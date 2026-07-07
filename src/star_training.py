import logging
import os
from pathlib import Path
from typing import List

import hydra
import torch
from datasets import Dataset, concatenate_datasets, load_dataset
from omegaconf import OmegaConf
from transformers import AutoTokenizer
from trl import SFTConfig, SFTTrainer

import wandb
from cerebrm_prompts import LIST_REWARD_PROMPT, LIST_REWARD_PROMPT_COT
from cerebrm_rewards import extract_boxed_contents_list
from configs.schema import Config
from utils import Prompt, get_generated_text, maybe_resume_training, run_inference

wandb.login()
logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
log.addHandler(ch)
os.environ["WANDB_ENTITY"] = "CodeShield"
os.environ["WANDB_PROJECT"] = "CerebRM-STAR"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
NUM_WORKERS = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else 1


def _create_prompts(example, cfg: Config, hinted=False, thinking=True):
    potential_answers = ["A", "B", "C", "D", "E"][: example["num_candidates"]]
    candidates = [f"[CANDIDATE_{i}]\n```{example['language']}\n{candidate}\n```\n[/CANDIDATE_{i}]" for i, candidate in zip(potential_answers, example["candidates"])]
    candidate_str = "\n\n".join(candidates)
    if hinted:
        candidate_str += f"\n\nHint: The correct answer is {example['chosen_answer']}"
    if thinking:
        example["prompt"] = [
            {
                "role": "user",
                "content": LIST_REWARD_PROMPT.format(
                    question=example["query"],
                    candidates=candidate_str,
                    valid_options=", ".join(potential_answers),
                ).strip(),
            },
        ]
    else:
        example["prompt"] = [
            {"role": "system", "content": LIST_REWARD_PROMPT_COT.format(valid_options=", ".join(potential_answers))},
            {
                "role": "user",
                "content": f"Here is the coding question followed by the candidate solutions:\n[QUESTION]\n{example['query']}\n[/QUESTION]\n\n{candidate_str}\n\nYour response should be exactly in the specified format, without any extra characters or spaces. Anything else will be considered invalid.",
            },
        ]

    return example


def generate_completions(cfg: Config, prompts: List[Prompt], model: str) -> List[str]:
    # Sampling K responses from the current model
    dp_size = cfg.star_params.gen_dp_size if cfg.star_params.gen_dp_size else 1
    tp_size = torch.cuda.device_count() // dp_size
    responses = run_inference(
        prompts,
        model,
        temperature=0.6,
        max_tokens=cfg.star_params.max_length - 4096,
        n=1,
        dp_size=dp_size,
        tp_size=tp_size,
        max_num_batched_tokens=20_000,
        max_num_seqs=256,
        max_model_len=cfg.star_params.max_length,
    )

    completions = get_generated_text(responses)
    return [x[0] for x in completions]


def score_completions(completions: List[str], ground_truths: List[str]) -> List[bool]:
    # Scoring the K responses using a verifiable reward
    model_answers = [extract_boxed_contents_list(x) for x in completions]
    scored_completions = [model_ans == gt for model_ans, gt in zip(model_answers, ground_truths)]
    return scored_completions


def train_star_model(
    cfg: Config,
    model_path_episode: str,
    stage3_data: Dataset,
    wandb_run_name: str,
    output_dir: str,
    eval_data: Dataset | None,
) -> None:
    kernel = "flash_attention_2"
    config = SFTConfig(
        model_init_kwargs={"attn_implementation": kernel},
        output_dir=f"{output_dir}/intermediate_checkpoints",
        overwrite_output_dir=cfg.star_params.overwrite_output_dir,
        completion_only_loss=True,
        max_steps=cfg.star_params.num_steps,
        # Training parameters
        bf16=cfg.star_params.use_bf16,
        eval_strategy="steps" if eval_data else "no",
        eval_steps=cfg.star_params.save_steps if eval_data else None,
        gradient_accumulation_steps=cfg.star_params.gradient_accumulation_steps,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=cfg.star_params.learning_rate,
        lr_scheduler_type=cfg.star_params.lr_scheduler_type,
        max_length=cfg.star_params.max_length,
        num_train_epochs=cfg.star_params.num_epochs,
        per_device_train_batch_size=cfg.star_params.batch_size,
        per_device_eval_batch_size=cfg.star_params.batch_size,
        seed=cfg.star_params.seed,
        warmup_ratio=cfg.star_params.warmup_ratio,
        weight_decay=cfg.star_params.weight_decay,
        # Logging parameters
        log_level=cfg.wandb_params.log_level,
        log_on_each_node=True,
        logging_steps=cfg.star_params.logging_steps,
        load_best_model_at_end=False,
        report_to="wandb",
        run_name=wandb_run_name,
        # Saving parameters
        hub_model_id=f"wetsoledrysoul/{wandb_run_name}",
        hub_private_repo=True,
        hub_strategy="end",
        save_strategy="steps",
        save_steps=cfg.star_params.save_steps,
        # Data parameters
        data_seed=cfg.star_params.seed,
        dataloader_drop_last=True,
        dataloader_num_workers=NUM_WORKERS,
        dataset_num_proc=NUM_WORKERS,
        remove_unused_columns=False,
        use_liger_kernel=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.star_params.model_path, model_max_length=cfg.star_params.max_length)
    if cfg.data.chat_template_path and Path(cfg.data.chat_template_path).exists():
        tokenizer.chat_template = Path(cfg.data.chat_template_path).read_text()
    if cfg.star_params.pad_token_id is not None:
        tokenizer.pad_token_id = cfg.star_params.pad_token_id
        tokenizer.pad_token = tokenizer.convert_ids_to_tokens(cfg.star_params.pad_token_id)
    trainer = SFTTrainer(model=model_path_episode, args=config, train_dataset=stage3_data, processing_class=tokenizer)
    trainer.train(resume_from_checkpoint=maybe_resume_training(config.output_dir))
    trainer.push_to_hub()


def _to_conversational(example, thinking=True):
    prompt, completion = example["prompt"], example["completion"]
    if prompt[-1]["role"] == "assistant":
        completion = prompt[-1]["content"] + completion
        prompt = prompt[:-1]
    if not completion.startswith("<think>") and thinking:
        completion = "<think>\n" + completion.strip()
    example["prompt"] = prompt
    example["completion"] = [{"role": "assistant", "content": completion}]
    return example


def does_file_exist(file: Path) -> bool:
    return file.exists()


def _by_idx(example, indices: List[str], membership=True) -> bool:
    if membership:
        return example["idx"] in indices
    return example["idx"] not in indices


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: Config):
    is_thinking_model = "deepseek" in cfg.star_params.model_path.lower()
    ### Sanity checks
    assert cfg.star_stage in [1, 2, 3], "Invalid star_stage. Must be 1 for generating and scoring completions, 2 for generating and scoring hinted completions, or 3 for training."
    assert cfg.star_episode >= 0, "Invalid star_episode. Must be a non-negative integer."
    model_short_name = cfg.star_params.model_path.split("/")[-1]
    wandb_run_name = f"star_{model_short_name}_ep{cfg.star_episode}"
    model_path_episode = cfg.star_params.model_path if cfg.star_episode == 0 else f"wetsoledrysoul/star_{model_short_name}_ep{cfg.star_episode - 1}"
    output_dir = Path(f"{os.getenv('WORK')}/star_output/{wandb_run_name}")
    if cfg.star_stage == 2:
        if not all([does_file_exist(output_dir / x) for x in ["stage1_correct_data.parquet"]]):
            raise ValueError(f"Stage 1 files are missing in {output_dir}. Please run stage 1 for episode {cfg.star_episode} before proceeding.")
    if cfg.star_stage == 3:
        if not all([does_file_exist(output_dir / x) for x in ["correct_data.parquet"]]):
            raise ValueError(f"Stage 2 files are missing in {output_dir}. Please run stage 2 for episode {cfg.star_episode} before proceeding.")
    ### End sanity checks

    ### Check if stage has been completed previously
    if cfg.star_stage == 1:
        if all([does_file_exist(output_dir / x) for x in ["stage1_correct_data.parquet"]]):
            log.info(f"Stage 1 files are already present in {output_dir}. Skipping.")
            return None
    elif cfg.star_stage == 2:
        if all([does_file_exist(output_dir / x) for x in ["correct_data.parquet"]]):
            log.info(f"Stage 2 files are already present in {output_dir}. Skipping.")
            return None
    elif cfg.star_stage == 3:
        if all([does_file_exist(output_dir / "intermediate_checkpoints" / x) for x in ["tokenizer.json", "config.json", "generation_config.json"]]):
            log.info(f"Stage 3 files are already present in {output_dir}. Skipping.")
            return None
    ### End Checks

    log.info(f"Config: {OmegaConf.to_yaml(OmegaConf.structured(cfg))}")
    train_data = load_dataset(cfg.data.train)["train"]

    output_dir.mkdir(parents=True, exist_ok=True)

    train_data = train_data.map(_create_prompts, fn_kwargs={"cfg": cfg, "thinking": is_thinking_model}, num_proc=NUM_WORKERS, desc="Creating prompts")
    unhinted_prompts = list(train_data["prompt"])
    all_indices = list(train_data["idx"])
    log.info(f"Starting star episode {cfg.star_episode}")
    if cfg.star_stage == 1:
        log.info(f"Processing {len(unhinted_prompts)} prompts for stage 1 with {model_path_episode}")
        stage1_completions = generate_completions(cfg, unhinted_prompts, model_path_episode)
        log.info(f"Scoring {len(stage1_completions)} prompts for stage 1")
        scored_completions = score_completions(stage1_completions, list(train_data["chosen_answer"]))

        s1_idx_completion_map = {i: completion for i, completion, score in zip(all_indices, stage1_completions, scored_completions) if score}
        log.info(f"Correct responses in S1 (Unhinted prompting): {len(s1_idx_completion_map)} ({(len(s1_idx_completion_map) / len(all_indices)) * 100:.2f}%)")

        stage1_save_data = train_data.filter(_by_idx, fn_kwargs={"indices": list(s1_idx_completion_map.keys())}, num_proc=NUM_WORKERS, desc="Creating stage 1 correct dataset")
        stage1_save_data = stage1_save_data.add_column("completion", [s1_idx_completion_map[i] for i in stage1_save_data["idx"]])
        # Saving
        stage1_save_data.to_parquet((output_dir / "stage1_correct_data.parquet").as_posix())
        log.info(f"Completed Stage 1 of star episode {cfg.star_episode}")

    elif cfg.star_stage == 2:
        stage1_data = load_dataset("parquet", data_files=(output_dir / "stage1_correct_data.parquet").as_posix())["train"]
        stage2_data = train_data.filter(_by_idx, fn_kwargs={"indices": stage1_data["idx"], "membership": False}, num_proc=NUM_WORKERS, desc="Filtering stage 2 data")
        stage2_data = stage2_data.map(_create_prompts, fn_kwargs={"cfg": cfg, "hinted": True, "thinking": is_thinking_model}, num_proc=NUM_WORKERS, desc="Creating prompts")

        stage2_prompts = list(stage2_data["prompt"])
        stage2_answers = list(stage2_data["chosen_answer"])
        stage2_indices = list(stage2_data["idx"])

        log.info(f"Processing {len(stage2_prompts)} prompts for stage 2")
        stage2_completions = generate_completions(cfg, stage2_prompts, model_path_episode)
        log.info(f"Scoring {len(stage2_completions)} prompts for stage 2")
        scored_completions = score_completions(stage2_completions, stage2_answers)
        log.info(f"Number of correct responses in stage 2 (Unhinted prompting): {sum(scored_completions)} ({(sum(scored_completions) / len(scored_completions)) * 100:.2f}%)")

        s2_idx_completion_map = {i: completion for i, completion, score in zip(stage2_indices, stage2_completions, scored_completions) if score}
        log.info(f"Correct responses in S2 (Hinted prompting): {len(s2_idx_completion_map)} ({(len(s2_idx_completion_map) / len(stage2_indices)) * 100:.2f}%)")

        stage2_save_data = train_data.filter(_by_idx, fn_kwargs={"indices": list(s2_idx_completion_map.keys())}, num_proc=NUM_WORKERS, desc="Creating stage 2 correct dataset")
        stage2_save_data = stage2_save_data.add_column("completion", [s2_idx_completion_map[i] for i in stage2_save_data["idx"]])

        save_data = concatenate_datasets([stage1_data, stage2_save_data])
        save_data.to_parquet((output_dir / "correct_data.parquet").as_posix())
        log.info(f"Completed Stage 2 of star episode {cfg.star_episode}")
    else:
        stage3_data = load_dataset("parquet", data_files=(output_dir / "correct_data.parquet").as_posix())["train"]
        stage3_data = stage3_data.select_columns(["prompt", "completion", "idx", "chosen_answer"])
        stage3_data = stage3_data.map(_to_conversational, fn_kwargs={"thinking": is_thinking_model}, num_proc=NUM_WORKERS, desc="Converting to conversational format")
        log.info(f"Training {cfg.star_params.model_path} on {len(stage3_data)} examples in stage three for episode {cfg.star_episode}")
        train_star_model(cfg, cfg.star_params.model_path, stage3_data, wandb_run_name, output_dir, eval_data=None)
        log.info(f"Completed Stage 3 of star episode {cfg.star_episode}")


if __name__ == "__main__":
    """
    STaR stages: 
    Stage 1: Sample 1 response from the current model for each prompt in the training set and score all the responses
    Stage 2: Sample and score responses with prompts augmented with a hint for the incorrect prompts in stage 1
    Stage 3: Train on the final dataset of correct responses from stage 1 and 2
    """
    main()
