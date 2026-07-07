import logging
import os
import pickle
from pathlib import Path
from typing import List

import hydra
import pandas as pd
import torch
from datasets import Dataset, load_dataset
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
os.environ["WANDB_PROJECT"] = "CerebRM-RESTEM"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
NUM_WORKERS = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else 1


def _create_prompts(example, cfg: Config, thinking=True):
    potential_answers = ["A", "B", "C", "D", "E"][: example["num_candidates"]]
    candidates = [f"[CANDIDATE_{i}]\n```{example['language']}\n{candidate}\n```\n[/CANDIDATE_{i}]" for i, candidate in zip(potential_answers, example["candidates"])]
    candidate_str = "\n\n".join(candidates)
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


def generate_completions(cfg: Config, prompts: List[Prompt], model: str) -> List[List[str]]:
    # Sampling K responses from the current model
    dp_size = cfg.restem_params.gen_dp_size if cfg.restem_params.gen_dp_size else 1
    tp_size = torch.cuda.device_count() // dp_size
    responses = run_inference(
        prompts,
        model,
        temperature=1.0,
        max_tokens=cfg.restem_params.max_length - 4096,
        n=cfg.restem_params.num_generations,
        dp_size=dp_size,
        tp_size=tp_size,
        max_num_batched_tokens=20_000,
        max_num_seqs=256,
        max_model_len=cfg.restem_params.max_length,
    )
    completions = get_generated_text(responses)
    return completions


def score_completions(completions: List[List[str]], ground_truths: List[str]) -> List[List[bool]]:
    # Scoring the K responses using a verifiable reward
    model_answers = [[extract_boxed_contents_list(y) for y in x] for x in completions]
    scored_completions = [[model_ans == gt for model_ans in model_ans_list] for model_ans_list, gt in zip(model_answers, ground_truths)]
    return scored_completions


def train_restem_model(
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
        overwrite_output_dir=cfg.restem_params.overwrite_output_dir,
        completion_only_loss=True,
        # Training parameters
        bf16=cfg.restem_params.use_bf16,
        eval_strategy="steps" if eval_data else "no",
        eval_steps=cfg.restem_params.save_steps if eval_data else None,
        gradient_accumulation_steps=cfg.restem_params.gradient_accumulation_steps,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=cfg.restem_params.learning_rate,
        lr_scheduler_type=cfg.restem_params.lr_scheduler_type,
        max_length=cfg.restem_params.max_length,
        num_train_epochs=cfg.restem_params.num_epochs,
        per_device_train_batch_size=cfg.restem_params.batch_size,
        per_device_eval_batch_size=cfg.restem_params.batch_size,
        seed=cfg.restem_params.seed,
        warmup_ratio=cfg.restem_params.warmup_ratio,
        weight_decay=cfg.restem_params.weight_decay,
        # Logging parameters
        log_level=cfg.wandb_params.log_level,
        log_on_each_node=True,
        logging_steps=cfg.restem_params.logging_steps,
        load_best_model_at_end=False,
        report_to="wandb",
        run_name=wandb_run_name,
        # Saving parameters
        hub_model_id=f"wetsoledrysoul/{wandb_run_name}",
        hub_private_repo=True,
        hub_strategy="end",
        save_strategy="steps",
        save_steps=cfg.restem_params.save_steps,
        # Data parameters
        data_seed=cfg.restem_params.seed,
        dataloader_drop_last=True,
        dataloader_num_workers=NUM_WORKERS,
        dataset_num_proc=NUM_WORKERS,
        remove_unused_columns=False,
        use_liger_kernel=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.restem_params.model_path, model_max_length=cfg.restem_params.max_length)
    if cfg.data.chat_template_path and Path(cfg.data.chat_template_path).exists():
        tokenizer.chat_template = Path(cfg.data.chat_template_path).read_text()
    if cfg.restem_params.pad_token_id is not None:
        tokenizer.pad_token_id = cfg.restem_params.pad_token_id
        tokenizer.pad_token = tokenizer.convert_ids_to_tokens(cfg.restem_params.pad_token_id)
    trainer = SFTTrainer(model=model_path_episode, args=config, train_dataset=stage3_data, processing_class=tokenizer)
    trainer.train(resume_from_checkpoint=maybe_resume_training(config.output_dir))
    trainer.push_to_hub()


def construct_data_for_training(cfg, dataset_indices, stage1_prompts, episode_completions, scored_completions, thinking=True) -> Dataset:
    prompts, completions, indices = [], [], []
    for idx, prompt, completions_list, scores in zip(dataset_indices, stage1_prompts, episode_completions, scored_completions):
        completions_list = [c for c, s in zip(completions_list, scores) if s]
        if not completions_list:
            continue
        for completion in completions_list:
            if prompt[-1]["role"] == "assistant":
                completion = prompt[-1]["content"] + completion
                prompt = prompt[:-1]
            if not completion.startswith("<think>") and thinking:
                completion = "<think>\n" + completion.strip()
            completion = [{"role": "assistant", "content": completion}]
            prompts.append(prompt)
            completions.append(completion)
            indices.append(idx)
    # Generate data with all possible correct samples
    full_dataset = pd.DataFrame({"prompt": prompts, "completion": completions, "indices": indices})

    # Smartly subsample to len(dataset_indices) - First keep atleast one from each available index, and then randomly sample the rest
    one_per_idx = full_dataset.sample(frac=1, random_state=cfg.restem_params.seed).drop_duplicates(subset="indices", keep="first").reset_index(names="full_dset_idx")
    log.info(f"Sampled {len(one_per_idx)} instances initially, with one entry per original training index")
    remaining_data = full_dataset[~full_dataset.index.isin(one_per_idx.full_dset_idx)]
    # sample either how much is needed to get to the original length, or how much is available
    num_to_sample = min(len(dataset_indices) - len(one_per_idx), len(remaining_data))
    additional_data = remaining_data.sample(n=num_to_sample, random_state=cfg.restem_params.seed).reset_index(drop=True)
    log.info(f"Sampled {len(additional_data)} additional instances randomly")
    final_ds = pd.concat([one_per_idx.drop("full_dset_idx", axis=1), additional_data], axis=0).reset_index(drop=True)
    log.info(f"Final dataset: {len(final_ds)} instances")
    return Dataset.from_pandas(final_ds)


def does_file_exist(file: Path) -> bool:
    return file.exists()


def _shard(prompts: list, n: int) -> list[list]:
    k, r = divmod(len(prompts), n)
    return [prompts[i * k + min(i, r) : (i + 1) * k + min(i + 1, r)] for i in range(n)]


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: Config):
    is_thinking_model = "deepseek" in cfg.restem_params.model_path.lower()
    ### Sanity checks
    assert cfg.restem_stage in [1, 2], "Invalid restem_stage. Must be 1 for generating and scoring completions, or 2 for training."
    assert cfg.restem_episode >= 0, "Invalid restem_episode. Must be a non-negative integer."
    model_short_name = cfg.restem_params.model_path.split("/")[-1]
    wandb_run_name = f"restem_{model_short_name}_ep{cfg.restem_episode}"
    model_path_episode = cfg.restem_params.model_path if cfg.restem_episode == 0 else f"wetsoledrysoul/restem_{model_short_name}_ep{cfg.restem_episode - 1}"
    output_dir = Path(f"{os.getenv('WORK')}/restem_output/{wandb_run_name}")
    if cfg.restem_stage == 2:
        if not all([does_file_exist(output_dir / x) for x in ["completions.pkl", "scored_completions.pkl"]]):
            raise ValueError(f"Stage 1 files are missing in {output_dir}. Please run stage 1 for episode {cfg.restem_episode} before proceeding.")
    ### End sanity checks
    ### Check if stage has been completed previously
    if cfg.restem_stage == 1:
        if all([does_file_exist(output_dir / x) for x in ["completions.pkl", "scored_completions.pkl"]]):
            log.info(f"Stage 1 files are already present in {output_dir}. Skipping.")
            return None
    elif cfg.restem_stage == 2:
        if all([does_file_exist(output_dir / "intermediate_checkpoints" / x) for x in ["tokenizer.json", "config.json", "generation_config.json"]]):
            log.info(f"Stage 2 files are already present in {output_dir}. Skipping.")
            return None
    ### End Checks

    log.info(f"Config: {OmegaConf.to_yaml(OmegaConf.structured(cfg))}")
    train_data = load_dataset(cfg.data.train)["train"]
    if cfg.data.val:
        eval_data = load_dataset(cfg.data.val)
        if "test_weak_easy" in eval_data:
            eval_data = eval_data["test_weak_easy"]
        elif "Full" in eval_data:
            eval_data = eval_data["Full"]
        else:
            eval_data = eval_data["test"]
    else:
        eval_data = None

    output_dir.mkdir(parents=True, exist_ok=True)

    train_data = train_data.map(_create_prompts, fn_kwargs={"cfg": cfg, "thinking": is_thinking_model}, num_proc=NUM_WORKERS, desc="Creating prompts")
    stage1_prompts = list(train_data["prompt"])
    dataset_indices = list(train_data["idx"])
    log.info(f"Starting restem episode {cfg.restem_episode}")
    if cfg.restem_stage == 1:
        num_shards = cfg.restem_params.stage1_num_saves if cfg.restem_params.stage1_num_saves else 1
        sharded_prompts = _shard(stage1_prompts, num_shards)
        for shard_idx, shard_prompts in enumerate(sharded_prompts):
            shard_outfiles = [f"completions_{shard_idx}.pkl", f"scored_completions_{shard_idx}.pkl"]
            if all([does_file_exist(output_dir / outfile) for outfile in shard_outfiles]):
                log.info(f"Shard {shard_idx} already processed. Skipping.")
                continue
            log.info(f"Processing shard {shard_idx + 1}/{num_shards}")
            # Check if shard has already been processed
            shard_completions = generate_completions(cfg, shard_prompts, model_path_episode)
            log.info(f"Scoring {len(shard_completions)} prompts")
            shard_scored_completions = score_completions(shard_completions, list(train_data["chosen_answer"]))
            with open(output_dir / f"completions_{shard_idx}.pkl", "wb") as f:
                pickle.dump(shard_completions, f)
            with open(output_dir / f"scored_completions_{shard_idx}.pkl", "wb") as f:
                pickle.dump(shard_scored_completions, f)
            log.info(f"Completed shard {shard_idx + 1}/{num_shards}")
        # Cleanup sharded files
        all_completions, all_scored_completions = [], []
        for shard_idx in range(num_shards):
            with open(output_dir / f"completions_{shard_idx}.pkl", "rb") as f:
                all_completions.extend(pickle.load(f))
            with open(output_dir / f"scored_completions_{shard_idx}.pkl", "rb") as f:
                all_scored_completions.extend(pickle.load(f))
            (output_dir / f"completions_{shard_idx}.pkl").unlink(missing_ok=True)
            (output_dir / f"scored_completions_{shard_idx}.pkl").unlink(missing_ok=True)
        log.info(f"Completed Stage 1 of restem episode {cfg.restem_episode}")
        with open(output_dir / "completions.pkl", "wb") as f:
            pickle.dump(all_completions, f)
        with open(output_dir / "scored_completions.pkl", "wb") as f:
            pickle.dump(all_scored_completions, f)
    else:
        with open(output_dir / "completions.pkl", "rb") as f:
            episode_completions = pickle.load(f)
        with open(output_dir / "scored_completions.pkl", "rb") as f:
            scored_completions = pickle.load(f)
        stage3_data = construct_data_for_training(cfg, dataset_indices, stage1_prompts, episode_completions, scored_completions, thinking=is_thinking_model)
        log.info(f"Training {model_path_episode} on {len(stage3_data)} examples in Stage 2")
        train_restem_model(cfg, model_path_episode, stage3_data, wandb_run_name, output_dir, eval_data)
        log.info(f"Completed Stage 2 of restem episode {cfg.restem_episode}")


if __name__ == "__main__":
    """
    Usage guide:
    Since only one vllm object can be created at a time (without running into vllm errors), the script is designed to be run in two `sub_stages` per restem episode.
    For each episode, run the following commands sequentially:
    
    restem_training.py +restem_stage=1 restem_episode=0
    restem_training.py +restem_stage=2 restem_episode=0
    restem_training.py +restem_stage=1 restem_episode=1
    restem_training.py +restem_stage=2 restem_episode=1
    
    and so on.
    """
    main()
