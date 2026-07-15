import logging
import os
from pathlib import Path
import torch
import hydra
from datasets import Dataset, load_dataset
from omegaconf import OmegaConf
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOConfig, DPOTrainer

import wandb
from configs.schema import Config
from utils import maybe_resume_training

wandb.login()
logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
log.addHandler(ch)
os.environ["WANDB_ENTITY"] = "CodeShield"
os.environ["WANDB_PROJECT"] = "CerebRM-DPO"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
NUM_WORKERS = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else 1


def conv_to_dpo_format(example):
    if not isinstance(example["prompt"], list):
        example["prompt"] = [{"role": "user", "content": example["prompt"]}]
    if not isinstance(example["chosen"], list):
        example["chosen"] = [{"role": "assistant", "content": example["chosen"]}]
    if not isinstance(example["rejected"], list):
        example["rejected"] = [{"role": "assistant", "content": example["rejected"]}]
    return example


def train_model(
    cfg: Config,
    model_name: str,
    data: Dataset,
    wandb_run_name: str,
    output_dir: str,
) -> None:
    kernel = "flash_attention_2"

    config = DPOConfig(
        # model_init_kwargs={"attn_implementation": kernel, 'dtype': torch.bfloat16},
        output_dir=f"{output_dir}/intermediate_checkpoints",
        # DPO Parameters
        beta=cfg.dpo_params.beta,
        use_liger_kernel=False,
        precompute_ref_log_probs=True,
        precompute_ref_batch_size=cfg.dpo_params.precompute_ref_batch_size,
        # Training parameters
        bf16=cfg.dpo_params.use_bf16,
        eval_strategy="no",
        eval_steps=None,
        gradient_accumulation_steps=cfg.dpo_params.gradient_accumulation_steps,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=cfg.dpo_params.learning_rate,
        lr_scheduler_type=cfg.dpo_params.lr_scheduler_type,
        max_length=cfg.dpo_params.max_length,
        num_train_epochs=cfg.dpo_params.num_epochs,
        per_device_train_batch_size=cfg.dpo_params.batch_size,
        per_device_eval_batch_size=None,
        seed=cfg.dpo_params.seed,
        warmup_ratio=cfg.dpo_params.warmup_ratio,
        weight_decay=cfg.dpo_params.weight_decay,
        # Logging parameters
        log_level=cfg.wandb_params.log_level,
        log_on_each_node=True,
        logging_steps=cfg.dpo_params.logging_steps,
        report_to="wandb",
        run_name=wandb_run_name,
        # Saving parameters
        hub_model_id=f"wetsoledrysoul/{wandb_run_name}",
        hub_private_repo=True,
        hub_strategy="end",
        save_strategy="steps",
        save_steps=cfg.dpo_params.save_steps,
        # Data parameters
        data_seed=cfg.dpo_params.seed,
        dataloader_drop_last=True,
        dataloader_num_workers=NUM_WORKERS,
        dataset_num_proc=NUM_WORKERS,
        remove_unused_columns=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if cfg.data.chat_template_path and Path(cfg.data.chat_template_path).exists():
        tokenizer.chat_template = Path(cfg.data.chat_template_path).read_text()
    if cfg.dpo_params.pad_token_id:
        tokenizer.pad_token_id = cfg.dpo_params.pad_token_id
        tokenizer.pad_token = tokenizer.convert_ids_to_tokens(cfg.dpo_params.pad_token_id)
    model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation=kernel, dtype=torch.bfloat16).to('cuda')
    # ref_model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation=kernel, dtype=torch.bfloat16).to('cuda')
    trainer = DPOTrainer(model=model_name, ref_model=model, args=config, train_dataset=data, processing_class=tokenizer)
    gen_config = trainer.model.generation_config
    if gen_config.temperature is not None or gen_config.top_p is not None or gen_config.top_k is not None:
        gen_config.do_sample = True

    trainer.train(resume_from_checkpoint=maybe_resume_training(config.output_dir))
    trainer.push_to_hub()


def does_file_exist(file: Path) -> bool:
    return file.exists()


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: Config):
    model_short_name = cfg.dpo_params.model_path.split("/")[-1]
    wandb_run_name = f"dpo_{model_short_name}"
    output_dir = Path(f"{os.getenv('WORK')}/dpo_output/{wandb_run_name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    if all([does_file_exist(output_dir / "intermediate_checkpoints" / x) for x in ["tokenizer.json", "config.json", "model.safetensors.index.json", "generation_config.json"]]):
        log.info(f"dpo training files are already present in {output_dir}. Skipping.")
        return None

    log.info(f"Config: {OmegaConf.to_yaml(OmegaConf.structured(cfg))}")
    train_data = load_dataset(cfg.data.train)["train"]
    train_data = train_data.map(conv_to_dpo_format, num_proc=NUM_WORKERS, desc="Converting to DPO format")
    train_data = train_data.shuffle(seed=cfg.dpo_params.seed)
    log.info(f"Training {cfg.dpo_params.model_path} on {len(train_data)} examples")
    train_model(cfg, cfg.dpo_params.model_path, train_data, wandb_run_name, output_dir)
    log.info(f"Completed training {cfg.dpo_params.model_path} on {len(train_data)} examples")


if __name__ == "__main__":
    main()
