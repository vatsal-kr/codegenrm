import logging
import os
from pathlib import Path

import hydra
import wandb
from configs.schema import Config
from datasets import Dataset, load_dataset
from omegaconf import OmegaConf
from transformers import AutoTokenizer
from utils import maybe_resume_training

from trl import SFTConfig, SFTTrainer

wandb.login()
logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
log.addHandler(ch)
os.environ["WANDB_ENTITY"] = "CodeShield"
os.environ["WANDB_PROJECT"] = "CerebRM-GenRM-CoT"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
NUM_WORKERS = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else 1


def _create_training_dataset(example):
    if not isinstance(example["prompt"], list):
        example["prompt"] = [{"role": "user", "content": example["prompt"]}]
    if not isinstance(example["completion"], list):
        example["completion"] = [{"role": "assistant", "content": example["completion"]}]
    return example


def train_model(
    cfg: Config,
    model_name: str,
    data: Dataset,
    wandb_run_name: str,
    output_dir: str,
    eval_data: Dataset | None = None,
) -> None:
    kernel = "flash_attention_2"

    config = SFTConfig(
        model_init_kwargs={"attn_implementation": kernel},
        output_dir=f"{output_dir}/intermediate_checkpoints",
        overwrite_output_dir=cfg.genrm_params.overwrite_output_dir,
        completion_only_loss=True,
        # Training parameters
        bf16=cfg.genrm_params.use_bf16,
        eval_strategy="steps" if eval_data else "no",
        eval_steps=cfg.genrm_params.eval_steps if eval_data else None,
        gradient_accumulation_steps=cfg.genrm_params.gradient_accumulation_steps,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=cfg.genrm_params.learning_rate,
        lr_scheduler_type=cfg.genrm_params.lr_scheduler_type,
        max_length=cfg.genrm_params.max_length,
        num_train_epochs=cfg.genrm_params.num_epochs,
        per_device_train_batch_size=cfg.genrm_params.batch_size,
        per_device_eval_batch_size=cfg.genrm_params.batch_size,
        seed=cfg.genrm_params.seed,
        warmup_ratio=cfg.genrm_params.warmup_ratio,
        weight_decay=cfg.genrm_params.weight_decay,
        # Logging parameters
        log_level=cfg.wandb_params.log_level,
        log_on_each_node=True,
        logging_steps=cfg.genrm_params.logging_steps,
        load_best_model_at_end=False,
        report_to="wandb",
        run_name=wandb_run_name,
        # Saving parameters
        hub_model_id=f"wetsoledrysoul/{wandb_run_name}",
        hub_private_repo=True,
        hub_strategy="end",
        save_strategy="steps",
        save_steps=cfg.genrm_params.save_steps,
        # Data parameters
        data_seed=cfg.genrm_params.seed,
        dataloader_drop_last=True,
        dataloader_num_workers=NUM_WORKERS,
        dataset_num_proc=NUM_WORKERS,
        remove_unused_columns=False,
        use_liger_kernel=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.genrm_params.model_path, model_max_length=cfg.genrm_params.max_length)
    if cfg.data.chat_template_path and Path(cfg.data.chat_template_path).exists():
        tokenizer.chat_template = Path(cfg.data.chat_template_path).read_text()
    if cfg.genrm_params.pad_token_id is not None:
        tokenizer.pad_token_id = cfg.genrm_params.pad_token_id
        tokenizer.pad_token = tokenizer.convert_ids_to_tokens(cfg.genrm_params.pad_token_id)
    trainer = SFTTrainer(model=model_name, args=config, train_dataset=data, processing_class=tokenizer)

    trainer.train(resume_from_checkpoint=maybe_resume_training(config.output_dir))
    trainer.push_to_hub()


def does_file_exist(file: Path) -> bool:
    return file.exists()


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: Config):
    model_short_name = cfg.genrm_params.model_path.split("/")[-1]
    wandb_run_name = f"genrm_cot_{model_short_name}_LR{cfg.genrm_params.learning_rate}"
    output_dir = Path(f"{os.getenv('WORK')}/genrm_output/{wandb_run_name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    if all([does_file_exist(output_dir / "intermediate_checkpoints" / x) for x in ["tokenizer.json", "config.json", "generation_config.json"]]):
        log.info(f"GenRM CoT training files are already present in {output_dir}. Skipping.")
        return None

    log.info(f"Config: {OmegaConf.to_yaml(OmegaConf.structured(cfg))}")
    train_data = load_dataset(cfg.data.train)["train"]

    train_data = train_data.map(_create_training_dataset, num_proc=NUM_WORKERS, desc="Creating prompts")
    train_data = train_data.shuffle(seed=cfg.genrm_params.seed)
    log.info(f"Training {cfg.genrm_params.model_path} on {len(train_data)} examples")
    train_model(cfg, cfg.genrm_params.model_path, train_data, wandb_run_name, output_dir)
    log.info(f"Completed training {cfg.genrm_params.model_path} on {len(train_data)} examples")


if __name__ == "__main__":
    main()
