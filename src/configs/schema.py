from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Data:
    train: str = None
    val: str = None
    chat_template_path: str = None


@dataclass
class GRPOParams:
    beta: float = None
    epsilon: float = None
    epsilon_high: float = None
    eval_batch_size: int = None
    learning_rate: float = None
    logging_steps: int = None
    loss_type: str = None
    lr_scheduler_type: str = None
    max_prompt_length: int = None
    num_epochs: int = None
    overwrite_output_dir: bool = None
    save_steps: float = None
    seed: int = None
    use_bf16: bool = None
    warmup_ratio: float = None
    weight_decay: float = None
    kl_penalty: str = None
    ref_model_sync_steps: int = None
    ref_model_mixup_alpha: float = None
    model_path: str = None
    batch_size: int = None
    gradient_accumulation_steps: int = None
    importance_sampling_level: str = None
    generate_every: int = None
    scale_rewards: str = None


@dataclass
class GenParams:
    max_completion_length: int = None
    num_generations: int = None
    temperature: float = None
    vllm_dtype: str = None
    vllm_gpu_memory_utilization: float = None
    vllm_max_model_length: int = None
    vllm_server_host: str = None
    vllm_server_port: int = None
    vllm_server_timeout: int = None
    vllm_tensor_parallel_size: int = None


@dataclass
class RESTEMParams:
    batch_size: int = None
    deepspeed_config_path: str = None
    gradient_accumulation_steps: int = None
    learning_rate: float = None
    logging_steps: int = None
    lr_scheduler_kwargs: dict = None
    lr_scheduler_type: str = None
    max_length: int = None
    model_path: str = None
    num_epochs: int = None
    overwrite_output_dir: bool = None
    save_steps: float = None
    seed: int = None
    use_bf16: bool = None
    warmup_ratio: float = None
    weight_decay: float = None
    num_generations: int = None
    gen_dp_size: int = None
    max_samples_to_keep: int = None
    stage1_num_saves: int = None
    pad_token_id: int = None


@dataclass
class STaRParams:
    batch_size: int = None
    deepspeed_config_path: str = None
    gradient_accumulation_steps: int = None
    learning_rate: float = None
    logging_steps: int = None
    lr_scheduler_kwargs: dict = None
    lr_scheduler_type: str = None
    max_length: int = None
    model_path: str = None
    num_epochs: int = None
    overwrite_output_dir: bool = None
    save_steps: float = None
    seed: int = None
    use_bf16: bool = None
    warmup_ratio: float = None
    weight_decay: float = None
    gen_dp_size: int = None
    pad_token_id: int = None


@dataclass
class DPOParams:
    learning_rate: float = None
    beta: float = None
    logging_steps: int = None
    lr_scheduler_type: str = None
    max_length: int = None
    num_epochs: int = None
    overwrite_output_dir: bool = None
    save_steps: int | float = None
    seed: int = None
    use_bf16: bool = None
    warmup_ratio: float = None
    weight_decay: float = None
    batch_size: int = None
    model_path: str = None
    gradient_accumulation_steps: int = None
    pad_token_id: int = None
    precompute_ref_batch_size: int = None


@dataclass
class GenRMParams:
    batch_size: int = None
    deepspeed_config_path: str = None
    gradient_accumulation_steps: int = None
    learning_rate: float = None
    logging_steps: int = None
    lr_scheduler_kwargs: dict = None
    lr_scheduler_type: str = None
    max_length: int = None
    model_path: str = None
    num_epochs: int = None
    overwrite_output_dir: bool = None
    save_steps: float = None
    seed: int = None
    use_bf16: bool = None
    warmup_ratio: float = None
    weight_decay: float = None
    pad_token_id: int = None


@dataclass
class WandbParams:
    log_level: str = None


@dataclass
class Config:
    data: Optional[Data] = field(default_factory=Data)
    gen_params: Optional[GenParams] = field(default_factory=GenParams)
    grpo_params: Optional[GRPOParams] = field(default_factory=GRPOParams)
    restem_params: Optional[RESTEMParams] = field(default_factory=RESTEMParams)
    genrm_params: Optional[GenRMParams] = field(default_factory=GenRMParams)
    star_params: Optional[STaRParams] = field(default_factory=STaRParams)
    dpo_params: Optional[DPOParams] = field(default_factory=DPOParams)
    wandb_params: Optional[WandbParams] = field(default_factory=WandbParams)
    grpo_reward_type: Optional[str] = None
    grpo_use_lora: Optional[bool] = None
    restem_stage: Optional[int] = None
    restem_episode: Optional[int] = None
    star_stage: Optional[int] = None
    star_episode: Optional[int] = None
