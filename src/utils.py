import logging
import os
import re
from dataclasses import dataclass
from multiprocessing import Manager, Process
from pathlib import Path
from time import sleep
from typing import List

import torch
from transformers import AutoConfig, AutoTokenizer
from vllm import LLM, RequestOutput, SamplingParams

try:
    from vllm.utils import get_open_port
except ImportError:
    from vllm.utils.network_utils import get_open_port

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
log.addHandler(ch)


@dataclass
class Message:
    role: str
    content: str


Prompt = List[Message]


def get_generated_text(responses: List[RequestOutput]) -> List[List[str]]:
    """Extract generated text from a list of RequestOutput objects.

    Args:
        responses (List[RequestOutput]): A list of RequestOutput objects containing generated responses.

    Returns:
        List[List[str]]: A list of lists containing the generated text for each prompt. The shape of the list is [num_prompts, num_responses_per_prompt].
    """
    return [[nth_response.text for nth_response in responses.outputs] for responses in responses]


def maybe_resume_training(base_dir: str) -> bool:
    """
    Find the latest valid checkpoint directory inside base_dir/checkpoint_x.
    A valid checkpoint must contain config.json, tokenizer.json, and at least
    one model weight file (pytorch_model.bin or model.safetensors).
    Returns a Path or None if no valid checkpoint exists.
    """
    base_path = Path(base_dir)
    if not base_path.is_dir():
        return False

    checkpoint_pattern = re.compile(r"checkpoint-(\d+)")

    candidates = []
    for subdir in base_path.iterdir():
        if subdir.is_dir():
            match = checkpoint_pattern.fullmatch(subdir.name)
            if not match:
                continue
            step = int(match.group(1))
            candidates.append((step, subdir))

    if not candidates:
        return False

    # Return the path of the checkpoint with the highest step
    return True


def get_best_config(model_name, dp_size, tp_size):
    config = AutoConfig.from_pretrained(model_name)
    attn_heads = config.num_attention_heads
    avail_gpus = torch.cuda.device_count()
    assert avail_gpus == dp_size * tp_size
    while attn_heads % tp_size:
        tp_size /= 2
        dp_size *= 2
    return dp_size, tp_size


def run_inference(
    prompts: List[Prompt],
    model: str,
    dp_size: int = None,
    tp_size: int = None,
    node_size: int = 1,
    node_rank: int = 0,
    master_addr: str = "",
    master_port: int = 0,
    enforce_eager: bool = False,
    trust_remote_code: bool = False,
    max_num_seqs: int = 256,
    max_model_len: int = None,
    gpu_memory_utilization: float = 0.95,
    compilation_config: int = None,
    quantization: str = None,
    enable_expert_parallel: bool = False,
    temperature=1.0,
    max_tokens=4096,
    n=1,
    enable_thinking: bool = False,
    tokenizer=None,
    max_num_batched_tokens: int = 2048,
    **kwargs,
) -> List[RequestOutput]:
    if not dp_size and not tp_size:
        dp_size = 1
        tp_size = torch.cuda.device_count()
    elif not dp_size:
        dp_size = torch.cuda.device_count() // tp_size
    elif not tp_size:
        tp_size = torch.cuda.device_count() // dp_size
    else:
        assert dp_size * tp_size == torch.cuda.device_count(), "dp_size * tp_size must equal the number of available GPUs"
    dp_size, tp_size = get_best_config(model, dp_size, tp_size)
    """
    Run data-parallel inference with configurable distributed and optimization settings.

    Args:
        prompts (List[Prompt]): A list of prompts to generate responses for. Each prompt is expected to be in conversational format - a list of dictionaries with "role" and "content" keys.
        model (str): Model name or path.
        dp_size (int): Data parallel size.
        tp_size (int): Tensor parallel size.
        node_size (int): Total number of nodes.
        node_rank (int): Rank of the current node.
        master_addr (str): Master node IP address.
        master_port (int): Master node port.
        enforce_eager (bool): Enforce eager mode execution.
        trust_remote_code (bool): Trust remote code execution.
        max_num_seqs (int): Max number of sequences per iteration.
        max_model_len (int): Max tokens processed per iteration.
        gpu_memory_utilization (float): Fraction of GPU memory vLLM can allocate (0.0, 1.0].
        compilation_config (int): Compilation optimization level (0-3).
        quantization (str): Quantization type or config.
        enable_expert_parallel (bool): Enable or disable expert parallelism.
        temperature (float): Sampling temperature.
        max_tokens (int): Maximum number of tokens to generate.
        n (int): Number of responses to generate.
    Returns:
        List[RequestOutput]: A list of generated responses per prompt. Each request output can contain multiple generations if n > 1.
    """
    if not tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=trust_remote_code)
    add_generation_prompt = not prompts[0][-1]["role"] == "assistant"
    prompts = tokenizer.apply_chat_template(
        prompts,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        continue_final_message=not add_generation_prompt,
        enable_thinking=enable_thinking,
    )

    if node_size == 1:
        dp_master_ip = "127.0.0.1"
        dp_master_port = get_open_port()
    else:
        dp_master_ip = master_addr
        dp_master_port = master_port

    assert dp_size % node_size == 0, "dp_size should be divisible by node_size"
    dp_per_node = dp_size // node_size
    manager = Manager()
    vllm_outputs = manager.list()

    floor = len(prompts) // dp_size
    remainder = len(prompts) % dp_size

    # Distribute prompts into even groups.
    def start(rank):
        return rank * floor + min(rank, remainder)

    procs = []
    for local_dp_rank, global_dp_rank in enumerate(range(node_rank * dp_per_node, (node_rank + 1) * dp_per_node)):
        start_idx = start(global_dp_rank)
        end_idx = start(global_dp_rank + 1)
        subset_prompts = prompts[start_idx:end_idx]
        if not subset_prompts:
            subset_prompts = ["__placeholder__"]
        proc = Process(
            target=_run_inference,
            args=(
                model,
                dp_size,
                local_dp_rank,
                global_dp_rank,
                dp_master_ip,
                dp_master_port,
                tp_size,
                enforce_eager,
                enable_expert_parallel,
                trust_remote_code,
                max_num_seqs,
                max_model_len,
                compilation_config,
                gpu_memory_utilization,
                quantization,
                temperature,
                max_tokens,
                n,
                max_num_batched_tokens,
            ),
            kwargs={
                **kwargs,
                "vllm_outputs": vllm_outputs,
                "subset_prompts": subset_prompts,
                "start_idx": start_idx,
            },
        )
        proc.start()
        procs.append(proc)
    exit_code = 0
    for proc in procs:
        proc.join()
        if proc.exitcode:
            exit_code = proc.exitcode
    if exit_code != 0:
        raise RuntimeError(f"Some processes failed with exit code {exit_code}")

    for proc in procs:
        proc.terminate()

    aggregated = sorted(list(vllm_outputs), key=lambda x: x["prompt_id"])
    outputs = [item["output"] for item in aggregated]

    return outputs


def _run_inference(
    model,
    dp_size,
    local_dp_rank,
    global_dp_rank,
    dp_master_ip,
    dp_master_port,
    GPUs_per_dp_rank,
    enforce_eager,
    enable_expert_parallel,
    trust_remote_code,
    max_num_seqs,
    max_model_len,
    compilation_config,
    gpu_memory_utilization,
    quantization,
    temperature,
    max_tokens,
    n,
    max_num_batched_tokens,
    vllm_outputs=None,
    subset_prompts=None,
    start_idx=0,
    **kwargs,
):
    os.environ["VLLM_DP_RANK"] = str(global_dp_rank)
    os.environ["VLLM_DP_RANK_LOCAL"] = str(local_dp_rank)
    os.environ["VLLM_DP_SIZE"] = str(dp_size)
    os.environ["VLLM_DP_MASTER_IP"] = dp_master_ip
    os.environ["VLLM_DP_MASTER_PORT"] = str(dp_master_port)

    # CUDA_VISIBLE_DEVICES for each DP rank is set automatically inside the
    # engine processes.

    if subset_prompts is None:
        subset_prompts = ["__placeholder__"]
    log.info(f"[DP Rank {global_dp_rank}] Processing {len(subset_prompts)} prompts")
    sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens, n=n, **kwargs)
    llm = LLM(
        model=model,
        tensor_parallel_size=GPUs_per_dp_rank,
        enforce_eager=enforce_eager,
        enable_expert_parallel=enable_expert_parallel,
        trust_remote_code=trust_remote_code,
        max_num_seqs=max_num_seqs,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        quantization=quantization,
        compilation_config=compilation_config,
        max_num_batched_tokens=max_num_batched_tokens,
    )
    outputs = llm.generate(subset_prompts, sampling_params)
    # Print the outputs.
    if vllm_outputs is not None:
        for i, output in enumerate(outputs):
            prompt_id = start_idx + i
            vllm_outputs.append({"prompt_id": prompt_id, "output": output})

    # Give engines time to pause their processing loops before exiting.
    sleep(1)
