"""Fix Olmo3 RoPE in transformers 5.12.1.

transformers 5.12.1 (pinned in setup.sh for vllm-0.23.0 compat) builds a single
Olmo3RotaryEmbedding from the flat `rope_parameters` (rope_type=yarn) and feeds
the same yarn-scaled cos/sin to ALL layers. Olmo3 was trained with YaRN on the
8 full-attention layers only; the 24 sliding-window layers use unscaled rope
(see transformers 4.57 modeling_olmo3: per-layer-type `rotary_embs` ModuleDict,
transformers >=5.13 nested per-layer `rope_parameters`, and vllm olmo2.py:
"Rope scaling is only applied on full attention layers").

Measured on Olmo-3-7B-Think-SFT scoring an Aletheia-DPO chosen sample
(3072 tokens, fp32): mean NLL 0.74 with this patch vs 1.12 without; the last
1k tokens go from 0.75 to 1.77 nats/token and the gap keeps growing with
length. Training through the unpatched forward optimizes the wrong model and
produces checkpoints that degenerate under vllm's (correct) implementation.

Import and call `patch_olmo3_rope()` BEFORE any Olmo3 model is constructed
(trainer-internal loads included). Class-level patch, so it covers policy,
reference, and precompute models alike. Remove once transformers is upgraded
to >=5.13 together with a vllm release that reads nested rope_parameters.
"""

import copy
import logging

import transformers
from packaging import version
from transformers.models.olmo3 import modeling_olmo3

log = logging.getLogger(__name__)

_PATCHED = False


def patch_olmo3_rope():
    global _PATCHED
    if _PATCHED:
        return
    tf_version = version.parse(transformers.__version__)
    # <5.0 (4.57+) uses a per-layer-type rotary_embs ModuleDict and >=5.13 uses
    # nested per-layer-type rope_parameters — both are already correct. Only the
    # 5.0..5.12 line has the single-yarn-rotary bug this patch fixes.
    if not (version.parse("5.0") <= tf_version < version.parse("5.13")):
        log.info(f"olmo3_rope_patch: transformers {transformers.__version__} not affected, skipping")
        return
    _PATCHED = True

    orig_model_init = modeling_olmo3.Olmo3Model.__init__
    orig_layer_forward = modeling_olmo3.Olmo3DecoderLayer.forward

    def patched_model_init(self, config):
        orig_model_init(self, config)
        default_cfg = copy.deepcopy(config)
        rope_theta = config.rope_parameters.get("rope_theta", 500000)
        default_cfg.rope_parameters = {"rope_type": "default", "rope_theta": rope_theta}
        self.rotary_emb_sliding = modeling_olmo3.Olmo3RotaryEmbedding(config=default_cfg)
        for layer in self.layers:
            layer._sliding_rope = self.rotary_emb_sliding

    def patched_layer_forward(self, hidden_states, *args, position_ids=None, position_embeddings=None, **kwargs):
        if self.self_attn.attention_type == "sliding_attention":
            position_embeddings = self._sliding_rope(hidden_states, position_ids)
        return orig_layer_forward(
            self, hidden_states, *args, position_ids=position_ids, position_embeddings=position_embeddings, **kwargs
        )

    modeling_olmo3.Olmo3Model.__init__ = patched_model_init
    modeling_olmo3.Olmo3DecoderLayer.forward = patched_layer_forward
