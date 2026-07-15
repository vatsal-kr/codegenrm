"""Fix trl colocate + vllm sleep-mode weight sync ordering.

With `vllm_mode="colocate"` and `vllm_enable_sleep_mode=True`, trl 1.7.x does,
per generation round:

  1. `sync_weights()`  - wake_up(weights), then push the CURRENT policy params
     into the vllm model via per-param `load_weights`.
  2. `generate()`      - wake_up (no-op) and `collective_rpc("reload_weights")`
     (workaround for vllm#29341, needed because level-2 sleep discards weight
     buffers AND non-parameter state such as rope caches).

Argument-less `reload_weights` re-reads the ORIGINAL checkpoint from disk
(`model_config.model`), so step 2 clobbers the weights pushed in step 1: vllm
samples from the frozen initial policy for the entire run and GRPO becomes
silently, increasingly off-policy. (Observed in the Jul 6/7 Olmo3 GRPO runs:
the reload's per-layer warnings appear right after every sync.)

This patch reorders the round to: wake -> reload from disk (restores valid
buffers) -> push current policy params -> generate (skip the clobbering
reload). If `generate()` ever runs without a fresh `sync_weights()` in the
same wake cycle, stock behavior is preserved.

Call `patch_trl_vllm_sleep_sync()` before the GRPOTrainer is constructed.
No-op for server mode and for `vllm_enable_sleep_mode=False`. Remove once trl
performs the disk reload before the param push upstream.
"""

import logging

log = logging.getLogger(__name__)

_PATCHED = False


def patch_trl_vllm_sleep_sync():
    global _PATCHED
    if _PATCHED:
        return
    _PATCHED = True

    import trl.generation.vllm_generation as _vg

    orig_sync_weights = _vg.VLLMGeneration.sync_weights
    orig_generate = _vg.VLLMGeneration.generate

    def patched_sync_weights(self):
        if self.mode == "colocate" and self.enable_sleep_mode:
            _vg.empty_cache()
            self.llm.wake_up(tags=["weights"])
            # Restore valid weight buffers and non-parameter state (rope
            # caches etc.) from disk BEFORE pushing the current policy, so the
            # push is the last write.
            try:
                self.llm.collective_rpc("reload_weights")
            except NotImplementedError:
                pass
            self._weights_valid_after_wake = True
        orig_sync_weights(self)

    def patched_generate(self, *args, **kwargs):
        if (
            self.mode == "colocate"
            and self.enable_sleep_mode
            and getattr(self, "_weights_valid_after_wake", False)
        ):
            # sync_weights already reloaded from disk and pushed the current
            # policy this wake cycle: make generate()'s own
            # collective_rpc("reload_weights") a no-op so it cannot overwrite
            # the just-synced policy with the on-disk checkpoint.
            orig_rpc = self.llm.collective_rpc

            def rpc_skip_reload(method, *rpc_args, **rpc_kwargs):
                if method == "reload_weights":
                    return None
                return orig_rpc(method, *rpc_args, **rpc_kwargs)

            self.llm.collective_rpc = rpc_skip_reload
            try:
                return orig_generate(self, *args, **kwargs)
            finally:
                self.llm.collective_rpc = orig_rpc
                # generate() ends with sleep(level=2): weights are gone again.
                self._weights_valid_after_wake = False
        return orig_generate(self, *args, **kwargs)

    _vg.VLLMGeneration.sync_weights = patched_sync_weights
    _vg.VLLMGeneration.generate = patched_generate
    log.info("trl_vllm_sleep_patch: patched VLLMGeneration sync/generate ordering")
