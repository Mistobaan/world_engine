# MLX Port Plan

This plan breaks the MLX port into stories and tasks. Each story lists a clear
outcome and a set of implementable tasks.

## Story 0: Feasibility and Baseline
Outcome: Confirm target platform constraints and define validation baselines.

Tasks:
- Identify target MLX version and minimum macOS + Apple Silicon requirements.
- Decide dtype strategy (bf16 vs fp16) and document implications.
- Define a minimal golden test set (prompt, controller inputs, seed) and
  baseline Torch outputs to compare against.
- Document acceptable numeric tolerances for latent and pixel outputs.

## Story 1: Core MLX Model Forward (No KV Cache)
Outcome: WorldModel forward pass works in MLX with dense attention.

Tasks:
- Implement MLX versions of core layers (Linear, Conv2d, MLP, SiLU).
- Port RMSNorm, AdaLN, AdaRMSNorm, and NoiseConditioner.
- Implement patchify/unpatchify and token reshaping in MLX.
- Implement OrthoRoPE in MLX.
- Implement dense attention (no KV cache), including GQA support.
- Replace TensorDict usage with MLX-native tensor struct or tuples.
- Validate single-frame forward against Torch for a fixed input.

## Story 2: Scheduler and Denoise Loop
Outcome: End-to-end denoise loop works in MLX (latent only).

Tasks:
- Port `scheduler_sigmas` and diff-based Euler update loop.
- Port controller conditioning and prompt conditioning logic paths.
- Ensure batching assumptions (B==1, N==1) are enforced or generalized.
- Add MLX-only inference wrapper similar to `WorldEngine.gen_frame`.
- Validate latent output stability across multiple denoise steps.

## Story 3: KV Cache and Attention Windowing
Outcome: Multi-frame rollout works with a functional KV cache in MLX.

Tasks:
- Implement ring-buffer KV cache structure (local/global windows).
- Implement block-mask or fallback windowed dense attention.
- Support pinned dilation policy for global layers.
- Validate attention visibility matches Torch for a small sequence.
- Add cache reset and freeze/unfreeze behavior.

## Story 4: AutoEncoder Port
Outcome: MLX autoencoder encode/decode matches Torch within tolerance.

Tasks:
- Port AE blocks including weight-norm replacements.
- Implement pixel shuffle/unshuffle and bicubic interpolate equivalents.
- Match input preprocessing and output postprocessing (uint8 scaling).
- Validate encode/decode on a small image set.

## Story 5: Prompt Encoder Strategy
Outcome: Prompt conditioning works end-to-end with MLX pipeline.

Tasks:
- Decide between MLX-native UMT5 or Torch sidecar.
- If MLX-native, port tokenizer + encoder and validate embeddings.
- If sidecar, define interop format and latency budget.
- Validate prompt-conditioned outputs with multiple prompts.

## Story 6: Optimization and Parity
Outcome: Performance and feature parity on macOS MLX backend.

Tasks:
- Add MLX-appropriate compile/jit hooks if available.
- Evaluate cached conditioning LUT feasibility with MLX dtype support.
- Decide on quantization strategy (disable or MLX-native).
- Profile end-to-end frame generation and tune hotspots.
- Document known gaps vs Torch version and mitigation plan.

## Story 7: Packaging and Developer Experience
Outcome: MLX backend is usable and documented.

Tasks:
- Add backend selection flags and environment detection.
- Update README with MLX setup steps and limitations.
- Add a minimal MLX example script under `examples/`.
- Add CI or local checks for MLX (if feasible).
