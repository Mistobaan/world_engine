# Architecture Overview (for MLX Port Planning)

This document summarizes the codebase layout and runtime behavior to support a
port to the Apple MLX backend for native macOS execution. It focuses on runtime
components, data flow, and all Torch-specific features that will need
equivalents or rewrites.

## Project Layout

- `src/world_engine.py`: Public `WorldEngine` API and inference orchestration.
- `src/model/world_model.py`: Core World Model (DiT-style transformer with
  controller + prompt conditioning).
- `src/model/attn.py`: Self-attention and cross-attention (RoPE, flex attention).
- `src/model/kv_cache.py`: Sparse KV cache, block-mask creation, and cache policy.
- `src/model/nn.py`: Shared NN utilities (AdaLN, AdaRMSNorm, NoiseConditioner).
- `src/ae.py`: Inference autoencoder loader + encode/decode helpers.
- `src/ae_nn.py`: Autoencoder architecture (encoder/decoder blocks).
- `src/patch_model.py`: Inference-only patches (cached conditioning, merged QKV).
- `src/quantize.py`: Optional quantization (NVFP4/FP8) and custom ops.
- `examples/`: Demo and benchmarking scripts (not part of the package API).

## High-Level Runtime Components

### WorldEngine
Entry-point for inference. Responsibilities:
- Load model config (via `WorldModel.load_config`).
- Load VAE/AutoEncoder (`InferenceAE`).
- Load text encoder (`PromptEncoder`) when prompt conditioning is enabled.
- Load World Model weights + apply inference patches.
- Initialize scheduler sigmas and KV cache.
- Maintain inference state: frame timestamp, KV cache, and prompt context.

Core public methods:
- `reset()`: resets KV cache and frame timestamp.
- `set_prompt(text)`: encodes text prompt to embeddings.
- `append_frame(img, ctrl)`: encode given image and update cache.
- `gen_frame(ctrl, return_img)`: generate next frame, update cache, optionally decode.

### WorldModel (DiT Transformer)
Architecture is a patchified Diffusion Transformer with:
- RoPE positional encoding over time + spatial axes.
- Controller conditioning via MLP fusion at periodic layers.
- Prompt conditioning via cross-attention at periodic layers.
- Conditioning on diffusion sigma via `NoiseConditioner` + AdaLN/AdaRMSNorm.
- KV cache for sparse attention over past frames.

`WorldModel.forward` expects `B==1` and `N==1` (single frame).

### AutoEncoder (InferenceAE)
Encoder/decoder pair loaded from safetensors:
- Input: uint8 image `[H, W, C]` -> normalized -> encode to latent.
- Output: latent -> decode -> uint8 `[H, W, 3]`.
- Uses weight-norm layers; `bake_weight_norm_` removes parametrizations at load.

### PromptEncoder (Text Conditioning)
Uses HF Transformers `UMT5EncoderModel`:
- Tokenizer + encoder with padding to max length 512.
- Returns embeddings + padding mask for cross-attention.

### KV Cache
Ring-buffer cache with block-sparse attention masks:
- One cache per layer.
- Local and global attention windows (configurable per layer).
- Sparse `BlockMask` from `torch.nn.attention.flex_attention`.

### Inference Patches
Applied in `apply_inference_patches`:
- Cached sigma embeddings via LUT (`CachedDenoiseStepEmb`).
- Cached conditioning head outputs (`CachedCondHead`).
- Merge QKV into a single projection (`MergedQKVAttn`).
- Split MLPFusion into separate linears (quantization-friendly).

### Quantization (Optional)
Quantizes linear layers:
- NVFP4 via FlashInfer (CUDA-only).
- FP8 (Torch built-ins).
Not active unless `quant` is passed to `WorldEngine`.

## Data Flow (Inference)

1) `WorldEngine.gen_frame`
   - Create noise `x` with shape `[1, 1, C, H, W]`.
   - Prepare controller inputs and prompt embeddings.
   - Denoise using scheduler sigmas and the World Model.
   - Update KV cache with the new frame.
   - Decode to image via VAE (optional).

2) `WorldEngine.append_frame`
   - Encode given image into latent.
   - Update KV cache with that latent (no denoise).

3) KV cache policy
   - Tail slice always holds the current frame.
   - Ring buffer stores past frames with optional global attention stride.
   - `BlockMask` determines which keys/values are visible to attention.

## Tensor Shapes and Conventions

- Image input: `[H, W, C]` uint8.
- Latent frame: `[B, N, C, H, W]` (in practice `B=N=1`).
- Tokens per frame: `tokens_per_frame = height * width` (post-patch).
- Patchify: Conv2d with kernel/stride `patch` -> tokens.
- Positional IDs: `t_pos`, `y_pos`, `x_pos` shaped `[1, T]`.
- Attention tensors: Q/K/V shape `[B, H, T, Dh]`.
- Controller inputs: button one-hot `[1, 1, n_buttons]`, mouse `[1, 1, 2]`,
  scroll `[1, 1, 1]`.

## Configuration Fields Used
Fields are read from `config.yaml` loaded via `OmegaConf`. The following
attributes are referenced in code:

- Model core: `d_model`, `n_heads`, `n_kv_heads`, `n_layers`, `mlp_ratio`.
- Input geometry: `channels`, `height`, `width`, `patch`, `tokens_per_frame`.
- Conditioning: `noise_conditioning`, `prompt_conditioning`,
  `prompt_embedding_dim`, `prompt_conditioning_period`,
  `ctrl_conditioning`, `ctrl_conditioning_period`,
  `ctrl_cond_dropout`, `prompt_cond_dropout`, `n_buttons`.
- Attention windows: `local_window`, `global_window`,
  `global_attn_period`, `global_attn_offset`, `global_pinned_dilation`.
- Scheduler: `scheduler_sigmas` (list of floats).
- Positional encoding: `n_frames` (used in RoPE).
- Misc: `value_residual`, `gated_attn`, `has_audio` (asserted false).

## Torch-Specific Features to Replace for MLX

The following are hard dependencies on PyTorch/CUDA that must be ported or
replaced for MLX:

1) Compilation and Dynamo
   - `torch.compile` and `torch._dynamo.config` tuning are used in multiple
     hot paths (`_denoise_pass`, `_cache_pass`, AE decode, prompt encoder).

2) Flex Attention + BlockMask
   - `torch.nn.attention.flex_attention` and `BlockMask` are used for sparse
     attention with KV cache; MLX needs equivalent sparse attention or a dense
     fallback.

3) CUDA Autocast + dtype assumptions
   - Many kernels assume `torch.bfloat16`, and AE decode explicitly uses
     `torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)`.
   - Cached conditioning relies on unique bf16 bit patterns.

4) CUDA-only quantization paths
   - FlashInfer NVFP4 custom op (`torch.library.custom_op`), plus FP8 paths
     using `torch._scaled_mm`, `torch.float8_e4m3fn`.

5) Torch-only ops & utilities
   - `torch._assert`, `torch._check`, `torch._scaled_mm`.
   - `torch.nn.utils.parametrize` and `weight_norm`.
   - `tensordict.TensorDict` used for positional IDs.
   - `einops` for rearrange; MLX equivalents required.

6) HF Transformers + Torch-based UMT5
   - `UMT5EncoderModel` is Torch-based. For a full MLX port, prompt encoding
     must be replaced with an MLX-native text encoder or kept in Torch with
     interop costs.

## Porting Notes (MLX Planning Checklist)

### Core Inference
- Replace `torch.compile` usage with MLX-appropriate compilation or remove.
- Implement RoPE and attention in MLX, including:
  - OrthoRoPE (time + 2D spatial).
  - Block-sparse attention or an approximation for KV cache.
- Recreate KV cache behavior:
  - Ring buffer with tail slice for current frame.
  - Per-layer local/global windows.
  - Pinned dilation for global attention layers.

### Conditioning & Scheduling
- Reimplement `NoiseConditioner` (Fourier features + MLP).
- Reimplement AdaRMSNorm and AdaLN logic.
- Ensure `scheduler_sigmas` iteration and update (diff-based Euler).
- Cached conditioning optimizations are optional but assume bf16 support.

### AutoEncoder
- Port AE blocks (weight-norm, pixel shuffle/unshuffle, bicubic interpolate).
- `LandscapeToSquare` and `SquareToLandscape` hardcode 512x512 and 360x640;
  ensure MLX interpolation matches Torch bicubic behavior.

### Text Encoder
- Decide on MLX-native UMT5 or a Torch-only sidecar:
  - If Torch remains, define a stable handoff format for embeddings + masks.
  - If MLX, ensure tokenizer and padding match HF behavior.

### Data Types
- Audit all bf16 assumptions:
  - Cached LUT lookup relies on bf16 bit patterns.
  - Quantization paths assume float8 support.
  - If MLX prefers fp16, disable cached conditioning or adjust lookup logic.

### Quantization
- NVFP4 and FP8 are CUDA-specific.
- For MLX, either:
  - Skip quantization entirely.
  - Implement MLX-specific quantization kernels.

### Loader + Checkpoints
- `BaseModel.from_pretrained` uses HF Hub snapshot download and safetensors.
- For macOS native usage, ensure HF download path or local folder support
  works without Torch device moves.

## External Dependencies
Key runtime deps (from `pyproject.toml`):
- `torch`, `torchvision`, `torchaudio` (CUDA test wheels).
- `einops`, `tensordict`, `transformers`, `diffusers`, `accelerate`,
  `huggingface-hub`, `omegaconf`, `ftfy`.
- `triton`/`triton-windows` (platform-specific).

## Example-Only Utilities
The `examples/` folder uses OpenCV and PyTest for rendering/benchmarking and is
not part of the core package API.

## Port Risk Areas
- Sparse attention via `flex_attention` (no MLX equivalent out of the box).
- Heavy reliance on bf16; cached LUT depends on bf16 bit identity.
- CUDA-only quantization and custom ops.
- Transformer text encoder dependency on Torch.
- Use of `torch.compile` and `_dynamo` for performance.

## MLX Port Plan (Milestones)

### Milestone 0: Feasibility + Baseline
- Confirm target MLX version and macOS hardware constraints.
- Decide dtype strategy (bf16 vs fp16) and autocast equivalents.
- Establish a small golden test set (1-2 prompts + fixed controller inputs)
  to validate visual parity and latent stats.

### Milestone 1: Core Model Forward in MLX (No KV Cache)
- Reimplement core tensor ops in MLX: linear, conv2d, layer norms, MLPs,
  RMSNorm, AdaLN/AdaRMSNorm, SiLU.
- Implement patchify/unpatchify and WorldDiT blocks.
- Implement OrthoRoPE in MLX.
- Implement dense attention (no cache, no block sparsity) for correctness.
- Replace `TensorDict` usage with MLX-native structures.

Exit criteria:
- Single-frame denoise pass produces numerically stable outputs vs Torch
  (approximate match; accept tolerance).

### Milestone 2: Scheduler + End-to-End Denoise Loop
- Port the sigma scheduler loop (`scheduler_sigmas` + `diff`) and Euler update.
- Integrate controller conditioning and prompt conditioning logic in MLX.
- Remove `torch.compile` usage; consider MLX compile/jit if available.

Exit criteria:
- End-to-end `gen_frame` works on MLX (return latent only).

### Milestone 3: KV Cache + Sparse Attention Strategy
- Rebuild ring-buffer KV cache structure in MLX.
- Decide on attention policy:
  - Option A: dense attention with windowed truncation (simpler).
  - Option B: custom sparse attention to match `flex_attention` + `BlockMask`.
- Implement mask logic and validate attention coverage matches Torch.

Exit criteria:
- Multi-frame rollout works with caching and stable memory usage.

### Milestone 4: AutoEncoder Port
- Port AE blocks (weight norm, pixel shuffle/unshuffle, bicubic up/downsample).
- Validate encode/decode outputs against Torch for a few images.

Exit criteria:
- End-to-end `gen_frame` returns decoded uint8 image.

### Milestone 5: Prompt Encoder Strategy
- Choose and implement:
  - MLX-native UMT5 encoder (preferred for pure MLX).
  - Torch sidecar for prompt encoding (fastest path, but cross-runtime).
- Match HF tokenizer behavior (padding, truncation to 512).

Exit criteria:
- Prompt conditioning behavior is equivalent across several prompts.

### Milestone 6: Optimization + Optional Features
- Optional cached conditioning (LUT) if bf16 bit identity is reliable in MLX.
- Optional quantization (MLX-specific; likely new path).
- Profiling and performance tuning.

Exit criteria:
- Inference throughput within acceptable range vs baseline.

## Validation Plan
- Numerical checks: latent mean/variance per layer, output pixel histograms.
- Functional tests: deterministic frames for fixed seeds and inputs.
- Performance tests: frames/sec on M-series GPU for N-frame rollout.
