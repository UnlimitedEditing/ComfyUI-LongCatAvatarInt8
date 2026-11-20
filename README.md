# ComfyUI-LongCatAvatarInt8

Runs Meituan's LongCat-Video-Avatar 1.6 **natively on Comfy-Org's ConvRot int8 weights**
(`Comfy-Org/LongCat-Video-Avatar` -> `diffusion_models/longcat-avatar-1.6_int8_convrot.safetensors`), with no int8 -> fp8 conversion.
The int8 linears run through core's `int8_tensorwise` ConvRot layers (comfy-kitchen `int8_linear`).

It is a port of Comfy-Org/ComfyUI#17831 (`kijai/ComfyUI` branch `longcat_avatar`) for a ComfyUI that does not have it yet.
Nothing in core is edited: the DiT (`longcat_avatar_int8_dit.py`), model base class and supported-model entry are registered
at import time, plus a detection hook for the checkpoint. If core already ships `supported_models.LongCatVideo` the pack steps aside.

Nodes (unique names, no clash with the PR's `LongCatAvatarToVideo`):
- `LongCatAvatarInt8ToVideo` single speaker image + Whisper audio -> conditioning + empty latent
- `LongCatAvatarInt8DistillSigmas` the 8-step distilled schedule, split at `mix_steps` for the DMD -> OPSD LoRA mix

Graph: `UNETLoader` -> `LoraLoaderModelOnly` (DMD) / (OPSD) -> two `SamplerCustomAdvanced` (euler, cfg 1, `BasicGuider`), the second with
`DisableNoise`; `CLIPLoader(type=wan)` umT5; `AudioEncoderLoader` Whisper large-v3; `VAELoader` wan_2.1_vae.
Needs comfy-kitchen >= 0.2.41 and a core with `comfy.ldm.pan2`, `qwen_image21`, `model_animate2` (ComfyUI >= 0.46).
Single speaker / single clip only (no multi-speaker masks, no continuation).
