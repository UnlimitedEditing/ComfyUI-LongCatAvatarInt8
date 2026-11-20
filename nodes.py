"""Nodes for LongCat-Video-Avatar 1.6 on the native ComfyUI int8 ConvRot path.

LongCatAvatarInt8ToVideo is the single-speaker, image-to-video part of Comfy-Org/ComfyUI#17831's LongCatAvatarToVideo
(same conditioning keys, so the ported model sees exactly what it would in core).
"""
import torch

import comfy.model_management
import comfy.utils
import node_helpers
from comfy_extras.nodes_wan import linear_interpolation
from comfy.ldm.wan.model_multitalk import get_audio_embeds


class LongCatAvatarInt8ToVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "positive": ("CONDITIONING",),
            "negative": ("CONDITIONING",),
            "vae": ("VAE",),
            "start_image": ("IMAGE", {"tooltip": "The speaker. Resized and centre-cropped to width x height."}),
            "audio_encoder_output": ("AUDIO_ENCODER_OUTPUT", {"tooltip": "Whisper large-v3 encoding of the speech; the video runs at 25 fps."}),
            "width": ("INT", {"default": 432, "min": 16, "max": 4096, "step": 16}),
            "height": ("INT", {"default": 240, "min": 16, "max": 4096, "step": 16}),
            "length": ("INT", {"default": 93, "min": 5, "max": 4097, "step": 4, "tooltip": "Frames at 25 fps, snapped to 1 + 4k."}),
            "audio_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01,
                                         "tooltip": "1.0 is the trained behaviour; other values cost no extra pass."}),
        }}

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "build"
    CATEGORY = "conditioning/video_models/longcat_int8"

    def build(self, positive, negative, vae, start_image, audio_encoder_output, width, height, length, audio_strength):
        width, height = max(16, width // 16 * 16), max(16, height // 16 * 16)
        length = ((length - 1) // 4) * 4 + 1
        img = comfy.utils.common_upscale(start_image[:1].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)[:, :, :, :3]
        cond = vae.encode(img)
        latent = torch.zeros([1, 16, ((length - 1) // 4) + 1, height // 8, width // 8], device=comfy.model_management.intermediate_device())

        # four 8-layer means and the last layer, resampled from Whisper's 50 Hz to the video's 25 fps (as in core's node)
        layers = torch.stack(audio_encoder_output["encoded_audio_all_layers"], dim=2)
        feats = [layers[:, :, i:i + 8].mean(dim=2) for i in range(0, 32, 8)] + [layers[:, :, 32]]
        encoded = [torch.stack([linear_interpolation(f, 50, 25) for f in feats], dim=2)[0]]
        frames = max(length, encoded[0].shape[0])
        encoded = [torch.cat([e, e[-1:].expand(frames - e.shape[0], -1, -1)]) for e in encoded]
        audio_embed = get_audio_embeds(encoded, 0, length).unsqueeze(0)

        values = {"audio_strength": audio_strength, "cond_latents": cond}
        pos = node_helpers.conditioning_set_values(positive, {**values, "audio_embed": audio_embed})
        neg = node_helpers.conditioning_set_values(negative, {**values, "audio_embed": torch.zeros_like(audio_embed)})
        return (pos, neg, {"samples": latent})


class LongCatAvatarInt8DistillSigmas:
    """The model card's 8-step distilled schedule (indices into a 50-step table, then the flow shift), split for the DMD -> OPSD mix."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "steps": ("INT", {"default": 8, "min": 2, "max": 50}),
            "shift": ("FLOAT", {"default": 12.0, "min": 1.0, "max": 100.0, "step": 0.1}),
            "mix_steps": ("INT", {"default": 4, "min": 0, "max": 50, "tooltip": "Steps 0..mix_steps-1 are the first stage (DMD LoRA), the rest the second (OPSD LoRA)."}),
        }}

    RETURN_TYPES = ("SIGMAS", "SIGMAS")
    RETURN_NAMES = ("stage1_sigmas", "stage2_sigmas")
    FUNCTION = "build"
    CATEGORY = "sampling/custom_sampling/schedulers"

    def build(self, steps, shift, mix_steps):
        table = torch.arange(1, 51, dtype=torch.float32) * 20.0
        idx = torch.floor(torch.linspace(0, 50, steps + 1)[:-1]).long()
        s = torch.flip(table, [0])[idx] / 1000.0
        s = shift * s / (1.0 + (shift - 1.0) * s)
        sigmas = torch.cat([s, s.new_zeros(1)])
        mix = max(0, min(mix_steps, steps))
        if mix == 0:
            return (sigmas[:1], sigmas)
        if mix == steps:
            return (sigmas, sigmas[-1:])
        return (sigmas[:mix + 1], sigmas[mix:])


NODE_CLASS_MAPPINGS = {
    "LongCatAvatarInt8ToVideo": LongCatAvatarInt8ToVideo,
    "LongCatAvatarInt8DistillSigmas": LongCatAvatarInt8DistillSigmas,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LongCatAvatarInt8ToVideo": "LongCat Avatar (int8) To Video",
    "LongCatAvatarInt8DistillSigmas": "LongCat Avatar Distill Sigmas",
}
