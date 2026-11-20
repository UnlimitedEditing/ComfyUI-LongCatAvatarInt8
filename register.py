"""Registers LongCat-Video-Avatar with a ComfyUI core that does not have it yet (Comfy-Org/ComfyUI#17831).

Nothing in core is edited: the detection hook and the supported-model class are added at import time. If core already ships
LongCat (supported_models.LongCatVideo exists) this does nothing and core's own implementation is used.
"""
import logging

import torch

import comfy.model_detection
import comfy.supported_models as supported_models

_DONE = False


def register():
    global _DONE
    if _DONE:
        return "already"
    if hasattr(supported_models, "LongCatVideo"):
        logging.info("[LongCatAvatarInt8] core already supports LongCat Avatar, not registering")
        _DONE = True
        return "core"

    from .longcat_avatar_int8_base import LongCatAvatarInt8Base

    class LongCatAvatarInt8Config(supported_models.WAN21_T2V):
        unet_config = {"image_model": "longcat_video"}
        sampling_settings = {"shift": 7.0}
        supported_inference_dtypes = [torch.bfloat16, torch.float32]

        def get_model(self, state_dict, prefix="", device=None):
            return LongCatAvatarInt8Base(self, device=device)

    original = comfy.model_detection.detect_unet_config

    def detect_unet_config(state_dict, key_prefix, *args, **kwargs):
        keys = state_dict.keys() if hasattr(state_dict, "keys") else state_dict
        if f"{key_prefix}blocks.0.ffn.w1.weight" in keys and f"{key_prefix}multitalk_audio_proj.proj1.weight" in keys:
            return {"image_model": "longcat_video"}  # LongCat-Video-Avatar
        return original(state_dict, key_prefix, *args, **kwargs)

    comfy.model_detection.detect_unet_config = detect_unet_config
    supported_models.models.insert(0, LongCatAvatarInt8Config)
    _DONE = True
    logging.info("[LongCatAvatarInt8] registered LongCat Avatar model (native int8 ConvRot path)")
    return "pack"
