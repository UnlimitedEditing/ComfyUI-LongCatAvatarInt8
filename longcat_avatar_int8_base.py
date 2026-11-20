# Model-base class for LongCat-Video-Avatar, ported from Comfy-Org/ComfyUI#17831 (comfy/model_base.py LongCatVideo).
import math

import torch

import comfy.conds
import comfy.model_management
import comfy.ldm.qwen_image21.model
from comfy.model_base import WAN21, ModelType

from .longcat_avatar_int8_dit import LongCatAvatarInt8Model


class LongCatAvatarInt8Base(WAN21):
    def __init__(self, model_config, model_type=ModelType.FLOW, device=None):
        super(WAN21, self).__init__(model_config, model_type, device=device, unet_model=LongCatAvatarInt8Model)

    @property
    def current_patcher(self):
        return self._current_patcher

    @current_patcher.setter
    def current_patcher(self, patcher):
        # set by pre_run / cleanup: the condition frame cache lives for one sampling run, off when hooks can repatch weights mid-run
        self._current_patcher = patcher
        diffusion_model = getattr(self, "diffusion_model", None)
        if diffusion_model is not None:
            diffusion_model.reset_cond_cache(patcher if patcher is not None and len(patcher.hook_patches) == 0 else None)

    def get_writable_model(self, conds, noise_shape, model_options, device):
        # auto mode with dynamic VRAM: the condition stream's block inputs go in writable slots managed like the weights, one per distinct
        # set of condition frames; the uncond shares its cond's slot, that stream sees no text or audio
        options = model_options.get("transformer_options", {}).get("longcat_cache", {})
        model = self.diffusion_model
        if options.get("device", "auto") != "auto" or model.cache_patcher is None:
            return None

        pt, ph, pw = model.patch_size
        slots, assigned = [], []
        for conditions in conds.values():
            for cond in conditions:
                c = cond["model_conds"]
                if "cond_latents" not in c:
                    continue
                frames = [c["cond_latents"].cond] + ([c["reference_latent"].cond.unsqueeze(2)] if "reference_latent" in c else [])
                slot = next((s for f, s in slots if len(f) == len(frames) and all(torch.equal(a, b) for a, b in zip(f, frames))), None)
                if slot is None:
                    tokens = sum(math.ceil(f.shape[2] / pt) for f in frames) * math.ceil(frames[0].shape[3] / ph) * math.ceil(frames[0].shape[4] / pw)
                    slot = comfy.ldm.qwen_image21.model.PrefixCacheSlot((1, tokens, model.dim), len(model.blocks), self.get_dtype_inference(), options.get("dtype", "default"))
                    slots.append((frames, slot))
                assigned.append((cond, slot))
        if not slots:
            return None

        cache = comfy.ldm.qwen_image21.model.KVCacheModel([slot for _, slot in slots])
        cache_bytes = sum(p.numel() * p.element_size() for p in cache.parameters())
        if not (self.current_patcher.get_free_memory(device) > 4 * cache_bytes or
                comfy.model_management.ensure_pin_budget(cache_bytes, evict_active=False)):
            return None

        for cond, slot in assigned:
            cond["model_conds"] = c = cond["model_conds"].copy()
            c.setdefault("cond_cache_slots", comfy.conds.CONDKVCache({})).cond[device] = slot
        return cache

    def _apply_model(self, x, t, c_concat=None, c_crossattn=None, control=None, transformer_options={}, **kwargs):
        denoised = super()._apply_model(x, t, c_concat, c_crossattn, control, transformer_options, **kwargs)
        # condition frames are not sampled: they denoise to the clean latents whatever state a split sampler hands over
        cond_latents = kwargs.get("cond_latents", None)
        if cond_latents is not None:
            denoised[:, :, :cond_latents.shape[2]] = cond_latents.to(denoised)
        return denoised

    def extra_conds(self, **kwargs):
        out = super().extra_conds(**kwargs)
        cond_latents = kwargs.get("cond_latents", None)
        if cond_latents is not None:
            out['cond_latents'] = comfy.conds.CONDRegular(self.process_latent_in(cond_latents))
        audio_embed = kwargs.get("audio_embed", None)
        if audio_embed is not None:
            out['audio_embed'] = comfy.conds.CONDRegular(audio_embed)
        for name in ("ref_frame_index", "ref_mask_frame_range", "speaker_masks", "audio_strength"):
            value = kwargs.get(name, None)
            if value is not None:
                out[name] = comfy.conds.CONDConstant(value)
        return out
