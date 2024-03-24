import sys
from os import path
sys.path.insert(0, path.dirname(__file__))
from .ldsrlib.LDSR import LDSR
from folder_paths import get_filename_list, get_full_path
from comfy.model_management import get_torch_device
from comfy.utils import ProgressBar
import torch


class LDSRModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        model_list = get_filename_list("upscale_models")
        candidates = [name for name in model_list if 'last.ckpt' in name]
        if len(candidates) > 0:
            default_path = candidates[0]
        else:
            default_path = 'last.ckpt'

        return {
            "required": {
                "model": (model_list, {'default': default_path}),
            }
        }

    RETURN_TYPES = ("UPSCALE_MODEL",)
    FUNCTION = "load"

    CATEGORY = "Flowty LDSR"

    def load(self, model):
        model_path = get_full_path("upscale_models", model)
        model = LDSR.load_model_from_path(model_path)
        model['model'].cpu()
        return (model, )


class LDSRUpscale:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "upscale_model": ("UPSCALE_MODEL",),
                "images": ("IMAGE",),
                "steps": (["6", "12", "25", "50", "100", "250", "500", "1000"], {"default": "100"}),
                "pre_downscale": (['None', '1/2', '1/4'], {"default": "None"}),
                "post_downscale": (['None', 'Original Size', '1/2', '1/4'], {"default": "None"}),
                "downsample_method": (['Nearest', 'Lanczos'], {"default": "Lanczos"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "upscale"

    CATEGORY = "Flowty LDSR"

    def upscale(self, upscale_model, images, steps, pre_downscale="None", post_downscale="None", downsample_method="Lanczos"):
        pbar = ProgressBar(int(steps))
        p = {"prev": 0}

        def prog(i):
            i = i + 1
            if i < p["prev"]:
                p["prev"] = 0
            pbar.update(i - p["prev"])
            p["prev"] = i

        ldsr = LDSR(model=upscale_model, on_progress=prog)

        outputs = []

        for image in images:
            outputs.append(ldsr.superResolution(image, int(steps), pre_downscale, post_downscale, downsample_method))

        return (torch.stack(outputs),)


class LDSRUpscaler:
    @classmethod
    def INPUT_TYPES(s):
        model_list = get_filename_list("upscale_models")
        candidates = [name for name in model_list if 'last.ckpt' in name]
        if len(candidates) > 0:
            default_path = candidates[0]
        else:
            default_path = 'last.ckpt'

        return {
            "required": {
                "model": (model_list, {'default': default_path}),
                "images": ("IMAGE",),
                "steps": (["6", "12", "25", "50", "100", "250", "500", "1000"], {"default": "100"}),
                "pre_downscale": (['None', '1/2', '1/4'], {"default": "None"}),
                "post_downscale": (['None', 'Original Size', '1/2', '1/4'], {"default": "None"}),
                "downsample_method": (['Nearest', 'Lanczos'], {"default": "Lanczos"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "upscale"

    CATEGORY = "Flowty LDSR"

    def upscale(self, model, images, steps, pre_downscale="None", post_downscale="None", downsample_method="Lanczos"):
        model_path = get_full_path("upscale_models", model)
        pbar = ProgressBar(int(steps))
        p = {"prev": 0}

        def prog(i):
            i = i + 1
            if i < p["prev"]:
                p["prev"] = 0
            pbar.update(i - p["prev"])
            p["prev"] = i

        ldsr = LDSR(modelPath=model_path, torchdevice=get_torch_device(), on_progress=prog)

        outputs = []

        for image in images:
            outputs.append(ldsr.superResolution(image, int(steps), pre_downscale, post_downscale, downsample_method))

        return (torch.stack(outputs),)


NODE_CLASS_MAPPINGS = {
    "LDSRUpscaler": LDSRUpscaler,
    "LDSRModelLoader": LDSRModelLoader,
    "LDSRUpscale": LDSRUpscale
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LDSRUpscaler": "LDSR Upscale (all-in-one)",
    "LDSRModelLoader": "Load LDSR Model",
    "LDSRUpscale": "LDSR Upscale"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
