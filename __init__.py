import sys
from os import path
sys.path.insert(0, path.dirname(__file__))
from .ldsrlib.LDSR import LDSR
from folder_paths import get_filename_list, get_full_path
from comfy.model_management import get_torch_device
from comfy.utils import ProgressBar

class LDSRUpscaler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (get_filename_list("upscale_models"),),
                "images": ("IMAGE",),
                "steps": (["25", "50", "100", "250", "500", "1000"], {"default": "100"}),
                "pre_downscale": (['None', '1/2', '1/4'],{"default": "None"}),
                "post_downscale": (['None', 'Original Size', '1/2', '1/4'],{"default": "None"}),
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
        p = {"prev":0}

        def prog(i):
            i = i + 1
            if i < p["prev"]:
                p["prev"] = 0
            pbar.update(i - p["prev"])
            p["prev"] = i

        ldsr = LDSR(model_path, get_torch_device(), prog)

        outputs = []

        for image in images:
            outputs.append(ldsr.superResolution(image, int(steps), pre_downscale, post_downscale, downsample_method))

        return (outputs, )


NODE_CLASS_MAPPINGS = {
    "LDSRUpscaler": LDSRUpscaler
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LDSRUpscaler": "LDSR Upscale"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
