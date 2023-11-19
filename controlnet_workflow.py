import os
import cv2
import numpy as np
import random
import sys
from typing import Sequence, Mapping, Any, Union
import torch


def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    """Returns the value at the given index of a sequence or mapping.

    If the object is a sequence (like list or string), returns the value at the given index.
    If the object is a mapping (like a dictionary), returns the value at the index-th key.

    Some return a dictionary, in these cases, we look for the "results" key

    Args:
        obj (Union[Sequence, Mapping]): The object to retrieve the value from.
        index (int): The index of the value to retrieve.

    Returns:
        Any: The value at the given index.

    Raises:
        IndexError: If the index is out of bounds for the object and the object is not a mapping.
    """
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]


def find_path(name: str, path: str = None) -> str:
    """
    Recursively looks at parent folders starting from the given path until it finds the given name.
    Returns the path as a Path object if found, or None otherwise.
    """
    # If no path is given, use the current working directory
    if path is None:
        path = os.getcwd()

    # Check if the current directory contains the name
    if name in os.listdir(path):
        path_name = os.path.join(path, name)
        print(f"{name} found: {path_name}")
        return path_name

    # Get the parent directory
    parent_directory = os.path.dirname(path)

    # If the parent directory is the same as the current directory, we've reached the root and stop the search
    if parent_directory == path:
        return None

    # Recursively call the function with the parent directory
    return find_path(name, parent_directory)


def add_comfyui_directory_to_sys_path() -> None:
    """
    Add 'ComfyUI' to the sys.path
    """
    comfyui_path = find_path("ComfyUI")
    if comfyui_path is not None and os.path.isdir(comfyui_path):
        sys.path.append(comfyui_path)
        print(f"'{comfyui_path}' added to sys.path")


def add_extra_model_paths() -> None:
    """
    Parse the optional extra_model_paths.yaml file and add the parsed paths to the sys.path.
    """
    from main import load_extra_path_config

    extra_model_paths = find_path("extra_model_paths.yaml")

    if extra_model_paths is not None:
        load_extra_path_config(extra_model_paths)
    else:
        print("Could not find the extra_model_paths config file.")


add_comfyui_directory_to_sys_path()
add_extra_model_paths()


def import_custom_nodes() -> None:
    """Find all custom nodes in the custom_nodes folder and add those node objects to NODE_CLASS_MAPPINGS

    This function sets up a new asyncio event loop, initializes the PromptServer,
    creates a PromptQueue, and initializes the custom nodes.
    """
    import asyncio
    import execution
    from nodes import init_custom_nodes
    import server

    # Creating a new event loop and setting it as the default loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Creating an instance of PromptServer with the loop
    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)

    # Initializing custom nodes
    init_custom_nodes()


from nodes import (
    NODE_CLASS_MAPPINGS,
    LoraLoader,
    VAEDecode,
    EmptyLatentImage,
    CLIPTextEncode,
    ControlNetApplyAdvanced,
    CheckpointLoaderSimple,
    ControlNetLoader,
    KSampler,
    SaveImage,
)


import_custom_nodes()


class LCMControlnetPipeline:
    def __init__(self, ckpt_name, lcm_lora_name, control_net_name, negative_prompt):
        with torch.inference_mode():
            self.checkpointloadersimple = CheckpointLoaderSimple()
            self.checkpointloadersimple_3 = self.checkpointloadersimple.load_checkpoint(
                ckpt_name=ckpt_name
            )

            self.loraloader = LoraLoader()
            self.loraloader_4 = self.loraloader.load_lora(
                lora_name=lcm_lora_name,
                strength_model=1,
                strength_clip=1,
                model=get_value_at_index(self.checkpointloadersimple_3, 0),
                clip=get_value_at_index(self.checkpointloadersimple_3, 1),
            )

            self.cliptextencode = CLIPTextEncode()
            self.cliptextencode_9 = self.cliptextencode.encode(
                text=negative_prompt,
                clip=get_value_at_index(self.checkpointloadersimple_3, 1),
            )

            self.controlnetloader = ControlNetLoader()
            self.controlnetloader_13 = self.controlnetloader.load_controlnet(
                control_net_name=control_net_name
            )

            self.emptylatentimage = EmptyLatentImage()

            self.modelsamplingdiscrete = NODE_CLASS_MAPPINGS["ModelSamplingDiscrete"]()
            self.manga2anime_lineart_preprocessor = NODE_CLASS_MAPPINGS[
                "Manga2Anime_LineArt_Preprocessor"
            ]()
            self.controlnetapplyadvanced = ControlNetApplyAdvanced()
            self.ksampler = KSampler()
            self.vaedecode = VAEDecode()
            self.saveimage = SaveImage()

    def __call__(self, np_img, prompt, control_weight):
        with torch.inference_mode():
            np_img = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)  # bgr -> rgb
            np_img = np_img.astype(np.float32) / 255.0
            t_img = torch.from_numpy(np_img)[None,]
            mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
            loadimage_11 = (t_img, mask.unsqueeze(0))

            emptylatentimage_25 = self.emptylatentimage.generate(
                width=np_img.shape[1], height=np_img.shape[0], batch_size=1
            )
            manga2anime_lineart_preprocessor_23 = (
                self.manga2anime_lineart_preprocessor.execute(
                    image=get_value_at_index(loadimage_11, 0)
                )
            )

            cliptextencode_6 = self.cliptextencode.encode(
                text=prompt, clip=get_value_at_index(self.checkpointloadersimple_3, 1)
            )

            modelsamplingdiscrete_5 = self.modelsamplingdiscrete.patch(
                sampling="lcm",
                zsnr=False,
                model=get_value_at_index(self.loraloader_4, 0),
            )

            controlnetapplyadvanced_16 = self.controlnetapplyadvanced.apply_controlnet(
                strength=control_weight,
                start_percent=0,
                end_percent=1,
                positive=get_value_at_index(cliptextencode_6, 0),
                negative=get_value_at_index(self.cliptextencode_9, 0),
                control_net=get_value_at_index(self.controlnetloader_13, 0),
                image=get_value_at_index(manga2anime_lineart_preprocessor_23, 0),
            )

            ksampler_1 = self.ksampler.sample(
                seed=random.randint(1, 2**64),
                steps=8,
                cfg=1.8,
                sampler_name="lcm",
                scheduler="simple",
                denoise=1.0,
                model=get_value_at_index(modelsamplingdiscrete_5, 0),
                positive=get_value_at_index(controlnetapplyadvanced_16, 0),
                negative=get_value_at_index(controlnetapplyadvanced_16, 1),
                latent_image=get_value_at_index(emptylatentimage_25, 0),
            )

            vaedecode_2 = self.vaedecode.decode(
                samples=get_value_at_index(ksampler_1, 0),
                vae=get_value_at_index(self.checkpointloadersimple_3, 2),
            )
            manga2anime_lineart_preprocessor_22 = (
                self.manga2anime_lineart_preprocessor.execute(
                    image=get_value_at_index(vaedecode_2, 0)
                )
            )

            color_image = vaedecode_2[0]
            color_image = 255.0 * color_image[0].cpu().numpy()
            color_image = np.clip(color_image, 0, 255).astype(np.uint8)
            line_image = manga2anime_lineart_preprocessor_22[0]
            line_image = 255.0 * line_image[0].cpu().numpy()
            line_image = np.clip(line_image, 0, 255).astype(np.uint8)
            line_image = 255 - line_image

            return color_image, line_image
