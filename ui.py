import gradio as gr
import numpy as np
from typing import Optional
from ComfyUI.controlnet_workflow import LCMControlnetPipeline
from PIL import Image


img_width = 512
img_height = 742
white_image = Image.fromarray(np.ones((img_height, img_width, 3), dtype=np.uint8) * 255)
init_image_path = "init.png"
white_image.save(init_image_path)
pipeline = LCMControlnetPipeline(
    ckpt_name="",
    lcm_lora_name="",
    control_net_name="",
    negative_prompt="",
)


def process_image(
    np_img: Optional[np.ndarray],
    prompt: str,
    control_weight: float,
):
    if np_img is None:
        return
    color_img, line_img = pipeline(np_img, prompt, control_weight)
    return color_img, line_img


with gr.Blocks() as ui:
    prompt_input = gr.Textbox(label="prompt", value="1girl")
    c_weight_input = gr.Slider(minimum=0, maximum=1, value=0.3, label="control weight")
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(
                source="upload",
                tool="color-sketch",
                value=init_image_path,
                width=img_width,
                height=img_height,
                interactive=True,
            )
        with gr.Column():
            image_line_output = gr.Image(width=img_width, height=img_height)
        with gr.Column():
            image_color_output = gr.Image(width=img_width, height=img_height)

    image_input.change(
        fn=process_image,
        inputs=[image_input, prompt_input, c_weight_input],
        outputs=[image_color_output, image_line_output],
        show_progress="hidden",
    )
    prompt_input.change(
        fn=process_image,
        inputs=[image_input, prompt_input, c_weight_input],
        outputs=[image_color_output, image_line_output],
        show_progress="hidden",
    )
    c_weight_input.change(
        fn=process_image,
        inputs=[image_input, prompt_input, c_weight_input],
        outputs=[image_color_output, image_line_output],
        show_progress="hidden",
    )

ui.queue()
ui.launch()
