import gradio as gr
import pandas as pd
from hub_utils import check_for_discussion, report_results
from model_utils import calculate_memory, get_model
from huggingface_hub.utils import HfHubHTTPError


def get_results(model_name: str, library: str, options: list, access_token: str):
    model = get_model(model_name, library, access_token)
    try:
        has_discussion = check_for_discussion(model_name)
    except HfHubHTTPError:
        has_discussion = True
    title = f"## Memory usage for '{model_name}'"
    data = calculate_memory(model, options)
    return [title, gr.update(visible=True, value=pd.DataFrame(data)), gr.update(visible=not has_discussion)]


with gr.Blocks() as demo:
    with gr.Column():
        gr.Markdown(
            "..."
        )
        out_text = gr.Markdown()
        out = gr.DataFrame(
            headers=["dtype", "Largest Layer", "Total Size", "Training using Adam"],
            interactive=False,
            visible=False,
        )
        with gr.Row():
            inp = gr.Textbox(label="Model Name or URL", value="bert-base-cased")
        with gr.Row():
            library = gr.Radio(["auto", "transformers", "timm"], label="Library", value="auto")
            options = gr.CheckboxGroup(
                ["float32", "float16/bfloat16", "int8", "int4"],
                value="float32",
                label="Model Precision",
            )
            access_token = gr.Textbox(label="API Token", placeholder="Optional (for gated models)")
        with gr.Row():
            btn = gr.Button("Calculate Memory Usage")
            post_to_hub = gr.Button(
                value="Report results in this model repo's discussions!\n(Will open in a new tab)", visible=False
            )

    btn.click(
        get_results,
        inputs=[inp, library, options, access_token],
        outputs=[out_text, out, post_to_hub],
        api_name=False,
    )

    post_to_hub.click(lambda: gr.Button.update(visible=False), outputs=post_to_hub, api_name=False).then(
        report_results, inputs=[inp, library, access_token]
    )


demo.launch()
