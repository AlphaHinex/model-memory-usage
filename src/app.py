import gradio as gr
import pandas as pd
from accelerate.utils import convert_bytes
from hub_utils import check_for_discussion, report_results
from huggingface_hub.utils import HfHubHTTPError
from model_utils import calculate_memory, get_model


def get_results(model_name: str, library: str, options: list, access_token: str):
    model = get_model(model_name, library, access_token)
    try:
        has_discussion = check_for_discussion(model_name)
    except HfHubHTTPError:
        has_discussion = True
    title = f"## Memory usage for '{model_name}'"
    data = calculate_memory(model, options)
    stages = {"model": [], "gradients": [], "optimizer": [], "step": []}
    for i, option in enumerate(data):
        for stage in stages:
            stages[stage].append(option["Training using Adam (Peak vRAM)"][stage])
        value = max(data[i]["Training using Adam (Peak vRAM)"].values())
        if value == -1:
            value = "N/A"
        else:
            value = convert_bytes(value)
        data[i]["Training using Adam (Peak vRAM)"] = value

    if any(value != -1 for value in stages["model"]):
        out_explain = "## Training using Adam explained:\n"
        out_explain += "When training on a batch size of 1, each stage of the training process is expected to have near the following memory results for each precision you selected:\n"
        memory_values = pd.DataFrame(
            columns=["dtype", "Model", "Gradient calculation", "Backward pass", "Optimizer step"]
        )
        for i, dtype in enumerate(options):
            if stages["model"][i] != -1:
                memory_values.loc[len(memory_values)] = [
                    dtype,
                    convert_bytes(stages["model"][i]),
                    convert_bytes(stages["gradients"][i]),
                    convert_bytes(stages["optimizer"][i]),
                    convert_bytes(stages["step"][i]),
                ]
        return [
            title,
            gr.update(visible=True, value=pd.DataFrame(data)),
            gr.update(visible=True, value=out_explain),
            gr.update(visible=True, value=memory_values),
            gr.update(visible=not has_discussion),
        ]
    return [
        title,
        gr.update(visible=True, value=pd.DataFrame(data)),
        gr.update(visible=False, value=""),
        gr.update(visible=False, value=pd.DataFrame()),
        gr.update(visible=not has_discussion),
    ]


with gr.Blocks() as demo:
    with gr.Column():
        gr.Markdown(
            """<img src="https://huggingface.co/spaces/hf-accelerate/model-memory-usage/resolve/main/measure_model_size.png" style="float: left;" width="250" height="250"><h1>🤗 Model Memory Calculator</h1>
    This tool will help you calculate how much vRAM is needed to train and perform big model inference
    on a model hosted on the 🤗 Hugging Face Hub. The minimum recommended vRAM needed for a model
    is denoted as the size of the "largest layer", and training of a model is roughly 4x its size (for Adam).
    These calculations are accurate within a few percent at most, such as `bert-base-cased` being 413.68 MB and the calculator estimating 413.18 MB.
    When performing inference, expect to add up to an additional 20% to this as found by [EleutherAI](https://blog.eleuther.ai/transformer-math/). 
    More tests will be performed in the future to get a more accurate benchmark for each model.
    Currently this tool supports all models hosted that use `transformers` and `timm`.
    To use this tool pass in the URL or model name of the model you want to calculate the memory usage for,
    select which framework it originates from ("auto" will try and detect it from the model metadata), and
    what precisions you want to use."""
        )
        out_text = gr.Markdown()
        out = gr.DataFrame(
            headers=["dtype", "Largest Layer", "Total Size", "Training using Adam (Peak vRAM)"],
            interactive=False,
            visible=False,
        )
        out_explain = gr.Markdown()
        memory_values = gr.DataFrame(
            headers=["dtype", "Model", "Gradient calculation", "Backward pass", "Optimizer step"],
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
        outputs=[out_text, out, out_explain, memory_values, post_to_hub],
        api_name=False,
    )

    post_to_hub.click(lambda: gr.Button.update(visible=False), outputs=post_to_hub, api_name=False).then(
        report_results, inputs=[inp, library, access_token]
    )


demo.launch()
