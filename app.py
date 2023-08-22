import re
import pandas as pd
import gradio as gr
from py_markdown_table.markdown_table import markdown_table
from model_sizer.utils import get_sizes, create_empty_model, convert_bytes


def convert_url_to_name(url:str):
    "Converts a model URL to its name on the Hub"
    results = re.findall(r"huggingface.co\/(.*?)#", url)
    if len(results) < 1:
        raise ValueError(f"URL {url} is not a valid model URL to the Hugging Face Hub")
    return results[0]

def calculate_memory(model_name:str, library:str, options:list):
    "Calculates the memory usage for a model"
    if library == "auto":
        library = None
    if "huggingface.co" in model_name:
        model_name = convert_url_to_name(model_name)
    model = create_empty_model(model_name, library_name=library)
    total_size, largest_layer = get_sizes(model)

    data = []

    title = f"Memory Usage for `{model_name}`"
    for dtype in options:
        dtype_total_size = total_size
        dtype_largest_layer = largest_layer[0]
        if dtype == "float16":
            dtype_total_size /= 2
            dtype_largest_layer /= 2
        elif dtype == "int8":
            dtype_total_size /= 4
            dtype_largest_layer /= 4
        elif dtype == "int4":
            dtype_total_size /= 8
            dtype_largest_layer /= 8
        dtype_training_size = convert_bytes(dtype_total_size * 4)
        dtype_total_size = convert_bytes(dtype_total_size)
        dtype_largest_layer = convert_bytes(dtype_largest_layer)
        data.append({
            "dtype": dtype,
            "Largest Layer": dtype_largest_layer,
            "Total Size": dtype_total_size,
            "Training using Adam": dtype_training_size
        })
    return f'## {title}', pd.DataFrame(data)

with gr.Blocks() as demo:
    gr.Markdown(
        """# Model Memory Calculator

        This tool will help you calculate how much vRAM is needed to train and perform big model inference
        on a model hosted on the :hugging_face: Hugging Face Hub. The minimum recommended vRAM needed for a model
        is denoted as the size of the "largest layer", and training of a model is roughly 4x its size (for Adam).
        
        Currently this tool supports all models hosted that use `transformers` and `timm`.

        To use this tool pass in the URL or model name of the model you want to calculate the memory usage for,
        select which framework it originates from ("auto" will try and detect it from the model metadata), and
        what precisions you want to use.  
        """
    )
    out_text = gr.Markdown()
    out = gr.DataFrame(
        headers=["dtype", "Largest Layer", "Total Size", "Training using Adam"],
    )

    inp = gr.Textbox(label="Model Name or URL")
    with gr.Row():
        library = gr.Radio(["auto", "transformers", "timm"], label="Library", value="auto")
        options = gr.CheckboxGroup(
            ["float32", "float16", "int8", "int4"],
            value="float32"
        )
    btn = gr.Button("Calculate Memory Usage", scale=0.5)

    btn.click(
        calculate_memory, inputs=[inp, library, options], outputs=[out_text, out],
    )

demo.launch()