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
    return pd.DataFrame(data)
    # return f"## {title}\n\n" + markdown_table(data).set_params(
    #         row_sep="markdown", quote=False,
    #     ).get_markdown()


options = gr.CheckboxGroup(
    ["float32", "float16", "int8", "int4"],
)

library = gr.Radio(["auto", "transformers", "timm"], label="Library", value="auto")

iface = gr.Interface(
    fn=calculate_memory,
    inputs=[
        "text",
        library,
        options,
    ],
    outputs="dataframe"
)

iface.launch()