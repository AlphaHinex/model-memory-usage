import re
import webbrowser
import pandas as pd
import gradio as gr
from huggingface_hub import HfApi
from accelerate.commands.estimate import create_empty_model
from accelerate.utils import convert_bytes, calculate_maximum_sizes

# We need to store them as globals because gradio doesn't have a way for us to pass them in to the button
HAS_DISCUSSION = True
MODEL_NAME = None
LIBRARY = None
TRUST_REMOTE_CODE = False

# We use this class to check if a discussion has been opened on the model by `huggingface_model_memory_bot`
hf_api = HfApi()

def check_for_discussion(model_name:str):
    "Checks if a discussion has been opened on the model"
    global hf_api
    discussions = list(hf_api.get_repo_discussions(model_name))
    return any(discussion.title == "[AUTOMATED] Model Memory Requirements" for discussion in discussions)

def report_results():
    "Reports the results of a memory calculation to the model's discussion"
    global MODEL_NAME, LIBRARY, TRUST_REMOTE_CODE
    _, results = calculate_memory(MODEL_NAME, LIBRARY, ["float32", "float16", "int8", "int4"], TRUST_REMOTE_CODE, raw=True)
    post = f"""# Model Memory Requirements\n
    
These calculations were measured from the [Model Memory Utility Space](https://hf.co/spaces/muellerzr/model-memory-utility) on the Hub.
    
The minimum recommended vRAM needed for this model to perform inference via [Accelerate or `device_map="auto"`](https://huggingface.co/docs/accelerate/usage_guides/big_modeling) is denoted by the size of the "largest layer" and training of the model is roughly 4x its total size (for Adam).

## Results

"""
    global hf_api
    post += results.to_markdown(index=False)
    # Uncomment when ready to go live
    discussion = hf_api.create_discussion(MODEL_NAME, "[AUTOMATED] Model Memory Requirements", description=post)
    webbrowser.open_new_tab(discussion.url)

def convert_url_to_name(url:str):
    "Converts a model URL to its name on the Hub"
    results = re.findall(r"huggingface.co\/(.*?)#", url)
    if len(results) < 1:
        raise ValueError(f"URL {url} is not a valid model URL to the Hugging Face Hub")
    return results[0]

def calculate_memory(model_name:str, library:str, options:list, trust_remote_code:bool, raw=False):
    "Calculates the memory usage for a model"
    if library == "auto":
        library = None
    if "huggingface.co" in model_name:
        model_name = convert_url_to_name(model_name)
    model = create_empty_model(model_name, library_name=library, trust_remote_code=trust_remote_code)
    total_size, largest_layer = calculate_maximum_sizes(model)

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
    global HAS_DISCUSSION, MODEL_NAME, LIBRARY, TRUST_REMOTE_CODE
    HAS_DISCUSSION = check_for_discussion(model_name)
    MODEL_NAME = model_name
    LIBRARY = library
    TRUST_REMOTE_CODE = trust_remote_code
    results = [f'## {title}', pd.DataFrame(data)]
    if not raw:
        results += [gr.update(visible=not HAS_DISCUSSION)]
    return results

with gr.Blocks() as demo:
    gr.Markdown(
        """# Model Memory Calculator

        This tool will help you calculate how much vRAM is needed to train and perform big model inference
        on a model hosted on the 🤗 Hugging Face Hub. The minimum recommended vRAM needed for a model
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
        interactive=False,
    )

    inp = gr.Textbox(label="Model Name or URL")
    with gr.Row():
        library = gr.Radio(["auto", "transformers", "timm"], label="Library", value="auto")
        options = gr.CheckboxGroup(
            ["float32", "float16", "int8", "int4"],
            value="float32"
        )
        trust_remote_code = gr.Checkbox(label="Trust Remote Code", value=False)
    btn = gr.Button("Calculate Memory Usage")
    post_to_hub = gr.Button(value = "Report results in this model repo's discussions!", visible=False)

    btn.click(
        calculate_memory, inputs=[inp, library, options, trust_remote_code], outputs=[out_text, out, post_to_hub],
    )
    
    post_to_hub.click(report_results)


demo.launch()