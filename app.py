import os
import re
import webbrowser
import pandas as pd
import gradio as gr
from huggingface_hub import HfApi
from huggingface_hub.utils import RepositoryNotFoundError, GatedRepoError
from accelerate.commands.estimate import create_empty_model, check_has_model
from accelerate.utils import convert_bytes, calculate_maximum_sizes
from urllib.parse import urlparse

# We need to store them as globals because gradio doesn't have a way for us to pass them in to the button
HAS_DISCUSSION = True
MODEL_NAME = None
LIBRARY = None
USER_TOKEN = None
TOKEN = os.environ.get("HUGGINGFACE_API_LOGIN", None)

def translate_llama2(text):
    "Translates llama-2 to its hf counterpart"
    if not text.endswith("-hf"):
        return text + "-hf"
    return text

def check_for_discussion(model_name:str):
    "Checks if an automated discussion has been opened on the model by `model-sizer-bot`"
    global TOKEN
    api = HfApi(token=TOKEN)
    discussions = list(api.get_repo_discussions(model_name))
    return any(discussion.title == "[AUTOMATED] Model Memory Requirements" and discussion.author == "model-sizer-bot" for discussion in discussions)

def report_results():
    "Reports the results of a memory calculation to the model's discussion page, and opens a new tab to it afterwards"
    global MODEL_NAME, LIBRARY, TOKEN, USER_TOKEN
    api = HfApi(token=TOKEN)
    results, data = calculate_memory(MODEL_NAME, LIBRARY, ["fp32", "fp16", "int8", "int4"], access_token=USER_TOKEN, raw=True)
    minimum = data[0]

    USER_TOKEN = None
    post = f"""# Model Memory Requirements\n

You will need about {minimum[1]} VRAM to load this model for inference, and {minimum[3]} VRAM to train it using Adam.
    
These calculations were measured from the [Model Memory Utility Space](https://hf.co/spaces/hf-accelerate/model-memory-utility) on the Hub.
    
The minimum recommended vRAM needed for this model assumes using [Accelerate or `device_map="auto"`](https://huggingface.co/docs/accelerate/usage_guides/big_modeling) and is denoted by the size of the "largest layer". 
When performing inference, expect to add up to an additional 20% to this, as found by [EleutherAI](https://blog.eleuther.ai/transformer-math/). More tests will be performed in the future to get a more accurate benchmark for each model.

When training with `Adam`, you can expect roughly 4x the reported results to be used. (1x for the model, 1x for the gradients, and 2x for the optimizer).

## Results:

{results}
"""
    discussion = api.create_discussion(MODEL_NAME, "[AUTOMATED] Model Memory Requirements", description=post)
    webbrowser.open_new_tab(discussion.url)

def extract_from_url(name:str):
    "Checks if `name` is a URL, and if so converts it to a model name"
    is_url = False
    try:
        result = urlparse(name)
        is_url = all([result.scheme, result.netloc])
    except:
        is_url = False
    # Pass through if not a URL
    if not is_url:
        return name
    else:
        path = result.path
        return path[1:]

def calculate_memory(model_name:str, library:str, options:list, access_token:str, raw=False):
    "Calculates the memory usage for a model"
    if "meta-llama" in model_name:
        model_name = translate_llama2(model_name)
    if library == "auto":
        library = None
    model_name = extract_from_url(model_name)
    try:
        model = create_empty_model(model_name, library_name=library, trust_remote_code=True, access_token=access_token)
    except GatedRepoError:
        raise gr.Error(f"Model `{model_name}` is a gated model, please ensure to pass in your access token and try again if you have access. You can find your access token here : https://huggingface.co/settings/tokens. ")
    except RepositoryNotFoundError:
        raise gr.Error(f"Model `{model_name}` was not found on the Hub, please try another model name.")
    except ValueError as e:
        raise gr.Error(f"Model `{model_name}` does not have any library metadata on the Hub, please manually select a library_name to use (such as `transformers`)")
    except (RuntimeError, OSError) as e:
        library = check_has_model(e)
        if library != "unknown":
            raise gr.Error(f"Tried to load `{model_name}` with `{library}` but a possible model to load was not found inside the repo.")
        raise gr.Error(f"Model `{model_name}` had an error, please open a discussion on the model's page with the error message and name: `{e}`")
    except ImportError:
        # hacky way to check if it works with `trust_remote_code=False`
        model = create_empty_model(model_name, library_name=library, trust_remote_code=False, access_token=access_token)
    except Exception as e:
        raise gr.Error(f"Model `{model_name}` had an error, please open a discussion on the model's page with the error message and name: `{e}`")
    total_size, largest_layer = calculate_maximum_sizes(model)

    data = []

    title = f"Memory Usage for '{model_name}'"
    for dtype in options:
        dtype_total_size = total_size
        dtype_largest_layer = largest_layer[0]
        if dtype in ("fp16",  "bf16", "float16/bfloat16"):
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
            "Largest Layer or Residual Group": dtype_largest_layer,
            "Total Size": dtype_total_size,
            "Training using Adam": dtype_training_size
        })
    global HAS_DISCUSSION, MODEL_NAME, LIBRARY
    HAS_DISCUSSION = check_for_discussion(model_name)
    MODEL_NAME = model_name
    LIBRARY = library

    if raw:
        return pd.DataFrame(data).to_markdown(index=False), data
    
    results = [
        f'## {title}', 
        gr.update(visible=True, value=pd.DataFrame(data)), 
        gr.update(visible=not HAS_DISCUSSION)
    ]
    return results

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
            post_to_hub = gr.Button(value = "Report results in this model repo's discussions!\n(Will open in a new tab)", visible=False)
    USER_TOKEN = access_token

    btn.click(
        calculate_memory, inputs=[inp, library, options, access_token], outputs=[out_text, out, post_to_hub],
    )
    
    post_to_hub.click(report_results).then(lambda: gr.Button.update(visible=False), outputs=post_to_hub)


demo.launch()