# Utilities related to searching and posting on the Hub
import os
import webbrowser

import pandas as pd
from huggingface_hub import HfApi
from model_utils import calculate_memory, extract_from_url, get_model


def check_for_discussion(model_name: str):
    "Checks if an automated discussion has been opened on the model by `model-sizer-bot`"
    api = HfApi(token=os.environ.get("HUGGINGFACE_API_LOGIN", None))
    model_name = extract_from_url(model_name)
    discussions = list(api.get_repo_discussions(model_name))
    return any(
        discussion.title == "[AUTOMATED] Model Memory Requirements" and discussion.author == "model-sizer-bot"
        for discussion in discussions
    )


def report_results(model_name, library, access_token):
    "Reports the results of a memory calculation to the model's discussion page, and opens a new tab to it afterwards"
    model = get_model(model_name, library, access_token)
    data = calculate_memory(model, ["float32", "float16/bfloat16", "int8", "int4"])
    df = pd.DataFrame(data).to_markdown(index=False)

    post = f"""# Model Memory Requirements\n

You will need about {data[1]} VRAM to load this model for inference, and {data[3]} VRAM to train it using Adam.

These calculations were measured from the [Model Memory Utility Space](https://huggingface.co/spaces/hf-accelerate/model-memory-usage) on the Hub.

The minimum recommended vRAM needed for this model assumes using [Accelerate or `device_map="auto"`](https://huggingface.co/docs/accelerate/usage_guides/big_modeling) and is denoted by the size of the "largest layer". 
When performing inference, expect to add up to an additional 20% to this, as found by [EleutherAI](https://blog.eleuther.ai/transformer-math/). More tests will be performed in the future to get a more accurate benchmark for each model.

When training with `Adam`, you can expect roughly 4x the reported results to be used. (1x for the model, 1x for the gradients, and 2x for the optimizer).

## Results:

{df}
"""
    api = HfApi(token=os.environ.get("HUGGINGFACE_API_LOGIN", None))
    discussion = api.create_discussion(model_name, "[AUTOMATED] Model Memory Requirements", description=post)
    webbrowser.open_new_tab(discussion.url)
