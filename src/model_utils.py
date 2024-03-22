# Utilities related to loading in and working with models/specific models
from urllib.parse import urlparse

import gradio as gr
import torch
from accelerate.commands.estimate import check_has_model, create_empty_model, estimate_training_usage
from accelerate.utils import calculate_maximum_sizes, convert_bytes
from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError


DTYPE_MODIFIER = {"float32": 1, "float16/bfloat16": 2, "int8": 4, "int4": 8}


def extract_from_url(name: str):
    "Checks if `name` is a URL, and if so converts it to a model name"
    is_url = False
    try:
        result = urlparse(name)
        is_url = all([result.scheme, result.netloc])
    except Exception:
        is_url = False
    # Pass through if not a URL
    if not is_url:
        return name
    else:
        path = result.path
        return path[1:]


def translate_llama2(text):
    "Translates llama-2 to its hf counterpart"
    if not text.endswith("-hf"):
        return text + "-hf"
    return text


def get_model(model_name: str, library: str, access_token: str):
    "Finds and grabs model from the Hub, and initializes on `meta`"
    if "meta-llama" in model_name:
        model_name = translate_llama2(model_name)
    if library == "auto":
        library = None
    model_name = extract_from_url(model_name)
    try:
        model = create_empty_model(model_name, library_name=library, trust_remote_code=True, access_token=access_token)
    except GatedRepoError:
        raise gr.Error(
            f"Model `{model_name}` is a gated model, please ensure to pass in your access token and try again if you have access. You can find your access token here : https://huggingface.co/settings/tokens. "
        )
    except RepositoryNotFoundError:
        raise gr.Error(f"Model `{model_name}` was not found on the Hub, please try another model name.")
    except ValueError:
        raise gr.Error(
            f"Model `{model_name}` does not have any library metadata on the Hub, please manually select a library_name to use (such as `transformers`)"
        )
    except (RuntimeError, OSError) as e:
        library = check_has_model(e)
        if library != "unknown":
            raise gr.Error(
                f"Tried to load `{model_name}` with `{library}` but a possible model to load was not found inside the repo."
            )
        raise gr.Error(
            f"Model `{model_name}` had an error, please open a discussion on the model's page with the error message and name: `{e}`"
        )
    except ImportError:
        # hacky way to check if it works with `trust_remote_code=False`
        model = create_empty_model(
            model_name, library_name=library, trust_remote_code=False, access_token=access_token
        )
    except Exception as e:
        raise gr.Error(
            f"Model `{model_name}` had an error, please open a discussion on the model's page with the error message and name: `{e}`"
        )
    return model


def calculate_memory(model: torch.nn.Module, options: list):
    "Calculates the memory usage for a model init on `meta` device"
    total_size, largest_layer = calculate_maximum_sizes(model)

    data = []
    for dtype in options:
        dtype_total_size = total_size
        dtype_largest_layer = largest_layer[0]

        modifier = DTYPE_MODIFIER[dtype]
        dtype_training_size = estimate_training_usage(
            dtype_total_size, dtype if dtype != "float16/bfloat16" else "float16"
        )
        dtype_total_size /= modifier
        dtype_largest_layer /= modifier

        dtype_total_size = convert_bytes(dtype_total_size)
        dtype_largest_layer = convert_bytes(dtype_largest_layer)
        data.append(
            {
                "dtype": dtype,
                "Largest Layer or Residual Group": dtype_largest_layer,
                "Total Size": dtype_total_size,
                "Training using Adam": dtype_training_size,
            }
        )
    return data
