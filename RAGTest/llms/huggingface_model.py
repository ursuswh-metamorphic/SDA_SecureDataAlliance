from functools import partial

import torch
import types
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from transformers.generation.utils import GenerationConfig
from transformers import BitsAndBytesConfig
from llama_index.llms.huggingface import HuggingFaceLLM
# pip install llama-index-llms-huggingface
from config import Config
cfg = Config()

load_tokenizer = []


def llama_model_and_tokenizer(name, auth_token):
    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(name, token=auth_token)

    # Create quantization config using BitsAndBytesConfig (new recommended way)
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
    )

    # Create model
    # For quantized models, we need to prevent accelerate from calling .to()
    # Solution: Patch the model's .to() method to handle quantized models gracefully
    # and patch dispatch_model to catch the error
    
    from transformers import modeling_utils
    from accelerate import big_modeling
    
    # Store original methods
    original_to = modeling_utils.PreTrainedModel.to
    original_dispatch = big_modeling.dispatch_model
    
    def safe_to(self, *args, **kwargs):
        # Check if model is quantized
        is_quantized = False
        try:
            for param in self.parameters():
                if hasattr(param, 'quant_state'):
                    is_quantized = True
                    break
            if not is_quantized and hasattr(self, 'config'):
                if hasattr(self.config, 'quantization_config') and self.config.quantization_config is not None:
                    is_quantized = True
        except:
            pass
        
        # If quantized, return self without moving (already on correct device)
        if is_quantized:
            return self
        # Otherwise, use original .to() method
        return original_to(self, *args, **kwargs)
    
    def safe_dispatch(model, device_map=None, **kwargs):
        try:
            return original_dispatch(model, device_map=device_map, **kwargs)
        except ValueError as e:
            # If error is about quantized model, return model as-is
            if ".to" in str(e) and ("4-bit" in str(e) or "8-bit" in str(e)):
                return model
            raise
    
    # Apply patches
    modeling_utils.PreTrainedModel.to = safe_to
    big_modeling.dispatch_model = safe_dispatch
    
    try:
        # Now load model - both .to() and dispatch_model are patched
        # When CUDA is available we still pass device_map="auto" so accelerate can shard the model
        # The patched dispatch will absorb the known ValueError for 4/8 bit models
        load_kwargs = {
            "token": auth_token,
            "torch_dtype": torch.float16,
            "rope_scaling": {"type": "dynamic", "factor": 2},
            "quantization_config": quantization_config,
            "low_cpu_mem_usage": True,
        }
        if torch.cuda.is_available():
            load_kwargs["device_map"] = "auto"
        # Only add device_map if CUDA is available and patches are in place
        # Actually, let's not use device_map at all for quantized models
        model = AutoModelForCausalLM.from_pretrained(name, **load_kwargs)
    finally:
        # Restore original methods
        modeling_utils.PreTrainedModel.to = original_to
        big_modeling.dispatch_model = original_dispatch
    
    # For quantized models, bitsandbytes handles device placement automatically
    # We cannot and should not call .to() on quantized models
    # Just set to eval mode
    model = model.eval()
    _patch_rotary_embeddings(model)

    return tokenizer, model


def llama_completion_to_prompt(completion):
    return f"""<s>[INST] <<SYS>>
        You are a helpful, respectful and honest assistant. Always answer as 
        helpfully as possible, while being safe. Your answers should not include
        any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content.
        Please ensure that your responses are socially unbiased and positive in nature.

        If a question does not make any sense, or is not factually coherent, explain 
        why instead of answering something not correct. If you don't know the answer 
        to a question, please don't share false information.

        Your goal is to provide answers relating to the financial performance of 
        the company.<</SYS>>
        {completion} [/INST]"""


def chatglm_model_and_tokenizer(name):
    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    load_tokenizer.append(tokenizer)

    # Create model
    model = AutoModel.from_pretrained(name, trust_remote_code=True).half().cuda().eval()

    return tokenizer, model


def chatglm_completion_to_prompt(completion):
    return "<|user|>\n " + completion + "<|assistant|>"


def qwen_model_and_tokenizer(name):
    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(name)
    load_tokenizer.append(tokenizer)

    # Create model
    model = AutoModelForCausalLM.from_pretrained(name,
                                                 torch_dtype=torch.float16,
                                                 device_map="auto").eval()

    return tokenizer, model


def qwen_completion_to_prompt(completion):
    tokenizer = load_tokenizer[0]
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": completion}
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )


def baichuan_model_and_tokenizer(name):
    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(name, use_fast=False, trust_remote_code=True)
    load_tokenizer.append(tokenizer)

    # Create model
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16, trust_remote_code=True, device_map="auto").eval()
    model.generation_config = GenerationConfig.from_pretrained("baichuan-inc/Baichuan2-7B-Chat")

    return tokenizer, model


def baichuan_completion_to_prompt(completion):
    return "<reserved_106>" + completion + "<reserved_107>"  # "You are a helpful assistant.<reserved_106>" + completion + "<reserved_107>""


def falcon_model_and_tokenizer(name):
    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(name)
    load_tokenizer.append(tokenizer)

    # Create model
    model = AutoModelForCausalLM.from_pretrained(name, trust_remote_code=True, device_map="auto").eval()

    return tokenizer, model


def falcon_completion_to_prompt(completion):
    return completion


def mpt_model_and_tokenizer(name):
    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    load_tokenizer.append(tokenizer)

    # Create model
    model = AutoModelForCausalLM.from_pretrained(name, trust_remote_code=True, device_map="auto").eval()

    return tokenizer, model


def mpt_completion_to_prompt(completion):
    return completion


def yi_model_and_tokenizer(name):
    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(name, use_fast=False)
    load_tokenizer.append(tokenizer)

    # Create model
    model = AutoModelForCausalLM.from_pretrained(
        name,
        torch_dtype=torch.float16,
        device_map="auto"
    ).eval()

    return tokenizer, model


def yi_completion_to_prompt(completion):
    return "<|im_start|> user\n" + completion + "<|im_end|> \n<|im_start|>assistant\n"


tokenizer_and_model_fn_dict = {
    "meta-llama/Llama-2-7b-chat-hf": partial(llama_model_and_tokenizer, auth_token=cfg.auth_token),
    "THUDM/chatglm3-6b": chatglm_model_and_tokenizer,
    "Qwen/Qwen1.5-7B-Chat": qwen_model_and_tokenizer,
    "Qwen/Qwen1.5-7B-Chat-GPTQ-Int8": qwen_model_and_tokenizer,
    "baichuan-inc/Baichuan2-7B-Chat": baichuan_model_and_tokenizer,
    "tiiuae/falcon-7b-instruct": falcon_model_and_tokenizer,
    "mosaicml/mpt-7b-chat": mpt_model_and_tokenizer,
    "01-ai/Yi-6B-Chat": yi_model_and_tokenizer,
}

completion_to_prompt_dict = {
    "meta-llama/Llama-2-7b-chat-hf": llama_completion_to_prompt,
    "THUDM/chatglm3-6b": chatglm_completion_to_prompt,
    "Qwen/Qwen1.5-7B-Chat": qwen_completion_to_prompt,
    "Qwen/Qwen1.5-7B-Chat-GPTQ-Int8": qwen_completion_to_prompt,
    "baichuan-inc/Baichuan2-7B-Chat": baichuan_completion_to_prompt,
    "tiiuae/falcon-7b-instruct": falcon_completion_to_prompt,
    "mosaicml/mpt-7b-chat": mpt_completion_to_prompt,
    "01-ai/Yi-6B-Chat": yi_completion_to_prompt,
}

# llm_argument_dict = {
#     "meta-llama/Llama-2-7b-chat-hf": {"context_window": 4096, "max_new_tokens": 256,
#                                       "generate_kwargs": {"temperature": 0.7, "top_k": 50, "top_p": 0.95}},
#     "THUDM/chatglm3-6b": {"context_window": 4096, "max_new_tokens": 256,
#                           "generate_kwargs": {"do_sample": True, "temperature": 0.7, "top_k": 50, "top_p": 0.95,
#                                               "eos_token_id": [2, 64795, 64797]}},
#     "Qwen/Qwen1.5-7B-Chat": {"context_window": 4096, "max_new_tokens": 256,
#                              "generate_kwargs": {"temperature": 0.7, "top_k": 50, "top_p": 0.95}},
#     "Qwen/Qwen1.5-7B-Chat-GPTQ-Int8": {"context_window": 4096, "max_new_tokens": 256,
#                              "generate_kwargs": {"temperature": 0.7, "top_k": 50, "top_p": 0.95}},
#     "baichuan-inc/Baichuan2-7B-Chat": {"context_window": 4096, "max_new_tokens": 256, "generate_kwargs": None},
#     "tiiuae/falcon-7b-instruct": {"context_window": 4096, "max_new_tokens": 256,
#                                   "generate_kwargs": {"do_sample": True, "temperature": 0.7, "top_k": 50,
#                                                       "top_p": 0.95}},
#     "mosaicml/mpt-7b-chat": {"context_window": 4096, "max_new_tokens": 256,
#                              "generate_kwargs": {"do_sample": True, "temperature": 0.7, "top_k": 50, "top_p": 0.95}},
#     "01-ai/Yi-6B-Chat": {"context_window": 4096, "max_new_tokens": 256,
#                          "generate_kwargs": {"temperature": 0.7, "top_k": 50, "top_p": 0.95}},
# }

llm_argument_dict = {
    "meta-llama/Llama-2-7b-chat-hf": {"context_window": 4096, "max_new_tokens": 256, "generate_kwargs": {"temperature": 0.7}},
    "THUDM/chatglm3-6b": {"context_window": 4096, "max_new_tokens": 256, "generate_kwargs": {"temperature": 0.7, "eos_token_id": [2, 64795, 64797]}},
    "Qwen/Qwen1.5-7B-Chat": {"context_window": 4096, "max_new_tokens": 256, "generate_kwargs": {"temperature": 0.7}},
    "Qwen/Qwen1.5-7B-Chat-GPTQ-Int8": {"context_window": 4096, "max_new_tokens": 256, "generate_kwargs": {"temperature": 0.7}},
    "baichuan-inc/Baichuan2-7B-Chat": {"context_window": 4096, "max_new_tokens": 256, "generate_kwargs": {"temperature": 0.7}},
    "tiiuae/falcon-7b-instruct": {"context_window": 4096, "max_new_tokens": 256, "generate_kwargs": {"temperature": 0.7}},
    "mosaicml/mpt-7b-chat": {"context_window": 4096, "max_new_tokens": 256, "generate_kwargs": {"temperature": 0.7}},
    "01-ai/Yi-6B-Chat": {"context_window": 4096, "max_new_tokens": 256, "generate_kwargs": {"temperature": 0.7}},
}

def get_huggingfacellm(name):
    print("name is " + name)
    tokenizer, model = tokenizer_and_model_fn_dict[name](name)

    # Check if model is quantized (8-bit or 4-bit)
    # Quantized models cannot be moved with .to(), so we don't pass device_map
    # Check for quantization by looking at model config or parameter attributes
    is_quantized = False
    try:
        # Check if model has quantization config
        if hasattr(model, 'config') and hasattr(model.config, 'quantization_config'):
            if model.config.quantization_config is not None:
                is_quantized = True
        # Check if any parameters are quantized (bitsandbytes adds quant_state)
        if not is_quantized:
            for param in model.parameters():
                if hasattr(param, 'quant_state'):
                    is_quantized = True
                    break
    except Exception:
        # If check fails, assume not quantized
        pass
    
    # For models loaded with load_in_8bit or load_in_4bit, device_map should not be passed to HuggingFaceLLM
    # because the model is already on the correct device and cannot be moved
    llm_kwargs = {
        "context_window": llm_argument_dict[name]["context_window"],
        "max_new_tokens": llm_argument_dict[name]["max_new_tokens"],
        "completion_to_prompt": completion_to_prompt_dict[name],
        "generate_kwargs": llm_argument_dict[name]["generate_kwargs"],
        "model": model,
        "tokenizer": tokenizer,
    }
    
    # Only add device_map if model is not quantized
    if not is_quantized:
        llm_kwargs["device_map"] = "auto"
    
    # Create a HF LLM using the llama index wrapper
    llm = HuggingFaceLLM(**llm_kwargs)
    return llm


def _patch_rotary_embeddings(model):
    try:
        from transformers.models.llama.modeling_llama import (
            LlamaRotaryEmbedding,
            LlamaLinearScalingRotaryEmbedding,
            LlamaDynamicNTKScalingRotaryEmbedding,
        )
    except ImportError:
        return

    rotary_classes = (
        LlamaRotaryEmbedding,
        LlamaLinearScalingRotaryEmbedding,
        LlamaDynamicNTKScalingRotaryEmbedding,
    )

    for module in model.modules():
        if isinstance(module, rotary_classes) and not getattr(module, "_device_patch_applied", False):
            original_forward = module.forward

            def patched_forward(self, x, *args, _orig_forward=original_forward, **kwargs):
                target_device = x.device

                # Ensure rotary buffers/bases are on the same device as the input
                if hasattr(self, "inv_freq") and torch.is_tensor(self.inv_freq) and self.inv_freq.device != target_device:
                    self.inv_freq = self.inv_freq.to(target_device)
                if hasattr(self, "base") and torch.is_tensor(self.base) and self.base.device != target_device:
                    self.base = self.base.to(target_device)
                if hasattr(self, "cos_cached") and torch.is_tensor(self.cos_cached) and self.cos_cached.device != target_device:
                    self.cos_cached = self.cos_cached.to(target_device)
                if hasattr(self, "sin_cached") and torch.is_tensor(self.sin_cached) and self.sin_cached.device != target_device:
                    self.sin_cached = self.sin_cached.to(target_device)

                # move positional inputs if needed
                if args:
                    args = list(args)
                    first_arg = args[0]
                    if torch.is_tensor(first_arg) and first_arg.device != target_device:
                        args[0] = first_arg.to(target_device)
                    args = tuple(args)

                return _orig_forward(x, *args, **kwargs)

            module.forward = types.MethodType(patched_forward, module)
            module._device_patch_applied = True
