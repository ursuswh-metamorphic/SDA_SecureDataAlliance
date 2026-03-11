from llama_index.llms.openai import OpenAI
from llama_index.llms.openai_like import OpenAILike

from llms.huggingface_model import get_huggingfacellm
from llms import chatglm4
from embs import chatglmemb
from config import Config
from llama_index.llms.ollama import Ollama
import os
import logging

logger = logging.getLogger(__name__)


llm_dict = {
    "qwen": "Qwen/Qwen1.5-7B-Chat",
    "llama": "meta-llama/Llama-2-7b-chat-hf",
    "chatglm": "THUDM/chatglm3-6b",
    "qwen14_int8": "Qwen/Qwen1.5-14B-Chat-GPTQ-Int8",
    "qwen7_int8": "Qwen/Qwen1.5-7B-Chat-GPTQ-Int8",
    "qwen1.8": "Qwen/Qwen1.5-1.8B-Chat",
    "baichuan": "baichuan-inc/Baichuan2-7B-Chat",
    "falcon": "tiiuae/falcon-7b-instruct",
    "mpt": "mosaicml/mpt-7b-chat",
    "yi": "01-ai/Yi-6B-Chat",
}


def get_openai(api_base, api_key, api_name):
    return OpenAI(api_key=api_key, api_base=api_base, temperature=0, model=api_name)


def get_nvidia_llm():
    """NVIDIA NIM API - OpenAI-compatible, open source models.
    
    Uses OpenAILike instead of OpenAI to avoid model name validation
    (NVIDIA models like meta/llama-3.1-70b-instruct are not in OpenAI's list).
    """
    cfg = Config()
    api_key = getattr(cfg, "nvidia_api_key", "") or os.environ.get("NVIDIA_API_KEY", "")
    api_base = getattr(cfg, "nvidia_api_base", "https://integrate.api.nvidia.com/v1")
    model = getattr(cfg, "nvidia_model", "meta/llama-3.1-70b-instruct")
    
    if not api_key:
        # Fallback to OpenAI if NVIDIA key not set (so app can start, user can add key via UI)
        openai_key = getattr(cfg, "api_key", "") or ""
        if openai_key:
            logger.warning(
                "NVIDIA API key not set. Falling back to OpenAI. "
                "Add nvidia_api_key in config or via UI to use NVIDIA models."
            )
            return get_openai(
                getattr(cfg, "api_base", "https://api.openai.com/v1"),
                openai_key,
                getattr(cfg, "api_name", "gpt-4o-mini"),
            )
        raise ValueError(
            "NVIDIA API key required. Set nvidia_api_key in config.toml or add via UI (Sidebar > LLM Settings). "
            "Get key at https://build.nvidia.com"
        )
    
    # Use OpenAILike for NVIDIA NIM (avoids model name validation)
    return OpenAILike(
        api_key=api_key,
        api_base=api_base,
        model=model,
        temperature=0,
        is_chat_model=True,
        context_window=128000,  # Llama 3.1 supports 128K context
    )


ollama_url = "http://localhost:11434"
os.environ['OLLAMA_HOST'] = ollama_url
def get_llm(name):
    if name in llm_dict.keys():
        return get_huggingfacellm(llm_dict[name])
    elif name == 'chatgpt-3.5':
        return get_openai(Config().api_base,Config().api_key,Config().api_name)
    elif name == 'gemini-2.5-pro':
        return get_gemini()
    elif name == 'gpt-4o-mini':
        return get_openai(Config().api_base, Config().api_key, Config().api_name)
    elif name == 'nvidia':
        return get_nvidia_llm()
    elif name == 'openai':
        return get_openai(Config().api_base, Config().api_key, Config().api_name)
    elif name == 'Llama3.1:8B':
        return Ollama(model="Llama3.1:8B", request_timeout=3600, base_url=ollama_url)
    elif name == 'deepseek-r1:7b':
        return Ollama(model="deepseek-r1:7b", request_timeout=3600, base_url=ollama_url)
    elif name == 'openthinker:latest':
        return Ollama(model="openthinker:latest", request_timeout=3600, base_url=ollama_url)
    elif name == 'mathstral:latest':
        return Ollama(model="mathstral:latest", request_timeout=3600, base_url=ollama_url)
    else:
        raise ValueError(f"no model name: {name}.")


