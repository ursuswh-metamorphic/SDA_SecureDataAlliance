# from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from typing import Any, List, Optional

from transformers import AutoModel, AutoTokenizer
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.bridge.pydantic import Field, PrivateAttr
import torch
from transformers import BertModel
import os

# Default downstream embedding backbone when path is missing / load fails.
# Must match upstream MedCPT Article Encoder training.
# https://huggingface.co/ncbi/MedCPT-Article-Encoder
DEFAULT_EMBEDDING_MODEL = "ncbi/MedCPT-Article-Encoder"
# MedCPT uses [CLS] pooling and 512 max tokens (see HF model card).
DEFAULT_EMBEDDING_POOLING = "cls"
DEFAULT_EMBEDDING_MAX_LENGTH = 512


class MedCPTEmbedding(BaseEmbedding):
    """LlamaIndex embedding compatible with MedCPT-style encoders.

    Uses transformers AutoModel/AutoTokenizer + [CLS] pooling as described
    in the MedCPT HF model card:
    https://huggingface.co/ncbi/MedCPT-Article-Encoder

    Also works for any BERT-family model (including BGE) if you explicitly
    want CLS pooling. Loading accepts either a HuggingFace model id or a
    local directory (e.g. a fine-tuned/converted checkpoint).
    """

    model_name: str = Field(description="HF model id or local path.")
    max_length: int = Field(
        default=DEFAULT_EMBEDDING_MAX_LENGTH,
        description="Max tokens per input.",
        gt=0,
    )
    normalize: bool = Field(
        default=True,
        description="L2-normalize output embeddings (cosine-ready).",
    )

    _tokenizer: Any = PrivateAttr()
    _model: Any = PrivateAttr()
    _device: str = PrivateAttr()

    def __init__(
        self,
        model_name: str,
        max_length: int = DEFAULT_EMBEDDING_MAX_LENGTH,
        normalize: bool = True,
        device: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model_name=model_name,
            max_length=max_length,
            normalize=normalize,
            **kwargs,
        )
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModel.from_pretrained(model_name).to(self._device)
        self._model.eval()

    @classmethod
    def class_name(cls) -> str:
        return "MedCPTEmbedding"

    def _encode(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        enc = self._tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        ).to(self._device)
        with torch.no_grad():
            out = self._model(**enc).last_hidden_state[:, 0, :]
        if self.normalize:
            out = torch.nn.functional.normalize(out, p=2, dim=-1)
        return out.cpu().tolist()

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._encode([query])[0]

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._encode([text])[0]

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self._encode(list(texts))

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)


def get_embedding(state_dict_path):    
    print("embedding_path: ", state_dict_path)

    # Handle relative paths - convert to absolute if it's a local path
    model_path = state_dict_path
    
    # Check if it's a relative path (not starting with / and not a HuggingFace model ID with /)
    if not os.path.isabs(state_dict_path) and not state_dict_path.startswith("http"):
        # Try to resolve relative path from project root (where config.toml is)
        # Get the directory containing this file (embs/)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Go up one level to get project root (finsaferag/finsaferag/)
        project_root = os.path.dirname(current_dir)
        # Try absolute path
        absolute_path = os.path.join(project_root, state_dict_path)
        if os.path.exists(absolute_path) or os.path.isdir(absolute_path):
            model_path = os.path.abspath(absolute_path)
            print(f"[OK] Resolved relative path to: {model_path}")
        elif os.path.exists(state_dict_path):
            # Path exists as-is (relative to current working directory)
            model_path = os.path.abspath(state_dict_path)
            print(f"[OK] Found model at relative path: {model_path}")
        else:
            # Assume it's a HuggingFace model ID
            print(f"[WARN] Path not found, treating as HuggingFace model ID: {state_dict_path}")
            model_path = state_dict_path
    elif os.path.isabs(state_dict_path):
        # Absolute path - check if exists
        if os.path.exists(state_dict_path) or os.path.isdir(state_dict_path):
            print(f"[OK] Found model at absolute path: {state_dict_path}")
        else:
            print(f"[WARN] Absolute path not found: {state_dict_path}")
            print(f"[WARN] Trying as HuggingFace model ID or falling back to default")
            # Try as-is first, will fallback in exception handler
            model_path = state_dict_path

    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device for embeddings: {device}")

    # Primary path: use our MedCPT-compatible wrapper (AutoModel + [CLS] pooling
    # + 512 max tokens). This matches the upstream MedCPT training exactly and
    # also works for any BERT-family checkpoint (including BGE/BGE-fine-tuned).
    try:
        embeddings = MedCPTEmbedding(
            model_name=model_path,
            max_length=DEFAULT_EMBEDDING_MAX_LENGTH,
            device=device,
        )
        print(f"[OK] Created MedCPTEmbedding (pooling=cls, max_length={DEFAULT_EMBEDDING_MAX_LENGTH}, device={device})")
        return embeddings
    except Exception as e_medcpt:
        print(f"[WARN] MedCPTEmbedding load failed: {e_medcpt}")
        print(f"[WARN] Falling back to default model via MedCPTEmbedding: {DEFAULT_EMBEDDING_MODEL}")
        try:
            embeddings = MedCPTEmbedding(
                model_name=DEFAULT_EMBEDDING_MODEL,
                max_length=DEFAULT_EMBEDDING_MAX_LENGTH,
                device=device,
            )
            print("[OK] Created MedCPTEmbedding with default model (pooling=cls)")
            return embeddings
        except Exception as e_default:
            print(f"[WARN] Default MedCPT model failed: {e_default}")

    # Fallback path: stock HuggingFaceEmbedding (SentenceTransformer).
    # WARNING: defaults to mean pooling for MedCPT — only kept for legacy models.
    print("[WARN] Falling back to HuggingFaceEmbedding (SentenceTransformer). "
          "Pooling may NOT be CLS for MedCPT in this path.")
    try:
        embeddings = HuggingFaceEmbedding(
            model_name=model_path,
            device=device,
            max_length=DEFAULT_EMBEDDING_MAX_LENGTH,
        )
        print("[OK] Created HuggingFaceEmbedding (fallback, device parameter)")
    except (TypeError, ValueError, Exception) as e:
        print(f"Device parameter approach failed: {e}, trying without device...")
        try:
            embeddings = HuggingFaceEmbedding(
                model_name=model_path,
                max_length=DEFAULT_EMBEDDING_MAX_LENGTH,
            )
            print("[OK] Created HuggingFaceEmbedding (fallback, no device param)")
        except Exception as e2:
            print(f"[FAIL] Failed to create HuggingFaceEmbedding: {e2}")
            print(f"[WARN] Last resort: default model {DEFAULT_EMBEDDING_MODEL}")
            try:
                embeddings = HuggingFaceEmbedding(
                    model_name=DEFAULT_EMBEDDING_MODEL,
                    max_length=DEFAULT_EMBEDDING_MAX_LENGTH,
                )
                print("[OK] Created default HuggingFaceEmbedding (fallback)")
            except Exception as e3:
                raise RuntimeError(f"Failed to load any embedding model. Last error: {e3}")
    
    # Ensure the underlying model is on the correct device
    # HuggingFaceEmbedding wraps a model, try different attribute names
    model_attr_names = ['_model', 'model', 'embed_model', '_embed_model', '_tokenizer']
    model_moved = False
    
    for attr_name in model_attr_names:
        if hasattr(embeddings, attr_name):
            obj = getattr(embeddings, attr_name)
            if obj is not None:
                # Handle model objects
                if hasattr(obj, 'to') and hasattr(obj, 'parameters'):
                    try:
                        # Check current device
                        try:
                            current_device = next(obj.parameters()).device
                            print(f"Embedding {attr_name} current device: {current_device}")
                        except StopIteration:
                            pass
                        
                        # Move to target device
                        obj = obj.to(device)
                        setattr(embeddings, attr_name, obj)
                        if hasattr(obj, 'eval'):
                            obj.eval()
                        model_moved = True
                        
                        # Verify device after moving
                        try:
                            actual_device = next(obj.parameters()).device
                            print(f"Successfully moved embedding {attr_name} to {device} (actual: {actual_device})")
                        except StopIteration:
                            print(f"Moved embedding {attr_name} to {device}")
                        break
                    except Exception as e:
                        print(f"Warning: Could not move {attr_name}: {e}")
                # Handle tokenizer objects (they might also need device)
                elif hasattr(obj, 'device') or (hasattr(obj, '__class__') and 'tokenizer' in str(type(obj)).lower()):
                    # Tokenizers usually don't need device, but log it
                    print(f"Found {attr_name} (likely tokenizer), skipping device move")
    
    # Additional check: try to access the model through __dict__ or dir()
    if not model_moved and torch.cuda.is_available():
        try:
            # Try to find model in all attributes (check private attributes that start with _ but not __)
            for attr in dir(embeddings):
                if not (attr.startswith('_') and not attr.startswith('__')):
                    continue
                try:
                    obj = getattr(embeddings, attr)
                    if obj is not None and hasattr(obj, 'parameters'):
                        try:
                            obj = obj.to(device)
                            setattr(embeddings, attr, obj)
                            if hasattr(obj, 'eval'):
                                obj.eval()
                            model_moved = True
                            print(f"Found and moved model via '{attr}'")
                            break
                        except:
                            pass
                except:
                    pass
        except Exception as e:
            print(f"Error during attribute search: {e}")
    
    if not model_moved and torch.cuda.is_available():
        print("Warning: Could not automatically move embedding model to CUDA. This may cause device mismatch errors.")
        print("Attempting alternative: loading model directly and wrapping...")
        try:
            # Last resort: Load model directly with transformers and wrap it
            from transformers import AutoModel
            direct_model = AutoModel.from_pretrained(
                state_dict_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            if not torch.cuda.is_available():
                direct_model = direct_model.to(device)
            direct_model.eval()
            print(f"Loaded model directly, device: {next(direct_model.parameters()).device}")
            
            # Try to set the model in embeddings
            for attr_name in ['_model', 'model']:
                if hasattr(embeddings, attr_name):
                    setattr(embeddings, attr_name, direct_model)
                    print(f"Set direct model to embeddings.{attr_name}")
                    model_moved = True
                    break
        except Exception as e:
            print(f"Alternative approach also failed: {e}")
            print("You may need to manually ensure the model is on CUDA or check HuggingFaceEmbedding documentation.")
    elif model_moved:
        print("[OK] Embedding model device placement verified")
    
    # embeddings = BertModel.from_pretrained(state_dict_path)
    return embeddings

'''
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

def get_embedding(name):
    encode_kwargs = {"batch_size": 128, 'device': 'cuda'}
    embeddings = HuggingFaceEmbeddings(
        model_name=name,
        encode_kwargs=encode_kwargs,
        # embed_batch_size=128,
    )
    return embeddings
'''