from typing import Any, List, Optional

from transformers import AutoModel, AutoTokenizer
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.bridge.pydantic import Field, PrivateAttr
import torch
import os

# Default embedding backbone — must match upstream MedCPT Article Encoder training.
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

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device for embeddings: {device}")

    # Primary path: MedCPT-compatible wrapper (AutoModel + [CLS] pooling + 512 max tokens).
    # Matches upstream MedCPT training exactly.
    try:
        embeddings = MedCPTEmbedding(
            model_name=state_dict_path,
            max_length=DEFAULT_EMBEDDING_MAX_LENGTH,
            device=device,
        )
        print(f"[OK] Created MedCPTEmbedding (pooling=cls, max_length={DEFAULT_EMBEDDING_MAX_LENGTH}, device={device})")
        return embeddings
    except Exception as e_medcpt:
        print(f"[WARN] MedCPTEmbedding load failed: {e_medcpt}")
        print(f"[WARN] Falling back to default model: {DEFAULT_EMBEDDING_MODEL}")
        try:
            embeddings = MedCPTEmbedding(
                model_name=DEFAULT_EMBEDDING_MODEL,
                max_length=DEFAULT_EMBEDDING_MAX_LENGTH,
                device=device,
            )
            print("[OK] Created MedCPTEmbedding with default model (pooling=cls)")
            return embeddings
        except Exception as e_default:
            print(f"[WARN] Default MedCPT model also failed: {e_default}")

    # Fallback: stock HuggingFaceEmbedding (SentenceTransformer).
    # WARNING: defaults to mean pooling — only kept for legacy/non-MedCPT models.
    print("[WARN] Falling back to HuggingFaceEmbedding (SentenceTransformer). "
          "Pooling may NOT be CLS for MedCPT in this path.")
    try:
        embeddings = HuggingFaceEmbedding(
            model_name=state_dict_path,
            device=device,
            max_length=DEFAULT_EMBEDDING_MAX_LENGTH,
        )
        print("[OK] Created HuggingFaceEmbedding (fallback)")
        return embeddings
    except Exception as e:
        print(f"[WARN] HuggingFaceEmbedding failed: {e}")
        embeddings = HuggingFaceEmbedding(
            model_name=DEFAULT_EMBEDDING_MODEL,
            max_length=DEFAULT_EMBEDDING_MAX_LENGTH,
        )
        print("[OK] Created default HuggingFaceEmbedding (last resort)")
        return embeddings
