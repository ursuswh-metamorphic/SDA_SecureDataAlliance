"""
train_data (torch.utils.data.Dataset),
test_data (torch.utils.data.Dataset),
and the model (torch.nn.Module) should be implemented here.

"""
import torch.nn
from transformers import AutoModel

# Upstream embedding backbone: MedCPT Article Encoder
# https://huggingface.co/ncbi/MedCPT-Article-Encoder
EMBEDDING_MODEL_NAME = "ncbi/MedCPT-Article-Encoder"

train_data = None
val_data = None
test_data = None
vocab = None
tokenizer = None

def get_model(*args, **kwargs) -> torch.nn.Module:
    # TODO 加载embedding模型 在largemodel
    model = AutoModel.from_pretrained(EMBEDDING_MODEL_NAME)
    return model