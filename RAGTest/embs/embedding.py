# from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from transformers import AutoModel,AutoTokenizer
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import torch
from transformers import BertModel
import os

def get_embedding(state_dict_path):    
    print("embedding_path: ", state_dict_path)

    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device for embeddings: {device}")

    # Try multiple approaches to ensure device placement
    # Approach 1: Try with model_kwargs (if supported)
    try:
        embeddings = HuggingFaceEmbedding(
            model_name=state_dict_path,
            model_kwargs={"device_map": "auto", "torch_dtype": torch.float16} if torch.cuda.is_available() else {},
            # embed_batch_size=128,
        )
        print("Created HuggingFaceEmbedding with model_kwargs")
    except (TypeError, ValueError) as e:
        print(f"model_kwargs approach failed: {e}, trying alternative...")
        # Approach 2: Try with device parameter
        try:
            embeddings = HuggingFaceEmbedding(
                model_name=state_dict_path,
                device=device,
                # embed_batch_size=128,
            )
            print("Created HuggingFaceEmbedding with device parameter")
        except TypeError:
            # Approach 3: Create without device, then move manually
            embeddings = HuggingFaceEmbedding(
                model_name=state_dict_path,
                # embed_batch_size=128,
            )
            print("Created HuggingFaceEmbedding without device parameter")
    
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
        print("✓ Embedding model device placement verified")
    
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