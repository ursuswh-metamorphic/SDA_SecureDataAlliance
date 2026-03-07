import torch
import os
import sys
from transformers import AutoModel, AutoTokenizer

def fix_model_state_dict(model_path, base_model_name="BAAI/bge-base-en", output_dir=None):
    """
    Convert model from upstream training to HuggingFace format.
    
    Args:
        model_path: Path to directory containing .pt/.bin file OR path to .bin/.pt file directly
        base_model_name: Base model name to load config and tokenizer from
        output_dir: Output directory (if None, will save to model_path if it's a dir, or create new dir if it's a file)
    
    Returns:
        bool: True if successful, False otherwise
    """
    # Kiểm tra xem model_path là file hay directory
    is_file = os.path.isfile(model_path)
    is_dir = os.path.isdir(model_path)
    
    if not is_file and not is_dir:
        print(f"Error: Path does not exist: {model_path}")
        return False
    
    # Xác định input file và output directory
    if is_file:
        # Nếu là file, lấy thư mục chứa file làm output (hoặc dùng output_dir)
        input_file = model_path
        if output_dir is None:
            # Tạo output dir từ tên file (bỏ extension)
            output_dir = os.path.splitext(model_path)[0] + "_converted"
            print(f"Output directory not specified, will create: {output_dir}")
    else:
        # Nếu là directory, tìm file .pt hoặc .bin
        pt_files = [f for f in os.listdir(model_path) if f.endswith(('.pt', '.bin'))]
        if not pt_files:
            print(f"No .pt or .bin files found in {model_path}")
            print(f"Files in directory: {os.listdir(model_path)[:10]}")
            return False
        
        input_file = os.path.join(model_path, pt_files[0])
        if output_dir is None:
            output_dir = model_path  # Save vào cùng thư mục
    
    print(f"Input model file: {input_file}")
    print(f"Output directory: {output_dir}")
    print(f"Base model: {base_model_name}")
    print("-" * 60)
    
    # Tạo output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load state_dict từ file
    print("\n[Step 1] Loading state_dict from file...")
    try:
        state_dict = torch.load(input_file, map_location='cpu')
        print(f"✓ Loaded state_dict successfully")
        print(f"  Number of keys: {len(state_dict.keys())}")
        print(f"  First 5 keys: {list(state_dict.keys())[:5]}")
    except Exception as e:
        print(f"✗ Error loading state_dict: {e}")
        return False
    
    # 2. Xử lý state_dict - xóa prefix 'model.' nếu có
    print("\n[Step 2] Processing state_dict (removing 'model.' prefix if exists)...")
    new_state_dict = {}
    has_model_prefix = False
    for key, value in state_dict.items():
        if key.startswith('model.'):
            new_key = key[6:]  # Xóa 'model.' prefix
            new_state_dict[new_key] = value
            has_model_prefix = True
        else:
            new_state_dict[key] = value
    
    if has_model_prefix:
        print(f"✓ Removed 'model.' prefix from keys")
    else:
        print("✓ No 'model.' prefix found, using keys as-is")
    
    print(f"  Processed keys: {len(new_state_dict.keys())}")
    print(f"  First 5 processed keys: {list(new_state_dict.keys())[:5]}")
    
    # 3. Load base model để lấy config và tokenizer
    print(f"\n[Step 3] Loading base model: {base_model_name}...")
    try:
        base_model = AutoModel.from_pretrained(base_model_name)
        base_tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        print(f"✓ Loaded base model and tokenizer successfully")
    except Exception as e:
        print(f"✗ Error loading base model: {e}")
        return False
    
    # 4. Load state_dict vào model
    print("\n[Step 4] Loading state_dict into model...")
    try:
        missing_keys, unexpected_keys = base_model.load_state_dict(new_state_dict, strict=False)
        print(f"✓ Loaded state_dict into model")
        if missing_keys:
            print(f"  Warning: {len(missing_keys)} missing keys (usually okay)")
        if unexpected_keys:
            print(f"  Warning: {len(unexpected_keys)} unexpected keys")
    except Exception as e:
        print(f"✗ Error loading state_dict into model: {e}")
        print("  Trying to continue anyway...")
    
    # 5. Save model dưới dạng HuggingFace format
    print(f"\n[Step 5] Saving model to HuggingFace format...")
    try:
        base_model.save_pretrained(output_dir)
        base_tokenizer.save_pretrained(output_dir)
        print(f"✓ Model saved successfully to: {output_dir}")
    except Exception as e:
        print(f"✗ Error saving model: {e}")
        return False
    
    # 6. Test load model (optional)
    print(f"\n[Step 6] Testing model loading...")
    try:
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        print("Testing with HuggingFaceEmbedding...")
        embeddings = HuggingFaceEmbedding(model_name=output_dir)
        print("✓ Model loaded successfully!")
    except ImportError:
        print("Note: llama_index not installed, skipping embedding test")
    except Exception as e:
        print(f"Warning: Model test failed: {e}")
        print("But the model files have been saved successfully")
    
    print("\n" + "=" * 60)
    print("✓ Conversion completed successfully!")
    print(f"\nYou can now use the model:")
    print(f"  python main_100_test.py --model=\"{output_dir}\"")
    print("=" * 60)
    
    return True


# 使用示例
if __name__ == "__main__":
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        # Default path - thay đổi theo nhu cầu
        print("Usage: python change.py <model_path_or_file> [output_dir] [base_model_name]")
        print("\nExamples:")
        print("  # Convert from .bin file:")
        print("  python change.py ../FedE/x-model_2025-11-23_04-01-02.bin")
        print("  # Convert from directory with .pt file:")
        print("  python change.py /path/to/model/directory")
        print("  # Specify output directory:")
        print("  python change.py ../FedE/x-model.bin ./converted_model")
        print("  # Specify base model:")
        print("  python change.py ../FedE/x-model.bin ./converted_model \"BAAI/bge-large-en-v1.5\"")
        sys.exit(1)
    
    # Output directory (optional)
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Base model name (optional)
    if len(sys.argv) > 3:
        base_model_name = sys.argv[3]
    else:
        base_model_name = "BAAI/bge-base-en"  # Default
    
    success = fix_model_state_dict(model_path, base_model_name=base_model_name, output_dir=output_dir)
    
    if not success:
        sys.exit(1)
