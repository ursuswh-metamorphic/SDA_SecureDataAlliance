# Setup GPU Server cho FedE Training

## 1. Yêu cầu tối thiểu
- GPU: NVIDIA với >= 12GB VRAM (khuyến nghị >= 24GB)
- RAM: >= 16GB
- Disk: >= 20GB trống
- CUDA: >= 12.1
- OS: Ubuntu 20.04+

## 2. Cài đặt Miniconda

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
bash /tmp/miniconda.sh -b -p ~/miniconda3
~/miniconda3/bin/conda init bash
source ~/.bashrc

# Chấp nhận Terms of Service
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
```

## 3. Tạo Conda Environment

```bash
conda create -n fedrag python=3.11 -y
conda activate fedrag
```

## 4. Cài đặt Dependencies

```bash
# PyTorch + CUDA 12.1
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

# Transformers + các thư viện khác
pip install transformers==4.35.0 "numpy<2" scipy prettytable ujson pyyaml pynvml matplotlib torchtext==0.16.0 torchdata==0.7.0
```

## 5. Clone Repo

```bash
cd ~
git clone https://github.com/ursuswh-metamorphic/SDA_SecureDataAlliance.git
cd SDA_SecureDataAlliance
git checkout trang/differential_privacy
```

## 6. Kiểm tra

```bash
conda activate fedrag
python -c "import torch; print(f'torch={torch.__version__}, cuda={torch.cuda.is_available()}, gpu={torch.cuda.get_device_name(0)}')"
```

## 7. Chạy Training

```bash
cd ~/SDA_SecureDataAlliance/FedE
mkdir -p logs

# Baseline (không DP)
nohup python main.py > logs/baseline_output.log 2> logs/baseline_error.log &

# DP eps=20 (khuyến nghị cho production)
nohup python main_dp_eps20.py > logs/dp_eps20_output.log 2> logs/dp_eps20_error.log &

# DP eps=8 (strict privacy)
nohup python main_dp.py > logs/dp_output.log 2> logs/dp_error.log &

# DP-LoRA eps=8 (parameter-efficient, chưa test)
nohup python main_dp_lora.py > logs/dp_lora_output.log 2> logs/dp_lora_error.log &
```

## 8. Theo dõi Training

```bash
# Xem output log
tail -f logs/dp_eps20_output.log

# Xem error/progress log
tail -f logs/dp_eps20_error.log

# Kiểm tra GPU
nvidia-smi

# Kiểm tra process
ps aux | grep main_dp
```

## 9. Tải Log về Local (từ máy Windows)

```bash
scp -P <port> root@<server-ip>:/path/to/SDA_SecureDataAlliance/FedE/logs/*.log ./FedE/logs/
```

## Lịch sử Server đã dùng

| Server | GPU | VRAM | SSH |
|---|---|---|---|
| Vast.ai | RTX 3060 | 12GB | `ssh -p 17187 root@91.150.160.38` |
| DigitalOcean | RTX 6000 Ada | 48GB | `ssh root@159.89.116.198` |
