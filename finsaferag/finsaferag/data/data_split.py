from math import ceil

from datasets import load_dataset
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import GroupedNaturalIdPartitioner

# 1) Đường dẫn tới train_data trên Hugging Face
data_files = "hf://datasets/DocAILab/FedE4RAG_Dataset/FEDE4FIN/train_data/data_*.json"

print("🔹 Loading train dataset (temporary) from:", data_files)

# 2) Load tạm dataset để tính số company và group_size
tmp_ds = load_dataset(
    "json",
    data_files=data_files,
    split="train",
)

print("Loaded temp train dataset")
print("Number of examples:", len(tmp_ds))

companies = tmp_ds.unique("company")
num_companies = len(companies)
print(f"Number of companies: {num_companies}")
target_num_clients = 7

# group_size ~ số company / số client mong muốn
group_size = max(1, ceil(num_companies / target_num_clients))
print(group_size)

print(f"🔹 Num companies: {num_companies}")
print(f"🔹 Target clients: {target_num_clients}, chosen group_size: {group_size}")

# 3) Tạo GroupedNaturalIdPartitioner dựa trên cột 'company'
partitioner = GroupedNaturalIdPartitioner(
    partition_by="company",
    group_size=group_size,
)

# 4) Tạo FederatedDataset "chuẩn Flower"
fds = FederatedDataset(
    dataset="json",                   # dùng HF json builder
    partitioners={"train": partitioner},
    shuffle=True,
    seed=42,
    data_files={"train": data_files}, # truyền vào load_dataset_kwargs
)

print("\n🔹 Federated Dataset created")

# 5) Gọi load_partition(0, "train") để trigger việc load + gán dataset cho partitioner
first_partition = fds.load_partition(0, "train")

# Sau khi dataset được chuẩn bị, partitioner mới biết có bao nhiêu partition
num_partitions = partitioner.num_partitions
print("🔹 Number of partitions (clients actually created):", num_partitions)

# 6) Lặp qua tối đa 7 client (hoặc ít hơn nếu số partition < 7)
max_clients_to_show = min(7, num_partitions)

for pid in range(max_clients_to_show):
    part = fds.load_partition(pid, "train")
    print(f"\n=== Client {pid} ===")
    print("Num examples:", len(part))

    companies_in_part = part.unique("company")
    print("Num unique companies:", len(companies_in_part))
    print("Some companies:", companies_in_part[:5])

# 7) Demo: xem 1 sample của client 0
if num_partitions > 0:
    partition0 = fds.load_partition(0, "train")
    print("\nSample from client 0:")
    print(partition0[0])
