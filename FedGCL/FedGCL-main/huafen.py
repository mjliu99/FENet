
import os
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

# 设置随机种子以确保可复现性
random_seed = 42

files_path = r'./feature'
labels_path = r'./labels.csv'
save_root = r"./"

# 获取文件列表并提取索引
files = os.listdir(files_path)
idx = [i.split(".")[0] for i in files]

# 读取标签文件
labels = pd.read_csv(labels_path)

# l=torch.tensor(labels)
# 从DataFrame中提取标签列
labels_column = labels

# 将标签列转换为PyTorch张量
labels_tensor = torch.tensor(labels_column.values)
torch.save(labels_tensor, os.path.join(save_root,"labels.pth"))

# 对训练-测试划分进行分层抽样
train_idx, test_idx, train_labels, test_labels = train_test_split(
    idx, labels, test_size=0.2, random_state=random_seed, stratify=labels
)

# 对训练-验证划分进行分层抽样
train_idx, val_idx, train_labels, val_labels = train_test_split(
    train_idx, train_labels, test_size=0.2, random_state=random_seed, stratify=train_labels
)
train_idx=sorted(train_idx)
val_idx=sorted(val_idx)
test_idx=sorted(test_idx)
k1,k2,k3=0,0,0
for i in range(len(idx)):
    if k1<len(train_idx) and idx[i]==train_idx[k1]:
        train_idx[k1]=i
        k1=k1+1
    if k2<len(test_idx) and idx[i]==test_idx[k2]:
        test_idx[k2]=i
        k2=k2+1
    if k3<len(val_idx) and idx[i]==val_idx[k3]:
        val_idx[k3]=i
        k3=k3+1
# 保存索引
# pd.DataFrame(train_idx).to_csv(os.path.join(save_root, "train_idx.csv"), index=False)
# pd.DataFrame(val_idx).to_csv(os.path.join(save_root, "val_idx.csv"), index=False)
# pd.DataFrame(test_idx).to_csv(os.path.join(save_root, "test_idx.csv"), index=False)
torch.save(torch.tensor(train_idx), os.path.join(save_root,"train_id.pth"))
torch.save(torch.tensor(test_idx), os.path.join(save_root,"test_id.pth"))
torch.save(torch.tensor(val_idx), os.path.join(save_root,"val_id.pth"))
print(1)
