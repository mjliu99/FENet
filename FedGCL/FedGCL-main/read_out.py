import torch
import os
from torch_geometric.nn import global_mean_pool


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch=torch.zeros(116,dtype=torch.long)
batch=batch.to(device)
folder_path = "./final"
file_list = sorted(os.listdir(folder_path))
pth_file_names = [file_name for file_name in file_list if file_name.endswith(".pth")]
for pth_file_name in pth_file_names:
    x=torch.load("./final/"+pth_file_name)
    x=x.t()
    y=global_mean_pool(x,batch)
    torch.save(y,"./read_out/"+pth_file_name)
i=0
for pth_file_name in pth_file_names:

    if i==0:
        y = torch.load("./read_out/" + pth_file_name)
    x = torch.load("./read_out/" + pth_file_name)
    if i != 0:
        y=torch.cat((y,x),dim=0)
    i=i+1
    if(i==175):
        torch.save(y,"./final.pth")


print(1)

