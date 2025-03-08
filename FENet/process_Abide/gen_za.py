
import os.path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 定义GCN层
class GraphConvolutionLayer(nn.Module):
    def __init__(self, in_features, out_features,weight):
        super(GraphConvolutionLayer, self).__init__()
        # self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        # self.bias = nn.Parameter(torch.FloatTensor(out_features))

        self.weight = weight
        # self.bias = bias
    def forward(self, adjacency_matrix, input_features):
        support = torch.mm(input_features, self.weight)
        output = torch.mm(adjacency_matrix, support)
        output=output*(1/116)
        # output=output+self.bias
        return output

class GCN(nn.Module):
    def __init__(self, input_features, hidden_features, output_features,weight1,weight2):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolutionLayer(input_features, hidden_features,weight1)
        self.gc2 = GraphConvolutionLayer(hidden_features, output_features,weight2)

    def forward(self, adjacency_matrix, input_features):
        x = F.relu(self.gc1(adjacency_matrix, input_features))
        x=self.gc2(adjacency_matrix, x)
        x = F.softmax(x,dim=0)
        # x = self.gc2(adjacency_matrix, x)
        return x
weight1 = torch.randn(116, 64)
weight2 = torch.randn(64, 116)

adj_root=r"F:\桌面\学习资料\学姐实验\BrainGNN\BrainGNN_Pytorch-main\data_116_raws\adjacency_matrix"
fea_root=r"F:\桌面\学习资料\学姐实验\BrainGNN\BrainGNN_Pytorch-main\data_116_raws\feature"
save_root=r"F:\桌面\学习资料\学姐实验\GCN_test\Graph-Learning\myGCN\my_data\gcn_output_116raws"
model = GCN(input_features=116, hidden_features=64, output_features=116, weight1=weight1, weight2=weight2)
# 遍历文件夹中的所有文件
for filename in os.listdir(adj_root):
    # 检查文件是否为CSV文件
    if filename.endswith('.csv'):
        number=filename.split("_")[0]
        adj_matrix = pd.read_csv(os.path.join(adj_root,filename))

        features_matrix = pd.read_csv(os.path.join(fea_root,number+".csv")).T
        adj_matrix = torch.tensor(adj_matrix.values, dtype=torch.float32)
        features_matrix = torch.tensor(features_matrix.values, dtype=torch.float32)


        # 定义模型并进行前向传播
        output_features = model(adj_matrix, features_matrix).T
        output_features=output_features.cuda()
        # aa=F.softmax(output_features)
        # output_features=np.array(output_features).T

        # 保存输出的特征矩阵
        # pd.DataFrame(output_features).to_csv(os.path.join(save_root,number+".csv"),index=False)
        torch.save(output_features, os.path.join(save_root,number+".pth"))

