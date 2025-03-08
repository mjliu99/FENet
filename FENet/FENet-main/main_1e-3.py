import argparse
import os
import torch
from torch_geometric.nn import global_mean_pool

from sklearn.metrics import roc_auc_score, recall_score, f1_score

from data.data_loader import Dataset_ECG, Dataset_Dhfm, Dataset_Solar, Dataset_Wiki
from model import CCA_SSG, LogReg
from aug import random_aug
from dataset import load
from torch.utils.data import DataLoader
import numpy as np
import torch as th
import torch.nn as nn

import warnings


warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='CCA-SSG')

parser.add_argument('--dataname', type=str, default='cora', help='Name of dataset.')
parser.add_argument('--gpu', type=int, default=0, help='GPU index.')
parser.add_argument('--epochs', type=int, default=100, help='Training epochs.')
parser.add_argument('--lr1', type=float, default=1e-3, help='Learning rate of CCA-SSG.')
parser.add_argument('--lr2', type=float, default=1e-2, help='Learning rate of linear evaluator.')
parser.add_argument('--wd1', type=float, default=0, help='Weight decay of CCA-SSG.')
parser.add_argument('--wd2', type=float, default=1e-4, help='Weight decay of linear evaluator.')

parser.add_argument('--lambd', type=float, default=1e-3, help='trade-off ratio.')
parser.add_argument('--n_layers', type=int, default=2, help='Number of GNN layers')

parser.add_argument('--use_mlp', action='store_true', default=False, help='Use MLP instead of GNN')

parser.add_argument('--der', type=float, default=0.2, help='Drop edge ratio.')
parser.add_argument('--dfr', type=float, default=0.2, help='Drop feature ratio.')

parser.add_argument("--hid_dim", type=int, default=512, help='Hidden layer dim.')
parser.add_argument("--out_dim", type=int, default=512, help='Output layer dim.')

args = parser.parse_args()

# check cuda
if args.gpu != -1 and th.cuda.is_available():
    args.device = 'cuda:{}'.format(args.gpu)
else:
    args.device = 'cpu'

if __name__ == '__main__':

    # 傅里叶GNN#
    parser1 = argparse.ArgumentParser(description='fourier graph network for multivariate time series forecasting')
    parser1.add_argument('--data', type=str, default='NYU', help='data set')
    parser1.add_argument('--feature_size', type=int, default=176, help='feature size')
    parser1.add_argument('--seq_length', type=int, default=30, help='inout length')
    parser1.add_argument('--pre_length', type=int, default=30, help='predict length')
    parser1.add_argument('--embed_size', type=int, default=128, help='hidden dimensions')
    parser1.add_argument('--hidden_size', type=int, default=256, help='hidden dimensions')
    parser1.add_argument('--train_epochs', type=int, default=100, help='train epochs')
    parser1.add_argument('--batch_size', type=int, default=176, help='input data batch size')
    parser1.add_argument('--learning_rate', type=float, default=0.00001, help='optimizer learning rate')
    parser1.add_argument('--exponential_decay_step', type=int, default=5)
    parser1.add_argument('--validate_freq', type=int, default=1)
    parser1.add_argument('--early_stop', type=bool, default=False)
    parser1.add_argument('--decay_rate', type=float, default=0.5)
    parser1.add_argument('--train_ratio', type=float, default=1)
    parser1.add_argument('--val_ratio', type=float, default=0)
    parser1.add_argument('--device', type=str, default='cuda:0', help='device')
    args1 = parser1.parse_args()

    # data set
    data_parser = {
        'traffic': {'root_path': 'data/traffic.npy', 'type': '0'},
        'ECG': {'root_path': 'data/ECG_data.csv', 'type': '1'},
        'COVID': {'root_path': 'data/covid.csv', 'type': '1'},
        'electricity': {'root_path': 'data/electricity.csv', 'type': '1'},
        'solar': {'root_path': '/data/solar', 'type': '1'},
        'metr': {'root_path': 'data/metr.csv', 'type': '1'},
        'wiki': {'root_path': 'data/wiki.csv', 'type': '1'},
        'NYU': {'root_path': 'data/NYU_50952.csv', 'type': '1'},
    }
    folder_path = "csv_data_NYU_shouhang/"
    file_list = sorted(os.listdir(folder_path))
    csv_file_names = [file_name for file_name in file_list if file_name.endswith(".csv")]
    epoch=0

    in_dim = 116
    model = CCA_SSG(in_dim, args.hid_dim, args.out_dim, args.n_layers, args.use_mlp)
    model = model.to(args.device)
    optimizer = th.optim.Adam(model.parameters(), lr=args.lr1, weight_decay=args.wd1)
    N = 116

    model.train()
    optimizer.zero_grad()

    for csv_file_name in csv_file_names:
        data_parser['NYU']['root_path'] = folder_path + csv_file_name

        if args1.data in data_parser.keys():
            data_info = data_parser[args1.data]

        data_dict = {
            'ECG': Dataset_ECG,
            'COVID': Dataset_ECG,
            'traffic': Dataset_Dhfm,
            'solar': Dataset_Solar,
            'wiki': Dataset_Wiki,
            'electricity': Dataset_ECG,
            'metr': Dataset_ECG,
            'NYU': Dataset_ECG
        }

        Data = data_dict[args1.data]
        # train val test
        train_set = Data(root_path=data_info['root_path'], flag='train', seq_len=args1.seq_length,
                         pre_len=args1.pre_length,
                         type=data_info['type'], train_ratio=args1.train_ratio, val_ratio=args1.val_ratio)
        test_set = Data(root_path=data_info['root_path'], flag='test', seq_len=args1.seq_length, pre_len=args1.pre_length,
                        type=data_info['type'], train_ratio=args1.train_ratio, val_ratio=args1.val_ratio)
        val_set = Data(root_path=data_info['root_path'], flag='val', seq_len=args1.seq_length, pre_len=args1.pre_length,
                       type=data_info['type'], train_ratio=args1.train_ratio, val_ratio=args1.val_ratio)

        train_dataloader = DataLoader(
            train_set,
            batch_size=args1.batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=False
        )
        for epoch1 in range(99):
            for index, (x, y) in enumerate(train_dataloader):
                x = x.float().to("cuda:0")
                z2 = model(x)
                if(epoch1==98):
                   th.save(x, "./x_1e-3/" + csv_file_name[0:5] + ".pth")
        z1 = th.load("./Za/"+csv_file_name[0:5]+".pth")
        z1 = z1.to(args.device)
        c = th.mm(z1, z2)
        c1 = th.mm(z1, z1.T)
        c2 = th.mm(z2.T, z2)
        c = c / N
        c1 = c1 / N
        c2 = c2 / N
        loss_inv = -th.diagonal(c).sum()
        iden = th.tensor(np.eye(c.shape[0])).to(args.device)
        loss_dec1 = (iden - c1).pow(2).sum()
        loss_dec2 = (iden - c2).pow(2).sum()

        loss = loss_inv + args.lambd * (loss_dec1 + loss_dec2)

        loss.backward()
        optimizer.step()
        epoch+=1

        print('Epoch={:03d}, loss={:.4f}'.format(epoch, loss.item()))


###############得到训练完成的频域数据#########################
    folder_path = "./x_1e-3"
    file_list = sorted(os.listdir(folder_path))
    pth_file_names = [file_name for file_name in file_list if file_name.endswith(".pth")]
    for pth_file_name in pth_file_names:
        f=th.load("./x_1e-3/"+pth_file_name)
        final=model.get_embedding(f)
        th.save(final,"./final_1e-3/"+pth_file_name)
        print(pth_file_name)
# ###############得到训练完成的频域数据#########################

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch = torch.zeros(116, dtype=torch.long)
    batch = batch.to(device)
    folder_path = "final"
    file_list = sorted(os.listdir(folder_path))
    pth_file_names = [file_name for file_name in file_list if file_name.endswith(".pth")]
    for pth_file_name in pth_file_names:
        x = torch.load("./final_1e-3/" + pth_file_name)
        x = x.t()
        y = global_mean_pool(x, batch)
        torch.save(y, "./read_out_1e-3/" + pth_file_name)
    i = 0
    for pth_file_name in pth_file_names:
        if i == 0:
            y = torch.load("./read_out_1e-3/" + pth_file_name)
        x = torch.load("./read_out_1e-3/" + pth_file_name)
        if i != 0:
            y = torch.cat((y, x), dim=0)
        i = i + 1
    torch.save(y, "./final_1e-3.pth")
    print(args.lambd)

