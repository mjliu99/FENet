import argparse
import os
from torch_geometric.nn import global_mean_pool
import random

import torch
from sklearn.metrics import roc_auc_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from data.data_loader import Dataset_ECG, Dataset_Dhfm, Dataset_Solar, Dataset_Wiki
from model import CCA_SSG, LogReg
from aug import random_aug
from dataset import load
from torch.utils.data import DataLoader
import numpy as np
import torch as th
import torch.nn as nn
import matplotlib.pyplot as plt
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
def main():
    min_acc = 2
    max_acc = 0
    mean_acc = 0
    auc = 0;
    recall = 0
    f1 = 0
    SD_acc = []
    SD_auc = []
    SD_recall = []
    SD_f1 = []
    for i in range(100):
        eval_acc, eval_auc, eval_recall, eval_f1 = lg(max_acc)
        SD_acc.append(float(eval_acc.to('cpu')))

        SD_auc.append(eval_auc)
        SD_f1.append(eval_f1)
        SD_recall.append(eval_recall)
        if eval_acc > max_acc:
            max_acc = eval_acc
            recall = eval_recall
            f1 = eval_f1
            auc = eval_auc
        if eval_acc < min_acc:
            min_acc = eval_acc
        mean_acc = mean_acc + eval_acc
    sd_acc = float(np.std(SD_acc))

    sd_auc = float(np.std(SD_auc))
    sd_recall = float(np.std(SD_recall))
    sd_f1 = float(np.std(SD_f1))
    mean_acc = mean_acc / 100
    print('mean_acc:{:.4f}'.format(mean_acc), 'max_acc:{:.4f}'.format(max_acc), 'min_acc:{:.4f}'.format(min_acc),
          'auc:{:.4f}'.format(auc), 'recall:{:.4f}'.format(recall), 'f1:{:.4f}'.format(f1)
          , 'SD_acc:{:.4f}'.format(sd_acc), 'SD_auc:{:.4f}'.format(sd_auc), 'SD_recall:{:.4f}'.format(sd_recall),
          'SD_f1:{:.4f}'.format(sd_f1))

def lg(maxacc):
    ''' Linear Evaluation '''
    final_embeds = th.load("./final.pth").to(args.device)

    train_idx = th.load("./120/train_id.pth").to(args.device)
    test_idx = th.load("./120/test_id.pth").to(args.device)
    val_idx = th.load("./120/val_id.pth").to(args.device)

    train_embs = final_embeds[train_idx]
    test_embs = final_embeds[test_idx]
    val_embs = final_embeds[val_idx]


    labels = th.load("./120/labels.pth").to(args.device)
    labels=labels.reshape(-1)
    train_labels = labels[train_idx]
    test_labels = labels[test_idx]
    val_labels = labels[val_idx]

    logreg = LogReg(176, 2)

    opt = th.optim.Adam(logreg.parameters(), lr=args.lr2, weight_decay=args.wd2)

    logreg = logreg.to(args.device)
    loss_fn = nn.CrossEntropyLoss()

    best_val_acc = 0
    eval_acc = 0

    eval_auc = 0

    best_val_recall = 0
    eval_recall = 0

    best_val_f1 = 0
    eval_f1 = 0
    best_epoch = 0
    for epoch in range(500):
        logreg.train()
        opt.zero_grad()
        logits = logreg(train_embs)
        preds = th.argmax(logits, dim=1)
        train_acc = th.sum(preds == train_labels).float() / train_labels.shape[0]

        loss = loss_fn(logits, train_labels.long())
        loss.backward()
        opt.step()

        logreg.eval()
        with th.no_grad():

            val_logits = logreg(val_embs)
            # print(val_logits.shape)
            test_logits = logreg(test_embs)

            val_preds = th.argmax(val_logits, dim=1)
            test_preds = th.argmax(test_logits, dim=1)

            val_acc = th.sum(val_preds == val_labels).float() / val_labels.shape[0]
            test_acc = th.sum(test_preds == test_labels).float() / test_labels.shape[0]

            # Calculate AUC
            val_auc = roc_auc_score(val_labels.cpu().numpy(), val_logits[:, 1].cpu().numpy())
            test_auc = roc_auc_score(test_labels.cpu().numpy(), test_logits[:, 1].cpu().numpy())

            # Calculate recall
            val_recall = recall_score(val_labels.cpu().numpy(), val_preds.cpu().numpy())
            test_recall = recall_score(test_labels.cpu().numpy(), test_preds.cpu().numpy())

            # Calculate F1 score
            val_f1 = f1_score(val_labels.cpu().numpy(), val_preds.cpu().numpy())
            test_f1 = f1_score(test_labels.cpu().numpy(), test_preds.cpu().numpy())

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                eval_acc = test_acc
                eval_auc = test_auc
                eval_recall = test_recall
                eval_f1 = test_f1
                if eval_acc > maxacc:
                    torch.save(logreg.state_dict(), './120/best_model.pth')
                best_epoch = epoch
                print(epoch)

    print('best_accuracy:{:.4f}'.format(eval_acc), 'auc:{:.4f}'.format(eval_auc), 'recall:{:.4f}'.format(eval_recall),
          'f1:{:.4f}'.format(eval_f1), 'best_acc_epoch:', best_epoch)
    return eval_acc, eval_auc, eval_recall, eval_f1


if __name__ == '__main__':
    main()

