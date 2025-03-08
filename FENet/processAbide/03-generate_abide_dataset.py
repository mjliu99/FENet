import deepdish as dd
import os.path as osp
import os

import numpy
import numpy as np
import argparse
from pathlib import Path
import pandas as pd
from scipy.io import loadmat
from scipy.signal import resample


def main():

    root_path=r"F:\桌面\学习资料\学姐实验\BrainGNN\BrainGNN_Pytorch-main\data_aal"
    # data_dir =  os.path.join(args.root_path, 'ABIDE_pcp/cpac/filt_noglobal/raw')
    site_path=r"target_all\NYU"
    data_dir =  os.path.join(root_path, 'ABIDE_pcp/cpac/filt_noglobal/raw')
    adjacency_matrix_path=r'F:\桌面\学习资料\学姐实验\BrainGNN\BrainGNN_Pytorch-main\adjacency_matrix_all'
    feature_all_path=r""
    csv_path=r'F:\桌面\学习资料\学姐实验\BrainGNN\BrainGNN_Pytorch-main\csv_data_NYU'

    timeseires = os.path.join(root_path, site_path)

    meta_file = os.path.join(root_path, 'ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv')

    meta_file = pd.read_csv(meta_file, header=0)

    id2site = meta_file[["subject", "SITE_ID"]]

    # pandas to map
    id2site = id2site.set_index("subject")
    id2site = id2site.to_dict()['SITE_ID']

    times = []

    labels = []
    pcorrs = []

    corrs = []

    site_list = []

    for f in os.listdir(timeseires):
        fname = f
        site = id2site[int(fname)]
        # 这里site为机构名称如PITT

        files = os.listdir(osp.join(timeseires, fname))

        file = list(filter(lambda x: x.endswith("1D"), files))[0]

        time = np.loadtxt(osp.join(timeseires, fname, file), skiprows=0)
        # time = np.loadtxt(osp.join(timeseires, fname, file), skiprows=0)

        pd.DataFrame(time).to_csv(os.path.join(csv_path,fname+".csv"), index=False)
        # 读取的1D时间序列文件，转置
        corr_mat_file=loadmat(osp.join(timeseires,fname,f+"_aal_correlation.mat"))

        f=f+".h5"
        temp = dd.io.load(osp.join(data_dir, f))
        print(temp)

        att=corr_mat_file['connectivity']
        pd.DataFrame(att).to_csv(os.path.join(adjacency_matrix_path,fname+"_adjacency_matrix.csv"), index=False)

        label = temp['label']



        times.append(time[:, :])
        labels.append(label[0])
        corrs.append(att)


        site_list.append(site)
    pd.DataFrame(labels).to_csv("./labels.csv", index=False)

    times=np.array(times)
    labels=np.array(labels)
    corrs=np.array(corrs)

#
# # 整理大于等于176时间序列的所有站点数据
def load_all():
      root_path=r"F:\桌面\学习资料\学姐实验\BrainGNN\BrainGNN_Pytorch-main\data_aal\ABIDE_pcp\cpac\filt_noglobal"
      save_adj_path=r"F:\桌面\学习资料\学姐实验\BrainGNN\BrainGNN_Pytorch-main\adjacency_matrix_all"
      save_fea_path=r"F:\桌面\学习资料\学姐实验\BrainGNN\BrainGNN_Pytorch-main\feature_all"
      raw_path=r"F:\桌面\学习资料\学姐实验\BrainGNN\BrainGNN_Pytorch-main\data_aal\ABIDE_pcp\cpac\filt_noglobal\raw"
      labels_path=r"./300/labels.csv"
      data_116_raws="../300"
      labels=[]
      meta_file = '../data_aal/ABIDE_pcp/Phenotypic_V1_0b_preprocessed1.csv'

      meta_file = pd.read_csv(meta_file, header=0)

      id2age = meta_file[["subject", "AGE_AT_SCAN"]]
      # pandas to map
      id2age = id2age.set_index("subject")
      id2age = id2age.to_dict()['AGE_AT_SCAN']

      id2group = meta_file[["subject", "DX_GROUP"]]
      # pandas to map
      id2group = id2group.set_index("subject")
      id2group = id2group.to_dict()['DX_GROUP']
      ids=[]
      max2age = 0
      min2age = 100
      max1age = 0
      min1age = 100
      count1 = 0
      count2 = 0
      i=-1
      id0=[]
      count=0
      for f in os.listdir(root_path):
          if f=="raw":
              break
          files = os.listdir(osp.join(root_path, f))
          path=os.path.join(root_path,f)
          data=np.loadtxt(os.path.join(path,files[2]))
          print(f)
          i=i+1
          if(len(data)>=176 and count<300):
              count=count+1
              group=id2group.get(int(f))
              age=id2age.get(int(f))
              if group==1:
                  count1=count1+1
                  # max1age=(max1age > age )? max1age : age
                  if age>max1age:
                      max1age=age
                  if age<min1age:
                      min1age=age
              if group==2:
                  count2=count2+1
                  if age>max2age:
                      max2age=age
                  if age<min2age:
                      min2age=age
              data=jcy(data)
              # pd.DataFrame(data).to_csv(os.path.join(data_116_raws,"feature/"+f+".csv"),index=False)
              # adj=loadmat(os.path.join(path,files[0]))['connectivity']
              temp = dd.io.load(osp.join(raw_path, f+".h5"))
              labels.append(temp["label"][0])
              ids.append(f)
              # pd.DataFrame(adj).to_csv(os.path.join(data_116_raws,"adjacency_matrix/"+f+"_adjacency_matrix.csv"),index=False)
            # id0.append(i)
      pd.DataFrame(labels).to_csv(os.path.join(data_116_raws,"labels_300.csv"),index=False)
      pd.DataFrame(ids).to_csv(os.path.join(data_116_raws,"ids_300.csv"),index=False)
      # pd.DataFrame(id0).to_csv(os.path.join(labels_path,"id0.csv"),index=False)

#     时间序列降采样到176


if __name__ == '__main__':
    # main()
    load_all()
