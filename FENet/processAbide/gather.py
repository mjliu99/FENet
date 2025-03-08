import os
import shutil
import numpy as np
import pandas as pd
from scipy.signal import resample


# todo.........收集数据.............

# source_dir = r"F:\桌面\学习资料\学姐实验\BrainGNN\BrainGNN_Pytorch-main\data_aal\ABIDE_pcp\cpac\filt_noglobal"
# target_dir = r"F:\桌面\学习资料\学姐实验\BrainGNN\BrainGNN_Pytorch-main\data_aal\target"
# # 遍历源目录下的所有子目录
# for dirpath, dirnames, filenames in os.walk(source_dir):
#     # 检查当前目录是否包含1D文件
#     if any(filename.endswith('.1D') for filename in filenames):
#         # 获取1D文件的文件名（不包含扩展名）
#         filename_without_extension = filenames[2].split('.')[0]
#         # 构建目标文件夹路径和源文件路径
#         target_folder = os.path.join(target_dir, filename_without_extension.split("_")[0])
#         source_file = os.path.join(dirpath, filenames[2])
#         # 如果目标文件夹不存在，则创建它
#         if not os.path.exists(target_folder):
#             os.makedirs(target_folder)
#             # 复制1D文件到目标文件夹
#         shutil.copy(source_file, target_folder)

# ............数文件数
# target_dir = r"F:\桌面\学习资料\学姐实验\BrainGNN\BrainGNN_Pytorch-main\data_aal\target"
#
# # 遍历目标目录下的所有文件夹
# for foldername in os.listdir(target_dir):
#     folderpath = os.path.join(target_dir, foldername)
#     if os.path.isdir(folderpath):
#         # 统计文件夹中的文件数量
#         file_count = len([f for f in os.listdir(folderpath) if os.path.isfile(os.path.join(folderpath, f))])
#         # 构建新的文件夹名称
#         new_foldername = f"{foldername}({file_count})"
#         # 重命名文件夹
#         os.rename(folderpath, os.path.join(target_dir, new_foldername))



# ......数行数
# target_dir = r"F:\桌面\学习资料\学姐实验\BrainGNN\BrainGNN_Pytorch-main\data_aal\target"
#
# 遍历目标目录下的所有文件夹
# for foldername in os.listdir(target_dir):
#     folderpath = os.path.join(target_dir, foldername)
#     for fn in os.listdir(folderpath):
#         data=np.loadtxt(os.path.join(folderpath,fn))
#         r=len(data)
#         new_folder=f"{folderpath}rows{str(r)}"
#         os.rename(folderpath, new_folder)
#         break


# todo 得到target_all数据
# source_dir = r"F:\桌面\学习资料\学姐实验\BrainGNN\BrainGNN_Pytorch-main\data_aal\ABIDE_pcp\cpac\filt_noglobal"
# target_dir = r"F:\桌面\学习资料\学姐实验\BrainGNN\BrainGNN_Pytorch-main\data_aal\target_all"
# # 遍历源目录下的所有子目录
# for dirpath, dirnames, filenames in os.walk(source_dir):
#     # 检查当前目录是否包含1D文件
#     if any(filename.endswith('.1D') for filename in filenames):
#         # 获取1D文件的文件名（不包含扩展名）
#         filename_without_extension = filenames[2].split('.')[0]
#         site=filenames[2].split("_")[0];
#         # 构建目标文件夹路径和源文件路径
#         target_folder = os.path.join(target_dir, site)
#         source_file = dirpath
#         # 如果目标文件夹不存在，则创建它
#         if not os.path.exists(target_folder):
#             os.makedirs(target_folder)
#
#         target_folder=os.path.join(target_folder,filenames[0].split("_")[0])
#         os.makedirs(target_folder)
#         for filename in filenames:
#             shutil.copy(os.path.join(source_file,filename), target_folder)

















