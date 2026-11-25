import glob
import os
import pandas as pd
import numpy as np

__all__=["make_files_pair","make_files_pair_with_indices_csv","make_files_pair_with_data_dir","make_pair_with_indices",
         "read_xyz_img_pair_file","read_crop_corner_file","make_pair_with_xyz_indices"]

def make_files_pair_with_data_dir(data_dir:str,csv_dir:str,num_lenth=3):
    """
    为多视数据生成对应关系
    Args:
        data_dir:文件路径
        csv_dir:csv文件的输出路径
        num_lenth:需要匹配的后缀长度
    Returns:
        nothing
    """
    files = glob.glob(os.path.join(data_dir, "*.tif"))
    make_files_pair(files,csv_dir, num_lenth)

def make_files_pair_with_indices_csv(indices_cvs:str,csv_dir:str,num_lenth=3):
    """
    making images pair files with indices file
    Args:
        indices_cvs:
        csv_dir:
        num_lenth:

    Returns:

    """
    indices=pd.read_csv(indices_cvs)
    ids=indices['name'].tolist()
    make_files_pair(ids,csv_dir, num_lenth)

def make_files_pair(files_list,csv_dir:str,num_lenth=3):

    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)
    info_file = open(csv_dir + '/' + 'pair.csv', 'w')

    file_dict={}
    for file in files_list:
        name=os.path.basename(file)
        file=file.split(".")[0]
        num="".join(file.split("_")[-num_lenth:])
        if num in file_dict.keys():
            file_dict[num].append(name)
        else:
            file_dict[num]=[name]

    for key,value in file_dict.items():
        info_file.write(" ".join(value))
        info_file.write("\n")

def make_pair_with_indices(files_list,num_lenth=3):
    """

    Args:
        files_list:
        num_lenth:

    Returns:

    """
    file_dict = {}
    for file in files_list:
        name = os.path.basename(file)
        file = file.split(".")[0]
        num = "".join(file.split("_")[-num_lenth:])
        num = num.split("-")[-1]
        if num in file_dict.keys():
            file_dict[num].append(name)
        else:
            file_dict[num] = [name]

    pairs_meta=[]
    for key, value in file_dict.items():
        pairs_meta.append(value)

    return pairs_meta

def make_pair_with_xyz_indices(
    xyz_files,
    feat_dir,
    xyz_num_length=3,
    feat_ext=".pt"
):
    """
    以物方 xyz index 作为 patch group 锚点，
    在 feature 目录中查找对应的多视角特征

    Args:
        xyz_files: List[str]
            物方 patch 文件名列表（来自 indices.csv）
        feat_dir: str
            像方特征目录
        xyz_num_length: int
            xyz index 使用末尾多少段（如 2 -> 012_034）
        feat_ext: str
            特征文件后缀

    Returns:
        pairs_meta: List[Tuple[str, List[str]]]
            每个元素是一个 tuple：(xyz_file, views)
    """

    # === 1. 预扫描 feature 文件，建立 index -> files 映射 ===
    feat_map = {}
    for fname in os.listdir(feat_dir):
        if not fname.endswith(feat_ext):
            continue

        stem = os.path.splitext(fname)[0]
        # Example STEM: CAM01_190073-6_0720_7278
        parts = stem.split("_")[-xyz_num_length:]
        
        # If the first part contains a '-', split it to extract the real region number '6'
        if '-' in parts[0]:
            parts[0] = parts[0].split('-')[-1]
            
        xyz_key = "_".join(parts)

        feat_map.setdefault(xyz_key, []).append(fname)

    # === 2. 以 xyz 为主索引，构建 patch group ===
    pairs_meta = []

    for xyz_file in xyz_files:
        xyz_name = os.path.splitext(os.path.basename(xyz_file))[0]
        xyz_key = "_".join(xyz_name.split("_")[-xyz_num_length:])

        views = feat_map.get(xyz_key, [])
        if len(views) == 0:
            # ⚠️ 强烈建议打印或统计
            # print(f"[WARN] No features found for xyz patch: {xyz_key}")
            pass

        pairs_meta.append((xyz_file, views))

    return pairs_meta

def read_xyz_img_pair_file(xyz_img_pair_file):
    with open(xyz_img_pair_file, "r") as f:
        scans = f.readlines()
        scans = [line.replace("\t", " ").strip().split(" ") for line in scans]
        scans = np.array(scans)
        scans_dict = dict(zip(scans[:, 1], scans[:, 0]))
    return scans_dict

def read_crop_corner_file(corner_coordination_file):
    """
    读取像方共线条件方程角点文件
    Args:
        corner_coordination_file:

    Returns:

    """
    with open(corner_coordination_file, "r") as f:
        scans = f.readlines()
        scans = scans[1:]
        # scans = [line.replace("\n", " ").strip().split(" ") for line in scans]
        scans=[line.rstrip().strip().split(" ")  for line in scans if line.strip()]
        for i, line in enumerate(scans):
            line = [float(l) for l in line]
            scans[i] = line
        return scans
    
if __name__=='__main__':
    import shutil
    image_dir=r'/path/to/user/NAS/gwx/DeepL/Xuzhou/IMG_MV/train/samples_none/image'
    xyz_dir=r'/path/to/user/NAS/gwx/DeepL/Xuzhou/XYZ/train/samples_none/image'
    files_list=glob.glob(os.path.join(image_dir,'*.tif'))
    files_pair=make_pair_with_indices(files_list)
    
    out_dir=r'/path/to/user/NAS/gwx/DeepL/Xuzhou/IMG_MV/train/3views_image'
    with open(r"/path/to/user/NAS/gwx/DeepL/Xuzhou/IMG_MV/XYZ_xy_name_map.txt", "r") as f:
            scans = f.readlines()
            scans = [line.replace("\t", " ").strip().split(" ") for line in scans]
            scans = np.array(scans)
            scans_dict = dict(zip(scans[:, 1], scans[:, 0]))
    n=0
    for file_pair in files_pair:
        if len(file_pair)==3:
            if not os.path.exists(os.path.join(out_dir,str(n))):
                os.makedirs(os.path.join(out_dir,str(n)))
            for file in file_pair:
                shutil.copy(os.path.join(image_dir,file),os.path.join(out_dir,str(n),file))
                
            image_name = os.path.splitext(file)[0]
            xyz_num = image_name.split("_")[-3]
            num = image_name.split("_")[-2:]
            num = "_".join(num)
            xyz_suffix=scans_dict[xyz_num] + "_" + num + ".tif"
                
            shutil.copy(os.path.join(xyz_dir,xyz_suffix),os.path.join(out_dir,str(n),xyz_suffix))
            n+=1
