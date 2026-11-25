import os
import shutil
import numpy as np
import torch

class FeatureStitcherAvg:
    """
    大图特征拼接器 (支持平均融合 / Averaging)
    原理：Sum / Count
    """
    def __init__(self, large_image_shape, save_dir, temp_dir="./temp_feat_avg_buffer"):
        self.H_orig, self.W_orig = large_image_shape
        self.save_dir = save_dir
        self.temp_dir = temp_dir
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir)
            
        self.strides = {'res2': 4, 'res3': 8, 'res4': 16, 'res5': 32}
        
        self.maps = {}    # 存放特征累加和 (Sum)
        self.counts = {}  # 存放叠加次数 (Count)
        self.shapes = {}
        self.dtypes = {}
        self.initialized = False

    def _init_maps(self, feature_dict):
        print(f"[Stitcher] Initializing Average Storage: {self.H_orig}x{self.W_orig}")
        
        for key, feat in feature_dict.items():
            if key not in self.strides: continue
                
            C = feat.shape[1]
            stride = self.strides[key]
            H_feat = int(np.ceil(self.H_orig / stride))
            W_feat = int(np.ceil(self.W_orig / stride))
            
            self.shapes[key] = (C, H_feat, W_feat)
            self.dtypes[key] = 'float32'
            
            # 1. 特征累加图 (float32)
            feat_path = os.path.join(self.temp_dir, f"{key}_sum.dat")
            self.maps[key] = np.memmap(
                feat_path, dtype='float32', mode='w+', shape=(C, H_feat, W_feat)
            )
            
            # 2. 计数图 (uint8 足够，假设重叠次数不超过 255)
            # 注意：计数图是 2D 的，所有通道共享同一个计数
            count_path = os.path.join(self.temp_dir, f"{key}_count.dat")
            self.counts[key] = np.memmap(
                count_path, dtype='uint8', mode='w+', shape=(H_feat, W_feat)
            )
            
            print(f"  - Level '{key}': Shape={self.shapes[key]}")
            
        self.initialized = True

    def update(self, feature_dict, top, left):
        if not self.initialized:
            self._init_maps(feature_dict)
            
        for key, feat_tensor in feature_dict.items():
            if key not in self.maps: continue
                
            # Tensor -> Numpy
            feat_data = feat_tensor.squeeze(0).cpu().detach().numpy()
            
            stride = self.strides[key]
            start_y, start_x = int(top // stride), int(left // stride)
            c, h_crop, w_crop = feat_data.shape
            
            max_h, max_w = self.shapes[key][1], self.shapes[key][2]
            end_y = min(start_y + h_crop, max_h)
            end_x = min(start_x + w_crop, max_w)
            
            valid_h = end_y - start_y
            valid_w = end_x - start_x
            
            if valid_h <= 0 or valid_w <= 0: continue

            # === 核心修改点 ===
            
            # 1. 特征累加 (In-place addition on disk)
            # 注意：memmap 的 += 操作会自动读写硬盘
            self.maps[key][:, start_y:end_y, start_x:end_x] += feat_data[:, :valid_h, :valid_w]
            
            # 2. 计数累加
            self.counts[key][start_y:end_y, start_x:end_x] += 1

    def save(self, filename_prefix="large_feat"):
        print("[Stitcher] Computing averages and saving...")
        saved_paths = []
        
        for key in self.maps.keys():
            # 刷新硬盘
            self.maps[key].flush()
            self.counts[key].flush()
            
            # 执行除法: Mean = Sum / Count
            # 这一步可能会比较慢，因为要全盘读写
            # 避免除以 0：取 max(count, 1)
            count_map = self.counts[key]
            feature_map = self.maps[key]
            
            # 利用 numpy 的广播机制：[C, H, W] / [H, W]
            # 为了防止内存溢出，我们逐块进行除法（或者相信虚拟内存足够大）
            # 这里直接操作 memmap 会自动处理内存，但速度取决于硬盘
            
            # 技巧：count 为 0 的地方（未覆盖区域），除以 1 保持为 0
            divisor = np.maximum(count_map, 1).astype('float32')
            
            # 执行除法并写回 self.maps (覆盖 Sum 变成 Mean)
            # 这一步会遍历整个大文件
            feature_map[:] /= divisor[None, :, :]
            feature_map.flush()

            # 移动最终文件
            final_dat_path = os.path.join(self.save_dir, f"{filename_prefix}_{key}.dat")
            temp_feat_path = os.path.join(self.temp_dir, f"{key}_sum.dat") # 现在已经是 mean 了
            
            # 删除引用以释放文件锁
            del self.maps[key]
            del self.counts[key]
            
            if os.path.exists(final_dat_path): os.remove(final_dat_path)
            shutil.move(temp_feat_path, final_dat_path)
            
            # 保存元数据
            meta_path = os.path.join(self.save_dir, f"{filename_prefix}_{key}_meta.txt")
            with open(meta_path, 'w') as f:
                shape_str = ",".join(map(str, self.shapes[key]))
                f.write(f"shape={shape_str}\n")
                f.write(f"dtype={self.dtypes[key]}\n")
                f.write(f"stride={self.strides[key]}\n")
                f.write("mode=average\n")

            print(f"  - Saved Average: {final_dat_path}")
            saved_paths.append(final_dat_path)
            
        # 清理
        self.maps = {}
        self.counts = {}
        if os.path.exists(self.temp_dir):
            try: shutil.rmtree(self.temp_dir)
            except: pass
            
        print("[Stitcher] Done.")
        return saved_paths
