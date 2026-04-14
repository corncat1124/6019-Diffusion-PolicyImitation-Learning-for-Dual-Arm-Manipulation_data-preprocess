import h5py
import numpy as np
import torch
from pathlib import Path
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
import io
from tqdm import tqdm
import json

class PreprocessPipeline:
    def __init__(self, data_dir, device="cpu"):
        self.data_dir = Path(data_dir)
        self.files = sorted(list(self.data_dir.glob("episode*.hdf5")))
        if not self.files:
            raise FileNotFoundError(f"在 {self.data_dir} 未找到 episode*.hdf5 文件")
        self.device = device
        weights = ResNet18_Weights.IMAGENET1K_V1
        self.model = resnet18(weights=weights)
        self.model.fc = torch.nn.Identity()
        self.model.eval().to(device)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def extract_episode(self, file):
        with h5py.File(file, "r") as f:
            actions = f["joint_action/vector"][:]
            combined_features = []
            cam_left = f["observation/left_camera/rgb"]
            cam_right = f["observation/right_camera/rgb"]
            for raw_l, raw_r in zip(cam_left, cam_right):
                img_l = Image.open(io.BytesIO(raw_l))
                feat_l = self._get_embedding(img_l)
                img_r = Image.open(io.BytesIO(raw_r))
                feat_r = self._get_embedding(img_r)
                combined = np.concatenate([feat_l, feat_r], axis=1)
                combined_features.append(combined)
            images = np.vstack(combined_features)
        return images, actions

    def _get_embedding(self, pil_img):
        img_t = self.transform(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.model(img_t).cpu().numpy()
        return feat

    def pad_sequence(self, arr, target_len):
        # arr shape: (T, D)
        pad_num = target_len - arr.shape[0]
        if pad_num <= 0:
            return arr[:target_len]
        if arr.shape[0] == 0:
            # 极端case，直接全零
            return np.zeros((target_len, arr.shape[1]), dtype=arr.dtype)
        last = arr[-1][np.newaxis, ...]
        pad = np.repeat(last, pad_num, axis=0)
        return np.concatenate([arr, pad], axis=0)

    def run(self, save_dir="preprocessed"):
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)

        all_images, all_actions = [], []
        fix_length = 75

        print(f" 开始处理.")
        for i, file in enumerate(tqdm(self.files)):
            imgs, acts = self.extract_episode(file)
            imgs = self.pad_sequence(imgs, fix_length)
            acts = self.pad_sequence(acts, fix_length)
            all_images.append(imgs)
            all_actions.append(acts)

        all_images = np.stack(all_images, axis=0)   # (N, 75, 1024)
        all_actions = np.stack(all_actions, axis=0) # (N, 75, 16)

        stats = {
            "action_min": all_actions.min(axis=(0, 1)).tolist(),
            "action_max": all_actions.max(axis=(0, 1)).tolist(),
        }

        np.save(save_path / "liftpot_images.npy", all_images)
        np.save(save_path / "liftpot_actions.npy", all_actions)
        with open(save_path / "stats.json", "w") as f:
            json.dump(stats, f)

        print(f"预处理完成！")
        print(f"图像特征: {all_images.shape} -> {save_path / 'liftpot_images.npy'}")
        print(f"动作向量: {all_actions.shape} -> {save_path / 'liftpot_actions.npy'}")
        print(f"归一化参数已存至 stats.json")

        # =========== 自动检查环节 =============
        print("\n==== 数据检查 ====")
        print("images.shape:", all_images.shape)
        print("actions.shape:", all_actions.shape)
        print("images dtype:", all_images.dtype)
        print("actions dtype:", all_actions.dtype)
        print("images min,max:", all_images.min(), all_images.max())
        print("actions min,max:", all_actions.min(), all_actions.max())

        print("\nimages[0,0,:10]:", all_images[0, 0, :10])   # 第一个episode第一帧 前10个特征
        print("actions[0,0]:", all_actions[0, 0])              # 第一个episode第一帧的动作
        print("actions[0,-1]:", all_actions[0, -1])            # 第一个episode最后一帧的动作

        # 检查是否有全0样本
        for idx in range(all_images.shape[0]):
            if np.all(all_images[idx] == 0):
                print(f" 第 {idx} 条 image embedding 全零异常")
            if np.all(all_actions[idx] == 0):
                print(f" 第 {idx} 条 action 全零异常")

        # 检查stats.json与实际数据一致性
        with open(save_path / "stats.json", "r") as f:
            stats_loaded = json.load(f)
        print("\n【stats.json 动作全局min】", stats_loaded["action_min"])
        print("【stats.json 动作全局max】", stats_loaded["action_max"])
        print("【代码实际min】", all_actions.min(axis=(0, 1)))
        print("【代码实际max】", all_actions.max(axis=(0, 1)))
        print("\n==== 检查完毕 ====\n")

if __name__ == "__main__":
    import os
    current_path = os.getcwd()
    pipeline = PreprocessPipeline(data_dir=current_path, device="cpu") 
    pipeline.run()