import json
import csv
import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import pickle
import random
from collections import deque
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import to_pil_image
import math
import cv2
import getopt
import sys

Image.MAX_IMAGE_PIXELS = None
DUMP_UNUSED_CUBE = False
orig_image = None

def load_index(index_path):
    """加载并解析马赛克索引文件"""
    with open(index_path, 'r') as f:
        data = json.load(f)
    return data


def save_index(index_data, output_path):
    """保存更新后的索引文件"""
    with open(output_path, 'w') as f:
        json.dump(index_data, f, indent=4)


def get_fade_weight(coord, img_size, fade_ratio=0.2):
    """计算坐标的羽化权重（0~1），越靠近中心权重越低"""
    x, y = coord
    width, height = img_size

    # 计算中心点坐标
    center_x, center_y = width / 2, height / 2

    # 计算到中心的最小相对距离（归一化到0~1）
    dx = abs(x - center_x) / (width * fade_ratio / 2)  # fade_ratio控制衰减范围比例
    rnd_px = random.random()
    if (x > 10000 - rnd_px * 3000 and x < 26400 + rnd_px * 5000) or y > 30000 + rnd_px * 15000:
        return 0.05
    dy = abs(y - center_y) / (height * fade_ratio / 2)
    distance = math.sqrt(dx ** 2 + dy ** 2)

    # 使用S型曲线过渡（Sigmoid衰减）
    weight = 1 / (1 + math.exp(-10 * (distance - 1)))  # 10控制衰减斜率
    return min(max(weight, 0), 1)  # 限制在0~1

def get_image_features(image_path, device):
    """返回：明度均值, 明度标准差, 色相矢量均值, 色相环形标准差, 缩放张量"""
    img = Image.open(image_path).convert('RGB')
    tensor = TF.to_tensor(img).to(device)
    resized_tensor = TF.resize(tensor, (24, 24),
                               interpolation=TF.InterpolationMode.BILINEAR,
                               antialias=True)

    # 计算明度特征
    luminance = (resized_tensor * torch.tensor([0.2126, 0.7152, 0.0722], device=device)[:, None, None]).sum(dim=0)
    lum_mean = luminance.mean()
    lum_std = luminance.std()

    # 计算色相特征（矢量平均）
    rgb = resized_tensor.permute(1, 2, 0)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    maxc, max_idx = torch.max(rgb, dim=2)
    minc = torch.min(rgb, dim=2)[0]
    delta = maxc - minc

    hue = torch.zeros_like(maxc)
    mask = delta != 0

    # 计算各通道色相
    r_mask = (max_idx == 0) & mask
    g_mask = (max_idx == 1) & mask
    b_mask = (max_idx == 2) & mask

    hue[r_mask] = 60 * ((g[r_mask] - b[r_mask]) / delta[r_mask]) % 360
    hue[g_mask] = 60 * ((b[g_mask] - r[g_mask]) / delta[g_mask] + 2) %360
    hue[b_mask] = 60 * ((r[b_mask] - g[b_mask]) / delta[b_mask] + 4) % 360

    # 矢量平均计算色相
    valid_hues = hue[mask]
    if valid_hues.numel() == 0:
        hue_mean = torch.tensor(0.0, device=device)
        hue_std = torch.tensor(0.0, device=device)
    else:
        hue_rad = torch.deg2rad(valid_hues)
        mean_sin = torch.sin(hue_rad).mean()
        mean_cos = torch.cos(hue_rad).mean()
        mean_hue_rad = torch.atan2(mean_sin, mean_cos)
        hue_mean = torch.rad2deg(mean_hue_rad) % 360

        # 环形标准差计算
        R = torch.sqrt(mean_sin ** 2 + mean_cos ** 2)
        hue_std = torch.sqrt(-2 * torch.log(R + 1e-6)) if R < 0.99 else torch.tensor(0.0)
        hue_std = torch.rad2deg(hue_std)

    return lum_mean.cpu(), lum_std.cpu(), hue_mean.cpu(), hue_std.cpu(), to_pil_image(resized_tensor.cpu())

def get_to_replace(index_data, orig_size, fade_ratio=0.2, threshold=1000):
    (orig_width, orig_height) = orig_size

    replace_data = {k: coords for k, coords in index_data.items() if len(coords) > threshold}

    coord_index = {}
    mask_img = Image.open("mask.png")
    assert mask_img.size == orig_size, "Mask size error"
    mask_array = np.array(mask_img)

    for k, coords in tqdm(replace_data.items(), desc="Masking"):
        for coord in coords:
            if np.any(mask_array[coord[1]:(coord[1]+24), coord[0]:(coord[0]+24)] == 1):
                coord_index.setdefault(k, []).append(coord)

    # 获取需要替换的文件及其坐标队列
    file_queues = {
        k: deque(v[1:])
        for k, v in coord_index.items()
        # if len(v) >= threshold
    }

    to_replace_files = list(file_queues.keys())

    # 生成广度优先的替换序列
    to_replace = []
    while any(queue for queue in file_queues.values()):
        # 遍历所有文件，每轮每文件取一个坐标
        for filename in list(file_queues.keys()):  # 转换为list避免字典修改错误
            if file_queues[filename]:
                coord = file_queues[filename].popleft()

                # 计算该坐标的替换概率权重
                weight = get_fade_weight(coord, (orig_width, orig_height), fade_ratio)

                # if random.random() < weight:
                #     # 按权重概率决定是否保留该坐标（权重越高越可能被替换）
                to_replace.append((filename, coord, weight))

    to_replace = sorted(to_replace, key=lambda x: x[2])

    print(f"Found {len(to_replace_files)} files, {len(to_replace)} block to replace")
    return to_replace

def main(base_path, image_dir, nick_csv, output):
    # 配置参数
    index_path = base_path+".index"
    output_index = os.path.splitext(output)[0]+".index"
    threshold = 50  # 需要替换的阈值
    fade_ratio = 0.8
    lum_weight = 0.9  # 明度权重
    hue_weight = 0.3  # 色相权重
    batch_size = 512  # 根据显存调整，RTX 3090建议512-1024
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载索引数据
    print("Loading index data...")
    index_data = load_index(index_path)

    # 加载所有nick并获取未使用的
    with open(nick_csv, 'r') as f:
        all_nicks = {row[1] for row in csv.reader(f)}
    used_nicks = set(index_data.keys())
    unused_nicks = list(all_nicks - used_nicks)
    print(f"Total unused avatars: {len(unused_nicks)}")

    to_replace = get_to_replace(index_data, orig_image.size, fade_ratio=fade_ratio, threshold=threshold)

    # 预加载特征
    if os.path.exists(base_path+"_features.pkl"):
        with open(base_path+"_features.pkl", "rb") as f:
            feature_data = pickle.load(f)
    else:
        feature_data = {"candidates": [], "old_feats": {}}
        # 加载候选特征
        for nick in tqdm(unused_nicks):
            path = os.path.join(image_dir, nick[0:2], nick)
            try:
                features = get_image_features(path, device)
                feature_data["candidates"].append((nick, features))
            except:
                continue

        with open(base_path+"_features.pkl", "wb") as f:
            pickle.dump(feature_data, f)

    feature_data['old_feats'] = {}
    # 加载旧特征
    for filename in tqdm({f[0] for f in to_replace}, desc="Loading old features"):
        path = os.path.join(image_dir, filename[0:2], filename)
        try:
            feature_data["old_feats"][filename] = get_image_features(path, device)
        except:
            continue

    # 准备张量
    old_features = []
    old_coords = []
    candidate_features = []
    valid_nicks = []
    valid_images = []
    for nick, (lum_m, lum_s, hue_m, hue_s, img) in feature_data["candidates"]:
        candidate_features.append(torch.stack([lum_m, hue_m]))
        valid_nicks.append(nick)
        valid_images.append(img)
    candidate_tensor = torch.stack(candidate_features).to(device)
    used_tensor = torch.zeros(len(valid_nicks), dtype=torch.int32, device=device)

    for filename, coord, _ in tqdm(to_replace, desc="Loading old features To GPU"):
        if filename not in feature_data["old_feats"]:
            continue
        (lum_m, lum_s, hue_m, hue_s, img) = feature_data["old_feats"][filename]
        old_features.append(torch.stack([lum_m, hue_m]))
        old_coords.append((filename, coord))

    old_tensor = torch.stack(old_features).to(device)

    if DUMP_UNUSED_CUBE:
        unused_cube_size = math.ceil(math.sqrt(len(unused_nicks)))
        unused_cube = Image.new("RGB", (unused_cube_size * 24, unused_cube_size * 24))
        idx = 0
        for nick in tqdm(unused_nicks):
            idx += 1
            path = os.path.join(image_dir, nick[0:2]+"/"+nick)
            try:
                unused_img = Image.open(path).resize( (24, 24)).convert('RGB')
                y = idx // unused_cube_size * 24
                x = idx % unused_cube_size * 24
                unused_cube.paste(unused_img, (x, y))
            except:
                continue
        unused_cube.save("unused_cube_size.png")

    num_old = old_tensor.size(0)

    # 初始化跟踪矩阵
    assigned_new = torch.full((candidate_tensor.size(0),), -1, dtype=torch.long, device=device)  # [N]
    assigned_old = torch.full((old_tensor.size(0),), -1, dtype=torch.long, device=device)  # [M]
    best_dist = torch.full_like(assigned_old, float('inf'), dtype=torch.float32)  # [M]

    # 转换数据类型
    old_tensor = old_tensor.to(torch.float32)
    new_tensor = candidate_tensor.to(torch.float32)

    # 分块处理旧特征
    for old_start in tqdm(range(0, num_old, batch_size), desc="Processing"):
        old_end = min(old_start + batch_size, num_old)
        old_batch = old_tensor[old_start:old_end]  # [bs, 2]

        # 全矩阵计算差异
        diff_lum = lum_weight * (old_batch[:, 0].view(-1, 1) - new_tensor[:, 0].view(1, -1))  # [bs, N]
        diff_hue = hue_weight * (old_batch[:, 1].view(-1, 1) - new_tensor[:, 1].view(1, -1))  # [bs, N]
        distances = torch.sqrt(diff_lum ** 2 + diff_hue ** 2)  # [bs, N]

        # 屏蔽已分配的新特征
        mask = (assigned_new != -1).view(1, -1)
        distances[mask.expand_as(distances)] = float('inf')

        # 找当前批最佳匹配
        batch_min, batch_indices = torch.min(distances, dim=1)

        # 更新全局最优解
        update_mask = batch_min < best_dist[old_start:old_end]
        global_indices = torch.where(update_mask, batch_indices, assigned_old[old_start:old_end])

        # 原子操作更新状态
        assigned_old[old_start:old_end] = torch.where(update_mask, global_indices, assigned_old[old_start:old_end])
        if torch.any(update_mask):
            assigned_new[global_indices] = torch.arange(old_start, old_end, device=device)[update_mask]

    # 生成最终分配映射
    valid_mask = assigned_old != -1
    allocation = {
        int(old_idx): int(new_idx)
        for old_idx, new_idx in zip(
            torch.where(valid_mask)[0].cpu().numpy(),
            assigned_old[valid_mask].cpu().numpy()
        )
    }

    allocation = dict(zip(allocation.values(),allocation))

    print("Allocation done, replace image %d " % len(allocation))
    # 执行替换
    for new_idx, old_idx in tqdm(allocation.items(), desc="Allocating new images"):
        filename, coord = old_coords[old_idx]
        new_nick = valid_nicks[new_idx]

        # 更新索引
        index_data[filename].remove(coord)
        index_data.setdefault(new_nick, []).append(coord)

        # 更新图像
        orig_image.paste(valid_images[new_idx], coord)

    # 保存结果
    print("Saving results...")
    save_index(index_data, output_index)

    print("Saving Image...")
    thumb_img = orig_image.copy()
    thumb_img.thumbnail((1280, int(1280 // (orig_image.size[0] / orig_image.size[1]))))
    thumb_img.save(os.path.splitext(output_index)[0]+"_thumb.jpg")
    orig_image.save(os.path.splitext(output_index)[0]+".png")

if __name__ == '__main__':
    opts, args = getopt.gnu_getopt(sys.argv[1:], 'i:t:o:n:', ['image=', 'tiles_dir=', 'outfile=', 'nick=', ""])

    tiles_dir = "images/"
    output = "postprocess.png"
    nick_csv = "nick.csv"

    for k, v in opts:
        if k in ("-i", "--image"):
            base_path = os.path.splitext(v)[0]
            orig_image = Image.open(v)
        if k in ("-t", "--tiles_dir"):
            tiles_dir = v
        if k in ("-o", "--outfile"):
            output = v
        if k in ("-n", "--nick"):
            nick_csv = v

    if orig_image is None:
        print("No input Image")
        sys.exit()
    main(base_path, tiles_dir, nick_csv, output)
