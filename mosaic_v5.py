#!/usr/local/bin/python3
#  --*-- coding:utf8 --*--
# Original Author: https://www.cnblogs.com/blamwq/p/11706844.html
# GPU Code by Deepseek & MagicBear

import getopt
import json
import sys
import os

import logging
from PIL import Image
from multiprocessing import Process, Queue, cpu_count, Manager, Pool, Value
from collections import defaultdict
import pickle
import glob
import time
import numpy as np
from functools import partial
import signal
import random

Image.MAX_IMAGE_PIXELS = None

from ipc_shm import ipc_shm

USAGE_GPU = True
if USAGE_GPU:
    try:
        import torch
        import torch.nn.functional as F
        import torch.distributed as dist
        import torch.nn as nn
        from torch.nn.parallel import DistributedDataParallel as DDP
        import torch.multiprocessing as mp
    except ImportError:
        print("Error: module 'torch' is not available, GPU is disabled.")
        USAGE_GPU = False

    try:
        from setproctitle import setproctitle
    except ImportError:
        setproctitle = None
        print("Warning: module 'setproctitle' is not available")

TILE_SIZE = 24  # 素材图片大小
TILE_MATCH_RES = 20  # 配置指数，值越大匹配度越高，执行时间越长
MAX_IMAGE_CACHE_MEMORY = 1000000
TerminatedFlag = Value('b', False)
SaveStateFlag = Value('b', True)

if 'MASTER_ADDR' not in os.environ:
    os.environ['MASTER_ADDR'] = 'localhost'

if 'MASTER_PORT' not in os.environ:
    os.environ['MASTER_PORT'] = '12355'

class ConfigObject:
    def __repr__(self):
        return json.dumps({
            "ENLARGEMENT": self.ENLARGEMENT,
            "MAX_USAGE": self.MAX_USAGE,
            "MAX_USAGE_WAY": self.MAX_USAGE_WAY,
            "USAGE_FACTOR": self.USAGE_FACTOR,
            "RESET_USAGE": self.RESET_USAGE,
            "TOP_N_RAND": self.TOP_N_RAND,
            "SELECTION_TEMPERATURE": self.SELECTION_TEMPERATURE,
            "GPU_SPLIT_ZONE": self.GPU_SPLIT_ZONE,
            "ZONE_ORDER": self.ZONE_ORDER,
            "ZONE_CENTER": self.ZONE_CENTER
        })

    ENLARGEMENT = 4  # 生成的图片是原始图片的多少倍
    MAX_USAGE = 5  # 单图最大使用次数
    MAX_USAGE_WAY = 0   # 单图最大使月次数计算方式
    USAGE_FACTOR = 1e6  # 单进程最大使用次数距离附加值
    RESET_USAGE = 0  # 每多少块区域重置使用次数
    TOP_N_RAND = 0   # 最接近素材随机数
    GPU_SPLIT_ZONE = 1  # GPU分割块数
    SELECTION_TEMPERATURE = 50000  # 控制选择随机性（越小越倾向于最佳）
    ZONE_ORDER = 0  # 乱序图像处理区域
    ZONE_CENTER = [50, 50]
    NCCL_PATH = 'env://'
    USED_UNIQUE_IMAGE = None
    RESET_CACHE = False
    LOAD_CONTINUOUS = False
    OUTPUT_BASE = None
    TILES_COUNT = Value('i', 0)

global_config = ConfigObject()

TILE_BLOCK_SIZE = int(TILE_SIZE / max(min(TILE_MATCH_RES, TILE_SIZE), 1))
WORKER_COUNT = max(cpu_count() - 4, 1)
if USAGE_GPU:
    WORKER_COUNT = 1
EOQ_VALUE = None
WARN_INFO = """ *缺少有效参数*
    参数:
        -i [--image]     : 原图片地址
        -t [--tiles_dir] : 素材目录地址
        -o [--outfile]   : 输出文件地址 【可选】
"""


class TileProcessor:
    def __init__(self, tiles_directory):
        self.tiles_directory = self._expand_directories(tiles_directory)

    # @staticmethod
    # def getCache():
        #
        #

    def _expand_directories(self, tiles_dirs):
        """展开包含子目录的路径"""
        expanded = []
        for dir_path in tiles_dirs:
            has_subdir = False
            for entry in os.scandir(dir_path):
                if entry.is_dir():
                    expanded.append(entry.path)
                    has_subdir = True
            if not has_subdir:
                expanded.append(dir_path)
        return sorted(expanded)

    @staticmethod
    def _process_single_dir(tiles_dir, tile_size, tile_block_size):
        """多进程 worker 函数"""
        cache_file = f"{tiles_dir}_{tile_size}_{tile_block_size}.pickle"

        # 无缓存则处理文件
        large_tiles = []
        small_tiles = []
        filenames = []

        for root, _, files in os.walk(tiles_dir):
            for filename in files:
                filepath = os.path.join(root, filename)
                try:
                    img = Image.open(filepath)
                    w, h = img.size
                    min_dim = min(w, h)
                    img = img.crop((
                        (w - min_dim) // 2,
                        (h - min_dim) // 2,
                        (w + min_dim) // 2,
                        (h + min_dim) // 2
                    ))
                    large = img.resize((tile_size, tile_size), Image.Resampling.LANCZOS).convert('RGB')
                    small = img.resize(
                        (tile_size // tile_block_size, tile_size // tile_block_size),
                        Image.Resampling.LANCZOS
                    ).convert('RGB')
                    filenames.append(os.path.basename(filepath).encode("utf8"))
                    large_tiles.append(large)
                    small_tiles.append(small)
                except Exception as e:
                    logging.warning(f"跳过 {filepath}: {str(e)}")

        large_tiles = list(map(lambda tile: list(tile.getdata()), large_tiles))
        small_tiles = list(map(lambda tile: list(tile.getdata()), small_tiles))

        # 保存缓存
        with open(cache_file, "wb") as f:
            pickle.dump((large_tiles, small_tiles, filenames), f)

        logging.info('从 \'%s\' 获取素材 %d 个' % (tiles_dir, len(large_tiles)))
        return (large_tiles, small_tiles, filenames)

    def get_tiles(self):
        large_tiles_shm = None
        small_tiles_shm = None
        filenames_shm = None
        shm_key = 0
        for tiles_path in self.tiles_directory:
            shm_key ^= ipc_shm.ftok(tiles_path, TILE_SIZE ^ TILE_BLOCK_SIZE)

        loadFromSharedMemory = True
        shm_size = MAX_IMAGE_CACHE_MEMORY * (TILE_SIZE * TILE_SIZE * 3 * 2 + 256)
        try:
            shm = ipc_shm(shm_key, shm_size, create=False)
        except:
            loadFromSharedMemory = False
            try:
                shm = ipc_shm(shm_key, shm_size, create=True)
            except Exception as e:
                shm = None
                logging.error("Error: Initalize share memory failed, %s" % e)

        if shm is not None:
            data_len = shm.nd_array((1), dtype=np.int32)
            if loadFromSharedMemory and data_len[0] > 0 and not global_config.RESET_CACHE:
                logging.info("从共享内存加载 %d " % (data_len[0]))
                large_tiles_shm = shm.nd_array((data_len[0], TILE_SIZE * TILE_SIZE, 3), offset=4, dtype=np.uint8)
                small_tiles_shm = shm.nd_array((data_len[0], (TILE_SIZE // TILE_BLOCK_SIZE) * (TILE_SIZE // TILE_BLOCK_SIZE), 3),
                                               offset=int(MAX_IMAGE_CACHE_MEMORY * TILE_SIZE * TILE_SIZE * 3 + 4), dtype=np.uint8)
                filenames_shm = shm.nd_array((data_len[0]), offset=int(MAX_IMAGE_CACHE_MEMORY * TILE_SIZE * TILE_SIZE * 3 * 2 + 4), dtype='|S256')
                return (large_tiles_shm, small_tiles_shm, filenames_shm)
            else:
                global_config.RESET_CACHE = False
                logging.info("加载至共享内存" % (data_len[0]))
                data_len[0] = 0

        all_large = []
        all_small = []
        all_filenames = []

        pool_size = min(cpu_count() - 4, len(self.tiles_directory))

        no_cache_dir = []
        for tiles_dir in self.tiles_directory:
            cache_file = f"{tiles_dir}_{TILE_SIZE}_{TILE_BLOCK_SIZE}.pickle"
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, "rb") as f:
                        (large_tiles, small_tiles, filenames) = pickle.load(f)

                        if len(large_tiles) != len(filenames):
                            raise TypeError("长度不匹配")

                        all_large.extend(large_tiles)
                        all_small.extend(small_tiles)
                        all_filenames.extend(filenames)

                        if shm is not None:
                            data_len[0] += len(large_tiles)

                        sys.stdout.write('从 \'%s\' 获取素材 %d 个, 共加载 %d 个\r' % (tiles_dir, len(large_tiles), len(all_large)))
                        continue
                except Exception as e:
                    logging.warning(f"缓存 {cache_file} 损坏，重新生成: {str(e)}")

            no_cache_dir.append(tiles_dir)

        if len(no_cache_dir) > 0:
            with Pool(processes=pool_size) as pool:
                # 每个目录一个任务
                worker = partial(
                    self._process_single_dir,
                    tile_size=TILE_SIZE,
                    tile_block_size=TILE_BLOCK_SIZE
                )
                results = pool.map(worker, no_cache_dir)

            # 合并结果
            for large_tiles, small_tiles, filenames in results:
                if shm is not None:
                    data_len[0] += len(large_tiles)

                all_large.extend(large_tiles)
                all_small.extend(small_tiles)
                all_filenames.extend(filenames)

        logging.info(f"共加载 {len(all_large)} 个有效素材, 转换至numpy格式")
        large_arr = np.array(all_large, dtype=np.uint8)
        small_arr = np.array(all_small, dtype=np.uint8)
        filenames_arr = np.array(all_filenames, dtype="|S256")

        if shm is not None:
            large_tiles_shm = shm.nd_array((data_len[0], TILE_SIZE * TILE_SIZE, 3), offset=4, dtype=np.uint8)
            small_tiles_shm = shm.nd_array(
                (data_len[0], (TILE_SIZE // TILE_BLOCK_SIZE) * (TILE_SIZE // TILE_BLOCK_SIZE), 3),
                offset=int(MAX_IMAGE_CACHE_MEMORY * TILE_SIZE * TILE_SIZE * 3 + 4), dtype=np.uint8)
            filenames_shm = shm.nd_array((data_len[0]),
                                         offset=int(MAX_IMAGE_CACHE_MEMORY * TILE_SIZE * TILE_SIZE * 3 * 2 + 4),
                                         dtype='|S256')
            logging.info(f"正在转换 {len(all_large)} 个素材至共享内存")
            large_tiles_shm[:, :, :] = large_arr
            small_tiles_shm[:, :, :] = small_arr
            filenames_shm[:] = filenames_arr

            return (large_tiles_shm, small_tiles_shm, filenames_shm)

        logging.info(f"完成 {len(all_large)} 个有效素材加载")
        return (large_arr, small_arr, filenames_arr)


class TargetImage:
    def __init__(self, image_path, config):
        self.image_path = image_path
        self.config = config

    def get_data(self):
        logging.info('处理主图片...')
        img = Image.open(self.image_path)
        w = img.size[0] * self.config.ENLARGEMENT
        h = img.size[1] * self.config.ENLARGEMENT
        logging.info('新图尺寸：%d x %d  马赛克总数：%d x %d = %d' % (w, h, w / TILE_SIZE, h / TILE_SIZE,
                                                                    (img.size[0] * self.config.ENLARGEMENT// TILE_SIZE) *
                                                                    (img.size[1] * self.config.ENLARGEMENT // TILE_SIZE)))
        large_img = img.resize((w, h), Image.Resampling.LANCZOS)
        w_diff = (w % TILE_SIZE) / 2
        h_diff = (h % TILE_SIZE) / 2

        # if necesary, crop the image slightly so we use a whole number of tiles horizontally and vertically
        if w_diff or h_diff:
            large_img = large_img.crop((w_diff, h_diff, w - w_diff, h - h_diff))

        small_img = large_img.resize((int(w / TILE_BLOCK_SIZE), int(h / TILE_BLOCK_SIZE)), Image.Resampling.LANCZOS)

        image_data = (large_img.convert('RGB'), small_img.convert('RGB'))

        logging.info('主图片处理完成.')

        return image_data


class TileFitter:
    def __init__(self, tiles_data, config):
        self.tiles_data = tiles_data
        self.config = config
        self.usage_counter = defaultdict(int)  # 进程独立计数器

    def __get_tile_diff(self, t1, t2, bail_out_value):
        diff = 0
        for i in range(len(t1)):
            # diff += (abs(t1[i][0] - t2[i][0]) + abs(t1[i][1] - t2[i][1]) + abs(t1[i][2] - t2[i][2]))
            diff += ((t1[i][0] - t2[i][0]) ** 2 + (t1[i][1] - t2[i][1]) ** 2 + (t1[i][2] - t2[i][2]) ** 2)
            if diff > bail_out_value:
                # we know already that this isnt going to be the best fit, so no point continuing with this tile
                return diff
        return diff

    def get_best_fit_tile(self, img_data):
        min_diff = sys.maxsize
        best_index = None

        # 第一轮：寻找未达使用上限的最佳匹配
        for index, tile_data in enumerate(self.tiles_data):
            if self.usage_counter[index] < self.config.MAX_USAGE:
                diff = self.__get_tile_diff(img_data, tile_data, min_diff)
                if diff < min_diff:
                    min_diff = diff
                    best_index = index

        # 第二轮：所有候选都超限时选择当前进程中使用最少的
        if best_index is None:
            sorted_indices = sorted(self.usage_counter.items(), key=lambda x: x[1])
            best_index = sorted_indices[0][0] if sorted_indices else 0

        self.usage_counter[best_index] += 1
        return best_index


def fit_tiles(work_queue, result_queue, tiles_data, config):
    # this function gets run by the worker processes, one on each CPU core
    tile_fitter = TileFitter(tiles_data, config)

    while True:
        try:
            img_data, img_coords = work_queue.get(True)
            if img_data == EOQ_VALUE:
                break
            tile_index = tile_fitter.get_best_fit_tile(img_data)
            result_queue.put((img_coords, tile_index))
        except KeyboardInterrupt:
            pass

    # let the result handler know that this worker has finished everything
    result_queue.put((EOQ_VALUE, EOQ_VALUE))


class DistributedTileMatcher:
    def __init__(self, tiles_tensor, total_tiles, config, rank, world_size, usage_counter=None):
        self.rank = rank
        self.world_size = world_size
        self.zone = rank * config.GPU_SPLIT_ZONE // world_size
        self.device = torch.device(f'cuda:{rank}')
        self.config = config
        self.tiles_len = total_tiles // config.GPU_SPLIT_ZONE
        self.idx_offset = self.tiles_len * self.zone

        self.tiles_tensor = tiles_tensor.to(self.device)

        assert self.tiles_len == len(tiles_tensor), "素材张量维度错误"

        # 使用次数计数器需要全局同步
        self.usage_counter = torch.zeros(total_tiles,
                                         dtype=torch.int32,
                                         device=self.device)
        if usage_counter is not None:
            self.usage_counter = torch.from_numpy(usage_counter.copy()).to(self.device)
        self.local_usage_counter = torch.zeros_like(self.usage_counter)
        self.zone_counter = torch.zeros_like(self.usage_counter)
        self.processed_zone = 0
        self.process_counter = 0

        # 在初始化时添加
        # self.scaler = torch.amp.GradScaler('cuda', enabled=True)

    def find_best_match(self, input_tensor):
        # 计算差异
        diff = torch.sum(
            (input_tensor.unsqueeze(1) - self.tiles_tensor.unsqueeze(0)) ** 2,
            dim=[2, 3, 4]  # 对C,H,W求和
        )

        if self.config.MAX_USAGE_WAY == 0:
            usage_penalty = ((self.local_usage_counter + self.usage_counter) / self.config.MAX_USAGE).floor() * self.config.USAGE_FACTOR
        else:
            usage_penalty = ((self.local_usage_counter + self.usage_counter) >= self.config.MAX_USAGE).float() * self.config.USAGE_FACTOR
        diff += usage_penalty[self.idx_offset:self.idx_offset+len(self.tiles_tensor)].unsqueeze(0)

        # 选择逻辑
        if self.config.TOP_N_RAND > 0:
            return self._select_with_random(diff)
        else:
            return self._select_direct(diff)

    def _select_direct(self, diff):
        best_idx = torch.argmin(diff, dim=1)
        best_idx = self.idx_offset + best_idx
        self._update_usage(best_idx)
        return best_idx

    def _select_with_random(self, diff):
        k = min(self.config.TOP_N_RAND, diff.shape[1])
        top_values, top_indices = torch.topk(-diff, k, dim=1)

        stable_values = top_values - torch.max(top_values, dim=1, keepdim=True).values
        weights = F.softmax(stable_values / self.config.SELECTION_TEMPERATURE, dim=1)

        selected_idx = torch.multinomial(weights.squeeze(1), 1).squeeze(1)
        best_idx = top_indices[torch.arange(len(top_indices)), selected_idx]

        best_idx = self.idx_offset + best_idx
        self._update_usage(best_idx)
        return best_idx

    def _update_usage(self, indices):
        self.local_usage_counter[indices] += 1
        self.process_counter += 1

        if self.process_counter >= 16:
            self.sync()
            self.process_counter = 0

        # 区域重置逻辑
        if self.config.RESET_USAGE > 0:
            self.processed_zone += 1
            self.zone_counter[indices] += 1

            if self.processed_zone >= self.config.RESET_USAGE:
                self.processed_zone = 0
                self.zone_counter = torch.zeros_like(self.usage_counter)

    def sync(self):
        dist.all_reduce(self.local_usage_counter, op=dist.ReduceOp.SUM)
        self.usage_counter += self.local_usage_counter
        self.local_usage_counter.zero_()

def gpu_fit_tiles(work_queue, result_queue, tiles_path, config, rank=None, world_size=None):
    if setproctitle is not None:
        setproctitle("python3 %s - GPU Worker %d" % (sys.argv[0], rank))

    # 初始化分布式环境
    device = torch.device(f'cuda:{rank}')
    print("初始化分布式环境数据 cuda:%d" % rank)
    if torch.cuda.get_device_name(0).split(" ")[0] == "AMD":
        dist.init_process_group("gloo", rank=rank, world_size=world_size)
    else:
        dist.init_process_group("nccl", init_method=config.NCCL_PATH, rank=rank, world_size=world_size)

    print("GPU进程：开始加载 Tiles ")
    large_tiles, tiles_data, _ = TileProcessor(tiles_path).get_tiles()
    print("CPU: 转换Tile为Torch格式")
    proc_tiles = len(large_tiles) // config.GPU_SPLIT_ZONE
    processed_tiles = []
    rank_zone_id = rank * config.GPU_SPLIT_ZONE // world_size
    for tiles_count in range(len(tiles_data)):
        # 确保为RGB格式
        arr = np.array(tiles_data[tiles_count]).reshape((TILE_SIZE, TILE_SIZE, 3))  # 现在形状为 [H, W, 3]
        zone_id = tiles_count // proc_tiles
        if rank_zone_id == zone_id:
            tensor = torch.from_numpy(arr).float()
            processed_tiles.append(tensor)
            if tiles_count % 1000 == 0:
                print("正在加载至CPU %d/%d" % (tiles_count, proc_tiles), end="\r")

    while len(processed_tiles) < proc_tiles:
        print("Fill empty slot to %d" % rank)
        processed_tiles.append(np.zeros_like(tiles_data[0]).reshape((TILE_SIZE, TILE_SIZE, 3)))

    # 准备数据
    tiles_tensor = torch.stack(processed_tiles).to(device)
    # 调整维度顺序 [N, C, H, W]
    tiles_tensor = tiles_tensor.permute(0, 3, 1, 2)  # 正确维度变换

    usage_stats = None
    if config.LOAD_CONTINUOUS and os.path.exists(config.OUTPUT_BASE+".pickle"):
        logging.info("正在加载断点数据")
        with open(config.OUTPUT_BASE+".pickle", "rb") as f:
            usage_stats, box_list, _ = pickle.load(f)
    matcher = DistributedTileMatcher(tiles_tensor, len(large_tiles), config, rank, world_size, usage_stats)
    print("初始化分布式环境数据 cuda:%d - 完成 加载项 %d" % (rank, len(processed_tiles)))

    zone_processed = 0

    while True:
        try:
            img_data, img_coords = work_queue.get(True)
            if img_data == EOQ_VALUE:
                matcher.sync()
                result_queue.put(('usage', matcher.usage_counter.cpu().numpy()))
                break

            # 转换输入数据到Tensor [3, H, W]
            input_tensor = torch.tensor(img_data, device=device).float()
            input_tensor = input_tensor.view(TILE_SIZE, TILE_SIZE, 3).permute(2, 0, 1).unsqueeze(0)

            input_tensor = input_tensor.to(matcher.device)
            # 批量处理
            batch_indices = matcher.find_best_match(input_tensor)
            zone_processed += 1
            if zone_processed % 100 == 0:
                config.USED_UNIQUE_IMAGE.value = torch.count_nonzero(matcher.usage_counter).item()

            result_queue.put((img_coords, batch_indices.item()))
        except KeyboardInterrupt:
            pass

    # 通知结果处理器
    result_queue.put((EOQ_VALUE, EOQ_VALUE))
    if matcher is not None:
        # 清理资源
        dist.destroy_process_group()

class ProgressCounter:
    def __init__(self, total):
        self.total = total
        self.counter = 0
        self.last_update_eta = time.time()
        self.last_write_log = 0
        self.last_counter = 0
        self.ps_counter = 0

    def update(self, used_unique_image, total_tiles = 0):
        self.counter += 1
        if self.counter % 50 == 0:
            if time.time() - self.last_update_eta >= 5:
                self.ps_counter = (self.counter - self.last_counter) / (time.time() - self.last_update_eta)
                self.last_counter = self.counter
                self.last_update_eta = time.time()
            eta_time = ((self.total - self.counter) / self.ps_counter) if self.ps_counter > 0 else 99 * 3600 + 99 * 60 - 1
            wlog = "进度: %.2f%% [%d / %d] (%d / %d - %.1f %%) ETA: %02d:%02d:%02d  Speed: %d i/s" % (100 * self.counter / self.total,
                                                self.counter, self.total,
                                                used_unique_image, total_tiles,
                                                used_unique_image / total_tiles * 100 if total_tiles > 0 else 0,
                                                                                      eta_time // 3600, eta_time // 60 % 60,
                                                                                      eta_time % 60, self.ps_counter)
            if time.time() - self.last_write_log >= 60:
                self.last_write_log = time.time()
                logging.info(wlog)
            else:
                sys.stdout.write(wlog + "      \r")



    sys.stdout.flush()


class MosaicImage:
    def __init__(self, original_img, outfile):
        self.image = Image.new(original_img.mode, original_img.size)
        if global_config.LOAD_CONTINUOUS:
            self.image = Image.open(outfile)
        self.x_tile_count = int(original_img.size[0] / TILE_SIZE)
        self.y_tile_count = int(original_img.size[1] / TILE_SIZE)
        self.total_tiles = self.x_tile_count * self.y_tile_count
        self.outfile = outfile

    def add_tile(self, tile_data, coords):
        img = Image.fromarray(tile_data.reshape((TILE_SIZE, TILE_SIZE, 3))).convert('RGB')
        self.image.paste(img, coords)

    def save(self, path):
        self.image.save(path)

    def save_thumbs(self, path):
        thumb_img = self.image.copy()
        ratio = self.x_tile_count / self.y_tile_count

        thumb_img.thumbnail((1280, int(1280 // ratio)))
        thumb_img.save(path)


def build_mosaic(result_queue, original_img_large, outfile, tiles_path, config):
    if setproctitle is not None:
        setproctitle("python3 %s - Mosaic Build Worker" % (sys.argv[0]))

    mosaic = MosaicImage(original_img_large, outfile)

    tiles_large, tiles_small, filename_map = TileProcessor(tiles_path).get_tiles()
    global_config.TILES_COUNT.value = len(tiles_large)
    total_tiles = len(tiles_large)

    active_workers = WORKER_COUNT

    img_coods_index = {}

    if config.LOAD_CONTINUOUS and os.path.exists(config.OUTPUT_BASE + ".pickle"):
        logging.info("正在加载断点数据")
        with open(config.OUTPUT_BASE + ".pickle", "rb") as f:
            _, _, img_coods_index = pickle.load(f)

    run_timer = time.time()
    mosaic_block = 0

    file_path = os.path.dirname(mosaic.outfile)
    file_base = os.path.basename(mosaic.outfile)
    if os.path.splitext(file_base)[1]:
        file_base = os.path.splitext(file_base)[0]

    with open(os.path.join(file_path, file_base + ".json"), 'w') as f:
        f.write(str(config))

    usage_stats = np.zeros((len(tiles_large),), dtype=np.int32)
    box_list = None
    while True:
        try:
            data = result_queue.get()

            if data == (EOQ_VALUE, EOQ_VALUE):
                active_workers -= 1
                if not active_workers:
                    if box_list is None and SaveStateFlag.value:
                        continue
                    else:
                        break
            elif isinstance(data, tuple) and data[0] == 'usage':
                # 累加各进程的使用统计
                usage_stats = data[1]
            elif isinstance(data, tuple) and data[0] == 'boxlist':
                logging.info("已获取剩余未完成box列表: %d" % (len(data[1])))
                box_list = data[1]
                if not active_workers:
                    break
            else:
                mosaic_block += 1
                img_coords, best_fit_tile_index = data
                tile_data = tiles_large[best_fit_tile_index]
                mosaic.add_tile(tile_data, img_coords)
                if best_fit_tile_index not in img_coods_index:
                    img_coods_index[best_fit_tile_index] = []
                img_coods_index[best_fit_tile_index].append(img_coords[0:2])

                if mosaic_block % 100 == 0 and time.time() - run_timer >= 15:
                    mosaic.save_thumbs(file_base+"_thumb.jpg")
                    run_timer = time.time()

        except KeyboardInterrupt:
            pass

    logging.info('正在导出图片至: %s' % mosaic.outfile)
    mosaic.save_thumbs(file_base + "_thumb.jpg")
    mosaic.save(mosaic.outfile)

    if SaveStateFlag is not None and SaveStateFlag.value and len(box_list) > 0:
        logging.info('保存过程进度文件至: %s.pickle' % file_base)
        with open(file_base+".pickle", 'wb') as f:
            pickle.dump((usage_stats, box_list, img_coods_index), f)
    elif os.path.exists(file_base+".pickle"):
        os.unlink(file_base+".pickle")

    logging.info('============ 生成成功 ============')
    # 输出使用统计
    sorted_indices = np.argsort(-usage_stats)
    usage_count = np.count_nonzero(usage_stats)
    logging.info("使用素材总量：%d / %d (%.1f%%)" % (usage_count, total_tiles, usage_count / total_tiles * 100))
    logging.info("使用素材次数：%d" % (np.sum(usage_stats)))
    print("\n使用次数最高的前20个素材：")
    for i, idx in enumerate(sorted_indices[:20], 1):
        print(f"{i:2d}. 素材{idx} - 使用次数: {usage_stats[idx]}")

    with open(os.path.join(file_path, file_base+".index"), 'w') as f:
        fn_map = {}
        for ind, coods in img_coods_index.items():
            fn_map[filename_map[ind].decode("utf-8")] = coods
        json.dump(fn_map, f)


def signal_handler(sig, frame):
    if sig == signal.SIGINT:
        TerminatedFlag.value = True

from collections import deque

def generate_full_coverage_order(cols, rows, zone_center=None):
    """生成确保全覆盖的辐射状坐标序列 (BFS算法)"""
    if zone_center is None:
        zone_center = [50, 50]
    visited = set()
    queue = deque()

    # 起始中心点
    # start_col = cols // 2
    # start_row = rows // 2
    start_col = int(cols * zone_center[0] / 100)
    start_row = int(rows * zone_center[1] / 100)
    queue.append((start_col, start_row))
    visited.add((start_col, start_row))

    # 8个扩散方向
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                  (-1, -1), (-1, 1), (1, -1), (1, 1)]

    while queue:
        col, row = queue.popleft()
        yield (col, row)

        # 扫描周围8个相邻点
        for dx, dy in directions:
            new_col = col + dx
            new_row = row + dy
            if (0 <= new_col < cols and 0 <= new_row < rows
                    and (new_col, new_row) not in visited):
                visited.add((new_col, new_row))
                queue.append((new_col, new_row))

def validate_coverage(cols, rows, generated_coords):
    """验证是否覆盖所有坐标"""
    all_coords = set((c, r) for c in range(cols) for r in range(rows))
    generated_set = set(generated_coords)
    missing = all_coords - generated_set
    if missing:
        print(f"警告: 缺失 {len(missing)} 个坐标，正在自动修复...")
        return list(generated_coords) + list(missing)
    return generated_coords


def compose(original_img, tiles_path, outfile):
    original_img_large, original_img_small = original_img

    logging.info('预加载小头像')
    # Build Share Memory cache
    tiles_data = TileProcessor(tiles_path).get_tiles()
    mosaic = MosaicImage(original_img_large, outfile)

    work_queue = Queue(WORKER_COUNT)
    result_queue = Queue()

    logging.info("生成配置:%s" % str(global_config))
    logging.info('生成图片中,按下 Ctrl-C 中断...')
    signal.signal(signal.SIGINT, signal_handler)
    box_list = []
    progress = None
    continue_boxlist_cnt = 0

    try:
        # start the worker processes that will perform the tile fitting
        if USAGE_GPU:
            world_size = torch.cuda.device_count()
            ctx = mp.get_context('spawn')
            exec_proc = ctx.Process
            work_queue = ctx.Queue(WORKER_COUNT)
            result_queue = ctx.Queue()
            global_config.USED_UNIQUE_IMAGE = ctx.Value('i', 0)
            processes = []
            for rank in range(world_size):
                p = exec_proc(
                    target=gpu_fit_tiles,
                    args=(work_queue, result_queue, tiles_path, global_config, rank if rank < world_size else world_size - 1, world_size)
                )
                p.start()
                processes.append(p)
        else:
            tiles_large, tiles_small = tiles_data
            for n in range(WORKER_COUNT):
                Process(target=fit_tiles, args=(work_queue, result_queue, tiles_small, global_config)).start()

        # start the worker processes that will build the mosaic image
        Process(target=build_mosaic,
                args=(result_queue, original_img_large, outfile, tiles_path, global_config)).start()

        progress = ProgressCounter(mosaic.x_tile_count * mosaic.y_tile_count)

        x_order = [x for x in range(mosaic.x_tile_count)]
        if global_config.ZONE_ORDER == 3 or global_config.ZONE_ORDER == 4:
            all_coods = list(generate_full_coverage_order(mosaic.x_tile_count, mosaic.y_tile_count, global_config.ZONE_CENTER))
            for idx, (x, y) in enumerate(all_coods):
                large_box = (x * TILE_SIZE, y * TILE_SIZE, (x + 1) * TILE_SIZE, (y + 1) * TILE_SIZE)
                small_box = (
                    x * TILE_SIZE / TILE_BLOCK_SIZE, y * TILE_SIZE / TILE_BLOCK_SIZE,
                    (x + 1) * TILE_SIZE / TILE_BLOCK_SIZE,
                    (y + 1) * TILE_SIZE / TILE_BLOCK_SIZE)
                box_list.append((large_box, small_box))

            logging.info("BFS生成区块：%d" % len(box_list))
            if global_config.ZONE_ORDER == 4:
                box_list = list(reversed(box_list))
        else:
            if global_config.ZONE_ORDER == 2:
                x_order = [mosaic.x_tile_count // 2 + (-1) ** x * ((x + 1) // 2) for x in range(mosaic.x_tile_count)]

            for x in x_order:
                for y in range(mosaic.y_tile_count):
                    large_box = (x * TILE_SIZE, y * TILE_SIZE, (x + 1) * TILE_SIZE, (y + 1) * TILE_SIZE)
                    small_box = (
                        x * TILE_SIZE / TILE_BLOCK_SIZE, y * TILE_SIZE / TILE_BLOCK_SIZE,
                        (x + 1) * TILE_SIZE / TILE_BLOCK_SIZE,
                        (y + 1) * TILE_SIZE / TILE_BLOCK_SIZE)
                    box_list.append((large_box, small_box))

            if global_config.ZONE_ORDER == 1:
                random.shuffle(box_list)

        if global_config.LOAD_CONTINUOUS and os.path.exists(global_config.OUTPUT_BASE+".pickle"):
            with open(global_config.OUTPUT_BASE+".pickle", "rb") as f:
                usage_stats, box_list, img_coods_index = pickle.load(f)
                progress.counter = progress.total - len(box_list)
                continue_boxlist_cnt = progress.counter
                logging.info("加载断点数据并继续执行, 已完成进度：%d" % progress.counter)

        for large_box, small_box in box_list:
            work_queue.put((list(original_img_small.crop(small_box).getdata()), large_box))
            progress.update(global_config.USED_UNIQUE_IMAGE.value if USAGE_GPU else 0,
                            global_config.TILES_COUNT.value)
            if TerminatedFlag.value:
                raise KeyboardInterrupt(None)

        result_queue.put(('boxlist', []))
    except KeyboardInterrupt:
        logging.info('\nHalting, saving partial image please wait...')
        if SaveStateFlag is not None and len(box_list) > 0:
            result_queue.put(('boxlist', box_list[(progress.counter - continue_boxlist_cnt):]))
            SaveStateFlag.value = True

    finally:
        # put these special values onto the queue to let the workers know they can terminate
        if USAGE_GPU:
            for n in range(world_size):
                work_queue.put((EOQ_VALUE, EOQ_VALUE))
        else:
            for n in range(WORKER_COUNT):
                work_queue.put((EOQ_VALUE, EOQ_VALUE))


def mosaic(img_path, tiles_path, outfile):
    image_data = TargetImage(img_path, global_config).get_data()
    compose(image_data, tiles_path, outfile)


if __name__ == '__main__':
    opts, args = getopt.gnu_getopt(sys.argv[1:], 'i:t:o:e:m:M:f:g:r:R:S:N:z:Cc', ['image=', 'tiles_dir=', 'outfile=',
                                                                    "enlargement=", "max_usage=", "max_usage_way=",
                                                                    "usage_factor=", "gpu_split_zone=",
                                                                    "reset=", "top_n_rand=", "select_temp=",
                                                                    "nccl_path=", "zone_order=", "zone_center=", "reset_cache", "continue",
                                                                    ""])

    base_image = None
    tiles_dir = []
    output = None
    for k, v in opts:
        if k in ("-i", "--image"):
            base_image = v
        if k in ("-t", "--tiles_dir"):
            tiles_dir.append(v)
        if k in ("-o", "--outfile"):
            output = v
        if k in ("-e", "--enlargement"):
            global_config.ENLARGEMENT = int(v)
        if k in ("-m", "--max_usage"):
            global_config.MAX_USAGE = int(v)
        if k in ("-m", "--max_usage_way"):
            global_config.MAX_USAGE_WAY = int(v)
        if k in ("-f", "--usage_factor"):
            global_config.USAGE_FACTOR = float(v)
        if k in ("-g", "--gpu_split_zone"):
            global_config.GPU_SPLIT_ZONE = int(v)
        if k in ("-r", "--reset"):
            global_config.RESET_USAGE = int(v)
        if k in ("-R", "--top_n_rand"):
            global_config.TOP_N_RAND = int(v)
        if k in ("-S", "--select_temp"):
            global_config.SELECTION_TEMPERATURE = int(v)
        if k in ("-n", "--nccl_path"):
            global_config.NCCL_PATH = v
        if k in ("-z", "--zone_order"):
            global_config.ZONE_ORDER = int(v)
        if k == "--zone_center":
            v = v.split(",")
            if len(v) != 2:
                raise TypeError("输入中心格式错误，需要为 x,y (%)")
            global_config.ZONE_CENTER = [int(v[0]), int(v[1])]
        if k in ("-C", "--reset_cache"):
            global_config.RESET_CACHE = True
        if k in ("-c", "--continue"):
            global_config.LOAD_CONTINUOUS = True

    for value in (base_image, tiles_dir):
        if value is None:
            logging.error(WARN_INFO)
            sys.exit()

    if output is None:
        output = './mosaic.jpg'

    counting = 0
    out_ = output
    if not global_config.LOAD_CONTINUOUS or not os.path.exists(os.path.splitext(output)[0] + ".pickle"):
        while os.path.exists(out_):
            if global_config.LOAD_CONTINUOUS and os.path.exists(os.path.splitext(output)[0]+"_"+str(counting)+".pickle"):
                out_ = os.path.splitext(output)[0]+"_"+str(counting)+os.path.splitext(output)[1]
                break
            counting += 1
            out_ = os.path.splitext(output)[0]+"_"+str(counting)+os.path.splitext(output)[1]
    output = out_

    global_config.OUTPUT_BASE = os.path.splitext(output)[0]

    if not os.path.exists(global_config.OUTPUT_BASE+".pickle"):
        global_config.LOAD_CONTINUOUS = False

    logging.basicConfig(filename=global_config.OUTPUT_BASE+'.log',
                        format='%(asctime)s  %(filename)s : %(levelname)s  %(message)s',
                        level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler())

    mosaic(base_image, tiles_dir, output)
