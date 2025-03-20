import os
from PIL import Image
from io import BytesIO
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2, 40).__str__()
import cv2
import math
import numpy as np

Image.MAX_IMAGE_PIXELS = None

class ImageNavigator:
    def __init__(self, CONFIG, cache_source=None):
        print("Loading image")
        if cache_source is not None:
            self.source = cache_source
        else:
            # self.source = Image.open(CONFIG['source_image']).convert('RGB')
            self.source = cv2.imread(CONFIG['source_image'])
        
        # self.width, self.height = self.source.size
        self.height, self.width = self.source.shape[0:2]
        print("Loaded")
        self.TILE_SIZE = CONFIG['tile_size']
        self.TILE_CACHE = CONFIG['tile_cache']
        self.zoom_levels = self.calculate_zoom_levels()
        self.max_coods = [[math.ceil(self.width / CONFIG['tile_size'] / zl), math.ceil(self.height / CONFIG['tile_size'] / zl)] for lv, zl in enumerate(self.zoom_levels)]
        self.CONFIG = CONFIG

    def calculate_zoom_levels(self):
        max_dim = max(self.width, self.height)
        return list(reversed([math.pow(2, i) for i in range(math.ceil(math.log(max_dim / self.TILE_SIZE, 2)) + 1)]))

    def get_tile(self, z, x, y):
        width = int(self.TILE_SIZE * self.zoom_levels[z])
        height = int(self.TILE_SIZE * self.zoom_levels[z])
        # 检查缓存
        cache_key = f"{z}_{x}_{y}.jpg"
        cache_path = os.path.join(self.TILE_CACHE, f"{z}/{cache_key}")

        if os.path.exists(cache_path):
            return cache_path

        # 获取实际切片区域
        box = (
            int(max(0, x * width)),
            int(max(0, y * height)),
            int(min(self.width, (x + 1) * width)),
            int(min(self.height, (y + 1) * height))
        )

        orig_w = box[2] - box[0]
        orig_h = box[3] - box[1]

        # 应用比例缩放
        new_w = int(orig_w / self.zoom_levels[z])
        new_h = int(orig_h / self.zoom_levels[z])

        tile = cv2.resize(self.source[box[1]:box[3], box[0]:box[2]], (new_w, new_h), interpolation=cv2.INTER_AREA)
        # tile = self.source.crop(box)
        # tile.thumbnail((new_w, new_h))
        # tile = np.array(tile.convert('RGB'))[:, :, ::-1]
        # tile = cv2.resize(tile, (new_w, new_h), interpolation=cv2.INTER_AREA)

        img = np.ones((self.TILE_SIZE, self.TILE_SIZE, 3), np.uint8)
        img *= 255
        img[:min(tile.shape[0], self.TILE_SIZE), :min(tile.shape[1], self.TILE_SIZE)] = tile
        
        is_success, img_data = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        img_io = BytesIO(img_data)

        # 缓存小尺寸
        if z <= 6:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, "wb") as f:
                f.write(img_data)

        return img_io

    def get_preview(self):
        preview_size = (int(self.width * self.CONFIG['preview_ratio']),
                        int(self.height * self.CONFIG['preview_ratio']))
        preview_path = os.path.join(self.CONFIG['tile_cache'], "preview.jpg")
        if not os.path.exists(preview_path):
            self.source.resize(preview_size).save(preview_path)
        return preview_path
