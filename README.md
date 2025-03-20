# 粉丝头像马赛克生成系统

通过粉丝头像自动生成艺术马赛克图片的系统，包含图片生成、后处理和互动定位三大模块

## 系统组成

### 🖼 马赛克生成器 (mosaic_v5.py)
**核心功能**：将源图片像素映射到粉丝头像，生成初步马赛克效果
```
python mosaic_v5.py -i 原图.jpg -t 头像库/ -o 马赛克初稿.png \
  -e 8 -m 50 -M smart
```

### 🎨 后处理器 (postprocess.py)
**优化功能**：增强使用的粉丝数量

```
python postprocess.py -i 原图.jpg -n 映射表.csv \
  -t 马赛克初稿.png -o 最终作品.png
```

### 🌐 互动定位网站 (web.py)
**功能特性**：
- 实时展示马赛克大图
- 姓名搜索定位

**启动方式**：
```
python web.py -i 最终作品.png -n 映射表.csv
```

### 典型应用场景

#### 明星应援活动

    # 生成阶段
    python mosaic_v5.py -i idol.jpg -t fans_avatars/ -o draft.png -e 10
    # 后处理阶段
    python postprocess.py -i idol.jpg -n fan_mapping.csv -o final.tiff
    # 部署查询网站
    nohup python web.py -i final.tiff -n fan_mapping.csv > web.log &

### 高级配置项
#### GPU加速配置

# 16卡的GPU分块处理
```
python mosaic_v5.py -g 16 ...
```

# 重置图片共享内存缓存
```
python mosaic_v5.py -C -i ...
```

# 继续上次生成进度
```
python mosaic_v5.py --continue -i ...
```
---

# 感谢

- [晨羽智云](https://www.chenyu.cn/) - 提供GPU运算平台
- [布拉莫维奇](https://www.cnblogs.com/blamwq/p/11706844.html) - 原始马赛克图片实现算法
- [DeepSeek](https://www.deepseek.com/) - 生成了部份代码
