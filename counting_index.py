import numpy as np
import json
import os
import csv
import sys
import re
import sys
import getopt

if __name__ == '__main__':
    opts, args = getopt.gnu_getopt(sys.argv[1:], 'i:n:', ['index=', 'nick=', ""])

    # 配置项
    CONFIG = {
        "csv_path": "nick.csv",
        "index_path": "gpusuma_12x_z3R40_15k_posthue.index"
    }

    for k, v in opts:
        if k in ("-i", "--index"):
            CONFIG['index_path'] = v
        if k in ("-n", "--nick"):
            CONFIG['csv_path'] = v

    if orig_image is None:
        print("No input Image")
        sys.exit()
    main(base_path, tiles_dir, nick_csv, output)

    # 初始化数据
    name_mapping = {}
    coord_index = {}

    # 加载数据文件
    def init_data():
        with open(CONFIG['csv_path'], encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                name_mapping[row[0]] = row[1]

        with open(CONFIG['index_path']) as f:
            coord_index.update(json.load(f))

        remove_keys = []
        for key, value in coord_index.items():
            if '_' in key:
                new_key = re.sub(r"_\d+", "", key)
                if new_key not in coord_index:
                    coord_index[new_key] = []
                coord_index[new_key].extend(value)
                remove_keys.append(key)

        if len(remove_keys) > 0:
            for k in remove_keys:
                del coord_index[k]

            with open(CONFIG['index_path'], 'w') as f:
                json.dump(coord_index, f)

    init_data()
    if len(sys.argv) > 1:
        name_file = name_mapping[sys.argv[1]]
        print(name_file)
        print(coord_index[name_file])

    unused_keys = len(set(name_mapping.values()) - set(coord_index.keys()))
    print("未使用素材头像次数: %d  %.02f%%" % (unused_keys, 100 * unused_keys / len(name_mapping)))
    name_mapping = {v:k for k, v in name_mapping.items()}

    coord_index = {k: len(item) for k, item in coord_index.items()}
    top_usage = {k: v for k, v in reversed(sorted(coord_index.items(), key=lambda item: item[1]))}
    print("\n使用次数最高的前50个素材：")
    for i, idx in enumerate(list(top_usage.keys())[:50], 1):
        print(f"{i:2d}. {name_mapping[idx]} - 使用次数: {top_usage[idx]}")


