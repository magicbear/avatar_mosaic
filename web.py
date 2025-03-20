import numpy as np
from flask import Flask, send_file, jsonify, request, render_template

import json
import math
import os
import csv
import image_nav
from io import BytesIO
import sys
import getopt

app = Flask(__name__, template_folder='template')
app.config['TEMPLATES_AUTO_RELOAD'] = True

# 配置项
CONFIG = {
    "csv_path": "nick.csv",
    "index_path": "gpusuma_12x_z3R40_15k_posthue.index",
    "source_image": "gpusuma_12x_z3R40_15k_posthue.png",
    "tile_cache": "cache",  # 瓦片缓存目录
    "preview_ratio": 0.1,  # 预览图缩放比例
    "tile_size": 256  # 缓存切片尺寸
}

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

# 路由定义
@app.route('/')
def index():
    return send_file('template/viewer.html', mimetype='text/html', max_age=86400 * 30)

@app.route('/ol.js')
def ol_js():
    return send_file('template/ol.js', mimetype='application/javascript', max_age=86400 * 30)

@app.route('/ol.css')
def ol_css():
    return send_file('template/ol.css', mimetype='text/css', max_age=86400 * 30)

@app.route('/ol.js.map')
def ol_js_map():
    return send_file('template/ol.js.map', mimetype='application/javascript', max_age=86400 * 30)

@app.route('/ol.css.map')
def ol_css_map():
    return send_file('template/ol.css.map', mimetype='text/css', max_age=86400 * 30)

@app.route('/avatar.jpg')
def avatar():
    return send_file('template/avatar.jpg', mimetype='text/css', max_age=86400 * 30)

@app.route('/tile/<int:z>_<int:x>_<int:y>')
def tile(z, x, y):
    return send_file(nav.get_tile(z, x, y), mimetype='image/jpeg', max_age=86400 * 30)

@app.route('/config')
def config():
    return send_file(BytesIO(json.dumps({
        "tileSize": CONFIG['tile_size'],
        "tileFactor": nav.zoom_levels,
        "origX": nav.width,
        "origY": nav.height,
        "max_zoom": len(nav.zoom_levels),
        "max_coods": nav.max_coods
    }).encode("utf-8")), mimetype='application/json', max_age=86400*30)

@app.route('/preview')
def preview():
    return send_file(nav.get_preview(), max_age=86400 * 30)

@app.route('/search')
def search():
    name = request.args.get('name')
    filename = name_mapping.get(name)
    if not filename:
        return jsonify({"error": "Name not found"}), 404

    coord = coord_index.get(filename)
    if not coord:
        return jsonify({"error": "Coordinate not found"}), 404

    return jsonify([{
                "x": c[0],
                "y": c[1]
            } for c in coord][:100]  # 返回所有坐标
    )

def reload_app():
    from importlib import reload
    global nav
    reload(image_nav)
    nav = image_nav.ImageNavigator(CONFIG, nav.source)
    app.run(host='0.0.0.0', port=5000)

if __name__ == '__main__':
    opts, args = getopt.gnu_getopt(sys.argv[1:], 'i:t:o:n:', ['image=', 'nick=', ""])

    for k, v in opts:
        if k in ("-i", "--image"):
            CONFIG['source_image'] = v
            CONFIG['index_path'] = os.path.splitext(v)[0]+".index"
        if k in ("-n", "--nick"):
            CONFIG['csv_path'] = v

    init_data()
    nav = image_nav.ImageNavigator(CONFIG)

    app.run(host='0.0.0.0', port=5000)
