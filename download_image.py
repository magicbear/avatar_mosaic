import csv
import os
import requests
from urllib.parse import urlparse
from multiprocessing import Pool, Manager
from tqdm import tqdm
import time


def download_worker(args):
    url, row_number, output_folder, progress_queue = args
    try:
        if not url:
            return (False, row_number, "URL为空")

        # 生成文件名
        parsed_url = urlparse(url)
        filename = os.path.basename(parsed_url.path) or f"image_{row_number}"
        if not os.path.splitext(filename)[1]:
            filename += ".jpg"

        os.makedirs(os.path.join(output_folder, filename[0:2]), exist_ok=True)
        # 处理重名文件
        save_path = os.path.join(output_folder, filename[0:2]+"/"+filename)

        if not os.path.exists(save_path):
            response = requests.get(url, timeout=10)
            response.raise_for_status()

            with open(save_path, 'wb') as f:
                f.write(response.content)

        progress_queue.put(1)
        return (True, row_number, url)
    except Exception as e:
        progress_queue.put(1)
        return (False, row_number, str(e))


def download_images_from_csv(csv_file, output_folder='images', processes=4):
    os.makedirs(output_folder, exist_ok=True)

    # 读取所有任务
    tasks = []
    with open(csv_file, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        for row_number, row in enumerate(reader, 1):
            if len(row) >= 2:
                url = row[1].strip()
                tasks.append((url, row_number))

    # 创建进度队列
    with Manager() as manager:
        progress_queue = manager.Queue()
        total = len(tasks)

        # 创建进程池
        with Pool(processes=processes) as pool:
            # 进度条线程
            pbar = tqdm(total=total, desc="下载进度", unit="file")

            # 启动下载任务
            results = pool.imap_unordered(download_worker,
                                          [(task[0], task[1], output_folder, progress_queue) for task in tasks])

            # 更新进度条
            success = 0
            failures = 0
            error_log = []

            def update_progress():
                while not progress_queue.empty():
                    pbar.update(1)
                    progress_queue.get()

            for result in results:
                update_progress()
                if result[0]:
                    success += 1
                else:
                    failures += 1
                    error_log.append(f"行 {result[1]}: {result[2]}")

            # 处理剩余进度
            while pbar.n < total:
                update_progress()
                time.sleep(0.1)

            pbar.close()

    # 输出统计信息
    print(f"\n下载完成！成功：{success} 个，失败：{failures} 个")
    if error_log:
        print("\n错误详情：")
        print("\n".join(error_log[-10:]))  # 显示最后10个错误


if __name__ == "__main__":
    csv_path = input("请输入CSV文件路径：")
    download_images_from_csv(csv_path, processes=os.cpu_count())