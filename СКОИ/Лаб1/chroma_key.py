import os
import sys
import time
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def chroma_key_lib(fg_path, bg_path, target_color, tolerance, output_path):
    fg = cv2.imread(fg_path)
    bg = cv2.imread(bg_path)
    bg = cv2.resize(bg, (fg.shape[1], fg.shape[0]))
    target_bgr = np.array([target_color[2], target_color[1], target_color[0]], dtype=np.int16)
    lower = np.clip(target_bgr - tolerance, 0, 255).astype(np.uint8)
    upper = np.clip(target_bgr + tolerance, 0, 255).astype(np.uint8)
    mask = cv2.inRange(fg, lower, upper)
    inv_mask = cv2.bitwise_not(mask)
    fg_part = cv2.bitwise_and(fg, fg, mask=inv_mask)
    bg_part = cv2.bitwise_and(bg, bg, mask=mask)
    result = cv2.add(fg_part, bg_part)
    cv2.imwrite(output_path, result)

def chroma_key_native(fg_path, bg_path, target_color, tolerance, output_path):
    fg_img = Image.open(fg_path).convert('RGB')
    bg_img = Image.open(bg_path).convert('RGB')
    bg_img = bg_img.resize(fg_img.size)
    w, h = fg_img.size
    fg_pixels = fg_img.load()
    bg_pixels = bg_img.load()
    out_img = Image.new('RGB', (w, h))
    out_pixels = out_img.load()
    tr, tg, tb = target_color
    for y in range(h):
        for x in range(w):
            r, g, b = fg_pixels[x, y]
            if abs(r - tr) <= tolerance and abs(g - tg) <= tolerance and abs(b - tb) <= tolerance:
                out_pixels[x, y] = bg_pixels[x, y]
            else:
                out_pixels[x, y] = (r, g, b)
    out_img.save(output_path)

def run_benchmarks(fg_path, bg_path, target_color, tolerance):
    resolutions = [(200, 150), (400, 300), (800, 600), (1280, 720), (1920, 1080), (2560, 1440)]
    lib_times = []
    native_times = []
    labels = []
    base_fg = Image.open(fg_path).convert('RGB')
    base_bg = Image.open(bg_path).convert('RGB')
    created_files = []
    for w, h in resolutions:
        temp_fg = f'temp_fg_{w}x{h}.png'
        temp_bg = f'temp_bg_{w}x{h}.png'
        temp_out_lib = f'temp_out_lib_{w}x{h}.png'
        temp_out_nat = f'temp_out_nat_{w}x{h}.png'
        base_fg.resize((w, h)).save(temp_fg)
        base_bg.resize((w, h)).save(temp_bg)
        created_files.extend([temp_fg, temp_bg, temp_out_lib, temp_out_nat])
        t0 = time.perf_counter()
        chroma_key_lib(temp_fg, temp_bg, target_color, tolerance, temp_out_lib)
        lib_times.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        chroma_key_native(temp_fg, temp_bg, target_color, tolerance, temp_out_nat)
        native_times.append(time.perf_counter() - t0)
        labels.append(f'{w}x{h}')
    for f in created_files:
        if os.path.exists(f):
            os.remove(f)
    plt.figure(figsize=(11, 6))
    x_indices = np.arange(len(resolutions))
    width = 0.35
    rects1 = plt.bar(x_indices - width / 2, lib_times, width, label='Библиотечная (OpenCV)', color='#2b5c8f')
    rects2 = plt.bar(x_indices + width / 2, native_times, width, label='Нативная (Python)', color='#e26d5c')
    plt.bar_label(rects1, padding=3, fmt='%.3f', fontsize=8)
    plt.bar_label(rects2, padding=3, fmt='%.3f', fontsize=8)
    plt.xticks(x_indices, labels, fontsize=10)
    plt.xlabel('Разрешение изображения', fontsize=11)
    plt.ylabel('Время выполнения (секунды)', fontsize=11)
    plt.title('Сравнение быстродействия библиотечной и нативной реализаций', fontsize=13)
    plt.legend(fontsize=11)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('benchmark.png', dpi=300)
    plt.savefig('benchmark_6res.png', dpi=300)
    plt.close()
    return resolutions, lib_times, native_times

def main():
    if len(sys.argv) == 7:
        fg_path = sys.argv[1]
        bg_path = sys.argv[2]
        r = int(sys.argv[3])
        g = int(sys.argv[4])
        b = int(sys.argv[5])
        tolerance = int(sys.argv[6])
    else:
        fg_path = 'foreground.png'
        bg_path = 'background.png'
        r = 0
        g = 255
        b = 0
        tolerance = 30
    target_color = (r, g, b)
    t0 = time.perf_counter()
    chroma_key_lib(fg_path, bg_path, target_color, tolerance, 'output_lib.png')
    t_lib = time.perf_counter() - t0
    t0 = time.perf_counter()
    chroma_key_native(fg_path, bg_path, target_color, tolerance, 'output_native.png')
    t_native = time.perf_counter() - t0
    print(f'Library execution time: {t_lib:.6f} s')
    print(f'Native execution time: {t_native:.6f} s')
    print(f'Acceleration factor: {t_native / t_lib:.2f}x')
    resolutions, lib_times, native_times = run_benchmarks(fg_path, bg_path, target_color, tolerance)
    for res, lt, nt in zip(resolutions, lib_times, native_times):
        print(f'{res[0]}x{res[1]}: OpenCV = {lt:.5f}s, Native = {nt:.5f}s, Speedup = {nt/lt:.1f}x')

if __name__ == '__main__':
    main()
