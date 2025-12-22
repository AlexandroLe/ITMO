import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import numpy as np

# ========== Настройки и стиль ==========
root = tk.Tk()
root.title("Лабораторная работа — Алгоритмы компьютерной графики №2")
root.geometry("980x760")
root.configure(bg="#f4f7fb")

style = ttk.Style(root)
try:
    style.theme_use('clam')
except Exception:
    pass

style.configure('TButton', font=('Segoe UI', 10), padding=6)
style.configure('TLabel', font=('Segoe UI', 10))
style.configure('Header.TLabel', font=('Segoe UI', 12, 'bold'))
style.configure('TFrame', background="#f4f7fb")

HEADER_FONT = ("Segoe UI", 11, "bold")
NORMAL_FONT = ("Segoe UI", 10)
SMALL_FONT = ("Segoe UI", 9)

render_timer = None

# ========== Утилиты для debounce ==========
def debounced_render(filter_name=None, delay=120):
    global render_timer
    if render_timer is not None:
        root.after_cancel(render_timer)
    render_timer = root.after(delay, lambda: render_all(filter_name))

# ========== Обработка изображений ==========
def compute_histogram(img):
    arr = np.array(img)
    hist = [np.histogram(arr[..., i], bins=256, range=(0, 256))[0] for i in range(3)]
    return hist

def draw_hist(canvas, hist):
    canvas.delete("all")
    W, H = 256, 100
    try:
        max_val = max(int(h.max()) for h in hist) if hist and all(h.size for h in hist) else 1
    except Exception:
        max_val = 1
    if max_val == 0:
        max_val = 1
    colors = ['#eb2f3b', '#6ede45', '#4e83de']
    for ch in range(3):
        color = colors[ch]
        for x in range(256):
            hval = hist[ch][x] / max_val * H
            canvas.create_rectangle(x, H - hval, x + 1, H, outline=color, fill=color)

def to_grayscale(img):
    arr = np.array(img, dtype=np.uint8)
    gray = np.mean(arr[..., :3], axis=2)
    gray_img = np.stack([gray, gray, gray], axis=2).astype(np.uint8)
    return Image.fromarray(gray_img)

def change_brightness(img, val):
    arr = np.array(img, dtype=np.int16)
    delta = int(float(val) * 2.55)
    arr[..., :3] = np.clip(arr[..., :3] + delta, 0, 255)
    return Image.fromarray(arr.astype(np.uint8))

def change_contrast(img, val):
    arr = np.array(img, dtype=np.int16)
    c = float(val) * 2.55
    if (259 - c) == 0:
        f = 1.0
    else:
        f = (259 * (c + 255)) / (255 * (259 - c))
    for ch in range(3):
        arr[..., ch] = np.clip(f * (arr[..., ch] - 128) + 128, 0, 255)
    return Image.fromarray(arr.astype(np.uint8))

def invert_colors(img):
    arr = np.array(img, dtype=np.uint8)
    inv = 255 - arr
    return Image.fromarray(inv)

def gaussian_blur(img, radius=1):
    arr = np.array(img, dtype=np.float32)
    if radius <= 0:
        return Image.fromarray(arr.astype(np.uint8))
    radius = int(radius)
    sigma = max(radius, 1.0)
    ax = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(ax ** 2) / (2 * sigma ** 2))
    kernel = kernel / np.sum(kernel)

    padded = np.pad(arr, ((radius, radius), (radius, radius), (0, 0)), mode='edge')
    result = np.empty_like(padded)
    for c in range(3):
        col_conv = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode='same'), axis=0, arr=padded[:, :, c])
        full_conv = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode='same'), axis=1, arr=col_conv)
        result[:, :, c] = full_conv

    cropped = result[radius:-radius, radius:-radius, :]
    cropped = np.clip(cropped, 0, 255).astype(np.uint8)
    return Image.fromarray(cropped)

# ========== GUI ==========
frame_top = ttk.Frame(root, padding=(12, 10))
frame_top.pack(fill='x')

btn_load = ttk.Button(frame_top, text="Загрузить изображение")
btn_demo1 = ttk.Button(frame_top, text="Демо: img1")
btn_demo2 = ttk.Button(frame_top, text="Демо: img2")

btn_load.pack(side='left', padx=6)
btn_demo1.pack(side='left', padx=6)
btn_demo2.pack(side='left', padx=6)

status_var = tk.StringVar(value="Файл: —     Размер: —")
status = ttk.Label(root, textvariable=status_var, style='TLabel')
status.pack(side='bottom', fill='x', pady=(0, 6))

scroll_container = ttk.Frame(root)
scroll_container.pack(fill='both', expand=True)

scroll_canvas = tk.Canvas(scroll_container, bg='#f4f7fb', highlightthickness=0)
scroll_canvas.pack(side='left', fill='both', expand=True)

scrollbar = ttk.Scrollbar(scroll_container, orient='vertical', command=scroll_canvas.yview)
scrollbar.pack(side='right', fill='y')

scroll_canvas.configure(yscrollcommand=scrollbar.set)

canvas_frame = ttk.Frame(scroll_canvas, padding=10)
scroll_canvas.create_window((0,0), window=canvas_frame, anchor='nw')

def _on_frame_configure(event):
    scroll_canvas.configure(scrollregion=scroll_canvas.bbox('all'))

canvas_frame.bind('<Configure>', _on_frame_configure)

def _on_mousewheel(event):
    try:
        scroll_canvas.yview_scroll(int(-1*(event.delta/120)), 'units')
    except Exception:
        pass

root.bind('<MouseWheel>', _on_mousewheel)
root.bind('<Button-4>', lambda e: scroll_canvas.yview_scroll(-1, 'units'))
root.bind('<Button-5>', lambda e: scroll_canvas.yview_scroll(1, 'units'))

labels = ["Оригинал", "Ч/Б", "Инверсия", "Яркость", "Контраст", "Блюр", "Комбинированное"]

canvases = []
cols = 3

bright_slider = None
contrast_slider = None
blur_slider = None

for i, lbl in enumerate(labels):
    row = i // cols
    col = i % cols
    box = ttk.LabelFrame(canvas_frame, text=lbl, padding=(8, 8))
    box.grid(row=row, column=col, padx=8, pady=8, sticky='n')

    img_canvas = tk.Canvas(box, width=256, height=256, bg="#e9eef6", highlightthickness=1, highlightbackground="#cfd8e3")
    img_canvas.pack()
    canvases.append(img_canvas)

    hist_canvas = tk.Canvas(box, width=256, height=100, bg="#ffffff", highlightthickness=1, highlightbackground="#cfd8e3")
    hist_canvas.pack(pady=6)
    canvases.append(hist_canvas)

    if lbl == "Яркость":
        lblv = ttk.Label(box, text="Яркость", font=SMALL_FONT)
        lblv.pack()
        s = tk.Scale(box, from_=-100, to=100, orient='horizontal', length=220)
        s.pack(pady=(0,4))
        bright_slider = s
    elif lbl == "Контраст":
        lblv = ttk.Label(box, text="Контраст", font=SMALL_FONT)
        lblv.pack()
        s = tk.Scale(box, from_=-100, to=100, orient='horizontal', length=220)
        s.pack(pady=(0,4))
        contrast_slider = s
    elif lbl == "Блюр":
        lblv = ttk.Label(box, text="Размытие (σ)", font=SMALL_FONT)
        lblv.pack()
        s = tk.Scale(box, from_=0, to=10, orient='horizontal', length=220)
        s.pack(pady=(0,4))
        blur_slider = s

if bright_slider is None:
    bright_slider = tk.Scale(root, from_=-100, to=100, orient='horizontal')
if contrast_slider is None:
    contrast_slider = tk.Scale(root, from_=-100, to=100, orient='horizontal')
if blur_slider is None:
    blur_slider = tk.Scale(root, from_=0, to=10, orient='horizontal')

bright_slider.config(command=lambda v: debounced_render(None))
contrast_slider.config(command=lambda v: debounced_render(None))
blur_slider.config(command=lambda v: debounced_render(None))

state = {'image': None, 'path': None}

def render_all(filter_name=None):
    if state['image'] is None:
        return
    base = state['image']

    imgs = {
        "orig": base,
        "gray": to_grayscale(base),
        "invert": invert_colors(base),
        "bright": change_brightness(base, bright_slider.get()),
        "contrast": change_contrast(base, contrast_slider.get()),
        "blur": gaussian_blur(base, blur_slider.get())
    }

    combined = change_brightness(base, bright_slider.get())
    combined = change_contrast(combined, contrast_slider.get())
    combined = gaussian_blur(combined, blur_slider.get())
    imgs["combined"] = combined

    keys = list(imgs.keys())
    mapping = {name: idx for idx, name in enumerate(keys)}

    if filter_name is None:
        indices = range(len(keys))
    else:
        indices = [mapping.get(filter_name, 0)]

    for i in indices:
        key = keys[i]
        img = imgs[key]
        img_small = img.copy()
        img_small.thumbnail((256, 256))
        tk_img = ImageTk.PhotoImage(img_small)
        c = canvases[i * 2]
        c.delete("all")
        c.create_image(128, 128, anchor='center', image=tk_img)
        c.image = tk_img
        draw_hist(canvases[i * 2 + 1], compute_histogram(img))

# ========== Загрузка изображений ==========
def open_file():
    path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.png *.jpeg *.bmp")])
    if path:
        try:
            img = Image.open(path).convert("RGB")
            state['image'] = img
            state['path'] = path
            status_var.set(f"Файл: {path}     Размер: {img.width}x{img.height}")
            render_all()
        except Exception as e:
            status_var.set(f"Ошибка при загрузке: {e}")

def load_demo1():
    try:
        img = Image.open("IMG1.jpeg").convert("RGB")
        state['image'] = img
        state['path'] = "IMG1.jpeg"
        status_var.set(f"Файл: IMG1.jpeg     Размер: {img.width}x{img.height}")
        render_all()
    except Exception as e:
        status_var.set(f"Не удалось загрузить demo img1: {e}")

def load_demo2():
    try:
        img = Image.open("IMG2.jpeg").convert("RGB")
        state['image'] = img
        state['path'] = "IMG2.jpeg"
        status_var.set(f"Файл: IMG2.jpeg     Размер: {img.width}x{img.height}")
        render_all()
    except Exception as e:
        status_var.set(f"Не удалось загрузить demo img2: {e}")

btn_load.config(command=open_file)
btn_demo1.config(command=load_demo1)
btn_demo2.config(command=load_demo2)

for b in (btn_load, btn_demo1, btn_demo2):
    b.bind('<Enter>', lambda e: e.widget.configure(cursor='hand2'))
    b.bind('<Leave>', lambda e: e.widget.configure(cursor=''))

root.mainloop()
