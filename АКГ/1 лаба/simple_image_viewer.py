import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os
from datetime import datetime
import math

IMAGES = {
    "foto 1": "im1.jpg",
    "foto 2": "im2.jpg",
    "foto 3": "im3.jpg",
}

THUMB_SIZE = (120, 90)
DISPLAY_MAX = (900, 650)

try:
    RESAMPLE = Image.Resampling.LANCZOS
except Exception:
    RESAMPLE = Image.ANTIALIAS


class SimpleImageAppDark(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Тёмный просмотрщик — метаданные и средние RGB")
        self.geometry("1150x760")
        self.minsize(920, 620)

        self.bg = "#121212"
        self.panel_bg = "#1c1c1c"
        self.card_bg = "#161616"
        self.accent = "#3399FF"
        self.fg = "#E6EEF6"
        self.sub_fg = "#AAB6C7"
        self.btn_bg = "#222222"

        self.configure(bg=self.bg)

        self.current_image = None
        self.current_photo = None
        self.thumbs = {}

        self.image_items = list(IMAGES.items())

        self.create_ui()

        if self.image_items:
            for lbl, p in self.image_items:
                if os.path.exists(p):
                    self.load_image(p, lbl)
                    break

        self.bind("<Configure>", self._on_configure)

    def create_ui(self):
        main = tk.Frame(self, bg=self.bg, padx=8, pady=8)
        main.pack(fill=tk.BOTH, expand=True)

        left = tk.Frame(main, bg=self.panel_bg, width=230)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0,8))
        left.pack_propagate(False)

        add_btn = tk.Button(left, text="Добавить...", bg=self.btn_bg, fg=self.fg, relief="flat",
                            command=self._on_add_images)
        add_btn.pack(fill=tk.X, pady=(10, 6), padx=8)

        tk.Label(left, text="Выбери изображение:", bg=self.panel_bg, fg=self.accent,
                 font=("Arial", 11, "bold")).pack(pady=(0, 6))

        thumbs_outer = tk.Frame(left, bg=self.panel_bg)
        thumbs_outer.pack(fill=tk.BOTH, expand=True, padx=4, pady=(0,8))

        nav_frame = tk.Frame(thumbs_outer, bg=self.panel_bg)
        nav_frame.pack(fill=tk.X, padx=4, pady=(0,6))
        prev_btn = tk.Button(nav_frame, text="Prev", width=8, command=lambda: self._thumb_scroll_by(-1))
        next_btn = tk.Button(nav_frame, text="Next", width=8, command=lambda: self._thumb_scroll_by(1))
        prev_btn.pack(side=tk.LEFT, padx=(0,6))
        next_btn.pack(side=tk.LEFT)

        canvas_frame = tk.Frame(thumbs_outer, bg=self.panel_bg)
        canvas_frame.pack(fill=tk.BOTH, expand=True)

        self.thumbs_canvas = tk.Canvas(canvas_frame, bg=self.panel_bg, highlightthickness=0)
        self.thumbs_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.thumbs_vsb = tk.Scrollbar(canvas_frame, orient="vertical", command=self._on_thumbs_vsb)
        self.thumbs_vsb.pack(side=tk.RIGHT, fill=tk.Y)

        self.thumb_scale = tk.Scale(canvas_frame, from_=0, to=100, orient="vertical",
                                    command=self._on_thumb_scale, showvalue=0, bg=self.panel_bg,
                                    troughcolor=self.btn_bg, fg=self.fg, bd=0, highlightthickness=0)
        self.thumb_scale.pack(side=tk.RIGHT, fill=tk.Y, padx=(4,0), pady=4)

        self.thumbs_canvas.configure(yscrollcommand=self._on_thumbs_scroll)

        self.thumbs_frame = tk.Frame(self.thumbs_canvas, bg=self.panel_bg)
        self.thumbs_window = self.thumbs_canvas.create_window((0, 0), window=self.thumbs_frame, anchor="nw")
        self.thumbs_frame.bind("<Configure>", lambda e: self._update_thumbs_scrollregion())

        self._bind_mousewheel(self.thumbs_canvas)

        for label, path in self.image_items:
            self._create_thumbnail_button(label, path)

        center = tk.Frame(main, bg=self.bg)
        center.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.img_frame = tk.LabelFrame(center, text="Изображение", bg=self.card_bg, fg=self.fg,
                                       labelanchor='n', padx=6, pady=6)
        self.img_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)

        self.img_label = tk.Label(self.img_frame, bg=self.card_bg)
        self.img_label.pack(expand=True, fill=tk.BOTH)

        info_outer = tk.LabelFrame(center, text="Информация", bg=self.card_bg, fg=self.fg,
                                   labelanchor='n', padx=0, pady=0)
        info_outer.pack(fill=tk.X, padx=8, pady=(0,8))

        self.info_canvas = tk.Canvas(info_outer, bg=self.card_bg, highlightthickness=0, height=220)
        self.info_vsb = tk.Scrollbar(info_outer, orient="vertical", command=self.info_canvas.yview)
        self.info_canvas.configure(yscrollcommand=self.info_vsb.set)

        self.info_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.info_vsb.pack(side=tk.RIGHT, fill=tk.Y)

        self.info_inner = tk.Frame(self.info_canvas, bg=self.card_bg)
        self.info_window = self.info_canvas.create_window((0, 0), window=self.info_inner, anchor="nw")
        self.info_inner.bind("<Configure>", lambda e: self.info_canvas.configure(scrollregion=self.info_canvas.bbox("all")))

        self.info_vars = {
            "name": tk.StringVar(value="N/A"),
            "format": tk.StringVar(value="N/A"),
            "resolution": tk.StringVar(value="N/A"),
            "filesize": tk.StringVar(value="N/A"),
            "mean_r": tk.StringVar(value="N/A"),
            "mean_g": tk.StringVar(value="N/A"),
            "mean_b": tk.StringVar(value="N/A"),
            "mode": tk.StringVar(value="N/A"),
            "channels": tk.StringVar(value="N/A"),
            "aspect": tk.StringVar(value="N/A"),
            "orientation": tk.StringVar(value="N/A"),
            "created": tk.StringVar(value="N/A"),
            "modified": tk.StringVar(value="N/A"),
            "dpi": tk.StringVar(value="N/A"),
            "unique_colors": tk.StringVar(value="N/A"),
            "path": tk.StringVar(value="N/A"),
        }

        rows = [
            ("Имя файла:", "name"),
            ("Полный путь:", "path"),
            ("Формат:", "format"),
            ("Режим (mode):", "mode"),
            ("Каналов:", "channels"),
            ("Разрешение (WxH):", "resolution"),
            ("Соотношение (W/H):", "aspect"),
            ("Ориентация:", "orientation"),
            ("Размер файла (МБ):", "filesize"),
            ("DPI:", "dpi"),
            ("Дата создания:", "created"),
            ("Дата изменения:", "modified"),
            ("Средний R:", "mean_r"),
            ("Средний G:", "mean_g"),
            ("Средний B:", "mean_b"),
            ("Уникальных цветов (примерно):", "unique_colors"),
        ]

        for i, (label_text, key) in enumerate(rows):
            lbl = tk.Label(self.info_inner, text=label_text, anchor="w", bg=self.card_bg, fg=self.sub_fg)
            val = tk.Label(self.info_inner, textvariable=self.info_vars[key], anchor="w", bg=self.card_bg, fg=self.fg, wraplength=560, justify="left")
            lbl.grid(row=i, column=0, sticky="w", padx=8, pady=4)
            val.grid(row=i, column=1, sticky="w", padx=8, pady=4)

        btns_frame = tk.Frame(self.info_inner, bg=self.card_bg)
        btns_frame.grid(row=len(rows), column=0, columnspan=2, sticky="w", padx=8, pady=(6,12))

        copy_btn = tk.Button(btns_frame, text="Копировать путь", bg=self.btn_bg, fg=self.fg, relief="flat", command=self._copy_path_to_clipboard)
        copy_btn.pack(side=tk.LEFT, padx=(0,8))

        save_copy_btn = tk.Button(btns_frame, text="Сохранить копию (уменьшённая)", bg=self.btn_bg, fg=self.fg, relief="flat", command=self._save_resized_copy)
        save_copy_btn.pack(side=tk.LEFT)

        bars_frame = tk.LabelFrame(main, text="Визуализация средних каналов", bg=self.panel_bg, fg=self.fg,
                                   labelanchor='n', padx=6, pady=6)
        bars_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(8,0), pady=6)
        self.canvas_rgb = tk.Canvas(bars_frame, width=260, height=260, bg=self.panel_bg, highlightthickness=0)
        self.canvas_rgb.pack(padx=6, pady=6)
        self.canvas_rgb.create_text(130, 12, text="R   G   B", font=("Arial", 10, "bold"), fill=self.sub_fg)

        self.bar_coords = {
            "r": (24, 36, 176, 76),
            "g": (24, 86, 176, 126),
            "b": (24, 136, 176, 176),
        }
        for _, coords in self.bar_coords.items():
            self.canvas_rgb.create_rectangle(*coords, outline="#333")

        self.bar_items = {
            "r": self.canvas_rgb.create_rectangle(24, 36, 24, 76, fill="#ff4d4d", outline=""),
            "g": self.canvas_rgb.create_rectangle(24, 86, 24, 126, fill="#28c76f", outline=""),
            "b": self.canvas_rgb.create_rectangle(24, 136, 24, 176, fill="#4d6bff", outline=""),
        }

        self.bar_value_texts = {
            "r": self.canvas_rgb.create_text(100, 26, text="0.00", fill=self.fg, font=("Arial", 9, "bold")),
            "g": self.canvas_rgb.create_text(100, 76, text="0.00", fill=self.fg, font=("Arial", 9, "bold")),
            "b": self.canvas_rgb.create_text(100, 126, text="0.00", fill=self.fg, font=("Arial", 9, "bold")),
        }

        self.canvas_rgb.create_text(192, 26, text="0", fill=self.sub_fg, font=("Arial", 8))
        self.canvas_RGB_scale_line = self.canvas_rgb.create_line(186, 36, 186, 176, fill="#333", dash=(2,))
        self.canvas_rgb.create_text(192, 176, text="255", fill=self.sub_fg, font=("Arial", 8))

        self.canvas_rgb.create_text(130, 196, text="Средний цвет:", fill=self.sub_fg, font=("Arial", 9))
        self.avg_color_box = self.canvas_rgb.create_rectangle(80, 206, 180, 246, fill="#222222", outline="#333")

        copy_rgb_btn = tk.Button(bars_frame, text="Копировать RGB", bg=self.btn_bg, fg=self.fg, relief="flat", command=self._copy_rgb_to_clipboard)
        copy_rgb_btn.pack(pady=(6,0))

    def _update_thumbs_scrollregion(self):
        self.thumbs_canvas.configure(scrollregion=self.thumbs_canvas.bbox("all"))
        try:
            w = self.thumbs_canvas.winfo_width()
            self.thumbs_canvas.itemconfig(self.thumbs_window, width=w)
        except Exception:
            pass
        try:
            first, last = self.thumbs_canvas.yview()
            self.thumb_scale.set(int(first * 100))
        except Exception:
            pass

    def _on_thumbs_scroll(self, first, last):
        try:
            self.thumbs_vsb.set(first, last)
            top_frac = float(first)
            self.thumb_scale.set(int(top_frac * 100))
        except Exception:
            pass

    def _on_thumbs_vsb(self, *args):
        try:
            self.thumbs_canvas.yview(*args)
            first, last = self.thumbs_canvas.yview()
            self.thumb_scale.set(int(first * 100))
        except Exception:
            pass

    def _on_thumb_scale(self, val):
        try:
            frac = float(val) / 100.0
            self.thumbs_canvas.yview_moveto(frac)
        except Exception:
            pass

    def _thumb_scroll_by(self, step_pages):
        try:
            self.thumbs_canvas.yview_scroll(step_pages, "pages")
            first, _ = self.thumbs_canvas.yview()
            self.thumb_scale.set(int(first * 100))
        except Exception:
            pass

    def _bind_mousewheel(self, widget):
        if os.name == 'nt':
            widget.bind_all("<MouseWheel>", self._on_mousewheel_thumbs)
        else:
            widget.bind_all("<Button-4>", self._on_mousewheel_thumbs)
            widget.bind_all("<Button-5>", self._on_mousewheel_thumbs)
            widget.bind_all("<MouseWheel>", self._on_mousewheel_thumbs)

    def _on_mousewheel_thumbs(self, event):
        x, y = self.winfo_pointerx(), self.winfo_pointery()
        widget_under_pointer = self.winfo_containing(x, y)
        if widget_under_pointer is None:
            return

        if widget_under_pointer == self.thumbs_canvas or widget_under_pointer.winfo_ismapped() and self._is_child_of(widget_under_pointer, self.thumbs_frame):
            try:
                if event.num == 4:
                    self.thumbs_canvas.yview_scroll(-1, "units")
                elif event.num == 5:
                    self.thumbs_canvas.yview_scroll(1, "units")
                else:
                    delta = 0
                    if hasattr(event, "delta"):
                        delta = event.delta
                    scroll_units = -1 * int(delta / 120) if delta != 0 else 0
                    if scroll_units != 0:
                        self.thumbs_canvas.yview_scroll(scroll_units, "units")
                first, _ = self.thumbs_canvas.yview()
                self.thumb_scale.set(int(first * 100))
            except Exception:
                pass

    def _is_child_of(self, widget, parent):
        while widget is not None:
            if widget == parent:
                return True
            widget = widget.master
        return False

    def _on_configure(self, event):
        try:
            canvas_w = self.info_canvas.winfo_width()
            self.info_canvas.itemconfig(self.info_window, width=canvas_w)
        except Exception:
            pass

        try:
            canvas_w2 = self.thumbs_canvas.winfo_width()
            self.thumbs_canvas.itemconfig(self.thumbs_window, width=canvas_w2)
        except Exception:
            pass

        if event.widget == self:
            if self.current_image:
                self.after(30, lambda: self._update_image_display(self.current_image))

    def _on_add_images(self):
        paths = filedialog.askopenfilenames(title="Выберите изображения",
                                            filetypes=[("Images", ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.gif")),
                                                       ("All files", "*.*")])
        if not paths:
            return
        added_any = False
        for p in paths:
            if not p:
                continue
            base = os.path.basename(p)
            label = base
            if any(p == existing_path for (_, existing_path) in self.image_items):
                continue
            existing_labels = {lbl for (lbl, _) in self.image_items}
            suffix = 1
            while label in existing_labels:
                label = f"{base} ({suffix})"
                suffix += 1
            self.image_items.append((label, p))
            self._create_thumbnail_button(label, p)
            added_any = True
        if added_any:
            self.load_image(self.image_items[-1][1], self.image_items[-1][0])
            self._update_thumbs_scrollregion()

    def _create_thumbnail_button(self, label, path):
        try:
            img = Image.open(path)
            img.thumbnail(THUMB_SIZE, RESAMPLE)
            thumb = ImageTk.PhotoImage(img)
            self.thumbs[label] = thumb
            btn = tk.Button(self.thumbs_frame, image=thumb, text=label, compound=tk.TOP,
                            bg=self.btn_bg, fg=self.fg, relief="flat",
                            command=lambda p=path, n=label: self.load_image(p, n))
        except Exception:
            btn = tk.Button(self.thumbs_frame, text=label, width=16,
                            bg=self.btn_bg, fg=self.fg, relief="flat",
                            command=lambda p=path, n=label: self.load_image(p, n))
        btn.pack(pady=6, padx=6, fill=tk.X)

    def _update_image_display(self, pil_image):
        display_width = self.img_label.winfo_width() or DISPLAY_MAX[0]
        display_height = self.img_label.winfo_height() or DISPLAY_MAX[1]

        if display_width <= 1 or display_height <= 1:
            return

        img_width, img_height = pil_image.size

        if img_width > display_width or img_height > display_height:
            ratio = min(display_width / img_width, display_height / img_height)
            new_width = max(1, int(img_width * ratio))
            new_height = max(1, int(img_height * ratio))
            resized_image = pil_image.resize((new_width, new_height), RESAMPLE)
        else:
            resized_image = pil_image

        self.current_photo = ImageTk.PhotoImage(resized_image)
        self.img_label.configure(image=self.current_photo)

    def load_image(self, path, name_for_button=None):
        if not os.path.exists(path):
            messagebox.showerror("Ошибка", f"Файл не найден:\n{path}")
            return
        try:
            pil = Image.open(path)
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не получилось открыть изображение:\n{e}")
            return

        self.current_image = pil

        self._update_image_display(pil)

        file_name = os.path.basename(path)
        img_format = pil.format or "N/A"
        resolution = f"{pil.width}x{pil.height}"
        try:
            file_size_mb = os.path.getsize(path) / (1024 * 1024)
            file_size_s = f"{file_size_mb:.2f} MB"
        except Exception:
            file_size_s = "N/A"

        mode = pil.mode or "N/A"
        channels = len(pil.getbands()) if hasattr(pil, "getbands") else "N/A"

        aspect_ratio = pil.width / pil.height if pil.height != 0 else 0
        aspect = f"{aspect_ratio:.2f}"
        if pil.width > pil.height:
            orientation = "Ландшафт"
        elif pil.width < pil.height:
            orientation = "Портрет"
        else:
            orientation = "Квадрат"

        dpi = pil.info.get("dpi") if isinstance(pil.info, dict) else None
        dpi_s = f"{dpi[0]}x{dpi[1]}" if dpi and isinstance(dpi, (tuple, list)) and len(dpi) >= 2 else "N/A"

        try:
            ctime = os.path.getctime(path)
            mtime = os.path.getmtime(path)
            created_s = datetime.fromtimestamp(ctime).strftime("%Y-%m-%d %H:%M:%S")
            modified_s = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            created_s = modified_s = "N/A"

        rgb_image = pil.convert("RGB")
        px_count = rgb_image.width * rgb_image.height
        max_px_for_mean = 300_000
        if px_count > max_px_for_mean:
            factor = math.sqrt(max_px_for_mean / px_count)
            new_w = max(1, int(rgb_image.width * factor))
            new_h = max(1, int(rgb_image.height * factor))
            rgb_for_calc = rgb_image.resize((new_w, new_h), RESAMPLE)
        else:
            rgb_for_calc = rgb_image

        pixels = list(rgb_for_calc.getdata())
        total = len(pixels) or 1
        sum_r = sum(p[0] for p in pixels)
        sum_g = sum(p[1] for p in pixels)
        sum_b = sum(p[2] for p in pixels)
        mean_r = sum_r / total
        mean_g = sum_g / total
        mean_b = sum_b / total

        try:
            unique_colors = len(set(pixels))
            unique_s = f"{unique_colors}" if unique_colors <= 50000 else f">{50000}"
        except Exception:
            unique_s = "N/A"

        self.info_vars["name"].set(file_name)
        self.info_vars["path"].set(os.path.abspath(path))
        self.info_vars["format"].set(img_format)
        self.info_vars["mode"].set(mode)
        self.info_vars["channels"].set(str(channels))
        self.info_vars["resolution"].set(resolution)
        self.info_vars["aspect"].set(aspect)
        self.info_vars["orientation"].set(orientation)
        self.info_vars["filesize"].set(file_size_s)
        self.info_vars["dpi"].set(dpi_s)
        self.info_vars["created"].set(created_s)
        self.info_vars["modified"].set(modified_s)
        self.info_vars["mean_r"].set(f"{mean_r:.2f}")
        self.info_vars["mean_g"].set(f"{mean_g:.2f}")
        self.info_vars["mean_b"].set(f"{mean_b:.2f}")
        self.info_vars["unique_colors"].set(unique_s)

        for ch, mean in (("r", mean_r), ("g", mean_g), ("b", mean_b)):
            left, top, right, bottom = self.bar_coords[ch]
            max_w = right - left
            new_w = int(left + (mean / 255.0) * max_w)
            self.canvas_rgb.coords(self.bar_items[ch], left, top, new_w, bottom)

            if ch == "r":
                self.canvas_rgb.coords(self.bar_value_texts["r"], left + 4, top - 10)
                self.canvas_rgb.itemconfig(self.bar_value_texts["r"], text=f"{mean:.2f}")
            elif ch == "g":
                self.canvas_rgb.coords(self.bar_value_texts["g"], left + 4, top - 10)
                self.canvas_rgb.itemconfig(self.bar_value_texts["g"], text=f"{mean:.2f}")
            else:
                self.canvas_rgb.coords(self.bar_value_texts["b"], left + 4, top - 10)
                self.canvas_rgb.itemconfig(self.bar_value_texts["b"], text=f"{mean:.2f}")

        mean_rgb_hex = "#%02x%02x%02x" % (int(mean_r), int(mean_g), int(mean_b))
        self.canvas_rgb.itemconfig(self.avg_color_box, fill=mean_rgb_hex, outline="#888")

        title = f"{file_name} — {resolution} — {img_format}"
        self.title(title)

        self._last_mean_rgb = (mean_r, mean_g, mean_b)

        self._update_thumbs_scrollregion()

    def _copy_path_to_clipboard(self):
        path = self.info_vars.get("path").get()
        if not path or path == "N/A":
            messagebox.showwarning("Нет пути", "Сначала откройте изображение.")
            return
        try:
            self.clipboard_clear()
            self.clipboard_append(path)
            messagebox.showinfo("Скопировано", "Путь к файлу скопирован в буфер обмена.")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось скопировать путь:\n{e}")

    def _save_resized_copy(self):
        if not self.current_image:
            messagebox.showwarning("Нет изображения", "Сначала откройте изображение.")
            return
        default_name = f"copy_{self.info_vars['name'].get()}"
        save_path = filedialog.asksaveasfilename(title="Сохранить копию как", initialfile=default_name,
                                                 defaultextension=".jpg", filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png"), ("All files", "*.*")])
        if not save_path:
            return
        try:
            img = self.current_image.copy()
            w, h = img.size
            if max(w, h) > 1200:
                factor = 1200 / max(w, h)
                img = img.resize((int(w * factor), int(h * factor)), RESAMPLE)
            img.save(save_path)
            messagebox.showinfo("Сохранено", f"Копия сохранена в:\n{save_path}")
        except Exception as e:
            messagebox.showerror("Ошибка сохранения", f"Не удалось сохранить копию:\n{e}")

    def _copy_rgb_to_clipboard(self):
        if not hasattr(self, "_last_mean_rgb"):
            messagebox.showwarning("Нет данных", "Сначала откройте изображение.")
            return
        r, g, b = self._last_mean_rgb
        s = f"{r:.2f},{g:.2f},{b:.2f}"
        try:
            self.clipboard_clear()
            self.clipboard_append(s)
            messagebox.showinfo("Скопировано", f"RGB ({s}) скопированы в буфер обмена.")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось скопировать RGB:\n{e}")


if __name__ == "__main__":
    app = SimpleImageAppDark()
    app.mainloop()