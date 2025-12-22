import math
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
from PIL import Image, ImageTk
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ----------------- Цветовая схема (Azure-like dark) -----------------
BG = "#0f1720"        # основной фон окна
PANEL = "#111520"     # фон панелей / карточек
PANEL2 = "#172029"    # чуть светлее для внутренних областей
FG = "#E6EEF3"        # основной светлый текст
ENTRY_BG = "#0f1620"  # поле ввода
BUTTON_BG = "#1f3a5a" # фон кнопок
ACCENT = "#3aa3ff"    # акцентный цвет (необязательно)

# ----------------- Утилиты -----------------
def format_sci(val):
    try:
        if val == 0:
            return "0"
        exp = int(math.floor(math.log10(abs(val))))
        mant = val / (10**exp)
        return f"{mant:.3f} * 10^{exp}"
    except:
        return str(val)

# ----------------- Константы по умолчанию -----------------
DEFAULTS = dict(
    W=1000.0, H=800.0,  # mm
    Wres=600, Hres=480,  # px
    xL=200.0, yL=-150.0, zL=600.0,  # mm
    I0=200.0,  # W/sr
    x0=0.0, y0=0.0, R=200.0,
)

# ----------------- Вычисления -----------------
def compute_fields(W, H, Wres, Hres, xL, yL, zL, I0, x0, y0, R):
    xs = np.linspace(-W / 2, W / 2, Wres)
    ys = np.linspace(-H / 2, H / 2, Hres)
    X, Y = np.meshgrid(xs, ys)

    RX = X - xL
    RY = Y - yL
    RZ = -zL
    d = np.sqrt(RX * RX + RY * RY + RZ * RZ) + 1e-12
    cos_theta = np.clip(zL / d, 0.0, 1.0)
    E = I0 * (cos_theta ** 2) / (d ** 2) # наша рабочая формула
    E = E * 1e6

    mask = (X - x0) ** 2 + (Y - y0) ** 2 <= R * R
    circle_vals = E[mask]
    norm_max = float(np.max(circle_vals)) if circle_vals.size else float(np.max(E))
    norm = np.where(mask, E / (norm_max + 1e-12), 0.0)
    img = (np.clip(norm, 0, 1) * 255.0).astype(np.uint8)

    def E_point(px, py):
        RXp, RYp, RZp = px - xL, py - yL, -zL
        dp = math.sqrt(RXp * RXp + RYp * RYp + RZp * RZp)
        ct = max(0.0, min(1.0, zL / dp))
        return I0 * (ct ** 2) / (dp ** 2) * 1e6

    five = dict(
        center=(x0, y0, E_point(x0, y0)),
        x_plus_R=(x0 + R, y0, E_point(x0 + R, y0)),
        x_minus_R=(x0 - R, y0, E_point(x0 - R, y0)),
        y_plus_R=(x0, y0 + R, E_point(x0, y0 + R)),
        y_minus_R=(x0, y0 - R, E_point(x0, y0 - R)),
    )

    stats = dict(
        E_max=float(np.max(circle_vals)) if circle_vals.size else float(np.max(E)),
        E_min=float(np.min(circle_vals)) if circle_vals.size else float(np.min(E)),
        E_mean=float(np.mean(circle_vals)) if circle_vals.size else float(np.mean(E)),
    )

    return dict(E=E, img=img, xs=xs, ys=ys, mask=mask, stats=stats, five=five, params=dict(
        W=W, H=H, Wres=Wres, Hres=Hres, xL=xL, yL=yL, zL=zL, I0=I0, x0=x0, y0=y0, R=R
    ))

# ----------------- Приложение -----------------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Lambert Plane — Irradiance (dark)")
        self.resizable(True, True)

        # применим общий фон окна
        self.configure(bg=BG)

        # --- Настройка ttk-стиля (clam удобно кастомизировать) ---
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except:
            pass

        # общие настройки стилей
        style.configure("TFrame", background=BG)
        style.configure("Card.TFrame", background=PANEL)
        style.configure("Inner.TFrame", background=PANEL2)

        style.configure("TLabel", background=BG, foreground=FG)
        style.configure("Card.TLabel", background=PANEL, foreground=FG)
        style.configure("Inner.TLabel", background=PANEL2, foreground=FG)

        style.configure("TButton",
                        background=BUTTON_BG,
                        foreground=FG,
                        relief="flat",
                        padding=4)
        style.map("TButton",
                  background=[("active", ACCENT), ("pressed", BUTTON_BG)],
                  foreground=[("disabled", "#777")])

        style.configure("TEntry",
                        fieldbackground=ENTRY_BG,
                        background=ENTRY_BG,
                        foreground=FG)
        style.configure("TCheckbutton", background=BG, foreground=FG)

        style.configure("Vertical.TScrollbar", troughcolor=PANEL, background=PANEL2)

        # переменные (контролы)
        self.vars = {}
        self.vars['W'] = tk.DoubleVar(value=DEFAULTS['W'])
        self.vars['H'] = tk.DoubleVar(value=DEFAULTS['H'])
        self.vars['Wres'] = tk.IntVar(value=DEFAULTS['Wres'])
        self.vars['Hres'] = tk.IntVar(value=DEFAULTS['Hres'])
        self.vars['zL'] = tk.DoubleVar(value=DEFAULTS['zL'])
        self.vars['I0'] = tk.DoubleVar(value=DEFAULTS['I0'])
        self.vars['R'] = tk.DoubleVar(value=DEFAULTS['R'])
        self.vars['xL'] = tk.DoubleVar(value=DEFAULTS['xL'])
        self.vars['yL'] = tk.DoubleVar(value=DEFAULTS['yL'])
        self.vars['x0'] = tk.DoubleVar(value=DEFAULTS['x0'])
        self.vars['y0'] = tk.DoubleVar(value=DEFAULTS['y0'])

        # главный фрейм
        frm = ttk.Frame(self, padding=6, style="TFrame")
        frm.grid(row=0, column=0, sticky="nsew")
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        # LEFT: контейнер с canvas (чтобы был scrollbar)
        left_container = ttk.Frame(frm, style="Card.TFrame")
        left_container.grid(row=0, column=0, sticky="nsw", padx=(0,6), pady=0)

        # canvas (фон панели)
        self.left_canvas = tk.Canvas(left_container, borderwidth=0, highlightthickness=0,
                                     width=720, height=680, bg=PANEL)
        self.left_canvas.grid(row=0, column=0, sticky="nsw")

        vsb = ttk.Scrollbar(left_container, orient="vertical", command=self.left_canvas.yview, style="Vertical.TScrollbar")
        vsb.grid(row=0, column=1, sticky="ns", padx=(2,0), pady=0)

        self.left_canvas.configure(yscrollcommand=vsb.set)

        # внутренний фрейм — сделаем обычным tk.Frame, чтобы легко настраивать bg
        self.left_inner = tk.Frame(self.left_canvas, bg=PANEL, highlightthickness=0)
        self.left_inner_id = self.left_canvas.create_window((0, 0), window=self.left_inner, anchor="nw")

        # корректируем scrollregion при изменениях
        self.left_inner.bind("<Configure>", lambda e: self.left_canvas.configure(scrollregion=self.left_canvas.bbox("all")))

        # при ресайзе канваса — подвинем ширину внутреннего фрейма
        def _on_canvas_config(event):
            canvas_width = event.width
            self.left_canvas.itemconfig(self.left_inner_id, width=canvas_width)
        self.left_canvas.bind("<Configure>", _on_canvas_config)

        # колесо мыши — прокрутка когда мышь над левым внутренним
        def _on_mousewheel(event):
            if event.delta:
                self.left_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
            else:
                if event.num == 4:
                    self.left_canvas.yview_scroll(-3, "units")
                elif event.num == 5:
                    self.left_canvas.yview_scroll(3, "units")

        def _bind_wheel(event):
            self.left_canvas.bind_all("<MouseWheel>", _on_mousewheel)
            self.left_canvas.bind_all("<Button-4>", _on_mousewheel)
            self.left_canvas.bind_all("<Button-5>", _on_mousewheel)

        def _unbind_wheel(event):
            self.left_canvas.unbind_all("<MouseWheel>")
            self.left_canvas.unbind_all("<Button-4>")
            self.left_canvas.unbind_all("<Button-5>")

        self.left_inner.bind("<Enter>", _bind_wheel)
        self.left_inner.bind("<Leave>", _unbind_wheel)

        # RIGHT панель для matplotlib
        right_panel = ttk.Frame(frm, style="TFrame")
        right_panel.grid(row=0, column=1, sticky="ne", padx=(12,0))

        # --- Заполняем left_inner контролами ---
        slider_params = [
            ('W, мм', 'W', 100.0, 10000.0, 100.0),
            ('H, мм', 'H', 100.0, 10000.0, 100.0),
            ('Wres, пикс', 'Wres', 200, 800, 1),
            ('zL, мм', 'zL', 100.0, 10000.0, 100.0),
            ('I0, Вт/ср', 'I0', 0.01, 10000.0, 0.01),
            ('R, мм', 'R', 10.0, 1000.0, 1.0),
        ]

        grid = tk.Frame(self.left_inner, bg=PANEL, padx=8, pady=8)
        grid.grid(row=0, column=0, sticky="nw")
        slider_length = 320

        # вспомогательная функция для создания меток в тёмной теме
        def dark_label(parent, text, r, c, **kw):
            lbl = tk.Label(parent, text=text, bg=PANEL, fg=FG, anchor="w")
            lbl.grid(row=r, column=c, sticky="w", padx=(0,6), pady=4)
            return lbl

        for i, (label_text, key, minv, maxv, resolution) in enumerate(slider_params):
            dark_label(grid, label_text, i, 0)
            scale = tk.Scale(grid, from_=minv, to=maxv, orient='horizontal',
                             resolution=resolution, length=slider_length, showvalue=False,
                             variable=self.vars[key], bg=PANEL2, troughcolor=PANEL, fg=FG,
                             highlightthickness=0)
            scale.grid(row=i, column=1, sticky="w")
            val_entry = ttk.Entry(grid, textvariable=self.vars[key], width=9, style="TEntry")
            # чтобы Entry имел тёмный фон — использовали TEntry стиль
            val_entry.grid(row=i, column=2, sticky="w", padx=(8,0))
            if key in ['W', 'H', 'Wres']:
                scale.bind("<ButtonRelease-1>", lambda e: self.update_square_pixels())
            else:
                scale.bind("<ButtonRelease-1>", lambda e: self.compute())
            val_entry.bind("<Return>", lambda e: self.compute())

        # Hres readonly row
        i = 2
        dark_label(grid, 'Hres, пикс', i, 0)
        hres_entry = ttk.Entry(grid, textvariable=self.vars['Hres'], width=9, style="TEntry", state="readonly")
        hres_entry.grid(row=i, column=2, sticky="w", padx=(8,0))

        # лог шкала
        chk = ttk.Checkbutton(grid, text="Логарифмическая шкала", variable=tk.BooleanVar(value=False),
                              command=lambda: None, style="TCheckbutton")  # placeholder, ниже переопределим
        # вместо временного чекбокса создадим свой с привязкой к self.log_scale
        self.log_scale = tk.BooleanVar(value=False)
        chk.destroy()
        chk = ttk.Checkbutton(grid, text="Логарифмическая шкала", variable=self.log_scale, command=self.compute,
                              style="TCheckbutton")
        chk.grid(row=len(slider_params), column=0, columnspan=3, sticky="w", pady=(10, 10))

        dark_label(grid, "Положение источника (xL, yL)", len(slider_params)+1, 0)

        # поля xL, yL
        lbl_xL = dark_label(grid, "xL (мм)", len(slider_params)+3, 0)
        entry_xL = ttk.Entry(grid, textvariable=self.vars['xL'], width=10, style="TEntry")
        entry_xL.bind("<FocusOut>", lambda e: self.round_xL())
        entry_xL.bind("<Return>", lambda e: (self.round_xL(), self.compute()))
        entry_xL.grid(row=len(slider_params)+3, column=1, sticky="w", pady=(0,6))

        lbl_yL = dark_label(grid, "yL (мм)", len(slider_params)+3, 2)
        entry_yL = ttk.Entry(grid, textvariable=self.vars['yL'], width=10, style="TEntry")
        entry_yL.bind("<FocusOut>", lambda e: self.round_yL())
        entry_yL.bind("<Return>", lambda e: (self.round_yL(), self.compute()))
        entry_yL.grid(row=len(slider_params)+3, column=3, sticky="w", pady=(0,6))

        # кнопки
        btns = tk.Frame(self.left_inner, bg=PANEL)
        btns.grid(row=1, column=0, pady=(10, 10), sticky="w", padx=(8,8))
        b1 = ttk.Button(btns, text="Пересчитать", command=self.compute, width=14, style="TButton")
        b2 = ttk.Button(btns, text="Сохранить PNG", command=self.save_outputs, width=14, style="TButton")
        b3 = ttk.Button(btns, text="График сечения", command=self.show_section, width=14, style="TButton")
        b4 = ttk.Button(btns, text="Сечение Y", command=self.show_section_y, width=14, style="TButton")
        b1.grid(row=0, column=0, padx=(0,6))
        b2.grid(row=0, column=1, padx=(0,6))
        b3.grid(row=0, column=2, padx=(0,6))
        b4.grid(row=0, column=3, padx=(0,6))

        # превью + точки (темный фон)
        self.preview = tk.Label(self.left_inner, bg=PANEL, bd=0)
        self.preview.grid(row=2, column=0, sticky="w", pady=(8, 0), padx=(8,8))

        self.points_frame = tk.Frame(self.left_inner, bg=PANEL)
        self.points_frame.grid(row=3, column=0, sticky="nw", pady=(8,0), padx=(8,8))

        # ----------------- Настройка matplotlib под тёмную тему -----------------
        plt.rcParams.update({
            'figure.facecolor': BG,
            'axes.facecolor': PANEL,
            'axes.edgecolor': FG,
            'axes.labelcolor': FG,
            'xtick.color': FG,
            'ytick.color': FG,
            'text.color': FG,
            'legend.facecolor': PANEL2,
            'legend.edgecolor': FG,
            'savefig.facecolor': BG,
            'grid.color': "#2a3440",
        })

        # figure справа
        fig, ax = plt.subplots(figsize=(3.6, 3.6))
        fig.patch.set_facecolor(BG)
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=FG)
        ax.xaxis.label.set_color(FG)
        ax.yaxis.label.set_color(FG)
        ax.title.set_color(FG)
        self.fig = fig
        self.ax = ax

        self.canvas = FigureCanvasTkAgg(fig, master=right_panel)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="ne")
        self.ax.set_xlabel("X, мм")
        self.ax.set_ylabel("Y, мм")
        self.ax.set_title("Вид: положение источника")
        self.ax.set_aspect('equal', 'box')
        self.ax.set_xlim(-10000, 10000)
        self.ax.set_ylim(-10000, 10000)
        self.scatter = None

        self.canvas.mpl_connect('button_press_event', self.on_plot_click)

        # начальные вычисления
        self.last = None
        self.compute()

    # ----------------- Логика UI -----------------
    def update_square_pixels(self, *args):
        try:
            W = float(self.vars["W"].get())
            H = float(self.vars["H"].get())
            Wres = int(self.vars["Wres"].get())
            ratio = H / W
            Hres = max(1, int(round(Wres * ratio)))
            self.vars["Hres"].set(Hres)
        except:
            pass
        self.compute()

    def _draw_light_source_point(self):
        xL = self.vars['xL'].get()
        yL = self.vars['yL'].get()
        self.ax.cla()
        # сохранить темную стилистику при перерисовке
        self.ax.set_facecolor(PANEL)
        self.ax.set_xlabel("X, мм")
        self.ax.set_ylabel("Y, мм")
        self.ax.set_title("Вид: положение источника")
        self.ax.set_aspect('equal', 'box')
        self.ax.set_xlim(-10000, 10000)
        self.ax.set_ylim(-10000, 10000)
        self.scatter = self.ax.scatter([xL], [yL], c='red', s=40, label='Источник света')
        lg = self.ax.legend(loc='upper right', fontsize='small')
        if lg:
            lg.get_frame().set_facecolor(PANEL2)
            lg.get_frame().set_edgecolor(FG)
            for text in lg.get_texts():
                text.set_color(FG)
        self.canvas.draw_idle()

    def on_plot_click(self, event):
        if event.inaxes != self.ax:
            return
        x = max(min(event.xdata, 10000), -10000)
        y = max(min(event.ydata, 10000), -10000)
        self.vars['xL'].set(x)
        self.vars['yL'].set(y)
        self._draw_light_source_point()
        self.compute()

    def read_params(self):
        try:
            vals = {}
            vals["W"] = float(self.vars["W"].get())
            vals["H"] = float(self.vars["H"].get())
            vals["Wres"] = int(self.vars["Wres"].get())
            vals["Hres"] = int(self.vars["Hres"].get())
            vals["xL"] = float(self.vars["xL"].get())
            vals["yL"] = float(self.vars["yL"].get())
            vals["zL"] = float(self.vars["zL"].get())
            vals["I0"] = float(self.vars["I0"].get())
            vals["x0"] = float(self.vars["x0"].get())
            vals["y0"] = float(self.vars["y0"].get())
            vals["R"] = float(self.vars["R"].get())
        except ValueError as e:
            messagebox.showerror("Ошибка ввода", f"Проверь параметры: {e}")
            return None
        vals["W"] = min(max(vals["W"], 100.0), 10000.0)
        vals["H"] = min(max(vals["H"], 100.0), 10000.0)
        vals["Wres"] = int(min(max(vals["Wres"], 200), 800))
        vals["Hres"] = int(min(max(vals["Hres"], 200), 800))
        vals["xL"] = min(max(vals["xL"], -10000.0), 10000.0)
        vals["yL"] = min(max(vals["yL"], -10000.0), 10000.0)
        vals["zL"] = min(max(vals["zL"], 100.0), 10000.0)
        vals["I0"] = min(max(vals["I0"], 0.01), 10000.0)
        return vals

    def compute(self):
        vals = self.read_params()
        if not vals:
            return
        vals["xL"] = int(vals["xL"])
        vals["yL"] = int(vals["yL"])
        # enforce square pixels
        ratio = vals["H"] / vals["W"]
        vals["Hres"] = max(1, int(round(vals["Wres"] * ratio)))
        self.vars["Hres"].set(vals["Hres"])
        self.vars['xL'].set(vals["xL"])
        self.vars['yL'].set(vals["yL"])
        self.last = compute_fields(**vals)

        # build base image array
        if self.log_scale.get():
            E_disp = np.log1p(self.last["E"])
            E_disp = E_disp / np.max(E_disp)
            arr = (E_disp * 255).astype(np.uint8)
        else:
            arr = self.last["img"]

        pil = Image.fromarray(arr, mode="L")

        # уменьшено максимальное превью
        max_w, max_h = 480, 480
        scale_w = max_w / pil.width
        scale_h = max_h / pil.height
        scale = min(1.0, scale_w, scale_h)
        new_size = (max(1, int(pil.width * scale)), max(1, int(pil.height * scale)))
        pil = pil.resize(new_size, Image.NEAREST)

        img_preview = pil
        self.tk_img = ImageTk.PhotoImage(img_preview.convert("RGB"))

        # поместим превью на темный фон
        self.preview.configure(image=self.tk_img, bg=PANEL)
        self.title(f"Lambert Plane — max {self.last['stats']['E_max']:.5g}, mean {self.last['stats']['E_mean']:.5g}")
        self._draw_light_source_point()

        for widget in self.points_frame.winfo_children():
            widget.destroy()
        for i, (name, (px, py, E_val)) in enumerate(self.last["five"].items()):
            text = f"{name}: (x={px:.1f}, y={py:.1f})  E={format_sci(E_val)}"
            lbl = tk.Label(self.points_frame, text=text, bg=PANEL, fg=FG, anchor="w")
            lbl.grid(row=i, column=0, sticky="w", pady=2)
        stats = self.last["stats"]
        base = len(self.last["five"])

        l1 = tk.Label(self.points_frame, text=f"E_max = {format_sci(stats['E_max'])}", bg=PANEL, fg=FG, anchor="w")
        l2 = tk.Label(self.points_frame, text=f"E_min = {format_sci(stats['E_min'])}", bg=PANEL, fg=FG, anchor="w")
        l3 = tk.Label(self.points_frame, text=f"E_mean = {format_sci(stats['E_mean'])}", bg=PANEL, fg=FG, anchor="w")
        l1.grid(row=base, column=0, sticky="w", pady=2)
        l2.grid(row=base + 1, column=0, sticky="w", pady=2)
        l3.grid(row=base + 2, column=0, sticky="w", pady=2)

    def show_section(self):
        if not self.last:
            return

        ys = self.last["ys"]
        x0 = self.last["params"]["x0"]
        y0 = self.last["params"]["y0"]
        xs = self.last["xs"]

        row_idx = int(np.argmin(np.abs(ys - y0)))

        # освещённость вдоль строки
        section = self.last["E"][row_idx, :]

        # маска круга вдоль строки
        mask_row = self.last["mask"][row_idx, :]
        section = np.where(mask_row, section, 0.0)

        plt.figure(facecolor=BG)
        ax = plt.gca()
        ax.set_facecolor(PANEL)
        plt.plot(xs, section)
        plt.xlabel("X, мм (сечение через y = y0)")
        plt.ylabel("E, Вт/м²")
        plt.title("Сечение по оси X через центр круга")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def show_section_y(self):
        if not self.last:
            return
        xs = self.last["xs"]
        x0 = self.last["params"]["x0"]
        y0 = self.last["params"]["y0"]
        ys = self.last["ys"]
        col_idx = int(np.argmin(np.abs(xs - x0)))
        section = self.last["E"][:, col_idx]
        mask_col = self.last["mask"][:, col_idx]
        section = np.where(mask_col, section, 0.0)
        plt.figure(facecolor=BG)
        ax = plt.gca()
        ax.set_facecolor(PANEL)
        plt.plot(ys, section)
        plt.xlabel("Y, мм (сечение через x = x0)")
        plt.ylabel("E, Вт/м²")
        plt.title("Сечение по оси Y через центр круга")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def round_xL(self): #округление
        try:
            self.vars['xL'].set(int(float(self.vars['xL'].get())))
        except:
            pass

    def round_yL(self):
        try:
            self.vars['yL'].set(int(float(self.vars['yL'].get())))
        except:
            pass

    def save_outputs(self):
        if not self.last:
            return
        outbase = filedialog.asksaveasfilename(
            title="Базовое имя файла (без расширения)",
            defaultextension=".png",
            filetypes=[("PNG", "*.png")]
        )
        if not outbase:
            return
        Image.fromarray(self.last["img"]).save(outbase)

        ys = self.last["ys"]
        y0 = self.last["params"]["y0"]
        xs = self.last["xs"]
        row_idx = int(np.argmin(np.abs(ys - y0)))
        section = self.last["E"][row_idx, :]
        plt.figure(facecolor=BG)
        ax = plt.gca()
        ax.set_facecolor(PANEL)
        plt.plot(xs, section)
        plt.xlabel("X, мм (сечение через y=y0)")
        plt.ylabel("E, Вт/м²")
        plt.title("Сечение освещённости — экспорт")
        plt.tight_layout()
        plt.savefig(outbase.rsplit(".", 1)[0] + "_section.png", dpi=180)
        plt.close()
        messagebox.showinfo("Готово", "PNG и график сечения сохранены.")

if __name__ == "__main__":
    App().mainloop()
