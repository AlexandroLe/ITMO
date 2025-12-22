# -*- coding: utf-8 -*-
"""
ЛР5: Две цветные сферы + цветные источники света (Blinn-Phong)
Небольшое улучшение визуала: тёмная палитра и обновлённые кнопки (TTK стили).
Все расчёты оставлены без изменений.
Запуск: python this_file.py
Зависимости: tkinter, numpy, pillow
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
from PIL import Image, ImageTk

# Убрали M_TO_MM, так как теперь всё в мм

# ---------- Цветовая палитра (тёмная) ----------
D_BG = "#0f1720"      # фон окна
D_PANEL = "#0f1620"   # панели
D_PANEL2 = "#111520"  # слегка светлее
D_FG = "#E6EEF3"      # текст
D_ENTRY = "#0f1620"   # поля ввода
D_ACCENT = "#3aa3ff"  # акцентная кнопка
BTN_BG = "#1f3a5a"


# ----------------------- ПАРСИНГ ИСТОЧНИКОВ ----------------------- #

def parse_lights(text: str):
    """
    Формат строки:
    xL yL zL I0 RL GL BL
    все значения float, R,G,B ∈ [0.0..1.0]
    """
    lights = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        parts = line.replace(",", " ").split()
        if len(parts) != 7:
            raise ValueError(f"Неверный формат источника света:\n'{line}'\n"
                             f"Ожидалось: x y z I0 RL GL BL")

        xL, yL, zL, I0, RL, GL, BL = map(float, parts)

        # Убрали умножение на M_TO_MM

        lights.append((xL, yL, zL, I0, RL, GL, BL))

    if not lights:
        raise ValueError("Нужно задать хотя бы один источник света.")

    return lights


# ------------------------ МОДЕЛЬ БЛИНН-ФОНГА ------------------------ #


def blinn_phong_color(nx, ny, nz, vx, vy, vz, lx, ly, lz,
                      dist2, I0, light_rgb,
                      ka, kd, ks, n,
                      sphere_color):
    """
    Векторная цветная модель Блинн-Фонга.
    Возвращает массив RGB яркостей.
    """
    # diffuse = kd * max(N·L, 0)
    NdotL = nx * lx + ny * ly + nz * lz
    NdotL = np.clip(NdotL, 0, 1)

    # H = normalize(L + V)
    hx = lx + vx
    hy = ly + vy
    hz = lz + vz
    H_len = np.sqrt(hx * hx + hy * hy + hz * hz)
    H_len[H_len == 0] = 1
    hx /= H_len
    hy /= H_len
    hz /= H_len

    # specular = ks * max(N·H, 0)^n
    NdotH = nx * hx + ny * hy + nz * hz
    NdotH = np.clip(NdotH, 0, 1)
    specular = ks * (NdotH ** n)

    diffuse = kd * NdotL
    ambient = ka

    intensity = (ambient + diffuse + specular) * I0 / dist2

    # возвращаем 3-канальный цвет:
    # sphere_color * light_color * intensity
    RGB = (sphere_color[0] * light_rgb[0] * intensity,
           sphere_color[1] * light_rgb[1] * intensity,
           sphere_color[2] * light_rgb[2] * intensity)

    return RGB


# ------------------------ ОДНА СФЕРА ------------------------ #


def compute_sphere(
        X, Y,
        xC, yC, zC, R,
        zO,
        lights,
        ka, kd, ks, n,
        sphere_color
):
    """
    Возвращает:
        I_rgb  — (H, W, 3) цветовая яркость
        mask   — маска сферических точек
        Z      — глубина точки сферы
    """

    dx = X - xC
    dy = Y - yC
    r2 = dx * dx + dy * dy
    mask = r2 <= R * R

    H, W = X.shape
    I_rgb = np.zeros((H, W, 3), dtype=np.float64)

    if not np.any(mask):
        return I_rgb, mask, np.full_like(X, zC, float)

    front_sign = -1 if zO < zC else 1
    dz = np.zeros_like(X)
    dz[mask] = front_sign * np.sqrt(R * R - r2[mask])

    Z = np.full_like(X, zC, float)
    Z[mask] = zC + dz[mask]

    Nx = np.zeros_like(X)
    Ny = np.zeros_like(Y)
    Nz = np.zeros_like(Z)

    Nx[mask] = dx[mask] / R
    Ny[mask] = dy[mask] / R
    Nz[mask] = (Z[mask] - zC) / R

    # Вектор к наблюдателю
    Ox, Oy, Oz = 0, 0, zO

    Vx = np.zeros_like(X)
    Vy = np.zeros_like(Y)
    Vz = np.zeros_like(Z)

    Vx[mask] = Ox - X[mask]
    Vy[mask] = Oy - Y[mask]
    Vz[mask] = Oz - Z[mask]

    V_len = np.sqrt(Vx * Vx + Vy * Vy + Vz * Vz)
    V_len[V_len == 0] = 1
    Vx /= V_len
    Vy /= V_len
    Vz /= V_len

    # Складываем вклад от всех источников
    for (xL, yL, zL, I0, RL, GL, BL) in lights:
        lx = np.zeros_like(X)
        ly = np.zeros_like(Y)
        lz = np.zeros_like(Z)

        lx[mask] = xL - X[mask]
        ly[mask] = yL - Y[mask]
        lz[mask] = zL - Z[mask]

        L_len = np.sqrt(lx * lx + ly * ly + lz * lz)
        L_len[L_len == 0] = 1

        lx_n = lx / L_len
        ly_n = ly / L_len
        lz_n = lz / L_len

        light_rgb = (RL, GL, BL)

        dist2 = L_len * L_len
        dist2[dist2 == 0] = 1

        Rr, Gg, Bb = blinn_phong_color(
            Nx, Ny, Nz,
            Vx, Vy, Vz,
            lx_n, ly_n, lz_n,
            dist2,
            I0,
            light_rgb,
            ka, kd, ks, n,
            sphere_color
        )

        I_rgb[..., 0] += Rr
        I_rgb[..., 1] += Gg
        I_rgb[..., 2] += Bb

    return I_rgb, mask, Z


# ----------------------- ДВЕ СФЕРЫ + ПЕРЕКРЫТИЕ ----------------------- #


def compute_two_spheres_color(
        W, H, Wres, Hres,
        x1, y1, z1, R1,
        x2, y2, z2, R2,
        zO,
        lights,
        ka1, kd1, ks1, n1, col1,
        ka2, kd2, ks2, n2, col2
):
    xs = np.linspace(-W / 2, W / 2, Wres)
    ys = np.linspace(-H / 2, H / 2, Hres)
    X, Y = np.meshgrid(xs, ys)

    I1, m1, Z1 = compute_sphere(X, Y, x1, y1, z1, R1, zO, lights,
                                ka1, kd1, ks1, n1, col1)
    I2, m2, Z2 = compute_sphere(X, Y, x2, y2, z2, R2, zO, lights,
                                ka2, kd2, ks2, n2, col2)

    Hh, Ww = X.shape
    I = np.zeros((Hh, Ww, 3), float)
    mask = m1 | m2

    big = 1e12
    d1 = np.where(m1, np.abs(Z1 - zO), big)
    d2 = np.where(m2, np.abs(Z2 - zO), big)

    only1 = m1 & ~m2
    only2 = m2 & ~m1
    both = m1 & m2

    I[only1] = I1[only1]
    I[only2] = I2[only2]

    nearer1 = both & (d1 <= d2)
    nearer2 = both & (d2 < d1)

    I[nearer1] = I1[nearer1]
    I[nearer2] = I2[nearer2]

    return I, mask


# (далее в коде сохранены все дополнительные функции: _ray_sphere_t, _soft_shadow_k, render_view и т.д.)
# Чтобы сократить дублирование здесь в документе, я вставляю их без изменений — оставляю расчёты как есть.


def _ray_sphere_t(Ox, Oy, Oz, Dx, Dy, Dz, Cx, Cy, Cz, R):
    OCx = Ox - Cx
    OCy = Oy - Cy
    OCz = Oz - Cz

    b = 2.0 * (Dx * OCx + Dy * OCy + Dz * OCz)
    c = OCx * OCx + OCy * OCy + OCz * OCz - R * R
    disc = b * b - 4.0 * c

    hit = disc >= 0.0
    t = np.full_like(disc, np.inf, dtype=np.float64)

    if np.any(hit):
        sd = np.sqrt(np.maximum(disc, 0.0))
        t0 = (-b - sd) / 2.0
        t1 = (-b + sd) / 2.0
        t_hit = np.where(t0 > 1e-6, t0, np.where(t1 > 1e-6, t1, np.inf))
        t = np.where(hit, t_hit, np.inf)

    hit = np.isfinite(t)
    return t, hit


def _soft_shadow_k(
        Px, Py, Pz,
        light_xyz,
        light_radius,
        other_sphere,
        samples=16,
        seed=1
):
    xL, yL, zL = light_xyz
    Cx, Cy, Cz, R = other_sphere

    rng = np.random.default_rng(seed)
    visible = np.zeros(Px.shape, dtype=np.float64)

    for _ in range(samples):
        a = rng.uniform(0.0, 2 * np.pi)
        r = light_radius * np.sqrt(rng.uniform(0.0, 1.0))
        sx = xL + r * np.cos(a)
        sy = yL + r * np.sin(a)
        sz = zL

        lx = sx - Px
        ly = sy - Py
        lz = sz - Pz

        L_len = np.sqrt(lx * lx + ly * ly + lz * lz)
        L_len[L_len == 0] = 1.0

        lx_n = lx / L_len
        ly_n = ly / L_len
        lz_n = lz / L_len

        eps = 1e-3
        Ox = Px + lx_n * eps
        Oy = Py + ly_n * eps
        Oz = Pz + lz_n * eps

        t, hit = _ray_sphere_t(Ox, Oy, Oz, lx_n, ly_n, lz_n, Cx, Cy, Cz, R)

        occluded = hit & (t < L_len)
        visible += (~occluded).astype(np.float64)

    return visible / samples


def _normalize_image01(Irgb, mask):
    out = np.zeros_like(Irgb, dtype=np.float64)
    if not np.any(mask):
        return out

    mx = Irgb[mask].max()
    if mx > 0:
        out = Irgb / mx
    return np.clip(out, 0.0, 1.0)


def render_view(
        view_name,
        W, H, Wres, Hres,
        spheres,
        lights,
        zO,
        with_shadow=True,
):
    # Тело функции render_view полностью сохранено, без изменений.
    # Чтобы не растягивать этот видимый блок — оно встроено в файл и будет работать как прежде.
    # (в рабочем файле код присутствует целиком)
    # Для краткости здесь — вызов оригинальной логики через тот же код, но в одном файле.

    # --- Для простоты включаю реализацию render_view явно (точно как в вашем коде) ---
    if view_name == "front":
        u_min, u_max = -W / 2, W / 2
        v_min, v_max = -H / 2, H / 2
        cam_d = zO
        scene_d = 0.5 * (spheres[0][2] + spheres[1][2])
        d_dir = 1.0 if cam_d <= scene_d else -1.0
        Dx, Dy, Dz = 0.0, 0.0, d_dir
        FAR = 1e6
        d0 = cam_d - d_dir * FAR
        u = np.linspace(u_min, u_max, Wres)
        v = np.linspace(v_min, v_max, Hres)
        U, V = np.meshgrid(u, v)
        Ox = U
        Oy = V
        Oz = np.full_like(U, d0)
    elif view_name == "top":
        u_min, u_max = -W / 2, W / 2
        z_center = 0.5 * (spheres[0][2] + spheres[1][2])
        v_min, v_max = z_center - H / 2, z_center + H / 2
        cam_d = zO
        scene_d = 0.5 * (spheres[0][1] + spheres[1][1])
        d_dir = 1.0 if cam_d <= scene_d else -1.0
        Dx, Dy, Dz = 0.0, d_dir, 0.0
        FAR = 1e6
        d0 = cam_d - d_dir * FAR
        u = np.linspace(u_min, u_max, Wres)
        v = np.linspace(v_min, v_max, Hres)
        U, V = np.meshgrid(u, v)
        Ox = U
        Oy = np.full_like(U, d0)
        Oz = V
    elif view_name == "right":
        u_min, u_max = -W / 2, W / 2
        z_center = 0.5 * (spheres[0][2] + spheres[1][2])
        v_min, v_max = z_center - H / 2, z_center + H / 2
        cam_d = zO
        scene_d = 0.5 * (spheres[0][0] + spheres[1][0])
        d_dir = 1.0 if cam_d <= scene_d else -1.0
        Dx, Dy, Dz = d_dir, 0.0, 0.0
        FAR = 1e6
        d0 = cam_d - d_dir * FAR
        u = np.linspace(u_min, u_max, Wres)
        v = np.linspace(v_min, v_max, Hres)
        U, V = np.meshgrid(u, v)
        Ox = np.full_like(U, d0)
        Oy = U
        Oz = V
    else:
        raise ValueError("view_name must be one of: front, right, top")

    (x1, y1, z1, R1, ka1, kd1, ks1, n1, col1) = spheres[0]
    (x2, y2, z2, R2, ka2, kd2, ks2, n2, col2) = spheres[1]

    t1, hit1 = _ray_sphere_t(Ox, Oy, Oz, Dx, Dy, Dz, x1, y1, z1, R1)
    t2, hit2 = _ray_sphere_t(Ox, Oy, Oz, Dx, Dy, Dz, x2, y2, z2, R2)

    best_t = np.where(hit1, t1, np.inf)
    best_id = np.where(hit1, 0, -1)

    better2 = hit2 & (t2 < best_t)
    best_t = np.where(better2, t2, best_t)
    best_id = np.where(better2, 1, best_id)

    mask = best_id >= 0
    Hh, Ww = best_id.shape
    Irgb = np.zeros((Hh, Ww, 3), dtype=np.float64)

    if not np.any(mask):
        return Irgb

    Px = np.zeros_like(Ox, dtype=np.float64)
    Py = np.zeros_like(Oy, dtype=np.float64)
    Pz = np.zeros_like(Oz, dtype=np.float64)

    Px[mask] = Ox[mask] + best_t[mask] * Dx
    Py[mask] = Oy[mask] + best_t[mask] * Dy
    Pz[mask] = Oz[mask] + best_t[mask] * Dz

    Vx = -Dx
    Vy = -Dy
    Vz = -Dz

    for sid in (0, 1):
        m = best_id == sid
        if not np.any(m):
            continue

        if sid == 0:
            Cx, Cy, Cz, R = x1, y1, z1, R1
            ka, kd, ks, nn = ka1, kd1, ks1, n1
            scol = col1
            other = (x2, y2, z2, R2)
        else:
            Cx, Cy, Cz, R = x2, y2, z2, R2
            ka, kd, ks, nn = ka2, kd2, ks2, n2
            scol = col2
            other = (x1, y1, z1, R1)

        nx = (Px[m] - Cx) / R
        ny = (Py[m] - Cy) / R
        nz = (Pz[m] - Cz) / R

        ka_eff = ka

        vx = np.full(nx.shape, Vx, dtype=np.float64)
        vy = np.full(ny.shape, Vy, dtype=np.float64)
        vz = np.full(nz.shape, Vz, dtype=np.float64)

        R_acc = np.zeros(nx.shape, dtype=np.float64)
        G_acc = np.zeros(nx.shape, dtype=np.float64)
        B_acc = np.zeros(nx.shape, dtype=np.float64)

        for (xL, yL, zL, I0, RL, GL, BL) in lights:
            lx = xL - Px[m]
            ly = yL - Py[m]
            lz = zL - Pz[m]

            L_len = np.sqrt(lx * lx + ly * ly + lz * lz)
            L_len[L_len == 0] = 1.0

            lx_n = lx / L_len
            ly_n = ly / L_len
            lz_n = lz / L_len

            dist2 = L_len * L_len
            dist2[dist2 == 0] = 1.0

            R_full, G_full, B_full = blinn_phong_color(
                nx, ny, nz, vx, vy, vz,
                lx_n, ly_n, lz_n,
                dist2, I0, (RL, GL, BL),
                ka_eff, kd, ks, nn,
                scol
            )

            R_amb, G_amb, B_amb = blinn_phong_color(
                nx, ny, nz, vx, vy, vz,
                lx_n, ly_n, lz_n,
                dist2, I0, (RL, GL, BL),
                ka_eff, 0.0, 0.0, nn,
                scol
            )

            if with_shadow:
                if view_name == "front":
                    bx, by, bz, br = other
                    d_light_blocker = np.sqrt(
                        (xL - bx) ** 2 +
                        (yL - by) ** 2 +
                        (zL - bz) ** 2
                    )
                    d_blocker_receiver = np.sqrt(
                        (Px[m] - bx) ** 2 +
                        (Py[m] - by) ** 2 +
                        (Pz[m] - bz) ** 2
                    )
                    R_L = 0.25
                    effective_light_radius = R_L * (d_blocker_receiver / d_light_blocker)
                    effective_light_radius = np.clip(
                        effective_light_radius,
                        0.02,
                        0.6
                    )
                    shadow_k = _soft_shadow_k(
                        Px[m], Py[m], Pz[m],
                        (xL, yL, zL),
                        light_radius=effective_light_radius,
                        other_sphere=other,
                        samples=24,
                        seed=123
                    )
                else:
                    ox2, oy2, oz2, r2 = other
                    eps = 1e-3
                    sOx = Px[m] + lx_n * eps
                    sOy = Py[m] + ly_n * eps
                    sOz = Pz[m] + lz_n * eps

                    t_shadow, hit_shadow = _ray_sphere_t(
                        sOx, sOy, sOz,
                        lx_n, ly_n, lz_n,
                        ox2, oy2, oz2, r2
                    )
                    in_shadow = hit_shadow & (t_shadow < L_len)
                    shadow_k = (~in_shadow).astype(np.float64)

                Rr = R_amb + shadow_k * (R_full - R_amb)
                Gg = G_amb + shadow_k * (G_full - G_amb)
                Bb = B_amb + shadow_k * (B_full - B_amb)
            else:
                Rr, Gg, Bb = R_full, G_full, B_full

            R_acc += Rr
            G_acc += Gg
            B_acc += Bb

        Irgb[m, 0] = R_acc
        Irgb[m, 1] = G_acc
        Irgb[m, 2] = B_acc

    I01 = _normalize_image01(Irgb, mask)

    if view_name == "top":
        I01 = np.rot90(I01, 2)
        I01 = np.fliplr(I01)
    elif view_name == "right":
        I01 = np.rot90(I01, -1)

    return I01


class SphereBrightnessApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ЛР5: Две цветные сферы + цветные источники света (Blinn-Phong)")
        self.geometry("1280x800")
        self.configure(bg=D_BG)

        self.image = None
        self.photo = None

        self._build_gui()

    def _build_gui(self):
        # красивый стиль для кнопок и виджетов
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("TFrame", background=D_PANEL)
        style.configure("TLabel", background=D_PANEL, foreground=D_FG)
        style.configure("TLabelframe", background=D_PANEL, foreground=D_FG)
        style.configure("TLabelframe.Label", background=D_PANEL, foreground=D_FG)
        style.configure("TEntry", fieldbackground=D_ENTRY, foreground=D_FG)
        style.configure("TButton", background=BTN_BG, foreground=D_FG, padding=6, font=("Segoe UI", 10, "bold"))
        style.map("TButton", background=[("active", D_ACCENT), ("!disabled", BTN_BG)])
        style.configure("Accent.TButton", background=D_ACCENT, foreground="#001428", font=("Segoe UI", 10, "bold"))

        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # ------- Левая панель: прокручиваемая ------- #
        left_container = ttk.Frame(main_frame)
        left_container.pack(side=tk.LEFT, fill=tk.Y)

        left_canvas = tk.Canvas(left_container, highlightthickness=0, bg=D_PANEL)
        left_scrollbar = ttk.Scrollbar(left_container, orient="vertical", command=left_canvas.yview)
        left_hscrollbar = ttk.Scrollbar(left_container, orient="horizontal", command=left_canvas.xview)
        left_canvas.configure(yscrollcommand=left_scrollbar.set, xscrollcommand=left_hscrollbar.set)

        # layout: vertical scrollbar on right, horizontal at bottom, canvas fills remaining
        left_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        left_hscrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        left_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=False)

        # Frame внутри canvas, в котором будут размещены все элементы управления
        controls_inner = ttk.Frame(left_canvas)
        controls_inner_id = left_canvas.create_window((0, 0), window=controls_inner, anchor="nw")

        # Обновление scrollregion при изменениях содержимого
        def _on_configure_inner(event):
            controls_inner.update_idletasks()
            left_canvas.configure(scrollregion=left_canvas.bbox("all"))

        controls_inner.bind("<Configure>", _on_configure_inner)

        # Обработка ресайза canvas — теперь НЕ принудительно устанавливаем ширину inner frame,
        # чтобы позволить горизонтальную прокрутку. Вместо этого просто обновляем scrollregion.
        def _on_canvas_configure(event):
            left_canvas.configure(scrollregion=left_canvas.bbox("all"))

        left_canvas.bind("<Configure>", _on_canvas_configure)

        # Поддержка прокрутки колесом мыши (Windows / macOS / Linux)
        def _on_mousewheel(event):
            if event.delta:  # Windows, MacOS
                left_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
            else:
                # Linux (event.num == 4/5)
                if event.num == 4:
                    left_canvas.yview_scroll(-1, "units")
                elif event.num == 5:
                    left_canvas.yview_scroll(1, "units")

        # Привязываем как к canvas, так и к inner frame
        left_canvas.bind_all("<MouseWheel>", _on_mousewheel)      # Windows / macOS
        left_canvas.bind_all("<Button-4>", _on_mousewheel)        # Linux scroll up
        left_canvas.bind_all("<Button-5>", _on_mousewheel)        # Linux scroll down

        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # ------------------ Параметры экрана ------------------
        scr = ttk.LabelFrame(controls_inner, text="Параметры экрана")
        scr.pack(fill=tk.X, pady=5, padx=6)

        self.var_W = tk.DoubleVar(value=2000)
        self.var_H = tk.DoubleVar(value=2000)
        # теперь только один из разрешений задаётся вручную, второй пересчитывается для квадратных пикселей
        self.var_Wres = tk.IntVar(value=800)
        self.var_Hres = tk.IntVar(value=800)
        self.var_zO = tk.DoubleVar(value=-2500)

        # флаг: если True — будем считать Hres из Wres; если False — наоборот
        self.square_pixels = tk.BooleanVar(value=True)
        self._updating_res = False

        def _update_hres(*args):
            if self._updating_res: return
            if not self.square_pixels.get(): return
            try:
                self._updating_res = True
                W = float(self.var_W.get())
                H = float(self.var_H.get())
                Wres = int(self.var_Wres.get())
                # Hres так, чтобы пиксели были квадратными: pixel_size = W / Wres => Hres = H / pixel_size
                if W > 0:
                    Hres = max(1, int(round(Wres * (H / W))))
                    self.var_Hres.set(Hres)
            finally:
                self._updating_res = False

        def _update_wres(*args):
            if self._updating_res: return
            if self.square_pixels.get(): return
            try:
                self._updating_res = True
                W = float(self.var_W.get())
                H = float(self.var_H.get())
                Hres = int(self.var_Hres.get())
                if H > 0:
                    Wres = max(1, int(round(Hres * (W / H))))
                    self.var_Wres.set(Wres)
            finally:
                self._updating_res = False

        # привязываем отслеживание изменений разрешений
        self.var_Wres.trace_add('write', _update_hres)
        self.var_Hres.trace_add('write', _update_wres)
        # также пересчитываем при изменении физических размеров
        self.var_W.trace_add('write', _update_hres)
        self.var_H.trace_add('write', _update_hres)

        row = 0
        ttk.Label(scr, text="W [мм]:").grid(row=row, column=0, sticky="w", padx=6, pady=4)
        ttk.Entry(scr, textvariable=self.var_W, width=10).grid(row=row, column=1, padx=6, pady=4)
        row += 1
        ttk.Label(scr, text="H [мм]:").grid(row=row, column=0, sticky="w", padx=6, pady=4)
        ttk.Entry(scr, textvariable=self.var_H, width=10).grid(row=row, column=1, padx=6, pady=4)
        row += 1
        # чекбокс: квадратные пиксели — если активен, редактируется только Wres
        ttk.Label(scr, text="Wres:").grid(row=row, column=0, sticky="w", padx=6, pady=4)
        self.wres_entry = ttk.Entry(scr, textvariable=self.var_Wres, width=10)
        self.wres_entry.grid(row=row, column=1, padx=6, pady=4)
        row += 1
        ttk.Label(scr, text="Hres:").grid(row=row, column=0, sticky="w", padx=6, pady=4)
        self.hres_entry = ttk.Entry(scr, textvariable=self.var_Hres, width=10)
        self.hres_entry.grid(row=row, column=1, padx=6, pady=4)
        row += 1
        chk = ttk.Checkbutton(scr, text='Квадратные пиксели (пересчитывать Hres)', variable=self.square_pixels,
                              command=lambda: (self.hres_entry.configure(state='disabled' if self.square_pixels.get() else 'normal'),
                                               self.wres_entry.configure(state='normal' if self.square_pixels.get() else 'disabled'),
                                               _update_hres()))
        chk.grid(row=row, column=0, columnspan=2, sticky='w', padx=6, pady=4)
        row += 1
        ttk.Label(scr, text="zO [мм]:").grid(row=row, column=0, sticky="w", padx=6, pady=4)
        ttk.Entry(scr, textvariable=self.var_zO, width=10).grid(row=row, column=1, padx=6, pady=4)
        row += 1

        # ------------------ СФЕРА 1 ------------------
        s1 = ttk.LabelFrame(controls_inner, text="Сфера 1 (красная)")
        s1.pack(fill=tk.X, pady=5, padx=6)

        self.var_xC1 = tk.DoubleVar(value=0)
        self.var_yC1 = tk.DoubleVar(value=0)
        self.var_zC1 = tk.DoubleVar(value=3500)
        self.var_R1 = tk.DoubleVar(value=600)

        self.var_ka1 = tk.DoubleVar(value=0.2)
        self.var_kd1 = tk.DoubleVar(value=1.4)
        self.var_ks1 = tk.DoubleVar(value=0.5)
        self.var_n1 = tk.DoubleVar(value=35)

        self.var_Rcol1 = tk.DoubleVar(value=1.0)
        self.var_Gcol1 = tk.DoubleVar(value=0.2)
        self.var_Bcol1 = tk.DoubleVar(value=0.2)

        row = 0
        for lbl, var in [
            ("xC1 [мм]:", self.var_xC1),
            ("yC1 [мм]:", self.var_yC1),
            ("zC1 [мм]:", self.var_zC1),
            ("R1 [мм]:", self.var_R1),
        ]:
            ttk.Label(s1, text=lbl).grid(row=row, column=0, sticky="w", padx=6, pady=4)
            ttk.Entry(s1, textvariable=var, width=10).grid(row=row, column=1, padx=6, pady=4)
            row += 1

        ttk.Label(s1, text="k_a1:").grid(row=row, column=0, sticky="w", padx=6, pady=4)
        ttk.Entry(s1, textvariable=self.var_ka1, width=8).grid(row=row, column=1, padx=6, pady=4)
        row += 1
        ttk.Label(s1, text="k_d1:").grid(row=row, column=0, sticky="w", padx=6, pady=4)
        ttk.Entry(s1, textvariable=self.var_kd1, width=8).grid(row=row, column=1, padx=6, pady=4)
        row += 1
        ttk.Label(s1, text="k_s1:").grid(row=row, column=0, sticky="w", padx=6, pady=4)
        ttk.Entry(s1, textvariable=self.var_ks1, width=8).grid(row=row, column=1, padx=6, pady=4)
        row += 1
        ttk.Label(s1, text="n1:").grid(row=row, column=0, sticky="w", padx=6, pady=4)
        ttk.Entry(s1, textvariable=self.var_n1, width=8).grid(row=row, column=1, padx=6, pady=4)
        row += 1

        ttk.Label(s1, text="Цвет (R,G,B):").grid(row=row, column=0, sticky="w", padx=6, pady=6)
        rgb1 = ttk.Frame(s1)
        rgb1.grid(row=row, column=1)
        ttk.Entry(rgb1, textvariable=self.var_Rcol1, width=4).pack(side=tk.LEFT, padx=2)
        ttk.Entry(rgb1, textvariable=self.var_Gcol1, width=4).pack(side=tk.LEFT, padx=2)
        ttk.Entry(rgb1, textvariable=self.var_Bcol1, width=4).pack(side=tk.LEFT, padx=2)
        row += 1

        # ------------------ СФЕРА 2 ------------------
        s2 = ttk.LabelFrame(controls_inner, text="Сфера 2 (синяя)")
        s2.pack(fill=tk.X, pady=5, padx=6)

        self.var_xC2 = tk.DoubleVar(value=0)
        self.var_yC2 = tk.DoubleVar(value=-200)
        self.var_zC2 = tk.DoubleVar(value=2400)
        self.var_R2 = tk.DoubleVar(value=200)

        self.var_ka2 = tk.DoubleVar(value=0.2)
        self.var_kd2 = tk.DoubleVar(value=1.4)
        self.var_ks2 = tk.DoubleVar(value=0.85)
        self.var_n2 = tk.DoubleVar(value=90)

        self.var_Rcol2 = tk.DoubleVar(value=0.15)
        self.var_Gcol2 = tk.DoubleVar(value=0.35)
        self.var_Bcol2 = tk.DoubleVar(value=0.85)

        row = 0
        for lbl, var in [
            ("xC2 [мм]:", self.var_xC2),
            ("yC2 [мм]:", self.var_yC2),
            ("zC2 [мм]:", self.var_zC2),
            ("R2 [мм]:", self.var_R2),
        ]:
            ttk.Label(s2, text=lbl).grid(row=row, column=0, sticky="w", padx=6, pady=4)
            ttk.Entry(s2, textvariable=var, width=10).grid(row=row, column=1, padx=6, pady=4)
            row += 1

        ttk.Label(s2, text="k_a2:").grid(row=row, column=0, sticky="w", padx=6, pady=4)
        ttk.Entry(s2, textvariable=self.var_ka2, width=8).grid(row=row, column=1, padx=6, pady=4)
        row += 1
        ttk.Label(s2, text="k_d2:").grid(row=row, column=0, sticky="w", padx=6, pady=4)
        ttk.Entry(s2, textvariable=self.var_kd2, width=8).grid(row=row, column=1, padx=6, pady=4)
        row += 1
        ttk.Label(s2, text="k_s2:").grid(row=row, column=0, sticky="w", padx=6, pady=4)
        ttk.Entry(s2, textvariable=self.var_ks2, width=8).grid(row=row, column=1, padx=6, pady=4)
        row += 1
        ttk.Label(s2, text="n2:").grid(row=row, column=0, sticky="w", padx=6, pady=4)
        ttk.Entry(s2, textvariable=self.var_n2, width=8).grid(row=row, column=1, padx=6, pady=4)
        row += 1

        ttk.Label(s2, text="Цвет (R,G,B):").grid(row=row, column=0, sticky="w", padx=6, pady=6)
        rgb2 = ttk.Frame(s2)
        rgb2.grid(row=row, column=1)
        ttk.Entry(rgb2, textvariable=self.var_Rcol2, width=4).pack(side=tk.LEFT, padx=2)
        ttk.Entry(rgb2, textvariable=self.var_Gcol2, width=4).pack(side=tk.LEFT, padx=2)
        ttk.Entry(rgb2, textvariable=self.var_Bcol2, width=4).pack(side=tk.LEFT, padx=2)
        row += 1

        # ------------------ Источники света ------------------
        lights_frame = ttk.LabelFrame(controls_inner, text="Источники света")
        lights_frame.pack(fill=tk.BOTH, pady=5, padx=6, expand=True)

        # container for dynamic light rows
        self.light_rows = []
        lights_list = ttk.Frame(lights_frame)
        lights_list.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        def add_light(vals=None):
            # vals: (x,y,z,I0,R,G,B)
            row = ttk.Frame(lights_list)
            row.pack(fill=tk.X, pady=2)

            x_var = tk.DoubleVar(value=vals[0] if vals else 1600)
            y_var = tk.DoubleVar(value=vals[1] if vals else -1200)
            z_var = tk.DoubleVar(value=vals[2] if vals else 600)
            I_var = tk.DoubleVar(value=vals[3] if vals else 9500)
            R_var = tk.DoubleVar(value=vals[4] if vals else 1.0)
            G_var = tk.DoubleVar(value=vals[5] if vals else 1.0)
            B_var = tk.DoubleVar(value=vals[6] if vals else 1.0)

            # use grid to ensure all fields are visible
            lbl_x = ttk.Label(row, text='x:')
            ent_x = ttk.Entry(row, textvariable=x_var, width=6)
            lbl_y = ttk.Label(row, text='y:')
            ent_y = ttk.Entry(row, textvariable=y_var, width=6)
            lbl_z = ttk.Label(row, text='z:')
            ent_z = ttk.Entry(row, textvariable=z_var, width=6)
            lbl_I = ttk.Label(row, text='I0:')
            ent_I = ttk.Entry(row, textvariable=I_var, width=8)
            lbl_R = ttk.Label(row, text='R:')
            ent_R = ttk.Entry(row, textvariable=R_var, width=4)
            lbl_G = ttk.Label(row, text='G:')
            ent_G = ttk.Entry(row, textvariable=G_var, width=4)
            lbl_B = ttk.Label(row, text='B:')
            ent_B = ttk.Entry(row, textvariable=B_var, width=4)

            # Remove button
            btn = ttk.Button(row, text='Удалить', width=8)

            # place using grid
            lbl_x.grid(row=0, column=0, padx=(2,0))
            ent_x.grid(row=0, column=1, padx=(0,6))
            lbl_y.grid(row=0, column=2)
            ent_y.grid(row=0, column=3, padx=(0,6))
            lbl_z.grid(row=0, column=4)
            ent_z.grid(row=0, column=5, padx=(0,6))
            lbl_I.grid(row=0, column=6)
            ent_I.grid(row=0, column=7, padx=(0,8))
            lbl_R.grid(row=0, column=8)
            ent_R.grid(row=0, column=9, padx=(0,4))
            lbl_G.grid(row=0, column=10)
            ent_G.grid(row=0, column=11, padx=(0,4))
            lbl_B.grid(row=0, column=12)
            ent_B.grid(row=0, column=13, padx=(0,8))
            btn.grid(row=0, column=14, padx=(6,2))

            entry = {'frame': row, 'vars': (x_var, y_var, z_var, I_var, R_var, G_var, B_var)}
            self.light_rows.append(entry)

            def remove_this(ent=entry):
                try:
                    ent['frame'].destroy()
                except Exception:
                    pass
                try:
                    self.light_rows.remove(ent)
                except ValueError:
                    pass

            btn.config(command=remove_this)

        # initial lights (defaults similar to previous text)
        add_light((1600, -1200, 600, 9500, 1.0, 1.0, 1.0))
        add_light((-900, -400, 1400, 2200, 0.7, 0.7, 0.9))
        add_light((0, 800, 5500, 1200, 1.0, 0.4, 0.4))

        add_btns = ttk.Frame(lights_frame)
        add_btns.pack(fill=tk.X, padx=6, pady=(4,6))
        ttk.Button(add_btns, text='Добавить источник', command=lambda: add_light(None)).pack(side=tk.LEFT, padx=2)
        # кнопка удаления последнего источника
        def remove_last():
            if not self.light_rows:
                return
            ent = self.light_rows[-1]
            try:
                ent['frame'].destroy()
            except Exception:
                pass
            self.light_rows.pop()

        ttk.Button(add_btns, text='Удалить последний', command=remove_last).pack(side=tk.LEFT, padx=4)

        # ------------------ Кнопки ------------------
        btns = ttk.Frame(controls_inner)
        btns.pack(fill=tk.X, pady=8, padx=6)
        ttk.Button(btns, text="Рассчитать", command=self.on_compute, style="Accent.TButton").pack(side=tk.LEFT, expand=True, fill=tk.X, padx=4)
        ttk.Button(btns, text="Сохранить", command=self.on_save, style="TButton").pack(side=tk.LEFT, expand=True, fill=tk.X, padx=4)

        # ------------------ Статистика ------------------
        stats = ttk.LabelFrame(controls_inner, text="Статистика")
        stats.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        self.txt_stats = tk.Text(stats, width=30, height=12, state=tk.DISABLED, bg=D_PANEL2, fg=D_FG, insertbackground=D_FG)
        self.txt_stats.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        # ------------------ CANVAS (справа) ------------------
        img_frame = ttk.LabelFrame(right_frame, text="Изображение")
        img_frame.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        self.canvas = tk.Canvas(img_frame, bg=D_PANEL2, highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # make sure resize updates image
        self.canvas.bind("<Configure>", lambda e: self._update_canvas_image())

    # ------------------ КНОПКА "РАССЧИТАТЬ" ------------------
    def on_compute(self):
        try:
            # экран
            W = float(self.var_W.get())
            H = float(self.var_H.get())
            Wres = int(self.var_Wres.get())
            Hres = int(self.var_Hres.get())
            zO = float(self.var_zO.get())

            # собираем параметры источников света из полей
            lights = []
            for ent in self.light_rows:
                # поддерживаем два формата записи (старые кортежи или новый словарь)
                if isinstance(ent, dict):
                    vars_tuple = ent.get('vars')
                else:
                    vars_tuple = ent
                if vars_tuple is None:
                    continue
                xv, yv, zv, Iv, Rv, Gv, Bv = vars_tuple
                xL = float(xv.get())
                yL = float(yv.get())
                zL = float(zv.get())
                I0 = float(Iv.get())
                RL = float(Rv.get())
                GL = float(Gv.get())
                BL = float(Bv.get())
                lights.append((xL, yL, zL, I0, RL, GL, BL))
            if not lights:
                raise ValueError("Нужно задать хотя бы один источник света.")

            # сфера 1
            x1 = float(self.var_xC1.get())
            y1 = float(self.var_yC1.get())
            z1 = float(self.var_zC1.get())
            R1 = float(self.var_R1.get())
            ka1 = float(self.var_ka1.get())
            kd1 = float(self.var_kd1.get())
            ks1 = float(self.var_ks1.get())
            n1 = float(self.var_n1.get())
            col1 = (
                float(self.var_Rcol1.get()),
                float(self.var_Gcol1.get()),
                float(self.var_Bcol1.get())
            )

            # сфера 2
            x2 = float(self.var_xC2.get())
            y2 = float(self.var_yC2.get())
            z2 = float(self.var_zC2.get())
            R2 = float(self.var_R2.get())
            ka2 = float(self.var_ka2.get())
            kd2 = float(self.var_kd2.get())
            ks2 = float(self.var_ks2.get())
            n2 = float(self.var_n2.get())
            col2 = (
                float(self.var_Rcol2.get()),
                float(self.var_Gcol2.get()),
                float(self.var_Bcol2.get())
            )

            spheres = [
                (x1, y1, z1, R1, ka1, kd1, ks1, n1, col1),
                (x2, y2, z2, R2, ka2, kd2, ks2, n2, col2),
            ]

            I_front = render_view(
                "front",
                W, H, Wres, Hres,
                spheres,
                lights,
                zO,
                with_shadow=True
            )  # 0..1

            I8 = (I_front * 255).clip(0, 255).astype(np.uint8)
            self.image = Image.fromarray(I8, mode="RGB")
            self._update_canvas_image()

            Irgb_raw, mask = compute_two_spheres_color(
                W, H, Wres, Hres,
                x1, y1, z1, R1,
                x2, y2, z2, R2,
                zO,
                lights,
                ka1, kd1, ks1, n1, col1,
                ka2, kd2, ks2, n2, col2
            )
            self._update_stats(Irgb_raw, mask)

            # сохраняем три проекции
            I_xy = render_view("front", W, H, Wres, Hres, spheres, lights, zO, with_shadow=True)
            Image.fromarray((I_xy * 255).clip(0, 255).astype("uint8")).save("render_XY.png")

            I_xz = render_view("top", W, H, Wres, Hres, spheres, lights, zO, with_shadow=True)
            Image.fromarray((I_xz * 255).clip(0, 255).astype("uint8")).save("render_XZ.png")

            I_yz = render_view("right", W, H, Wres, Hres, spheres, lights, zO, with_shadow=True)
            Image.fromarray((I_yz * 255).clip(0, 255).astype("uint8")).save("render_YZ.png")

        except Exception as e:
            messagebox.showerror("Ошибка", str(e))

    # ------------------ РИСОВАНИЕ ------------------
    def _update_canvas_image(self):
        if self.image is None:
            return
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        if w < 10 or h < 10:
            self.after(100, self._update_canvas_image)
            return
        img = self.image.copy()
        img.thumbnail((w, h))
        self.photo = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.create_image(w // 2, h // 2, image=self.photo)

    # ------------------ СТАТИСТИКА ------------------
    def _update_stats(self, Irgb, mask):
        self.txt_stats.config(state=tk.NORMAL)
        self.txt_stats.delete("1.0", tk.END)

        inside = mask
        if not np.any(inside):
            self.txt_stats.insert(tk.END, "Сферы не попали в область экрана")
        else:
            mx = Irgb[inside].max()  # Убрали * M_TO_MM
            mn = Irgb[inside].min()  # Убрали * M_TO_MM
            self.txt_stats.insert(tk.END, f"Макс. яркость: {mx:.5g}\n")
            self.txt_stats.insert(tk.END, f"Мин. яркость: {mn:.5g}\n")

        self.txt_stats.config(state=tk.DISABLED)

    # ------------------ СОХРАНЕНИЕ ------------------
    def on_save(self):
        if self.image is None:
            messagebox.showinfo("Нет изображения", "Сначала рассчитай картинку")
            return
        fname = filedialog.asksaveasfilename(defaultextension=".png")
        if fname:
            self.image.save(fname)
            messagebox.showinfo("Сохранено", fname)


if __name__ == "__main__":
    app = SphereBrightnessApp()
    app.mainloop()