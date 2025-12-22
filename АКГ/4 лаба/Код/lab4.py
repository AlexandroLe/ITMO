# -*- coding: utf-8 -*-
"""
Single-file PyQt6 GUI for sphere brightness (Lambert emission + Blinn-Phong).
Dark theme by default, switchable to light theme via a button.
Output now includes units of brightness (Вт/м^2).
Left controls are inside a scroll area so they don't overflow the screen.
"""
import sys
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QLineEdit,
    QTextEdit, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox, QFileDialog,
    QMessageBox, QSizePolicy, QSpinBox, QScrollArea
)
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt

# ----------------- Theme colors (used in stylesheets) -----------------
D_BG = "#0f1720"        # main bg
D_PANEL = "#111520"     # panels
D_PANEL2 = "#172029"
D_FG = "#E6EEF3"
D_ENTRY = "#0f1620"
D_BUTTON = "#1f3a5a"
D_ACCENT = "#3aa3ff"

L_BG = "#f3f4f6"
L_PANEL = "#ffffff"
L_PANEL2 = "#f0f4f8"
L_FG = "#0b1320"
L_ENTRY = "#ffffff"
L_BUTTON = "#e6eef3"
L_ACCENT = "#1d4ed8"

# ----------------- Utility math -----------------
def normalize(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / (n + 1e-20)

def solve_ray_sphere(O, D, C, R):
    OC = O - C
    a = np.sum(D * D, axis=-1)
    b = 2.0 * np.sum(D * OC, axis=-1)
    c = np.sum(OC * OC, axis=-1) - R * R
    disc = b * b - 4 * a * c
    t = np.full(disc.shape, np.inf, dtype=float)
    hit = disc >= 0
    if np.any(hit):
        sqrt_disc = np.sqrt(np.maximum(disc, 0.0))
        t1 = (-b - sqrt_disc) / (2 * a)
        t2 = (-b + sqrt_disc) / (2 * a)
        tpos = np.where((t1 > 1e-9), t1, np.where((t2 > 1e-9), t2, np.inf))
        t = np.where(hit, tpos, np.inf)
    hit_mask = np.isfinite(t)
    return hit_mask, t

# ----------------- Renderer -----------------
def render_brightness(
    screen_W_mm, screen_H_mm, Wres, observer_mm, sphere_center_mm, sphere_R_mm,
    lights_mm_I0, ambient, kd, ks, shininess, max_side_pixels = 1000
):
    screen_W = float(screen_W_mm)
    screen_H = float(screen_H_mm)
    Wres = int(Wres)
    Hres = max(1, int(round(Wres * (screen_H / screen_W))))
    max_side = max(Hres, Wres)
    if max_side > max_side_pixels:
        scale = max_side_pixels / max_side
        Wres = max(2, int(round(Wres * scale)))
        Hres = max(2, int(round(Hres * scale)))

    mm2m = 1e-3
    screen_W_m = screen_W * mm2m
    screen_H_m = screen_H * mm2m
    sphere_center = np.array(sphere_center_mm, dtype=float)
    screen_z_m = sphere_center[2] * mm2m

    xs = np.linspace(sphere_center[0]*mm2m - screen_W_m/2, sphere_center[0]*mm2m + screen_W_m/2, Wres)
    ys = np.linspace(sphere_center[1]*mm2m + screen_H_m/2, sphere_center[1]*mm2m - screen_H_m/2, Hres)
    X, Y = np.meshgrid(xs, ys)
    pixel_pos = np.stack([X, Y, np.full_like(X, screen_z_m)], axis=-1)

    O = np.array(observer_mm, dtype=float) * mm2m
    D = pixel_pos - O
    D = normalize(D)

    C = sphere_center * mm2m
    R = float(sphere_R_mm) * mm2m

    O_grid = np.broadcast_to(O, D.shape)
    hit_mask, t_vals = solve_ray_sphere(O_grid, D, C, R)
    P = O_grid + D * t_vals[..., np.newaxis]

    N = np.zeros_like(P)
    V = np.zeros_like(P)
    valid = hit_mask
    if np.any(valid):
        N[valid] = normalize(P[valid] - C)
        V[valid] = normalize(O - P[valid])

    E = np.zeros((Hres, Wres), dtype=float)

    for (lx,ly,lz,I0) in lights_mm_I0:
        Lpos = np.array([lx,ly,lz], dtype=float) * mm2m
        if np.any(valid):
            Lvec = np.zeros_like(P)
            Lvec[valid] = normalize(Lpos - P[valid])
            r = np.zeros((Hres, Wres), dtype=float)
            r[valid] = np.linalg.norm(Lpos - P[valid], axis=-1)

            src_axis = normalize((C - Lpos).reshape((1,1,3)))[0,0]
            dir_light_to_point = np.zeros_like(P)
            dir_light_to_point[valid] = normalize(P[valid] - Lpos)
            cos_emit = np.zeros((Hres, Wres), dtype=float)
            cos_emit[valid] = np.maximum(0.0, np.sum(src_axis * dir_light_to_point[valid], axis=-1))

            cos_inc = np.zeros((Hres, Wres), dtype=float)
            cos_inc[valid] = np.maximum(0.0, np.sum(N[valid] * Lvec[valid], axis=-1))

            denom = np.where(r > 1e-12, r*r, np.inf)
            E_light = np.zeros((Hres, Wres), dtype=float)
            E_light[valid] = I0 * cos_emit[valid] * cos_inc[valid] / denom[valid]

            E_diff = kd * E_light

            Hvec = np.zeros_like(P)
            Hvec[valid] = normalize(Lvec[valid] + V[valid])
            Nh = np.zeros((Hres, Wres), dtype=float)
            Nh[valid] = np.maximum(0.0, np.sum(N[valid] * Hvec[valid], axis=-1))
            E_spec = ks * (Nh ** shininess) * I0 * cos_emit / denom

            E += E_diff + E_spec

    E += ambient
    E_masked = np.where(hit_mask, E, 0.0)

    vals_on_sphere = E_masked[hit_mask]
    if vals_on_sphere.size > 0:
        E_max = float(np.max(vals_on_sphere))
        E_min = float(np.min(vals_on_sphere))
        E_mean = float(np.mean(vals_on_sphere))
    else:
        E_max = E_min = E_mean = 0.0

    img = np.zeros_like(E_masked)
    if vals_on_sphere.size > 0:
        vmin = float(np.min(vals_on_sphere))
        vmax = float(np.max(vals_on_sphere))
        if vmax - vmin > 1e-20:
            img[hit_mask] = (E_masked[hit_mask] - vmin) / (vmax - vmin)
        else:
            img[hit_mask] = 1.0
    img_uint8 = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)

    stats = {'E_max': E_max, 'E_min': E_min, 'E_mean': E_mean, 'Wres': Wres, 'Hres': Hres}
    return img_uint8, E_masked, stats, P, hit_mask

# ----------------- GUI -----------------
class SingleWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Яркость на сфере — GUI (с темами)")
        self.resize(1200, 760)

        self.theme = "dark"  # default

        central = QWidget()
        self.setCentralWidget(central)
        main_h = QHBoxLayout(central)

        # ---------------- Left controls inside scroll area ----------------
        left_container = QWidget()
        left_container_layout = QVBoxLayout(left_container)
        left_container.setObjectName("leftContainer")   # <<< important: object name for stylesheet
        left_container.setMinimumWidth(350)      # keep same visual width
        left_container.setMaximumWidth(350)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(6,6,6,6)
        left_layout.setSpacing(8)

        gb_screen = QGroupBox("Экран (мм) / Разрешение (px)")
        fl = QFormLayout(gb_screen)
        self.edit_W = QLineEdit("1000"); self.edit_H = QLineEdit("800")
        self.spin_Wres = QSpinBox(); self.spin_Wres.setRange(200, 1600); self.spin_Wres.setValue(800)
        fl.addRow("W (мм):", self.edit_W); fl.addRow("H (мм):", self.edit_H); fl.addRow("Wres (px):", self.spin_Wres)
        left_layout.addWidget(gb_screen)

        gb_obs = QGroupBox("Наблюдатель (мм)")
        fo = QFormLayout(gb_obs)
        self.edit_obs_x = QLineEdit("0"); self.edit_obs_y = QLineEdit("0"); self.edit_obs_z = QLineEdit("-1500")
        fo.addRow("X:", self.edit_obs_x); fo.addRow("Y:", self.edit_obs_y); fo.addRow("Z:", self.edit_obs_z)
        left_layout.addWidget(gb_obs)

        gb_sph = QGroupBox("Сфера (мм)")
        fs = QFormLayout(gb_sph)
        self.edit_cx = QLineEdit("0"); self.edit_cy = QLineEdit("0"); self.edit_cz = QLineEdit("300"); self.edit_r = QLineEdit("250")
        fs.addRow("Cx:", self.edit_cx); fs.addRow("Cy:", self.edit_cy); fs.addRow("Cz:", self.edit_cz); fs.addRow("R:", self.edit_r)
        left_layout.addWidget(gb_sph)

        gb_l = QGroupBox("Источник 1 (мм, Вт/ср)")
        fl1 = QFormLayout(gb_l)
        self.l1x = QLineEdit("800"); self.l1y = QLineEdit("100"); self.l1z = QLineEdit("0"); self.l1I = QLineEdit("6000")
        fl1.addRow("x:", self.l1x); fl1.addRow("y:", self.l1y); fl1.addRow("z:", self.l1z); fl1.addRow("I0:", self.l1I)
        left_layout.addWidget(gb_l)

        gb_l2 = QGroupBox("Источник 2 (мм, Вт/ср)")
        fl2 = QFormLayout(gb_l2)
        self.l2x = QLineEdit("-1000"); self.l2y = QLineEdit("0"); self.l2z = QLineEdit("-400"); self.l2I = QLineEdit("4000")
        fl2.addRow("x:", self.l2x); fl2.addRow("y:", self.l2y); fl2.addRow("z:", self.l2z); fl2.addRow("I0:", self.l2I)
        left_layout.addWidget(gb_l2)

        gb_bp = QGroupBox("Blinn-Phong")
        fb = QFormLayout(gb_bp)
        self.edit_ambient = QLineEdit("0.35"); self.edit_kd = QLineEdit("1.0"); self.edit_ks = QLineEdit("0.9"); self.edit_n = QLineEdit("180")
        fb.addRow("ambient (Ka):", self.edit_ambient); fb.addRow("diffuse (Kd):", self.edit_kd)
        fb.addRow("specular (Ks):", self.edit_ks); fb.addRow("shininess (n):", self.edit_n)
        left_layout.addWidget(gb_bp)

        # buttons row (4 buttons)
        btn_layout = QHBoxLayout()
        self.btn_calc = QPushButton("Рассчитать")
        self.btn_save_img = QPushButton("Сохранить PNG")
        self.btn_save_info = QPushButton("Сохранить Info")
        self.btn_theme = QPushButton("Тема: Тёмная")
        self.btn_theme.setObjectName("themeBtn")
        btn_layout.addWidget(self.btn_calc); btn_layout.addWidget(self.btn_save_img)
        btn_layout.addWidget(self.btn_save_info); btn_layout.addWidget(self.btn_theme)
        left_layout.addLayout(btn_layout)

        left_layout.addStretch(1)  # push content to top

        # put 'left' into 'left_container'
        left_container_layout.addWidget(left)

        # create scroll area and put left_container inside it
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(left_container)
        scroll.setMinimumWidth(380)
        scroll.setMaximumWidth(380)

        main_h.addWidget(scroll)

        # ---------------- Right: preview + stats ----------------
        right = QWidget()
        right_layout = QVBoxLayout(right)
        self.preview_label = QLabel("Здесь будет изображение")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.preview_label.setMinimumSize(320,320)
        right_layout.addWidget(self.preview_label)

        self.stats_text = QTextEdit(); self.stats_text.setReadOnly(True); self.stats_text.setFixedHeight(320)
        right_layout.addWidget(self.stats_text)
        main_h.addWidget(right, 1)

        # connections
        self.btn_calc.clicked.connect(self.on_calc)
        self.btn_save_img.clicked.connect(self.on_save_img)
        self.btn_save_info.clicked.connect(self.on_save_info)
        self.btn_theme.clicked.connect(self.toggle_theme)

        # storage
        self.last_img = None
        self.last_brightness = None
        self.last_stats = None
        self.last_P = None
        self.last_hit = None
        self.last_params = None

        # apply initial theme
        self.apply_theme(self.theme)

    def apply_theme(self, theme):
        """Apply stylesheet for 'dark' or 'light'"""
        if theme == "dark":
            # use % formatting to avoid f-string brace conflicts
            style = (
                "QMainWindow { background: %s; color: %s; }\n"
                "QWidget#leftContainer { background: %s; }\n"
                "QGroupBox { background: %s; color: %s; border: 1px solid #22303a; border-radius: 8px; margin-top: 10px; padding: 6px; }\n"
                "QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px 0 3px; }\n"
                "QLabel, QLineEdit, QSpinBox, QComboBox { color: %s; background: %s; border: 1px solid #22303a; border-radius: 6px; padding: 4px; }\n"
                "QTextEdit { color: %s; background: %s; border: 1px solid #22303a; border-radius: 6px; padding: 6px; }\n"
                "QPushButton { background: %s; color: %s; border-radius: 8px; padding: 6px 8px; }\n"
                "QPushButton#themeBtn { background: %s; color: #001428; font-weight: bold; }\n"
            ) % (D_BG, D_FG, D_PANEL, D_PANEL, D_FG, D_FG, D_ENTRY, D_FG, D_PANEL2, D_BUTTON, D_FG, D_ACCENT)
            self.btn_theme.setText("Тема: Тёмная")
            self.theme = "dark"
        else:
            style = (
                "QMainWindow { background: %s; color: %s; }\n"
                "QWidget#leftContainer { background: %s; }\n"
                "QGroupBox { background: %s; color: %s; border: 1px solid #cbd5e1; border-radius: 8px; margin-top: 10px; padding: 6px; }\n"
                "QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px 0 3px; }\n"
                "QLabel, QLineEdit, QSpinBox, QComboBox { color: %s; background: %s; border: 1px solid #cbd5e1; border-radius: 6px; padding: 4px; }\n"
                "QTextEdit { color: %s; background: %s; border: 1px solid #cbd5e1; border-radius: 6px; padding: 6px; }\n"
                "QPushButton { background: %s; color: %s; border-radius: 8px; padding: 6px 8px; }\n"
                "QPushButton#themeBtn { background: %s; color: #ffffff; font-weight: bold; }\n"
            ) % (L_BG, L_FG, L_PANEL, L_PANEL, L_FG, L_FG, L_ENTRY, L_FG, L_PANEL2, L_BUTTON, L_FG, L_ACCENT)
            self.btn_theme.setText("Тема: Светлая")
            self.theme = "light"

        self.setStyleSheet(style)

        # adjust preview background to match theme
        if self.theme == "dark":
            self.preview_label.setStyleSheet(f"background: {D_PANEL2}; border: 1px solid #22303a;")
        else:
            self.preview_label.setStyleSheet(f"background: {L_PANEL2}; border: 1px solid #cbd5e1;")

    def toggle_theme(self):
        self.apply_theme("light" if self.theme == "dark" else "dark")

    def read_inputs(self):
        try:
            W = float(self.edit_W.text()); H = float(self.edit_H.text())
            Wres = int(self.spin_Wres.value())
            obs = np.array([float(self.edit_obs_x.text()), float(self.edit_obs_y.text()), float(self.edit_obs_z.text())], dtype=float)
            sph_c = np.array([float(self.edit_cx.text()), float(self.edit_cy.text()), float(self.edit_cz.text())], dtype=float)
            R = float(self.edit_r.text())
            l1 = (float(self.l1x.text()), float(self.l1y.text()), float(self.l1z.text()), float(self.l1I.text()))
            l2 = (float(self.l2x.text()), float(self.l2y.text()), float(self.l2z.text()), float(self.l2I.text()))
            ambient = float(self.edit_ambient.text())
            kd = float(self.edit_kd.text()); ks = float(self.edit_ks.text()); n = float(self.edit_n.text())

            if not (100 <= W <= 10000 and 100 <= H <= 10000):
                raise ValueError("W,H должны быть в 100..10000 мм")
            if R <= 0:
                raise ValueError("R должен быть положительным")
            return W,H,Wres,obs,sph_c,R,[l1,l2],ambient,kd,ks,n
        except Exception as e:
            QMessageBox.critical(self, "Ошибка ввода", str(e))
            return None

    def on_calc(self):
        data = self.read_inputs()
        if data is None:
            return
        W,H,Wres,obs,sph_c,R,lights,ambient,kd,ks,n = data

        try:
            img_uint8, brightness, stats, P, hit_mask = render_brightness(
                W, H, Wres, obs, sph_c, R, lights, ambient, kd, ks, n, max_side_pixels=1000
            )
        except Exception as e:
            QMessageBox.critical(self, "Ошибка расчета", str(e))
            return

        self.last_img = img_uint8
        self.last_brightness = brightness
        self.last_stats = stats
        self.last_P = P
        self.last_hit = hit_mask
        self.last_params = {'W':W,'H':H,'Wres':stats['Wres'],'Hres':stats['Hres'],'observer':obs,'sphere_center':sph_c,'R':R,'lights':lights,'ambient':ambient,'kd':kd,'ks':ks,'n':n}

        h,w = img_uint8.shape
        qimg = QImage(img_uint8.data, w, h, w, QImage.Format.Format_Grayscale8)
        pix = QPixmap.fromImage(qimg)
        target_w = max(320, min(1000, self.preview_label.width()))
        target_h = max(320, min(1000, self.preview_label.height()))
        scaled = pix.scaled(target_w, target_h, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.preview_label.setPixmap(scaled)

        # ---- Форматированный вывод (5 точек) + единицы яркости ----
        lines = []
        lines.append("--- Расчетные значения яркости (абсолютные величины) ---\n")
        lines.append(f"Генерируемое разрешение изображения: {stats['Wres']}x{stats['Hres']} пикселей\n\n")
        lines.append("Единицы яркости: Вт/м^2\n\n")

        visible_indices = np.argwhere(brightness > 0)
        if visible_indices.size == 0:
            lines.append("Сфера не видна / нет видимых пикселей.\n")
        else:
            rng = np.random.default_rng(seed=12345)
            num = min(5, visible_indices.shape[0])
            chosen_idx = rng.choice(visible_indices.shape[0], size=num, replace=False)
            chosen_pixels = visible_indices[chosen_idx]  # array of (py, px)
            for i, (py, px) in enumerate(chosen_pixels, start=1):
                pixel_str = f"(np.int64({int(px)}), np.int64({int(py)}))"
                world_m = self.last_P[py, px]  # meters
                world_mm = world_m * 1000.0
                wx, wy, wz = world_mm
                world_str = f"[{wx:.2f}, {wy:.2f}, {wz:.2f}] мм"
                brightness_val = float(self.last_brightness[py, px])
                lines.append(f"Точка {i}:\n")
                lines.append(f"  Пиксель: {pixel_str}\n")
                lines.append(f"  Мировые коорд.: {world_str}\n")
                lines.append(f"  Яркость (абс.): {brightness_val:.10f}\n")
                lines.append("---\n")

        vals = brightness[brightness > 0]
        if vals.size > 0:
            Emax = float(np.max(vals)); Emin = float(np.min(vals)); Emean = float(np.mean(vals))
        else:
            Emax = Emin = Emean = 0.0
        lines.append(f"\nМаксимальная яркость на сфере: {Emax:.7f}\n")
        lines.append(f"Минимальная яркость на сфере (ненулевая): {Emin:.7f}\n")
        lines.append(f"Средняя яркость на сфере (ненулевая): {Emean:.7f}\n")

        self.stats_text.setPlainText("".join(lines))

    def on_save_img(self):
        if self.last_img is None:
            QMessageBox.warning(self, "Нет изображения", "Сначала рассчитайте изображение.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Сохранить PNG", "", "PNG files (*.png)")
        if not path:
            return
        h,w = self.last_img.shape
        qimg = QImage(self.last_img.data, w, h, w, QImage.Format.Format_Grayscale8)
        ok = qimg.save(path, "PNG")
        if ok:
            QMessageBox.information(self, "Сохранено", f"Сохранено в {path}")
        else:
            QMessageBox.critical(self, "Ошибка", "Не удалось сохранить изображение")

    def on_save_info(self):
        if self.last_params is None:
            QMessageBox.warning(self, "Нет данных", "Сначала рассчитайте изображение.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Сохранить Info", "", "Text files (*.txt)")
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self.stats_text.toPlainText())
            QMessageBox.information(self, "Сохранено", f"Инфо сохранено в {path}")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка записи", str(e))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = SingleWindow()
    win.show()
    sys.exit(app.exec())
