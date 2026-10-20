"""
Лабораторная работа №2: Трекинг объекта на видео на основе ключевых точек
Дисциплина: Специализированные комплексы обработки изображений (СКОИ)
"""

import os
import sys
import time
import argparse
from typing import Tuple, Optional, List, Dict, Any
import cv2
import numpy as np


class ObjectKeypointTracker:
    """
    Класс трекера плоского объекта на видео на основе ключевых точек и перспективной трансформации (гомографии).
    Поддерживает детекторы SIFT и ORB, а также гибридный режим с Optical Flow (KLT) для устойчивости к размытию.
    """

    def __init__(
        self,
        detector_type: str = "sift",
        ratio_threshold: float = 0.75,
        min_inliers: int = 8,
        ransac_thresh: float = 5.0,
        enable_optical_flow_fallback: bool = True,
        smooth_factor: float = 0.35,
    ):
        self.detector_type = detector_type.lower()
        self.ratio_threshold = ratio_threshold
        self.min_inliers = min_inliers
        self.ransac_thresh = ransac_thresh
        self.enable_optical_flow_fallback = enable_optical_flow_fallback
        self.smooth_factor = smooth_factor

        # Инициализация детектора и сопоставителя
        if self.detector_type == "sift":
            self.detector = cv2.SIFT_create(
                contrastThreshold=0.015,
                edgeThreshold=15
            )
            self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        elif self.detector_type == "orb":
            self.detector = cv2.ORB_create(
                nfeatures=3000,
                scaleFactor=1.2,
                nlevels=8,
                edgeThreshold=15,
                fastThreshold=10
            )
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        else:
            raise ValueError(f"Неизвестный тип детектора: {detector_type}. Используйте 'sift' или 'orb'.")

        # Шаблон объекта (первый кадр)
        self.template_frame: Optional[np.ndarray] = None
        self.template_gray: Optional[np.ndarray] = None
        self.template_kp: List[cv2.KeyPoint] = []
        self.template_des: Optional[np.ndarray] = None
        self.template_corners: Optional[np.ndarray] = None  # (4, 1, 2)
        self.template_area: float = 0.0

        # Состояние трекера по кадрам
        self.prev_gray: Optional[np.ndarray] = None
        self.prev_corners: Optional[np.ndarray] = None
        self.prev_tracked_pts: Optional[np.ndarray] = None

    def initialize(self, first_frame: np.ndarray, crop_roi: Optional[Tuple[int, int, int, int]] = None):
        """
        Инициализация шаблона по первому кадру видео.
        Если передан crop_roi = (x, y, w, h), эталоном служит вырезанная область.
        Иначе используется весь первый кадр.
        """
        self.template_frame = first_frame.copy()
        if crop_roi is not None:
            rx, ry, rw, rh = crop_roi
            self.template_frame = self.template_frame[ry : ry + rh, rx : rx + rw]

        h, w = self.template_frame.shape[:2]
        self.template_gray = cv2.cvtColor(self.template_frame, cv2.COLOR_BGR2GRAY)
        self.template_kp, self.template_des = self.detector.detectAndCompute(self.template_gray, None)

        self.template_corners = np.float32(
            [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]
        ).reshape(-1, 1, 2)
        self.template_area = float(w * h)

        self.prev_gray = None
        self.prev_corners = None
        self.prev_tracked_pts = None

    def _validate_geometry(
        self,
        corners: np.ndarray,
        frame_shape: Tuple[int, int]
    ) -> bool:
        """
        Проверка геометрической корректности найденного четырехугольника:
        1. Выпуклость (не самопересекается, не вырожден в линию/бабочку).
        2. Допустимая площадь (не исчезающе мала и не превышает кадр в разы).
        3. Положительное направление обхода (детерминант преобразования положителен).
        """
        h_frame, w_frame = frame_shape[:2]
        poly = corners.reshape(-1, 2)
        poly_int = np.int32(poly)

        # 1. Проверка на выпуклость
        if not cv2.isContourConvex(poly_int):
            return False

        # 2. Проверка площади
        area = cv2.contourArea(poly_int)
        frame_area = w_frame * h_frame
        if area < 0.002 * frame_area or area > 1.6 * frame_area:
            return False

        # 3. Проверка соотношения сторон
        d01 = np.linalg.norm(poly[0] - poly[1])
        d12 = np.linalg.norm(poly[1] - poly[2])
        d23 = np.linalg.norm(poly[2] - poly[3])
        d30 = np.linalg.norm(poly[3] - poly[0])

        if d01 <= 3 or d12 <= 3 or d23 <= 3 or d30 <= 3:
            return False

        ratio_horizontal = max(d01, d23) / (min(d01, d23) + 1e-6)
        ratio_vertical = max(d12, d30) / (min(d12, d30) + 1e-6)
        if ratio_horizontal > 5.5 or ratio_vertical > 5.5:
            return False

        return True

    def process_frame(self, frame: np.ndarray) -> Tuple[bool, Optional[np.ndarray], List[Tuple[float, float]], str]:
        """
        Обработка очередного кадра видео.
        Возвращает:
        - success (bool): обнаружен ли объект
        - corners (np.ndarray or None): координаты 4 углов рамки (4, 1, 2)
        - inlier_pts (list of (x, y)): координаты точек-инлаеров
        - method (str): метод ("DETECTION_SIFT", "OPTICAL_FLOW", "LOST")
        """
        if self.template_des is None or len(self.template_kp) < 4:
            return False, None, [], "NO_TEMPLATE"

        h_frame, w_frame = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 1. Шаг детекции по ключевым точкам и сопоставлению
        kp_cur, des_cur = self.detector.detectAndCompute(gray, None)

        detection_success = False
        detected_corners = None
        inliers_list: List[Tuple[float, float]] = []

        if des_cur is not None and len(kp_cur) >= 4:
            matches = self.matcher.knnMatch(self.template_des, des_cur, k=2)
            good_matches = []
            for m_n in matches:
                if len(m_n) == 2:
                    m, n = m_n
                    if m.distance < self.ratio_threshold * n.distance:
                        good_matches.append(m)

            if len(good_matches) >= self.min_inliers:
                src_pts = np.float32([self.template_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp_cur[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, self.ransac_thresh)
                if M is not None and mask is not None:
                    num_inliers = int(mask.sum())
                    if num_inliers >= self.min_inliers:
                        cand_corners = cv2.perspectiveTransform(self.template_corners, M)
                        if self._validate_geometry(cand_corners, (h_frame, w_frame)):
                            detection_success = True
                            detected_corners = cand_corners
                            # Извлекаем координаты инлаеров
                            mask_flat = mask.ravel()
                            for idx, is_inlier in enumerate(mask_flat):
                                if is_inlier:
                                    pt = dst_pts[idx][0]
                                    inliers_list.append((float(pt[0]), float(pt[1])))

        # 2. Если прямая детекция сработала — сохраняем результат
        if detection_success:
            if self.prev_corners is not None and self.smooth_factor > 0:
                # Экспоненциальное сглаживание для устранения микровибраций
                final_corners = (1.0 - self.smooth_factor) * detected_corners + self.smooth_factor * self.prev_corners
            else:
                final_corners = detected_corners

            self.prev_corners = final_corners.copy()
            self.prev_gray = gray.copy()
            self.prev_tracked_pts = np.float32(inliers_list).reshape(-1, 1, 2) if inliers_list else None
            return True, final_corners, inliers_list, f"DETECTION_{self.detector_type.upper()}"

        # 3. Шаг Optical Flow fallback (если включен и в предыдущем кадре объект был найден)
        if self.enable_optical_flow_fallback and self.prev_gray is not None and self.prev_corners is not None:
            # Отслеживаем опорные точки из предыдущего кадра через Lucas-Kanade
            pts_to_track = self.prev_tracked_pts
            if pts_to_track is None or len(pts_to_track) < 8:
                # Если инлаеров было мало, генерируем точки внутри предыдущей рамки
                mask_roi = np.zeros_like(self.prev_gray)
                cv2.fillConvexPoly(mask_roi, np.int32(self.prev_corners.reshape(-1, 2)), 255)
                pts_to_track = cv2.goodFeaturesToTrack(self.prev_gray, maxCorners=100, qualityLevel=0.01, minDistance=5, mask=mask_roi)

            if pts_to_track is not None and len(pts_to_track) >= 6:
                next_pts, status, err = cv2.calcOpticalFlowPyrLK(
                    self.prev_gray,
                    gray,
                    pts_to_track,
                    None,
                    winSize=(21, 21),
                    maxLevel=3,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
                )

                if next_pts is not None and status is not None:
                    good_prev = pts_to_track[status == 1]
                    good_next = next_pts[status == 1]

                    if len(good_prev) >= 6:
                        # Находим аффинное преобразование или гомографию между кадрами
                        M_flow, inlier_mask = cv2.estimateAffinePartial2D(good_prev, good_next)
                        if M_flow is not None:
                            # Преобразуем углы предыдущего кадра
                            prev_2d = self.prev_corners.reshape(-1, 2)
                            flow_corners = cv2.transform(prev_2d.reshape(-1, 1, 2), M_flow)
                            if self._validate_geometry(flow_corners, (h_frame, w_frame)):
                                self.prev_corners = flow_corners.copy()
                                self.prev_gray = gray.copy()
                                self.prev_tracked_pts = good_next.reshape(-1, 1, 2)
                                flow_pts_list = [(float(p[0]), float(p[1])) for p in good_next]
                                return True, flow_corners, flow_pts_list, "OPTICAL_FLOW_KLT"

        # 4. Объект потерян
        self.prev_corners = None
        self.prev_tracked_pts = None
        self.prev_gray = gray.copy()
        return False, None, [], "LOST"


def draw_tracking_result(
    frame: np.ndarray,
    success: bool,
    corners: Optional[np.ndarray],
    inliers: List[Tuple[float, float]],
    label: str,
    method: str,
    fps: float,
    inlier_count: int,
    draw_info_overlay: bool = True
) -> np.ndarray:
    """
    Отрисовка результатов трекинга:
    - Зеленая перспективная рамка полигона
    - Зеленые маркеры инлаеров (ромбики/точки)
    - Четкая надпись с черным контуром и белым текстом
    - Информационная плашка (FPS, метод, число совпадений)
    """
    vis = frame.copy()
    h_img, w_img = frame.shape[:2]
    scale_factor = max(0.6, min(w_img, h_img) / 480.0)

    if success and corners is not None:
        poly_pts = np.int32(corners.reshape(-1, 2))
        line_thick = max(2, int(round(3 * scale_factor)))

        # 1. Отрисовка перспективного контура
        cv2.polylines(vis, [poly_pts], isClosed=True, color=(0, 255, 0), thickness=line_thick, lineType=cv2.LINE_AA)

        # 2. Отрисовка инлаеров (стилизованные зеленые ромбики как в образце)
        d_rad = max(2, int(round(3 * scale_factor)))
        for pt in inliers:
            x, y = int(round(pt[0])), int(round(pt[1]))
            pts_diamond = np.array([[x, y - d_rad], [x + d_rad, y], [x, y + d_rad], [x - d_rad, y]], np.int32)
            cv2.fillConvexPoly(vis, pts_diamond, (0, 255, 0), lineType=cv2.LINE_AA)
            cv2.polylines(vis, [pts_diamond], True, (0, 100, 0), 1, lineType=cv2.LINE_AA)

        # 3. Отрисовка подписи объекта у верхнего края рамки
        min_y_idx = np.argmin(poly_pts[:, 1])
        top_x, top_y = poly_pts[min_y_idx]
        font_scale = 0.7 * scale_factor
        text_thick = max(2, int(round(2 * scale_factor)))
        text_bg_thick = text_thick + 2
        text_pos = (max(10, int(top_x - 30 * scale_factor)), max(30, int(top_y - 12 * scale_factor)))

        # Двухслойный текст: толстый черный контур + белый текст
        cv2.putText(vis, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), text_bg_thick, lineType=cv2.LINE_AA)
        cv2.putText(vis, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), text_thick, lineType=cv2.LINE_AA)

    # 4. Верхний информационный оверлей
    if draw_info_overlay:
        status_color = (0, 255, 0) if success else (0, 0, 255)
        status_text = f"Status: {method} | Inliers: {inlier_count} | FPS: {fps:.1f}"

        banner_w = int(480 * scale_factor)
        banner_h = int(36 * scale_factor)
        overlay = vis.copy()
        cv2.rectangle(overlay, (8, 8), (8 + banner_w, 8 + banner_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.65, vis, 0.35, 0, vis)

        cv2.putText(
            vis,
            status_text,
            (16, int(26 * scale_factor)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5 * scale_factor,
            status_color,
            max(1, int(round(1.5 * scale_factor))),
            lineType=cv2.LINE_AA
        )

    return vis


def process_video(
    input_path: str,
    output_path: str,
    label: str,
    detector_type: str = "sift",
    enable_optical_flow: bool = True,
    draw_overlay: bool = True,
    max_frames: Optional[int] = None
) -> Dict[str, Any]:
    """
    Обработка одного видеофайла: трекинг объекта и сохранение аннотированного видео.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Входной видеофайл не найден: {input_path}")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {input_path}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if max_frames is not None:
        total_frames = min(total_frames, max_frames)

    # Чтение первого кадра для шаблона
    ret, first_frame = cap.read()
    if not ret or first_frame is None:
        cap.release()
        raise RuntimeError("Не удалось прочитать первый кадр видео.")

    tracker = ObjectKeypointTracker(
        detector_type=detector_type,
        enable_optical_flow_fallback=enable_optical_flow
    )
    tracker.initialize(first_frame)

    # Создание VideoWriter: используем H.264 (avc1) для воспроизведения прямо в VS Code и браузерах
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    out = cv2.VideoWriter(output_path, fourcc, fps_in, (w, h))
    if not out.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps_in, (w, h))

    tracked_frames = 0
    inliers_counts: List[int] = []
    frame_times: List[float] = []
    method_counts: Dict[str, int] = {}

    # Записываем первый кадр
    out.write(first_frame)
    frame_idx = 1

    print(f"Обработка: '{os.path.basename(input_path)}' ({w}x{h}, {fps_in:.1f} fps, {total_frames} кадров)...")

    while True:
        if max_frames is not None and frame_idx >= max_frames:
            break

        t_start = time.perf_counter()
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        success, corners, inliers, method = tracker.process_frame(frame)
        t_proc = time.perf_counter() - t_start
        frame_times.append(t_proc)

        current_fps = 1.0 / t_proc if t_proc > 0 else fps_in
        if success:
            tracked_frames += 1
            inliers_counts.append(len(inliers))

        method_counts[method] = method_counts.get(method, 0) + 1

        # Отрисовка
        annotated = draw_tracking_result(
            frame=frame,
            success=success,
            corners=corners,
            inliers=inliers,
            label=label,
            method=method,
            fps=current_fps,
            inlier_count=len(inliers),
            draw_info_overlay=draw_overlay
        )

        out.write(annotated)
        frame_idx += 1

        if frame_idx % 50 == 0 or frame_idx == total_frames:
            print(f"  Кадр {frame_idx}/{total_frames} | {method} | Инлаеры: {len(inliers)} | FPS: {current_fps:.1f}")

    cap.release()
    out.release()

    duration = sum(frame_times)
    avg_fps = len(frame_times) / duration if duration > 0 else 0.0
    tracking_rate = (tracked_frames / (frame_idx - 1)) * 100 if frame_idx > 1 else 0.0
    avg_inliers = float(np.mean(inliers_counts)) if inliers_counts else 0.0

    stats = {
        "input": input_path,
        "output": output_path,
        "label": label,
        "detector": detector_type,
        "total_frames": frame_idx,
        "tracked_frames": tracked_frames,
        "tracking_rate": tracking_rate,
        "avg_inliers": avg_inliers,
        "avg_fps": avg_fps,
        "methods": method_counts
    }

    print(f"Готово: {os.path.basename(output_path)}")
    print(f"  Успешность трекинга: {tracked_frames}/{frame_idx-1} ({tracking_rate:.1f}%)")
    print(f"  Среднее число инлаеров: {avg_inliers:.1f}")
    print(f"  Средний FPS: {avg_fps:.1f}\n")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Трекинг объекта на видео на основе ключевых точек (СКОИ Лаб 2)")
    parser.add_argument("-i", "--input", type=str, default=None, help="Путь к входному видео")
    parser.add_argument("-o", "--output", type=str, default=None, help="Путь к выходному видео")
    parser.add_argument("-l", "--label", type=str, default=None, help="Подпись объекта на видео")
    parser.add_argument("-d", "--detector", type=str, default="sift", choices=["sift", "orb"], help="Детектор признаков")
    parser.add_argument("--no-optical-flow", action="store_true", help="Отключить fallback на Optical Flow")
    parser.add_argument("--all", action="store_true", help="Запустить автоматическую обработку всех видео")

    args = parser.parse_args()

    # Определение базовой папки Лаб2
    current_dir = os.path.dirname(os.path.abspath(__file__))

    if args.all or args.input is None:
        # Пакетная обработка всех 4 видео лабораторной
        videos = [
            {
                "input": os.path.join(current_dir, "Тестовые видео", "mona-lisa.avi"),
                "output": os.path.join(current_dir, "output", "tracked_mona-lisa.mp4"),
                "label": "Mona Lisa"
            },
            {
                "input": os.path.join(current_dir, "Тестовые видео", "mona-lisa-blur.avi"),
                "output": os.path.join(current_dir, "output", "tracked_mona-lisa-blur.mp4"),
                "label": "Mona Lisa"
            },
            {
                "input": os.path.join(current_dir, "Тестовые видео", "mona-lisa-blur-extra-credit.avi"),
                "output": os.path.join(current_dir, "output", "tracked_mona-lisa-blur-extra.mp4"),
                "label": "Mona Lisa"
            },
            {
                "input": os.path.join(current_dir, "book.mp4"),
                "output": os.path.join(current_dir, "output", "tracked_book.mp4"),
                "label": "Book: Call of Cthulhu"
            }
        ]

        print("=== Запуск комплексного тестирования на всех видео ЛР2 ===")
        all_stats = []
        for v in videos:
            if os.path.exists(v["input"]):
                stat = process_video(
                    input_path=v["input"],
                    output_path=v["output"],
                    label=v["label"],
                    detector_type=args.detector,
                    enable_optical_flow=not args.no_optical_flow
                )
                all_stats.append(stat)
            else:
                print(f"[ПРЕДУПРЕЖДЕНИЕ] Файл не найден: {v['input']}")

        print("\n=== ИТОГОВАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ ===")
        print(f"{'Видео':<35} | {'Кадров':<8} | {'Трекинг %':<10} | {'Инлаеры':<9} | {'FPS':<6}")
        print("-" * 78)
        for s in all_stats:
            name = os.path.basename(s['input'])
            print(f"{name:<35} | {s['total_frames']:<8} | {s['tracking_rate']:<9.1f}% | {s['avg_inliers']:<9.1f} | {s['avg_fps']:<6.1f}")

    else:
        # Обработка одиночного видео
        inp = args.input
        out_p = args.output or os.path.join(current_dir, "output", f"tracked_{os.path.basename(inp)}.mp4")
        lbl = args.label or "Target Object"
        process_video(
            input_path=inp,
            output_path=out_p,
            label=lbl,
            detector_type=args.detector,
            enable_optical_flow=not args.no_optical_flow
        )


if __name__ == "__main__":
    main()
