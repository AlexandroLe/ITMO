from __future__ import annotations

from math import cos, sin
from html import escape
from pathlib import Path

from .lighting import calculate_point, point_inside_triangle_by_lengths, triangle_normal
from .models import PointResult, Scene, SurfacePoint
from .vector import Vec3


def tone_map(rgb: tuple[float, float, float], scale: float) -> tuple[int, int, int]:
    return tuple(max(0, min(255, int(channel * scale))) for channel in rgb)


def write_ppm_render(path: Path, scene: Scene, width: int = 320, height: int = 240) -> None:
    edge01_len = (scene.triangle.p1 - scene.triangle.p0).length()
    edge02_len = (scene.triangle.p2 - scene.triangle.p0).length()
    samples: list[tuple[float, float, tuple[float, float, float] | None]] = []
    max_channel = 0.0

    for row in range(height):
        y = edge02_len * (row + 0.5) / height
        for col in range(width):
            x = edge01_len * (col + 0.5) / width
            if point_inside_triangle_by_lengths(scene.triangle, x, y):
                result = calculate_point(scene, SurfacePoint("pixel", x, y))
                max_channel = max(max_channel, *result.brightness_rgb)
                samples.append((x, y, result.brightness_rgb))
            else:
                samples.append((x, y, None))

    scale = 230.0 / max_channel if max_channel > 0 else 1.0
    lines = [f"P3\n{width} {height}\n255\n"]
    for _, _, rgb in samples:
        color = (18, 20, 24) if rgb is None else tone_map(rgb, scale)
        lines.append(f"{color[0]} {color[1]} {color[2]}\n")

    path.write_text("".join(lines), encoding="ascii")


def _project_isometric(point: Vec3) -> tuple[float, float]:
    x = (point.x - point.y) * 0.8660254038
    y = (point.x + point.y) * 0.5 - point.z
    return (x, y)


def _star_points(cx: float, cy: float, outer: float = 9.0, inner: float = 4.0) -> str:
    values: list[str] = []
    for index in range(10):
        radius = outer if index % 2 == 0 else inner
        angle = -1.5707963268 + index * 0.6283185307
        x = cx + radius * cos(angle)
        y = cy + radius * sin(angle)
        values.append(f"{x:.2f},{y:.2f}")
    return " ".join(values)


def write_scene_scheme_svg(
    path: Path,
    scene: Scene,
    results: list[PointResult],
    width: int = 960,
    height: int = 680,
) -> None:
    triangle = scene.triangle
    vertices = [triangle.p0, triangle.p1, triangle.p2]
    calc_points = [result.position for result in results]
    centroid = (triangle.p0 + triangle.p1 + triangle.p2) / 3.0
    normal_end = centroid + triangle_normal(triangle) * 1.3
    axis_endpoints = [Vec3(6, 0, 0), Vec3(0, 5, 0), Vec3(0, 0, 4)]

    world_points = (
        vertices
        + calc_points
        + [scene.observer, centroid, normal_end]
        + [light.position for light in scene.lights]
        + [light.position + light.axis.normalized() * 0.9 for light in scene.lights]
        + [Vec3(0, 0, 0)]
        + axis_endpoints
    )

    projected = [_project_isometric(point) for point in world_points]
    min_x = min(point[0] for point in projected)
    max_x = max(point[0] for point in projected)
    min_y = min(point[1] for point in projected)
    max_y = max(point[1] for point in projected)
    margin = 70.0
    scale_x = (width - margin * 2) / max(max_x - min_x, 1.0)
    scale_y = (height - margin * 2) / max(max_y - min_y, 1.0)
    scale = min(scale_x, scale_y)

    def screen(point: Vec3) -> tuple[float, float]:
        px, py = _project_isometric(point)
        sx = margin + (px - min_x) * scale
        sy = height - margin - (py - min_y) * scale
        return (sx, sy)

    def line(start: Vec3, end: Vec3, stroke: str, width_px: float = 2.0, extra: str = "") -> str:
        x1, y1 = screen(start)
        x2, y2 = screen(end)
        return (
            f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
            f'stroke="{stroke}" stroke-width="{width_px}" {extra}/>'
        )

    p0, p1, p2 = [screen(point) for point in vertices]
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<defs>",
        '<marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">',
        '<path d="M 0 0 L 10 5 L 0 10 z" fill="#242933"/>',
        "</marker>",
        "</defs>",
        f'<rect width="{width}" height="{height}" fill="#f7f8fb"/>',
        '<text x="32" y="42" font-family="Arial" font-size="24" font-weight="700" fill="#20242b">3D scene scheme</text>',
        '<text x="32" y="68" font-family="Arial" font-size="14" fill="#596070">triangle, lights, observer and calculation points</text>',
    ]

    origin = Vec3(0, 0, 0)
    elements.append(line(origin, axis_endpoints[0], "#d15b47", 2.5, 'marker-end="url(#arrow)"'))
    elements.append(line(origin, axis_endpoints[1], "#2f8f65", 2.5, 'marker-end="url(#arrow)"'))
    elements.append(line(origin, axis_endpoints[2], "#3c72c9", 2.5, 'marker-end="url(#arrow)"'))
    for label, endpoint, color in [("X", axis_endpoints[0], "#d15b47"), ("Y", axis_endpoints[1], "#2f8f65"), ("Z", axis_endpoints[2], "#3c72c9")]:
        x, y = screen(endpoint)
        elements.append(f'<text x="{x + 8:.2f}" y="{y - 6:.2f}" font-family="Arial" font-size="16" font-weight="700" fill="{color}">{label}</text>')

    elements.append(
        f'<polygon points="{p0[0]:.2f},{p0[1]:.2f} {p1[0]:.2f},{p1[1]:.2f} {p2[0]:.2f},{p2[1]:.2f}" '
        'fill="#dce8ff" stroke="#2f5fb3" stroke-width="3" opacity="0.9"/>'
    )

    for label, vertex in [("P0", triangle.p0), ("P1", triangle.p1), ("P2", triangle.p2)]:
        x, y = screen(vertex)
        elements.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5" fill="#2f5fb3"/>')
        elements.append(f'<text x="{x + 8:.2f}" y="{y - 8:.2f}" font-family="Arial" font-size="14" fill="#1f3768">{label}</text>')

    elements.append(line(centroid, normal_end, "#242933", 2.4, 'marker-end="url(#arrow)"'))
    nx, ny = screen(normal_end)
    elements.append(f'<text x="{nx + 8:.2f}" y="{ny - 6:.2f}" font-family="Arial" font-size="14" font-weight="700" fill="#242933">N</text>')

    for result in results:
        x, y = screen(result.position)
        elements.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="7" fill="#e23d4f" stroke="#ffffff" stroke-width="2"/>')
        elements.append(f'<text x="{x + 10:.2f}" y="{y + 4:.2f}" font-family="Arial" font-size="13" fill="#9d1f31">{escape(result.name)}</text>')

    for light in scene.lights:
        lx, ly = screen(light.position)
        axis_end = light.position + light.axis.normalized() * 0.9
        elements.append(line(light.position, axis_end, "#d89b00", 2.0, 'marker-end="url(#arrow)"'))
        for result in results:
            elements.append(line(light.position, result.position, "#e7b64b", 1.2, 'stroke-dasharray="5 5" opacity="0.55"'))
        elements.append(f'<polygon points="{_star_points(lx, ly)}" fill="#ffc533" stroke="#a06a00" stroke-width="1.5"/>')
        elements.append(f'<text x="{lx + 13:.2f}" y="{ly - 10:.2f}" font-family="Arial" font-size="13" fill="#765100">{escape(light.name)}</text>')

    ox, oy = screen(scene.observer)
    elements.append(f'<path d="M {ox - 12:.2f} {oy:.2f} Q {ox:.2f} {oy - 10:.2f} {ox + 12:.2f} {oy:.2f} Q {ox:.2f} {oy + 10:.2f} {ox - 12:.2f} {oy:.2f}" fill="#ffffff" stroke="#333a45" stroke-width="2"/>')
    elements.append(f'<circle cx="{ox:.2f}" cy="{oy:.2f}" r="4" fill="#333a45"/>')
    elements.append(f'<text x="{ox + 15:.2f}" y="{oy + 5:.2f}" font-family="Arial" font-size="14" font-weight="700" fill="#333a45">observer</text>')

    elements.append('<rect x="710" y="505" width="205" height="118" rx="6" fill="#ffffff" stroke="#d5d9e1"/>')
    legend = [
        ("#2f5fb3", "triangle vertices"),
        ("#e23d4f", "calculation points"),
        ("#ffc533", "light sources"),
        ("#333a45", "observer"),
    ]
    for index, (color, text) in enumerate(legend):
        y = 530 + index * 22
        elements.append(f'<circle cx="732" cy="{y}" r="6" fill="{color}" stroke="#777" stroke-width="0.5"/>')
        elements.append(f'<text x="748" y="{y + 5}" font-family="Arial" font-size="13" fill="#363b44">{text}</text>')

    elements.append("</svg>")
    path.write_text("\n".join(elements), encoding="utf-8")
