#!/usr/bin/env python3
"""Denoise a rendered PPM image with an edge-preserving bilateral filter."""

from __future__ import annotations

import argparse
import math
from pathlib import Path


Pixel = tuple[int, int, int]


def _read_token(data: bytes, pos: int) -> tuple[str, int]:
    n = len(data)
    while pos < n:
        c = data[pos]
        if c == 35:  # '#'
            while pos < n and data[pos] not in (10, 13):
                pos += 1
        elif chr(c).isspace():
            pos += 1
        else:
            break

    start = pos
    while pos < n and not chr(data[pos]).isspace() and data[pos] != 35:
        pos += 1
    return data[start:pos].decode("ascii"), pos


def read_ppm(path: Path) -> tuple[int, int, int, list[Pixel]]:
    data = path.read_bytes()
    pos = 0

    magic, pos = _read_token(data, pos)
    if magic not in {"P3", "P6"}:
        raise ValueError(f"Unsupported PPM format {magic!r}; expected P3 or P6")

    width_s, pos = _read_token(data, pos)
    height_s, pos = _read_token(data, pos)
    maxval_s, pos = _read_token(data, pos)
    width, height, maxval = int(width_s), int(height_s), int(maxval_s)

    if maxval <= 0 or maxval > 255:
        raise ValueError("Only 8-bit PPM files with maxval in 1..255 are supported")

    count = width * height * 3
    if magic == "P3":
        values: list[int] = []
        for _ in range(count):
            token, pos = _read_token(data, pos)
            if not token:
                raise ValueError("PPM ended before all pixel values were read")
            values.append(round(int(token) * 255 / maxval))
    else:
        while pos < len(data) and chr(data[pos]).isspace():
            pos += 1
        raw = data[pos : pos + count]
        if len(raw) != count:
            raise ValueError("PPM ended before all binary pixel values were read")
        values = list(raw)
        if maxval != 255:
            values = [round(v * 255 / maxval) for v in values]

    pixels = [(values[i], values[i + 1], values[i + 2]) for i in range(0, count, 3)]
    return width, height, 255, pixels


def write_ppm(path: Path, width: int, height: int, maxval: int, pixels: list[Pixel]) -> None:
    with path.open("w", encoding="ascii") as f:
        f.write(f"P3\n{width} {height}\n{maxval}\n")
        for i, (r, g, b) in enumerate(pixels):
            f.write(f"{r} {g} {b}")
            f.write("\n" if (i + 1) % 4 == 0 else "  ")


def _clamp_to_byte(value: float) -> int:
    return max(0, min(255, int(round(value))))


def bilateral_filter(
    pixels: list[Pixel],
    width: int,
    height: int,
    radius: int,
    sigma_spatial: float,
    sigma_range: float,
    passes: int,
) -> list[Pixel]:
    if radius < 1:
        return pixels[:]
    if sigma_spatial <= 0 or sigma_range <= 0:
        raise ValueError("sigma_spatial and sigma_range must be positive")

    current = pixels[:]
    range_scale = 2.0 * (sigma_range * 255.0) ** 2
    spatial: list[tuple[int, int, float]] = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            d2 = dx * dx + dy * dy
            if d2 <= radius * radius:
                weight = math.exp(-d2 / (2.0 * sigma_spatial * sigma_spatial))
                spatial.append((dx, dy, weight))

    for _ in range(passes):
        filtered: list[Pixel] = []
        for y in range(height):
            row = y * width
            for x in range(width):
                cr, cg, cb = current[row + x]
                wr = wg = wb = wsum = 0.0

                for dx, dy, sw in spatial:
                    nx = min(width - 1, max(0, x + dx))
                    ny = min(height - 1, max(0, y + dy))
                    nr, ng, nb = current[ny * width + nx]
                    dr = nr - cr
                    dg = ng - cg
                    db = nb - cb
                    color_d2 = dr * dr + dg * dg + db * db
                    weight = sw * math.exp(-color_d2 / range_scale)
                    wr += weight * nr
                    wg += weight * ng
                    wb += weight * nb
                    wsum += weight

                filtered.append(
                    (
                        _clamp_to_byte(wr / wsum),
                        _clamp_to_byte(wg / wsum),
                        _clamp_to_byte(wb / wsum),
                    )
                )
        current = filtered
    return current


def mean_absolute_difference(a: list[Pixel], b: list[Pixel]) -> float:
    total = 0
    for pa, pb in zip(a, b):
        total += abs(pa[0] - pb[0]) + abs(pa[1] - pb[1]) + abs(pa[2] - pb[2])
    return total / (len(a) * 3)


def high_frequency_energy(pixels: list[Pixel], width: int, height: int) -> float:
    if width < 3 or height < 3:
        return 0.0
    total = 0.0
    samples = 0
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            i = y * width + x
            for c in range(3):
                center = pixels[i][c] * 4
                lap = (
                    center
                    - pixels[i - 1][c]
                    - pixels[i + 1][c]
                    - pixels[i - width][c]
                    - pixels[i + width][c]
                )
                total += abs(lap)
                samples += 1
    return total / samples


def gradient_magnitudes(pixels: list[Pixel], width: int, height: int) -> list[float]:
    if width < 3 or height < 3:
        return []
    gradients: list[float] = []
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            left = pixels[y * width + x - 1]
            right = pixels[y * width + x + 1]
            up = pixels[(y - 1) * width + x]
            down = pixels[(y + 1) * width + x]
            dx = sum((right[c] - left[c]) ** 2 for c in range(3))
            dy = sum((down[c] - up[c]) ** 2 for c in range(3))
            gradients.append(math.sqrt(dx + dy))
    return gradients


def edge_retention(before: list[Pixel], after: list[Pixel], width: int, height: int) -> float:
    before_g = gradient_magnitudes(before, width, height)
    after_g = gradient_magnitudes(after, width, height)
    if not before_g:
        return 0.0

    threshold = sorted(before_g)[int(len(before_g) * 0.90)]
    edge_indices = [i for i, g in enumerate(before_g) if g >= threshold]
    before_mean = sum(before_g[i] for i in edge_indices) / len(edge_indices)
    after_mean = sum(after_g[i] for i in edge_indices) / len(edge_indices)
    return after_mean / before_mean if before_mean else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply bilateral denoising to render_better.ppm while preserving image edges."
    )
    parser.add_argument("input", nargs="?", default="render_better.ppm", type=Path)
    parser.add_argument("-o", "--output", default=Path("render_better_bilateral.ppm"), type=Path)
    parser.add_argument("--radius", default=4, type=int)
    parser.add_argument("--sigma-spatial", default=2.2, type=float)
    parser.add_argument(
        "--sigma-range",
        default=0.12,
        type=float,
        help="Color-domain sigma in normalized RGB units; smaller values preserve stronger edges.",
    )
    parser.add_argument("--passes", default=1, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    width, height, maxval, pixels = read_ppm(args.input)
    filtered = bilateral_filter(
        pixels,
        width,
        height,
        radius=args.radius,
        sigma_spatial=args.sigma_spatial,
        sigma_range=args.sigma_range,
        passes=args.passes,
    )
    write_ppm(args.output, width, height, maxval, filtered)

    before_hf = high_frequency_energy(pixels, width, height)
    after_hf = high_frequency_energy(filtered, width, height)
    print(f"input: {args.input} ({width}x{height})")
    print(f"output: {args.output}")
    print(
        "filter: "
        f"radius={args.radius}, sigma_spatial={args.sigma_spatial}, "
        f"sigma_range={args.sigma_range}, passes={args.passes}"
    )
    print(f"mean absolute change: {mean_absolute_difference(pixels, filtered):.3f} levels")
    print(f"high-frequency energy: {before_hf:.3f} -> {after_hf:.3f}")
    print(f"strong-edge retention: {edge_retention(pixels, filtered, width, height):.3f}")


if __name__ == "__main__":
    main()
