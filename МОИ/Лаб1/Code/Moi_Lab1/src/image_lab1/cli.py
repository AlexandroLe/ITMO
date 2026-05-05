from __future__ import annotations

import argparse
from pathlib import Path

from .io import load_input, write_csv, write_json
from .lighting import calculate_scene
from .render import write_ppm_render, write_scene_scheme_svg


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="moi-lab1",
        description="Calculate RGB brightness for points on a triangle surface.",
    )
    parser.add_argument("--scenario", type=Path, default=Path("examples/demo_scenario.json"))
    parser.add_argument("--out", type=Path, default=Path("exports"))
    parser.add_argument("--render", action="store_true", help="Create a simple PPM visualization.")
    parser.add_argument("--scheme", action="store_true", help="Create a 3D SVG scene scheme.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    scene, points = load_input(args.scenario)
    results = calculate_scene(scene, points)

    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / "results.json", results)
    write_csv(args.out / "results.csv", results)
    if args.render:
        write_ppm_render(args.out / "triangle_render.ppm", scene)
    if args.scheme:
        write_scene_scheme_svg(args.out / "scene_scheme.svg", scene, results)

    for result in results:
        rgb = result.brightness_rgb
        pos = result.position
        print(
            f"{result.name}: P=({pos.x:.4f}, {pos.y:.4f}, {pos.z:.4f}), "
            f"L_rgb=({rgb[0]:.6f}, {rgb[1]:.6f}, {rgb[2]:.6f})"
        )
        for contrib in result.contributions:
            e = contrib.irradiance_rgb
            print(
                f"  {contrib.light_name}: E_rgb=({e[0]:.6f}, {e[1]:.6f}, {e[2]:.6f})"
            )


if __name__ == "__main__":
    main()
