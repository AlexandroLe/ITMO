from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .models import Light, Material, PointResult, Scene, SurfacePoint, Triangle
from .vector import rgb_from_list, vec3_from_list


def load_input(path: Path) -> tuple[Scene, list[SurfacePoint]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    scene = Scene(
        triangle=Triangle(
            p0=vec3_from_list(data["triangle"]["p0"]),
            p1=vec3_from_list(data["triangle"]["p1"]),
            p2=vec3_from_list(data["triangle"]["p2"]),
        ),
        observer=vec3_from_list(data["observer"]),
        material=Material(
            color_rgb=rgb_from_list(data["material"]["color_rgb"]),
            kd=float(data["material"]["kd"]),
            ks=float(data["material"]["ks"]),
            ke=float(data["material"]["ke"]),
        ),
        lights=[
            Light(
                name=str(light.get("name", f"light_{index + 1}")),
                position=vec3_from_list(light["position"]),
                axis=vec3_from_list(light["axis"]).normalized(),
                intensity_rgb=rgb_from_list(light["intensity_rgb"]),
            )
            for index, light in enumerate(data["lights"])
        ],
    )
    points = [
        SurfacePoint(
            name=str(point.get("name", f"point_{index + 1}")),
            x=float(point["x"]),
            y=float(point["y"]),
        )
        for index, point in enumerate(data["points"])
    ]
    return scene, points


def point_result_to_dict(result: PointResult) -> dict[str, Any]:
    return {
        "name": result.name,
        "local_xy": list(result.local_xy),
        "position": list(result.position.as_tuple()),
        "normal": list(result.normal.as_tuple()),
        "brightness_rgb": list(result.brightness_rgb),
        "contributions": [
            {
                "light_name": item.light_name,
                "distance2": item.distance2,
                "cos_alpha": item.cos_alpha,
                "cos_theta": item.cos_theta,
                "irradiance_rgb": list(item.irradiance_rgb),
                "brdf_rgb": list(item.brdf_rgb),
                "radiance_rgb": list(item.radiance_rgb),
            }
            for item in result.contributions
        ],
    }


def write_json(path: Path, results: list[PointResult]) -> None:
    payload = [point_result_to_dict(result) for result in results]
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, results: list[PointResult]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["point", "x", "y", "px", "py", "pz", "nx", "ny", "nz", "r", "g", "b"])
        for result in results:
            writer.writerow(
                [
                    result.name,
                    result.local_xy[0],
                    result.local_xy[1],
                    result.position.x,
                    result.position.y,
                    result.position.z,
                    result.normal.x,
                    result.normal.y,
                    result.normal.z,
                    result.brightness_rgb[0],
                    result.brightness_rgb[1],
                    result.brightness_rgb[2],
                ]
            )

