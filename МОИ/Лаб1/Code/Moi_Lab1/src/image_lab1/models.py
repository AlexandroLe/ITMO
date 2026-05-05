from __future__ import annotations

from dataclasses import dataclass

from .vector import Vec3


RGB = tuple[float, float, float]


@dataclass(frozen=True)
class Triangle:
    p0: Vec3
    p1: Vec3
    p2: Vec3


@dataclass(frozen=True)
class Light:
    name: str
    position: Vec3
    axis: Vec3
    intensity_rgb: RGB


@dataclass(frozen=True)
class Material:
    color_rgb: RGB
    kd: float
    ks: float
    ke: float


@dataclass(frozen=True)
class Scene:
    triangle: Triangle
    observer: Vec3
    material: Material
    lights: list[Light]


@dataclass(frozen=True)
class SurfacePoint:
    name: str
    x: float
    y: float


@dataclass(frozen=True)
class LightContribution:
    light_name: str
    distance2: float
    cos_alpha: float
    cos_theta: float
    irradiance_rgb: RGB
    brdf_rgb: RGB
    radiance_rgb: RGB


@dataclass(frozen=True)
class PointResult:
    name: str
    local_xy: tuple[float, float]
    position: Vec3
    normal: Vec3
    brightness_rgb: RGB
    contributions: list[LightContribution]

