from __future__ import annotations

from math import pi

from .models import LightContribution, PointResult, Scene, SurfacePoint, Triangle
from .vector import EPS, Vec3, rgb_add, rgb_mul, rgb_scale


BLACK = (0.0, 0.0, 0.0)


def triangle_normal(triangle: Triangle) -> Vec3:
    """Normal from the formula N = normalize((P2 - P0) x (P1 - P0))."""
    return (triangle.p2 - triangle.p0).cross(triangle.p1 - triangle.p0).normalized()


def orient_normal_to_observer(normal: Vec3, point: Vec3, observer: Vec3) -> Vec3:
    view = (observer - point).normalized()
    if normal.dot(view) < 0:
        return normal * -1.0
    return normal


def point_from_local_edges(triangle: Triangle, x: float, y: float) -> Vec3:
    edge01 = (triangle.p1 - triangle.p0).normalized()
    edge02 = (triangle.p2 - triangle.p0).normalized()
    return triangle.p0 + edge01 * x + edge02 * y


def point_inside_triangle_by_lengths(triangle: Triangle, x: float, y: float) -> bool:
    edge01_len = (triangle.p1 - triangle.p0).length()
    edge02_len = (triangle.p2 - triangle.p0).length()
    p = point_from_local_edges(triangle, x, y)
    area = (triangle.p1 - triangle.p0).cross(triangle.p2 - triangle.p0).length()
    a0 = (triangle.p1 - p).cross(triangle.p2 - p).length()
    a1 = (p - triangle.p0).cross(triangle.p2 - triangle.p0).length()
    a2 = (triangle.p1 - triangle.p0).cross(p - triangle.p0).length()
    return x >= -EPS and y >= -EPS and x <= edge01_len + EPS and y <= edge02_len + EPS and abs((a0 + a1 + a2) - area) <= 1e-8


def calculate_point(scene: Scene, surface_point: SurfacePoint) -> PointResult:
    point = point_from_local_edges(scene.triangle, surface_point.x, surface_point.y)
    normal = orient_normal_to_observer(triangle_normal(scene.triangle), point, scene.observer)
    view_dir = (scene.observer - point).normalized()

    total = BLACK
    contributions: list[LightContribution] = []

    for light in scene.lights:
        # In the task s = P_T - P_L is the light propagation direction.
        from_light_to_point = point - light.position
        distance2 = from_light_to_point.length2()
        if distance2 < EPS:
            raise ValueError(f"Point {surface_point.name} matches light {light.name} position")

        distance = distance2 ** 0.5
        light_dir_to_surface = from_light_to_point / distance
        surface_to_light = light_dir_to_surface * -1.0

        cos_theta = max(0.0, light_dir_to_surface.dot(light.axis.normalized()))
        cos_alpha = max(0.0, surface_to_light.dot(normal))

        emitted = rgb_scale(light.intensity_rgb, cos_theta)
        irradiance = rgb_scale(emitted, cos_alpha / distance2)

        if cos_alpha <= 0.0 or cos_theta <= 0.0:
            brdf = BLACK
            radiance = BLACK
        else:
            half_vector_sum = view_dir + surface_to_light
            if half_vector_sum.length() < EPS:
                specular_factor = 0.0
            else:
                half_vector = half_vector_sum.normalized()
                # The highlight is used only when the half-vector is directed with the oriented normal.
                specular_factor = max(0.0, half_vector.dot(normal)) ** scene.material.ke
            reflectance = scene.material.kd + scene.material.ks * specular_factor
            brdf = rgb_scale(scene.material.color_rgb, reflectance)
            radiance = rgb_scale(rgb_mul(irradiance, brdf), 1.0 / pi)

        total = rgb_add(total, radiance)
        contributions.append(
            LightContribution(
                light_name=light.name,
                distance2=distance2,
                cos_alpha=cos_alpha,
                cos_theta=cos_theta,
                irradiance_rgb=irradiance,
                brdf_rgb=brdf,
                radiance_rgb=radiance,
            )
        )

    return PointResult(
        name=surface_point.name,
        local_xy=(surface_point.x, surface_point.y),
        position=point,
        normal=normal,
        brightness_rgb=total,
        contributions=contributions,
    )


def calculate_scene(scene: Scene, points: list[SurfacePoint]) -> list[PointResult]:
    return [calculate_point(scene, point) for point in points]

