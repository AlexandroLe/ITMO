from __future__ import annotations

import argparse
import math
import os
import random
import time
from dataclasses import dataclass
from typing import Iterable


EPS = 1e-5
GAMMA = 1.0 / 2.2


@dataclass(frozen=True)
class Vec3:
    x: float
    y: float
    z: float

    def __add__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, value: float | "Vec3") -> "Vec3":
        if isinstance(value, Vec3):
            return Vec3(self.x * value.x, self.y * value.y, self.z * value.z)
        return Vec3(self.x * value, self.y * value, self.z * value)

    __rmul__ = __mul__

    def __truediv__(self, value: float) -> "Vec3":
        return Vec3(self.x / value, self.y / value, self.z / value)

    def dot(self, other: "Vec3") -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: "Vec3") -> "Vec3":
        return Vec3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def length(self) -> float:
        return math.sqrt(self.dot(self))

    def normalized(self) -> "Vec3":
        length = self.length()
        if length <= 0.0:
            return self
        return self / length

    def max_component(self) -> float:
        return max(self.x, self.y, self.z)


BLACK = Vec3(0.0, 0.0, 0.0)
WHITE = Vec3(1.0, 1.0, 1.0)


def luminance(c: Vec3) -> float:
    return 0.2126 * c.x + 0.7152 * c.y + 0.0722 * c.z


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


@dataclass(frozen=True)
class Ray:
    origin: Vec3
    direction: Vec3


@dataclass(frozen=True)
class Material:
    name: str
    diffuse: Vec3
    mirror: Vec3 = BLACK
    emission: Vec3 = BLACK

    def check_energy(self) -> None:
        total = self.diffuse + self.mirror
        if total.x > 1.0 + EPS or total.y > 1.0 + EPS or total.z > 1.0 + EPS:
            raise ValueError(f"Material {self.name} reflects more than one unit of energy")


@dataclass(frozen=True)
class Triangle:
    a: Vec3
    b: Vec3
    c: Vec3
    material: Material

    @property
    def normal(self) -> Vec3:
        return (self.b - self.a).cross(self.c - self.a).normalized()

    @property
    def area(self) -> float:
        return 0.5 * (self.b - self.a).cross(self.c - self.a).length()

    def sample(self, rng: random.Random) -> Vec3:
        r1 = math.sqrt(rng.random())
        r2 = rng.random()
        return self.a * (1.0 - r1) + self.b * (r1 * (1.0 - r2)) + self.c * (r1 * r2)


@dataclass(frozen=True)
class PointLight:
    position: Vec3
    intensity: Vec3

    def flux(self) -> Vec3:
        # Isotropic point source: Phi [W] = integral I dOmega = 4*pi*I.
        return self.intensity * (4.0 * math.pi)


@dataclass(frozen=True)
class AreaLight:
    triangle: Triangle
    radiance: Vec3

    def flux(self) -> Vec3:
        # Lambert emitter: Phi [W] = integral_A integral_hemisphere L cos(theta)dOmega dA.
        return self.radiance * (math.pi * self.triangle.area)


@dataclass
class Hit:
    t: float
    point: Vec3
    normal: Vec3
    triangle: Triangle


@dataclass
class Scene:
    triangles: list[Triangle]
    point_lights: list[PointLight]
    area_lights: list[AreaLight]
    light_cdf: list[float]
    light_flux_sum: float

    @classmethod
    def build(cls, triangles: list[Triangle], point_lights: list[PointLight], area_lights: list[AreaLight]) -> "Scene":
        weights = [luminance(light.flux()) for light in [*point_lights, *area_lights]]
        flux_sum = sum(weights)
        if flux_sum <= 0.0:
            raise ValueError("Scene must contain at least one light with positive flux")
        cdf: list[float] = []
        acc = 0.0
        for w in weights:
            acc += w / flux_sum
            cdf.append(acc)
        cdf[-1] = 1.0
        for tri in triangles:
            tri.material.check_energy()
        return cls(triangles, point_lights, area_lights, cdf, flux_sum)

    def intersect(self, ray: Ray, max_t: float = float("inf")) -> Hit | None:
        best_t = max_t
        best: Hit | None = None
        for tri in self.triangles:
            hit_t = intersect_triangle(ray, tri, best_t)
            if hit_t is not None:
                best_t = hit_t
                n = tri.normal
                if n.dot(ray.direction) > 0.0:
                    n = n * -1.0
                best = Hit(hit_t, ray.origin + ray.direction * hit_t, n, tri)
        return best

    def visible(self, origin: Vec3, target: Vec3) -> bool:
        to_target = target - origin
        dist = to_target.length()
        if dist <= EPS:
            return False
        ray = Ray(origin, to_target / dist)
        return self.intersect(ray, dist - EPS) is None

    def sample_light(self, rng: random.Random) -> tuple[PointLight | AreaLight, float]:
        u = rng.random()
        idx = 0
        while idx < len(self.light_cdf) - 1 and u > self.light_cdf[idx]:
            idx += 1
        prev = 0.0 if idx == 0 else self.light_cdf[idx - 1]
        p_select = self.light_cdf[idx] - prev
        lights: list[PointLight | AreaLight] = [*self.point_lights, *self.area_lights]
        return lights[idx], p_select


def intersect_triangle(ray: Ray, tri: Triangle, max_t: float) -> float | None:
    e1 = tri.b - tri.a
    e2 = tri.c - tri.a
    p = ray.direction.cross(e2)
    det = e1.dot(p)
    if -EPS < det < EPS:
        return None
    inv_det = 1.0 / det
    s = ray.origin - tri.a
    u = inv_det * s.dot(p)
    if u < 0.0 or u > 1.0:
        return None
    q = s.cross(e1)
    v = inv_det * ray.direction.dot(q)
    if v < 0.0 or u + v > 1.0:
        return None
    t = inv_det * e2.dot(q)
    if EPS < t < max_t:
        return t
    return None


def orthonormal_basis(n: Vec3) -> tuple[Vec3, Vec3]:
    if abs(n.x) > abs(n.z):
        tangent = Vec3(-n.y, n.x, 0.0).normalized()
    else:
        tangent = Vec3(0.0, -n.z, n.y).normalized()
    bitangent = n.cross(tangent)
    return tangent, bitangent


def cosine_sample_hemisphere(n: Vec3, rng: random.Random) -> Vec3:
    r1 = rng.random()
    r2 = rng.random()
    radius = math.sqrt(r1)
    phi = 2.0 * math.pi * r2
    x = radius * math.cos(phi)
    y = radius * math.sin(phi)
    z = math.sqrt(max(0.0, 1.0 - r1))
    tangent, bitangent = orthonormal_basis(n)
    return (tangent * x + bitangent * y + n * z).normalized()


def reflect(direction: Vec3, normal: Vec3) -> Vec3:
    return (direction - normal * (2.0 * direction.dot(normal))).normalized()


def direct_light(scene: Scene, hit: Hit, rng: random.Random) -> Vec3:
    light, p_select = scene.sample_light(rng)
    if p_select <= 0.0:
        return BLACK
    material = hit.triangle.material
    f_lambert = material.diffuse * (1.0 / math.pi)
    if f_lambert.max_component() <= 0.0:
        return BLACK

    if isinstance(light, PointLight):
        to_light = light.position - hit.point
        dist2 = to_light.dot(to_light)
        dist = math.sqrt(dist2)
        wi = to_light / dist
        cos_x = max(0.0, hit.normal.dot(wi))
        if cos_x <= 0.0 or not scene.visible(hit.point + hit.normal * EPS, light.position):
            return BLACK
        return f_lambert * light.intensity * (cos_x / (dist2 * p_select))

    light_point = light.triangle.sample(rng)
    to_light = light_point - hit.point
    dist2 = to_light.dot(to_light)
    dist = math.sqrt(dist2)
    wi = to_light / dist
    light_normal = light.triangle.normal
    cos_x = max(0.0, hit.normal.dot(wi))
    cos_l = max(0.0, light_normal.dot(wi * -1.0))
    if cos_x <= 0.0 or cos_l <= 0.0:
        return BLACK
    if not scene.visible(hit.point + hit.normal * EPS, light_point):
        return BLACK
    pdf_area = 1.0 / light.triangle.area
    return f_lambert * light.radiance * (cos_x * cos_l / (dist2 * p_select * pdf_area))


def trace(scene: Scene, ray: Ray, rng: random.Random, max_depth: int) -> Vec3:
    radiance = BLACK
    throughput = WHITE
    specular_path = True

    for depth in range(max_depth):
        hit = scene.intersect(ray)
        if hit is None:
            break

        mat = hit.triangle.material

        if mat.emission.max_component() > 0.0 and specular_path:
            radiance += throughput * mat.emission

        if mat.diffuse.max_component() > 0.0:
            radiance += throughput * direct_light(scene, hit, rng)

        w_diffuse = luminance(mat.diffuse)
        w_mirror = luminance(mat.mirror)
        w_sum = w_diffuse + w_mirror
        if w_sum <= 0.0:
            break

        keep = 1.0
        if depth >= 2:
            keep = clamp01(throughput.max_component())
            keep = max(0.05, min(0.95, keep))

        p_rr = 1.0 - keep
        p_diffuse = keep * (w_diffuse / w_sum)
        p_mirror = keep * (w_mirror / w_sum)

        u = rng.random()

        if u < p_rr:
            break

        if keep < 1.0:
            u = (u - p_rr) / keep

        if u < (w_diffuse / w_sum):
            event_pdf = p_diffuse

            new_dir = cosine_sample_hemisphere(hit.normal, rng)
            cos_theta = max(0.0, hit.normal.dot(new_dir))
            bsdf = mat.diffuse * (1.0 / math.pi)
            pdf_dir = cos_theta / math.pi

            if pdf_dir <= 0.0 or event_pdf <= 0.0:
                break

            throughput = throughput * bsdf * (cos_theta / (pdf_dir * event_pdf))
            specular_path = False

        else:
            event_pdf = p_mirror

            new_dir = reflect(ray.direction, hit.normal)
            if event_pdf <= 0.0:
                break

            throughput = throughput * mat.mirror * (1.0 / event_pdf)
            specular_path = True

        ray = Ray(hit.point + hit.normal * EPS, new_dir)

    return radiance


def make_camera_ray(x: float, y: float, width: int, height: int, fov_deg: float) -> Ray:
    origin = Vec3(0.0, 1.0, 4.2)
    forward = Vec3(0.0, -0.08, -1.0).normalized()
    right = forward.cross(Vec3(0.0, 1.0, 0.0)).normalized()
    up = right.cross(forward).normalized()
    aspect = width / height
    scale = math.tan(math.radians(fov_deg) * 0.5)
    px = (2.0 * x / width - 1.0) * aspect * scale
    py = (1.0 - 2.0 * y / height) * scale
    return Ray(origin, (forward + right * px + up * py).normalized())


def render(scene: Scene, width: int, height: int, samples: int, max_depth: int, seed: int) -> list[Vec3]:
    rng = random.Random(seed)
    pixels = [BLACK for _ in range(width * height)]
    counts = [0 for _ in range(width * height)]

    # Pixel state is explicit: each pass samples every pixel once, and the counter
    # can be used for adaptive policies without changing the estimator.
    for sample in range(samples):
        started = time.time()
        for y in range(height):
            for x in range(width):
                idx = y * width + x
                jx = rng.random()
                jy = rng.random()
                ray = make_camera_ray(x + jx, y + jy, width, height, 50.0)
                pixels[idx] += trace(scene, ray, rng, max_depth)
                counts[idx] += 1
        elapsed = time.time() - started
        print(f"sample {sample + 1}/{samples}: {elapsed:.2f}s", flush=True)

    return [pixels[i] / max(1, counts[i]) for i in range(width * height)]


def write_ppm(path: str, pixels: Iterable[Vec3], width: int, height: int, exposure: float) -> None:
    data = list(pixels)
    if exposure <= 0.0:
        max_l = max((luminance(p) for p in data), default=1.0)
        exposure = 1.0 / max(max_l, EPS)
    with open(path, "w", encoding="ascii") as f:
        f.write(f"P3\n{width} {height}\n255\n")
        for p in data:
            mapped = Vec3(
                clamp01(p.x * exposure),
                clamp01(p.y * exposure),
                clamp01(p.z * exposure),
            )
            r = int(255.0 * (mapped.x ** GAMMA) + 0.5)
            g = int(255.0 * (mapped.y ** GAMMA) + 0.5)
            b = int(255.0 * (mapped.z ** GAMMA) + 0.5)
            f.write(f"{r} {g} {b}\n")


def rect(a: Vec3, b: Vec3, c: Vec3, d: Vec3, material: Material) -> list[Triangle]:
    return [Triangle(a, b, c, material), Triangle(a, c, d, material)]


def cornell_scene() -> Scene:
    white = Material("white lambert", Vec3(0.72, 0.72, 0.72))
    red = Material("red lambert", Vec3(0.75, 0.12, 0.10))
    green = Material("green lambert", Vec3(0.10, 0.62, 0.16))
    mirror = Material("mirror+diffuse", Vec3(0.12, 0.12, 0.12), Vec3(0.72, 0.72, 0.72))
    light_mat = Material("visible area light", BLACK, BLACK, Vec3(12.0, 11.0, 9.0))

    tris: list[Triangle] = []
    tris += rect(Vec3(-1.5, 0, -1.5), Vec3(1.5, 0, -1.5), Vec3(1.5, 0, 1.5), Vec3(-1.5, 0, 1.5), white)
    tris += rect(Vec3(-1.5, 3, -1.5), Vec3(-1.5, 3, 1.5), Vec3(1.5, 3, 1.5), Vec3(1.5, 3, -1.5), white)
    tris += rect(Vec3(-1.5, 0, -1.5), Vec3(-1.5, 3, -1.5), Vec3(1.5, 3, -1.5), Vec3(1.5, 0, -1.5), white)
    tris += rect(Vec3(-1.5, 0, 1.5), Vec3(-1.5, 3, 1.5), Vec3(-1.5, 3, -1.5), Vec3(-1.5, 0, -1.5), red)
    tris += rect(Vec3(1.5, 0, -1.5), Vec3(1.5, 3, -1.5), Vec3(1.5, 3, 1.5), Vec3(1.5, 0, 1.5), green)

    tris += rect(Vec3(-0.9, 0.0, -0.6), Vec3(-0.15, 0.0, -0.75), Vec3(-0.15, 1.35, -0.75), Vec3(-0.9, 1.35, -0.6), white)
    tris += rect(Vec3(-0.15, 0.0, -0.75), Vec3(0.05, 0.0, 0.15), Vec3(0.05, 1.35, 0.15), Vec3(-0.15, 1.35, -0.75), white)
    tris += rect(Vec3(0.05, 0.0, 0.15), Vec3(-0.7, 0.0, 0.3), Vec3(-0.7, 1.35, 0.3), Vec3(0.05, 1.35, 0.15), white)
    tris += rect(Vec3(-0.7, 0.0, 0.3), Vec3(-0.9, 0.0, -0.6), Vec3(-0.9, 1.35, -0.6), Vec3(-0.7, 1.35, 0.3), white)
    tris += rect(Vec3(-0.9, 1.35, -0.6), Vec3(-0.15, 1.35, -0.75), Vec3(0.05, 1.35, 0.15), Vec3(-0.7, 1.35, 0.3), white)

    tris += rect(Vec3(0.45, 0.0, -0.15), Vec3(1.05, 0.0, -0.15), Vec3(1.05, 0.85, -0.15), Vec3(0.45, 0.85, -0.15), mirror)
    tris += rect(Vec3(1.05, 0.0, -0.15), Vec3(1.05, 0.0, 0.55), Vec3(1.05, 0.85, 0.55), Vec3(1.05, 0.85, -0.15), mirror)
    tris += rect(Vec3(1.05, 0.0, 0.55), Vec3(0.45, 0.0, 0.55), Vec3(0.45, 0.85, 0.55), Vec3(1.05, 0.85, 0.55), mirror)
    tris += rect(Vec3(0.45, 0.0, 0.55), Vec3(0.45, 0.0, -0.15), Vec3(0.45, 0.85, -0.15), Vec3(0.45, 0.85, 0.55), mirror)
    tris += rect(Vec3(0.45, 0.85, -0.15), Vec3(1.05, 0.85, -0.15), Vec3(1.05, 0.85, 0.55), Vec3(0.45, 0.85, 0.55), mirror)

    light_tri = Triangle(Vec3(-0.35, 2.99, -0.45), Vec3(0.35, 2.99, -0.45), Vec3(0.0, 2.99, 0.35), light_mat)
    tris.append(light_tri)

    point_lights = [PointLight(Vec3(-0.85, 1.75, 0.85), Vec3(1.4, 1.8, 3.3))]
    area_lights = [AreaLight(light_tri, light_mat.emission)]
    return Scene.build(tris, point_lights, area_lights)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Small path tracer for lab 4.")
    parser.add_argument("--width", type=int, default=500)
    parser.add_argument("--height", type=int, default=500)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--exposure", type=float, default=0.0, help="0 means normalize by max image luminance")
    parser.add_argument("--output", default="render.ppm")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scene = cornell_scene()
    started = time.time()
    pixels = render(scene, args.width, args.height, args.samples, args.max_depth, args.seed)
    write_ppm(args.output, pixels, args.width, args.height, args.exposure)
    elapsed = time.time() - started
    print(f"wrote {os.path.abspath(args.output)} in {elapsed:.2f}s")


if __name__ == "__main__":
    main()
