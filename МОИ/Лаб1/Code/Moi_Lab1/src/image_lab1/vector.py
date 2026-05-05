from __future__ import annotations

from dataclasses import dataclass
from math import sqrt


EPS = 1e-12


@dataclass(frozen=True)
class Vec3:
    x: float
    y: float
    z: float

    def __add__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, value: float) -> "Vec3":
        return Vec3(self.x * value, self.y * value, self.z * value)

    __rmul__ = __mul__

    def __truediv__(self, value: float) -> "Vec3":
        if abs(value) < EPS:
            raise ValueError("Division by near-zero value")
        return Vec3(self.x / value, self.y / value, self.z / value)

    def dot(self, other: "Vec3") -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: "Vec3") -> "Vec3":
        return Vec3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def length2(self) -> float:
        return self.dot(self)

    def length(self) -> float:
        return sqrt(self.length2())

    def normalized(self) -> "Vec3":
        length = self.length()
        if length < EPS:
            raise ValueError("Cannot normalize zero vector")
        return self / length

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)


def vec3_from_list(values: list[float]) -> Vec3:
    if len(values) != 3:
        raise ValueError(f"Expected 3 values, got {len(values)}")
    return Vec3(float(values[0]), float(values[1]), float(values[2]))


def rgb_mul(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (a[0] * b[0], a[1] * b[1], a[2] * b[2])


def rgb_scale(a: tuple[float, float, float], value: float) -> tuple[float, float, float]:
    return (a[0] * value, a[1] * value, a[2] * value)


def rgb_add(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def rgb_from_list(values: list[float]) -> tuple[float, float, float]:
    if len(values) != 3:
        raise ValueError(f"Expected RGB with 3 values, got {len(values)}")
    return (float(values[0]), float(values[1]), float(values[2]))

