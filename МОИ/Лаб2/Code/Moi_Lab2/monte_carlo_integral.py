from __future__ import annotations

import math
from pathlib import Path
from dataclasses import dataclass
from typing import Callable

import random


A = 2.0
B = 5.0
TRUE_INTEGRAL = (B**3 - A**3) / 3.0
SAMPLE_COUNTS = [100, 1_000, 10_000, 100_000]
SEED = 42
PLOTS_DIR = Path("plots")


def f(x: float) -> float:
    return x * x


def true_error_estimate(n: int) -> float:
    """Estimate required by the task: Delta I = I_true / sqrt(N)."""
    return TRUE_INTEGRAL / math.sqrt(n)


def normalized_power_density(power: int) -> Callable[[float], float]:
    norm = (power + 1) / (B ** (power + 1) - A ** (power + 1))

    def density(x: float) -> float:
        return norm * (x**power)

    return density


def sample_power_density(power: int, rng: random.Random) -> float:
    """Inverse transform sampling for p(x) proportional to x**power on [A, B]."""
    u = rng.random()
    value = A ** (power + 1) + u * (B ** (power + 1) - A ** (power + 1))
    return value ** (1.0 / (power + 1))


def simple_monte_carlo(n: int, rng: random.Random) -> float:
    total = 0.0
    for _ in range(n):
        x = rng.uniform(A, B)
        total += f(x)
    return (B - A) * total / n


def stratified_monte_carlo(n: int, step: float, rng: random.Random) -> float:
    intervals: list[tuple[float, float]] = []
    left = A
    while left < B - 1e-12:
        right = min(left + step, B)
        intervals.append((left, right))
        left = right

    base = n // len(intervals)
    remainder = n % len(intervals)
    estimate = 0.0

    for index, (left, right) in enumerate(intervals):
        samples_in_stratum = base + (1 if index < remainder else 0)
        subtotal = 0.0
        for _ in range(samples_in_stratum):
            x = rng.uniform(left, right)
            subtotal += f(x)
        estimate += (right - left) * subtotal / samples_in_stratum

    return estimate


def importance_sampling(n: int, power: int, rng: random.Random) -> float:
    density = normalized_power_density(power)
    total = 0.0
    for _ in range(n):
        x = sample_power_density(power, rng)
        total += f(x) / density(x)
    return total / n


def multiple_importance_sampling(n: int, use_squared_weights: bool, rng: random.Random) -> float:
    p1 = normalized_power_density(1)
    p2 = normalized_power_density(3)
    n1 = n // 2
    n2 = n - n1

    def weight_first(x: float) -> float:
        d1 = p1(x)
        d2 = p2(x)
        if use_squared_weights:
            return (d1 * d1) / (d1 * d1 + d2 * d2)
        return d1 / (d1 + d2)

    estimate = 0.0
    subtotal = 0.0
    for _ in range(n1):
        x = sample_power_density(1, rng)
        subtotal += weight_first(x) * f(x) / p1(x)
    estimate += subtotal / n1

    subtotal = 0.0
    for _ in range(n2):
        x = sample_power_density(3, rng)
        w2 = 1.0 - weight_first(x)
        subtotal += w2 * f(x) / p2(x)
    estimate += subtotal / n2

    return estimate


def russian_roulette(n: int, survival_probability: float, rng: random.Random) -> float:
    """Uniform MC with Russian roulette survival probability R.

    A sample is either discarded or multiplied by 1/R, so the estimator remains
    unbiased in expectation.
    """
    total = 0.0
    for _ in range(n):
        x = rng.uniform(A, B)
        if rng.random() <= survival_probability:
            total += f(x) / survival_probability
    return (B - A) * total / n


@dataclass(frozen=True)
class Result:
    method: str
    n: int
    estimate: float

    @property
    def absolute_error(self) -> float:
        return abs(self.estimate - TRUE_INTEGRAL)

    @property
    def delta_i(self) -> float:
        return true_error_estimate(self.n)


def collect_results() -> list[Result]:
    rng = random.Random(SEED)
    results: list[Result] = []

    for n in SAMPLE_COUNTS:
        results.append(Result("Simple MC", n, simple_monte_carlo(n, rng)))
        results.append(Result("Stratified MC, step=1", n, stratified_monte_carlo(n, 1.0, rng)))
        results.append(Result("Stratified MC, step=0.5", n, stratified_monte_carlo(n, 0.5, rng)))

        for power in (1, 2, 3):
            results.append(Result(f"Importance sampling, p~x^{power}", n, importance_sampling(n, power, rng)))

        results.append(Result("Multiple IS, balance weights", n, multiple_importance_sampling(n, False, rng)))
        results.append(Result("Multiple IS, squared weights", n, multiple_importance_sampling(n, True, rng)))

        for cutoff in (0.5, 0.75, 0.95):
            results.append(Result(f"Russian roulette, R={cutoff}", n, russian_roulette(n, cutoff, rng)))

    return results


def print_table(results: list[Result]) -> None:
    print(f"Analytical integral for f(x)=x^2 on [{A:g}, {B:g}]: {TRUE_INTEGRAL:.10f}")
    print(f"Random seed: {SEED}")
    print()
    print("| Method | N | I_true | I_MC | Absolute error | Delta I |")
    print("|---|---:|---:|---:|---:|---:|")
    for row in results:
        print(
            f"| {row.method} | {row.n} | {TRUE_INTEGRAL:.10f} | "
            f"{row.estimate:.10f} | {row.absolute_error:.10f} | {row.delta_i:.10f} |"
        )


def grouped_by_method(results: list[Result]) -> dict[str, list[Result]]:
    grouped: dict[str, list[Result]] = {}
    for row in results:
        grouped.setdefault(row.method, []).append(row)
    for rows in grouped.values():
        rows.sort(key=lambda item: item.n)
    return grouped


def make_line_chart(
    grouped: dict[str, list[Result]],
    filename: Path,
    title: str,
    y_label: str,
    value_getter: Callable[[Result], float],
    draw_true_integral: bool = False,
) -> None:
    width = 1250
    height = 760
    margin_left = 90
    margin_right = 360
    margin_top = 70
    margin_bottom = 90
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    x_values = [math.log10(n) for n in SAMPLE_COUNTS]
    x_min = min(x_values)
    x_max = max(x_values)

    all_y_values = [value_getter(row) for rows in grouped.values() for row in rows]
    if draw_true_integral:
        all_y_values.append(TRUE_INTEGRAL)
    y_min = min(all_y_values)
    y_max = max(all_y_values)
    if math.isclose(y_min, y_max):
        y_min -= 1.0
        y_max += 1.0
    padding = (y_max - y_min) * 0.08
    y_min -= padding
    y_max += padding

    def sx(n: int) -> float:
        return margin_left + (math.log10(n) - x_min) / (x_max - x_min) * plot_width

    def sy(value: float) -> float:
        return margin_top + (y_max - value) / (y_max - y_min) * plot_height

    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
        "#111827",
    ]

    lines: list[str] = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="{0}" height="{1}" viewBox="0 0 {0} {1}">'.format(
            width, height
        ),
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2:.1f}" y="35" text-anchor="middle" '
        'font-family="Arial" font-size="22" font-weight="700">'
        f"{title}</text>",
        f'<line x1="{margin_left}" y1="{margin_top + plot_height}" '
        f'x2="{margin_left + plot_width}" y2="{margin_top + plot_height}" stroke="#222"/>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" '
        f'y2="{margin_top + plot_height}" stroke="#222"/>',
    ]

    for n in SAMPLE_COUNTS:
        x = sx(n)
        lines.append(
            f'<line x1="{x:.2f}" y1="{margin_top}" x2="{x:.2f}" '
            f'y2="{margin_top + plot_height}" stroke="#e5e7eb"/>'
        )
        lines.append(
            f'<text x="{x:.2f}" y="{margin_top + plot_height + 28}" '
            f'text-anchor="middle" font-family="Arial" font-size="13">{n}</text>'
        )

    for i in range(6):
        value = y_min + (y_max - y_min) * i / 5
        y = sy(value)
        lines.append(
            f'<line x1="{margin_left}" y1="{y:.2f}" x2="{margin_left + plot_width}" '
            f'y2="{y:.2f}" stroke="#e5e7eb"/>'
        )
        lines.append(
            f'<text x="{margin_left - 12}" y="{y + 4:.2f}" text-anchor="end" '
            f'font-family="Arial" font-size="12">{value:.3f}</text>'
        )

    lines.append(
        f'<text x="{margin_left + plot_width / 2:.1f}" y="{height - 28}" '
        'text-anchor="middle" font-family="Arial" font-size="15">Sample size N</text>'
    )
    lines.append(
        f'<text x="24" y="{margin_top + plot_height / 2:.1f}" text-anchor="middle" '
        f'font-family="Arial" font-size="15" transform="rotate(-90 24 '
        f'{margin_top + plot_height / 2:.1f})">{y_label}</text>'
    )

    if draw_true_integral:
        y = sy(TRUE_INTEGRAL)
        lines.append(
            f'<line x1="{margin_left}" y1="{y:.2f}" x2="{margin_left + plot_width}" '
            f'y2="{y:.2f}" stroke="#000" stroke-width="2" stroke-dasharray="8 6"/>'
        )

    for method_index, (method, rows) in enumerate(grouped.items()):
        color = colors[method_index % len(colors)]
        points = " ".join(f"{sx(row.n):.2f},{sy(value_getter(row)):.2f}" for row in rows)
        lines.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="2.2" points="{points}"/>'
        )
        for row in rows:
            lines.append(
                f'<circle cx="{sx(row.n):.2f}" cy="{sy(value_getter(row)):.2f}" '
                f'r="3.5" fill="{color}"/>'
            )

    legend_x = margin_left + plot_width + 30
    legend_y = margin_top + 12
    if draw_true_integral:
        lines.append(
            f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 28}" '
            f'y2="{legend_y}" stroke="#000" stroke-width="2" stroke-dasharray="8 6"/>'
        )
        lines.append(
            f'<text x="{legend_x + 38}" y="{legend_y + 4}" '
            'font-family="Arial" font-size="12">True integral</text>'
        )
        legend_y += 24

    for method_index, method in enumerate(grouped):
        color = colors[method_index % len(colors)]
        y = legend_y + method_index * 24
        lines.append(
            f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 28}" y2="{y}" '
            f'stroke="{color}" stroke-width="2.2"/>'
        )
        lines.append(
            f'<text x="{legend_x + 38}" y="{y + 4}" '
            f'font-family="Arial" font-size="12">{method}</text>'
        )

    lines.append("</svg>")
    filename.write_text("\n".join(lines), encoding="utf-8")


def save_plots(results: list[Result]) -> None:
    PLOTS_DIR.mkdir(exist_ok=True)
    grouped = grouped_by_method(results)
    make_line_chart(
        grouped,
        PLOTS_DIR / "convergence.svg",
        "Convergence of Monte Carlo Integral Estimates",
        "Integral estimate",
        lambda row: row.estimate,
        draw_true_integral=True,
    )
    make_line_chart(
        grouped,
        PLOTS_DIR / "absolute_error.svg",
        "Absolute Error vs Sample Size",
        "Absolute error",
        lambda row: row.absolute_error,
    )


def main() -> None:
    results = collect_results()
    print_table(results)
    save_plots(results)
    print()
    print(f"Plots saved to: {PLOTS_DIR / 'convergence.svg'} and {PLOTS_DIR / 'absolute_error.svg'}")


if __name__ == "__main__":
    main()
