"""Траектории градиентного спуска на двух квадратичных функциях."""

import warnings

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from optimization import gradient_descent
from oracles import QuadraticOracle
from plot_trajectory_2d import plot_levels, plot_trajectory


MATRICES = {1: np.diag([8., 8.]), 16: np.diag([0.5, 8.])}
STARTS = [(3, 4), (5, 0)]
OPTIONS = {
    "Constant": {"method": "Constant", "c": 3 / 16},
    "Armijo": {"method": "Armijo", "alpha_0": 1., "c1": 1e-4},
    "Wolfe": {"method": "Wolfe", "alpha_0": 1., "c1": 1e-4, "c2": 0.9},
}


def run_experiment():
    histories = {}
    for kappa, matrix in MATRICES.items():
        oracle = QuadraticOracle(matrix, np.zeros(2))
        for start in STARTS:
            for method, options in OPTIONS.items():
                _, status, history = gradient_descent(
                    oracle, np.array(start, dtype=float), tolerance=1e-8,
                    max_iter=1000, line_search_options=options, trace=True,
                )
                if status != "success":
                    raise RuntimeError(f"κ={kappa}, x₀={start}, {method}: {status}")
                histories[kappa, start, method] = history
    return histories


def steps_table(histories):
    rows = ["| κ | Начальная точка | Constant | Armijo | Wolfe |",
            "|---:|:---|---:|---:|---:|"]
    for kappa in MATRICES:
        for start in STARTS:
            steps = [len(histories[kappa, start, method]["x"]) - 1 for method in OPTIONS]
            rows.append(f"| {kappa} | {start} | " + " | ".join(map(str, steps)) + " |")
    return "\n".join(rows)


def trajectory_figure(histories, kappa):
    oracle = QuadraticOracle(MATRICES[kappa], np.zeros(2))
    colors = ["#18638b", "#bc511b"]
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.9), sharex=True, sharey=True)
    for ax, method in zip(axes, OPTIONS):
        plt.sca(ax)
        with warnings.catch_warnings():
            # Учебная утилита передаёт устаревшее имя аргумента толщины линий.
            warnings.filterwarnings("ignore", message="The following kwargs were not used by contour.*")
            plot_levels(oracle.func, [-3.3, 5.8], [-4.8, 4.8],
                        levels=[1, 4, 16, 36, 64, 100])
        for contours in ax.collections:
            contours.set_edgecolor("#a8b1b6")
            contours.set_linewidth(0.9)
        for label in ax.texts:
            label.set_color("#5b6367")
        ax.grid(True, color="#edf0f2", linewidth=0.6)
        for start, color in zip(STARTS, colors):
            path = histories[kappa, start, method]["x"]
            plot_trajectory(oracle.func, path)
            ax.lines[-1].set(color=color, marker="o", markersize=2.8, linewidth=2.2)
            ax.scatter(*start, s=47, marker="s", color=color, zorder=5)
        ax.scatter(0, 0, s=110, marker="*", color="#191d22", zorder=6)
        ax.set(xlim=(-3.3, 5.8), ylim=(-4.8, 4.8), aspect="equal",
               xlabel="$x_1$", title=method)
    axes[0].set_ylabel("$x_2$")
    handles = [Line2D([], [], color=color, marker="s", linewidth=2.2,
                      label=f"Старт {start}") for start, color in zip(STARTS, colors)]
    handles.append(Line2D([], [], color="#191d22", marker="*", markersize=12,
                          linestyle="none", label="Минимум (0, 0)"))
    fig.suptitle(f"Траектории градиентного спуска при κ = {kappa}", fontsize=15)
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False)
    fig.subplots_adjust(left=0.055, right=0.99, bottom=0.19, top=0.86, wspace=0.12)
    return fig
