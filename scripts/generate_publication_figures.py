#!/usr/bin/env python3
"""Generate publication-quality figures for the DualPath-CEM manuscript.

The diagrams are deterministic vector artwork. Quantitative panels are read
from the formal experiment summary, and the qualitative panel only rearranges
the already exported reconstruction pixels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib as mpl

mpl.use("Agg")
mpl.rcParams.update(
    {
        "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.size": 7.8,
        "axes.titlesize": 8.6,
        "axes.labelsize": 7.8,
        "axes.linewidth": 0.8,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "legend.fontsize": 6.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    }
)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image, ImageDraw, ImageFont


INK = "#17212B"
MUTED = "#5B6773"
LINE = "#9AA6B2"
GRID = "#DDE3E8"
PANEL = "#F6F8FA"
WHITE = "#FFFFFF"
SPATIAL = "#0072B2"
SPATIAL_LIGHT = "#DCEEF8"
SEMANTIC = "#009E73"
SEMANTIC_LIGHT = "#DCF3EB"
NOISE = "#E69F00"
NOISE_LIGHT = "#FFF0C9"
FUSION = "#6F5CC2"
FUSION_LIGHT = "#ECE8FA"
ATTACK = "#D55E00"
ATTACK_LIGHT = "#FBE4DA"
BASELINE = "#6B7280"
GOOD = "#2F855A"
GOOD_LIGHT = "#E7F4EC"


def _save(figure: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output_dir / f"{stem}.pdf",
        bbox_inches="tight",
        pad_inches=0.035,
    )
    figure.savefig(
        output_dir / f"{stem}.png",
        dpi=450,
        bbox_inches="tight",
        pad_inches=0.035,
    )
    plt.close(figure)


def _panel(
    axis: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    facecolor: str = PANEL,
    edgecolor: str = GRID,
    linewidth: float = 0.8,
    radius: float = 0.10,
    zorder: int = 0,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle=f"round,pad=0.02,rounding_size={radius}",
        linewidth=linewidth,
        edgecolor=edgecolor,
        facecolor=facecolor,
        zorder=zorder,
    )
    axis.add_patch(patch)
    return patch


def _box(
    axis: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    detail: str = "",
    *,
    facecolor: str = WHITE,
    edgecolor: str = LINE,
    title_color: str = INK,
    fontsize: float = 7.2,
    detail_size: float = 5.6,
    linewidth: float = 1.0,
    zorder: int = 3,
) -> FancyBboxPatch:
    patch = _panel(
        axis,
        x,
        y,
        width,
        height,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        radius=0.08,
        zorder=zorder,
    )
    multiline_title = "\n" in title
    title_y = y + height * (0.67 if detail and multiline_title else (0.60 if detail else 0.50))
    axis.text(
        x + width / 2,
        title_y,
        title,
        ha="center",
        va="center",
        color=title_color,
        fontsize=fontsize,
        fontweight="bold",
        linespacing=1.15,
        zorder=zorder + 1,
    )
    if detail:
        axis.text(
            x + width / 2,
            y + height * (0.17 if multiline_title else 0.24),
            detail,
            ha="center",
            va="center",
            color=MUTED,
            fontsize=detail_size,
            linespacing=1.15,
            zorder=zorder + 1,
        )
    return patch


def _arrow(
    axis: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = MUTED,
    linewidth: float = 1.15,
    style: str = "-|>",
    connectionstyle: str = "arc3",
    zorder: int = 2,
) -> FancyArrowPatch:
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=10.5,
        linewidth=linewidth,
        color=color,
        connectionstyle=connectionstyle,
        shrinkA=2,
        shrinkB=2,
        zorder=zorder,
    )
    axis.add_patch(patch)
    return patch


def _pill(
    axis: plt.Axes,
    x: float,
    y: float,
    text: str,
    *,
    facecolor: str,
    color: str,
    fontsize: float = 5.2,
) -> None:
    axis.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight="bold",
        color=color,
        bbox={
            "boxstyle": "round,pad=0.25,rounding_size=0.65",
            "facecolor": facecolor,
            "edgecolor": "none",
        },
        zorder=8,
    )


def _panel_label(axis: plt.Axes, label: str, x: float = 0.0, y: float = 1.04) -> None:
    axis.text(
        x,
        y,
        label,
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.2,
        fontweight="bold",
        color=INK,
    )


def generate_architecture(output_dir: Path) -> None:
    figure, axis = plt.subplots(figsize=(9.2, 4.10))
    axis.set_xlim(0, 16)
    axis.set_ylim(0, 9)
    axis.axis("off")

    _panel(axis, 0.12, 0.28, 6.08, 8.35, facecolor="#FAFBFC", radius=0.14)
    _panel(axis, 6.45, 0.28, 9.42, 8.35, facecolor="#FAFBFC", radius=0.14)
    axis.text(0.38, 8.30, "CLIENT", fontsize=8.1, fontweight="bold", color=MUTED)
    axis.text(6.72, 8.30, "SERVER AND EVALUATOR", fontsize=8.1, fontweight="bold", color=MUTED)
    axis.axvline(6.32, ymin=0.055, ymax=0.95, color=LINE, linewidth=1.0, linestyle=(0, (4, 3)))
    axis.text(
        6.32,
        8.70,
        "collaborative-inference boundary",
        ha="center",
        va="center",
        fontsize=6.8,
        color=MUTED,
    )

    _box(axis, 0.42, 3.72, 1.12, 1.25, "Input", "$64\\times64$ RGB face", facecolor=WHITE)
    axis.text(2.00, 7.72, "SPATIAL PATH", fontsize=7.1, fontweight="bold", color=SPATIAL)
    axis.text(2.00, 3.14, "SEMANTIC PATH", fontsize=7.1, fontweight="bold", color=SEMANTIC)

    _box(
        axis,
        2.00,
        6.28,
        2.25,
        1.15,
        "Slot-CEM\nspatial encoder",
        "VGG11-BN client + bottleneck",
        facecolor=SPATIAL_LIGHT,
        edgecolor=SPATIAL,
        title_color=SPATIAL,
    )
    _pill(axis, 2.38, 6.17, "FROZEN", facecolor="#E8EDF1", color=MUTED)
    _box(
        axis,
        4.72,
        6.28,
        1.08,
        1.15,
        "Gaussian",
        "$\\sigma_s=0.31$",
        facecolor=NOISE_LIGHT,
        edgecolor=NOISE,
        title_color="#9A6700",
    )
    _pill(axis, 5.26, 7.61, "$16\\times16\\times16$", facecolor=SPATIAL_LIGHT, color=SPATIAL)

    _box(
        axis,
        2.00,
        1.72,
        2.25,
        1.15,
        "Semantic encoder",
        "MobileNetV3-Large + GAP\n+ 256-D projection",
        facecolor=SEMANTIC_LIGHT,
        edgecolor=SEMANTIC,
        title_color=SEMANTIC,
    )
    _pill(axis, 2.54, 1.61, "TRAINED S1+S2", facecolor=SEMANTIC_LIGHT, color=SEMANTIC)
    _box(
        axis,
        4.72,
        1.72,
        1.08,
        1.15,
        "Gaussian",
        "$\\sigma_t=0.10$",
        facecolor=NOISE_LIGHT,
        edgecolor=NOISE,
        title_color="#9A6700",
    )
    _pill(axis, 5.26, 3.05, "256-D", facecolor=SEMANTIC_LIGHT, color=SEMANTIC)

    _arrow(axis, (1.54, 4.34), (2.00, 6.80), color=SPATIAL, connectionstyle="arc3,rad=-0.10")
    _arrow(axis, (1.54, 4.34), (2.00, 2.28), color=SEMANTIC, connectionstyle="arc3,rad=0.10")
    _arrow(axis, (4.25, 6.86), (4.72, 6.86), color=SPATIAL)
    _arrow(axis, (4.25, 2.29), (4.72, 2.29), color=SEMANTIC)

    _box(
        axis,
        6.78,
        6.28,
        2.05,
        1.15,
        "Spatial\nserver head",
        "VGG cloud + identity\nclassifier",
        facecolor=SPATIAL_LIGHT,
        edgecolor=SPATIAL,
        title_color=SPATIAL,
    )
    _pill(axis, 7.80, 6.17, "ADAPTED S2", facecolor=SPATIAL_LIGHT, color=SPATIAL)
    _box(
        axis,
        6.78,
        1.72,
        2.05,
        1.15,
        "Semantic\nserver head",
        "linear identity classifier",
        facecolor=SEMANTIC_LIGHT,
        edgecolor=SEMANTIC,
        title_color=SEMANTIC,
    )
    _pill(axis, 7.80, 1.61, "TRAINED S1+S2", facecolor=SEMANTIC_LIGHT, color=SEMANTIC)

    _box(
        axis,
        9.76,
        3.72,
        2.15,
        1.25,
        "Calibrated\nlogit fusion",
        "learned weight +\ntwo temperatures",
        facecolor=FUSION_LIGHT,
        edgecolor=FUSION,
        title_color=FUSION,
    )
    _box(axis, 12.72, 3.72, 1.42, 1.25, "Identity", "526 classes", facecolor=WHITE)
    _arrow(axis, (5.80, 6.86), (6.78, 6.86), color=SPATIAL)
    _arrow(axis, (5.80, 2.29), (6.78, 2.29), color=SEMANTIC)
    _arrow(axis, (8.83, 6.86), (9.88, 4.88), color=SPATIAL, connectionstyle="arc3,rad=-0.05")
    _arrow(axis, (8.83, 2.29), (9.88, 3.82), color=SEMANTIC, connectionstyle="arc3,rad=0.05")
    _arrow(axis, (11.91, 4.34), (12.72, 4.34), color=FUSION)

    _panel(axis, 9.30, 0.55, 6.12, 1.72, facecolor=ATTACK_LIGHT, edgecolor="#E9B6A1", radius=0.10)
    axis.text(9.58, 1.93, "JOINT INVERSION EVALUATOR", fontsize=7.0, fontweight="bold", color=ATTACK)
    _box(
        axis,
        9.58,
        0.83,
        2.62,
        0.78,
        "Residual / GAN /\nadaptive attacker",
        "observes both tensors",
        facecolor=WHITE,
        edgecolor=ATTACK,
        title_color=ATTACK,
        fontsize=7.3,
        detail_size=6.4,
    )
    _box(
        axis,
        13.00,
        0.83,
        1.98,
        0.78,
        "Reconstructed\nimage",
        "after target freeze",
        facecolor=WHITE,
        edgecolor=ATTACK,
        title_color=ATTACK,
        fontsize=7.3,
        detail_size=6.2,
    )
    _arrow(axis, (12.20, 1.22), (13.00, 1.22), color=ATTACK)
    _arrow(axis, (6.18, 6.53), (9.66, 1.60), color=ATTACK, linewidth=1.0, connectionstyle="arc3,rad=0.13")
    _arrow(axis, (6.18, 2.05), (9.58, 1.18), color=ATTACK, linewidth=1.0, connectionstyle="arc3,rad=-0.08")

    axis.text(
        6.32,
        0.05,
        "Transmitted payload: 4,096 spatial + 256 semantic FP32 values = 17,408 bytes per image",
        ha="center",
        va="bottom",
        fontsize=7.0,
        color=MUTED,
    )
    axis.text(0.12, 8.87, "End-to-end DualPath-CEM architecture", fontsize=9.4, fontweight="bold", color=INK)
    _save(figure, output_dir, "architecture")


def generate_slot_cem_mechanism(output_dir: Path) -> None:
    figure, axis = plt.subplots(figsize=(9.2, 3.10))
    axis.set_xlim(0, 16)
    axis.set_ylim(0, 5.4)
    axis.axis("off")

    steps = [
        (0.18, 0.48, 3.05, 4.48, "1", "Same-identity\nevidence"),
        (3.48, 0.48, 3.65, 4.48, "2", "Competitive\nassignment"),
        (7.38, 0.48, 3.70, 4.48, "3", "Geometric privacy\nstatistic"),
        (11.33, 0.48, 4.48, 4.48, "4", "Effect on the\nspatial encoder"),
    ]
    for x, y, width, height, number, title in steps:
        _panel(axis, x, y, width, height, facecolor="#FAFBFC", radius=0.10)
        axis.text(x + 0.28, y + height - 0.40, number, fontsize=8.0, fontweight="bold", color=WHITE,
                  bbox={"boxstyle": "circle,pad=0.28", "facecolor": SPATIAL, "edgecolor": "none"})
        axis.text(x + 0.72, y + height - 0.40, title, fontsize=6.2, fontweight="bold", color=INK, va="center", linespacing=1.0)

    rng = np.random.default_rng(7)
    batch = np.array([[0.80, 3.20], [1.22, 3.45], [1.68, 3.08], [2.12, 3.36]])
    axis.scatter(batch[:, 0], batch[:, 1], s=45, color=SPATIAL, edgecolors=WHITE, linewidths=0.8, zorder=5)
    axis.text(1.53, 3.83, "current mini-batch", ha="center", fontsize=7.0, color=MUTED)
    memory_x = np.linspace(0.72, 2.64, 8)
    memory_y = 1.55 + rng.normal(0, 0.08, len(memory_x))
    axis.scatter(memory_x, memory_y, s=24, color="#7DB7D8", edgecolors=WHITE, linewidths=0.6, zorder=5)
    axis.text(1.66, 2.03, "per-class memory bank", ha="center", fontsize=7.0, color=INK, fontweight="bold")
    axis.text(1.66, 1.08, "up to 64 stored features", ha="center", fontsize=6.8, color=MUTED)
    _arrow(axis, (1.62, 2.85), (1.62, 2.20), color=SPATIAL)
    axis.text(1.62, 2.56, "+", ha="center", va="center", fontsize=11, fontweight="bold", color=SPATIAL)

    _box(
        axis,
        3.78,
        3.42,
        1.35,
        0.74,
        "Fixed projection",
        "64-D",
        facecolor=SPATIAL_LIGHT,
        edgecolor=SPATIAL,
        title_color=SPATIAL,
        fontsize=7.1,
        detail_size=6.1,
    )
    slot_colours = [SPATIAL, SEMANTIC, NOISE, FUSION, ATTACK, "#56B4E9", "#CC79A7", "#7A8B99"]
    slot_positions = []
    for row in range(2):
        for col in range(4):
            sx = 4.05 + col * 0.72
            sy = 1.72 + row * 0.72
            slot_positions.append((sx, sy))
            circle = Circle((sx, sy), 0.19, facecolor=slot_colours[row * 4 + col], edgecolor=WHITE, linewidth=1.0, zorder=5)
            axis.add_patch(circle)
    _arrow(axis, (4.45, 3.40), (4.45, 2.72), color=SPATIAL)
    axis.text(5.20, 2.88, "3 iterative updates", fontsize=6.8, color=MUTED)
    axis.text(5.50, 1.04, "8 class-conditioned slots", ha="center", fontsize=7.1, color=INK, fontweight="bold")
    _box(
        axis,
        5.58,
        3.42,
        1.22,
        0.74,
        "Softmax",
        "assignment weights",
        facecolor=WHITE,
        edgecolor=SPATIAL,
        title_color=SPATIAL,
        fontsize=7.1,
        detail_size=6.0,
    )
    _arrow(axis, (5.13, 3.79), (5.58, 3.79), color=SPATIAL)

    centers = np.array([[8.05, 1.58], [8.95, 2.00], [9.82, 1.50], [8.56, 3.20], [9.55, 3.35]])
    colors = [SPATIAL, SEMANTIC, NOISE, FUSION, ATTACK]
    for idx, center in enumerate(centers):
        points = center + rng.normal(0, 0.19, size=(5, 2))
        axis.scatter(points[:, 0], points[:, 1], s=15, color=colors[idx], alpha=0.48, edgecolors="none", zorder=4)
        axis.scatter([center[0]], [center[1]], s=75, marker="X", color=colors[idx], edgecolors=WHITE, linewidths=0.8, zorder=6)
    query = np.array([9.15, 2.62])
    axis.scatter([query[0]], [query[1]], s=42, color=INK, edgecolors=WHITE, linewidths=0.8, zorder=7)
    nearest = centers[3]
    axis.plot([query[0], nearest[0]], [query[1], nearest[1]], color=INK, linewidth=1.15, linestyle=(0, (3, 2)), zorder=5)
    axis.text(9.18, 2.90, "$z_i$", fontsize=7.5, color=INK)
    axis.text(8.40, 2.48, "nearest centroid", fontsize=6.5, color=MUTED, rotation=38)
    axis.text(9.23, 1.00, "distance measured in the full\nsmashed-feature geometry", ha="center", fontsize=7.0, color=INK, linespacing=1.25)

    _box(
        axis,
        11.72,
        3.18,
        3.72,
        0.92,
        "$V_y = |\\mathcal{B}_y|^{-1} \\sum_i \\min_j \\|z_i-c_{y,j}\\|_2^2$",
        "no learned variance head",
        facecolor=SPATIAL_LIGHT,
        edgecolor=SPATIAL,
        title_color=INK,
        fontsize=8.1,
        detail_size=6.6,
    )
    _arrow(axis, (13.58, 3.10), (13.58, 2.60), color=SPATIAL)
    _box(
        axis,
        11.72,
        1.40,
        3.72,
        1.08,
        "CEM privacy objective",
        "pushes the encoder toward larger\nclass-conditional ambiguity",
        facecolor=GOOD_LIGHT,
        edgecolor=GOOD,
        title_color=GOOD,
        fontsize=7.7,
        detail_size=6.6,
    )
    axis.text(13.58, 0.92, "Checkpoint is then frozen for DualPath-CEM", ha="center", fontsize=6.9, color=MUTED)

    for x in (3.32, 7.22, 11.17):
        _arrow(axis, (x, 2.72), (x + 0.16, 2.72), color=LINE, linewidth=1.0)
    axis.text(0.12, 5.18, "Slot-CEM foundation used by the spatial path", fontsize=9.4, fontweight="bold", color=INK)
    _save(figure, output_dir, "slot_cem_mechanism")


def _status_row(
    axis: plt.Axes,
    x: float,
    y: float,
    label: str,
    status: str,
    color: str,
    *,
    width: float = 3.30,
) -> None:
    axis.text(x, y, label, fontsize=6.9, color=INK, va="center")
    _pill(
        axis,
        x + width,
        y,
        status,
        facecolor="#E8EDF1" if status == "FROZEN" else (SEMANTIC_LIGHT if color == SEMANTIC else SPATIAL_LIGHT),
        color=MUTED if status == "FROZEN" else color,
        fontsize=5.7,
    )


def generate_training_protocol(output_dir: Path) -> None:
    figure, axis = plt.subplots(figsize=(9.2, 3.40))
    axis.set_xlim(0, 16)
    axis.set_ylim(0, 5.75)
    axis.axis("off")

    panels = [
        (0.18, 0.62, 4.75, 4.50, "STAGE 1", "semantic warm-up", SEMANTIC),
        (5.22, 0.62, 4.75, 4.50, "STAGE 2", "noise adaptation", SPATIAL),
        (10.26, 0.62, 5.56, 4.50, "FORMAL EVAL.", "freeze, then attack", ATTACK),
    ]
    for x, y, width, height, stage, title, color in panels:
        _panel(axis, x, y, width, height, facecolor="#FAFBFC", radius=0.11)
        axis.add_patch(Rectangle((x, y + height - 0.62), width, 0.62, facecolor=color, edgecolor="none", zorder=2))
        axis.text(x + 0.24, y + height - 0.31, stage, fontsize=7.2, fontweight="bold", color=WHITE, va="center")
        axis.text(x + width - 0.24, y + height - 0.31, title, fontsize=5.7, color=WHITE, va="center", ha="right")

    _status_row(axis, 0.48, 3.96, "Complete spatial branch", "FROZEN", SPATIAL)
    _status_row(axis, 0.48, 3.44, "Semantic encoder + head", "TRAINED", SEMANTIC)
    _status_row(axis, 0.48, 2.92, "Fusion calibration", "TRAINED", SEMANTIC)
    axis.text(0.48, 2.28, "$\\mathcal{L}_1=\\mathrm{CE}(\\ell,Y)+0.35\\,\\mathrm{CE}(\\ell_t,Y)$", fontsize=7.4, color=INK)
    axis.text(0.48, 1.75, "$\\sigma_s=0.025,\\;\\sigma_t=0.05$", fontsize=7.2, color=MUTED)
    _pill(axis, 1.13, 1.18, "80 epochs", facecolor=SEMANTIC_LIGHT, color=SEMANTIC, fontsize=6.2)
    axis.text(2.10, 1.18, "AdamW + cosine decay", fontsize=6.7, color=MUTED, va="center")

    _status_row(axis, 5.52, 3.96, "Spatial client", "FROZEN", SPATIAL)
    _status_row(axis, 5.52, 3.44, "Spatial server + head", "TRAINED", SPATIAL)
    _status_row(axis, 5.52, 2.92, "Semantic path + fusion", "TRAINED", SEMANTIC)
    axis.text(5.52, 2.28, "$\\mathcal{L}_2=\\mathrm{CE}(\\ell,Y)+0.50\\,\\mathrm{CE}(\\ell_t,Y)$", fontsize=7.4, color=INK)
    axis.text(5.52, 1.75, "$\\sigma_s=0.22,\\;\\sigma_t=0.10$", fontsize=7.2, color=MUTED)
    _pill(axis, 6.14, 1.18, "40 epochs", facecolor=SPATIAL_LIGHT, color=SPATIAL, fontsize=6.2)
    axis.text(7.10, 1.18, "clip 5 + label smoothing 0.05", fontsize=6.7, color=MUTED, va="center")

    _box(
        axis,
        10.58,
        3.63,
        2.12,
        0.86,
        "Five target models",
        "seeds 126-130",
        facecolor=WHITE,
        edgecolor=FUSION,
        title_color=FUSION,
        fontsize=7.2,
        detail_size=6.3,
    )
    _box(
        axis,
        13.35,
        3.63,
        2.08,
        0.86,
        "Fixed deployment",
        "$\\sigma_s=0.31,\\;\\sigma_t=0.10$",
        facecolor=NOISE_LIGHT,
        edgecolor=NOISE,
        title_color="#9A6700",
        fontsize=7.2,
        detail_size=6.3,
    )
    _arrow(axis, (12.70, 4.06), (13.35, 4.06), color=FUSION)
    _box(
        axis,
        10.58,
        2.16,
        2.12,
        0.86,
        "Target checkpoint",
        "frozen before attacks",
        facecolor="#E8EDF1",
        edgecolor=BASELINE,
        title_color=BASELINE,
        fontsize=7.2,
        detail_size=6.3,
    )
    _box(
        axis,
        13.35,
        2.16,
        2.08,
        0.86,
        "Joint attackers",
        "residual / GAN /\nadaptive",
        facecolor=ATTACK_LIGHT,
        edgecolor=ATTACK,
        title_color=ATTACK,
        fontsize=7.2,
        detail_size=5.2,
    )
    _arrow(axis, (11.64, 3.63), (11.64, 3.02), color=FUSION)
    _arrow(axis, (12.70, 2.59), (13.35, 2.59), color=ATTACK)
    axis.text(10.58, 1.54, "3 attacker seeds x 2 knowledge settings", fontsize=6.8, color=INK)
    axis.text(10.58, 1.10, "No reconstruction loss enters target training", fontsize=6.8, color=ATTACK, fontweight="bold")

    _arrow(axis, (4.93, 2.87), (5.22, 2.87), color=LINE, linewidth=1.3)
    _arrow(axis, (9.97, 2.87), (10.26, 2.87), color=LINE, linewidth=1.3)
    axis.text(0.12, 5.52, "Optimisation and attack-evaluation protocol", fontsize=9.4, fontweight="bold", color=INK)
    _save(figure, output_dir, "training_protocol")


def _orth_arrow(
    axis: plt.Axes,
    points: list[tuple[float, float]],
    *,
    color: str = MUTED,
    linewidth: float = 1.05,
    dashed: bool = False,
    zorder: int = 3,
) -> None:
    if len(points) < 2:
        return
    style = (0, (3, 2)) if dashed else "-"
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    axis.plot(
        xs,
        ys,
        color=color,
        linewidth=linewidth,
        linestyle=style,
        solid_capstyle="round",
        solid_joinstyle="round",
        zorder=zorder,
    )
    _arrow(
        axis,
        points[-2],
        points[-1],
        color=color,
        linewidth=linewidth,
        zorder=zorder + 1,
    )


def _editorial_card(
    axis: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    detail: str,
    *,
    color: str,
    fill: str,
    badge: str | None = None,
    title_size: float = 6.8,
    detail_size: float = 5.3,
) -> None:
    shadow = FancyBboxPatch(
        (x + 0.28, y - 0.28),
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.55",
        linewidth=0,
        facecolor="#D8DEE4",
        alpha=0.42,
        zorder=1,
    )
    axis.add_patch(shadow)
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.55",
        linewidth=0.95,
        edgecolor=color,
        facecolor=fill,
        zorder=2,
    )
    axis.add_patch(patch)
    axis.add_patch(
        Rectangle(
            (x, y),
            1.15,
            height,
            facecolor=color,
            edgecolor="none",
            zorder=3,
        )
    )
    axis.text(
        x + 1.75,
        y + height - (2.02 if badge else 1.42),
        title,
        ha="left",
        va="center",
        fontsize=title_size,
        fontweight="bold",
        color=INK,
        linespacing=1.0,
        zorder=4,
    )
    axis.text(
        x + 1.75,
        y + 1.48,
        detail,
        ha="left",
        va="center",
        fontsize=detail_size,
        color=MUTED,
        linespacing=1.05,
        zorder=4,
    )
    if badge:
        axis.text(
            x + width - 0.65,
            y + height - 0.68,
            badge,
            ha="right",
            va="center",
            fontsize=4.4,
            fontweight="bold",
            color=color,
            bbox={
                "boxstyle": "round,pad=0.22,rounding_size=0.65",
                "facecolor": WHITE,
                "edgecolor": "none",
                "alpha": 0.96,
            },
            zorder=5,
        )


def _draw_tensor_icon(
    axis: plt.Axes,
    x: float,
    y: float,
    *,
    color: str,
    label: str,
) -> None:
    for offset, alpha in ((1.15, 0.18), (0.58, 0.32), (0.0, 0.62)):
        axis.add_patch(
            Rectangle(
                (x + offset, y + offset * 0.45),
                3.7,
                4.4,
                linewidth=0.8,
                edgecolor=color,
                facecolor=mpl.colors.to_rgba(color, alpha),
                zorder=4,
            )
        )
    axis.text(
        x + 2.45,
        y - 0.78,
        label,
        ha="center",
        va="top",
        fontsize=4.9,
        color=MUTED,
        linespacing=1.0,
    )


def _draw_token_icon(axis: plt.Axes, x: float, y: float, *, color: str) -> None:
    for row in range(4):
        for column in range(4):
            alpha = 0.28 + 0.08 * ((row + column) % 3)
            axis.add_patch(
                Rectangle(
                    (x + column * 0.95, y + row * 0.95),
                    0.68,
                    0.68,
                    linewidth=0.45,
                    edgecolor=color,
                    facecolor=mpl.colors.to_rgba(color, alpha),
                    zorder=4,
                )
            )
    axis.text(x + 1.78, y - 0.72, "256-D token", ha="center", va="top", fontsize=4.9, color=MUTED)


def _draw_noise_channel(
    axis: plt.Axes,
    x: float,
    y: float,
    sigma: str,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        5.2,
        5.3,
        boxstyle="round,pad=0.02,rounding_size=0.55",
        linewidth=0.9,
        edgecolor=NOISE,
        facecolor=NOISE_LIGHT,
        zorder=3,
    )
    axis.add_patch(patch)
    axis.text(x + 2.6, y + 3.45, "NOISE", ha="center", va="center", fontsize=5.2, fontweight="bold", color="#8B5E00")
    axis.text(x + 2.6, y + 1.55, sigma, ha="center", va="center", fontsize=5.7, color=INK)


def _draw_face_input(axis: plt.Axes, x: float, y: float) -> None:
    patch = FancyBboxPatch(
        (x, y),
        7.2,
        8.2,
        boxstyle="round,pad=0.02,rounding_size=0.7",
        linewidth=0.9,
        edgecolor=LINE,
        facecolor=WHITE,
        zorder=3,
    )
    axis.add_patch(patch)
    axis.add_patch(Circle((x + 3.6, y + 5.15), 1.35, facecolor="#E5EAF0", edgecolor=BASELINE, linewidth=0.65, zorder=4))
    axis.add_patch(
        FancyBboxPatch(
            (x + 1.55, y + 1.35),
            4.1,
            1.8,
            boxstyle="round,pad=0.02,rounding_size=0.9",
            linewidth=0.65,
            edgecolor=BASELINE,
            facecolor="#E5EAF0",
            zorder=4,
        )
    )
    axis.text(x + 3.6, y - 0.72, "$64\\times64$ RGB", ha="center", va="top", fontsize=5.0, color=MUTED)


def _architecture_tiles(source: Path | None) -> tuple[Image.Image, Image.Image] | None:
    if source is None or not source.is_file():
        return None
    with Image.open(source) as image:
        grid = image.convert("RGB")
    scale = max(1, round(grid.height / 212))
    padding = 5 * scale
    tile = 64 * scale
    label_width = 120 * scale
    if label_width + tile > grid.width or padding + 2 * (tile + padding) > grid.height:
        return None
    original = grid.crop((label_width, padding, label_width + tile, padding + tile))
    reconstruction_y = padding + tile + padding
    reconstruction = grid.crop(
        (label_width, reconstruction_y, label_width + tile, reconstruction_y + tile)
    )
    return original, reconstruction


def generate_architecture_editorial(
    output_dir: Path,
    qualitative_source: Path | None = None,
) -> None:
    figure, axis = plt.subplots(figsize=(7.15, 4.55))
    figure.subplots_adjust(left=0.008, right=0.992, bottom=0.012, top=0.988)
    axis.set_xlim(0, 100)
    axis.set_ylim(0, 64)
    axis.axis("off")

    def section_title(x: float, y: float, label: str, title: str) -> None:
        axis.text(x, y, label, ha="left", va="center", fontsize=5.8, fontweight="bold", color=WHITE,
                  bbox={"boxstyle": "round,pad=0.27,rounding_size=0.7", "facecolor": INK, "edgecolor": "none"}, zorder=8)
        axis.text(x + 3.8, y, title, ha="left", va="center", fontsize=6.3, fontweight="semibold", color=INK, zorder=8)

    def compact_card(
        x: float,
        y: float,
        width: float,
        height: float,
        title: str,
        detail: str,
        color: str,
        fill: str,
        badge: str | None = None,
    ) -> None:
        patch = FancyBboxPatch(
            (x, y), width, height,
            boxstyle="round,pad=0.02,rounding_size=0.55",
            linewidth=0.9, edgecolor=color, facecolor=fill, zorder=4,
        )
        axis.add_patch(patch)
        axis.add_patch(Rectangle((x, y), 0.72, height, facecolor=color, edgecolor="none", zorder=5))
        axis.text(x + 1.35, y + height * 0.64, title, ha="left", va="center", fontsize=5.1,
                  fontweight="semibold", color=INK, linespacing=1.0, zorder=6)
        axis.text(x + 1.35, y + height * 0.27, detail, ha="left", va="center", fontsize=4.15,
                  color=MUTED, linespacing=1.0, zorder=6)
        if badge:
            axis.text(x + width - 0.45, y + height - 0.48, badge, ha="right", va="center",
                      fontsize=3.9, fontweight="bold", color=color,
                      bbox={"boxstyle": "round,pad=0.20,rounding_size=0.55", "facecolor": WHITE,
                            "edgecolor": "none", "alpha": 0.96}, zorder=7)

    def mini_node(
        x: float,
        y: float,
        width: float,
        title: str,
        detail: str,
        color: str,
        fill: str,
    ) -> None:
        _panel(axis, x, y, width, 6.8, facecolor=fill, edgecolor=color, linewidth=0.82, radius=0.48, zorder=4)
        axis.text(x + width / 2, y + 4.45, title, ha="center", va="center", fontsize=4.5,
                  fontweight="semibold", color=INK, linespacing=1.0, zorder=6)
        axis.text(x + width / 2, y + 1.85, detail, ha="center", va="center", fontsize=3.9,
                  color=MUTED, linespacing=1.0, zorder=6)

    # Panel (a): complete protected inference path.
    _panel(axis, 0.7, 25.8, 98.6, 37.2, facecolor=WHITE, edgecolor="#CBD5DF", radius=0.72, zorder=0)
    axis.add_patch(Rectangle((1.0, 27.1), 50.0, 31.8, facecolor="#F7FAFC", edgecolor="none", zorder=0))
    axis.add_patch(Rectangle((52.4, 27.1), 45.6, 31.8, facecolor="#FBF9F7", edgecolor="none", zorder=0))
    section_title(2.0, 60.7, "a", "Protected dual-path collaborative inference")
    axis.text(3.0, 57.8, "TRUSTED CLIENT", fontsize=4.4, fontweight="bold", color=MUTED)
    axis.text(54.1, 57.8, "UNTRUSTED CLOUD", fontsize=4.4, fontweight="bold", color=MUTED)
    axis.plot([51.7, 51.7], [27.2, 59.3], color="#8795A3", linewidth=0.8, linestyle=(0, (3, 2)), zorder=2)
    axis.text(51.7, 61.2, "TRANSMISSION", ha="center", va="center", fontsize=3.9,
              fontweight="bold", color=MUTED)

    tiles = _architecture_tiles(qualitative_source)
    face_x, face_y, face_w, face_h = 3.2, 39.5, 8.2, 8.2
    face_border = FancyBboxPatch((face_x, face_y), face_w, face_h,
                                 boxstyle="round,pad=0.02,rounding_size=0.65",
                                 facecolor=WHITE, edgecolor=BASELINE, linewidth=0.85, zorder=4)
    axis.add_patch(face_border)
    if tiles is not None:
        artist = axis.imshow(np.asarray(tiles[0]), extent=(face_x, face_x + face_w, face_y, face_y + face_h),
                             zorder=5, interpolation="lanczos", aspect="auto")
        artist.set_clip_path(face_border)
    else:
        axis.add_patch(Circle((face_x + face_w / 2, face_y + 5.3), 1.45,
                              facecolor="#E4EAF0", edgecolor=BASELINE, linewidth=0.6, zorder=5))
    axis.text(face_x + face_w / 2, face_y - 1.0, "private face  $x$\n$64\\times64$ RGB",
              ha="center", va="top", fontsize=4.1, color=MUTED, linespacing=1.0)

    axis.text(15.0, 53.9, "SPATIAL / PRIVACY PATH", fontsize=4.1, fontweight="bold", color=SPATIAL)
    axis.text(15.0, 38.2, "SEMANTIC / UTILITY PATH", fontsize=4.1, fontweight="bold", color=SEMANTIC)
    compact_card(15.0, 45.1, 15.0, 7.3, "Slot-CEM spatial encoder", "VGG11-BN + C16 bottleneck", SPATIAL, SPATIAL_LIGHT, "FROZEN")
    compact_card(15.0, 29.4, 15.0, 7.3, "Semantic encoder", "MobileNetV3-L + 256-D proj.", SEMANTIC, SEMANTIC_LIGHT, "S1 + S2")

    _draw_tensor_icon(axis, 32.1, 46.3, color=SPATIAL, label="$z_s:16\\times16\\times16$")
    _draw_token_icon(axis, 32.5, 31.0, color=SEMANTIC)
    _draw_noise_channel(axis, 40.6, 45.7, "$\\sigma_s=0.31$")
    _draw_noise_channel(axis, 40.6, 30.0, "$\\sigma_t=0.10$")

    compact_card(54.0, 45.1, 14.4, 7.3, "Spatial server", "VGG cloud + identity head", SPATIAL, SPATIAL_LIGHT, "ADAPTED S2")
    compact_card(54.0, 29.4, 14.4, 7.3, "Token classifier", "linear identity head", SEMANTIC, SEMANTIC_LIGHT, "S1 + S2")
    compact_card(75.0, 37.2, 12.6, 8.1, "Calibrated fusion", "$p=\\alpha p_s+(1-\\alpha)p_t$", FUSION, FUSION_LIGHT, "S1 + S2")
    compact_card(90.2, 37.2, 6.5, 8.1, "Identity", "526 classes", BASELINE, WHITE)

    split_x, split_y = 12.8, 43.6
    axis.add_patch(Circle((split_x, split_y), 0.42, facecolor=INK, edgecolor=WHITE, linewidth=0.45, zorder=8))
    _orth_arrow(axis, [(11.4, 43.6), (split_x, split_y), (split_x, 48.7), (15.0, 48.7)], color=SPATIAL, linewidth=1.0)
    _orth_arrow(axis, [(split_x, split_y), (split_x, 33.0), (15.0, 33.0)], color=SEMANTIC, linewidth=1.0)
    _orth_arrow(axis, [(30.0, 48.7), (32.1, 48.7)], color=SPATIAL)
    _orth_arrow(axis, [(36.9, 48.7), (40.6, 48.7)], color=SPATIAL)
    _orth_arrow(axis, [(45.8, 48.7), (51.7, 48.7), (54.0, 48.7)], color=SPATIAL, linewidth=1.0)
    _orth_arrow(axis, [(30.0, 33.0), (32.5, 33.0)], color=SEMANTIC)
    _orth_arrow(axis, [(36.3, 33.0), (40.6, 33.0)], color=SEMANTIC)
    _orth_arrow(axis, [(45.8, 33.0), (51.7, 33.0), (54.0, 33.0)], color=SEMANTIC, linewidth=1.0)
    _orth_arrow(axis, [(68.4, 48.7), (71.5, 48.7), (71.5, 42.8), (75.0, 42.8)], color=SPATIAL)
    _orth_arrow(axis, [(68.4, 33.0), (71.5, 33.0), (71.5, 39.4), (75.0, 39.4)], color=SEMANTIC)
    _orth_arrow(axis, [(87.6, 41.2), (90.2, 41.2)], color=FUSION)
    axis.text(51.7, 27.9, "4,096 + 256 FP32 values  |  17,408 bytes / image",
              ha="center", va="bottom", fontsize=4.0, color=MUTED)

    # Panel (b): the privacy mechanism is shown at the same level as the main path.
    _panel(axis, 0.7, 0.8, 63.0, 23.5, facecolor="#FBFCFD", edgecolor="#CBD5DF", radius=0.72, zorder=0)
    section_title(2.0, 22.0, "b", "Slot-CEM privacy regularisation")
    mini_node(2.3, 11.0, 10.2, "Class pool", "$\\mathcal{B}_y + \\mathcal{M}_y$\n$K\\leq64$ features", SPATIAL, SPATIAL_LIGHT)
    mini_node(14.8, 11.0, 10.0, "Fixed assignment", "$4096\\rightarrow64$\nno learned variance", SPATIAL, WHITE)
    mini_node(27.1, 11.0, 9.3, "Competitive slots", "$S=8,\\;T=3$\nsoft class structure", FUSION, FUSION_LIGHT)
    mini_node(38.7, 11.0, 10.8, "Full-space geometry", "$\\min_j\\|z_i-c_{y,j}\\|_2^2$\ndetached centroids", NOISE, NOISE_LIGHT)
    mini_node(51.8, 11.0, 9.5, "CEM objective", "geometric variance\nmaximise $H(X|Z)$", GOOD, GOOD_LIGHT)
    for start, end, color in [
        ((12.5, 14.4), (14.8, 14.4), SPATIAL),
        ((24.8, 14.4), (27.1, 14.4), SPATIAL),
        ((36.4, 14.4), (38.7, 14.4), FUSION),
        ((49.5, 14.4), (51.8, 14.4), NOISE),
    ]:
        _orth_arrow(axis, [start, end], color=color, linewidth=0.9)
    # Compact visual cues inside the mechanism nodes.
    for index in range(5):
        axis.add_patch(Circle((4.2 + index * 1.35, 11.65 + (index % 2) * 0.34), 0.22,
                              facecolor=SPATIAL, edgecolor=WHITE, linewidth=0.3, zorder=7))
    slot_colors = [SPATIAL, SEMANTIC, NOISE, FUSION]
    for index, color in enumerate(slot_colors):
        axis.add_patch(Circle((29.2 + index * 1.55, 11.65), 0.33, facecolor=color,
                              edgecolor=WHITE, linewidth=0.4, zorder=7))
    axis.text(31.75, 6.4,
              "$L_{\\mathrm{CEM}}=\\operatorname{mean}_{y,j}\\,\\mathrm{ReLU}[\\log(V_y+\\gamma)-\\log\\tau]$",
              ha="center", va="center", fontsize=4.65, color=INK,
              bbox={"boxstyle": "round,pad=0.35,rounding_size=0.55", "facecolor": WHITE,
                    "edgecolor": "#D2D9E0", "linewidth": 0.7}, zorder=6)
    _orth_arrow(axis, [(56.6, 11.0), (56.6, 9.3), (48.0, 9.3), (48.0, 7.4)], color=GOOD, linewidth=0.95)
    axis.text(31.75, 3.6, "privacy gradient updates the spatial encoder only",
              ha="center", fontsize=4.0, fontweight="semibold", color=GOOD)

    # Panel (c): post-hoc threat model and visible privacy outcome.
    _panel(axis, 65.0, 0.8, 34.3, 23.5, facecolor="#FFFDFC", edgecolor="#E5C5B7", radius=0.72, zorder=0)
    section_title(66.3, 22.0, "c", "Frozen-target inversion evaluation")
    axis.text(67.2, 18.5, "observed pair", fontsize=4.0, color=MUTED)
    _panel(axis, 67.0, 11.0, 7.6, 6.6, facecolor=WHITE, edgecolor=ATTACK, linewidth=0.8, radius=0.45, zorder=4)
    axis.text(70.8, 15.2, "$\\tilde z_s \\oplus \\tilde z_t$", ha="center", va="center", fontsize=5.2,
              fontweight="semibold", color=INK, zorder=6)
    axis.text(70.8, 12.6, "target frozen", ha="center", va="center", fontsize=3.9, color=ATTACK, zorder=6)
    _orth_arrow(axis, [(74.6, 14.3), (77.0, 14.3)], color=ATTACK)
    _panel(axis, 77.0, 11.0, 11.0, 6.6, facecolor=ATTACK_LIGHT, edgecolor=ATTACK, linewidth=0.85, radius=0.45, zorder=4)
    axis.text(82.5, 15.25, "Joint attacker suite", ha="center", va="center", fontsize=4.7,
              fontweight="semibold", color=ATTACK, zorder=6)
    axis.text(82.5, 12.7, "residual  |  GAN  |  adaptive", ha="center", va="center",
              fontsize=3.9, color=MUTED, zorder=6)
    _orth_arrow(axis, [(88.0, 14.3), (90.1, 14.3)], color=ATTACK)

    recon_x, recon_y, recon_w, recon_h = 90.1, 10.9, 6.8, 6.8
    recon_border = FancyBboxPatch((recon_x, recon_y), recon_w, recon_h,
                                  boxstyle="round,pad=0.02,rounding_size=0.5",
                                  facecolor=WHITE, edgecolor=ATTACK, linewidth=0.85, zorder=4)
    axis.add_patch(recon_border)
    if tiles is not None:
        artist = axis.imshow(np.asarray(tiles[1]), extent=(recon_x, recon_x + recon_w, recon_y, recon_y + recon_h),
                             zorder=5, interpolation="lanczos", aspect="auto")
        artist.set_clip_path(recon_border)
    axis.text(82.0, 6.7, "MSE $\\uparrow$   LPIPS $\\uparrow$   SSIM $\\downarrow$   ID cosine $\\downarrow$",
              ha="center", va="center", fontsize=4.0, fontweight="semibold", color=INK)
    axis.text(82.0, 3.8, "reconstruction losses never update the target",
              ha="center", va="center", fontsize=4.0, color=ATTACK)
    axis.plot([50.9, 50.9, 51.4], [48.7, 25.0, 25.0], color=ATTACK,
              linewidth=0.8, linestyle=(0, (3, 2)), zorder=2)
    axis.plot([51.5, 51.5, 51.4], [33.0, 25.0, 25.0], color=ATTACK,
              linewidth=0.8, linestyle=(0, (3, 2)), zorder=2)
    axis.add_patch(Circle((51.4, 25.0), 0.30, facecolor=ATTACK, edgecolor=WHITE, linewidth=0.4, zorder=6))
    axis.plot([51.4, 64.2, 64.2, 70.8], [25.0, 25.0, 18.1, 18.1], color=ATTACK,
              linewidth=0.8, linestyle=(0, (3, 2)), zorder=2)
    _arrow(axis, (64.2, 18.1), (70.8, 18.1), color=ATTACK, linewidth=0.8, zorder=3)

    # A small legend makes the visual grammar explicit without competing with the method.
    legend_y = 1.65
    axis.plot([4.0, 7.0], [legend_y, legend_y], color=INK, linewidth=1.0)
    _arrow(axis, (6.0, legend_y), (7.0, legend_y), color=INK, linewidth=1.0)
    axis.text(7.7, legend_y, "forward inference", va="center", fontsize=3.8, color=MUTED)
    axis.plot([19.0, 22.0], [legend_y, legend_y], color=GOOD, linewidth=0.9, linestyle=(0, (3, 2)))
    axis.text(22.7, legend_y, "privacy gradient", va="center", fontsize=3.8, color=MUTED)
    axis.plot([34.0, 37.0], [legend_y, legend_y], color=ATTACK, linewidth=0.9, linestyle=(0, (3, 2)))
    axis.text(37.7, legend_y, "attacker observation", va="center", fontsize=3.8, color=MUTED)

    axis.set_aspect("auto")
    _save(figure, output_dir, "architecture")


def _step_heading(axis: plt.Axes, x: float, y: float, number: int, title: str, color: str) -> None:
    axis.add_patch(Circle((x, y), 1.45, facecolor=color, edgecolor=WHITE, linewidth=0.8, zorder=6))
    axis.text(x, y, str(number), ha="center", va="center", fontsize=6.0, fontweight="bold", color=WHITE, zorder=7)
    axis.text(x + 2.2, y, title, ha="left", va="center", fontsize=5.8, fontweight="bold", color=INK, zorder=7)


def generate_slot_cem_editorial(output_dir: Path) -> None:
    figure, axis = plt.subplots(figsize=(7.15, 3.0))
    figure.subplots_adjust(left=0.012, right=0.988, bottom=0.02, top=0.98)
    axis.set_xlim(0, 100)
    axis.set_ylim(0, 40)
    axis.axis("off")
    ranges = [(1, 25), (27, 54), (56, 79), (81, 99)]
    for start, end in ranges:
        _panel(axis, start, 2.0, end - start, 36.0, facecolor="#FAFBFC", edgecolor="#D7DEE5", radius=0.65)

    _step_heading(axis, 4.0, 35.0, 1, "Class-conditioned pool", SPATIAL)
    _step_heading(axis, 30.0, 35.0, 2, "Competitive slots", SPATIAL)
    _step_heading(axis, 59.0, 35.0, 3, "Full-space geometry", FUSION)
    _step_heading(axis, 84.0, 35.0, 4, "CEM update", GOOD)

    axis.text(5.0, 28.6, "current batch", fontsize=5.0, color=MUTED)
    batch_points = [(5.5, 25.0), (9.1, 27.0), (12.5, 24.5), (16.0, 26.2)]
    axis.scatter([point[0] for point in batch_points], [point[1] for point in batch_points], s=23, color=SPATIAL, edgecolors=WHITE, linewidths=0.6, zorder=5)
    axis.text(5.0, 17.1, "64-feature class memory", fontsize=5.0, color=MUTED)
    for row in range(3):
        for column in range(7):
            alpha = 0.24 + row * 0.10
            axis.add_patch(
                Rectangle(
                    (5.2 + column * 2.0 + row * 0.35, 10.8 + row * 1.55),
                    1.1,
                    0.9,
                    linewidth=0.4,
                    edgecolor=SPATIAL,
                    facecolor=mpl.colors.to_rgba(SPATIAL, alpha),
                    zorder=4,
                )
            )
    _orth_arrow(axis, [(19.8, 25.5), (22.2, 25.5), (22.2, 18.8)], color=SPATIAL)
    _orth_arrow(axis, [(19.8, 13.8), (22.2, 13.8), (22.2, 18.8)], color=SPATIAL)
    axis.add_patch(Circle((22.2, 18.8), 1.65, facecolor=SPATIAL_LIGHT, edgecolor=SPATIAL, linewidth=0.9, zorder=5))
    axis.text(22.2, 18.8, "$\\mathcal{B}_y$", ha="center", va="center", fontsize=6.4, color=SPATIAL, zorder=6)

    _editorial_card(
        axis,
        30.0,
        24.0,
        10.2,
        5.6,
        "Fixed projection",
        "64-D assignment space",
        color=SPATIAL,
        fill=SPATIAL_LIGHT,
        title_size=5.8,
        detail_size=4.4,
    )
    _orth_arrow(axis, [(40.2, 26.8), (43.0, 26.8)], color=SPATIAL)
    slot_colors = [SPATIAL, SEMANTIC, NOISE, FUSION, ATTACK, "#56B4E9", "#CC79A7", "#84919C"]
    slot_positions = []
    for row in range(2):
        for column in range(4):
            position = (44.8 + column * 2.25, 22.6 - row * 4.2)
            slot_positions.append(position)
            axis.add_patch(Circle(position, 0.78, facecolor=slot_colors[row * 4 + column], edgecolor=WHITE, linewidth=0.65, zorder=5))
    axis.text(48.2, 13.4, "8 slots  x  3 updates", ha="center", fontsize=5.2, fontweight="bold", color=INK)
    arc = mpl.patches.Arc((48.2, 20.5), 11.5, 11.0, theta1=215, theta2=505, color=SPATIAL, linewidth=0.8, linestyle=(0, (3, 2)), zorder=2)
    axis.add_patch(arc)
    axis.text(34.8, 7.0, "soft competitive assignment", fontsize=4.7, color=MUTED)

    rng = np.random.default_rng(19)
    centers = np.array([[61.7, 12.0], [68.0, 23.7], [74.0, 14.0]])
    geometry_colors = [SPATIAL, FUSION, NOISE]
    for center, color in zip(centers, geometry_colors):
        samples = center + rng.normal(0, [1.25, 1.35], size=(7, 2))
        axis.scatter(samples[:, 0], samples[:, 1], s=12, color=color, alpha=0.42, edgecolors="none", zorder=4)
        axis.scatter([center[0]], [center[1]], marker="X", s=58, color=color, edgecolors=WHITE, linewidths=0.7, zorder=6)
    query = (67.0, 17.8)
    axis.scatter([query[0]], [query[1]], s=27, color=INK, edgecolors=WHITE, linewidths=0.6, zorder=7)
    axis.plot([query[0], centers[1, 0]], [query[1], centers[1, 1]], color=INK, linewidth=0.9, linestyle=(0, (3, 2)), zorder=5)
    axis.text(64.1, 29.7, "assignment chooses $c_{y,j}$", fontsize=4.8, color=MUTED)
    axis.text(
        67.5,
        6.8,
        "distance $\\|z_i-c_{y,j}\\|_2^2$ is measured\nin the full smashed-feature space",
        ha="center",
        fontsize=4.7,
        color=INK,
        linespacing=1.08,
    )

    _box(
        axis,
        83.5,
        24.0,
        13.0,
        5.5,
        "$V_y=\\frac{1}{n_y}\\sum_i\\min_j d_{ij}$",
        "geometric statistic",
        facecolor=SPATIAL_LIGHT,
        edgecolor=SPATIAL,
        title_color=INK,
        fontsize=4.9,
        detail_size=4.5,
    )
    _orth_arrow(axis, [(90.0, 24.0), (90.0, 20.2)], color=GOOD)
    _box(
        axis,
        83.5,
        14.1,
        13.0,
        5.5,
        "CEM objective",
        "maximise $H(X|Z)$",
        facecolor=GOOD_LIGHT,
        edgecolor=GOOD,
        title_color=GOOD,
        fontsize=5.2,
        detail_size=4.6,
    )
    _orth_arrow(axis, [(90.0, 14.1), (90.0, 10.3)], color=GOOD)
    axis.text(
        90.0,
        7.1,
        "FROZEN SPATIAL TARGET",
        ha="center",
        va="center",
        fontsize=4.5,
        fontweight="bold",
        color=GOOD,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": WHITE, "edgecolor": GOOD, "linewidth": 0.8},
    )
    for x_value in (26.0, 55.0, 80.0):
        _orth_arrow(axis, [(x_value - 0.7, 20.0), (x_value + 0.7, 20.0)], color=LINE, linewidth=0.9)

    _save(figure, output_dir, "slot_cem_mechanism")


def _draw_lock(axis: plt.Axes, x: float, y: float, color: str) -> None:
    axis.add_patch(Rectangle((x, y), 1.7, 1.45, facecolor=color, edgecolor="none", zorder=6))
    arc = mpl.patches.Arc((x + 0.85, y + 1.45), 1.25, 1.35, theta1=0, theta2=180, color=color, linewidth=1.0, zorder=6)
    axis.add_patch(arc)


def _status_cell(
    axis: plt.Axes,
    x: float,
    y: float,
    width: float,
    text_value: str,
    *,
    color: str,
    frozen: bool,
) -> None:
    fill = "#EEF1F4" if frozen else mpl.colors.to_rgba(color, 0.13)
    edge = LINE if frozen else color
    _panel(axis, x, y, width, 4.8, facecolor=fill, edgecolor=edge, linewidth=0.7, radius=0.45, zorder=2)
    if frozen:
        _draw_lock(axis, x + 1.2, y + 1.45, BASELINE)
        text_x = x + 3.8
    else:
        axis.add_patch(Circle((x + 2.0, y + 2.4), 0.72, facecolor=color, edgecolor=WHITE, linewidth=0.5, zorder=6))
        text_x = x + 3.5
    axis.text(text_x, y + 2.4, text_value, ha="left", va="center", fontsize=5.0, fontweight="bold", color=BASELINE if frozen else color, zorder=6)


def generate_training_protocol_editorial(output_dir: Path) -> None:
    figure, axis = plt.subplots(figsize=(7.15, 3.25))
    figure.subplots_adjust(left=0.012, right=0.988, bottom=0.02, top=0.98)
    axis.set_xlim(0, 100)
    axis.set_ylim(0, 44)
    axis.axis("off")

    axis.text(2.0, 41.5, "TARGET OPTIMISATION", fontsize=5.6, fontweight="bold", color=MUTED)
    axis.text(61.0, 41.5, "FORMAL PRIVACY EVALUATION", fontsize=5.6, fontweight="bold", color=MUTED)
    _panel(axis, 1.0, 2.0, 56.5, 37.8, facecolor="#FAFBFC", edgecolor="#D7DEE5", radius=0.65)
    _panel(axis, 59.5, 2.0, 39.5, 37.8, facecolor="#FAFBFC", edgecolor="#D7DEE5", radius=0.65)

    axis.add_patch(Rectangle((15.5, 34.5), 24.0, 4.0, facecolor=SEMANTIC, edgecolor="none", zorder=3))
    axis.add_patch(Rectangle((39.5, 34.5), 16.0, 4.0, facecolor=SPATIAL, edgecolor="none", zorder=3))
    axis.text(27.5, 36.5, "STAGE 1  |  80 epochs", ha="center", va="center", fontsize=5.7, fontweight="bold", color=WHITE)
    axis.text(47.5, 36.5, "STAGE 2  |  40 epochs", ha="center", va="center", fontsize=5.0, fontweight="bold", color=WHITE)
    axis.text(13.7, 36.5, "MODULE", ha="right", va="center", fontsize=5.1, fontweight="bold", color=MUTED)

    rows = [
        ("Spatial client", True, True),
        ("Spatial server", True, False),
        ("Semantic path", False, False),
        ("Fusion", False, False),
    ]
    row_y = [28.7, 22.4, 16.1, 9.8]
    for (label, frozen_stage1, frozen_stage2), y_value in zip(rows, row_y):
        axis.text(13.7, y_value + 2.4, label, ha="right", va="center", fontsize=5.3, color=INK)
        _status_cell(axis, 16.0, y_value, 22.6, "FROZEN" if frozen_stage1 else "TRAINED", color=SEMANTIC, frozen=frozen_stage1)
        _status_cell(axis, 40.3, y_value, 14.4, "FROZEN" if frozen_stage2 else "TRAINED", color=SPATIAL if label.startswith("Spatial") else SEMANTIC, frozen=frozen_stage2)
    axis.text(16.0, 5.7, "$\\sigma_s=.025,\\;\\sigma_t=.05,\\;\\lambda_t=.35$", fontsize=4.8, color=MUTED)
    axis.text(40.3, 5.7, "$\\sigma_s=.22,\\;\\sigma_t=.10,\\;\\lambda_t=.50$", fontsize=4.8, color=MUTED)

    axis.add_patch(Rectangle((61.5, 34.5), 35.5, 4.0, facecolor=ATTACK, edgecolor="none", zorder=3))
    axis.text(79.25, 36.5, "FREEZE TARGET   |   TRAIN ATTACKERS", ha="center", va="center", fontsize=5.2, fontweight="bold", color=WHITE)
    _box(
        axis,
        62.5,
        27.2,
        12.0,
        5.0,
        "Deployment",
        "$\\sigma_s=.31,\\;\\sigma_t=.10$",
        facecolor=NOISE_LIGHT,
        edgecolor=NOISE,
        title_color="#8B5E00",
        fontsize=5.5,
        detail_size=4.7,
    )
    for index in range(5):
        axis.add_patch(Circle((78.3 + index * 2.1, 29.7), 0.62, facecolor=FUSION, edgecolor=WHITE, linewidth=0.5, zorder=6))
        axis.text(78.3 + index * 2.1, 27.9, str(126 + index), ha="center", va="top", fontsize=3.9, color=MUTED)
    axis.text(82.5, 32.0, "five independent targets", ha="center", fontsize=4.7, color=INK)
    _orth_arrow(axis, [(74.5, 29.7), (76.5, 29.7)], color=FUSION)

    _box(
        axis,
        62.5,
        17.7,
        12.0,
        5.0,
        "Frozen target",
        "hash verified",
        facecolor="#EEF1F4",
        edgecolor=BASELINE,
        title_color=BASELINE,
        fontsize=5.0,
        detail_size=4.5,
    )
    _orth_arrow(axis, [(68.5, 27.2), (68.5, 22.7)], color=BASELINE)
    _draw_lock(axis, 76.3, 19.35, BASELINE)

    _box(
        axis,
        80.5,
        17.2,
        15.5,
        6.0,
        "Attack suite",
        "residual | GAN | adaptive",
        facecolor=ATTACK_LIGHT,
        edgecolor=ATTACK,
        title_color=ATTACK,
        fontsize=5.7,
        detail_size=4.5,
    )
    _orth_arrow(axis, [(78.2, 20.2), (80.5, 20.2)], color=ATTACK)
    axis.text(
        88.25,
        14.5,
        "3 seeds per attack  |  2 knowledge settings",
        ha="center",
        fontsize=4.0,
        color=MUTED,
    )

    _panel(axis, 62.5, 5.0, 33.5, 6.1, facecolor=WHITE, edgecolor="#D7DEE5", radius=0.45)
    metrics = [("MSE", SPATIAL), ("LPIPS", FUSION), ("SSIM", SEMANTIC), ("ID cosine", ATTACK)]
    for index, (metric, color) in enumerate(metrics):
        x_value = 66.0 + index * 7.7
        axis.add_patch(Circle((x_value, 8.0), 0.72, facecolor=color, edgecolor=WHITE, linewidth=0.5, zorder=5))
        axis.text(x_value + 1.15, 8.0, metric, ha="left", va="center", fontsize=4.8, color=INK)
    axis.text(79.2, 3.2, "No reconstruction loss enters target training", ha="center", fontsize=5.0, fontweight="bold", color=ATTACK)
    _orth_arrow(axis, [(88.25, 17.2), (88.25, 11.1)], color=ATTACK)

    _save(figure, output_dir, "training_protocol")


def _mean_component(summary: dict, component: str) -> float:
    values = [
        float(row["accuracy"])
        for row in summary["component_records"]
        if row["component"] == component
    ]
    return float(np.mean(values))


def generate_results_overview(summary: dict, output_dir: Path) -> None:
    figure = plt.figure(figsize=(7.15, 4.55))
    grid = figure.add_gridspec(
        2,
        2,
        height_ratios=[1.0, 0.72],
        width_ratios=[1.02, 0.98],
        hspace=0.38,
        wspace=0.36,
    )
    ax_tradeoff = figure.add_subplot(grid[0, 0])
    ax_gain = figure.add_subplot(grid[0, 1])
    ax_ablation = figure.add_subplot(grid[1, :])

    published = summary["published"]
    observed = summary["observed"]
    accuracies = np.array([float(row["accuracy"]) * 100.0 for row in summary["utility_records"]])
    mse_values = np.array(summary["target_attack_means"]["decoder_inference_mse"], dtype=float)
    seeds = [int(row["target_seed"]) for row in summary["utility_records"]]
    x_mean = float(accuracies.mean())
    y_mean = float(mse_values.mean())

    published_accuracy = float(published["accuracy"]) * 100.0
    published_mse = float(published["decoder_inference_mse"])
    ax_tradeoff.add_patch(
        Rectangle(
            (published_accuracy, published_mse),
            82.40 - published_accuracy,
            0.0388 - published_mse,
            facecolor=GOOD_LIGHT,
            edgecolor="none",
            alpha=0.72,
            zorder=0,
        )
    )
    ax_tradeoff.scatter(
        accuracies,
        mse_values,
        s=34,
        color=SPATIAL,
        edgecolors=WHITE,
        linewidths=0.8,
        zorder=4,
        label="DualPath-CEM targets",
    )
    seed_offsets = [(-8, 6), (0, 11), (2, -11), (5, 3), (-3, 10)]
    for x, y, seed, offset in zip(accuracies, mse_values, seeds, seed_offsets):
        ax_tradeoff.annotate(
            str(seed),
            (x, y),
            xytext=offset,
            textcoords="offset points",
            fontsize=6.1,
            color=MUTED,
        )
    ax_tradeoff.scatter(
        [float(published["accuracy"]) * 100.0],
        [float(published["decoder_inference_mse"])],
        marker="s",
        s=54,
        color=BASELINE,
        edgecolors=WHITE,
        linewidths=0.8,
        zorder=5,
        label="Published CEM",
    )
    ax_tradeoff.errorbar(
        [x_mean],
        [y_mean],
        xerr=[[
            x_mean - float(summary["intervals"]["accuracy"]["lower"]) * 100.0
        ], [
            float(summary["intervals"]["accuracy"]["upper"]) * 100.0 - x_mean
        ]],
        yerr=[[
            y_mean - float(summary["intervals"]["decoder_inference_mse"]["lower"])
        ], [
            float(summary["intervals"]["decoder_inference_mse"]["upper"]) - y_mean
        ]],
        fmt="D",
        markersize=6.0,
        markerfacecolor=ATTACK,
        markeredgecolor=WHITE,
        ecolor=ATTACK,
        elinewidth=1.1,
        capsize=2.5,
        zorder=6,
        label="Mean (95% CI)",
    )
    ax_tradeoff.axvline(float(published["accuracy"]) * 100.0, color=BASELINE, linewidth=0.8, linestyle=(0, (3, 2)), alpha=0.75)
    ax_tradeoff.axhline(float(published["decoder_inference_mse"]), color=BASELINE, linewidth=0.8, linestyle=(0, (3, 2)), alpha=0.75)
    ax_tradeoff.text(81.48, 0.0347, "higher utility\nand privacy", ha="center", va="center", fontsize=6.2, color=GOOD, fontweight="semibold")
    ax_tradeoff.annotate("", xy=(81.82, 0.0363), xytext=(81.55, 0.0351), arrowprops={"arrowstyle": "-|>", "color": GOOD, "lw": 0.9})
    ax_tradeoff.set_xlim(80.05, 82.40)
    ax_tradeoff.set_ylim(0.0190, 0.0388)
    ax_tradeoff.set_xlabel("Top-1 accuracy (%)")
    ax_tradeoff.set_ylabel("Decoder inference MSE")
    ax_tradeoff.grid(True, color=GRID, linewidth=0.65, zorder=0)
    ax_tradeoff.legend(frameon=False, loc="lower right", handletextpad=0.45, labelspacing=0.35)
    ax_tradeoff.set_title("(a) Cross-seed privacy-utility trade-off", loc="left", color=INK, fontweight="semibold", pad=6)

    metric_rows = [
        ("Accuracy", "accuracy"),
        ("Decoder train", "decoder_training_mse"),
        ("Decoder inference", "decoder_inference_mse"),
        ("GAN train", "gan_training_mse"),
        ("GAN inference", "gan_inference_mse"),
    ]
    gains = [
        100.0 * (float(observed[key]) / float(published[key]) - 1.0)
        for _, key in metric_rows
    ]
    y_positions = np.arange(len(metric_rows))[::-1]
    colors = [SEMANTIC] + [SPATIAL] * 4
    bars = ax_gain.barh(y_positions, gains, color=colors, height=0.58, edgecolor="none", zorder=3)
    ax_gain.axvline(0, color=INK, linewidth=0.8)
    ax_gain.set_yticks(y_positions, [label for label, _ in metric_rows])
    ax_gain.set_xlim(0, max(gains) * 1.22)
    ax_gain.set_xlabel("Relative improvement (%)")
    ax_gain.grid(True, axis="x", color=GRID, linewidth=0.65, zorder=0)
    for bar, value in zip(bars, gains):
        ax_gain.text(value + 1.3, bar.get_y() + bar.get_height() / 2, f"+{value:.1f}%", va="center", fontsize=7.0, color=INK, fontweight="bold")
    ax_gain.text(0.0, -0.23, "Accuracy measures utility; MSE measures reconstruction privacy.",
                 transform=ax_gain.transAxes, fontsize=5.5, color=MUTED)
    ax_gain.set_title("(b) Gain over published CEM", loc="left", color=INK, fontweight="semibold", pad=6)

    ablation_labels = ["Spatial only", "Semantic only", "Stage 1 fusion", "Stage 2 fusion"]
    ablation_values = [
        100.0 * _mean_component(summary, "legacy"),
        100.0 * _mean_component(summary, "semantic"),
        100.0 * float(np.mean([row["accuracy"] for row in summary["stage1_records"]])),
        100.0 * _mean_component(summary, "fused"),
    ]
    ablation_colors = [SPATIAL, SEMANTIC, FUSION, ATTACK]
    y_ablation = np.arange(len(ablation_labels))[::-1]
    ax_ablation.hlines(y_ablation, 55.0, ablation_values, color=ablation_colors, linewidth=3.0, alpha=0.38, zorder=2)
    ax_ablation.scatter(ablation_values, y_ablation, s=58, color=ablation_colors, edgecolors=WHITE, linewidths=0.9, zorder=4)
    baseline_accuracy = float(published["accuracy"]) * 100.0
    ax_ablation.axvline(baseline_accuracy, color=BASELINE, linewidth=1.0, linestyle=(0, (3, 2)), zorder=1)
    ax_ablation.text(
        baseline_accuracy - 0.25,
        2.60,
        "Published CEM\n80.33",
        ha="right",
        va="center",
        fontsize=6.2,
        color=BASELINE,
        bbox={"facecolor": WHITE, "edgecolor": "none", "alpha": 0.82, "pad": 1.2},
    )
    for value, y_value in zip(ablation_values, y_ablation):
        ax_ablation.text(value + 0.45, y_value, f"{value:.2f}", va="center", fontsize=7.0, color=INK, fontweight="bold")
    ax_ablation.set_yticks(y_ablation, ablation_labels)
    ax_ablation.set_xlim(55.0, 83.2)
    ax_ablation.set_xlabel("Top-1 accuracy (%)")
    ax_ablation.grid(True, axis="x", color=GRID, linewidth=0.65, zorder=0)
    ax_ablation.set_title("(c) Contribution of each prediction path", loc="left", color=INK, fontweight="semibold", pad=6)

    for axis in (ax_tradeoff, ax_gain, ax_ablation):
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_color(LINE)
        axis.spines["bottom"].set_color(LINE)
        axis.tick_params(colors=MUTED, length=3)

    figure.subplots_adjust(left=0.12, right=0.985, top=0.93, bottom=0.105)
    _save(figure, output_dir, "results_overview")


def generate_attack_robustness(summary: dict, output_dir: Path) -> None:
    figure = plt.figure(figsize=(7.15, 2.75))
    grid = figure.add_gridspec(1, 2, width_ratios=[1.02, 0.98], wspace=0.38)
    ax_mse = figure.add_subplot(grid[0, 0])
    ax_metrics = figure.add_subplot(grid[0, 1])

    attacks = ["decoder", "gan", "adaptive"]
    attack_labels = ["Residual decoder", "GAN", "Adaptive decoder"]
    means: dict[tuple[str, str, str], float] = {}
    for attack in attacks:
        for knowledge in ("training", "inference"):
            rows = [
                row
                for row in summary["attack_records"]
                if row["attack"] == attack and row["knowledge"] == knowledge
            ]
            for metric in ("mse", "lpips", "identity_cosine_similarity"):
                means[(attack, knowledge, metric)] = float(
                    np.mean([float(row[metric]) for row in rows])
                )

    x_positions = np.arange(len(attacks))
    training = np.array([means[(attack, "training", "mse")] for attack in attacks])
    inference = np.array([means[(attack, "inference", "mse")] for attack in attacks])
    for x, train_value, inference_value in zip(x_positions, training, inference):
        ax_mse.plot([x, x], [train_value, inference_value], color=LINE, linewidth=2.2, zorder=1)
    ax_mse.scatter(x_positions, training, s=46, color=BASELINE, edgecolors=WHITE, linewidths=0.8, zorder=3, label="Training knowledge")
    ax_mse.scatter(x_positions, inference, s=52, color=ATTACK, edgecolors=WHITE, linewidths=0.8, zorder=4, label="Inference knowledge")
    for x, value in zip(x_positions, inference):
        ax_mse.text(x, value + 0.00045, f"{value:.4f}", ha="center", va="bottom", fontsize=6.4, color=ATTACK, fontweight="bold")
    ax_mse.set_xticks(x_positions, ["Residual", "GAN", "Adaptive"])
    ax_mse.set_ylim(0.024, 0.043)
    ax_mse.set_ylabel("Reconstruction MSE")
    ax_mse.grid(True, axis="y", color=GRID, linewidth=0.65, zorder=0)
    ax_mse.legend(frameon=False, loc="upper left", ncols=1, handletextpad=0.4, labelspacing=0.3)
    ax_mse.set_title("(a) MSE by attacker and knowledge setting", loc="left", color=INK, fontweight="semibold", pad=6)

    raw = np.array(
        [
            [
                means[(attack, "inference", "mse")],
                means[(attack, "inference", "lpips")],
                means[(attack, "inference", "identity_cosine_similarity")],
            ]
            for attack in attacks
        ]
    )
    scores = np.zeros_like(raw)
    for column in range(raw.shape[1]):
        values = raw[:, column]
        span = max(float(values.max() - values.min()), 1e-12)
        normalised = (values - values.min()) / span
        scores[:, column] = 1.0 - normalised if column == 2 else normalised
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "privacy_blue", ["#F4F8FB", "#9DCAE1", SPATIAL]
    )
    ax_metrics.imshow(scores, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    headers = ["MSE\n$\\uparrow$", "LPIPS\n$\\uparrow$", "ID cosine\n$\\downarrow$"]
    ax_metrics.set_xticks(np.arange(3), headers)
    ax_metrics.xaxis.tick_top()
    ax_metrics.tick_params(top=False, bottom=False, left=False)
    ax_metrics.set_yticks(np.arange(3), attack_labels)
    formats = ["{:.4f}", "{:.4f}", "{:.4f}"]
    for row in range(raw.shape[0]):
        for column in range(raw.shape[1]):
            text_color = WHITE if scores[row, column] > 0.62 else INK
            ax_metrics.text(
                column,
                row,
                formats[column].format(raw[row, column]),
                ha="center",
                va="center",
                fontsize=7.0,
                fontweight="bold",
                color=text_color,
            )
    for position in np.arange(-0.5, 3, 1):
        ax_metrics.axhline(position, color=WHITE, linewidth=2.0)
        ax_metrics.axvline(position, color=WHITE, linewidth=2.0)
    ax_metrics.set_title("(b) Inference-knowledge privacy profile", loc="left", color=INK, fontweight="semibold", pad=8)
    ax_metrics.text(
        0.5,
        -0.19,
        "Arrows indicate the defender-favourable direction",
        transform=ax_metrics.transAxes,
        ha="center",
        fontsize=6.3,
        color=MUTED,
    )

    for axis in (ax_mse, ax_metrics):
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_color(LINE)
        axis.spines["bottom"].set_color(LINE)
        axis.tick_params(colors=MUTED)
    figure.subplots_adjust(left=0.11, right=0.985, top=0.77, bottom=0.24)
    _save(figure, output_dir, "attack_robustness")


def _load_font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates: Iterable[str]
    if bold:
        candidates = (
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "DejaVuSans-Bold.ttf",
        )
    else:
        candidates = (
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "DejaVuSans.ttf",
        )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def refine_qualitative_grid(source: Path, destination: Path) -> None:
    with Image.open(source) as image:
        source_image = image.convert("RGB")

    rows = 3
    columns = 8
    scale = max(1, round(source_image.height / 212))
    source_padding = 5 * scale
    source_tile = 64 * scale
    source_stride = source_tile + source_padding
    label_width = 120 * scale
    expected_size = (
        label_width + columns * source_stride + source_padding,
        rows * source_stride + source_padding,
    )
    if source_image.size != expected_size:
        raise ValueError(f"unexpected qualitative-grid geometry: {source_image.size}")

    tile = 280
    top = 62
    left = 184
    gap = 12
    row_gap = 13
    outer = 16
    output_width = left + columns * tile + (columns - 1) * gap + outer
    output_height = top + rows * tile + (rows - 1) * row_gap + outer
    canvas = Image.new("RGB", (output_width, output_height), "white")
    draw = ImageDraw.Draw(canvas)
    font_small = _load_font(25)
    font_label = _load_font(31, bold=True)

    for column in range(columns):
        x = left + column * (tile + gap)
        label = f"Example {column + 1}"
        box = draw.textbbox((0, 0), label, font=font_small)
        draw.text((x + (tile - (box[2] - box[0])) / 2, 17), label, fill="#5B6773", font=font_small)

    row_labels = ["Original", "Residual\ndecoder", "GAN"]
    row_colors = ["#6B7280", SPATIAL, ATTACK]
    row_fills = ["#F6F8FA", "#F4F9FC", "#FDF7F3"]
    for row in range(rows):
        y = top + row * (tile + row_gap)
        draw.rounded_rectangle(
            (outer, y - 5, output_width - outer, y + tile + 5),
            radius=7,
            fill=row_fills[row],
            outline="#E2E7EB",
            width=2,
        )
        draw.rounded_rectangle((outer + 8, y + 18, outer + 18, y + tile - 18), radius=4, fill=row_colors[row])
        label_lines = row_labels[row].split("\n")
        line_height = 36
        label_y = y + tile / 2 - len(label_lines) * line_height / 2
        for index, label_line in enumerate(label_lines):
            draw.text((outer + 34, label_y + index * line_height), label_line, fill="#17212B", font=font_label)
        for column in range(columns):
            source_x = label_width + column * source_stride
            source_y = source_padding + row * source_stride
            crop = source_image.crop(
                (source_x, source_y, source_x + source_tile, source_y + source_tile)
            )
            if source_tile != tile:
                crop = crop.resize((tile, tile), Image.Resampling.LANCZOS)
            target_x = left + column * (tile + gap)
            canvas.paste(crop, (target_x, y))
            draw.rectangle((target_x, y, target_x + tile - 1, y + tile - 1), outline="#D5DCE2", width=2)

    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, format="PNG", dpi=(450, 450), optimize=True)


def generate_all(summary: dict, output_dir: Path, qualitative_source: Path | None = None) -> None:
    generate_architecture_editorial(output_dir, qualitative_source)
    generate_slot_cem_editorial(output_dir)
    generate_training_protocol_editorial(output_dir)
    generate_results_overview(summary, output_dir)
    generate_attack_robustness(summary, output_dir)
    if qualitative_source is not None and qualitative_source.is_file():
        refine_qualitative_grid(qualitative_source, output_dir / "qualitative_grid_refined.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-summary", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--qualitative-source", type=Path)
    args = parser.parse_args()
    summary = json.loads(args.formal_summary.read_text(encoding="utf-8"))
    generate_all(summary, args.output_dir, args.qualitative_source)
    print(
        json.dumps(
            {
                "status": "PASS",
                "output_dir": str(args.output_dir),
                "figures": [
                    "architecture",
                    "slot_cem_mechanism",
                    "training_protocol",
                    "results_overview",
                    "attack_robustness",
                    "qualitative_grid_refined" if args.qualitative_source else None,
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
