#!/usr/bin/env python3
"""Generate editable SVG artwork plus publication-ready PDF/PNG figures.

The drawing layer is native SVG rather than Matplotlib.  This keeps diagrams
editable, preserves vector text and paths in the manuscript PDF, and gives the
method and result figures one consistent visual grammar. CairoSVG produces the
vector PDF, while Chromium is used only for high-resolution PNG previews.
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
import math
from pathlib import Path
from statistics import mean
from typing import Sequence

import cairosvg
from PIL import Image
from playwright.sync_api import sync_playwright


INK = "#17212B"
MUTED = "#667382"
SUBTLE = "#94A1AE"
LINE = "#CBD5DF"
GRID = "#E3E9EF"
PANEL = "#F6F8FA"
WHITE = "#FFFFFF"
BLUE = "#0B73B7"
BLUE_LIGHT = "#E2F1F9"
GREEN = "#008F72"
GREEN_LIGHT = "#E2F4EE"
AMBER = "#D88900"
AMBER_LIGHT = "#FFF2D2"
PURPLE = "#6C5BC6"
PURPLE_LIGHT = "#ECE9FA"
ATTACK = "#CE5300"
ATTACK_LIGHT = "#FCE8DC"
GRAY = "#6F7A88"
GRAY_LIGHT = "#EDF1F4"
GOOD = "#2C8154"
GOOD_LIGHT = "#E7F4EC"

FONT = "Arial Unicode MS"


TEXT = {
    "en": {
        "architecture_title": "DualPath-CEM: dual-path protected inference and Slot-CEM pretraining",
        "trusted_client": "TRUSTED CLIENT",
        "transmission": "OBSERVED TRANSMISSION",
        "untrusted_cloud": "UNTRUSTED CLOUD",
        "input_faces": "private face",
        "spatial_path": "SPATIAL / PRIVACY PATH",
        "semantic_path": "SEMANTIC / UTILITY PATH",
        "spatial_encoder": "Slot-CEM spatial encoder",
        "spatial_encoder_detail": "VGG11-BN client + C16 bottleneck",
        "semantic_encoder": "Semantic encoder",
        "semantic_encoder_detail": "MobileNetV3-L + GAP\n256-D projection",
        "spatial_server": "Spatial server",
        "spatial_server_detail": "VGG cloud + identity head",
        "token_classifier": "Token classifier",
        "token_classifier_detail": "linear identity head",
        "noise": "Gaussian channel",
        "fusion": "Calibrated logit fusion",
        "identity": "Identity",
        "classes": "526 identities",
        "frozen": "FROZEN",
        "adapted_s2": "ADAPTED S2",
        "trained_s12": "TRAINED S1+S2",
        "payload": "4,096 spatial + 256 semantic FP32 values  |  17,408 bytes / image",
        "privacy_zoom": "SLOT-CEM PRIVACY REGULARISATION",
        "class_pool": "Same-identity pool",
        "class_pool_detail": "mini-batch +\n64-feature memory",
        "fixed_projection": "Learned slot-space\nprojection",
        "fixed_projection_detail": "4,096 to 64\npretraining only",
        "competitive_slots": "Competitive\nslots",
        "competitive_slots_detail": "8 slots\n3 iterative updates",
        "full_geometry": "Soft within-slot\ngeometry",
        "full_geometry_detail": "weighted distance to\ndetached centroids",
        "cem_objective": "CEM objective",
        "cem_objective_detail": "maximise\nconditional entropy",
        "privacy_gradient": "gradient reaches encoder (scheduled lambda=10), projection, and Slot Attention",
        "evaluation": "FROZEN-TARGET JOINT INVERSION",
        "observed_pair": "attacker observes both released tensors",
        "attackers": "Residual decoder\nGAN\nAdaptive decoder",
        "target_frozen": "target parameters remain frozen",
        "reconstruction": "reconstruction",
        "forward": "forward inference",
        "grad": "privacy gradient",
        "attack_obs": "attacker observation",
        "slot_title": "How Slot-CEM replaces per-batch Gaussian-mixture fitting",
        "evidence": "Class-conditioned evidence",
        "assignment": "Competitive assignment",
        "geometry": "Geometric privacy statistic",
        "update": "Parameter update",
        "current_batch": "current mini-batch",
        "memory_bank": "same-class memory bank",
        "assignment_space": "learned 64-D\nslot space",
        "attention_matrix": "feature-to-slot attention",
        "soft_assignment": "soft assignment, repeated 3 times",
        "choose_centroid": "soft assignments define c_y,j",
        "measure_full": "weighted squared distances are evaluated in the 64-D slot space",
        "detached": "centroids detached",
        "no_shortcut": "Variance is computed from feature distances rather than\npredicted by a separate variance head.",
        "sparse_batch": "Memory supplies same-identity evidence\nwhen a mini-batch is sparse.",
        "training_title": "Two-stage target optimisation and locked formal evaluation",
        "stage0": "SLOT-CEM CHECKPOINT",
        "stage1": "STAGE 1  |  80 epochs",
        "stage2": "STAGE 2  |  40 epochs",
        "formal_eval": "FORMAL PRIVACY EVALUATION",
        "module": "MODULE",
        "spatial_client": "Spatial client",
        "spatial_cloud": "Spatial server",
        "semantic_branch": "Semantic path",
        "fusion_module": "Fusion",
        "locked": "LOCKED",
        "trained": "TRAINED",
        "formal_targets": "5 independent frozen targets",
        "target_seeds": "target seeds 126-130",
        "attacker_matrix": "Residual/GAN: 3 seeds each  |  Adaptive: 1 seed  |  70 runs",
        "residual": "Residual",
        "gan": "GAN",
        "adaptive": "Adaptive",
        "training_knowledge": "training knowledge",
        "inference_knowledge": "inference knowledge",
        "aggregate": "aggregate across target seeds",
        "bootstrap": "mean + 95% bootstrap CI",
        "no_recon_grad": "No reconstruction gradient enters target training",
        "results_a": "(a) Cross-seed privacy-utility plane",
        "results_b": "(b) Gain over the published CEM reference",
        "results_c": "(c) Where the utility gain comes from",
        "accuracy": "Top-1 accuracy (%)",
        "recon_mse": "Reconstruction MSE",
        "higher_better": "higher utility and privacy",
        "published_cem": "Published CEM",
        "our_targets": "DualPath-CEM targets",
        "mean_ci": "mean + 95% CI",
        "relative_gain": "relative gain",
        "decoder_train": "Decoder · train knowledge",
        "decoder_infer": "Decoder · inference knowledge",
        "gan_train": "GAN · train knowledge",
        "gan_infer": "GAN · inference knowledge",
        "spatial_only": "Spatial only",
        "semantic_only": "Semantic only",
        "stage1_fusion": "Stage-1 fusion",
        "stage2_fusion": "Stage-2 fusion",
        "attack_title_a": "(a) Reconstruction error by attacker knowledge",
        "attack_title_b": "(b) Inference-knowledge privacy profile",
        "train": "Training knowledge",
        "infer": "Inference knowledge",
        "defender_direction": "arrowheads mark the defender-favourable direction",
        "mse": "MSE",
        "lpips": "LPIPS",
        "ssim": "SSIM",
        "id_cos": "ID cosine",
        "qual_title": "Fixed qualitative examples under the inference-knowledge protocol",
        "original": "Original",
        "residual_decoder": "Residual\ndecoder",
        "mean_mse": "mean MSE",
        "examples_note": "same declared examples in every row  |  quantitative results use the complete evaluation split",
    },
    "zh": {
        "architecture_title": "DualPath-CEM：双路径受保护推理与 Slot-CEM 预训练",
        "trusted_client": "可信客户端",
        "transmission": "攻击者可观察的传输",
        "untrusted_cloud": "不可信云端",
        "input_faces": "隐私人脸",
        "spatial_path": "空间 / 隐私路径",
        "semantic_path": "语义 / 效用路径",
        "spatial_encoder": "Slot-CEM 空间编码器",
        "spatial_encoder_detail": "VGG11-BN 客户端 + C16 瓶颈",
        "semantic_encoder": "语义编码器",
        "semantic_encoder_detail": "MobileNetV3-L + 全局池化\n256维投影",
        "spatial_server": "空间服务器",
        "spatial_server_detail": "VGG 云端 + 身份分类头",
        "token_classifier": "令牌分类器",
        "token_classifier_detail": "线性身份分类头",
        "noise": "高斯信道",
        "fusion": "校准 logit 融合",
        "identity": "身份",
        "classes": "530 类",
        "frozen": "冻结",
        "adapted_s2": "阶段2适应",
        "trained_s12": "阶段1+2训练",
        "payload": "4,096 个空间值 + 256 个语义 FP32 值  |  每图 17,408 字节",
        "privacy_zoom": "SLOT-CEM 隐私正则化",
        "class_pool": "同身份特征池",
        "class_pool_detail": "当前批次 +\n64 特征记忆库",
        "fixed_projection": "可学习 slot 空间投影",
        "fixed_projection_detail": "4,096 到 64\n仅用于预训练",
        "competitive_slots": "竞争 slot",
        "competitive_slots_detail": "8 个 slot\n3 次迭代更新",
        "full_geometry": "slot 内软几何统计",
        "full_geometry_detail": "到停止梯度质心的\n加权距离",
        "cem_objective": "CEM 目标",
        "cem_objective_detail": "最大化条件熵",
        "privacy_gradient": "梯度更新编码器（调度 lambda=10）、投影层和 Slot Attention",
        "evaluation": "冻结目标后的联合反演",
        "observed_pair": "攻击者观察两条路径释放的全部张量",
        "attackers": "残差解码器\nGAN\n自适应解码器",
        "target_frozen": "目标模型参数始终冻结",
        "reconstruction": "重建结果",
        "forward": "前向推理",
        "grad": "隐私梯度",
        "attack_obs": "攻击者观察",
        "slot_title": "Slot-CEM 如何替代逐批次高斯混合拟合",
        "evidence": "类条件证据",
        "assignment": "竞争分配",
        "geometry": "几何隐私统计量",
        "update": "参数更新",
        "current_batch": "当前 mini-batch",
        "memory_bank": "同类别记忆库",
        "assignment_space": "可学习 64 维\nslot 空间",
        "attention_matrix": "特征到 slot 的注意力",
        "soft_assignment": "软分配，重复更新 3 次",
        "choose_centroid": "软分配确定 c_y,j",
        "measure_full": "加权平方距离在 64 维 slot 空间中计算",
        "detached": "质心停止梯度",
        "no_shortcut": "方差由特征距离直接计算，\n而不是由单独的方差预测头给出。",
        "sparse_batch": "当 mini-batch 中同身份样本不足时，\n记忆库补充类内证据。",
        "training_title": "两阶段目标优化与锁定后的正式评估",
        "stage0": "SLOT-CEM 检查点",
        "stage1": "阶段 1  |  80 epochs",
        "stage2": "阶段 2  |  40 epochs",
        "formal_eval": "正式隐私评估",
        "module": "模块",
        "spatial_client": "空间客户端",
        "spatial_cloud": "空间服务器",
        "semantic_branch": "语义路径",
        "fusion_module": "融合模块",
        "locked": "冻结",
        "trained": "训练",
        "formal_targets": "5 个独立冻结目标模型",
        "target_seeds": "目标种子 126-130",
        "attacker_matrix": "残差/GAN：各 3 个种子  |  自适应：1 个种子  |  共 70 次",
        "residual": "残差",
        "gan": "GAN",
        "adaptive": "自适应",
        "training_knowledge": "训练知识",
        "inference_knowledge": "推理知识",
        "aggregate": "跨目标种子汇总",
        "bootstrap": "均值 + 95% bootstrap 区间",
        "no_recon_grad": "重建梯度不会进入目标模型训练",
        "results_a": "(a) 跨种子隐私-效用平面",
        "results_b": "(b) 相对已发表 CEM 参考点的提升",
        "results_c": "(c) 效用提升来自哪些组件",
        "accuracy": "Top-1 准确率 (%)",
        "recon_mse": "重建 MSE",
        "higher_better": "更高效用与更强隐私",
        "published_cem": "已发表 CEM",
        "our_targets": "DualPath-CEM 目标模型",
        "mean_ci": "均值 + 95% 区间",
        "relative_gain": "相对提升",
        "decoder_train": "解码器 · 训练知识",
        "decoder_infer": "解码器 · 推理知识",
        "gan_train": "GAN · 训练知识",
        "gan_infer": "GAN · 推理知识",
        "spatial_only": "仅空间路径",
        "semantic_only": "仅语义路径",
        "stage1_fusion": "阶段1融合",
        "stage2_fusion": "阶段2融合",
        "attack_title_a": "(a) 不同攻击者知识下的重建误差",
        "attack_title_b": "(b) 推理知识设置下的隐私指标",
        "train": "训练知识",
        "infer": "推理知识",
        "defender_direction": "箭头表示对防御者有利的方向",
        "mse": "MSE",
        "lpips": "LPIPS",
        "ssim": "SSIM",
        "id_cos": "身份余弦",
        "qual_title": "推理知识协议下的固定定性样例",
        "original": "原图",
        "residual_decoder": "残差\n解码器",
        "mean_mse": "平均 MSE",
        "examples_note": "每行使用同一组预先声明的样例  |  定量结果基于完整评估划分",
    },
}


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


class SVG:
    def __init__(self, width: int, height: int, title: str):
        self.width = width
        self.height = height
        self.title = title
        self.parts: list[str] = []
        self.defs: list[str] = []
        self._clip_index = 0
        for name, color in {
            "ink": INK,
            "muted": MUTED,
            "blue": BLUE,
            "green": GREEN,
            "amber": AMBER,
            "purple": PURPLE,
            "attack": ATTACK,
            "gray": GRAY,
            "good": GOOD,
        }.items():
            self.defs.append(
                f'<marker id="arrow-{name}" viewBox="0 0 10 10" refX="9" refY="5" '
                f'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
                f'<path d="M0,0 L10,5 L0,10 z" fill="{color}"/></marker>'
            )
        self.defs.append(
            '<filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">'
            '<feDropShadow dx="0" dy="3" stdDeviation="5" flood-color="#1F2937" flood-opacity="0.10"/>'
            '</filter>'
        )

    def rect(self, x: float, y: float, w: float, h: float, *, fill=WHITE, stroke="none", sw=1.5,
             rx=0, opacity=1.0, shadow=False, dash: str | None = None) -> None:
        attrs = [f'x="{x}"', f'y="{y}"', f'width="{w}"', f'height="{h}"', f'fill="{fill}"',
                 f'stroke="{stroke}"', f'stroke-width="{sw}"', f'rx="{rx}"', f'opacity="{opacity}"']
        if shadow:
            attrs.append('filter="url(#shadow)"')
        if dash:
            attrs.append(f'stroke-dasharray="{dash}"')
        self.parts.append(f'<rect {" ".join(attrs)}/>')

    def line(self, x1, y1, x2, y2, *, stroke=INK, sw=2, dash: str | None = None, opacity=1.0) -> None:
        extra = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" '
            f'stroke-width="{sw}" opacity="{opacity}"{extra}/>'
        )

    def circle(self, cx, cy, r, *, fill=WHITE, stroke="none", sw=1.5, opacity=1.0) -> None:
        self.parts.append(
            f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}" stroke="{stroke}" '
            f'stroke-width="{sw}" opacity="{opacity}"/>'
        )

    def ellipse(self, cx, cy, rx, ry, *, fill="none", stroke=INK, sw=1.5, opacity=1.0,
                dash: str | None = None, rotate: float = 0) -> None:
        extra = f' stroke-dasharray="{dash}"' if dash else ""
        transform = f' transform="rotate({rotate} {cx} {cy})"' if rotate else ""
        self.parts.append(
            f'<ellipse cx="{cx}" cy="{cy}" rx="{rx}" ry="{ry}" fill="{fill}" stroke="{stroke}" '
            f'stroke-width="{sw}" opacity="{opacity}"{extra}{transform}/>'
        )

    def path(self, d: str, *, stroke=INK, sw=2, fill="none", dash: str | None = None,
             marker: str | None = None, opacity=1.0) -> None:
        extra = f' stroke-dasharray="{dash}"' if dash else ""
        marker_attr = f' marker-end="url(#arrow-{marker})"' if marker else ""
        self.parts.append(
            f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" '
            f'stroke-linecap="round" stroke-linejoin="round" opacity="{opacity}"{extra}{marker_attr}/>'
        )

    def polygon(self, points: Sequence[tuple[float, float]], *, fill=WHITE, stroke=INK, sw=1.5,
                opacity=1.0) -> None:
        p = " ".join(f"{x},{y}" for x, y in points)
        self.parts.append(
            f'<polygon points="{p}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" opacity="{opacity}"/>'
        )

    def text(self, x, y, value: str | Sequence[str], *, size=24, fill=INK, weight=400,
             anchor="start", italic=False, line_height=1.18, opacity=1.0,
             letter_spacing=0, family=FONT, rotate: float | None = None) -> None:
        lines = value.split("\n") if isinstance(value, str) else list(value)
        style = "font-style:italic;" if italic else ""
        transform = f' transform="rotate({rotate} {x} {y})"' if rotate is not None else ""
        self.parts.append(
            f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" font-weight="{weight}" '
            f'fill="{fill}" text-anchor="{anchor}" opacity="{opacity}" letter-spacing="{letter_spacing}" '
            f'style="{style}"{transform}>'
        )
        for index, line in enumerate(lines):
            dy = 0 if index == 0 else size * line_height
            self.parts.append(f'<tspan x="{x}" dy="{dy}">{esc(line)}</tspan>')
        self.parts.append('</text>')

    def image(self, x, y, w, h, data_uri: str, *, rx=0, stroke=LINE, sw=1.5) -> None:
        clip_id = f"clip-{self._clip_index}"
        self._clip_index += 1
        self.defs.append(f'<clipPath id="{clip_id}"><rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}"/></clipPath>')
        self.parts.append(
            f'<image x="{x}" y="{y}" width="{w}" height="{h}" href="{data_uri}" '
            f'preserveAspectRatio="xMidYMid slice" clip-path="url(#{clip_id})"/>'
        )
        self.rect(x, y, w, h, fill="none", stroke=stroke, sw=sw, rx=rx)

    def render(self) -> str:
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
            f'width="{self.width}" height="{self.height}" viewBox="0 0 {self.width} {self.height}" '
            f'role="img" aria-label="{esc(self.title)}">'
            f'<title>{esc(self.title)}</title><defs>{"".join(self.defs)}</defs>'
            f'<rect width="100%" height="100%" fill="#FFFFFF"/>{"".join(self.parts)}</svg>'
        )


def panel(svg: SVG, x, y, w, h, *, fill=PANEL, stroke=LINE, rx=12, shadow=False) -> None:
    svg.rect(x, y, w, h, fill=fill, stroke=stroke, sw=1.6, rx=rx, shadow=shadow)


def badge(svg: SVG, x, y, value: str, color: str, *, width: float | None = None, fill=WHITE) -> None:
    width = width or max(76, 13 * len(value))
    svg.rect(x, y, width, 27, fill=fill, stroke=color, sw=1.4, rx=13.5)
    svg.text(x + width / 2, y + 19, value, size=14, fill=color, weight=700, anchor="middle")


def card(svg: SVG, x, y, w, h, title: str, detail: str, color: str, light: str,
         *, state: str | None = None, title_size=22, detail_size=16) -> None:
    svg.rect(x, y, w, h, fill=light, stroke=color, sw=2.2, rx=9, shadow=True)
    svg.rect(x, y, 8, h, fill=color, rx=4)
    title_lines = title.split("\n")
    svg.text(x + 24, y + 39, title_lines, size=title_size, weight=650, line_height=1.05)
    detail_y = y + h - 25
    svg.text(x + 24, detail_y, detail, size=detail_size, fill=MUTED, line_height=1.1)
    if state:
        badge(svg, x + w - max(86, 13 * len(state)) - 9, y - 13, state, color,
              width=max(86, 13 * len(state)), fill=WHITE)


def arrow(svg: SVG, points: Sequence[tuple[float, float]], color: str, marker: str, *, sw=3,
          dash: str | None = None, opacity=1.0) -> None:
    d = f"M {points[0][0]} {points[0][1]} " + " ".join(f"L {x} {y}" for x, y in points[1:])
    svg.path(d, stroke=color, sw=sw, dash=dash, marker=marker, opacity=opacity)


def section_tag(svg: SVG, x, y, label: str, title: str) -> None:
    if label:
        svg.circle(x + 15, y + 15, 15, fill=INK)
        svg.text(x + 15, y + 21, label, size=17, fill=WHITE, weight=700, anchor="middle")
        title_x = x + 42
    else:
        title_x = x
    svg.text(title_x, y + 22, title, size=27, weight=650)


def feature_tensor(svg: SVG, x, y, color=BLUE) -> None:
    for index, opacity in enumerate((0.20, 0.34, 0.58)):
        svg.rect(x + index * 10, y - index * 7, 58, 58, fill=color, stroke=color, sw=1.5, rx=2, opacity=opacity)


def token_grid(svg: SVG, x, y, color=GREEN) -> None:
    for row in range(4):
        for col in range(4):
            svg.rect(x + col * 16, y + row * 16, 11, 11, fill=color, stroke=color, sw=0.8, rx=1,
                     opacity=0.28 + 0.10 * ((row + col) % 3))


def noise_node(svg: SVG, x, y, sigma: str, t: dict) -> None:
    svg.rect(x, y, 135, 82, fill=AMBER_LIGHT, stroke=AMBER, sw=2.2, rx=10, shadow=True)
    svg.text(x + 67.5, y + 29, t["noise"], size=15, fill="#8C5C00", weight=650, anchor="middle")
    svg.text(x + 67.5, y + 61, sigma, size=17, weight=650, anchor="middle")


def image_to_data_uri(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def qualitative_tiles(path: Path) -> dict[str, list[str]]:
    image = Image.open(path).convert("RGB")
    scale = max(1, round(image.height / 212))
    padding = 5 * scale
    tile = 64 * scale
    label_width = 120 * scale
    output: dict[str, list[str]] = {"original": [], "residual": [], "gan": []}
    for row_index, row_name in enumerate(("original", "residual", "gan")):
        y = padding + row_index * (tile + padding)
        for column in range(8):
            x = label_width + column * (tile + padding)
            crop = image.crop((x, y, x + tile, y + tile))
            output[row_name].append(image_to_data_uri(crop))
    return output


def draw_face_stack(svg: SVG, x: float, y: float, tiles: dict[str, list[str]], t: dict) -> None:
    svg.image(x + 30, y + 22, 102, 102, tiles["original"][2], rx=7, stroke=LINE)
    svg.image(x + 14, y + 10, 102, 102, tiles["original"][1], rx=7, stroke=LINE)
    svg.image(x, y, 102, 102, tiles["original"][0], rx=7, stroke=GRAY, sw=2)
    svg.text(x + 61, y + 145, t["input_faces"], size=17, fill=MUTED, weight=600, anchor="middle")
    svg.text(x + 61, y + 168, "64 x 64 RGB", size=15, fill=MUTED, anchor="middle")


def draw_slot_points(svg: SVG, x: float, y: float, scale: float = 1.0) -> None:
    colors = [BLUE, GREEN, AMBER, PURPLE, ATTACK, "#4FA9D8", "#C66A9A", GRAY]
    for index, color in enumerate(colors):
        row, col = divmod(index, 4)
        svg.circle(x + col * 27 * scale, y + row * 27 * scale, 8 * scale, fill=color, stroke=WHITE, sw=1)


def draw_architecture(tiles: dict[str, list[str]], lang: str) -> SVG:
    t = TEXT[lang]
    svg = SVG(1800, 1060, t["architecture_title"])

    section_tag(svg, 28, 20, "a", t["architecture_title"])
    svg.rect(24, 72, 1772, 494, fill=WHITE, stroke=LINE, sw=1.6, rx=5)
    svg.rect(24, 72, 842, 494, fill="#F5F9FC", stroke="none")
    svg.rect(866, 72, 930, 494, fill="#FCFAF7", stroke="none")
    svg.line(866, 72, 866, 566, stroke=SUBTLE, sw=2.2, dash="8 6")
    svg.text(48, 106, t["trusted_client"], size=16, fill=BLUE, weight=700, letter_spacing=0.7)
    svg.text(892, 106, t["untrusted_cloud"], size=16, fill=ATTACK, weight=700, letter_spacing=0.7)
    badge(svg, 756, 126, t["transmission"], ATTACK, width=220)

    draw_face_stack(svg, 52, 234, tiles, t)
    svg.circle(212, 320, 6, fill=INK)
    branch_y = ((188, BLUE), (390, GREEN))
    for y, color in branch_y:
        arrow(svg, [(212, 320), (242, 320), (242, y + 52), (288, y + 52)], color,
              "blue" if color == BLUE else "green", sw=3.2)

    svg.text(288, 150, t["spatial_path"], size=16, fill=BLUE, weight=700)
    svg.rect(288, 174, 268, 126, fill=BLUE_LIGHT, stroke=BLUE, sw=2.2, rx=5)
    svg.rect(288, 174, 9, 126, fill=BLUE)
    svg.text(316, 212, t["spatial_encoder"], size=20, weight=700)
    svg.text(316, 244, t["spatial_encoder_detail"], size=14, fill=MUTED)
    badge(svg, 440, 265, t["frozen"], BLUE, width=101)
    for i, size in enumerate((66, 54, 42)):
        svg.rect(604 + i * 18, 202 + i * 6, size, size, fill=BLUE, stroke=WHITE, sw=1.2,
                 opacity=0.62 - i * 0.12)
    svg.text(646, 288, "Z_s  16 × 16 × 16", size=14, fill=BLUE, weight=650, anchor="middle")
    noise_node(svg, 720, 195, "σ_s = 0.31", t)
    arrow(svg, [(556, 237), (604, 237)], BLUE, "blue")
    arrow(svg, [(682, 237), (720, 237)], BLUE, "blue")

    svg.text(288, 352, t["semantic_path"], size=16, fill=GREEN, weight=700)
    svg.rect(288, 376, 268, 126, fill=GREEN_LIGHT, stroke=GREEN, sw=2.2, rx=5)
    svg.rect(288, 376, 9, 126, fill=GREEN)
    svg.text(316, 414, t["semantic_encoder"], size=20, weight=700)
    svg.text(316, 438, t["semantic_encoder_detail"], size=13, fill=MUTED, line_height=1.05)
    badge(svg, 386, 467, t["trained_s12"], GREEN, width=155)
    token_grid(svg, 606, 406, GREEN)
    svg.text(638, 482, "z_t  256-D", size=14, fill=GREEN, weight=650, anchor="middle")
    noise_node(svg, 720, 397, "σ_t = 0.10", t)
    arrow(svg, [(556, 439), (604, 439)], GREEN, "green")
    arrow(svg, [(670, 439), (720, 439)], GREEN, "green")

    spatial_server_x, semantic_server_x = 928, 928
    card(svg, spatial_server_x, 174, 248, 126, t["spatial_server"], t["spatial_server_detail"],
         BLUE, BLUE_LIGHT, state=t["adapted_s2"], title_size=19, detail_size=14)
    card(svg, semantic_server_x, 376, 248, 126, t["token_classifier"], t["token_classifier_detail"],
         GREEN, GREEN_LIGHT, state=t["trained_s12"], title_size=19, detail_size=14)
    arrow(svg, [(855, 237), (928, 237)], BLUE, "blue", sw=3.4)
    arrow(svg, [(855, 439), (928, 439)], GREEN, "green", sw=3.4)

    for y, color, label in ((237, BLUE, "ℓ_s / τ_s"), (439, GREEN, "ℓ_t / τ_t")):
        arrow(svg, [(1176, y), (1254, y), (1254, 294 if y < 300 else 370), (1304, 294 if y < 300 else 370)],
              color, "blue" if color == BLUE else "green", sw=3.2)
        svg.text(1208, y - 13, label, size=16, fill=color, weight=650)
    svg.polygon([(1304, 254), (1518, 254), (1556, 332), (1518, 410), (1304, 410), (1342, 332)],
                fill=PURPLE_LIGHT, stroke=PURPLE, sw=2.3)
    svg.text(1428, 307, t["fusion"], size=21, fill=PURPLE, weight=700, anchor="middle")
    fusion_eq = "ℓ = (1−α) ℓ_s/τ_s\n+ α ℓ_t/τ_t"
    svg.text(1428, 337, fusion_eq, size=14, weight=650, anchor="middle", line_height=1.05)
    svg.text(1428, 382, "alpha and temperatures learned" if lang == "en" else "学习 alpha 与温度参数",
             size=14, fill=MUTED, anchor="middle")
    svg.rect(1620, 276, 132, 112, fill=WHITE, stroke=INK, sw=2, rx=4)
    svg.text(1686, 320, t["identity"], size=21, weight=700, anchor="middle")
    svg.text(1686, 354, t["classes"], size=15, fill=MUTED, anchor="middle")
    arrow(svg, [(1556, 332), (1620, 332)], PURPLE, "purple", sw=4)
    svg.rect(624, 526, 484, 26, fill=WHITE, stroke=LINE, sw=1.1, rx=3)
    svg.text(866, 545, t["payload"], size=13, fill=MUTED, weight=600, anchor="middle")

    # The lower strip explains the training-only Slot-CEM mechanism. Attack
    # evaluation is intentionally kept out of the main architecture figure.
    svg.rect(24, 594, 1772, 418, fill="#F8FBFD", stroke=LINE, sw=1.6, rx=5)
    section_tag(svg, 44, 614, "b", t["privacy_zoom"])
    step_centers = [166, 520, 874, 1228, 1582]
    step_titles = [t["class_pool"], t["fixed_projection"], t["competitive_slots"],
                   t["full_geometry"], t["cem_objective"]]
    step_colors = [BLUE, BLUE, PURPLE, AMBER, GOOD]
    for idx, (x, title, color) in enumerate(zip(step_centers, step_titles, step_colors), 1):
        svg.circle(x, 704, 22, fill=color)
        svg.text(x, 712, str(idx), size=18, fill=WHITE, weight=700, anchor="middle")
        svg.text(x, 758, title, size=16, weight=650, anchor="middle", line_height=1.06)
        if idx < 5:
            arrow(svg, [(x + 28, 704), (step_centers[idx] - 28, 704)], color,
                  ["blue", "blue", "purple", "amber"][idx - 1], sw=2.5)
    for i in range(24):
        svg.circle(98 + (i % 8) * 18, 822 + (i // 8) * 20, 5, fill=BLUE,
                   opacity=0.3 + 0.1 * (i % 4))
    svg.text(166, 902, "B_y + sg(M_y)  |  K <= 64", size=14, fill=BLUE, weight=650, anchor="middle")
    svg.text(166, 932, "complete features from different images", size=12, fill=MUTED, anchor="middle")

    svg.rect(448, 816, 144, 76, fill=WHITE, stroke=BLUE, sw=1.8, rx=3)
    svg.text(520, 850, "W_p", size=23, fill=BLUE, weight=700, anchor="middle")
    svg.text(520, 877, "4096 to 64", size=13, fill=MUTED, anchor="middle")
    badge(svg, 440, 920, "PRETRAINING ONLY" if lang == "en" else "仅用于预训练", BLUE, width=160)

    draw_attention_matrix(svg, 790, 806, rows=8, cols=13)
    draw_slot_points(svg, 828, 922, 0.9)
    svg.text(874, 792, "S=8  ·  T=3", size=14, fill=PURPLE, weight=650, anchor="middle")

    for cx, cy, color in ((1168, 854, BLUE), (1232, 822, PURPLE), (1294, 884, AMBER)):
        svg.ellipse(cx, cy, 28, 17, fill=color, stroke=color, sw=1, opacity=.13)
        svg.circle(cx, cy, 6, fill=color, stroke=WHITE, sw=1)
    svg.text(1228, 924, "V_y,j = weighted distance", size=14, fill=AMBER, weight=650, anchor="middle")
    svg.text(1228, 952, "to detached soft centroids", size=12, fill=MUTED, anchor="middle")

    svg.rect(1466, 810, 232, 112, fill=GOOD_LIGHT, stroke=GOOD, sw=1.9, rx=4)
    svg.text(1582, 846, "L_slot", size=22, fill=GOOD, weight=700, anchor="middle")
    svg.text(1582, 876, "mean ReLU[", size=12, anchor="middle")
    svg.text(1582, 897, "log(V+gamma) - log(vartheta)]", size=11, anchor="middle")
    arrow(svg, [(1582, 922), (1582, 970), (520, 970), (520, 894)], GOOD, "good", sw=2.4, dash="7 5")
    svg.text(1110, 992, ("privacy gradient updates the spatial encoder; memory and centroids are stop-gradient" if lang == "en" else
                         "隐私梯度更新空间编码器；记忆特征和质心均停止梯度"),
             size=13, fill=GOOD, weight=650, anchor="middle")
    return svg


def draw_attention_matrix(svg: SVG, x: float, y: float, rows: int = 8, cols: int = 10) -> None:
    for row in range(rows):
        for col in range(cols):
            distance = abs((col % rows) - row)
            opacity = max(0.08, 0.78 - 0.16 * distance)
            color = [BLUE, GREEN, AMBER, PURPLE, ATTACK, "#4FA9D8", "#C66A9A", GRAY][row]
            svg.rect(x + col * 13, y + row * 13, 10, 10, fill=color, stroke="none", opacity=opacity, rx=1)


def draw_slot_mechanism(lang: str) -> SVG:
    t = TEXT[lang]
    svg = SVG(1800, 790, t["slot_title"])
    section_tag(svg, 28, 20, "", t["slot_title"])
    svg.line(28, 72, 1772, 72, stroke=INK, sw=2)

    # A continuous scientific pipeline replaces the previous collection of cards.
    stage_x = [46, 382, 730, 1088, 1452]
    headings = [t["evidence"], t["fixed_projection"], t["assignment"], t["geometry"], t["update"]]
    colors = [BLUE, BLUE, PURPLE, AMBER, GOOD]
    for i, (x, title, color) in enumerate(zip(stage_x, headings, colors), 1):
        svg.circle(x + 17, 112, 17, fill=color)
        svg.text(x + 17, 119, str(i), size=16, fill=WHITE, weight=700, anchor="middle")
        svg.text(x + 46, 120, title, size=21, fill=color, weight=700, line_height=1.05)
    for x in (350, 698, 1056, 1420):
        svg.line(x, 92, x, 608, stroke=LINE, sw=1.5)

    # Same-class evidence: live samples carry gradients; memory entries are detached.
    svg.text(54, 178, t["current_batch"], size=16, fill=INK, weight=650)
    batch = [(84, 232), (134, 207), (185, 243), (238, 214), (292, 235)]
    for i, (x, y) in enumerate(batch):
        svg.circle(x, y, 10, fill=BLUE, stroke=WHITE, sw=1.5, opacity=.75 + .05 * (i % 3))
        svg.line(x, y + 12, x, 280, stroke=BLUE, sw=1, opacity=.3)
    svg.text(54, 330, t["memory_bank"], size=16, fill=INK, weight=650)
    for row in range(4):
        for col in range(8):
            svg.rect(60 + col * 32 + row * 3, 361 + row * 30, 21, 17, fill=BLUE,
                     stroke=WHITE, sw=1, rx=1, opacity=.17 + .07 * row)
    svg.path("M 310 203 L 330 203 L 330 464 L 310 464", stroke=BLUE, sw=2.5)
    svg.text(170, 529, "B_y + M_y", size=24, fill=BLUE, weight=700, anchor="middle")
    svg.text(170, 558, "K <= 64", size=15, fill=MUTED, anchor="middle")
    svg.text(170, 590, t["sparse_batch"], size=13, fill=MUTED, anchor="middle", line_height=1.05)

    # Projection is learned during Slot-CEM pretraining; it is not the transmitted tensor.
    for row in range(9):
        for col in range(14):
            opacity = .08 + .38 * abs(math.sin((row + 1) * (col + 2)))
            svg.rect(410 + col * 17, 187 + row * 17, 13, 13, fill=BLUE, opacity=opacity, rx=1)
    svg.text(529, 372, "W_p in R^(64 x 4096)", size=20, fill=BLUE, weight=650, anchor="middle")
    arrow(svg, [(529, 393), (529, 451)], BLUE, "blue", sw=3)
    svg.rect(435, 451, 188, 82, fill=BLUE_LIGHT, stroke=BLUE, sw=2, rx=3)
    svg.text(529, 485, "h_i = W_p z_i", size=20, weight=650, anchor="middle")
    svg.text(529, 513, t["assignment_space"], size=14, fill=MUTED, anchor="middle", line_height=1.05)
    badge(svg, 449, 565, "PRETRAINING ONLY" if lang == "en" else "仅用于预训练", BLUE, width=160)

    # Slot Attention: assignment matrix and iterative slots.
    draw_attention_matrix(svg, 758, 188, rows=8, cols=12)
    svg.text(837, 322, "A_y in R^(N_y x 8)", size=18, fill=PURPLE, weight=650, anchor="middle")
    draw_slot_points(svg, 790, 388, 1.15)
    svg.ellipse(838, 420, 118, 82, stroke=PURPLE, sw=2.2, dash="7 5")
    svg.path("M 914 380 C 1018 402, 1018 520, 906 545", stroke=PURPLE, sw=2.3,
             dash="7 5", marker="purple")
    svg.text(870, 561, t["soft_assignment"], size=15, fill=MUTED, anchor="middle")
    svg.text(870, 596, "S = 8     T = 3", size=18, fill=PURPLE, weight=700, anchor="middle")

    # Geometric statistic in the learned slot space.
    centers = [(1165, 244, BLUE, -16), (1270, 205, PURPLE, 17), (1325, 338, AMBER, -6)]
    offsets = [(-31, -7), (-20, 19), (3, -22), (23, 9), (31, -13), (7, 23), (-36, 14)]
    for cx, cy, color, rotation in centers:
        svg.ellipse(cx, cy, 67, 40, fill=color, stroke=color, sw=1.5, opacity=.12, rotate=rotation)
        for dx, dy in offsets:
            svg.circle(cx + dx, cy + dy, 6, fill=color, opacity=.38)
        svg.circle(cx, cy, 11, fill=color, stroke=WHITE, sw=2)
        svg.line(cx - 8, cy - 8, cx + 8, cy + 8, stroke=WHITE, sw=2)
        svg.line(cx - 8, cy + 8, cx + 8, cy - 8, stroke=WHITE, sw=2)
    svg.circle(1235, 355, 8, fill=INK, stroke=WHITE, sw=1.5)
    svg.path("M 1235 355 L 1270 205", stroke=INK, sw=2, dash="6 5")
    svg.text(1260, 423, t["choose_centroid"], size=15, fill=MUTED, anchor="middle")
    svg.rect(1116, 462, 276, 94, fill=WHITE, stroke=AMBER, sw=1.8, rx=3)
    svg.text(1254, 496, "d_ij = ||h_i - c_yj||^2 / 64", size=18, weight=650, anchor="middle")
    svg.text(1254, 528, t["full_geometry_detail"], size=13, fill=MUTED, anchor="middle", line_height=1.05)
    badge(svg, 1179, 575, t["detached"], AMBER, width=150)

    # Thresholded geometric loss; centroids and memory provide context but receive no gradient.
    svg.rect(1472, 180, 274, 105, fill=AMBER_LIGHT, stroke=AMBER, sw=2, rx=3)
    svg.text(1609, 213, "V_y = (1/S) sum_j", size=17, weight=650, anchor="middle")
    svg.text(1609, 243, "sum_i a_ij d_ij / sum_i a_ij", size=16, weight=650, anchor="middle")
    svg.text(1609, 262, t["full_geometry_detail"], size=11, fill=MUTED, anchor="middle", line_height=1.0)
    arrow(svg, [(1609, 285), (1609, 340)], GOOD, "good", sw=3)
    svg.rect(1472, 340, 274, 132, fill=GOOD_LIGHT, stroke=GOOD, sw=2.2, rx=3)
    svg.text(1609, 379, t["cem_objective"], size=21, fill=GOOD, weight=700, anchor="middle")
    svg.text(1609, 416, "L_slot = mean ReLU", size=18, weight=650, anchor="middle")
    svg.text(1609, 447, "[log(V_y+gamma) - log tau]", size=17, weight=650, anchor="middle")
    svg.text(1609, 527, t["no_shortcut"], size=13, fill=MUTED, anchor="middle", line_height=1.05)

    # Explicit gradient-routing band.
    svg.rect(28, 642, 1744, 106, fill="#F6FAF7", stroke=GOOD, sw=1.5, rx=4)
    svg.text(64, 678, "BACKWARD" if lang == "en" else "反向传播", size=14, fill=GOOD, weight=700)
    arrow(svg, [(1609, 618), (1609, 697), (529, 697), (529, 610)], GOOD, "good", sw=3, dash="8 6")
    svg.text(1068, 688, t["privacy_gradient"], size=15, fill=GOOD, weight=650, anchor="middle")
    svg.text(1068, 727, ("memory features and centroids: stop-gradient context" if lang == "en" else
                         "记忆特征和质心仅提供停止梯度的上下文"),
             size=14, fill=MUTED, anchor="middle")

    for x in (350, 698, 1056, 1420):
        arrow(svg, [(x - 8, 322), (x + 24, 322)], SUBTLE, "gray", sw=2.2)
    return svg


def status_cell(svg: SVG, x: float, y: float, w: float, state: str, color: str, frozen: bool) -> None:
    fill = GRAY_LIGHT if frozen else (BLUE_LIGHT if color == BLUE else GREEN_LIGHT)
    edge = GRAY if frozen else color
    svg.rect(x, y, w, 58, fill=fill, stroke=edge, sw=1.7, rx=7)
    if frozen:
        svg.rect(x + 20, y + 24, 18, 20, fill=GRAY, rx=2)
        svg.path(f"M {x+23} {y+24} V {y+17} C {x+23} {y+5}, {x+35} {y+5}, {x+35} {y+17} V {y+24}",
                 stroke=GRAY, sw=3)
    else:
        svg.circle(x + 30, y + 29, 10, fill=color, stroke=WHITE, sw=1.5)
    svg.text(x + 54, y + 37, state, size=17, fill=edge, weight=700)


def draw_training_protocol(lang: str) -> SVG:
    t = TEXT[lang]
    svg = SVG(1800, 720, t["training_title"])
    section_tag(svg, 24, 20, "", t["training_title"])

    stages = [
        (40, 92, 260, t["stage0"], GRAY, GRAY_LIGHT),
        (330, 92, 410, t["stage1"], GREEN, GREEN_LIGHT),
        (770, 92, 330, t["stage2"], BLUE, BLUE_LIGHT),
        (1160, 92, 600, t["formal_eval"], ATTACK, ATTACK_LIGHT),
    ]
    for x, y, w, title, color, fill in stages:
        svg.rect(x, y, w, 64, fill=fill, stroke=color, sw=2, rx=8)
        svg.text(x + w / 2, y + 40, title, size=20, fill=color, weight=700, anchor="middle")
    for (x, _, w, *_), (nx, *_) in zip(stages, stages[1:]):
        arrow(svg, [(x + w, 124), (nx, 124)], GRAY, "gray", sw=2.5)

    panel(svg, 40, 182, 1060, 474, fill="#FAFBFC", stroke=LINE, rx=12)
    svg.text(70, 221, t["module"], size=17, fill=MUTED, weight=700)
    svg.text(500, 221, t["stage1"], size=17, fill=GREEN, weight=700, anchor="middle")
    svg.text(877, 221, t["stage2"], size=17, fill=BLUE, weight=700, anchor="middle")
    rows = [
        (t["spatial_client"], True, True, BLUE),
        (t["spatial_cloud"], True, False, BLUE),
        (t["semantic_branch"], False, False, GREEN),
        (t["fusion_module"], False, False, PURPLE),
    ]
    for index, (label, f1, f2, color) in enumerate(rows):
        y = 252 + index * 78
        svg.text(265, y + 38, label, size=19, weight=600, anchor="end")
        status_cell(svg, 310, y, 380, t["locked"] if f1 else t["trained"], color, f1)
        status_cell(svg, 730, y, 320, t["locked"] if f2 else t["trained"], color, f2)
    svg.rect(310, 578, 380, 48, fill=WHITE, stroke=LINE, sw=1.3, rx=7)
    svg.text(500, 609, "spatial .025  |  semantic .05  |  lambda_t .35", size=17, fill=MUTED, weight=600, anchor="middle")
    svg.rect(730, 578, 320, 48, fill=WHITE, stroke=LINE, sw=1.3, rx=7)
    svg.text(890, 609, "spatial .22  |  semantic .10  |  lambda_t .50", size=17, fill=MUTED, weight=600, anchor="middle")

    panel(svg, 1160, 182, 600, 474, fill="#FFFCFA", stroke="#E8C6B2", rx=12)
    svg.rect(1190, 215, 220, 82, fill=AMBER_LIGHT, stroke=AMBER, sw=2, rx=8)
    svg.text(1300, 250, "DEPLOYMENT" if lang == "en" else "部署设置", size=17, fill="#8C5C00", weight=700, anchor="middle")
    svg.text(1300, 278, "spatial .31  |  semantic .10", size=18, weight=600, anchor="middle")
    arrow(svg, [(1410, 256), (1470, 256)], PURPLE, "purple")
    svg.text(1595, 224, t["formal_targets"], size=17, weight=650, anchor="middle")
    for index in range(5):
        svg.circle(1495 + index * 50, 260, 14, fill=PURPLE, stroke=WHITE, sw=2)
        svg.text(1495 + index * 50, 291, str(126 + index), size=13, fill=MUTED, anchor="middle")
    svg.text(1595, 318, t["target_seeds"], size=14, fill=MUTED, anchor="middle")

    svg.rect(1190, 347, 540, 150, fill=WHITE, stroke=ATTACK, sw=2, rx=9)
    svg.text(1460, 378, t["attacker_matrix"], size=17, fill=ATTACK, weight=650, anchor="middle")
    attacks = [t["residual"], t["gan"], t["adaptive"]]
    for col, label in enumerate(attacks):
        x = 1352 + col * 116
        svg.text(x, 414, label, size=15, weight=650, anchor="middle")
    knowledge = [(t["training_knowledge"], 446, GRAY), (t["inference_knowledge"], 478, ATTACK)]
    for label, y, color in knowledge:
        svg.text(1210, y + 5, label, size=13, fill=MUTED)
        for col in range(3):
            x = 1352 + col * 116
            seed_count = 1 if col == 2 else 3
            offsets = [0] if seed_count == 1 else [-14, 0, 14]
            for seed, offset in enumerate(offsets):
                svg.circle(x + offset, y, 5, fill=color, opacity=0.62 + seed * 0.16)
    arrow(svg, [(1460, 497), (1460, 535)], ATTACK, "attack")
    svg.rect(1250, 535, 420, 70, fill=GRAY_LIGHT, stroke=GRAY, sw=1.7, rx=8)
    svg.text(1460, 564, t["aggregate"], size=17, weight=650, anchor="middle")
    svg.text(1460, 589, t["bootstrap"], size=15, fill=MUTED, anchor="middle")
    svg.text(1460, 638, t["no_recon_grad"], size=16, fill=ATTACK, weight=700, anchor="middle")
    return svg


def map_value(value: float, lo: float, hi: float, start: float, end: float) -> float:
    return start + (value - lo) / (hi - lo) * (end - start)


def draw_results(summary: dict, lang: str) -> SVG:
    t = TEXT[lang]
    svg = SVG(1800, 1060, "DualPath-CEM results")
    svg.rect(24, 24, 850, 642, fill=WHITE, stroke=LINE, sw=1.6, rx=4)
    svg.rect(902, 24, 874, 642, fill=WHITE, stroke=LINE, sw=1.6, rx=4)
    svg.rect(24, 694, 1752, 338, fill=WHITE, stroke=LINE, sw=1.6, rx=4)
    svg.text(52, 70, t["results_a"], size=25, weight=700)
    svg.text(930, 70, t["results_b"], size=25, weight=700)
    svg.text(52, 740, t["results_c"], size=25, weight=700)

    plot_x0, plot_x1, plot_y0, plot_y1 = 120, 830, 565, 118
    x_lo, x_hi = 80.0, 82.35
    y_lo, y_hi = 0.020, 0.0382
    published = summary["published"]
    px = map_value(published["accuracy"] * 100, x_lo, x_hi, plot_x0, plot_x1)
    py = map_value(published["decoder_inference_mse"], y_lo, y_hi, plot_y0, plot_y1)
    svg.rect(px, plot_y1, plot_x1 - px, py - plot_y1, fill=GOOD_LIGHT, stroke="none", opacity=0.75)
    svg.text(plot_x1 - 12, plot_y1 + 25, "PARETO DOMINANCE" if lang == "en" else "PARETO 优势区",
             size=12, fill=GOOD, weight=700, anchor="end")
    for value in (80.0, 80.5, 81.0, 81.5, 82.0):
        x = map_value(value, x_lo, x_hi, plot_x0, plot_x1)
        svg.line(x, plot_y0, x, plot_y1, stroke=GRID, sw=1.5)
        svg.text(x, plot_y0 + 28, f"{value:.1f}", size=16, fill=MUTED, anchor="middle")
    for value in (0.020, 0.025, 0.030, 0.035):
        y = map_value(value, y_lo, y_hi, plot_y0, plot_y1)
        svg.line(plot_x0, y, plot_x1, y, stroke=GRID, sw=1.5)
        svg.text(plot_x0 - 12, y + 6, f"{value:.3f}", size=16, fill=MUTED, anchor="end")
    svg.line(plot_x0, plot_y0, plot_x1, plot_y0, stroke=SUBTLE, sw=2)
    svg.line(plot_x0, plot_y0, plot_x0, plot_y1, stroke=SUBTLE, sw=2)
    svg.line(px, plot_y0, px, plot_y1, stroke=GRAY, sw=2, dash="7 6")
    svg.line(plot_x0, py, plot_x1, py, stroke=GRAY, sw=2, dash="7 6")
    svg.rect(px - 11, py - 11, 22, 22, fill=GRAY, stroke=WHITE, sw=2)
    svg.text(px + 16, py - 16, t["published_cem"], size=15, fill=GRAY, weight=650)
    accuracies = [row["accuracy"] * 100 for row in summary["utility_records"]]
    mses = summary["target_attack_means"]["decoder_inference_mse"]
    seeds = [row["target_seed"] for row in summary["utility_records"]]
    for accuracy, mse, seed in zip(accuracies, mses, seeds):
        x = map_value(accuracy, x_lo, x_hi, plot_x0, plot_x1)
        y = map_value(mse, y_lo, y_hi, plot_y0, plot_y1)
        svg.circle(x, y, 10, fill=BLUE, stroke=WHITE, sw=2)
        seed_offsets = {126: (-10, -18), 127: (0, -34), 128: (10, 28), 129: (0, -26), 130: (0, -42)}
        dx, dy = seed_offsets.get(seed, (0, -18))
        svg.text(x + dx, y + dy, str(seed), size=13, fill=MUTED, weight=600, anchor="middle")
    interval_x = summary["intervals"]["accuracy"]
    interval_y = summary["intervals"]["decoder_inference_mse"]
    mx = map_value(interval_x["mean"] * 100, x_lo, x_hi, plot_x0, plot_x1)
    my = map_value(interval_y["mean"], y_lo, y_hi, plot_y0, plot_y1)
    xlow = map_value(interval_x["lower"] * 100, x_lo, x_hi, plot_x0, plot_x1)
    xhigh = map_value(interval_x["upper"] * 100, x_lo, x_hi, plot_x0, plot_x1)
    ylow = map_value(interval_y["lower"], y_lo, y_hi, plot_y0, plot_y1)
    yhigh = map_value(interval_y["upper"], y_lo, y_hi, plot_y0, plot_y1)
    svg.line(xlow, my, xhigh, my, stroke=ATTACK, sw=4)
    svg.line(mx, ylow, mx, yhigh, stroke=ATTACK, sw=4)
    svg.polygon([(mx, my - 15), (mx + 15, my), (mx, my + 15), (mx - 15, my)], fill=ATTACK, stroke=WHITE, sw=2)
    arrow(svg, [(566, 283), (674, 204)], GOOD, "good", sw=3)
    svg.text(555, 306, t["higher_better"], size=15, fill=GOOD, weight=650)
    svg.text((plot_x0 + plot_x1) / 2, 625, t["accuracy"], size=18, weight=650, anchor="middle")
    svg.text(48, 345, t["recon_mse"], size=18, weight=650, anchor="middle", rotate=-90)

    rows = [
        (t["accuracy"], published["accuracy"] * 100, summary["observed"]["accuracy"] * 100, GREEN, "%"),
        (t["decoder_train"], published["decoder_training_mse"], summary["observed"]["decoder_training_mse"], BLUE, ""),
        (t["decoder_infer"], published["decoder_inference_mse"], summary["observed"]["decoder_inference_mse"], BLUE, ""),
        (t["gan_train"], published["gan_training_mse"], summary["observed"]["gan_training_mse"], PURPLE, ""),
        (t["gan_infer"], published["gan_inference_mse"], summary["observed"]["gan_inference_mse"], PURPLE, ""),
    ]
    svg.text(1275, 112, t["published_cem"], size=14, fill=GRAY, weight=700, anchor="middle")
    svg.text(1633, 112, "DualPath-CEM", size=14, fill=ATTACK, weight=700, anchor="middle")
    for index, (label, baseline, ours, color, suffix) in enumerate(rows):
        y = 163 + index * 91
        svg.text(1148, y + 6, label, size=16, fill=INK, weight=650, anchor="end")
        # Endpoints are shown on a common visual baseline; exact values remain printed.
        left, right = 1275, 1633
        svg.line(left, y, right, y, stroke=LINE, sw=8, opacity=.75)
        svg.circle(left, y, 10, fill=GRAY, stroke=WHITE, sw=2)
        svg.circle(right, y, 12, fill=color, stroke=WHITE, sw=2)
        baseline_text = f"{baseline:.2f}{suffix}" if suffix else f"{baseline:.4f}"
        ours_text = f"{ours:.2f}{suffix}" if suffix else f"{ours:.4f}"
        gain = 100 * (ours / baseline - 1)
        svg.text(left, y + 34, baseline_text, size=14, fill=GRAY, weight=650, anchor="middle")
        svg.text(right, y + 34, ours_text, size=14, fill=color, weight=700, anchor="middle")
        svg.text((left + right) / 2, y - 14, f"+{gain:.1f}%", size=14, fill=color, weight=700,
                 anchor="middle")
        arrow(svg, [(left + 18, y), (right - 18, y)], color,
              "green" if color == GREEN else ("blue" if color == BLUE else "purple"), sw=2.2)
    svg.text(1454, 628, ("exact means; arrows point towards the defender-favourable endpoint" if lang == "en" else
                         "均为精确均值；箭头指向对防御者有利的一端"),
             size=13, fill=MUTED, anchor="middle")

    components = [
        (t["spatial_only"], mean(r["accuracy"] for r in summary["component_records"] if r["component"] == "legacy") * 100, BLUE),
        (t["semantic_only"], mean(r["accuracy"] for r in summary["component_records"] if r["component"] == "semantic") * 100, GREEN),
        (t["stage1_fusion"], mean(r["accuracy"] for r in summary["stage1_records"]) * 100, PURPLE),
        (t["stage2_fusion"], summary["observed"]["accuracy"] * 100, ATTACK),
    ]
    bar_x0, bar_x1 = 260, 1695
    axis_y = 984
    min_u, max_u = 55.0, 84.0
    for value in (60, 65, 70, 75, 80):
        x = map_value(value, min_u, max_u, bar_x0, bar_x1)
        svg.line(x, 776, x, axis_y, stroke=GRID, sw=1.2)
        svg.text(x, axis_y + 27, str(value), size=14, fill=MUTED, anchor="middle")
    baseline_x = map_value(published["accuracy"] * 100, min_u, max_u, bar_x0, bar_x1)
    svg.line(baseline_x, 768, baseline_x, axis_y, stroke=GRAY, sw=2, dash="7 5")
    svg.text(baseline_x, 764, f"{t['published_cem']} 80.33", size=13, fill=GRAY, weight=650, anchor="middle")
    for index, (label, value, color) in enumerate(components):
        y = 806 + index * 46
        x = map_value(value, min_u, max_u, bar_x0, bar_x1)
        svg.text(238, y + 6, label, size=15, weight=650, anchor="end")
        svg.rect(bar_x0, y - 12, x - bar_x0, 24, fill=color, stroke="none", rx=2, opacity=.82)
        svg.circle(x, y, 8, fill=color, stroke=WHITE, sw=1.5)
        svg.text(x + 17, y + 6, f"{value:.2f}", size=15, fill=color, weight=700)
    svg.text((bar_x0 + bar_x1) / 2, 1024, t["accuracy"], size=16, weight=650, anchor="middle")
    return svg


def grouped_attack_means(summary: dict) -> dict[tuple[str, str], dict[str, float]]:
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in summary["attack_records"]:
        groups.setdefault((row["attack"], row["knowledge"]), []).append(row)
    return {
        key: {metric: mean(float(row[metric]) for row in rows)
              for metric in ("mse", "lpips", "ssim", "identity_cosine_similarity")}
        for key, rows in groups.items()
    }


def draw_attack_robustness(summary: dict, lang: str) -> SVG:
    t = TEXT[lang]
    svg = SVG(1800, 710, "Attack robustness")
    svg.rect(24, 24, 822, 662, fill=WHITE, stroke=LINE, sw=1.6, rx=4)
    svg.rect(874, 24, 902, 662, fill=WHITE, stroke=LINE, sw=1.6, rx=4)
    svg.text(54, 70, t["attack_title_a"], size=25, weight=700)
    svg.text(904, 70, t["attack_title_b"], size=25, weight=700)
    groups = grouped_attack_means(summary)

    attacks = [("decoder", t["residual"]), ("gan", t["gan"]), ("adaptive", t["adaptive"])]
    x_lo, x_hi = 0.024, 0.042
    x0, x1 = 220, 790
    for value in (0.025, 0.030, 0.035, 0.040):
        x = map_value(value, x_lo, x_hi, x0, x1)
        svg.line(x, 125, x, 597, stroke=GRID, sw=1.4)
        svg.text(x, 626, f"{value:.3f}", size=15, fill=MUTED, anchor="middle")
    for index, (key, label) in enumerate(attacks):
        y = 190 + index * 145
        train_value = groups[(key, "training")]["mse"]
        infer_value = groups[(key, "inference")]["mse"]
        xt = map_value(train_value, x_lo, x_hi, x0, x1)
        xi = map_value(infer_value, x_lo, x_hi, x0, x1)
        svg.text(192, y + 6, label, size=17, weight=650, anchor="end")
        svg.line(xt, y, xi, y, stroke=LINE, sw=8)
        svg.circle(xt, y, 12, fill=GRAY, stroke=WHITE, sw=2)
        svg.circle(xi, y, 14, fill=ATTACK, stroke=WHITE, sw=2)
        arrow(svg, [(xt + 15, y), (xi - 16, y)], ATTACK, "attack", sw=2.2)
        svg.text(xt, y + 34, f"{train_value:.4f}", size=14, fill=GRAY, weight=650, anchor="middle")
        svg.text(xi, y - 18, f"{infer_value:.4f}", size=14, fill=ATTACK, weight=700, anchor="middle")
    svg.circle(225, 103, 8, fill=GRAY)
    svg.text(243, 109, t["train"], size=14, fill=MUTED)
    svg.circle(408, 103, 8, fill=ATTACK)
    svg.text(426, 109, t["infer"], size=14, fill=MUTED)
    svg.text((x0 + x1) / 2, 666, t["recon_mse"] + "  ↑", size=17, weight=650, anchor="middle")

    metrics = [
        ("mse", t["mse"], "↑"),
        ("lpips", t["lpips"], "↑"),
        ("ssim", t["ssim"], "↓"),
        ("identity_cosine_similarity", t["id_cos"], "↓"),
    ]
    heat_x0, col_w = 1085, 158
    row_values = [[groups[(key, "inference")][metric] for metric, *_ in metrics] for key, _ in attacks]
    ranges = []
    for col, (metric, _, direction) in enumerate(metrics):
        values = [row[col] for row in row_values]
        lo, hi = min(values), max(values)
        pad = max((hi - lo) * .18, 1e-6)
        ranges.append((lo - pad, hi + pad, direction))
        x = heat_x0 + col * col_w
        svg.text(x + col_w / 2, 118, metric.upper() if metric == "mse" else metrics[col][1],
                 size=17, fill=MUTED, weight=700, anchor="middle")
        svg.text(x + col_w / 2, 147, direction, size=21, fill=GOOD, weight=700, anchor="middle")
    for row_index, (key, label) in enumerate(attacks):
        y = 176 + row_index * 138
        svg.text(1055, y + 58, label, size=17, weight=650, anchor="end")
        for col, (metric, _, direction) in enumerate(metrics):
            value = groups[(key, "inference")][metric]
            lo, hi, _ = ranges[col]
            favourable = (value - lo) / (hi - lo) if direction == "↑" else (hi - value) / (hi - lo)
            favourable = min(1.0, max(0.0, favourable))
            opacity = 0.16 + 0.70 * favourable
            x = heat_x0 + col * col_w
            svg.rect(x + 3, y, col_w - 6, 112, fill=BLUE, stroke=WHITE, sw=3, rx=2, opacity=opacity)
            text_color = WHITE if opacity > 0.56 else INK
            svg.text(x + col_w / 2, y + 66, f"{value:.4f}", size=18, fill=text_color,
                     weight=650, anchor="middle")
    svg.rect(1032, 606, 690, 47, fill="#F6FAF7", stroke=GOOD, sw=1.2, rx=3)
    svg.text(1377, 636, t["defender_direction"], size=14, fill=MUTED, anchor="middle")
    return svg


def draw_qualitative(tiles: dict[str, list[str]], summary: dict, lang: str) -> SVG:
    t = TEXT[lang]
    svg = SVG(1800, 690, t["qual_title"])
    svg.text(24, 40, t["qual_title"], size=27, weight=650)
    gap = 12
    tile_w = 190
    x0 = 182
    y_rows = [84, 274, 464]
    for index in range(8):
        svg.text(x0 + index * (tile_w + gap) + tile_w / 2, 72, f"{index + 1}", size=15, fill=MUTED,
                 weight=650, anchor="middle")
    rows = [
        ("original", t["original"], GRAY, None),
        ("residual", t["residual_decoder"], BLUE, summary["observed"]["decoder_inference_mse"]),
        ("gan", t["gan"], ATTACK, summary["observed"]["gan_inference_mse"]),
    ]
    for row_index, (key, label, color, mse) in enumerate(rows):
        y = y_rows[row_index]
        svg.rect(24, y, 1752, 174, fill=WHITE if row_index == 0 else "#FBFCFD", stroke=LINE, sw=1.2, rx=8)
        svg.rect(24, y + 12, 7, 150, fill=color, rx=3.5)
        svg.text(50, y + 77, label, size=20, weight=700, line_height=1.08)
        if mse is not None:
            svg.text(50, y + 133, f"{t['mean_mse']} {mse:.4f}", size=13, fill=MUTED, weight=600)
        for index, uri in enumerate(tiles[key]):
            x = x0 + index * (tile_w + gap)
            svg.image(x, y + 7, tile_w, 160, uri, rx=3, stroke=WHITE, sw=2)
    svg.text(900, 676, t["examples_note"], size=15, fill=MUTED, anchor="middle")
    return svg


def export_svg(svg: SVG, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    svg_text = svg.render()
    svg_path = output_dir / f"{stem}.svg"
    svg_path.write_text(svg_text, encoding="utf-8")
    cairosvg.svg2pdf(bytestring=svg_text.encode("utf-8"), write_to=str(output_dir / f"{stem}.pdf"))
    html_text = (
        "<!doctype html><html><head><meta charset='utf-8'><style>"
        f"@page{{size:{svg.width}px {svg.height}px;margin:0}}html,body{{margin:0;padding:0;width:{svg.width}px;"
        f"height:{svg.height}px;overflow:hidden;background:white}}svg{{display:block}}</style></head><body>"
        f"{svg_text}</body></html>"
    )
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="chrome", headless=True)
        page = browser.new_page(viewport={"width": svg.width, "height": svg.height}, device_scale_factor=2)
        page.set_content(html_text, wait_until="load")
        page.screenshot(path=str(output_dir / f"{stem}.png"), full_page=True)
        browser.close()


def generate_all(summary: dict, qualitative_source: Path, output_dir: Path, lang: str) -> None:
    tiles = qualitative_tiles(qualitative_source)
    figures = {
        "architecture": draw_architecture(tiles, lang),
        "slot_cem_mechanism": draw_slot_mechanism(lang),
        "training_protocol": draw_training_protocol(lang),
        "results_overview": draw_results(summary, lang),
        "attack_robustness": draw_attack_robustness(summary, lang),
        "qualitative_grid_refined": draw_qualitative(tiles, summary, lang),
    }
    for stem, svg in figures.items():
        export_svg(svg, output_dir, stem)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--qualitative-source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lang", choices=("en", "zh"), default="en")
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    generate_all(summary, args.qualitative_source, args.output_dir, args.lang)
    print(json.dumps({"status": "PASS", "lang": args.lang, "output_dir": str(args.output_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
