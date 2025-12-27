# ml_robotarm_masked_ink_train.py
# 요구사항 통합(학습 중심):
# - 비정형 단순 다각형(꼭짓점>=6, 평행 변 1쌍 이상, 자기교차X, 비정형 강화 조건)
# - polygon -> mask -> safe_mask(margin erosion) (+ optional SDF)
# - 선은 safe_mask 밖으로 절대 기록되지 않음(clip_segment_to_mask + InkRecorder 게이트)
# - Safety Shield(행동 수정: 밖으로 나가려 하면 pen up + 내부 투영)
# - 스타일 토큰 기반 PlanGenerator(설계도/회로도 느낌), 매번 변주
# - 가상 필압(굵기/농도/덧그리기 패스)
# - Gymnasium Env(2-link arm FK/IK) + BC 학습(모방학습)
# - Preview(도형 저투명) / Export(잉크만)

from __future__ import annotations
import os, math, time, json, random, argparse
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional, Dict

import numpy as np
from PIL import Image, ImageDraw
import yaml

# Gymnasium (Env 표준)
try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception as e:
    raise RuntimeError("gymnasium이 필요합니다. `pip install gymnasium` 후 다시 실행하세요.") from e

# Torch (BC 학습)
import torch
import torch.nn as nn
import torch.optim as optim


# -------------------------
# Utils: seed, run dir, IO
# -------------------------
def now_timestamp():
    return time.strftime("%Y%m%d_%H%M%S")

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def non_deterministic_seed():
    # 런타임 변주는 비결정적으로
    return int.from_bytes(os.urandom(4), "little")

def save_yaml(path: str, obj: dict):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=True)

def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def save_json(path: str, obj: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


# -------------------------
# Geometry: segment intersect, simple polygon check
# -------------------------
def orient(a, b, c) -> float:
    # cross((b-a),(c-a))
    return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])

def on_segment(a, b, p) -> bool:
    return (min(a[0], b[0]) - 1e-9 <= p[0] <= max(a[0], b[0]) + 1e-9 and
            min(a[1], b[1]) - 1e-9 <= p[1] <= max(a[1], b[1]) + 1e-9)

def seg_intersect(a, b, c, d) -> bool:
    # Proper + collinear handling
    o1 = orient(a, b, c)
    o2 = orient(a, b, d)
    o3 = orient(c, d, a)
    o4 = orient(c, d, b)
    if (o1 == 0 and on_segment(a, b, c)) or (o2 == 0 and on_segment(a, b, d)) or \
       (o3 == 0 and on_segment(c, d, a)) or (o4 == 0 and on_segment(c, d, b)):
        return True
    return (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0)

def polygon_self_intersects(pts: List[Tuple[float,float]]) -> bool:
    n = len(pts)
    edges = []
    for i in range(n):
        a = pts[i]
        b = pts[(i+1)%n]
        edges.append((a,b))
    for i in range(n):
        for j in range(i+1, n):
            # 인접 엣지(공유 꼭짓점)는 교차 검사 제외
            if abs(i-j) <= 1 or (i==0 and j==n-1):
                continue
            a,b = edges[i]
            c,d = edges[j]
            if seg_intersect(a,b,c,d):
                return True
    return False

def polygon_area(pts: List[Tuple[float,float]]) -> float:
    x = np.array([p[0] for p in pts])
    y = np.array([p[1] for p in pts])
    return 0.5 * float(np.abs(np.dot(x, np.roll(y,-1)) - np.dot(y, np.roll(x,-1))))

def ensure_ccw(pts: List[Tuple[float,float]]) -> List[Tuple[float,float]]:
    x = np.array([p[0] for p in pts])
    y = np.array([p[1] for p in pts])
    signed = 0.5 * float(np.dot(x, np.roll(y,-1)) - np.dot(y, np.roll(x,-1)))
    if signed < 0:
        return list(reversed(pts))
    return pts

def min_edge_len(pts: List[Tuple[float,float]]) -> float:
    n = len(pts)
    m = 1e9
    for i in range(n):
        a = np.array(pts[i])
        b = np.array(pts[(i+1)%n])
        m = min(m, float(np.linalg.norm(b-a)))
    return m

def edge_len_ratio(pts: List[Tuple[float,float]]) -> float:
    n = len(pts)
    lens = []
    for i in range(n):
        a = np.array(pts[i]); b = np.array(pts[(i+1)%n])
        lens.append(float(np.linalg.norm(b-a))+1e-9)
    return max(lens)/min(lens)

def angle_dispersion_score(pts: List[Tuple[float,float]]) -> float:
    # 정다각형 느낌 억제: 각도 분산이 어느 정도 있어야 함
    n = len(pts)
    angles = []
    for i in range(n):
        p_prev = np.array(pts[(i-1)%n])
        p = np.array(pts[i])
        p_next = np.array(pts[(i+1)%n])
        v1 = p_prev - p
        v2 = p_next - p
        v1 /= (np.linalg.norm(v1)+1e-9)
        v2 /= (np.linalg.norm(v2)+1e-9)
        dot = float(np.clip(np.dot(v1, v2), -1, 1))
        ang = math.acos(dot)
        angles.append(ang)
    return float(np.std(angles))

def count_concave_vertices(pts: List[Tuple[float,float]]) -> int:
    pts = ensure_ccw(pts)
    n = len(pts)
    concave = 0
    for i in range(n):
        a = np.array(pts[(i-1)%n])
        b = np.array(pts[i])
        c = np.array(pts[(i+1)%n])
        if orient(tuple(a), tuple(b), tuple(c)) < 0:
            concave += 1
    return concave

def bbox_and_centroid(pts: List[Tuple[float,float]]):
    arr = np.array(pts, dtype=np.float32)
    mn = arr.min(axis=0); mx = arr.max(axis=0)
    centroid = arr.mean(axis=0)
    return mn, mx, centroid


# -------------------------
# Mask: rasterize polygon, erosion(safe margin), optional SDF
# -------------------------
def rasterize_polygon_mask(H: int, W: int, pts: List[Tuple[float,float]]) -> np.ndarray:
    img = Image.new("L", (W,H), 0)
    draw = ImageDraw.Draw(img)
    draw.polygon(pts, fill=255)
    m = (np.array(img) > 0).astype(np.uint8)
    return m

def erode_mask_manhattan(mask: np.ndarray, radius: int) -> np.ndarray:
    # 빠르고 의존성 없는 erosion(맨해튼 반경). radius가 클수록 안쪽으로 줄어듦.
    m = mask.astype(np.uint8)
    if radius <= 0:
        return m
    out = m.copy()
    for _ in range(radius):
        up = np.roll(out, -1, axis=0)
        dn = np.roll(out,  1, axis=0)
        lf = np.roll(out, -1, axis=1)
        rt = np.roll(out,  1, axis=1)
        out = out & up & dn & lf & rt
        # 경계 wrap 방지
        out[0,:] = 0; out[-1,:] = 0; out[:,0] = 0; out[:,-1] = 0
    return out

def inside_mask(mask: np.ndarray, p: np.ndarray) -> bool:
    H, W = mask.shape
    x = int(round(float(p[0])))
    y = int(round(float(p[1])))
    if x < 0 or x >= W or y < 0 or y >= H:
        return False
    return bool(mask[y, x] == 1)

def nearest_inside_point_bruteforce(mask: np.ndarray, p: np.ndarray, max_r: int = 40) -> np.ndarray:
    # SDF 없을 때 내부 투영용(가벼운 탐색)
    H,W = mask.shape
    x0 = int(round(float(p[0]))); y0 = int(round(float(p[1])))
    if 0 <= x0 < W and 0 <= y0 < H and mask[y0, x0] == 1:
        return np.array([x0, y0], dtype=np.float32)
    best = None
    for r in range(1, max_r+1):
        for dy in range(-r, r+1):
            for dx in range(-r, r+1):
                if abs(dx)+abs(dy) != r:
                    continue
                x = x0+dx; y = y0+dy
                if 0 <= x < W and 0 <= y < H and mask[y,x] == 1:
                    return np.array([x,y], dtype=np.float32)
    # 못 찾으면 클램프
    x = int(np.clip(x0, 0, W-1)); y = int(np.clip(y0, 0, H-1))
    return np.array([x,y], dtype=np.float32)


# -------------------------
# Segment clipping to mask (hard guarantee)
# -------------------------
def sample_line_points(p0: np.ndarray, p1: np.ndarray, step: float = 1.0) -> np.ndarray:
    d = p1 - p0
    dist = float(np.linalg.norm(d))
    if dist < 1e-6:
        return p0[None, :]
    n = max(2, int(math.ceil(dist/step)) + 1)
    t = np.linspace(0, 1, n, dtype=np.float32)[:, None]
    return p0[None, :] * (1-t) + p1[None, :] * t

def refine_boundary(mask: np.ndarray, a: np.ndarray, b: np.ndarray, a_inside: bool, iters: int = 12) -> np.ndarray:
    # a와 b 사이에서 inside<->outside 경계 교차점을 이분탐색
    lo = a.copy(); hi = b.copy()
    lo_in = a_inside
    for _ in range(iters):
        mid = (lo+hi)/2
        mid_in = inside_mask(mask, mid)
        if mid_in == lo_in:
            lo = mid
        else:
            hi = mid
    return (lo+hi)/2

def clip_segment_to_mask(p0: np.ndarray, p1: np.ndarray, safe_mask: np.ndarray, step: float = 1.0) -> List[Tuple[np.ndarray,np.ndarray]]:
    # 샘플링으로 inside 구간 찾고, 경계는 이분탐색으로 정밀화
    pts = sample_line_points(p0, p1, step=step)
    inside = np.array([inside_mask(safe_mask, pt) for pt in pts], dtype=np.bool_)
    segs = []
    if not inside.any():
        return segs

    # 연속 inside runs
    n = len(pts)
    i = 0
    while i < n:
        if not inside[i]:
            i += 1
            continue
        j = i
        while j < n and inside[j]:
            j += 1
        # run [i, j-1]
        q0 = pts[i].copy()
        q1 = pts[j-1].copy()

        # 앞 경계 보정(i>0 이면 i-1은 outside일 수 있음)
        if i > 0 and inside[i] and not inside[i-1]:
            q0 = refine_boundary(safe_mask, pts[i-1], pts[i], a_inside=False)
        # 뒤 경계 보정(j<n 이면 j는 outside)
        if j < n and inside[j-1] and not inside[j]:
            q1 = refine_boundary(safe_mask, pts[j-1], pts[j], a_inside=True)

        if float(np.linalg.norm(q1-q0)) >= 1e-3:
            segs.append((q0, q1))
        i = j
    return segs


# -------------------------
# Ink recording + rendering (pressure)
# -------------------------
@dataclass
class InkSeg:
    p0: Tuple[float,float]
    p1: Tuple[float,float]
    width: float
    alpha: float
    passes: int

class InkRecorder:
    def __init__(self):
        self.segs: List[InkSeg] = []

    def add_clipped(self, clipped: List[Tuple[np.ndarray,np.ndarray]], pressure: float):
        # pressure -> width/alpha/passes
        p = float(np.clip(pressure, 0.0, 1.0))
        width = 1.0 + 4.0*p
        alpha = 0.25 + 0.75*p
        passes = 1 + int(round(2*p))
        for q0, q1 in clipped:
            self.segs.append(InkSeg(
                p0=(float(q0[0]), float(q0[1])),
                p1=(float(q1[0]), float(q1[1])),
                width=width,
                alpha=alpha,
                passes=passes
            ))

    def render_png(self, H: int, W: int, out_path: str, ink_only: bool = True,
                   poly_pts: Optional[List[Tuple[float,float]]] = None, poly_alpha: float = 0.12):
        # RGBA로 잉크 레이어 렌더
        img = Image.new("RGBA", (W,H), (255,255,255,0 if ink_only else 255))
        draw = ImageDraw.Draw(img, "RGBA")

        # preview용 폴리곤 저투명
        if (not ink_only) and (poly_pts is not None):
            a = int(255*np.clip(poly_alpha, 0, 1))
            draw.polygon(poly_pts, outline=(0,0,0,a), fill=(0,0,0,int(a*0.25)))

        for s in self.segs:
            a = int(255*np.clip(s.alpha, 0, 1))
            # passes로 덧그리기(농도↑ 느낌)
            for _ in range(s.passes):
                draw.line([s.p0, s.p1], fill=(0,0,0,a), width=int(round(s.width)))
        img.save(out_path)

    def render_svg(self, H: int, W: int, out_path: str):
        # 단순 SVG(선분만). alpha/width 반영.
        lines = []
        lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">')
        for s in self.segs:
            a = np.clip(s.alpha, 0, 1)
            lines.append(
                f'<line x1="{s.p0[0]:.2f}" y1="{s.p0[1]:.2f}" x2="{s.p1[0]:.2f}" y2="{s.p1[1]:.2f}" '
                f'stroke="black" stroke-width="{s.width:.2f}" stroke-opacity="{a:.3f}" stroke-linecap="round" />'
            )
        lines.append("</svg>")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))


# -------------------------
# Shape generator: simple polygon + parallel edges enforced
# -------------------------
@dataclass
class ShapeSpec:
    n_min: int = 6
    n_max: int = 12
    min_area: float = 16000.0
    min_edge_len: float = 18.0
    min_span: float = 140.0
    edge_len_ratio_min: float = 1.5
    angle_std_min: float = 0.20
    concave_max: int = 2
    require_axis_edge: bool = True
    parallel_tol: float = 1e-3  # 실제 강제 생성이므로 tol은 크게 의미 없음
    max_tries: int = 400

def make_base_star_polygon(cx, cy, r_min, r_max, n) -> List[Tuple[float,float]]:
    # 반지름 랜덤 + 각도 정렬 => self-intersection 가능성 낮음
    angles = np.sort(np.random.uniform(0, 2*np.pi, size=n))
    rs = np.random.uniform(r_min, r_max, size=n)
    pts = []
    for a, r in zip(angles, rs):
        pts.append((cx + r*np.cos(a), cy + r*np.sin(a)))
    return pts

def enforce_parallel_pair(pts: List[Tuple[float,float]]) -> List[Tuple[float,float]]:
    # 한 쌍의 평행 변을 강제: i번째 엣지 벡터를 골라, j번째 엣지를 같은 방향(스케일)으로 세팅
    n = len(pts)
    pts = [np.array(p, dtype=np.float32) for p in pts]
    i = np.random.randint(0, n)
    j = (i + np.random.randint(2, n-1)) % n  # 인접 피함
    a0 = pts[i]
    a1 = pts[(i+1)%n]
    v = a1 - a0
    v_norm = v / (np.linalg.norm(v)+1e-9)

    b0 = pts[j]
    b1 = pts[(j+1)%n]
    L = float(np.linalg.norm(b1-b0))
    if L < 1e-6:
        L = float(np.linalg.norm(v)) * np.random.uniform(0.6, 1.3)
    # b1을 b0 + v_dir * L 로 재배치 (평행 강제)
    pts[(j+1)%n] = b0 + v_norm * L

    # 소폭 jitter로 정다각형 느낌 억제
    for k in range(n):
        pts[k] += np.random.uniform(-2.5, 2.5, size=2).astype(np.float32)

    out = [(float(p[0]), float(p[1])) for p in pts]
    return out

def include_axis_aligned_edge(pts: List[Tuple[float,float]]) -> List[Tuple[float,float]]:
    # 거의 수평/수직 변 1개 포함(설계도 느낌 강화)
    n = len(pts)
    pts = [np.array(p, dtype=np.float32) for p in pts]
    i = np.random.randint(0, n)
    a = pts[i]
    b = pts[(i+1)%n]
    if np.random.rand() < 0.5:
        # 수평: y 맞추기
        b[1] = a[1] + np.random.uniform(-1.5, 1.5)
    else:
        # 수직: x 맞추기
        b[0] = a[0] + np.random.uniform(-1.5, 1.5)
    pts[(i+1)%n] = b
    return [(float(p[0]), float(p[1])) for p in pts]

def generate_polygon(H: int, W: int, spec: ShapeSpec) -> List[Tuple[float,float]]:
    # 재시도 기반: 조건 불만족 시 계속 생성
    for _ in range(spec.max_tries):
        n = int(np.random.randint(spec.n_min, spec.n_max+1))
        cx = np.random.uniform(W*0.35, W*0.65)
        cy = np.random.uniform(H*0.35, H*0.65)
        r_max = np.random.uniform(min(H,W)*0.22, min(H,W)*0.34)
        r_min = r_max * np.random.uniform(0.35, 0.6)

        pts = make_base_star_polygon(cx, cy, r_min, r_max, n)
        pts = enforce_parallel_pair(pts)
        if spec.require_axis_edge:
            pts = include_axis_aligned_edge(pts)

        pts = ensure_ccw(pts)

        # 바운딩/스팬 검사
        mn, mx, _ = bbox_and_centroid(pts)
        span = float(np.min(mx - mn))
        if span < spec.min_span:
            continue

        # 자기교차 검사
        if polygon_self_intersects(pts):
            continue

        # 면적/최소변/비정형성
        if polygon_area(pts) < spec.min_area:
            continue
        if min_edge_len(pts) < spec.min_edge_len:
            continue
        if edge_len_ratio(pts) < spec.edge_len_ratio_min:
            continue
        if angle_dispersion_score(pts) < spec.angle_std_min:
            continue
        if count_concave_vertices(pts) > spec.concave_max:
            continue

        # 화면 밖으로 나가면 살짝 클램프
        clamped = []
        for x,y in pts:
            clamped.append((float(np.clip(x, 5, W-6)), float(np.clip(y, 5, H-6))))
        return clamped

    raise RuntimeError("폴리곤 생성 실패: 제약이 너무 강하거나 캔버스가 너무 작습니다.")


# -------------------------
# Style tokens + Plan generator (설계도/회로도 느낌)
# -------------------------
@dataclass
class StyleConfig:
    # 토큰 빈도/스케일(변주)
    backbone_prob: float = 0.90
    loop_prob: float = 0.70
    node_prob: float = 0.85
    joint_prob: float = 0.75
    hatch_prob: float = 0.55
    aux_prob: float = 0.65

    nodes_min: int = 6
    nodes_max: int = 18
    loops_min: int = 1
    loops_max: int = 4

    # pressure 레벨(토큰별 기본)
    p_backbone: float = 0.55
    p_loop: float = 0.45
    p_node: float = 0.85
    p_joint: float = 0.75
    p_hatch: float = 0.25
    p_aux: float = 0.18

@dataclass
class Stroke:
    points: List[Tuple[float,float]]  # polyline points
    pressure: float
    pen_down: bool = True

def sample_points_in_mask(mask: np.ndarray, k: int) -> List[Tuple[float,float]]:
    ys, xs = np.where(mask == 1)
    if len(xs) == 0:
        return []
    idx = np.random.choice(len(xs), size=min(k, len(xs)), replace=False)
    pts = [(float(xs[i]), float(ys[i])) for i in idx]
    return pts

def jitter_point(p: Tuple[float,float], s: float) -> Tuple[float,float]:
    return (p[0] + float(np.random.uniform(-s,s)), p[1] + float(np.random.uniform(-s,s)))

def clip_stroke_to_mask(stroke: Stroke, safe_mask: np.ndarray) -> List[Stroke]:
    # polyline을 segment 단위로 클리핑 후 inside 구간만 남김
    pts = [np.array(p, dtype=np.float32) for p in stroke.points]
    out_strokes: List[Stroke] = []
    current: List[Tuple[float,float]] = []
    for i in range(len(pts)-1):
        p0, p1 = pts[i], pts[i+1]
        clipped = clip_segment_to_mask(p0, p1, safe_mask, step=1.0)
        if len(clipped) == 0:
            if len(current) >= 2:
                out_strokes.append(Stroke(points=current, pressure=stroke.pressure, pen_down=True))
            current = []
            continue
        # 여러 구간이 나올 수 있음
        for (q0, q1) in clipped:
            q0t = (float(q0[0]), float(q0[1]))
            q1t = (float(q1[0]), float(q1[1]))
            if not current:
                current = [q0t, q1t]
            else:
                # 이어붙일 수 있으면 붙이고 아니면 새 stroke
                if l2(np.array(current[-1],dtype=np.float32), np.array(q0t,dtype=np.float32)) < 2.0:
                    current.append(q1t)
                else:
                    if len(current) >= 2:
                        out_strokes.append(Stroke(points=current, pressure=stroke.pressure, pen_down=True))
                    current = [q0t, q1t]
    if len(current) >= 2:
        out_strokes.append(Stroke(points=current, pressure=stroke.pressure, pen_down=True))
    return out_strokes

def plan_generator(safe_mask: np.ndarray, style_cfg: StyleConfig, z: np.ndarray) -> List[Stroke]:
    # z: 변주 코드 (간단 난수 벡터)
    H,W = safe_mask.shape
    plan: List[Stroke] = []

    # z로 토큰 밀도/확률 조절
    dens = float(np.clip(0.6 + 0.8*z[0], 0.35, 1.4))
    hatch_boost = float(np.clip(0.5 + 1.2*z[1], 0.2, 1.8))
    loop_boost = float(np.clip(0.6 + 1.0*z[2], 0.2, 1.6))
    jitter = float(np.clip(2.0 + 4.0*z[3], 1.0, 8.0))

    # 노드 샘플
    n_nodes = int(np.clip(np.random.randint(style_cfg.nodes_min, style_cfg.nodes_max+1) * dens, 4, 28))
    nodes = sample_points_in_mask(safe_mask, n_nodes)
    if len(nodes) < 4:
        return plan

    # 1) 뼈대선(backbone): 노드들을 MST 비슷하게 연결(간단히 kNN 연결)
    if np.random.rand() < style_cfg.backbone_prob:
        pts = np.array(nodes, dtype=np.float32)
        # kNN (k=2~3) 연결
        k = 2 if np.random.rand() < 0.6 else 3
        for i in range(len(nodes)):
            d = np.linalg.norm(pts - pts[i:i+1], axis=1)
            nn = np.argsort(d)[1:k+1]
            for j in nn:
                if np.random.rand() < 0.65:
                    p0 = jitter_point(nodes[i], jitter)
                    p1 = jitter_point(nodes[j], jitter)
                    s = Stroke(points=[p0, p1], pressure=style_cfg.p_backbone, pen_down=True)
                    plan.extend(clip_stroke_to_mask(s, safe_mask))

    # 2) 폴리곤 루프(loop): 내부 랜덤 루프 1~4개
    if np.random.rand() < style_cfg.loop_prob * loop_boost:
        n_loops = int(np.clip(np.random.randint(style_cfg.loops_min, style_cfg.loops_max+1) * loop_boost, 1, 6))
        for _ in range(n_loops):
            # 루프 중심
            c = nodes[np.random.randint(0, len(nodes))]
            # 루프 크기
            rad = float(np.random.uniform(18, 70) * dens)
            m = int(np.random.randint(3, 7))
            ang0 = float(np.random.uniform(0, 2*np.pi))
            loop = []
            for t in range(m):
                a = ang0 + t*(2*np.pi/m) + float(np.random.uniform(-0.25, 0.25))
                r = rad * float(np.random.uniform(0.75, 1.15))
                loop.append(jitter_point((c[0]+r*np.cos(a), c[1]+r*np.sin(a)), jitter*0.6))
            loop.append(loop[0])
            s = Stroke(points=loop, pressure=style_cfg.p_loop, pen_down=True)
            plan.extend(clip_stroke_to_mask(s, safe_mask))

    # 3) 노드(node): 점/접점(짧은 cross/점)
    if np.random.rand() < style_cfg.node_prob:
        for p in nodes:
            if np.random.rand() < 0.7:
                # 작은 + 형태
                size = float(np.random.uniform(2.5, 6.0))
                s1 = Stroke(points=[(p[0]-size, p[1]), (p[0]+size, p[1])], pressure=style_cfg.p_node, pen_down=True)
                s2 = Stroke(points=[(p[0], p[1]-size), (p[0], p[1]+size)], pressure=style_cfg.p_node, pen_down=True)
                plan.extend(clip_stroke_to_mask(s1, safe_mask))
                plan.extend(clip_stroke_to_mask(s2, safe_mask))

    # 4) 조인트 디테일(joint): 작은 사각/스텝 패턴
    if np.random.rand() < style_cfg.joint_prob:
        for _ in range(int(6*dens)):
            p = nodes[np.random.randint(0,len(nodes))]
            w = float(np.random.uniform(6, 16))
            h = float(np.random.uniform(4, 12))
            # 직교 사각(회로도 느낌)
            rect = [
                (p[0], p[1]),
                (p[0]+w, p[1]),
                (p[0]+w, p[1]+h),
                (p[0], p[1]+h),
                (p[0], p[1])
            ]
            s = Stroke(points=[jitter_point(q, jitter*0.3) for q in rect], pressure=style_cfg.p_joint, pen_down=True)
            plan.extend(clip_stroke_to_mask(s, safe_mask))

    # 5) 해칭(hatch): 부분 톤
    if np.random.rand() < style_cfg.hatch_prob * hatch_boost:
        # 해칭 영역을 랜덤으로 고르고 평행선 뽑기
        n_h = int(np.clip(np.random.randint(10, 26)*hatch_boost, 8, 60))
        angle = float(np.random.choice([0, np.pi/4, np.pi/2, 3*np.pi/4]) + np.random.uniform(-0.12, 0.12))
        dirv = np.array([math.cos(angle), math.sin(angle)], dtype=np.float32)
        # 기준점들
        base = sample_points_in_mask(safe_mask, min(n_h, 120))
        for i in range(min(n_h, len(base))):
            p = base[i]
            length = float(np.random.uniform(24, 120) * dens)
            p0 = (p[0]-dirv[0]*length*0.5, p[1]-dirv[1]*length*0.5)
            p1 = (p[0]+dirv[0]*length*0.5, p[1]+dirv[1]*length*0.5)
            s = Stroke(points=[jitter_point(p0, jitter*0.35), jitter_point(p1, jitter*0.35)],
                       pressure=style_cfg.p_hatch, pen_down=True)
            if np.random.rand() < 0.5:
                plan.extend(clip_stroke_to_mask(s, safe_mask))

    # 6) 보조선(aux): 얇고 옅은 레이어(긴 직선 몇 개)
    if np.random.rand() < style_cfg.aux_prob:
        n_aux = int(np.clip(np.random.randint(2, 7)*dens, 2, 10))
        for _ in range(n_aux):
            p0 = nodes[np.random.randint(0,len(nodes))]
            p1 = nodes[np.random.randint(0,len(nodes))]
            if l2(np.array(p0), np.array(p1)) < 40:
                continue
            s = Stroke(points=[jitter_point(p0, jitter*0.6), jitter_point(p1, jitter*0.6)],
                       pressure=style_cfg.p_aux, pen_down=True)
            plan.extend(clip_stroke_to_mask(s, safe_mask))

    # 매번 다른 결과 보장(일부만 선택/분할/순서 랜덤)
    np.random.shuffle(plan)
    keep = int(np.clip(len(plan) * float(np.clip(0.7 + 0.6*z[4], 0.35, 1.0)), 8, len(plan)))
    return plan[:keep]


# -------------------------
# Robot Arm (2-link) + FK/IK
# -------------------------
@dataclass
class ArmConfig:
    l1: float = 140.0
    l2: float = 120.0
    joint_limit: float = math.pi  # [-pi, pi]
    max_dtheta: float = 0.12
    max_dxy: float = 8.0

def fk(theta: np.ndarray, cfg: ArmConfig) -> np.ndarray:
    t1, t2 = float(theta[0]), float(theta[1])
    x = cfg.l1*math.cos(t1) + cfg.l2*math.cos(t1+t2)
    y = cfg.l1*math.sin(t1) + cfg.l2*math.sin(t1+t2)
    return np.array([x,y], dtype=np.float32)

def ik(xy: np.ndarray, cfg: ArmConfig, elbow_up: bool = True) -> Optional[np.ndarray]:
    x, y = float(xy[0]), float(xy[1])
    L1, L2 = cfg.l1, cfg.l2
    r2 = x*x + y*y
    c2 = (r2 - L1*L1 - L2*L2) / (2*L1*L2)
    if c2 < -1.0 or c2 > 1.0:
        return None
    s2 = math.sqrt(max(0.0, 1.0 - c2*c2))
    if not elbow_up:
        s2 = -s2
    t2 = math.atan2(s2, c2)
    k1 = L1 + L2*c2
    k2 = L2*s2
    t1 = math.atan2(y, x) - math.atan2(k2, k1)
    # normalize
    t1 = (t1 + math.pi)%(2*math.pi) - math.pi
    t2 = (t2 + math.pi)%(2*math.pi) - math.pi
    return np.array([t1,t2], dtype=np.float32)


# -------------------------
# Gymnasium Env
# -------------------------
@dataclass
class EnvConfig:
    H: int = 512
    W: int = 512
    safe_margin: int = 10
    max_steps: int = 1800
    lookahead: int = 1
    action_mode: str = "ee_delta"  # "ee_delta" or "joint_delta"
    shield: bool = True

    # reward weights
    w_progress: float = 1.0
    w_dist: float = 0.08
    w_smooth: float = 0.02
    w_toggle: float = 0.02

class MaskedInkArmEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, env_cfg: EnvConfig, arm_cfg: ArmConfig, shape_spec: ShapeSpec, style_cfg: StyleConfig):
        super().__init__()
        self.env_cfg = env_cfg
        self.arm_cfg = arm_cfg
        self.shape_spec = shape_spec
        self.style_cfg = style_cfg

        # action: (dx,dy, pen_toggle, pressure_delta)
        # pen_toggle: [-1,1] continuous, sign으로 up/down
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        # observation: theta(2), ee_xy(2), target_xy(2), delta_to_target(2), progress(1), pen(1), style_z(5)
        self.obs_dim = 2+2+2+2+1+1+5
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)

        # state
        self.poly_pts: List[Tuple[float,float]] = []
        self.mask = None
        self.safe_mask = None
        self.style_z = None
        self.plan: List[Stroke] = []
        self.plan_idx = 0
        self.pt_idx = 0

        self.theta = np.zeros(2, dtype=np.float32)
        self.ee = np.zeros(2, dtype=np.float32)
        self.origin = np.array([env_cfg.W/2, env_cfg.H/2], dtype=np.float32)  # arm base at center
        self.pen_down = False
        self.pressure = 0.5
        self.ink = InkRecorder()

        self.prev_action = np.zeros(4, dtype=np.float32)
        self.steps = 0
        self.shield_trigger_count = 0
        self.pen_toggle_count = 0

        # runtime target
        self._target = np.zeros(2, dtype=np.float32)

    def _workspace_to_canvas(self, xy: np.ndarray) -> np.ndarray:
        # 로봇 좌표(원점 기준) -> 캔버스 픽셀
        return self.origin + xy

    def _canvas_to_workspace(self, pxy: np.ndarray) -> np.ndarray:
        return pxy - self.origin

    def _set_target(self):
        if self.plan_idx >= len(self.plan):
            self._target = self.ee.copy()
            return
        stroke = self.plan[self.plan_idx]
        pts = stroke.points
        j = min(self.pt_idx + self.env_cfg.lookahead, len(pts)-1)
        self._target = np.array(pts[j], dtype=np.float32)

    def _advance_plan_if_needed(self, tol: float = 6.0):
        if self.plan_idx >= len(self.plan):
            return
        stroke = self.plan[self.plan_idx]
        pts = stroke.points
        if self.pt_idx >= len(pts):
            self.plan_idx += 1
            self.pt_idx = 0
            return
        # 가까우면 다음 포인트
        if l2(self.ee, np.array(pts[self.pt_idx], dtype=np.float32)) < tol:
            self.pt_idx += 1
            if self.pt_idx >= len(pts):
                self.plan_idx += 1
                self.pt_idx = 0

    def _expert_action(self) -> np.ndarray:
        # 교사: target로 ee를 부드럽게 이동, stroke 구간에서는 pen_down True
        self._set_target()
        d = self._target - self.ee
        dist = float(np.linalg.norm(d))
        if dist > 1e-6:
            step = min(self.arm_cfg.max_dxy, dist)
            move = d / dist * step
        else:
            move = np.zeros(2, dtype=np.float32)

        # pen: 계획 stroke가 남아 있으면 down, 아니면 up
        pen = 1.0 if self.plan_idx < len(self.plan) else -1.0

        # pressure: stroke 토큰 pressure 사용(경계 근처는 감소)
        if self.plan_idx < len(self.plan):
            p = float(np.clip(self.plan[self.plan_idx].pressure, 0, 1))
        else:
            p = 0.3
        # 경계 착시 방지: safe_mask 밖으로 두꺼운 선 튀어 보임 방지 위해 near-boundary면 p 감소
        # (간단히 safe_margin을 한 번 더 적용한 더 작은 마스크로 검사)
        inner = erode_mask_manhattan(self.safe_mask, radius=2)
        if not inside_mask(inner, self.ee):
            p *= 0.55

        # action range [-1,1]로 정규화
        dx = float(np.clip(move[0] / self.arm_cfg.max_dxy, -1, 1))
        dy = float(np.clip(move[1] / self.arm_cfg.max_dxy, -1, 1))
        pt = float(np.clip((p - 0.5)/0.5, -1, 1))
        return np.array([dx, dy, pen, pt], dtype=np.float32)

    def _obs(self) -> np.ndarray:
        self._set_target()
        target = self._target
        d = target - self.ee
        prog = 1.0 - (self.plan_idx / max(1, len(self.plan)))
        pen = 1.0 if self.pen_down else 0.0
        obs = np.concatenate([
            self.theta,
            self.ee,
            target,
            d,
            np.array([prog], dtype=np.float32),
            np.array([pen], dtype=np.float32),
            self.style_z
        ]).astype(np.float32)
        return obs

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        cfg = self.env_cfg

        # 런타임 변주용 seed는 비결정적(학습은 config로 고정 가능)
        # 여기서는 gym seed와 별개로 style/shape에 random 사용
        self.poly_pts = generate_polygon(cfg.H, cfg.W, self.shape_spec)
        self.mask = rasterize_polygon_mask(cfg.H, cfg.W, self.poly_pts)
        self.safe_mask = erode_mask_manhattan(self.mask, radius=cfg.safe_margin)

        # style z: 5-dim
        self.style_z = np.random.uniform(0, 1, size=(5,)).astype(np.float32)
        self.plan = plan_generator(self.safe_mask, self.style_cfg, self.style_z)

        self.plan_idx = 0
        self.pt_idx = 0

        # 초기 ee를 safe_mask 내부에서 샘플
        pts = sample_points_in_mask(self.safe_mask, 1)
        if not pts:
            raise RuntimeError("safe_mask가 비었습니다. safe_margin이 너무 크거나 도형이 너무 작습니다.")
        self.ee = np.array(pts[0], dtype=np.float32)

        # 초기 theta는 IK로
        ws = self._canvas_to_workspace(self.ee)
        th = ik(ws, self.arm_cfg, elbow_up=True)
        if th is None:
            # workspace 밖이면 근처로 투영
            self.ee = nearest_inside_point_bruteforce(self.safe_mask, self.ee, max_r=40)
            ws = self._canvas_to_workspace(self.ee)
            th = ik(ws, self.arm_cfg, elbow_up=True)
            if th is None:
                th = np.zeros(2, dtype=np.float32)
        self.theta = th
        self.pen_down = False
        self.pressure = 0.5
        self.ink = InkRecorder()

        self.steps = 0
        self.shield_trigger_count = 0
        self.pen_toggle_count = 0
        self.prev_action = np.zeros(4, dtype=np.float32)

        obs = self._obs()
        info = {"shield_trigger_count": 0, "outside_ink_ratio": 0.0}
        return obs, info

    def step(self, action: np.ndarray):
        cfg = self.env_cfg
        self.steps += 1

        action = np.array(action, dtype=np.float32)
        dx = float(np.clip(action[0], -1, 1)) * self.arm_cfg.max_dxy
        dy = float(np.clip(action[1], -1, 1)) * self.arm_cfg.max_dxy
        pen_cmd = float(action[2])
        p_cmd = float(action[3])

        prev_pen = self.pen_down
        if pen_cmd >= 0:
            self.pen_down = True
        else:
            self.pen_down = False
        if prev_pen != self.pen_down:
            self.pen_toggle_count += 1

        # pressure update
        self.pressure = float(np.clip(0.5 + 0.5*p_cmd, 0.0, 1.0))

        # 후보 ee 이동
        prev_ee = self.ee.copy()
        cand = self.ee + np.array([dx, dy], dtype=np.float32)

        # Safety Shield: 밖으로 나가려 하면 pen up + 내부 투영
        if cfg.shield and (not inside_mask(self.safe_mask, cand)):
            self.shield_trigger_count += 1
            self.pen_down = False
            cand = nearest_inside_point_bruteforce(self.safe_mask, cand, max_r=50)

        # 상태 업데이트
        self.ee = cand

        # theta update (IK)
        ws = self._canvas_to_workspace(self.ee)
        th = ik(ws, self.arm_cfg, elbow_up=True)
        if th is not None:
            # joint limit clamp
            th = np.clip(th, -self.arm_cfg.joint_limit, self.arm_cfg.joint_limit)
            self.theta = th

        # Ink recording (hard): pen_down이면 segment clip 후 내부 구간만 기록
        if self.pen_down:
            clipped = clip_segment_to_mask(prev_ee, self.ee, self.safe_mask, step=1.0)
            self.ink.add_clipped(clipped, pressure=self.pressure)

        # 진행
        self._advance_plan_if_needed()

        # reward 구성
        self._set_target()
        dist = l2(self.ee, self._target)
        progress = 1.0 if dist < 6.0 else 0.0
        smooth = float(np.linalg.norm(action[:2] - self.prev_action[:2]))
        toggle = 1.0 if prev_pen != self.pen_down else 0.0
        reward = (cfg.w_progress*progress) - (cfg.w_dist*dist) - (cfg.w_smooth*smooth) - (cfg.w_toggle*toggle)

        self.prev_action = action.copy()

        terminated = (self.plan_idx >= len(self.plan))  # 계획 완료
        truncated = (self.steps >= cfg.max_steps)

        obs = self._obs()

        # outside_ink_ratio는 구조상 0 (클리핑 게이트)
        info = {
            "shield_trigger_count": self.shield_trigger_count,
            "pen_toggle_count": self.pen_toggle_count,
            "outside_ink_ratio": 0.0,
            "plan_done": terminated,
            "dist_to_target": dist
        }
        return obs, reward, terminated, truncated, info

    def render_preview(self) -> np.ndarray:
        # preview: 배경 + 도형(저투명) + 잉크
        H,W = self.env_cfg.H, self.env_cfg.W
        tmp = Image.new("RGBA", (W,H), (255,255,255,255))
        draw = ImageDraw.Draw(tmp, "RGBA")
        # polygon faint
        a = int(255*0.12)
        draw.polygon(self.poly_pts, outline=(0,0,0,a), fill=(0,0,0,int(a*0.20)))
        # ink over
        ink_img = Image.new("RGBA", (W,H), (0,0,0,0))
        ink_draw = ImageDraw.Draw(ink_img, "RGBA")
        for s in self.ink.segs:
            aa = int(255*np.clip(s.alpha, 0, 1))
            for _ in range(s.passes):
                ink_draw.line([s.p0, s.p1], fill=(0,0,0,aa), width=int(round(s.width)))
        tmp = Image.alpha_composite(tmp, ink_img)
        return np.array(tmp.convert("RGB"))

    def export_ink(self, out_png: str, out_svg: Optional[str] = None):
        H,W = self.env_cfg.H, self.env_cfg.W
        self.ink.render_png(H,W,out_png, ink_only=True)
        if out_svg is not None:
            self.ink.render_svg(H,W,out_svg)


# -------------------------
# BC dataset + policy
# -------------------------
class MLPPolicy(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, act_dim),
            nn.Tanh()  # action range [-1,1]
        )
    def forward(self, x):
        return self.net(x)

@dataclass
class TrainConfig:
    exp_name: str = "masked_ink_bc"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    train_seed: int = 12345
    runtime_nondet: bool = True

    demos_episodes: int = 40
    demos_max_steps: int = 1600

    batch_size: int = 1024
    epochs: int = 20
    lr: float = 3e-4

    eval_episodes: int = 6

def collect_demos(env: MaskedInkArmEnv, cfg: TrainConfig) -> Tuple[np.ndarray, np.ndarray, Dict[str,float]]:
    obs_buf = []
    act_buf = []
    shield_counts = []
    dists = []

    for ep in range(cfg.demos_episodes):
        obs, _ = env.reset()
        for t in range(cfg.demos_max_steps):
            act = env._expert_action()
            obs_buf.append(obs.copy())
            act_buf.append(act.copy())
            obs, r, term, trunc, info = env.step(act)
            shield_counts.append(info["shield_trigger_count"])
            dists.append(info["dist_to_target"])
            if term or trunc:
                break

    obs_arr = np.array(obs_buf, dtype=np.float32)
    act_arr = np.array(act_buf, dtype=np.float32)
    stats = {
        "demo_steps": float(len(obs_arr)),
        "avg_dist": float(np.mean(dists)) if dists else 0.0,
        "max_shield": float(np.max(shield_counts)) if shield_counts else 0.0,
    }
    return obs_arr, act_arr, stats

def train_bc(obs: np.ndarray, act: np.ndarray, obs_dim: int, act_dim: int, cfg: TrainConfig):
    device = torch.device(cfg.device)
    model = MLPPolicy(obs_dim, act_dim).to(device)
    opt = optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.MSELoss()

    X = torch.from_numpy(obs).to(device)
    Y = torch.from_numpy(act).to(device)

    N = X.shape[0]
    idx = np.arange(N)

    for epoch in range(cfg.epochs):
        np.random.shuffle(idx)
        total = 0.0
        for i in range(0, N, cfg.batch_size):
            b = idx[i:i+cfg.batch_size]
            xb = X[b]
            yb = Y[b]
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(b)
        print(f"[BC] epoch {epoch+1:02d}/{cfg.epochs}  loss={total/N:.6f}")
    return model

@torch.no_grad()
def eval_policy(env: MaskedInkArmEnv, model: nn.Module, episodes: int, device: str) -> Dict[str,float]:
    dev = torch.device(device)
    model.eval()
    done_rates = []
    avg_dists = []
    shield = []
    steps = []

    for ep in range(episodes):
        obs, _ = env.reset()
        dsum = 0.0
        for t in range(env.env_cfg.max_steps):
            x = torch.from_numpy(obs).to(dev).unsqueeze(0)
            act = model(x).squeeze(0).cpu().numpy().astype(np.float32)
            obs, r, term, trunc, info = env.step(act)
            dsum += float(info["dist_to_target"])
            if term or trunc:
                done_rates.append(1.0 if term else 0.0)
                avg_dists.append(dsum / (t+1))
                shield.append(float(info["shield_trigger_count"]))
                steps.append(float(t+1))
                break

    return {
        "done_rate": float(np.mean(done_rates)) if done_rates else 0.0,
        "avg_dist": float(np.mean(avg_dists)) if avg_dists else 0.0,
        "avg_shield": float(np.mean(shield)) if shield else 0.0,
        "avg_steps": float(np.mean(steps)) if steps else 0.0
    }

def make_runs_dir(exp_name: str) -> str:
    run_dir = os.path.join("runs", exp_name, now_timestamp())
    ensure_dir(run_dir)
    ensure_dir(os.path.join(run_dir, "samples"))
    ensure_dir(os.path.join(run_dir, "ckpt"))
    return run_dir


# -------------------------
# CLI Entrypoints
# -------------------------
def build_default_config_dict() -> dict:
    return {
        "train": asdict(TrainConfig()),
        "env": asdict(EnvConfig()),
        "arm": asdict(ArmConfig()),
        "shape": asdict(ShapeSpec()),
        "style": asdict(StyleConfig()),
    }

def build_env_from_cfg(cfg: dict) -> MaskedInkArmEnv:
    env_cfg = EnvConfig(**cfg["env"])
    arm_cfg = ArmConfig(**cfg["arm"])
    shape_spec = ShapeSpec(**cfg["shape"])
    style_cfg = StyleConfig(**cfg["style"])
    env = MaskedInkArmEnv(env_cfg, arm_cfg, shape_spec, style_cfg)
    return env

def cmd_init_config(args):
    cfg = build_default_config_dict()
    save_yaml(args.out, cfg)
    print(f"saved: {args.out}")

def cmd_train_bc(args):
    cfg = load_yaml(args.config)
    tcfg = TrainConfig(**cfg["train"])

    # 학습 재현용 seed는 고정 가능
    set_global_seed(tcfg.train_seed)

    run_dir = make_runs_dir(tcfg.exp_name)

    # 런타임 변주: shape/style는 비결정적로 하고 싶다면(전시/데모)
    if tcfg.runtime_nondet:
        np.random.seed(non_deterministic_seed())

    env = build_env_from_cfg(cfg)

    # 데모 수집
    obs, act, demo_stats = collect_demos(env, tcfg)
    print("[DEMO]", demo_stats)

    # BC 학습
    model = train_bc(obs, act, env.obs_dim, 4, tcfg)

    # 평가
    metrics = eval_policy(env, model, tcfg.eval_episodes, tcfg.device)
    print("[EVAL]", metrics)

    # 샘플 생성(프리뷰/익스포트)
    for k in range(3):
        obs0, _ = env.reset()
        for _ in range(env.env_cfg.max_steps):
            x = torch.from_numpy(obs0).to(torch.device(tcfg.device)).unsqueeze(0)
            a = model(x).squeeze(0).detach().cpu().numpy().astype(np.float32)
            obs0, r, term, trunc, info = env.step(a)
            if term or trunc:
                break
        # preview
        prev = env.render_preview()
        Image.fromarray(prev).save(os.path.join(run_dir, "samples", f"preview_{k:02d}.png"))
        # export ink only
        env.export_ink(
            out_png=os.path.join(run_dir, "samples", f"export_ink_{k:02d}.png"),
            out_svg=os.path.join(run_dir, "samples", f"export_ink_{k:02d}.svg") if args.svg else None
        )

    # 체크포인트/메타/런타임 패키징
    ckpt_path = os.path.join(run_dir, "ckpt", "policy.pt")
    torch.save({"state_dict": model.state_dict(), "obs_dim": env.obs_dim, "act_dim": 4}, ckpt_path)

    # export용 onnx (선택)
    onnx_path = None
    if args.onnx:
        onnx_path = os.path.join(run_dir, "ckpt", "policy.onnx")
        dummy = torch.zeros(1, env.obs_dim, device=torch.device(tcfg.device))
        torch.onnx.export(model, dummy, onnx_path, input_names=["obs"], output_names=["act"], opset_version=17)

    model_meta = {
        "obs_dim": env.obs_dim,
        "act_dim": 4,
        "canvas": {"H": env.env_cfg.H, "W": env.env_cfg.W},
        "version": "0.1.0",
        "action_spec": {"dx_dy": "[-1,1] scaled by max_dxy", "pen": "sign(act[2])", "pressure": "act[3]->[0,1]"},
    }
    style_meta = {
        "token_set": ["backbone", "loop", "node", "joint", "hatch", "aux"],
        "notes": "token-driven plan + random z for variation",
    }
    runtime_yaml = {
        "default": {
            "export_format": "png",
            "preview_polygon_alpha": 0.12,
            "safe_margin": env.env_cfg.safe_margin,
            "shield": env.env_cfg.shield,
            "nondeterministic": True
        }
    }

    save_json(os.path.join(run_dir, "model_meta.json"), model_meta)
    save_json(os.path.join(run_dir, "style_meta.json"), style_meta)
    save_yaml(os.path.join(run_dir, "runtime.yaml"), runtime_yaml)

    # 로그(간단)
    log = {"demo_stats": demo_stats, "eval": metrics, "paths": {"pt": ckpt_path, "onnx": onnx_path}}
    save_json(os.path.join(run_dir, "train_log.json"), log)

    print("run_dir:", run_dir)
    print("outside_ink_ratio는 구조상 0(클리핑 게이트)")

def cmd_demo(args):
    cfg = load_yaml(args.config)
    env = build_env_from_cfg(cfg)

    # 런타임은 비결정적 변주(요구사항)
    np.random.seed(non_deterministic_seed())

    # 무학습 데모: expert(플랜 추종)로 한 장 생성
    obs,_ = env.reset()
    for _ in range(env.env_cfg.max_steps):
        a = env._expert_action()
        obs, r, term, trunc, info = env.step(a)
        if term or trunc:
            break

    ensure_dir(args.out_dir)
    Image.fromarray(env.render_preview()).save(os.path.join(args.out_dir, "preview.png"))
    env.export_ink(
        out_png=os.path.join(args.out_dir, "export_ink.png"),
        out_svg=os.path.join(args.out_dir, "export_ink.svg") if args.svg else None
    )
    print("saved to:", args.out_dir)

def main():
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest="cmd", required=True)

    p_init = sp.add_parser("init_config")
    p_init.add_argument("--out", type=str, default="config.yaml")
    p_init.set_defaults(func=cmd_init_config)

    p_train = sp.add_parser("train_bc")
    p_train.add_argument("--config", type=str, default="config.yaml")
    p_train.add_argument("--onnx", action="store_true")
    p_train.add_argument("--svg", action="store_true")
    p_train.set_defaults(func=cmd_train_bc)

    p_demo = sp.add_parser("demo_expert")
    p_demo.add_argument("--config", type=str, default="config.yaml")
    p_demo.add_argument("--out_dir", type=str, default="demo_out")
    p_demo.add_argument("--svg", action="store_true")
    p_demo.set_defaults(func=cmd_demo)

    args = p.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
