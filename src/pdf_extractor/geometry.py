import numpy as np

def point_to_segment_dist2(px, py, ax, ay, bx, by):
    """
    (px,py) と 線分 A(ax,ay)-B(bx,by) の最短距離^2 を返す（ベクトル対応）
    px,py: scalar
    ax,ay,bx,by: shape (N,)
    return: shape (N,)
    """
    abx = bx - ax
    aby = by - ay
    apx = px - ax
    apy = py - ay

    denom = abx ** 2 + aby ** 2  # shape (N,)
    # 長さ0線分対策（ゼロ割回避）
    denom_safe = np.where(denom > 0, denom, 1.0)

    t = (apx * abx + apy * aby) / denom_safe
    t = np.clip(t, 0.0, 1.0)

    cx = ax + t * abx
    cy = ay + t * aby

    dx = px - cx
    dy = py - cy
    return dx * dx + dy * dy