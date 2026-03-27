import numpy as np

def trim_lines(lines: np.ndarray, bbox: tuple) -> np.ndarray:
    """
    bbox内に線分の両端点が含まれる線分を抽出する
    """
    if lines.size == 0:
        return lines

    x_min, y_min, x_max, y_max = bbox

    # 各端点がbbox内にあるかどうかの真偽値テーブルを作成
    # lines[:, 0] は全行の x0
    start_in = (lines[:, 0] >= x_min) & (lines[:, 0] <= x_max) & \
            (lines[:, 1] >= y_min) & (lines[:, 1] <= y_max)

    # lines[:, 2] は全行の x1
    end_in = (lines[:, 2] >= x_min) & (lines[:, 2] <= x_max) & \
            (lines[:, 3] >= y_min) & (lines[:, 3] <= y_max)

    # 「点0 または 点1 が範囲内」というマスクを作成
    mask = start_in | end_in

    # マスクを適用して抽出
    return lines[mask]


def remove_short_lines(lines: np.ndarray, min_length: float) -> np.ndarray:
    """
    指定した長さ未満の短い線分を除去する
    """
    if lines.size == 0:
        return lines

    # 各要素ごとの差分を計算
    dx = lines[:, 2] - lines[:, 0]  # x1 - x0
    dy = lines[:, 3] - lines[:, 1]  # y1 - y0

    # 全ての線分の長さを一斉に計算
    lengths = np.sqrt(dx**2 + dy**2)

    # 長さが閾値以上のものだけを抽出するマスクを作成
    mask = lengths >= min_length

    return lines[mask]