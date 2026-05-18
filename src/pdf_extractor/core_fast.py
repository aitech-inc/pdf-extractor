import fitz
import io
import numpy as np
from typing import List, Dict, Optional, Literal

def get_lines_from_pdf(
    pdf_bytes: bytes,
    dpi: int = 400,
    page_numbers: Optional[List[int]] = None,
    axis_aligned_only: bool = False,
    min_length: Optional[float] = None,
    include_lines: bool = True,
    include_rects: bool = True,
    include_curves: bool = True
):# -> Dict[int, np.ndarray]:
    """
    PDFから線分情報を取得する。

    :param pdf_bytes: PDFファイルの中身（バイト列）
    :param dpi: 解析時の解像度
    :param page_numbers: 抽出したいページのリスト（例: [0, 1]）。Noneの場合は全ページ。
    :param axis_aligned_only: Trueの場合、水平・垂直線のみ抽出する
    :param min_length: 指定した長さ未満の線分を除外する。Noneの場合は除外しない。
    :param include_lines: Trueの場合、線分エッジも線分として抽出する
    :param include_rects: Trueの場合、矩形エッジも線分として抽出する
    :param include_curves: Trueの場合、曲線エッジも線分として抽出する
    :return: ページ番号をキー、線分座標 [x0, y0, x1, y1] の numpy配列を値とした辞書
    """
    results = {}
    scale = dpi / 72  # PDFのデフォルトの解像度は72dpiなので、スケーリングが必要
    # 画像上の長さをpdf上の長さに変換するためのスケーリング
    min_length_pdf = min_length / scale if min_length is not None else None

    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        target_pages = page_numbers if page_numbers is not None else range(len(doc))

        for p_idx in target_pages:
            if p_idx >= len(doc):
                continue

            page = doc[p_idx]

            lines = []
            drawings = page.get_drawings()
            warning_flag = False
            for d in drawings:
                for item in d["items"]:
                    if include_lines and item[0] == 'l':  # line
                        p0, p1 = item[1], item[2]
                        x0, y0, x1, y1 = p0.x, p0.y, p1.x, p1.y
                        if _line_filter(x0, y0, x1, y1, axis_aligned_only, min_length_pdf):
                            lines.append((x0, y0, x1, y1))

                    elif include_rects and item[0] == 're':  # rect
                        # 現状、斜めに回転した長方形は強制的に追加されてしまう
                        r = item[1]
                        x0, y0, x1, y1 = r.x0, r.y0, r.x1, r.y1
                        lines.append((x0, y0, x0, y1))
                        lines.append((x0, y1, x1, y1))
                        lines.append((x1, y1, x1, y0))
                        lines.append((x1, y0, x0, y0))

                    elif include_curves and item[0] == 'c':
                        N = len(item) - 1
                        for i in range(1, N - 1):
                            p0, p1 = item[i], item[i + 1]
                            x0, y0, x1, y1 = p0.x, p0.y, p1.x, p1.y
                            if _line_filter(x0, y0, x1, y1, axis_aligned_only, min_length_pdf):
                                lines.append((x0, y0, x1, y1))

                    elif include_rects and item[0] == 'qu':
                        # 現状、斜めに回転した長方形は強制的に追加されてしまう
                        quad = item[1]
                        pts = [quad.ul, quad.ur, quad.lr, quad.ll, quad.ul]
                        for i in range(4):
                            p0, p1 = pts[i], pts[i + 1]
                            x0, y0, x1, y1 = p0.x, p0.y, p1.x, p1.y
                            lines.append((x0, y0, x1, y1))

                    else:
                        warning_flag = True

            # 1ページの線分を取得後、PNG画像上への座標変化換を行う
            arr = np.asarray(lines, dtype=np.float32)
            arr = transform_lines_array(arr, page, scale)
            results[p_idx] = arr
    return results


def transform_lines_array(arr, page, scale):
    if len(arr) == 0:
        return arr

    rot = page.rotation
    out = arr.copy()

    x0 = arr[:, 0]
    y0 = arr[:, 1]
    x1 = arr[:, 2]
    y1 = arr[:, 3]

    if rot == 0:
        out[:, :4] = arr[:, :4] * scale

    elif rot == 270:
        w = page.mediabox.width
        out[:, 0] = y0 * scale
        out[:, 1] = (w - x0) * scale
        out[:, 2] = y1 * scale
        out[:, 3] = (w - x1) * scale

    elif rot == 90:
        h = page.mediabox.height
        out[:, 0] = (h - y0) * scale
        out[:, 1] = x0 * scale
        out[:, 2] = (h - y1) * scale
        out[:, 3] = x1 * scale

    elif rot == 180:
        w = page.mediabox.width
        h = page.mediabox.height
        out[:, 0] = (w - x0) * scale
        out[:, 1] = (h - y0) * scale
        out[:, 2] = (w - x1) * scale
        out[:, 3] = (h - y1) * scale

    return out


def _line_filter(
    x0, y0, x1, y1,
    axis_aligned_only=False,
    min_length=None,
    axis_eps: float = 1e-3
):
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    length = max(dx, dy)

    if min_length is not None and length < min_length:
        return False

    if axis_aligned_only:
        is_axis = (dx <= axis_eps) or (dy <= axis_eps)
        if not is_axis:
            return False

    return True
