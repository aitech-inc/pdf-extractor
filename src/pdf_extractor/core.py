import io
import pdfplumber, fitz
import numpy as np
from typing import Optional, List, Dict, Literal

"""
pdfから直接抽出する系の処理
・get_lines_from_pdf
・get_images_from_pdf
"""


def get_lines_from_pdf(
    pdf_bytes: bytes,
    dpi: int = 400,
    page_numbers: Optional[List[int]] = None,
    axis_aligned_only: bool = False,
    min_length: Optional[float] = None
) -> Dict[int, np.ndarray]:
    """
    PDFから線分情報を取得する。

    :param pdf_bytes: PDFファイルの中身（バイト列）
    :param dpi: 解析時の解像度
    :param page_numbers: 抽出したいページのリスト（例: [0, 1]）。Noneの場合は全ページ。
    :param axis_aligned_only: Trueの場合、水平・垂直線のみ抽出する
    :param min_length: 指定した長さ未満の線分を除外する。Noneの場合は除外しない。
    :return: ページ番号をキー、線分座標 [x0, y0, x1, y1] の numpy配列を値とした辞書
    """
    results = {}

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        total_pages = len(pdf.pages)
        # 指定がなければ全ページ、指定があればそのページのみ対象にする
        target_pages = page_numbers if page_numbers is not None else range(total_pages)

        for p_idx in target_pages:
            if p_idx >= total_pages:
                continue
            page = pdf.pages[p_idx]
            lines = page.objects.get("line", [])
            rects = page.objects.get("rect", [])
            curves = page.objects.get("curve", [])

            # 線分と曲線エッジを統合
            all_edges = _get_edge(
                lines,
                rects,
                curves,
                dpi,
                axis_aligned_only,
                min_length
            )
            results[p_idx] = all_edges

    return results


def _get_edge(lines, rects, curves, dpi, axis_aligned_only, min_length):
    l_arr = _get_lines(lines, dpi, axis_aligned_only, min_length)
    r_arr = _get_rect_edge_lines(rects, dpi, axis_aligned_only, min_length)
    c_arr = _get_curve_edge_lines(curves, dpi, axis_aligned_only, min_length)

    arrays = [arr for arr in (l_arr, r_arr, c_arr) if len(arr) > 0]

    if not arrays:
        return np.empty((0, 4), dtype=np.float32)

    return np.concatenate(arrays, axis=0)


def _get_lines(raw_lines, dpi, axis_aligned_only, min_length):
    scale = dpi / 72
    lines = []

    for line in raw_lines:
        (x0, y0), (x1, y1) = line["pts"]

        x0, y0 = x0 * scale, y0 * scale
        x1, y1 = x1 * scale, y1 * scale

        if _line_filter(
            x0, y0, x1, y1,
            axis_aligned_only=axis_aligned_only,
            min_length=min_length,
        ):
            lines.append([x0, y0, x1, y1])

    if not lines:
        return np.empty((0, 4), dtype=np.float32)

    return np.asarray(lines, dtype=np.float32)


def _get_rect_edge_lines(raw_rects, dpi, axis_aligned_only, min_length):
    scale = dpi / 72
    rect_edge_lines = []
    for rect in raw_rects:
        pts = rect['pts']
        N = len(pts)
        for i in range(N):
            x0, y0 = pts[i]
            x1, y1 = pts[(i + 1) % N]  # 最後の点は最初の点とつなげる
            x0, y0 = x0 * scale, y0 * scale
            x1, y1 = x1 * scale, y1 * scale
            if _line_filter(
                x0, y0, x1, y1,
                axis_aligned_only=axis_aligned_only,
                min_length=min_length
            ):
                rect_edge_lines.append([x0, y0, x1, y1])

    if not rect_edge_lines:
        return np.empty((0, 4), dtype=np.float32)

    return np.asarray(rect_edge_lines, dtype=np.float32)


def _get_curve_edge_lines(raw_curve_edge_lines, dpi, axis_aligned_only, min_length):
    scale = dpi / 72
    curve_edge_lines = []
    for line in raw_curve_edge_lines:
        pts = line['pts']
        for i in range(len(pts) - 1):
            x0, y0 = pts[i]
            x1, y1 = pts[i + 1]
            x0, y0 = x0 * scale, y0 * scale
            x1, y1 = x1 * scale, y1 * scale
            if _line_filter(
                x0, y0, x1, y1,
                axis_aligned_only=axis_aligned_only,
                min_length=min_length
            ):
                curve_edge_lines.append([x0, y0, x1, y1])

    if not curve_edge_lines:
        return np.empty((0, 4), dtype=np.float32)

    return np.asarray(curve_edge_lines, dtype=np.float32)


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


def get_images_from_pdf(
    pdf_bytes: bytes,
    dpi: int = 400,
    page_numbers: Optional[List[int]] = None,
    mode: Literal["RGB", "L"] = "RGB"  # "RGB"はカラー、"L"はグレースケール
) -> Dict[int, np.ndarray]:
    results = {}

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        total_pages = len(pdf.pages)
        target_pages = page_numbers if page_numbers is not None else range(total_pages)

        for p_idx in target_pages:
            if p_idx >= total_pages:
                continue

            page = pdf.pages[p_idx]

            pil_img = page.to_image(resolution=dpi).original

            # モード変換（"L" を指定するとグレースケールに変換される）
            if pil_img.mode != mode:
                pil_img = pil_img.convert(mode)

            # numpy配列に変換
            im_np = np.array(pil_img)
            results[p_idx] = im_np

    return results


def get_text_from_pdf(
    pdf_bytes: bytes,
    dpi: int = 400,
    page_numbers: Optional[List[int]] = None
) -> Dict[int, List[dict]]:
    """
    PDFからテキスト情報（座標付き）を抽出します。
    """
    scale = dpi / 72
    results = {}

    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        # 指定がなければ全ページ、あればそのページのみ対象
        target_pages = page_numbers if page_numbers is not None else range(len(doc))

        for p_idx in target_pages:
            if p_idx >= len(doc):
                continue

            page = doc[p_idx]
            rect = page.rect
            width = rect.width
            words = page.get_text("words")  # (x0, y0, x1, y1, "word", block_no, line_no, word_no)

            page_text_infos = []
            for w in words:
                # 元のコードの計算式を維持（座標変換ロジック）
                x0 = (width - w[3]) * scale
                x1 = (width - w[1]) * scale
                y0 = w[0] * scale
                y1 = w[2] * scale

                page_text_infos.append({
                    'x0': x0, 'y0': y0,
                    'x1': x1, 'y1': y1,
                    "cx": (float(x0) + float(x1)) / 2,
                    "cy": (float(y0) + float(y1)) / 2,
                    "text": str(w[4]),
                })

            results[p_idx] = page_text_infos

    return results
