import io
import pdfplumber
import numpy as np
from typing import Optional, List, Dict


def get_lines_from_pdf(
    pdf_bytes: bytes,
    dpi: int = 400,
    page_numbers: Optional[List[int]] = None
) -> Dict[int, np.ndarray]:
    """
    PDFから線分情報を取得する。

    :param pdf_bytes: PDFファイルの中身（バイト列）
    :param dpi: 解析時の解像度
    :param page_numbers: 抽出したいページのリスト（例: [0, 1]）。Noneの場合は全ページ。
    :return: ページ番号をキー、線分座標 [x0, y0, x1, y1] の numpy配列を値とした辞書
    """
    results = {}

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        # 指定がなければ全ページ、指定があればそのページのみ対象にする
        target_pages = page_numbers if page_numbers is not None else range(len(pdf.pages))

        for p_idx in target_pages:
            page = pdf.pages[p_idx]
            lines = page.objects.get("line", [])
            curves = page.objects.get("curve", [])

            # 線分と曲線エッジを統合
            all_edges = _get_edge(lines, curves, dpi)
            results[p_idx] = all_edges

    return results


def _get_edge(lines, curves, dpi):
    # numpyの空配列処理を考慮して、中身がない場合のハンドリングを入れるとより安全
    l_arr = _get_lines(lines, dpi)
    c_arr = _get_curve_edge_lines(curves, dpi)

    # どちらかが空でも動くように concatenate する
    if len(l_arr) == 0 and len(c_arr) == 0:
        return np.empty((0, 4))
    if len(l_arr) == 0: return c_arr
    if len(c_arr) == 0: return l_arr

    return np.concatenate([l_arr, c_arr], axis=0)

def _get_lines(raw_lines, dpi):
    lines = np.empty((len(raw_lines), 4))
    scale = dpi / 72
    for i, line in enumerate(raw_lines):
        (x0, y0), (x1, y1) = line['pts']
        x0, y0 = x0 * scale, y0 * scale
        x1, y1 = x1 * scale, y1 * scale
        lines[i] = [x0, y0, x1, y1]
    return lines

def _get_curve_edge_lines(raw_curve_edge_lines, dpi):
    scale = dpi / 72
    curve_edge_lines = []
    for i, line in enumerate(raw_curve_edge_lines):
        pts = line['pts']
        for i in range(len(pts) - 1):
            x0, y0 = pts[i]
            x1, y1 = pts[i+1]
            x0, y0 = x0 * scale, y0 * scale
            x1, y1 = x1 * scale, y1 * scale
            curve_edge_lines.append([x0, y0, x1, y1])
    return np.array(curve_edge_lines)