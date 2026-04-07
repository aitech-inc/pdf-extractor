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
            all_edges = _get_edge(lines, rects, curves, dpi)
            results[p_idx] = all_edges

    return results


def _get_edge(lines, rects, curves, dpi):
    # numpyの空配列処理を考慮して、中身がない場合のハンドリングを入れるとより安全
    l_arr = _get_lines(lines, dpi)
    r_arr = _get_rect_edge_lines(rects, dpi)
    c_arr = _get_curve_edge_lines(curves, dpi)

    arrays = [arr for arr in (l_arr, r_arr, c_arr) if len(arr) > 0]

    if not arrays:
        return np.empty((0, 4))

    return np.concatenate(arrays, axis=0)


def _get_lines(raw_lines, dpi):
    lines = np.empty((len(raw_lines), 4))
    scale = dpi / 72
    for i, line in enumerate(raw_lines):
        (x0, y0), (x1, y1) = line['pts']
        x0, y0 = x0 * scale, y0 * scale
        x1, y1 = x1 * scale, y1 * scale
        lines[i] = [x0, y0, x1, y1]
    return lines

def _get_rect_edge_lines(raw_rects, dpi):
    """
    pts [(220.7999955, 123.96000225), (226.44000975, 123.96000225), (226.44000975, 171.24000824999996), (220.7999955, 171.24000824999996)]
    """
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
            rect_edge_lines.append([x0, y0, x1, y1])
    return np.array(rect_edge_lines)


def _get_curve_edge_lines(raw_curve_edge_lines, dpi):
    scale = dpi / 72
    curve_edge_lines = []
    for i, line in enumerate(raw_curve_edge_lines):
        pts = line['pts']
        for i in range(len(pts) - 1):
            x0, y0 = pts[i]
            x1, y1 = pts[i + 1]
            x0, y0 = x0 * scale, y0 * scale
            x1, y1 = x1 * scale, y1 * scale
            curve_edge_lines.append([x0, y0, x1, y1])
    return np.array(curve_edge_lines)


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