import fitz, pdfplumber
import io
import numpy as np
from typing import List, Dict, Optional, Literal

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


def get_texts_from_pdf(
    pdf_bytes: bytes,
    dpi: int = 400,
    page_numbers: Optional[List[int]] = None,
):
    scale = dpi / 72.0
    text_json = {}

    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        target_pages = page_numbers if page_numbers is not None else range(len(doc))

        for page_num in target_pages:
            if page_num >= len(doc):
                continue

            page = doc[page_num]
            page_rotation = page.rotation

            rect = page.rect
            page_width = rect.width * scale
            page_height = rect.height * scale

            infos = []
            blocks = page.get_text("dict")["blocks"]
            for b in blocks:
                for line in b.get("lines", []):
                    # text = line['spans'][0]['text']
                    text = "".join(
                        span.get("text", "")
                        for span in line.get("spans", [])
                    ).strip()
                    bbox = line['bbox']
                    x0, y0, x1, y1 = _rotate_bbox(
                        bbox[0] * scale, bbox[1] * scale,
                        bbox[2] * scale, bbox[3] * scale,
                        page_width, page_height, page_rotation
                    )

                    dx, dy = line['dir']

                    is_rotated = abs(dy) > abs(dx)
                    if page_rotation in (90, 270):
                        is_rotated = not is_rotated

                    infos.append({
                        'x0': x0, 'y0': y0,
                        'x1': x1, 'y1': y1,
                        "cx": (float(x0) + float(x1)) / 2,
                        "cy": (float(y0) + float(y1)) / 2,
                        "text": text,
                        'is_rotated': is_rotated
                    })
            text_json[page_num] = infos
    return text_json




def _rotate_bbox(x0, y0, x1, y1, page_width, page_height, rotation):
    if rotation == 0:
        return x0, y0, x1, y1

    elif rotation == 90:
        return (
            page_height - y1,
            x0,
            page_height - y0,
            x1,
        )

    elif rotation == 180:
        return (
            page_width - x1,
            page_height - y1,
            page_width - x0,
            page_height - y0,
        )

    elif rotation == 270:
        return (
            y0,
            page_width - x1,
            y1,
            page_width - x0,
        )

    return x0, y0, x1, y1

def get_pdfbytes(pdf_path):
    with open(pdf_path, 'rb') as f:
        pdf_bytes = f.read()
    return pdf_bytes