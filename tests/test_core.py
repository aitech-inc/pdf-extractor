import os
from pdf_extractor.core import get_images_from_pdf, get_lines_from_pdf


def test_extraction():
    # テスト用PDFのパス（実行ディレクトリからの相対パス）
    pdf_path = os.path.join(os.path.dirname(__file__), "data/sample.pdf")

    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    # 画像抽出のテスト（RGB）
    images = get_images_from_pdf(pdf_bytes, dpi=72, mode="RGB")
    print(f"Page 0 image shape: {images[0].shape}")  # (H, W, 3)のはず

    # 画像抽出のテスト（グレースケール）
    images_gray = get_images_from_pdf(pdf_bytes, dpi=72, mode="L")
    print(f"Page 0 gray shape: {images_gray[0].shape}")  # (H, W)のはず

    # 線分抽出のテスト
    lines = get_lines_from_pdf(pdf_bytes)
    print(f"Found {len(lines[0])} lines on page 0")


if __name__ == "__main__":
    test_extraction()