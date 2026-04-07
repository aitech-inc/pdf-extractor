# pdf-extractor

PDFから線分やテキストなどの情報を抽出するための汎用ライブラリです。

---

## インストール方法

このリポジトリを直接 `pip` または `uv` でインストールして使用します。

### pip の場合
```bash
pip install git+https://github.com/aitech-inc/pdf-extractor.git
```
最新のコミットを反映させたいとき
```bash
pip install --upgrade --force-reinstall git+https://github.com/aitech-inc/pdf-extractor.git
```

### uv の場合
```bash
uv add git+https://github.com/aitech-inc/pdf-extractor.git
```

## 使い方
```python
import io
from pdf_extractor.core import get_lines_from_pdf

# 1. PDFファイルをバイナリ(bytes)として読み込む
with open("sample.pdf", "rb") as f:
    pdf_bytes = f.read()

# 2. 線分情報を取得
# dpi: 出力座標の解像度（デフォルト400）
# page_numbers: 抽出したいページのリスト。Noneの場合は全ページ。
lines_dict = get_lines_from_pdf(pdf_bytes, dpi=400, page_numbers=[0])

# 3. 結果の利用
# 戻り値は { ページ番号: numpy配列(N, 4) } の辞書形式です。
# 配列の中身は [x0, y0, x1, y1] です。
for page_idx, lines in lines_dict.items():
    print(f"Page {page_idx}: {len(lines)} lines found.")
    if len(lines) > 0:
        print(f"First line coordinates: {lines[0]}")
```
