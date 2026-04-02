import io
import numpy as np
from PIL import Image


def ndarray_to_png_bytes(arr: np.ndarray, normalize: bool = True) -> bytes:
    arr = np.asarray(arr)

    if arr.ndim == 2:
        if normalize:
            arr = arr.astype(np.float32)
            mn, mx = arr.min(), arr.max()
            if mx == mn:
                img = np.zeros_like(arr, dtype=np.uint8)
            else:
                img = ((arr - mn) / (mx - mn) * 255).astype(np.uint8)
        else:
            img = arr.astype(np.uint8)

        pil_img = Image.fromarray(img, mode="L")

    elif arr.ndim == 3 and arr.shape[2] in (3, 4):
        img = arr.astype(np.uint8)
        pil_img = Image.fromarray(img)

    else:
        raise ValueError(f"Unsupported array shape: {arr.shape}")

    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    return buffer.getvalue()