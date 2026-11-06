"""Render every preset on a colourful test image -> samples/*.png.

Usage:  python scripts/generate_samples.py [input.png]

With no argument it builds a synthetic hue/brightness test card. Use this to
eyeball gate combinations and pick demo presets (handoff step 5).
"""

import colorsys
import os
import sys

import numpy as np
from PIL import Image

from quantimg import presets
from quantimg.runner import run_spec

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "samples")


def test_card(h=160, w=240):
    """Hue sweep across x, brightness ramp down y, plus a white disc."""
    img = np.zeros((h, w, 3))
    for y in range(h):
        for x in range(w):
            img[y, x] = colorsys.hsv_to_rgb(x / w, 0.85, 0.35 + 0.6 * (1 - y / h))
    yy, xx = np.mgrid[0:h, 0:w]
    img[((xx - w * 0.7) ** 2 + (yy - h * 0.45) ** 2) < 32 ** 2] = [0.98, 0.98, 0.98]
    return (img * 255).astype(np.uint8)


def main():
    if len(sys.argv) > 1:
        img = np.asarray(Image.open(sys.argv[1]).convert("RGB"))
    else:
        img = test_card()

    os.makedirs(OUT_DIR, exist_ok=True)
    Image.fromarray(img).save(os.path.join(OUT_DIR, "_input.png"))
    for name in presets.PRESETS:
        out = run_spec(img, presets.get_preset(name))
        Image.fromarray(out).save(os.path.join(OUT_DIR, f"{name}.png"))
        print(f"  {name}")
    print(f"wrote {len(presets.PRESETS) + 1} images to {os.path.normpath(OUT_DIR)}")


if __name__ == "__main__":
    main()
