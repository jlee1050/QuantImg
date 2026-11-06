"""Redraw the deck's illustrations in the QuantImg app palette and swap them
into the styled .pptx (replacing the dark-theme circuit diagrams, Bloch sphere,
and flow arrows). Keeps each image's original pixel dimensions so placement and
aspect ratio are preserved.

Usage:  python scripts/redraw_assets.py DECK.pptx
"""

import io
import sys
import zipfile

from PIL import Image, ImageDraw, ImageFont

S = 3  # supersample factor

CREAM = (248, 241, 229, 255)
BEIGE = (233, 228, 210, 255)
PANEL = (211, 204, 191, 255)
INK = (36, 36, 36, 255)
MUTED = (94, 85, 76, 255)
ACCENT = (122, 74, 69, 255)
ACCENTD = (94, 56, 51, 255)
MERID = (169, 120, 110, 255)
# muted, warm-compatible channel colours
R_W = (183, 90, 74, 255)
G_W = (120, 140, 92, 255)
B_W = (110, 126, 151, 255)
POS_W = (96, 88, 78, 255)
WIRES = [("R", R_W), ("G", G_W), ("B", B_W), ("X", POS_W), ("Y", POS_W)]

FONT = "C:/Windows/Fonts/consolab.ttf"


def font(px):
    return ImageFont.truetype(FONT, int(px))


def _ctext(d, xy, s, f, fill):
    b = d.textbbox((0, 0), s, font=f)
    d.text((xy[0] - (b[2] - b[0]) / 2, xy[1] - (b[3] - b[1]) / 2 - b[1]),
           s, font=f, fill=fill)


def draw_circuit(w, h, gates, cnots, path):
    W, H = w * S, h * S
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    x0, x1 = 0.14 * W, 0.97 * W
    ys = [(0.13 + 0.185 * i) * H for i in range(5)]
    wire_w = max(2, int(0.006 * H))
    lab = font(0.072 * H)
    gfont = font(0.085 * H)
    chip = 0.14 * H
    rad = 0.028 * H

    for i, (name, col) in enumerate(WIRES):
        d.line([(x0, ys[i]), (x1, ys[i])], fill=col, width=wire_w)
        _ctext(d, (0.065 * W, ys[i]), name, lab, INK)

    for col_frac, wire, letter in gates:
        cx, cy = col_frac * W, ys[wire]
        box = [cx - chip / 2, cy - chip / 2, cx + chip / 2, cy + chip / 2]
        d.rounded_rectangle(box, radius=rad, fill=CREAM, outline=ACCENT,
                            width=max(2, int(0.007 * H)))
        _ctext(d, (cx, cy), letter, gfont, INK)

    for col_frac, ctrl, tgt in cnots:
        cx = col_frac * W
        d.line([(cx, ys[ctrl]), (cx, ys[tgt])], fill=ACCENT,
               width=max(2, int(0.006 * H)))
        cr = 0.022 * H
        d.ellipse([cx - cr, ys[ctrl] - cr, cx + cr, ys[ctrl] + cr], fill=ACCENT)
        tr = 0.055 * H
        d.ellipse([cx - tr, ys[tgt] - tr, cx + tr, ys[tgt] + tr],
                  fill=CREAM, outline=ACCENT, width=max(2, int(0.008 * H)))
        d.line([(cx - tr, ys[tgt]), (cx + tr, ys[tgt])], fill=ACCENT,
               width=max(2, int(0.006 * H)))
        d.line([(cx, ys[tgt] - tr), (cx, ys[tgt] + tr)], fill=ACCENT,
               width=max(2, int(0.006 * H)))

    img.resize((w, h), Image.LANCZOS).save(path, format="PNG")


def draw_bloch(n, path):
    N = n * S
    img = Image.new("RGBA", (N, N), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    cx, cy, r = N / 2, N / 2, 0.38 * N
    lw = max(2, int(0.006 * N))
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(233, 228, 210, 90),
              outline=INK, width=lw)
    d.ellipse([cx - r, cy - 0.33 * r, cx + r, cy + 0.33 * r],
              outline=ACCENT, width=lw)                       # equator
    d.ellipse([cx - 0.34 * r, cy - r, cx + 0.34 * r, cy + r],
              outline=MERID, width=lw)                        # meridian
    # dashed axis
    y = cy - r
    while y < cy + r:
        d.line([(cx, y), (cx, min(y + 0.04 * N, cy + r))],
               fill=(94, 85, 76, 150), width=lw)
        y += 0.075 * N
    f = font(0.082 * N)
    _ctext(d, (cx, cy - r - 0.075 * N), "0", f, INK)
    _ctext(d, (cx, cy + r + 0.075 * N), "1", f, INK)
    # state vector up-right
    import math
    ang = math.radians(32)
    tx, ty = cx + r * 0.66 * math.cos(ang), cy - r * 0.66 * math.sin(ang)
    d.line([(cx, cy), (tx, ty)], fill=ACCENT, width=max(3, int(0.011 * N)))
    dr = 0.035 * N
    d.ellipse([tx - dr, ty - dr, tx + dr, ty + dr], fill=ACCENT)
    img.resize((n, n), Image.LANCZOS).save(path, format="PNG")


def draw_arrow(w, h, path):
    W, H = w * S, h * S
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.polygon([(0.05 * W, 0.33 * H), (0.55 * W, 0.33 * H), (0.55 * W, 0.12 * H),
               (0.95 * W, 0.5 * H), (0.55 * W, 0.88 * H), (0.55 * W, 0.67 * H),
               (0.05 * W, 0.67 * H)], fill=ACCENT)
    img.resize((w, h), Image.LANCZOS).save(path, format="PNG")


def main(deck):
    assets = {}

    buf = io.BytesIO()
    draw_circuit(1520, 1440,
                 gates=[(0.30, 0, "H"), (0.30, 2, "Z"), (0.30, 3, "H")],
                 cnots=[(0.57, 3, 0), (0.78, 1, 2)], path=buf)
    assets["ppt/media/image-1-1.png"] = buf.getvalue()

    buf = io.BytesIO()
    draw_circuit(1240, 680,
                 gates=[(0.26, 0, "H"), (0.26, 3, "H"), (0.80, 2, "Z")],
                 cnots=[(0.52, 3, 0)], path=buf)
    assets["ppt/media/image-7-1.png"] = buf.getvalue()

    buf = io.BytesIO()
    draw_bloch(1120, buf)
    assets["ppt/media/image-5-1.png"] = buf.getvalue()

    for name in ("image-12-1.png", "image-12-2.png", "image-12-3.png"):
        buf = io.BytesIO()
        draw_arrow(112, 80, buf)
        assets[f"ppt/media/{name}"] = buf.getvalue()

    # rewrite the zip, replacing just those media entries
    src = zipfile.ZipFile(deck, "r")
    tmp = deck + ".tmp"
    out = zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED)
    for item in src.infolist():
        data = assets.get(item.filename, src.read(item.filename))
        out.writestr(item, data)
    src.close()
    out.close()
    import os
    os.replace(tmp, deck)
    print(f"redrew {len(assets)} assets in {deck}")


if __name__ == "__main__":
    main(sys.argv[1])
