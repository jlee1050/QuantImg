"""Restyle the hackathon pitch deck to match the QuantImg app's visual identity.

The deck ships in a dark teal/pink "quantum" theme; the app is a warm light
theme (cream background with faint accent polka-dots, brown-red accent, beige
panels, Rubik / IBM Plex / Caveat fonts). This script maps one onto the other.

Usage:  python scripts/restyle_deck.py INPUT.pptx OUTPUT.pptx
"""

import sys

from PIL import Image, ImageDraw
from pptx import Presentation
from pptx.oxml.ns import qn
from pptx.util import Emu

# --- app palette ---------------------------------------------------------- #
CREAM = "F8F1E5"
BEIGE = "E9E4D2"
PANEL = "D3CCBF"
INK = "242424"
MUTED = "5E554C"
ACCENT = "7A4A45"
ACCENTD = "5E3833"

# --- app fonts ------------------------------------------------------------- #
FONT_MAP = {
    "Space Grotesk": "Rubik",
    "Inter": "IBM Plex Sans Condensed",
    "JetBrains Mono": "IBM Plex Mono",
}
DISPLAY = "Space Grotesk"   # original display face (pre-remap)
MONO_HINT = "Mono"

# --- original colour buckets ---------------------------------------------- #
DARK_FILLS = {"15131F", "1F1B2E"}
LIGHTS = {"F3EFE7", "FFFFFF", "D8D2E6", "A9A3BD", "BCB8B5"}
DARKS = {"15131F", "2B2933"}
MUTED_SET = {"3D3950", "5B5670"}
ACC_SET = {"5AD4C8", "FF5C8A", "C2305E"}
EYEBROW = {"A9A3BD"}          # dim mono labels

ACCENT_BG_SLIDES = {2, 8}     # hero statement + closing -> full accent bg


# --------------------------------------------------------------------------- #
def make_dot_png(path, w=2000, h=1125):
    """Cream canvas with faint accent polka-dots, like the app background."""
    img = Image.new("RGB", (w, h), f"#{CREAM}")
    dots = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(dots)
    r, step = 3, 46
    ar, ag, ab = (int(ACCENT[i:i + 2], 16) for i in (0, 2, 4))
    for y in range(step // 2, h, step):
        for x in range(step // 2, w, step):
            d.ellipse([x - r, y - r, x + r, y + r], fill=(ar, ag, ab, 34))
    img = Image.alpha_composite(img.convert("RGBA"), dots).convert("RGB")
    img.save(path)


def set_bg(slide, hex_color):
    bg = slide.element.find(qn("p:cSld") + "/" + qn("p:bg"))
    if bg is None:
        return
    clr = bg.find(".//" + qn("a:srgbClr"))
    if clr is not None:
        clr.set("val", hex_color)


def add_back_picture(slide, prs, path):
    pic = slide.shapes.add_picture(path, 0, 0, prs.slide_width, prs.slide_height)
    sp = pic._element
    tree = sp.getparent()
    tree.remove(sp)
    tree.insert(2, sp)            # after nvGrpSpPr + grpSpPr -> behind everything


def rgb_of(shape):
    try:
        if shape.fill.type == 1:
            return str(shape.fill.fore_color.rgb)
    except Exception:
        pass
    return None


def center_in(boxes, shape):
    try:
        cx = shape.left + shape.width / 2
        cy = shape.top + shape.height / 2
    except TypeError:
        return False
    return any(l <= cx <= l + w and t <= cy <= t + h for l, t, w, h in boxes)


def fill_color(h, in_card):
    if h in DARK_FILLS:
        return ACCENT
    if h == "FFFFFF":
        return BEIGE
    if h in MUTED_SET:
        return ACCENTD if in_card else PANEL
    if h == "C2305E":
        return ACCENTD
    if h in {"5AD4C8", "FF5C8A"}:
        return CREAM if in_card else ACCENT
    if h == "F3EFE7":
        return CREAM
    return None


def text_color(orig, font, on_acc):
    is_disp = font == DISPLAY
    is_mono = MONO_HINT in (font or "")
    if on_acc:
        if orig in ACC_SET or orig in EYEBROW:
            return BEIGE
        return CREAM
    if orig in ACC_SET:
        return ACCENT
    if is_mono:
        return ACCENT if (orig in ACC_SET or orig in EYEBROW or orig is None) else INK
    if is_disp:
        return INK
    if orig in MUTED_SET:
        return MUTED
    if orig in LIGHTS:
        return MUTED if font == "Inter" else INK
    if orig in DARKS:
        return INK
    return MUTED if font == "Inter" else INK


def restyle_runs(shape, on_acc, force=None):
    if not shape.has_text_frame:
        return
    for para in shape.text_frame.paragraphs:
        for run in para.runs:
            font = run.font
            origfont = font.name
            orig = None
            try:
                if font.color is not None and font.color.type is not None:
                    orig = str(font.color.rgb)
            except Exception:
                orig = None
            font.color.rgb = _rgb(force or text_color(orig, origfont, on_acc))
            if origfont in FONT_MAP:
                font.name = FONT_MAP[origfont]


def _rgb(hex_str):
    from pptx.dml.color import RGBColor
    return RGBColor.from_string(hex_str)


def add_shadow(shape):
    """Layered drop shadow on a hero title, echoing the app wordmark."""
    spPr = shape._element.spPr
    for old in spPr.findall(qn("a:effectLst")):
        spPr.remove(old)
    import lxml.etree as ET
    ns = "http://schemas.openxmlformats.org/drawingml/2006/main"
    eff = ET.SubElement(spPr, qn("a:effectLst"))
    shdw = ET.SubElement(eff, qn("a:outerShdw"))
    shdw.set("blurRad", "0")
    shdw.set("dist", "38100")           # ~3pt offset
    shdw.set("dir", "2700000")          # down-right
    shdw.set("rotWithShape", "0")
    clr = ET.SubElement(shdw, qn("a:srgbClr"))
    clr.set("val", ACCENTD)


def update_theme_fonts(prs):
    for master in prs.slide_masters:
        tp = [r.target_part for r in master.part.rels.values()
              if "theme" in r.reltype][0]
        import lxml.etree as ET
        root = ET.fromstring(tp.blob)
        fs = root.find(".//" + qn("a:fontScheme"))
        for tag, face in (("majorFont", "Rubik"),
                          ("minorFont", "IBM Plex Sans Condensed")):
            latin = fs.find(qn("a:" + tag) + "/" + qn("a:latin"))
            if latin is not None:
                latin.set("typeface", face)
        tp._blob = ET.tostring(root, xml_declaration=True,
                               encoding="UTF-8", standalone=True)


def main(inp, outp):
    prs = Presentation(inp)
    update_theme_fonts(prs)
    dot_png = "scripts/_dot_bg.png"
    make_dot_png(dot_png)

    for i, slide in enumerate(prs.slides):
        on_acc_slide = i in ACCENT_BG_SLIDES

        # accent-card boxes (shapes that will become brown-red cards)
        boxes = []
        dark_shapes = []
        for sh in slide.shapes:
            if rgb_of(sh) in DARK_FILLS:
                try:
                    box = (sh.left, sh.top, sh.width, sh.height)
                    boxes.append(box)
                    dark_shapes.append((sh, box))
                except TypeError:
                    pass

        # a small dark shape nested inside a larger dark card is a "chip"
        # (e.g. gate buttons on the GATES panel) -> cream chip, accent text.
        chip_ids = set()
        for sh, (l, t, w, h) in dark_shapes:
            cx, cy = l + w / 2, t + h / 2
            for other, (ol, ot, ow, oh) in dark_shapes:
                if other is sh:
                    continue
                if ow * oh > w * h * 1.3 and ol <= cx <= ol + ow and ot <= cy <= ot + oh:
                    chip_ids.add(id(sh._element))
                    break

        # background
        if on_acc_slide:
            set_bg(slide, ACCENT)
        else:
            set_bg(slide, CREAM)
            add_back_picture(slide, prs, dot_png)

        # shapes: fills + text
        for sh in slide.shapes:
            h = rgb_of(sh)
            in_card = center_in(boxes, sh)
            is_chip = id(sh._element) in chip_ids
            if h is not None:
                new = CREAM if is_chip else fill_color(h, in_card)
                if new is not None:
                    sh.fill.solid()
                    sh.fill.fore_color.rgb = _rgb(new)
            if is_chip:
                restyle_runs(sh, False, force=ACCENT)
            else:
                on_acc = on_acc_slide or (h in DARK_FILLS) or in_card
                restyle_runs(sh, on_acc)

    # hero title flourishes
    _hero_title(prs)
    prs.save(outp)
    print(f"wrote {outp}")


def _hero_title(prs):
    # slide 0 wordmark -> accent + shadow + Caveat subtitle
    s0 = prs.slides[0]
    for sh in s0.shapes:
        if sh.has_text_frame and sh.text_frame.text.strip() == "quantimg":
            for p in sh.text_frame.paragraphs:
                for r in p.runs:
                    r.font.color.rgb = _rgb(ACCENT)
                    r.font.name = "Rubik"
            add_shadow(sh)
        if sh.has_text_frame and sh.text_frame.text.startswith("A drag-and-drop"):
            for p in sh.text_frame.paragraphs:
                for r in p.runs:
                    r.font.name = "Caveat"
                    r.font.color.rgb = _rgb(ACCENT)
    # closing wordmark shadow
    for sh in prs.slides[8].shapes:
        if sh.has_text_frame and sh.text_frame.text.strip() == "QUANTIMG":
            add_shadow(sh)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
