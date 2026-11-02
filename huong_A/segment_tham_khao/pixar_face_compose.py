"""Stage 2+3 of pixar-seg3: build the face base, SAVE the parts, then assemble the face from the saved files only.

  python pixar_face_compose.py --img <portrait dir> --seg <root with parts/<id>/*.png and lab/<id>.png> --out <out root> [--mix N]
Face base = the whole skin region (face oval + hair-closed head + ears + neck from the SegFormer class map). It keeps the original pixels except the
removed parts (hair, hat, glasses, brows, eyes, iris, lashes, nose, mouth; dilated); those holes get a smooth fill by multi-scale normalised convolution
(Pixar skin is smooth, so the diffusion keeps the cheek shading and never redraws features). Extra part from the class map: hat (glasses are only cut out of the base; no cloth, no neck).
Saved parts (<seg>/parts_soft_crop/<id>/): the base `face.png` and every part, each cropped to its box with a feathered ring of its own original pixels
(HALO px around the mask, so the blank base needs no matching shading) + `layout.json` {"size": [w, h], "parts": {name: [x0, y0]}}.
Assembly (assemble_from_disk) reads ONLY those files: face, mouth, nose, eyes, iris, lashes, brows, hair, hat pasted at their offsets.
Mix: every feature of donor B (tight mask) moves onto the matching feature of A (box centre + width ratio), A keeps hair / hat.
Writes <seg>/parts/<id>/{face,hat}.png, <seg>/parts_soft_crop/<id>/*, <out>/compose/<id>.png (RGBA assembled from disk),
<out>/compose_full/<id>.png (assembled person over a diffused background plate), <out>/mix/<A>_with_<B>.png, <out>/compose.csv.
"""
import argparse, csv, glob, json, os

import cv2, numpy as np
from PIL import Image

PNG_PARTS = ("mouth", "nose", "eyes", "iris_left", "iris_right", "lashes_left", "lashes_right", "brows", "hair")  # saved by the SAM 2 stage
EXTRA = {"glasses": (3,), "hat": (14,), "earring": (15,)}  # from the SegFormer class map
HOLE = ("hair", "brows", "eyes", "nose", "mouth", "lashes_left", "lashes_right", "iris_left", "iris_right", "glasses", "hat")  # removed from the base
BELOW = ()  # nothing under the base (no cloth)
ORDER = ("mouth", "nose", "eyes", "iris_left", "iris_right", "lashes_left", "lashes_right", "brows", "earring", "hair", "glasses", "hat")  # bottom -> top; glasses are an accessory part: cut out of the base, pasted back on top of eyes / brows / hair
DONOR = tuple(p for p in ORDER if p not in ("hair", "hat"))
GROUP = {"iris_left": "eyes", "iris_right": "eyes", "lashes_left": "eyes", "lashes_right": "eyes", "glasses": "eyes"}  # these move with the eye opening
SKIN = (1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12)  # skin, nose, eyes, brows, ears, mouth, lips (no neck, no necklace; earrings are their own part)
HALO, FEATHER = 15, 4  # each saved part carries the original pixels within HALO px of its mask (the base is blank that far out), fading over FEATHER px

ap = argparse.ArgumentParser()
ap.add_argument("--img"); ap.add_argument("--seg"); ap.add_argument("--out"); ap.add_argument("--mix", type=int, default=8)
ap.add_argument("--faithful_base", action="store_true", help="base = the photo itself everywhere except under the parts and the glasses (the parts hide that area when pasted back): the assembly reproduces the photo")
ap.add_argument("--clean_base", action="store_true", help="base = smooth skin only (diffusion from all the skin outside the parts, then blurred): parts are pasted on it as they are, no halos")
args = ap.parse_args()
for d in ("compose", "compose_full", "mix"):
    os.makedirs(f"{args.out}/{d}", exist_ok=True)


def load(pid, part, size):
    f = f"{args.seg}/parts/{pid}/{part}.png"
    return np.array(Image.open(f).resize(size, Image.NEAREST)) if os.path.exists(f) else None


def lab_layer(rgb, lab, ids, min_px=150):
    """RGBA layer (original pixels) of the given class ids, small specks dropped"""
    m = np.isin(lab, ids).astype(np.uint8)
    n, cl, st, _ = cv2.connectedComponentsWithStats(m)
    m = np.isin(cl, [i for i in range(1, n) if st[i, 4] >= min_px])
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) > 0
    return np.dstack([rgb, m * 255]).astype(np.uint8) if m.sum() >= 30 else None


def head_region(oval, hair):
    """oval + hair closed together (skin between them), limited to the neighbourhood of the oval and kept inside the hair's outer edge"""
    u = (oval | (cv2.erode(hair.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0)).astype(np.uint8)
    u = cv2.morphologyEx(u, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41)))
    mid = int(np.nonzero(oval)[0].mean())
    pad = 130  # empty rows above the image so the closing cannot bridge from the image edge to the hair
    tall = cv2.morphologyEx(cv2.copyMakeBorder(u, pad, 0, 0, 0, cv2.BORDER_CONSTANT, value=0), cv2.MORPH_CLOSE, np.ones((121, 1), np.uint8))[pad:]  # forehead between the hair edge and the top of the oval can be wider than 41 px
    tall[mid:] = u[mid:]  # only above the middle of the face: below it a vertical bridge would paint the neck / background
    gap = (tall > 0) & (u == 0)  # the bridged forehead is exempt from the "near" limit below (it can be 70 px tall)
    u = tall
    cs, _ = cv2.findContours(u, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    head = np.zeros_like(u); cv2.drawContours(head, cs, -1, 1, -1)
    near = cv2.dilate(oval.astype(np.uint8), np.ones((101, 101), np.uint8))
    return ((head & near) > 0) | gap


def fill_holes(rgb, valid, scales=(3, 6, 12, 24, 48, 96), soft=2):
    """smooth fill of ~valid pixels: small blur scales first so the fill follows the nearby shading, bigger scales for deep holes"""
    w = valid.astype(np.float32); num = rgb.astype(np.float32) * w[..., None]
    est = rgb.astype(np.float32).copy(); done = valid.copy()
    for s in scales:
        d = cv2.GaussianBlur(w, (0, 0), s); n = cv2.GaussianBlur(num, (0, 0), s)
        take = ~done & (d > 0.02)
        est[take] = n[take] / d[take][:, None]; done |= take
    est[~done] = rgb[valid].mean(0) if valid.any() else 128
    hole = ~valid
    soft = cv2.GaussianBlur(est, (0, 0), soft)  # the diffused colours are smooth already; this only removes scale seams
    est[hole] = soft[hole]
    return est.astype(np.uint8)


def over(layers):
    """over-composite RGBA layers (bottom -> top) with premultiplied accumulation: straight colour + real alpha"""
    h, w = layers[0].shape[:2]
    cp = np.zeros((h, w, 3), np.float32); a = np.zeros((h, w, 1), np.float32)
    for L in layers:
        la = L[..., 3:4].astype(np.float32) / 255.0
        cp = L[..., :3].astype(np.float32) * la + cp * (1 - la)
        a = la + a * (1 - la)
    return (cp / np.maximum(a, 1e-4)).clip(0, 255).astype(np.uint8), a[..., 0]


def compose(base, layers):
    """preview flattened on grey (mix images)"""
    c, a = over([layers[p] for p in BELOW if p in layers] + [base] + [layers[p] for p in ORDER if p in layers])
    return (c * a[..., None] + 128 * (1 - a[..., None])).astype(np.uint8)


def soft_layers(layers, region):
    """layers with a feathered ring of their own (original) pixels around the mask, inside `region` only"""
    out = {}
    for p, a in layers.items():
        dist = cv2.distanceTransform((a[..., 3] <= 127).astype(np.uint8), cv2.DIST_L2, 3)
        ring = np.clip((HALO - dist) / FEATHER, 0, 1) * region
        o = a.copy(); o[..., 3] = (np.maximum(a[..., 3].astype(np.float32) / 255.0, ring) * 255).astype(np.uint8)
        out[p] = o
    return out


def save_parts(pid, base, soft):
    """the saved parts: base + every part cropped to its box, plus the offsets to paste them back"""
    d = f"{args.seg}/parts_soft_crop/{pid}"; os.makedirs(d, exist_ok=True)
    parts = {}
    for name, a in [("face", base)] + list(soft.items()):
        ys, xs = np.nonzero(a[..., 3] > 0)
        if len(ys) == 0:
            continue
        x0, y0 = int(xs.min()), int(ys.min())
        Image.fromarray(a[y0:ys.max() + 1, x0:xs.max() + 1]).save(f"{d}/{name}.png"); parts[name] = [x0, y0]
    json.dump({"size": [int(base.shape[1]), int(base.shape[0])], "parts": parts}, open(f"{d}/layout.json", "w"))


def assemble_from_disk(pid):
    """paste the saved parts onto the saved base, reading nothing else"""
    d = f"{args.seg}/parts_soft_crop/{pid}"; lay = json.load(open(f"{d}/layout.json")); w, h = lay["size"]

    def get(name):
        c = np.zeros((h, w, 4), np.uint8); im = np.array(Image.open(f"{d}/{name}.png")); x0, y0 = lay["parts"][name]
        c[y0:y0 + im.shape[0], x0:x0 + im.shape[1]] = im
        return c
    return over([get(p) for p in BELOW + ("face",) + ORDER if p in lay["parts"]])


def box(rgba):
    ys, xs = np.nonzero(rgba[..., 3] > 127); return xs.min(), xs.max(), ys.min(), ys.max()


def transfer(rgba, src, dst):
    """affine move of a layer from feature box src (donor) onto feature box dst (receiver): centres coincide, uniform scale = width ratio"""
    s = float(np.clip((dst[1] - dst[0] + 1) / (src[1] - src[0] + 1), 0.6, 1.6))
    M = np.array([[s, 0, (dst[0] + dst[1]) / 2 - s * (src[0] + src[1]) / 2], [0, s, (dst[2] + dst[3]) / 2 - s * (src[2] + src[3]) / 2]], np.float32)
    return cv2.warpAffine(rgba, M, (rgba.shape[1], rgba.shape[0]), flags=cv2.INTER_LINEAR, borderValue=0)


ids = sorted(os.path.basename(d) for d in glob.glob(f"{args.seg}/parts/*"))
rows, cache = [], {}
for pid in ids:
    img = Image.open(f"{args.img}/{pid}.png").convert("RGB"); rgb = np.array(img); h, w = rgb.shape[:2]
    flat = load(pid, "face_flat", (w, h)); oval = flat[..., 3] > 0
    layers = {p: a for p in PNG_PARTS for a in [load(pid, p, (w, h))] if a is not None and (a[..., 3] > 127).sum() >= 30}
    for miss, have in (("lashes_left", "lashes_right"), ("lashes_right", "lashes_left")):  # v11 found no lashes on one side (< 30 px): mirror the other side about the iris midline
        if miss not in layers and have in layers and "iris_left" in layers and "iris_right" in layers:
            cx = [np.nonzero(layers[k][..., 3] > 127)[1].mean() for k in ("iris_left", "iris_right")]
            M = np.array([[-1, 0, cx[0] + cx[1]], [0, 1, 0]], np.float32)
            layers[miss] = cv2.warpAffine(layers[have], M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
            Image.fromarray(layers[miss]).save(f"{args.seg}/parts/{pid}/{miss}.png")
    labf = f"{args.seg}/lab/{pid}.png"
    lab = np.array(Image.open(labf).resize((w, h), Image.NEAREST)) if os.path.exists(labf) else None
    if lab is not None:
        for name, cls in EXTRA.items():
            lay = lab_layer(rgb, lab, cls, 40 if name == "earring" else 150)  # studs are tiny
            if lay is not None:
                layers[name] = lay
                Image.fromarray(lay).save(f"{args.seg}/parts/{pid}/{name}.png")
    feat = np.zeros((h, w), bool)
    for p in HOLE:
        if p in layers:
            feat |= layers[p][..., 3] > 127
    head = oval.copy() if lab is not None else head_region(oval, layers["hair"][..., 3] > 127 if "hair" in layers else np.zeros((h, w), bool))
    if lab is not None:  # whole skin (ears, neck, forehead under the hair edge) joins the face base; no hull, so background between hair and neck stays out
        skin = cv2.morphologyEx(np.isin(lab, SKIN).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8)) > 0
        head = head | skin
        under = np.zeros((h, w), bool)  # the base also reaches under the hair / hat edge: no sliver of nothing where they meet the skin
        for p in ("hair", "hat"):
            if p in layers:
                under |= layers[p][..., 3] > 20
        head |= (cv2.dilate(head.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0) & under
        n, cl, st, _ = cv2.connectedComponentsWithStats(head.astype(np.uint8))  # stray skin bits (hands, neck scraps) away from the face are not part of the base
        if n > 2:
            head = np.isin(cl, [i for i in range(1, n) if st[i, 4] >= 0.15 * st[1:, 4].max()])
    hole = cv2.dilate(feat.astype(np.uint8), np.ones((5, 5) if args.faithful_base else (9, 9) if args.clean_base else (15, 15), np.uint8)) > 0  # must stay inside the solid part of the halos (HALO - FEATHER): a wider hole leaves a visible seam
    valid = (cv2.erode(head.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0) & ~hole
    seed = valid
    if valid.sum() >= 2000 and not (args.clean_base or args.faithful_base):  # keep neutral skin only (as v11 make_face): colour near the median, mid-tone luma
        labc = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32); luma = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        lo, hi = np.percentile(luma[valid], (10, 90))
        good = valid & (np.linalg.norm(labc - np.median(labc[valid], axis=0), axis=2) < 30) & (luma >= lo) & (luma <= hi)
        if good.sum() >= 1000:
            seed = cv2.morphologyEx(good.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0  # seeds the diffusion only: skin outside the holes keeps its own pixels
    if valid.sum() < 500:  # no usable skin: fall back to the v11 flat colour
        base = flat
    else:
        if args.faithful_base:
            base = np.dstack([np.where(valid[..., None] | (head & ~hole)[..., None], rgb, fill_holes(rgb, valid)), head * 255]).astype(np.uint8)
        elif args.clean_base:  # only the low frequencies of the skin survive: no eye sockets, nostril shadows or lip creases left in the base
            est = cv2.GaussianBlur(fill_holes(rgb, seed, (4, 8, 16, 32, 64, 128), 3), (0, 0), 5)  # measured on the 50 portraits: a luma/colour-filtered seed set made the base 18-56 L brighter than the photo
            base = np.dstack([np.where(head[..., None], est, 0), head * 255]).astype(np.uint8)
        else:
            est = fill_holes(rgb, seed)
            base = np.dstack([np.where((valid & ~hole)[..., None], rgb, est), head * 255]).astype(np.uint8)  # own pixels outside the holes, diffusion inside
    Image.fromarray(base).save(f"{args.seg}/parts/{pid}/face.png")
    ring_ok = head.astype(np.float32)
    if "glasses" in layers:  # no frame pixels in the halos; the edge fades out so no hard band is left where the frames were
        ring_ok *= 1 - np.clip(cv2.GaussianBlur(cv2.dilate((layers["glasses"][..., 3] > 127).astype(np.uint8), np.ones((9, 9), np.uint8)).astype(np.float32), (0, 0), 5) * 1.5, 0, 1)
    save_parts(pid, base, soft_layers({p: a for p, a in layers.items() if p in ORDER}, ring_ok))
    comp, a_s = assemble_from_disk(pid)  # the re-assembly: saved base + saved parts, nothing from memory
    bg = fill_holes(rgb, ~(cv2.dilate((a_s > 0.02).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0))  # background plate: the person region diffused away
    Image.fromarray((comp.astype(np.float32) * a_s[..., None] + bg.astype(np.float32) * (1 - a_s[..., None])).astype(np.uint8)).save(f"{args.out}/compose_full/{pid}.png")
    solid = a_s >= 0.99  # where the assembled person is fully opaque: it is meant to equal the portrait there
    Image.fromarray(np.dstack([comp, (a_s * 255).astype(np.uint8)])).save(f"{args.out}/compose/{pid}.png")
    psnr = float(10 * np.log10(255.0 ** 2 / max(1e-9, ((comp.astype(float) - rgb.astype(float)) ** 2)[solid].mean())))
    rows.append({"id": pid, "fill_frac": round(float((hole & head).sum() / max(1, head.sum())), 3), "psnr": round(psnr, 1)})
    cache[pid] = (base, layers)
    print(pid, rows[-1], flush=True)
with open(f"{args.out}/compose.csv", "w", newline="") as fh:
    wr = csv.DictWriter(fh, fieldnames=["id", "fill_frac", "psnr"]); wr.writeheader(); wr.writerows(rows)
for k in range(min(args.mix, len(ids))):  # new faces: features of B on the base of A, A keeps hair / hat
    a, b = ids[k], ids[(k + 7) % len(ids)]
    if a == b:
        continue
    base, la = cache[a]; _, lb = cache[b]
    mixed = {}
    for p in DONOR:
        g = GROUP.get(p, p)  # reference feature whose box is used for this layer
        if p in lb and g in la and g in lb:
            mixed[p] = transfer(lb[p], box(lb[g]), box(la[g]))
    for p in ("hair", "hat"):
        if p in la:
            mixed[p] = la[p]
    Image.fromarray(compose(base, mixed)).save(f"{args.out}/mix/{a}_with_{b}.png")
print("done", len(rows), "faces, mean PSNR", round(float(np.mean([r["psnr"] for r in rows])), 1))
