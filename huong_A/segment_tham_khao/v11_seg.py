
import glob, os
import numpy as np, torch, cv2, urllib.request
import mediapipe as mp
from mediapipe.tasks import python as mpt
from mediapipe.tasks.python import vision
from PIL import Image
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

M = "jonathandinu/face-parsing"
proc, net = SegformerImageProcessor.from_pretrained(M), SegformerForSemanticSegmentation.from_pretrained(M).eval()
urllib.request.urlretrieve("https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task", "lm.task")
lmk = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(base_options=mpt.BaseOptions(model_asset_path="lm.task")))
EYE = {"right": [33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246],
       "left": [263,249,390,373,374,380,381,382,362,398,384,385,386,387,388,466]}
IRIS = {"right": [468, 469, 470, 471, 472], "left": [473, 474, 475, 476, 477]}
BROW = {"right": [46,53,52,65,55,70,63,105,66,107], "left": [276,283,282,295,285,300,293,334,296,336]}

def refine(rgb, m, pts):
    """m: dict of segformer masks (modified in place); pts: Nx2 landmarks in pixels"""
    h, w = rgb.shape[:2]
    luma = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    eyes = np.zeros((h, w), np.uint8); lash = {}
    for side, idx in EYE.items():
        e = np.zeros((h, w), np.uint8); cv2.fillPoly(e, [pts[idx].astype(np.int32)], 1)
        seg = (m["eyes"] & True).astype(np.uint8)
        n_c, cl = cv2.connectedComponents(seg)  # segformer iris blob whose centre falls inside the landmark eye
        cx, cy = pts[idx].mean(0); sel = np.zeros((h, w), np.uint8)
        for c in range(1, n_c):
            yy, xx = np.where(cl == c)
            if abs(xx.mean() - cx) < 0.6 * (pts[idx, 0].max() - pts[idx, 0].min()) and abs(yy.mean() - cy) < 0.6 * (pts[idx, 0].max() - pts[idx, 0].min()):
                sel[cl == c] = 1
        both = np.argwhere((e | sel) > 0)[:, ::-1].astype(np.int32)
        e = np.zeros((h, w), np.uint8); cv2.fillConvexPoly(e, cv2.convexHull(both), 1)  # hull: covers whites + iris
        if len(pts) > 477:  # pupil = iris disc from the landmarker's iris points (iris + black pupil + highlight)
            ic = pts[IRIS[side][0]]; r = np.linalg.norm(pts[IRIS[side][1:]] - ic, axis=1).mean()
            pm = cv2.circle(np.zeros((h, w), np.uint8), (int(ic[0]), int(ic[1])), int(round(1.4 * r)), 1, -1)  # 1.4x: landmark ring is smaller than a big Pixar iris
            iris = (pm & e) > 0
            # clean rim: drop sclera (bright, unsaturated) and eyelid skin from the outer ring; the highlight inside is kept by the hull fill below
            Rd = 1.4 * r; yy, xx = np.mgrid[0:h, 0:w]
            outer = np.hypot(xx - ic[0], yy - ic[1]) > 0.6 * Rd
            sat = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[..., 1]
            bad = (luma > 140) & (sat < 80)
            if m["nose"].sum() > 50:
                labc = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
                bad |= np.linalg.norm(labc - np.median(labc[m["nose"]], axis=0), axis=2) < 25
            iris &= ~(outer & bad)
            iris = cv2.morphologyEx(iris.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            pn, pl = cv2.connectedComponents(iris)
            if pn > 1:  # biggest blob, convex-hull filled (smooth rim, highlight hole closed), still inside the eye
                yy2, xx2 = np.nonzero(pl == max(range(1, pn), key=lambda i: (pl == i).sum()))
                iris = (cv2.fillConvexPoly(np.zeros((h, w), np.uint8), cv2.convexHull(np.stack([xx2, yy2], 1).astype(np.int32)), 1) > 0) & (e > 0)
            else:
                iris = iris > 0
            lash["iris_" + side] = iris  # clean round iris incl. black pupil and highlight, no sclera
        eyes |= e
        x0, x1 = pts[idx, 0].min(), pts[idx, 0].max(); y0, y1 = pts[idx, 1].min(), pts[idx, 1].max()
        k = max(5, int(0.3 * (x1 - x0))) | 1
        ring = cv2.dilate(e, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) > 0
        band = ring & (e == 0) & (np.arange(h)[:, None] < (y0 + y1) / 2)
        if band.sum():
            c = band & (luma < np.percentile(luma[band], 35))
            lash["lashes_" + side] = cv2.morphologyEx(c.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)) > 0
    m["eyes"] = eyes > 0
    m.update(lash)
    ey = np.where(eyes > 0)[0]
    m["nose"][: int(ey.max() - 0.25 * (ey.max() - ey.min())), :] = False  # nose starts below the middle of the eyes
    for side, idx in BROW.items():
        hull = cv2.convexHull(pts[idx].astype(np.int32)); b = np.zeros((h, w), np.uint8); cv2.fillConvexPoly(b, hull, 1)
        bw = pts[idx, 0].max() - pts[idx, 0].min(); kk = max(3, int(0.08 * bw)) | 1
        reg = cv2.dilate(b, np.ones((kk, kk), np.uint8)) > 0
        reg &= (eyes == 0) & ~m["hair"]
        if reg.sum():
            cand = m["brows"] | (reg & (luma < np.percentile(luma[reg], 40)))
            n_c, cl = cv2.connectedComponents(cand.astype(np.uint8))
            keep = np.unique(cl[m["brows"] & reg]); keep = keep[keep > 0]  # only blobs that touch the segformer brow
            m["brows"] = (m["brows"] & ~reg) | np.isin(cl, keep)

    # main-face cleanup: hair only around the main face, nose only the blob at the nose tip
    fw = pts[:, 0].max() - pts[:, 0].min(); cx = pts[:, 0].mean(); eye_y = pts[EYE["right"], 1].mean()
    X = np.arange(w)[None, :]; Y = np.arange(h)[:, None]
    zone = (np.abs(X - cx) < 0.9 * fw) & (Y < pts[:, 1].max() + 2.0 * fw)
    oval = cv2.fillConvexPoly(np.zeros((h, w), np.uint8), cv2.convexHull(pts.astype(np.int32)), 1) > 0
    m["hair"] &= zone & ~(oval & (Y > eye_y))  # beard / face skin / hands inside the face are not hair
    if m["nose"].sum() > 50:  # hands / skin mislabeled as hair: hair pixels below the eyes that look like the face skin
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        skin = np.median(lab[m["nose"]], axis=0)
        m["hair"] &= ~((np.linalg.norm(lab - skin, axis=2) < 22) & (Y > eye_y))
        m["hair"] = cv2.morphologyEx(m["hair"].astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
    hn, hl, hs, _ = cv2.connectedComponentsWithStats(m["hair"].astype(np.uint8))
    if hn > 1:
        m["hair"] = np.isin(hl, [i for i in range(1, hn) if hs[i, 4] >= 0.1 * hs[1:, 4].max()])
    nn, nl = cv2.connectedComponents(m["nose"].astype(np.uint8))
    if nn > 2:
        tip = pts[1].astype(int); tip = (min(tip[1], h - 1), min(tip[0], w - 1))
        d = [(np.hypot(*(np.argwhere(nl == i).mean(0) - tip)), i) for i in range(1, nn)]
        m["nose"] = nl == min(d)[1]
    return m


dev = "cuda" if torch.cuda.is_available() else "cpu"; net.to(dev)
# CelebAMask-HQ ids: 1 skin 2 nose 4/5 l/r eye 6/7 brows 10 mouth 11/12 lips 13 hair
CLS = {"hair": [13], "brows": [6, 7], "eyes": [4, 5], "nose": [2], "mouth": [10, 11, 12]}
OVAL = [10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109]


def make_face(rgb, lab, m, pts):
    """face base: face-oval shape, eyes / brows / nose / mouth / hair / glasses replaced by smooth skin colour (None if no skin found)"""
    h, w = lab.shape
    if pts is not None:
        shape = np.zeros((h, w), np.uint8); cv2.fillPoly(shape, [pts[OVAL].astype(np.int32)], 1)
    else:  # fallback: convex hull of skin + feature pixels
        u = np.isin(lab, [1, 2, 3, 4, 5, 6, 7, 10, 11, 12]).astype(np.uint8)
        n_c, cl, st, _ = cv2.connectedComponentsWithStats(u)
        if n_c < 2:
            return None
        big = cl == 1 + np.argmax(st[1:, 4])
        shape = cv2.fillConvexPoly(np.zeros((h, w), np.uint8), cv2.convexHull(np.argwhere(big)[:, ::-1].astype(np.int32)), 1)
    shape = shape > 0
    feat = np.isin(lab, [3, 13, 14, 16, 17, 18]).astype(np.uint8)  # glasses, hair, hat, neck, cloth
    for p in ("hair", "brows", "eyes", "nose", "mouth"):
        feat |= m[p].astype(np.uint8)
    feat = cv2.dilate(feat, np.ones((7, 7), np.uint8)) > 0
    valid = cv2.erode((shape & (lab == 1) & ~feat).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    if valid.sum() < 200:
        return None
    labc = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    valid &= np.linalg.norm(labc - np.median(labc[valid], axis=0), axis=2) < 28  # beard / moustache shadows are not skin
    if valid.sum() < 200:
        return None
    luma = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    lo, hi = np.percentile(luma[valid], (25, 75))
    valid = cv2.erode((valid & (luma >= lo) & (luma <= hi)).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0  # mid-tones only: no eye / nose / hair shadows or highlights
    if valid.sum() < 200:
        return None
    ys, xs = np.nonzero(shape); x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()

    def basis(yy, xx):  # plane in face-normalised coordinates (higher orders extrapolate wildly where little skin is left)
        u = (xx - (x0 + x1) / 2) / max(1, (x1 - x0) / 2); v = (yy - (y0 + y1) / 2) / max(1, (y1 - y0) / 2)
        return np.stack([np.ones_like(u), u, v], -1)
    vy, vx = np.nonzero(valid)
    if len(vy) > 20000:
        k = np.random.default_rng(0).choice(len(vy), 20000, replace=False); vy, vx = vy[k], vx[k]
    coef = np.linalg.lstsq(basis(vy.astype(np.float32), vx.astype(np.float32)), rgb[vy, vx].astype(np.float32), rcond=None)[0]
    out = np.zeros((h, w, 3), np.float32)
    sk = rgb[vy, vx].astype(np.float32)
    plane = np.clip(basis(ys.astype(np.float32), xs.astype(np.float32)) @ coef, np.percentile(sk, 2, axis=0), np.percentile(sk, 98, axis=0))
    out[ys, xs] = sk.mean(0) + 0.5 * (plane - sk.mean(0))  # halve the gradient: flatter, like the reference face texture  # one flat skin colour with a gentle gradient, like a face texture
    return out.astype(np.uint8), shape


files = sorted(glob.glob("/kaggle/input/**/*.png", recursive=True))
W = "/kaggle/working"
COL = {"hair": (255, 0, 0), "brows": (0, 255, 255), "eyes": (0, 255, 0), "lashes_left": (255, 128, 0), "lashes_right": (255, 128, 0), "iris_left": (255, 255, 0), "iris_right": (255, 255, 0), "nose": (0, 128, 255), "mouth": (255, 0, 255)}
n = 0
for f in files:
    pid = os.path.splitext(os.path.basename(f))[0]
    img = Image.open(f).convert("RGB"); rgb = np.array(img); h, w = rgb.shape[:2]
    with torch.no_grad():
        lg = net(**{k: v.to(dev) for k, v in proc(images=img, return_tensors="pt").items()}).logits
    lab = torch.nn.functional.interpolate(lg, size=(h, w), mode="bilinear").argmax(1)[0].cpu().numpy()
    m = {p: np.isin(lab, c) for p, c in CLS.items()}
    res = lmk.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    pts_arr = np.array([[l.x * w, l.y * h] for l in res.face_landmarks[0]]) if res.face_landmarks else None
    rgbs = {}
    if res.face_landmarks:  # landmarks fix eyes / nose / brows and give lashes
        refine(rgb, m, pts_arr)
    else:  # fallback: segformer only, lashes guessed from the band above the eyes
        # lashes: dark pixels in the upper band above each eye (no lash class in the model); one mask per eye
        luma = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        n_c, cl, st, _ = cv2.connectedComponentsWithStats(m["eyes"].astype(np.uint8))
        eyes_cc = sorted(sorted(range(1, n_c), key=lambda i: -st[i, 4])[:2], key=lambda i: st[i, 0])
        for side, i in zip(("right", "left"), eyes_cc):  # image-left eye is the subject's right eye
            em = cl == i; x0, y0, bw, bh = st[i, :4]
            k = max(5, int(0.35 * bw)) | 1
            ring = cv2.dilate(em.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) > 0
            band = ring & ~em & (np.arange(h)[:, None] < y0 + 0.5 * bh)
            if band.sum() == 0:
                continue
            m["lashes_" + side] = cv2.morphologyEx((band & (luma < np.percentile(luma[band], 35))).astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)) > 0
    fc = make_face(rgb, lab, m, pts_arr)  # after refine: m holds the final feature masks
    if fc is not None:
        rgbs["face"], m["face"] = fc
    for d in ("parts", "parts_crop"):
        os.makedirs(f"{W}/{d}/{pid}", exist_ok=True)
    os.makedirs(f"{W}/overlay", exist_ok=True)
    ov = rgb.copy()
    for p, mk in m.items():
        if mk.sum() < 30:
            print("MISSING", pid, p); continue
        mk8 = cv2.morphologyEx(mk.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))  # fill pin-holes
        rgba = np.dstack([rgbs.get(p, rgb), mk8 * 255]).astype(np.uint8)
        Image.fromarray(rgba).save(f"{W}/parts/{pid}/{p}.png")  # full canvas: same position as the portrait, for the rig
        yy, xx = np.where(mk8 > 0)
        Image.fromarray(rgba[yy.min():yy.max() + 1, xx.min():xx.max() + 1]).save(f"{W}/parts_crop/{pid}/{p}.png")
        if p != "face":
            ov[mk8 > 0] = (0.5 * ov[mk8 > 0] + 0.5 * np.array(COL[p])).astype(np.uint8)
    Image.fromarray(np.hstack([rgb, ov])).save(f"{W}/overlay/{pid}.png")
    n += 1
print(n, "portraits")
assert n > 0
