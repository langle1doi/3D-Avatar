"""Stage 1 of pixar-seg2: sharpen the v11 part masks with SAM 2 and give the hair a soft alpha with ViTMatte.

  python pixar_seg2_masks.py --img <portraits> --v11 <v11 output root> --out <seg2 root> [--limit N]
v11 (SegFormer + MediaPipe) stays the prompt generator: SAM 2 gets the v11 box + one point per blob, and its mask is accepted only if it
overlaps the v11 mask (IoU >= MIN_IOU) and stays near it; otherwise the v11 mask is kept. Writes <out>/parts/<id>/<part>.png (full-canvas RGBA)
and <out>/stage1.csv.
"""
import argparse, csv, glob, os, shutil

import cv2, numpy as np, torch
from PIL import Image
from transformers import Sam2Model, Sam2Processor, VitMatteForImageMatting, VitMatteImageProcessor

SAM_PARTS = ("hair",)  # eyes / brows / nose / mouth keep the v11 masks: SAM 2 made the eyes ragged, shrank the nose (0.75x in 38/50) and left the mouth edge jagged with the lip tips cut
MIN_IOU = 0.5
MIN_AREA = 0.85  # a SAM 2 mask smaller than this fraction of the v11 mask lost part of the feature: keep v11

ap = argparse.ArgumentParser()
ap.add_argument("--img"); ap.add_argument("--v11"); ap.add_argument("--out"); ap.add_argument("--limit", type=int, default=0)
args = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
sam_p = Sam2Processor.from_pretrained("facebook/sam2.1-hiera-large")
sam = Sam2Model.from_pretrained("facebook/sam2.1-hiera-large").to(dev).eval()
mat_p = VitMatteImageProcessor.from_pretrained("hustvl/vitmatte-small-composition-1k")
mat = VitMatteForImageMatting.from_pretrained("hustvl/vitmatte-small-composition-1k").to(dev).eval()


def blob_points(m):
    """one point per blob (>= 5% of the biggest): the pixel deepest inside it"""
    n, cl, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8))
    pts = []
    for i in range(1, n):
        if st[i, 4] < 0.05 * st[1:, 4].max():
            continue
        d = cv2.distanceTransform((cl == i).astype(np.uint8), cv2.DIST_L2, 5)
        y, x = np.unravel_index(d.argmax(), d.shape); pts.append([int(x), int(y)])
    return pts


def sam_mask(img, m):
    ys, xs = np.nonzero(m)
    pts = blob_points(m)
    inp = sam_p(images=img, input_boxes=[[[int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]]],
                input_points=[[pts]], input_labels=[[[1] * len(pts)]], return_tensors="pt").to(dev)
    with torch.no_grad():
        out = sam(**inp, multimask_output=False)
    return sam_p.post_process_masks(out.pred_masks.cpu(), inp["original_sizes"])[0].reshape(-1, *m.shape)[0].numpy() > 0


def hair_alpha(img, m):
    """ViTMatte soft alpha for the hair: trimap = eroded mask (sure fg) / outside dilated mask (sure bg) / band in between"""
    k = np.ones((9, 9), np.uint8)
    fg = cv2.erode(m.astype(np.uint8), k) > 0; near = cv2.dilate(m.astype(np.uint8), k) > 0
    tri = np.where(fg, 255, np.where(near, 128, 0)).astype(np.uint8)
    inp = mat_p(images=img, trimaps=Image.fromarray(tri), return_tensors="pt").to(dev)
    with torch.no_grad():
        a = mat(**inp).alphas[0, 0].cpu().numpy()[: m.shape[0], : m.shape[1]]
    return np.where(fg, 1.0, np.where(near, np.clip(a, 0, 1), 0.0))


os.makedirs(f"{args.out}/parts", exist_ok=True)
rows = []
ids = sorted(os.path.basename(d) for d in glob.glob(f"{args.v11}/parts/*"))
for pid in (ids[: args.limit] if args.limit else ids):
    os.makedirs(f"{args.out}/parts/{pid}", exist_ok=True)
    img = Image.open(f"{args.img}/{pid}.png").convert("RGB"); rgb = np.array(img)
    for f in glob.glob(f"{args.v11}/parts/{pid}/*.png"):
        part = os.path.splitext(os.path.basename(f))[0]
        dst = f"{args.out}/parts/{pid}/{'face_flat' if part == 'face' else part}.png"  # stage 3 decides between face_flat and face_inpaint
        if part not in SAM_PARTS:
            shutil.copy(f, dst); continue
        m0 = np.array(Image.open(f))[..., 3] > 0
        alpha = m0.astype(np.float32); status = "v11"; iou = 0.0
        try:
            ms = sam_mask(img, m0) & (cv2.dilate(m0.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0)
            iou = float((ms & m0).sum() / max(1, (ms | m0).sum()))
            if iou >= MIN_IOU and ms.sum() >= MIN_AREA * m0.sum():
                alpha = ms.astype(np.float32); status = "sam2"
        except Exception as e:  # keep the v11 mask, but say so
            status = f"v11 (sam error {type(e).__name__}: {e})"
        if part == "hair":
            try:
                alpha = hair_alpha(img, alpha > 0); status += "+matte"
            except Exception as e:
                status += f" (matte error {type(e).__name__}: {e})"
        Image.fromarray(np.dstack([rgb, (alpha * 255).astype(np.uint8)])).save(dst)
        rows.append({"id": pid, "part": part, "status": status, "iou_v11": round(iou, 3), "area_v11": int(m0.sum()), "area_new": int((alpha > 0.5).sum())})
    print(pid, flush=True)
with open(f"{args.out}/stage1.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["id", "part", "status", "iou_v11", "area_v11", "area_new"]); w.writeheader(); w.writerows(rows)
n_sam = sum(r["status"].startswith("sam2") for r in rows)
print(f"{n_sam}/{len(rows)} parts refined by SAM 2")
assert not any("error" in r["status"] for r in rows), "SAM 2 / ViTMatte raised errors, see stage1.csv"
