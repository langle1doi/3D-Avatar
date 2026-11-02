"""Save the SegFormer face-parsing class map (CelebAMask-HQ ids) of every portrait, for the face base / extra parts of pixar_face_compose.py.

  python pixar_seg_lab.py --img <portrait dir> --out <root>
ids: 1 skin 2 nose 3 glasses 4/5 eyes 6/7 brows 8/9 ears 10 mouth 11/12 lips 13 hair 14 hat 15 earring 16 necklace 17 neck 18 cloth.
Writes <root>/lab/<id>.png (uint8 class ids, same size as the portrait).
"""
import argparse, glob, os

import numpy as np, torch
from PIL import Image
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

ap = argparse.ArgumentParser()
ap.add_argument("--img"); ap.add_argument("--out")
args = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
M = "jonathandinu/face-parsing"
proc, net = SegformerImageProcessor.from_pretrained(M), SegformerForSemanticSegmentation.from_pretrained(M).eval().to(dev)
os.makedirs(f"{args.out}/lab", exist_ok=True)
for f in sorted(glob.glob(f"{args.img}/*.png")):
    pid = os.path.splitext(os.path.basename(f))[0]
    img = Image.open(f).convert("RGB"); w, h = img.size
    with torch.no_grad():
        lg = net(**{k: v.to(dev) for k, v in proc(images=img, return_tensors="pt").items()}).logits
    lab = torch.nn.functional.interpolate(lg, size=(h, w), mode="bilinear").argmax(1)[0].cpu().numpy().astype(np.uint8)
    Image.fromarray(lab).save(f"{args.out}/lab/{pid}.png")
print("done", len(glob.glob(f"{args.out}/lab/*.png")), "maps")
