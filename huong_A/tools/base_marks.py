# Điểm mốc trên nhân vật nền: dò trên ảnh chụp thẳng demo/shots/ortho_<char>.png rồi đổi sang toạ độ thế giới (x, y).
# Dùng: python base_marks.py rain 0.42637 1.45335     ghi blender/out/rain_marks.npy, chạy lại khi nhân vật nền đổi.
import os, sys
import numpy as np
from PIL import Image, ImageDraw
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from photo2params import landmarks, landmarks_z

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
c, S, cy = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
path = os.path.join(ROOT, "demo", "shots", f"ortho_{c}.png")
_, L = landmarks(path)
if L is None: sys.exit("không dò được mặt")
L = np.asarray(L, float); N = Image.open(path).width
P = np.stack([(L[:, 0] / N - 0.5) * S, cy + (0.5 - L[:, 1] / N) * S], 1)
np.save(os.path.join(ROOT, "blender", "out", c + "_marks.npy"), P)
np.save(os.path.join(ROOT, "blender", "out", c + "_marks_z.npy"), landmarks_z(path) * S / N)
im = Image.open(path).convert("RGB"); d = ImageDraw.Draw(im)
for x, y in L[:, :2]: d.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(0, 200, 0))
im.save(os.path.join(ROOT, "demo", "shots", f"ortho_{c}_marks.png"))
print(c, len(P), "x", P[:, 0].min().round(3), P[:, 0].max().round(3), "y", P[:, 1].min().round(3), P[:, 1].max().round(3))
