# Ảnh Pixar -> tham số avatar kiểu RPM (không chiếu ảnh lên mặt, chỉ đo rồi đặt thanh trượt).
# Dùng: python photo2params.py --pop thu_muc_anh          # đo trung bình nhóm ảnh Pixar, ghi pop_ref.json
#       python photo2params.py anh.png ra.json           # đo một ảnh
# Cần: mediapipe, pillow, numpy + file face_landmarker.task cạnh script.
import json, os, sys
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(HERE, "pop_ref.json")
det = FaceLandmarker.create_from_options(FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=os.path.join(HERE, "face_landmarker.task")), num_faces=1))


def landmarks(path):
    img = np.asarray(Image.open(path).convert("RGB"))
    r = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=img))
    if not r.face_landmarks: return img, None
    h, w = img.shape[:2]
    return img, np.array([(p.x * w, p.y * h) for p in r.face_landmarks[0]])


def landmarks_z(path):
    """Chiều sâu ước lượng của từng điểm mốc (điểm ảnh, số âm là gần máy ảnh). Tách riêng để các phép đo 2D không đổi."""
    img = np.asarray(Image.open(path).convert("RGB"))
    r = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=img))
    return np.array([p.z * img.shape[1] for p in r.face_landmarks[0]]) if r.face_landmarks else None


def ratios(L):
    D = lambda a, b: float(np.linalg.norm(L[a] - L[b]))
    d = D(468, 473)  # khoảng cách hai tâm mống mắt: thước đo chung
    fw = D(234, 454)
    return {"faceWidth": fw / d, "faceLong": D(168, 152) / d, "jaw": D(172, 397) / fw,
            "eyeW": (D(33, 133) + D(362, 263)) / 2 / d, "eyeH": (D(159, 145) + D(386, 374)) / 2 / d, "eyeSpacing": D(133, 362) / fw,
            "noseW": D(129, 358) / d, "noseH": D(168, 2) / d, "mouthW": D(61, 291) / d, "lip": D(0, 17) / d}


# Dáng nền kiểu Pixar trên Rain, cộng với độ lệch của người này so với trung bình nhóm ảnh Pixar (so với Rain thì ai cũng chạm trần vì Rain khác cả nhóm).
# gain: thanh trượt = 1 làm tỉ lệ đổi bao nhiêu (lấy từ biên độ morph trong rain_bake.py); AMP: phóng đại khác biệt cho dễ nhận ra.
BASE = {"faceWidth": 0.4, "faceLong": 0.3, "jawNarrow": -0.3, "cheek": 0.5, "eyeSize": 0.6, "eyeSpacing": 0.0, "noseSize": -0.3, "noseWidth": -0.3, "mouthWidth": 0.3, "lipFull": -0.1}
BASE_SNOW = {"faceWidth": 0.2, "faceLong": 0.0, "jawNarrow": -0.2, "cheek": 0.2, "eyeSize": 0.5, "eyeSpacing": 0.0, "noseSize": 0.0, "noseWidth": 0.0, "mouthWidth": 0.1, "lipFull": 0.0}
AMP = 1.2


def shapes(r, pop, base):
    q = {k: r[k] / pop[k] - 1 for k in r}
    d = {"faceWidth": q["faceWidth"] / 0.18, "faceLong": q["faceLong"] / 0.13, "jawNarrow": -q["jaw"] / 0.2, "cheek": q["jaw"] / 0.15,
         "eyeSize": (((1 + q["eyeW"]) * (1 + q["eyeH"])) ** 0.5 - 1) / 0.3, "eyeSpacing": q["eyeSpacing"] / 0.5,
         "noseWidth": q["noseW"] / 0.45, "noseSize": q["noseH"] / 0.3, "mouthWidth": q["mouthW"] / 0.35, "lipFull": q["lip"] / 0.55}
    return {k: float(np.clip(base[k] + AMP * d[k], -1, 1)) for k in base}


def hexc(c): return "#%02x%02x%02x" % tuple(int(np.clip(v, 0, 255)) for v in c)


def patch(img, p, r):
    x, y = int(p[0]), int(p[1]); h, w = img.shape[:2]
    return img[max(0, y - r):min(h, y + r + 1), max(0, x - r):min(w, x + r + 1)].reshape(-1, 3).astype(float)


def params(path, ref, force=None):
    img, L = landmarks(path)
    if L is None: return None
    fh = float(np.linalg.norm(L[10] - L[152])); fwp = float(np.linalg.norm(L[234] - L[454])); R = max(2, int(fh * 0.02))
    skin = np.median(np.vstack([patch(img, L[i], R) for i in (50, 280, 151)]), axis=0)
    skin_lit = skin; skin = skin * 0.85  # ảnh đã có sẵn ánh sáng, cảnh 3D chiếu thêm lần nữa
    lip = np.median(np.vstack([patch(img, (L[0] + L[13]) / 2, 1), patch(img, (L[14] + L[17]) / 2, 2)]), axis=0)
    # mắt: vòng trong mống mắt, bỏ con ngươi tối
    iris = []
    for c, e in ((468, 469), (473, 474)):
        rad = np.linalg.norm(L[c] - L[e])
        for a in np.linspace(0, 2 * np.pi, 16, endpoint=False):
            iris.append(patch(img, L[c] + 0.7 * rad * np.array([np.cos(a), np.sin(a)]), 1))
    iris = np.vstack(iris); eyes = np.median(iris[iris.sum(1) > np.median(iris.sum(1))], axis=0)
    # tóc: nhiều điểm trên trán và thái dương; loại màu giống da và màu nền xám/không sắc, lấy nửa tối (điểm sáng thường lẫn da)
    up = (L[10] - L[152]) / max(fh, 1); pts = [L[10] + up * fh * k for k in (0.06, 0.1, 0.14, 0.18, 0.22)] + [L[i] - (L[152] - L[i]) * 0.12 for i in (103, 332)]
    cand = [c for c in (np.median(patch(img, p, R), axis=0) for p in pts) if np.abs(c - skin_lit).sum() > 90 and (np.ptp(c) > 12 or c.mean() < 70)]
    if not cand: cand = [np.median(patch(img, L[234] - np.array([fwp * 0.08, 0]), R), axis=0)]
    cand = sorted(cand, key=lambda c: c.sum()); hair = np.median(cand[:max(1, (len(cand) + 1) // 2)], axis=0)
    is_hair = lambda p: np.abs(np.median(patch(img, p, R), axis=0) - hair).sum() < 140
    # kính: gờ tối giữa mi dưới và má, hoặc ở sống mũi (so với da)
    lum = lambda c: float(np.mean(c)); dark = 0
    for a, b in ((145, 50), (374, 280)):
        vals = [lum(np.median(patch(img, L[a] + (L[b] - L[a]) * t, 1), axis=0)) for t in np.linspace(0.15, 0.9, 12)]
        dark += min(vals) < 0.4 * lum(skin_lit)
    dark_bridge = lum(np.median(patch(img, L[6], 2), axis=0)) < 0.45 * lum(skin_lit)
    glasses = bool(dark == 2 or (dark >= 1 and dark_bridge))
    # độ dài tóc: có tóc hai bên dưới cằm thì dài, ngang cằm thì ngang vai
    side = lambda y: is_hair(L[234] + np.array([-fwp * 0.1, y])) or is_hair(L[454] + np.array([fwp * 0.1, y]))
    cy = L[152][1] - L[234][1]
    jaw_lv = side(cy * 0.5)  # ngang hàm phải có tóc: áo tối ở dưới cằm không được tính là tóc
    style = "rain" if not jaw_lv else "long" if side(cy + fh * 0.15) else "bob"
    # râu: cằm và nhân trung tối hơn da rõ rệt
    # râu/ria: vùng trên môi tối hẳn so với da (10 ảnh mẫu: có râu 0.01-0.15, không râu từ 0.31)
    beard = min(lum(np.median(patch(img, L[i], 1), axis=0)) for i in (164, 167, 393)) / lum(skin_lit) < 0.2
    # ponytail: MediaPipe không phân loại giới tính -> chỉ râu mới chắc là nam; nam không râu cần --char snow (sản phẩm thật: để người dùng chọn, RPM cũng hỏi)
    char = force or ("snow" if beard else "rain")
    if char == "snow": style = "short"
    r = ratios(L)
    return {"char": char, "skin": hexc(skin), "hair": hexc(hair), "eyes": hexc(eyes), "lip": hexc(lip), "style": style, "glasses": glasses, "beard": bool(beard and char == "snow"),
            "shapes": shapes(r, ref, BASE_SNOW if char == "snow" else BASE), "ratios": r}


if __name__ == "__main__":
    if sys.argv[1] == "--pop":  # trung bình nhóm ảnh Pixar
        rs = [ratios(L) for L in (landmarks(os.path.join(sys.argv[2], f))[1] for f in sorted(os.listdir(sys.argv[2])) if f.startswith(sys.argv[3] if len(sys.argv) > 3 else "")) if L is not None]
        json.dump({k: float(np.mean([r[k] for r in rs])) for k in rs[0]}, open(REF, "w")); print("pop", len(rs), "anh")
    else:
        force = sys.argv[sys.argv.index("--char") + 1] if "--char" in sys.argv else None
        p = params(sys.argv[1], json.load(open(REF)), force)
        if p is None: sys.exit("không dò được mặt: " + sys.argv[1])
        json.dump(p, open(sys.argv[2], "w"), ensure_ascii=False, indent=1)
        print(os.path.basename(sys.argv[1]), {k: p[k] for k in ("char", "skin", "hair", "style", "glasses", "beard")}, {k: round(v, 2) for k, v in p["shapes"].items()})
