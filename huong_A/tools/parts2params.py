# Các phần đã cắt của ảnh Pixar (results/pixar_parts_seg3_clean) -> tham số avatar + ảnh mống mắt.
# Dùng: python parts2params.py 0002 [--gpt] [--char rain|snow]     ghi demo/params/parts_0002.json và parts_0002_iris.png
# --gpt: gửi ảnh Pixar lên OpenAI để nhận nam/nữ, kiểu râu, kính (key đọc từ generative ai/.env, không in ra).
import base64, io, json, os, re, sys, urllib.request
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from photo2params import landmarks, landmarks_z, ratios, shapes, hexc, params, BASE, BASE_SNOW, REF

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEN = os.path.join(os.path.dirname(ROOT), "generative ai")
PARTS = os.environ.get("AVATAR_PARTS") or os.path.join(GEN, "results", "pixar_parts_seg3_clean")
OUT = os.path.join(ROOT, "demo", "params")
cl = lambda x, a=-1.0, b=1.0: float(np.clip(x, a, b))


def part(i, n):
    return np.asarray(Image.open(os.path.join(PARTS, "parts", i, n + ".png")).convert("RGBA"), float)


def med(a, q=50):  # màu đại diện của phần đã cắt; q > 50 để tránh vùng bóng đổ
    v = a[a[..., 3] > 200][:, :3]; l = v.mean(1)
    return np.median(v[(l >= np.percentile(l, max(q - 15, 0))) & (l <= np.percentile(l, min(q + 15, 100)))], axis=0)


def iris_disk(i, size=256):
    """Ghép 2 mống mắt thành đĩa tròn đủ: mí che phần trên nên điền từng vòng bằng trung vị của phần thấy được."""
    rings, R = [], 40
    for n in ("iris_left", "iris_right"):
        a = part(i, n); ys, xs = np.nonzero(a[..., 3] > 128)
        r = (xs.max() - xs.min() + 1) / 2; cx = (xs.max() + xs.min()) / 2; cy = ys.max() - r  # đáy mống mắt không bị che
        th = np.linspace(0, 2 * np.pi, 180, endpoint=False); rr = np.linspace(0, 1, R)
        px = cx + np.outer(rr, np.cos(th)) * r; py = cy + np.outer(rr, np.sin(th)) * r
        s = a[np.clip(py.round().astype(int), 0, 511), np.clip(px.round().astype(int), 0, 511)]
        s[s[..., :3].mean(-1) > 215] = (0, 0, 0, 0)  # bỏ đốm sáng in sẵn: cảnh 3D có phản chiếu riêng
        rings.append(s)
    s = np.concatenate(rings, axis=1)  # gộp 2 mắt, 360 góc
    col = np.zeros((R, 3))
    for k in range(R):
        v = s[k][s[k, :, 3] > 128]
        col[k] = np.median(v[:, :3], axis=0) if len(v) else col[k - 1]
    g = (np.arange(size) + 0.5) / size * 2 - 1; xx, yy = np.meshgrid(g, g); rad = np.hypot(xx, yy)
    idx = np.clip(rad * (R - 1), 0, R - 1); lo = idx.astype(int); hi = np.minimum(lo + 1, R - 1); f = (idx - lo)[..., None]
    out = np.zeros((size, size, 4)); out[..., :3] = col[lo] * (1 - f) + col[hi] * f
    out[..., 3] = np.clip((1 - rad) / 0.04, 0, 1) * 255  # mép mờ
    return Image.fromarray(out.astype(np.uint8), "RGBA")


def hair_shape(i, L):
    """Đo hình bóng tóc so với mặt (đơn vị: chiều cao/bề rộng mặt) rồi đổi sang thanh trượt của tóc."""
    a = part(i, "hair")[..., 3] > 128
    ys, xs = np.nonzero(a); H, W = a.shape
    chin = L[152][1]; top = L[10][1]; fh = chin - top; fw = np.linalg.norm(L[234] - L[454]); cx = (L[234][0] + L[454][0]) / 2
    clipped = ys.max() >= H - 2
    side = a.copy(); side[:, int(L[234][0]):int(L[454][0])] = False  # tóc hai bên, bỏ phần sau cổ giữa
    ye = np.nonzero(side.any(1))[0]
    lr = (ye.max() - chin) / fh if len(ye) else -0.5  # dài bao nhiêu dưới cằm
    band = a[int(chin - 0.1 * fh):int(chin + 0.1 * fh)]; bx = np.nonzero(band.any(0))[0]
    vol = (bx.max() - bx.min()) / fw if len(bx) else 1.0
    # gợn sóng: độ lệch mép ngoài so với đường cong khớp, phần dưới tai
    edge = []
    for y in range(int(L[234][1]), int(min(ye.max() if len(ye) else H - 1, H - 2))):
        c = np.nonzero(a[y])[0]
        if len(c): edge.append((y, c.min(), c.max()))
    ws = 0.0
    if len(edge) > 20:
        e = np.array(edge, float)
        for k in (1, 2): ws += np.std(e[:, k] - np.polyval(np.polyfit(e[:, 0], e[:, k], 2), e[:, 0])) / 2 / fw
    # chân tóc: ở giữa trán, tóc dừng cách lông mày bao nhiêu
    brow = np.nonzero((part(i, "brows")[..., 3] > 128).any(1))[0].min()
    col = np.nonzero(a[:brow, int(cx) - 25:int(cx) + 25].any(1))[0]
    hairline = (brow - col.max()) / fh if len(col) else 0.5
    # ôm mặt: ngang mắt, mép trong của tóc cách mép mặt bao nhiêu (0 = tóc sát mặt)
    y = int((L[33][1] + L[263][1]) / 2); row = np.nonzero(a[y])[0]
    li = row[row < cx]; ri = row[row > cx]
    gap = (((L[234][0] - li.max()) if len(li) else fw * 0.15) + ((ri.min() - L[454][0]) if len(ri) else fw * 0.15)) / 2 / fw
    # rẽ ngôi: đỉnh cao nhất của đường chân tóc trong khoảng hai lông mày
    xs0 = np.arange(int(L[234][0] + 0.2 * fw), int(L[454][0] - 0.2 * fw)); low = np.array([np.nonzero(a[:brow, x])[0].max() if a[:brow, x].any() else brow for x in xs0], float)
    low = np.convolve(low, np.ones(9) / 9, "same"); part_off = (xs0[4 + int(np.argmin(low[4:-4]))] - cx) / fw
    crown = (top - ys.min()) / fh  # tóc cao bao nhiêu trên trán
    m = {"part": round(float(part_off), 3), "crown": round(float(crown), 3), "len": round(float(lr), 3), "clipped": bool(clipped), "vol": round(float(vol), 3), "wave": round(float(ws), 4), "hairline": round(float(hairline), 3), "gap": round(float(gap), 3)}
    s = {"hairLen": max(cl((lr * 0.2 - 0.17) / 0.17), 0.2) if clipped else cl((lr * 0.2 - 0.17) / 0.17),  # 0.2 m = chiều cao mặt Rain, 0.17 m = độ dài gốc của lưới tóc
          "hairVol": cl((vol - 1.3) / 0.4), "hairWave": cl((ws - 0.008) / 0.02), "hairline": cl((0.42 - hairline) / 0.2 + 0.4), "hairFront": cl(1 - max(gap, 0) / 0.08), "hairPart": cl(part_off / 0.2)}
    b = part(i, "brows")[..., 3] > 128; t = np.median(b.sum(0)[b.sum(0) > 0]) / fh
    s["browThin"] = cl((0.06 - t) / 0.03, -0.5, 1)
    m["brow"] = round(float(t), 4)
    return m, s


def smile(i, L):
    a = part(i, "mouth")[..., 3] > 128; ys, xs = np.nonzero(a)
    w = xs.max() - xs.min(); c = (ys[xs == xs.max()].mean() + ys[xs == xs.min()].mean()) / 2
    mid = ys[np.abs(xs - (xs.max() + xs.min()) / 2) < 4].mean()
    return cl((mid - c) / w / 0.2, 0, 0.9)


def gpt(i):
    key = next(l.split("=", 1)[1].strip().strip('"\'') for l in open(os.path.join(GEN, ".env"), encoding="utf-8") if l.startswith("OPENAI_API_KEY="))
    buf = io.BytesIO(); Image.open(os.path.join(PARTS, "assembled", i + ".png")).convert("RGB").resize((384, 384)).save(buf, "JPEG", quality=85)
    url = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    ask = 'Stylized cartoon portrait for an avatar creator. Reply with JSON only: {"presentation":"male"|"female","facial_hair":"none"|"mustache"|"goatee"|"mustache_goatee"|"full_beard"|"stubble","glasses":true|false,"bangs":true|false}'
    for model in ("gpt-4o", "gpt-4-turbo", "gpt-4o-mini"):
        body = json.dumps({"model": model, "max_tokens": 80, "messages": [{"role": "user", "content": [{"type": "text", "text": ask}, {"type": "image_url", "image_url": {"url": url, "detail": "low"}}]}]}).encode()
        req = urllib.request.Request("https://api.openai.com/v1/chat/completions", body, {"Authorization": "Bearer " + key, "Content-Type": "application/json"})
        try:
            r = json.load(urllib.request.urlopen(req, timeout=60))
            t = r["choices"][0]["message"]["content"] or ""
            js = re.search(r"\{.*\}", t, re.S)
            if js: return json.loads(js.group(0)), model
            print("GPT", model, "khong tra JSON:", t[:90].replace("\n", " "))
        except urllib.error.HTTPError as e:  # chỉ in mã lỗi
            print("GPT", model, "HTTP", e.code)
    return None, None


def run(i, force=None, use_gpt=False):
    os.makedirs(OUT, exist_ok=True)
    _, L = landmarks(os.path.join(PARTS, "assembled", i + ".png"))
    if L is None: sys.exit("không dò được mặt: " + i)
    desc, model = gpt(i) if use_gpt else (None, None)
    skin = med(part(i, "face"), 70); ha = part(i, "hair"); yy = np.nonzero((ha[..., 3] > 200).any(1))[0]
    top_a = ha.copy(); top_a[int((yy.min() + yy.max()) / 2):] = 0; hair = med(top_a, 78)  # màu chính lấy ở nửa trên, đuôi tóc có thể nhuộm khác
    h50 = med(top_a, 50)
    if h50.mean() < 70 and np.ptp(h50) < 22: hair = h50 * 0.5 + h50.mean() * 0.5  # tóc đen (tối và gần như không có sắc màu, tóc nâu thì không): vùng sáng là ánh xanh phản chiếu, lấy màu giữa và bớt sắc
    if skin.max() > 240: skin = skin * 240 / skin.max()  # da quá sáng nhân với texture sẽ ra tím nhạt
    tip_a = ha.copy(); tip_a[:int(yy.max() - 0.2 * (yy.max() - yy.min()))] = 0; tip = med(tip_a, 70)  # đuôi tóc (tóc nhuộm hai màu)
    hair_tip = hexc(tip) if np.abs(tip - hair).sum() > 70 else None
    mo = part(i, "mouth"); red = mo[mo[..., 3] > 200]; lip = np.median(red[red[:, 0] - red[:, 1] > 30][:, :3], axis=0) if (red[:, 0] - red[:, 1] > 30).sum() > 20 else med(mo)
    loc = params(os.path.join(PARTS, "assembled", i + ".png"), json.load(open(REF)))  # kính, râu: nhận ngay trên máy
    char = force or ("snow" if (desc["presentation"] == "male" if desc else loc["beard"]) else "rain")
    fh_ = (desc or {}).get("facial_hair", "full_beard" if loc["beard"] else "none")
    m, s = hair_shape(i, L)
    base_ = shapes(ratios(L), json.load(open(REF)), BASE_SNOW if char == "snow" else BASE)
    base_["noseSize"] = max(base_["noseSize"], -0.3); base_["noseWidth"] = max(base_["noseWidth"], -0.2)  # ép mạnh thì mũi bẹt, lộ lỗ mũi
    p = {"char": char, "skin": hexc(skin), "hair": hexc(hair), "eyes": "#2a7fa8", "lip": hexc(lip), "style": "short" if char == "snow" else ("rain" if m["len"] < -0.2 else "gen"), "glasses": bool((desc or {}).get("glasses", loc["glasses"])), "hair_tip": hair_tip,
         "beard": fh_ not in ("none", "stubble") and char == "snow", "shapes": {**base_, **s, **({"hairVol": cl((m["crown"] - 0.13) / 0.12), "hairline": cl((0.42 - m["hairline"]) / 0.2)} if char == "snow" else {})},
         "expr": {"Cười": smile(i, L)}, "iris": f"params/parts_{i}_iris.png", "hair_measure": m, "gpt": desc, "gpt_model": model}
    iris_disk(i).save(os.path.join(OUT, f"parts_{i}_iris.png"))
    json.dump(p, open(os.path.join(OUT, f"parts_{i}.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return p


if __name__ == "__main__":
    i = sys.argv[1]; p = run(i, sys.argv[sys.argv.index("--char") + 1] if "--char" in sys.argv else None, "--gpt" in sys.argv)
    print(i, {k: p[k] for k in ("char", "skin", "hair", "lip", "glasses", "beard")}, p["hair_measure"])
