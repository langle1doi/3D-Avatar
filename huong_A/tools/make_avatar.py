# Bước cuối của luồng: các phần đã cắt của ảnh Pixar -> một file nhân vật 3D .glb hoàn chỉnh, không cần Blender.
# Dùng: python make_avatar.py 0002 [0007 ...] [--char rain|snow] [--out thu_muc]     (--all: mọi ảnh trong thư mục đã cắt)
# Nhân vật nền (blender/out/rain_rpm.glb, snow_rpm.glb) do rain_bake.py tạo sẵn một lần. Thư mục đã cắt: biến AVATAR_PARTS hoặc mặc định trong parts2params.py.
# Kết quả: màu, tóc, kính, khuôn mặt đã cố định; xương và biểu cảm vẫn giữ để cử động. Bản quyền ghi ở asset.copyright (CC-BY, bắt buộc giữ).
import io, json, os, struct, sys, time
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import parts2params as P2
import face_fit as FF

ROOT = P2.ROOT
BASE_DIR = os.path.join(ROOT, "blender", "out")
COMP = {5120: "i1", 5121: "u1", 5122: "i2", 5123: "u2", 5125: "u4", 5126: "f4"}
LEN = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}
HAIRS = {"rain": {"gen": ["hair_gen"], "rain": ["hair_main", "hair_ponytail", "hair_strand", "hairband"]}, "snow": {"short": ["hair_base"]}}
lin = lambda s: np.where(s <= 0.04045, s / 12.92, ((s + 0.055) / 1.055) ** 2.4)
srgb = lambda x: np.where(x <= 0.0031308, x * 12.92, 1.055 * np.maximum(x, 1e-9) ** (1 / 2.4) - 0.055)
hex2lin = lambda h: lin(np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)], float) / 255)
to_img = lambda a: Image.fromarray((np.clip(srgb(a), 0, 1) * 255).astype(np.uint8))


class Glb:
    def __init__(self, path):
        b = open(path, "rb").read()
        jl = struct.unpack("<I", b[12:16])[0]; self.j = json.loads(b[20:20 + jl])
        bl = struct.unpack("<I", b[20 + jl:24 + jl])[0]; self.bin = b[28 + jl:28 + jl + bl]
        self.new = {}  # chỉ số image -> bytes ảnh đã sửa

    def view(self, i):
        v = self.j["bufferViews"][i]
        return self.bin[v.get("byteOffset", 0):v.get("byteOffset", 0) + v["byteLength"]]

    def read(self, i):
        a = self.j["accessors"][i]; dt = np.dtype("<" + COMP[a["componentType"]]); k = LEN[a["type"]]
        if "bufferView" in a:
            out = np.frombuffer(self.view(a["bufferView"]), dt, a["count"] * k, a.get("byteOffset", 0)).copy()
        else:
            out = np.zeros(a["count"] * k, dt)
        out = out.reshape(-1, k)
        if "sparse" in a:
            s = a["sparse"]; n = s["count"]
            idx = np.frombuffer(self.view(s["indices"]["bufferView"]), "<" + COMP[s["indices"]["componentType"]], n, s["indices"].get("byteOffset", 0))
            val = np.frombuffer(self.view(s["values"]["bufferView"]), dt, n * k, s["values"].get("byteOffset", 0))
            out[idx] = val.reshape(-1, k)
        return out

    def image_of(self, mat_name):
        m = next(x for x in self.j["materials"] if x["name"] == mat_name)
        tex = self.j["textures"][m["pbrMetallicRoughness"]["baseColorTexture"]["index"]]
        return tex.get("source", tex.get("extensions", {}).get("EXT_texture_webp", {}).get("source"))

    def get_image(self, i):
        return Image.open(io.BytesIO(self.new.get(i) or self.view(self.j["images"][i]["bufferView"]))).convert("RGB")

    def set_image(self, i, im):
        buf = io.BytesIO(); im.save(buf, "WEBP", quality=88); self.new[i] = buf.getvalue()

    def tint(self, mat_name, hex_, neutral=0.8):
        """Nhân màu thẳng vào ảnh (không dùng baseColorFactor vì giá trị đó bị giới hạn ở 1)."""
        i = self.image_of(mat_name)
        self.set_image(i, to_img(lin(np.asarray(self.get_image(i), float) / 255) * hex2lin(hex_) / neutral))


def set_factor(g, name, rgb):
    for m in g.j["materials"]:
        if m["name"] == name:
            p = m.setdefault("pbrMetallicRoughness", {}); p.pop("baseColorTexture", None); p["baseColorFactor"] = [float(x) for x in np.clip(rgb, 0, 1)] + [1.0]


def bake_shapes(g, p, char):
    """Trộn các morph khuôn mặt/tóc vào lưới gốc rồi bỏ chúng đi; chỉ biểu cảm được giữ. Morph bù mắt to cộng vào biểu cảm chớp mắt."""
    sh = p["shapes"]; eye = sh.get("eyeSize", 0.0)
    smile = min(p.get("expr", {}).get("Cười", 0.0), 0.3 if char == "snow" else 1.0)
    for m in g.j["meshes"]:
        for pr in m["primitives"]:  # màu đỉnh chỉ là dữ liệu trung gian; để lại sẽ nhân vào vật liệu làm tối cả mesh
            for a in ("COLOR_0", "COLOR_1"): pr["attributes"].pop(a, None)
        names = m.get("extras", {}).get("targetNames")
        if not names: continue
        keep = [n for n, t in enumerate(names) if not t.startswith(("shape_", "fix_"))]
        for pr in m["primitives"]:
            pos = g.read(pr["attributes"]["POSITION"]).astype(float); tg = pr["targets"]
            for n, t in enumerate(names):
                if t.startswith("shape_") and sh.get(t[6:]): pos += sh[t[6:]] * g.read(tg[n]["POSITION"])
            for n, t in enumerate(names):
                if t.startswith("fix_eyeBlink") and eye:  # fix_eyeBlinkLeft_eyeSize -> cộng vào eyeBlinkLeft
                    base = names.index(t.split("_")[1]); tg[base] = {"POSITION": (g.read(tg[base]["POSITION"]) + eye * g.read(tg[n]["POSITION"])).astype("<f4")}
            pr["attributes"]["POSITION"] = pos.astype("<f4")
            pr["targets"] = [tg[n] for n in keep]
        kept = [names[n] for n in keep]
        if kept:
            m["extras"]["targetNames"] = kept
            m["weights"] = [smile if t in ("mouthSmileLeft", "mouthSmileRight") else p.get("expr", {}).get("Há", 0.0) if t == "jawOpen" else 0.0 for t in kept]
        else:
            m.pop("extras", None); m.pop("weights", None)
            for pr in m["primitives"]: pr.pop("targets", None)


def paint_beard(g, hair_hex):
    """Trọng số râu nằm trên từng đỉnh của đầu (COLOR_1): vẽ ra ảnh theo UV rồi pha màu râu vào da."""
    head = next(m for m in g.j["meshes"] if any(g.j["materials"][p["material"]]["name"] == "head" for p in m["primitives"]))
    i = g.image_of("head"); im = g.get_image(i); W, H = im.size
    mask = Image.new("L", (W // 2, H // 2)); dr = ImageDraw.Draw(mask)
    for pr in head["primitives"]:
        if "COLOR_1" not in pr["attributes"]: continue
        c = g.read(pr["attributes"]["COLOR_1"])[:, 0].astype(float)
        if c.max() > 1: c = c / 65535
        uv = g.read(pr["attributes"]["TEXCOORD_0"]) * (W // 2, H // 2); tri = g.read(pr["indices"]).reshape(-1, 3)
        for t in tri:
            w = c[t].mean()
            if w > 0.01: dr.polygon([tuple(uv[k]) for k in t], fill=int(w * 255))
    mask = np.asarray(mask.filter(ImageFilter.GaussianBlur(3)).resize((W, H), Image.BILINEAR), float)[..., None] / 255 * 0.93
    g.set_image(i, to_img(lin(np.asarray(im, float) / 255) * (1 - mask) + hex2lin(hair_hex) * mask))


def paint_lip(g, lip_hex):
    i = g.image_of("head"); im = g.get_image(i); W, H = im.size
    m = np.asarray(Image.open(os.path.join(BASE_DIR, "rain_tex", "lip_mask.png")).convert("L").resize((W, H)), float)[..., None] / 255
    a = lin(np.asarray(im, float) / 255); l = np.clip(a.mean(-1, keepdims=True) / 0.3, 0.4, 1.3)
    g.set_image(i, to_img(a * (1 - m) + hex2lin(lip_hex) * l * m))


def paint_iris(g, iris_path, disk):
    i = g.image_of("eyes"); im = g.get_image(i).convert("RGBA"); W, H = im.size; u, v, r = disk
    ir = Image.open(iris_path).convert("RGBA").resize((int(2 * r * W), int(2 * r * H)), Image.LANCZOS)
    im.alpha_composite(ir, (int((u - r) * W), int((v - r) * H))); g.set_image(i, im.convert("RGB"))


def FF_lip(g, char):
    """Các điểm thuộc môi trên lưới nền (x, y), để đo viền môi thật."""
    head = next(m for m in g.j["meshes"] if any(g.j["materials"][p["material"]]["name"] == "head" for p in m["primitives"]))
    if char == "snow":
        return np.load(os.path.join(BASE_DIR, "snow_marks.npy"))[[61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]]  # Snow không có vùng môi riêng
    pr = head["primitives"][0]; v = g.read(pr["attributes"]["POSITION"]).astype(float); uv = g.read(pr["attributes"]["TEXCOORD_0"])
    a = np.asarray(Image.open(os.path.join(BASE_DIR, "rain_tex", "lip_mask.png")).convert("L")); H, W = a.shape
    return v[a[(uv[:, 1] * H).astype(int).clip(0, H - 1), (uv[:, 0] * W).astype(int).clip(0, W - 1)] > 128][:, :2]


def save(g, path, credit):
    """Ghi lại, chỉ giữ dữ liệu còn dùng (nhờ vậy file nhẹ đi khi bỏ tóc thừa và các morph đã trộn)."""
    j = g.j; meshes, mmap = [], {}
    for n in j["nodes"]:
        if "mesh" not in n: continue
        if n["mesh"] not in mmap: mmap[n["mesh"]] = len(meshes); meshes.append(j["meshes"][n["mesh"]])
        n["mesh"] = mmap[n["mesh"]]
    j["meshes"] = meshes
    out = bytearray(); views = []; vmap = {}; accs = []; amap = {}

    def add(raw, target=None):
        out.extend(b"\0" * (-len(out) % 4)); v = {"buffer": 0, "byteOffset": len(out), "byteLength": len(raw)}
        if target: v["target"] = target
        out.extend(raw); views.append(v); return len(views) - 1

    def old_view(i):
        if i not in vmap: vmap[i] = add(g.view(i), j["bufferViews"][i].get("target"))
        return vmap[i]

    def acc(x, target=None):
        if isinstance(x, np.ndarray):  # lưới đã sửa
            a = {"bufferView": add(x.tobytes(), target), "componentType": {"f4": 5126, "u4": 5125, "u2": 5123, "u1": 5121}[x.dtype.str[1:]], "count": len(x), "type": {1: "SCALAR", 2: "VEC2", 3: "VEC3", 4: "VEC4"}[x.shape[1]]}
            if x.dtype.kind == "f": a["min"], a["max"] = x.min(0).tolist(), x.max(0).tolist()
            accs.append(a)
            return len(accs) - 1
        if x not in amap:
            a = dict(j["accessors"][x])
            if "bufferView" in a: a["bufferView"] = old_view(a["bufferView"])
            if "sparse" in a:
                s = json.loads(json.dumps(a["sparse"])); s["indices"]["bufferView"] = old_view(s["indices"]["bufferView"]); s["values"]["bufferView"] = old_view(s["values"]["bufferView"]); a["sparse"] = s
            accs.append(a); amap[x] = len(accs) - 1
        return amap[x]

    for m in meshes:
        for pr in m["primitives"]:
            pr["attributes"] = {k: acc(v, 34962) for k, v in pr["attributes"].items()}
            if "indices" in pr: pr["indices"] = acc(pr["indices"], 34963)
            if "targets" in pr: pr["targets"] = [{k: acc(v) for k, v in t.items()} for t in pr["targets"]]
    for s in j.get("skins", []):
        if "inverseBindMatrices" in s: s["inverseBindMatrices"] = acc(s["inverseBindMatrices"])
    used_imgs = set()
    for k in {pr["material"] for m in meshes for pr in m["primitives"] if "material" in pr}:
        t = j["materials"][k].get("pbrMetallicRoughness", {}).get("baseColorTexture")
        if t:
            tx = j["textures"][t["index"]]; used_imgs.add(tx.get("source", tx.get("extensions", {}).get("EXT_texture_webp", {}).get("source")))
    tiny = io.BytesIO(); Image.new("RGB", (1, 1)).save(tiny, "WEBP")
    for i, im in enumerate(j["images"]):
        im["bufferView"] = add(g.new.get(i) or (g.view(im["bufferView"]) if i in used_imgs else tiny.getvalue()))
    j["accessors"] = accs; j["bufferViews"] = views; j["buffers"] = [{"byteLength": len(out)}]
    j["asset"]["copyright"] = credit
    js = json.dumps(j, separators=(",", ":")).encode(); js += b" " * (-len(js) % 4); out.extend(b"\0" * (-len(out) % 4))
    with open(path, "wb") as f:
        f.write(b"glTF" + struct.pack("<II", 2, 28 + len(js) + len(out)) + struct.pack("<I", len(js)) + b"JSON" + js + struct.pack("<I", len(out)) + b"BIN\0" + bytes(out))


def make(i, force=None, out_dir=None):
    cm = json.load(open(os.path.join(ROOT, "tools", "char_map.json"), encoding="utf-8"))
    p = P2.run(i, force or ("snow" if i in cm["snow"] else "rain"))
    p["glasses"] = os.path.exists(os.path.join(P2.PARTS, "parts", i, "glasses.png"))  # bước cắt đã biết có kính hay không
    p["beard"] = p["char"] == "snow" and (p["beard"] or i in cm["beard"])
    char = p["char"]; cfg = json.load(open(os.path.join(BASE_DIR, char + "_rpm.json"), encoding="utf-8"))
    g = Glb(os.path.join(BASE_DIR, char + "_rpm.glb"))
    p["style"] = "gen" if char == "rain" else "short"
    drop = [m for st, ms in HAIRS[char].items() if st != p["style"] for m in ms] + ([] if p["glasses"] else ["glasses"]) + ["hairband"]
    for n in g.j["nodes"]:
        if n.get("name") in drop: n.pop("mesh", None); n.pop("skin", None)
    rgb_ = lambda h: np.array([int(h[k:k + 2], 16) for k in (1, 3, 5)], float)
    bm, bc = (None, None)
    _, L = P2.landmarks(os.path.join(P2.PARTS, "assembled", i + ".png"))
    ys, xs = np.nonzero(P2.part(i, "mouth")[..., 3] > 128); ha = P2.part(i, "hair")[..., 3] > 128; hair_px = ha.sum(); ht = np.nonzero(ha.any(1))[0].min() if hair_px > 3000 else None
    hv = g.read(next(pr for m in g.j["meshes"] for pr in m["primitives"] if g.j["materials"][pr["material"]]["name"] == "head")["attributes"]["POSITION"])
    fit = FF.Fit(L, np.load(os.path.join(BASE_DIR, char + "_marks.npy")), FF_lip(g, char), np.c_[xs, ys].astype(float), float(hv[:, 1].max()), float(hv[:, 0].max()), None if ht in (None, 0) else ht,
                 P2.landmarks_z(os.path.join(P2.PARTS, "assembled", i + ".png")), np.load(os.path.join(BASE_DIR, char + "_marks_z.npy")))
    if char == "snow": bm, bc = FF.beard_mask(Image.open(os.path.join(P2.PARTS, "assembled", i + ".png")), L, P2.part(i, "mouth")[..., 3], rgb_(p["skin"]))
    if p["beard"] and bm is None: paint_beard(g, p["hair"])  # râu bạc không tách được khỏi da: dùng râu mẫu
    p["shapes"] = {}; p["expr"] = {"Cười": 0.3, "Há": min(fit.gap * 2.5, 0.2)} if fit.gap > 0.02 else {}  # hình mặt, tóc giờ theo ảnh; chỉ ảnh cười hở răng mới cần biểu cảm
    bake_shapes(g, p, char)
    FF.warp(g, fit)
    wpts = lambda a: fit.px(*np.nonzero(a)[::-1])
    FF.strip_fit(g, "eyebrows", wpts(P2.part(i, "brows")[..., 3] > 128))
    FF.strip_fit(g, "eyelashes", wpts(FF.biggest(P2.part(i, "lashes_left")) | FF.biggest(P2.part(i, "lashes_right"))), thin=0.5)
    if char == "rain": FF.lash_follow_lid(g)
    br = P2.part(i, "brows"); brow = np.median(br[br[..., 3] > 200][:, :3], axis=0) / 255
    if hair_px > 3000:
        FF.hair(g, fit, Image.open(os.path.join(P2.PARTS, "parts", i, "hair.png")), L, "hair_gen" if char == "rain" else "hair_base", ("head", "body", "top", "scarf", "shirt"), skin=rgb_(p["skin"]))
    if p["glasses"] and not FF.glasses(g, fit, os.path.join(P2.PARTS, "parts", i, "glasses.png"), lin):
        p["glasses"] = False
        for n in g.j["nodes"]:
            if n.get("name") == "glasses": n.pop("mesh", None); n.pop("skin", None)
    sk = hex2lin(p["skin"]); sk = (sk.mean() + (sk - sk.mean()) * 1.18) * 0.93  # ánh sáng web làm bạc màu vùng sáng: bù độ đậm và độ tươi
    skhex = "#%02x%02x%02x" % tuple((np.clip(srgb(sk), 0, 1) * 255).astype(int))
    if char == "snow":  # Snow: mày và mi dựng mới, Rain giữ khối gốc đã nắn
        FF.brows(g, fit, P2.part(i, "brows"), lin); FF.lashes(g, fit, [P2.part(i, "lashes_left"), P2.part(i, "lashes_right")], lin)
    g.tint("head", skhex); g.tint("body", skhex)
    hi = g.image_of("head"); skin = rgb_(skhex)
    g.set_image(hi, to_img(FF.paint_face(g, fit, Image.open(os.path.join(P2.PARTS, "parts", i, "face.png")), g.get_image(hi), lin, srgb, skin, None if char == "rain" else rgb_(p["lip"]), bm, bc)))
    if char == "rain": paint_lip(g, p["lip"])
    else: set_factor(g, "lip", hex2lin(p["lip"]) * 1.5)
    if hair_px <= 3000: g.tint("hair_gen" if p["style"] == "gen" else "hair", p["hair"])
    set_factor(g, "eyebrows", lin(brow))
    paint_iris(g, os.path.join(ROOT, "demo", p["iris"]), cfg["iris_disk"])
    out_dir = out_dir or os.path.join(ROOT, "output"); os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, i + ".glb"); save(g, path, cfg["credit"])
    return path, p


if __name__ == "__main__":
    a = sys.argv[1:]
    force = a[a.index("--char") + 1] if "--char" in a else None
    out = a[a.index("--out") + 1] if "--out" in a else None
    ids = sorted(os.listdir(os.path.join(P2.PARTS, "parts"))) if "--all" in a else [x for x in a if not x.startswith("--") and x not in (force, out)]
    bad = []
    for i in ids:
        t = time.time()
        try:
            path, p = make(i, force, out)
            print(i, p["char"], "kinh" if p["glasses"] else "-", "rau" if p["beard"] else "-", os.path.getsize(path) // 1024, "KB", round(time.time() - t, 1), "s", flush=True)
        except Exception as e:  # một ảnh lỗi không được dừng cả lô
            bad.append(i); print(i, "LOI", type(e).__name__, str(e)[:120], flush=True)
    print("xong", len(ids) - len(bad), "/", len(ids), "loi:", bad)
