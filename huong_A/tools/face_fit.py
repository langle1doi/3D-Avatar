# Nắn nhân vật nền cho khớp ảnh: dùng điểm mốc trên ảnh và trên nền (base_marks.py) làm một trường biến dạng (x, y) cho cả lưới,
# nên mặt, mắt, miệng, lông mày theo đúng ảnh mà chớp mắt, cười vẫn chạy. Kính dựng thẳng từ parts/<id>/glasses.png.
import cv2
import numpy as np
from PIL import Image

OVAL = [234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323, 454, 10, 109, 338]
EYES = [33, 133, 159, 145, 263, 362, 386, 374, 468, 473]
BROWS = [70, 63, 105, 66, 107, 336, 296, 334, 293, 300]
NOSE = [1, 94, 129, 358]
MOUTH = [61, 291, 0, 17, 13, 14, 37, 267, 84, 314]
ANCH = OVAL + EYES + BROWS + NOSE + MOUTH
EYE_NODES = ("eyes", "eye_cornea", "eye_corneas", "eye_dots")


LOCK = [33, 133, 159, 145, 263, 362, 386, 374, 234, 454, 70, 300]
DEEP = [1, 4, 94, 2, 129, 358, 50, 280, 101, 330, 205, 425, 187, 411, 152, 199, 10, 151]
DEPTH = 0.7  # ước lượng từ một ảnh không chắc: chỉ lấy một phần


def sstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def tps_fit(P, D, reg=2e-6):
    def U(r2): return 0.5 * r2 * np.log(r2 + 1e-12)
    n = len(P); K = U(((P[:, None] - P[None]) ** 2).sum(-1)) + reg * np.eye(n); A = np.c_[np.ones(n), P]
    w = np.linalg.solve(np.block([[K, A], [A.T, np.zeros((3, 3))]]), np.r_[D, np.zeros((3, 2))])
    return lambda X: U(((X[:, None] - P[None]) ** 2).sum(-1)) @ w[:n] + np.c_[np.ones(len(X)), X] @ w[n:]


def similarity(src, dst):
    """Xoay, co giãn, dời đều src sang dst (không làm méo tỉ lệ)."""
    a, b = src.mean(0), dst.mean(0); x, y = src - a, dst - b
    U, S, Vt = np.linalg.svd(y.T @ x); R = U @ np.diag([1, np.sign(np.linalg.det(U @ Vt))]) @ Vt
    s = S.sum() / (x ** 2).sum()
    return s, R, b - s * R @ a


def lip_profile(p, n=7):
    """Khóe miệng và viền trên, dưới của môi tại n cột, tính từ tập điểm môi."""
    x0, x1 = p[:, 0].min(), p[:, 0].max(); w = (x1 - x0) * 0.05; out = [p[p[:, 0].argmin()], p[p[:, 0].argmax()]]
    for t in np.linspace(0.12, 0.88, n):
        c = x0 + t * (x1 - x0); q = p[np.abs(p[:, 0] - c) < w, 1]
        if len(q) < 2: q = p[np.argsort(np.abs(p[:, 0] - c))[:4], 1]  # ít điểm (Snow)
        out += [(c, q.max()), (c, q.min())]
    return np.array(out, float)


class Fit:
    def __init__(self, L, base, lip_base, lip_px, head_top, head_w, hair_top=None, z_img=None, z_base=None):
        L = np.asarray(L, float)[:, :2] * (1, -1); self.base = base; k = EYES + MOUTH + NOSE
        self.s, self.R, self.t = similarity(L[k], base[k])
        self.tgt = self.to_world(L)
        top, chin = base[10, 1], base[152, 1]; self.chin = chin
        A = OVAL + EYES + BROWS + NOSE
        d = self.tgt[A] - base[A]; mid = (self.tgt[13] + self.tgt[14]) / 2 - (base[13] + base[14]) / 2
        lb, lt = lip_profile(lip_base), lip_profile(self.px(lip_px[:, 0], lip_px[:, 1]))
        self.gap = float(np.linalg.norm(self.tgt[13] - self.tgt[14]) / np.linalg.norm(self.tgt[10] - self.tgt[152]))
        if self.gap > 0.02:  # miệng há: không kéo môi dày ra, giữ độ dày nền, phần há do biểu cảm lo
            my = lt[:, 1].mean(); r = (lb[2::2, 1] - lb[3::2, 1]).max() / max((lt[2::2, 1] - lt[3::2, 1]).max(), 1e-4); lt[:, 1] = my + (lt[:, 1] - my) * r
        neck = np.array([[x, chin - 0.055] for x in (-0.09, -0.05, 0, 0.05, 0.09)])
        crown = np.array([[0, head_top], [-0.09, head_top - 0.035], [0.09, head_top - 0.035]])
        dc = d[[OVAL.index(10), OVAL.index(109), OVAL.index(338)]]
        if hair_top is not None:  # sọ nền cao hơn tóc trong ảnh thì hạ xuống, nếu không sẽ lộ da trên đỉnh
            ty = self.px(np.array([0.0]), np.array([float(hair_top)]))[0, 1] - 0.012
            dc[:, 1] = np.minimum(dc[:, 1], ty - crown[:, 1] + (0, -0.012, -0.012))
        ey = base[[33, 263], 1].mean(); ears = np.array([[-head_w, ey], [head_w, ey]])
        de = np.array([[self.tgt[234, 0] - 0.03 + head_w, d[0, 1]], [self.tgt[454, 0] + 0.03 - head_w, d[20, 1]]])  # tai nền to hơn tai trong ảnh
        self.P = np.r_[base[A], lb, [(base[13] + base[14]) / 2], neck, crown, ears]; self.D = np.r_[d, lt - lb, [mid], np.zeros((5, 2)), dc, de]
        self.u = tps_fit(self.P, self.D); self.w = None
        if z_img is not None and z_base is not None:  # độ cao mũi, độ đầy má: ước lượng từ ảnh, giữ yên quanh mắt để mí vẫn khép
            dz = -(z_img * self.s - z_base); dz -= np.median(dz[LOCK]); dz = np.clip(dz * DEPTH, -0.014, 0.014); dz[LOCK] = 0
            k = LOCK + DEEP; self.w = tps_fit(base[k], np.c_[dz[k], np.zeros(len(k))], 1e-5)

    def to_world(self, xy):  # xy đã lật trục y
        return self.s * xy @ self.R.T + self.t

    def px(self, x, y):  # điểm ảnh -> thế giới
        return self.to_world(np.stack([x, -y], -1).astype(float))

    def inv(self, W):  # thế giới -> điểm ảnh
        xy = (W - self.t) @ self.R / self.s
        return xy * (1, -1)

    def field(self, X):
        return self.u(X) * sstep(self.chin - 0.075, self.chin - 0.025, X[:, 1])[:, None]


def warp(g, f):
    """Áp trường biến dạng lên mọi lưới. Nhãn cầu giữ tròn: chỉ dời và co giãn đều. Biểu cảm được biến đổi theo để mi vẫn khép đúng."""
    rd = lambda x: x if isinstance(x, np.ndarray) else g.read(x)
    ew = lambda a: np.linalg.norm(a[33] - a[133]) + np.linalg.norm(a[263] - a[362])
    es = ew(f.tgt) / ew(f.base)
    node = {}
    for n in g.j["nodes"]:
        if "mesh" in n: node.setdefault(n["mesh"], n.get("name", ""))
    ev = np.concatenate([rd(pr["attributes"]["POSITION"]) for k, nm in node.items() if nm == "eyes" for pr in g.j["meshes"][k]["primitives"]]).astype(float)
    piv = {sd: np.r_[ev[sd * ev[:, 0] > 0][:, :2].mean(0), ev[sd * ev[:, 0] > 0][:, 2].max()] for sd in (-1, 1)}
    mv = {sd: f.field(piv[sd][:2][None])[0] for sd in piv}
    for k, nm in node.items():
        for pr in g.j["meshes"][k]["primitives"]:
            v = rd(pr["attributes"]["POSITION"]).astype(float)
            if nm in EYE_NODES:
                out = v.copy()
                for sd in (-1, 1):
                    m = sd * v[:, 0] > 0; out[m] = (v[m] - piv[sd]) * es + piv[sd]; out[m, :2] += mv[sd]
                pr["attributes"]["POSITION"] = out.astype("<f4"); continue
            X = v[:, :2]; d = f.field(X); e = 1e-3
            dx = (f.field(X + (e, 0)) - d) / e; dy = (f.field(X + (0, e)) - d) / e
            out = v.copy(); out[:, :2] += d
            if nm == "head" and f.w is not None:
                gate = lambda q: f.w(q)[:, 0] * sstep(0.03, 0.08, v[:, 2]) * sstep(f.chin - 0.03, f.chin + 0.005, q[:, 1]); z0 = gate(X); out[:, 2] += z0; dzx, dzy = (gate(X + (e, 0)) - z0) / e, (gate(X + (0, e)) - z0) / e
            else: dzx = dzy = None
            pr["attributes"]["POSITION"] = out.astype("<f4")
            if "NORMAL" in pr["attributes"]:  # pháp tuyến theo nghịch đảo chuyển vị của ma trận biến dạng
                n = rd(pr["attributes"]["NORMAL"]).astype(float); a, b, c, h = 1 + dx[:, 0], dy[:, 0], dx[:, 1], 1 + dy[:, 1]
                m = n.copy(); m[:, 0] = h * n[:, 0] - c * n[:, 1]; m[:, 1] = -b * n[:, 0] + a * n[:, 1]
                if dzx is not None: m[:, 0] -= dzx * n[:, 2]; m[:, 1] -= dzy * n[:, 2]
                pr["attributes"]["NORMAL"] = (m / np.linalg.norm(m, axis=1, keepdims=True).clip(1e-9)).astype("<f4")
            for t in pr.get("targets", []):
                dv = rd(t["POSITION"]).astype(float); nz = np.nonzero(np.abs(dv).sum(1))[0]
                if not len(nz): continue
                dv[nz, :2] += f.field(X[nz] + dv[nz, :2]) - d[nz]; t["POSITION"] = dv.astype("<f4")


def frame_mask(path):
    a = np.asarray(Image.open(path).convert("RGBA"), float); al = a[..., 3] > 128; xs = np.nonzero(al.any(0))[0]; lum = a[..., :3].mean(-1)
    for th in (55, 90, 125):  # gọng mảnh, sáng màu cần ngưỡng cao hơn
        m = (al & (lum < th)).astype(np.uint8); n, lab, st, _ = cv2.connectedComponentsWithStats(m)
        keep = np.isin(lab, 1 + np.nonzero(st[1:, cv2.CC_STAT_AREA] >= 25)[0]); xk = np.nonzero(keep.any(0))[0]
        if len(xk) and np.ptp(xk) > 0.7 * np.ptp(xs) and keep.sum() > 0.08 * al.sum(): break
    else:
        return None, None
    m = keep; col = np.median(a[m][:, :3], axis=0)
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(1 - m)  # lấp lỗ nhỏ (đinh tán, nhiễu), giữ hai tròng kính
    for c in range(1, n):
        if st[c, cv2.CC_STAT_AREA] < 400: m[lab == c] = 1
    return m > 0, col


def glasses(g, f, path, lin):
    """Gọng kính: đúng hình trong ảnh, đùn thành khối, uốn theo mặt. Càng kính dùng dáng chung vì ảnh không thấy."""
    m, col = frame_mask(path)
    if m is None: return True  # không tách được gọng: giữ kính dựng sẵn
    nid = next(n["mesh"] for n in g.j["nodes"] if n.get("name") == "glasses" and "mesh" in n)
    old = g.j["meshes"][nid]["primitives"][0]
    oz = old["attributes"]["POSITION"][:, 2].max()  # warp đã đổi thành mảng
    H = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 1.4); H2 = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 5.0); st = 2
    gy, gx = np.mgrid[0:H.shape[0] // st + 1, 0:H.shape[1] // st + 1] * st
    def sam(x, y):  # lấy mẫu song tuyến
        x = np.clip(x, 0, H.shape[1] - 2); y = np.clip(y, 0, H.shape[0] - 2); i, j = x.astype(int), y.astype(int); a, b = x - i, y - j
        return H[j, i] * (1 - a) * (1 - b) + H[j, i + 1] * a * (1 - b) + H[j + 1, i] * (1 - a) * b + H[j + 1, i + 1] * a * b
    C = (sam(gx[:-1, :-1].ravel() + 1.0, gy[:-1, :-1].ravel() + 1.0) > 0.5).reshape(gx[:-1, :-1].shape)
    idx = -np.ones(gx.shape, int); V = []; quads = []
    for y, x in zip(*np.nonzero(C)):
        q = []
        for yy, xx in ((y, x), (y, x + 1), (y + 1, x + 1), (y + 1, x)):
            if idx[yy, xx] < 0: idx[yy, xx] = len(V); V.append((gx[yy, xx], gy[yy, xx]))
            q.append(idx[yy, xx])
        quads.append(q)
    V = np.array(V, float); Q = np.array(quads)
    E = {e for a, b, c, d in Q for e in ((a, b), (b, c), (c, d), (d, a))}
    edge = np.array([e for e in E if (e[1], e[0]) not in E]); bd = np.unique(edge)
    for _ in range(4):  # kéo mép lưới về đúng đường viền trơn
        x, y = V[bd, 0], V[bd, 1]; gxx = (sam(x + 1, y) - sam(x - 1, y)) / 2; gyy = (sam(x, y + 1) - sam(x, y - 1)) / 2
        s = np.clip((0.5 - sam(x, y)) / (gxx ** 2 + gyy ** 2 + 1e-5), -2.5, 2.5); V[bd, 0] += s * gxx; V[bd, 1] += s * gyy
    W = f.px(V[:, 0], V[:, 1]); th = 0.0055
    if not ((W[:, 0] < 0).any() and (W[:, 0] > 0).any()): return True
    zf = lambda x: oz - 2.2 * x ** 2
    z = zf(W[:, 0]); front = np.c_[W, z]; back = np.c_[W, z - th]
    ix0, iy0 = V[:, 0].round().astype(int).clip(3, H.shape[1] - 4), V[:, 1].round().astype(int).clip(3, H.shape[0] - 4)
    ex, ey = H2[iy0, ix0 + 3] - H2[iy0, ix0 - 3], H2[iy0 + 3, ix0] - H2[iy0 - 3, ix0]
    nf = np.c_[-ex * 0.9, ey * 0.9, np.ones(len(V))]; nf /= np.linalg.norm(nf, axis=1, keepdims=True)
    n = len(V); pos = [front, back]; nor = [nf, np.tile((0.0, 0.0, -1.0), (n, 1))]
    tri = [Q[:, [0, 2, 1]], Q[:, [0, 3, 2]], Q[:, [0, 1, 2]] + n, Q[:, [0, 2, 3]] + n]
    a, b = edge[:, 0], edge[:, 1]; d = W[b] - W[a]; sn = np.c_[d[:, 1], -d[:, 0], np.zeros(len(d))]; sn /= np.linalg.norm(sn, axis=1, keepdims=True).clip(1e-9)
    b0 = n * 2; k = len(edge)
    pos.append(np.r_[front[a], front[b], back[a], back[b]]); nor.append(np.tile(sn, (4, 1)))
    ix = np.arange(k) + b0; tri += [np.c_[ix, ix + k, ix + 3 * k], np.c_[ix, ix + 3 * k, ix + 2 * k]]
    for sd in (-1, 1):  # càng kính: thanh dẹt từ bản lề ra sau tai
        side = W[sd * W[:, 0] > 0]; e = side[np.argmax(np.abs(side[:, 0]))]; hy = np.percentile(side[np.abs(side[:, 0]) > np.abs(e[0]) - 0.006, 1], 70)
        z0, z1 = zf(e[0]) - th, -0.005; S = 10; b0 = sum(len(p) for p in pos); pts = []
        for t in np.linspace(0, 1, S):
            x, y, zz = e[0] + sd * 0.012 * t, hy - 0.012 * t ** 3, z0 + (z1 - z0) * t
            pts += [(x - 0.0014, y + 0.0025, zz), (x + 0.0014, y + 0.0025, zz), (x + 0.0014, y - 0.0025, zz), (x - 0.0014, y - 0.0025, zz)]
        pos.append(np.array(pts)); nor.append(np.tile((sd * 1.0, 0.0, 0.0), (len(pts), 1)))
        for s in range(S - 1):
            for c in range(4):
                p, q = b0 + s * 4 + c, b0 + s * 4 + (c + 1) % 4; tri.append(np.array([[p, q, q + 4], [p, q + 4, p + 4]]))
    P = np.concatenate(pos).astype("<f4"); N = np.concatenate(nor).astype("<f4"); T = np.concatenate(tri).astype("<u4").reshape(-1, 1)
    J = g.read(old["attributes"]["JOINTS_0"]); W0 = g.read(old["attributes"]["WEIGHTS_0"]); c = len(P)
    pr = {"attributes": {"POSITION": P, "NORMAL": N, "JOINTS_0": np.tile(J[0], (c, 1)), "WEIGHTS_0": np.tile(W0[0], (c, 1)).astype("<f4")}, "indices": T, "material": old["material"]}
    g.j["meshes"][nid]["primitives"] = [pr]
    g.j["meshes"][nid].pop("weights", None); g.j["meshes"][nid].pop("extras", None)
    mt = g.j["materials"][old["material"]]; b = mt.setdefault("pbrMetallicRoughness", {}); b.pop("baseColorTexture", None)
    b["baseColorFactor"] = [float(x) for x in lin(col / 255)] + [1.0]; b["metallicFactor"] = 0.0; b["roughnessFactor"] = 0.42; mt["doubleSided"] = True
    return True


def beard_mask(a, L, mouth, skin):
    """Râu trong ảnh: vùng tối hẳn so với da, dưới mũi, ngoài miệng. Trả về mặt nạ và màu râu."""
    a = np.asarray(a.convert("RGBA"), np.float32); L = np.asarray(L, float); lum = a[..., :3].mean(-1); m = (a[..., 3] > 128) & (lum < skin.mean() * 0.5)
    m[:int(L[1, 1])] = False; hull = np.zeros(m.shape, np.uint8); cv2.fillPoly(hull, [cv2.convexHull(np.int32(L[OVAL[:21], :2]))], 1); m &= hull > 0
    m &= cv2.dilate((mouth > 128).astype(np.uint8), np.ones((5, 5), np.uint8)) == 0
    cx = int(L[[234, 454], 0].mean()); fl = cv2.warpAffine(m[:, ::-1].astype(np.uint8), np.float32([[1, 0, 2 * cx - m.shape[1]], [0, 1, 0]]), m.shape[1::-1])
    m &= cv2.dilate(fl, np.ones((17, 17), np.uint8)) > 0  # râu có ở cả hai bên; bóng đổ thường chỉ một bên
    n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8))
    for c in range(1, n):
        if st[c, cv2.CC_STAT_AREA] < 60: m[lab == c] = False
    return (cv2.GaussianBlur(m.astype(np.float32), (0, 0), 1.6), np.median(a[m][:, :3], axis=0)) if m.sum() > 600 else (None, None)


def biggest(a):
    """Mảng liền lớn nhất của một phần cắt, bỏ các đốm nhiễu."""
    m = (a[..., 3] > 128).astype(np.uint8); n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    return lab == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA])) if n > 1 else m > 0


def strip_fit(g, name, pts, thin=None):
    """Lông mày, mi là dải dài: dồn dải 3D về đúng đường cong và độ dày đo trên ảnh, từng bên một."""
    nid = next((n["mesh"] for n in g.j["nodes"] if n.get("name") == name and "mesh" in n), None)
    if nid is None: return
    def prof(q):
        x0, x1 = q[:, 0].min(), q[:, 0].max(); t = (q[:, 0] - x0) / max(x1 - x0, 1e-6); us = (np.arange(10) + 0.5) / 10; top, bot, uu = [], [], []
        for u in us:
            y = q[np.abs(t - u) < 0.08, 1]
            if len(y) > 1: uu.append(u); top.append(y.max()); bot.append(y.min())
        d = min(3, len(uu) - 1)
        return x0, x1, np.polyfit(uu, top, d), np.polyfit(uu, bot, d)
    for pr in g.j["meshes"][nid]["primitives"]:
        v = pr["attributes"]["POSITION"]; v = (v if isinstance(v, np.ndarray) else g.read(v)).astype(float); out = v.copy()
        for sd in (-1, 1):
            m = sd * v[:, 0] > 0; t = pts[sd * pts[:, 0] > 0]
            if m.sum() < 4 or len(t) < 12: continue
            a0, a1, at, ab = prof(v[m, :2]); b0, b1, bt, bb = prof(t)
            u = (v[m, 0] - a0) / (a1 - a0); c0 = (np.polyval(at, u) + np.polyval(ab, u)) / 2; h0 = (np.polyval(at, u) - np.polyval(ab, u)).clip(2e-4)
            c1 = (np.polyval(bt, u) + np.polyval(bb, u)) / 2; h1 = (np.polyval(bt, u) - np.polyval(bb, u)).clip(6e-4)
            out[m, 0] = b0 + u * (b1 - b0); out[m, 1] = c1 + (v[m, 1] - c0) * h1 / h0
            if thin:  # mi: giữ đường cong theo mí vừa nắn, chỉ mỏng lại; co ngang vừa phải để khi chớp còn phủ hết mí
                out[m, 1] = c0 + (v[m, 1] - c0) * thin; out[m, 0] = (out[m, 0] + v[m, 0]) / 2
        pr["attributes"]["POSITION"] = out.astype("<f4")


def lash_follow_lid(g, lift=0.0015):
    """Khi chớp, mi phải nằm trên mí đã khép: nền gốc để mi thấp hơn mặt da khoảng 5mm nên phần giữa mi bị da che, chỉ hai đầu mi lòi ra."""
    rd = lambda x: x if isinstance(x, np.ndarray) else g.read(x)
    mesh = lambda name: next((g.j["meshes"][n["mesh"]] for n in g.j["nodes"] if n.get("name") == name and "mesh" in n), None)
    hm, lm = mesh("head"), mesh("eyelashes")
    if hm is None or lm is None or "extras" not in hm or "extras" not in lm: return
    hn, ln = hm["extras"]["targetNames"], lm["extras"]["targetNames"]; hp = next((p for p in hm["primitives"] if p.get("targets")), None)
    if hp is None: return
    hv = rd(hp["attributes"]["POSITION"]).astype(float)
    for pr in lm["primitives"]:
        lv = rd(pr["attributes"]["POSITION"]).astype(float)
        for name in ("eyeBlinkLeft", "eyeBlinkRight"):
            if name not in ln or name not in hn: continue
            t = pr["targets"][ln.index(name)]; dl = rd(t["POSITION"]).astype(float); mv = np.abs(dl).sum(1) > 1e-6
            if not mv.any(): continue
            ch = hv + rd(hp["targets"][hn.index(name)]["POSITION"]).astype(float); ch = ch[ch[:, 2] > 0.05]; pc = lv[mv] + dl[mv]
            y0 = pc[:, 1].min(); pc[:, 1] = y0 + 0.35 * (pc[:, 1] - y0)  # mi nhắm: dải mi dẹp lại sát đường mí, đuôi mi không vươn lên dọc theo mí
            near = ((pc[:, None, :2] - ch[None, :, :2]) ** 2).sum(-1).argsort(1)[:, :6]
            pc[:, 2] = np.maximum(pc[:, 2], ch[near, 2].max(1) + lift); dl[mv] = pc - lv[mv]; t["POSITION"] = dl.astype("<f4")


def hair(g, f, im, L, node, bodies, skin=None):
    """Khối tóc mới theo đúng viền trong ảnh: mặt trước ôm đầu, mặt sau khép kín, ảnh tóc dán thẳng lên. Ảnh cắt ngang tóc thì tự nối dài."""
    a0 = np.array(im.convert("RGBA")); H, W = a0.shape[:2]; T, B = 80, 70; st = 3; HH = T + H + B; L = np.asarray(L, float)
    al = cv2.morphologyEx((a0[..., 3] > 128).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    if skin is not None:  # viền da còn sót ở mép ảnh tóc (quanh khuôn mặt, ngoài rìa): thay bằng màu tóc lân cận
        band = (al > 0) & (cv2.erode(al, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))) == 0)
        sk_ = cv2.dilate((band & (np.linalg.norm(a0[..., :3].astype(np.float32) - np.asarray(skin, np.float32), axis=-1) < 75)).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        ok_ = ((al > 0) & ~sk_).astype(np.float32); w_ = cv2.GaussianBlur(ok_, (0, 0), 4)
        col_ = cv2.GaussianBlur(a0[..., :3].astype(np.float32) * ok_[..., None], (0, 0), 4) / (w_[..., None] + 1e-4)
        fix_ = sk_ & (w_ > 0.02); a0[..., :3][fix_] = col_[fix_].clip(0, 255).astype(np.uint8)
    n, lab, stt, _ = cv2.connectedComponentsWithStats(al)
    for c in range(1, n):  # đốm nhỏ (kẹp tóc, nhiễu) không dựng thành khối
        if stt[c, cv2.CC_STAT_AREA] < 300: al[lab == c] = 0
    A = np.zeros((HH, W), np.uint8); A[T:T + H] = al; rgb = np.zeros((HH, W, 3), np.float32); rgb[T:T + H] = a0[..., :3]
    if al[0].any():  # đỉnh tóc bị cắt: nối thành vòm
        xs = np.nonzero(al[0])[0]; c, hw = (xs.min() + xs.max()) / 2, (xs.max() - xs.min()) / 2; tt = min(T - 4, 0.3 * hw)
        for k in range(1, int(tt)):
            h = hw * np.sqrt(1 - (k / tt) ** 2); A[T - k, int(c - h):int(c + h)] = 1
    if al[-1].any():
        for y in range(1, B):
            k = 2 * int(70 * (1 - np.sqrt(1 - (y / B) ** 2))) + 1; A[T + H + y - 1] = cv2.erode(al[-1:], np.ones((1, k), np.uint8)).ravel()
    E = cv2.erode(A, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))); e_ = cv2.GaussianBlur(E.astype(np.float32), (0, 0), 3)[..., None] + 1e-4
    in_ = (A[..., None] > 0) & (np.arange(HH)[:, None, None] >= T) & (np.arange(HH)[:, None, None] < T + H)
    rgb = np.where(in_ & (E[..., None] == 0), cv2.GaussianBlur(rgb * E[..., None], (0, 0), 3) / e_, rgb)  # mép ảnh cắt lẫn màu đen của nền: lấy màu từ trong tóc
    rgb[:T] = rgb[2 * T - 1:T - 1:-1]; rgb[T + H:] = rgb[T + H - 1:T + H - B - 1:-1]  # phần nối dài lấy đối xứng để sợi tóc liền mạch
    gy, gx = np.mgrid[0:HH // st + 1, 0:W // st + 1]; gx = gx * st; gy = gy * st; gh, gw = gx.shape
    zf0 = np.full(gx.shape, -9.0, np.float32); zb0 = np.full(gx.shape, 9.0, np.float32)
    uu, vv = np.meshgrid(np.linspace(0, 1, 6), np.linspace(0, 1, 6)); ok = uu + vv <= 1; bary = np.c_[uu[ok], vv[ok], 1 - uu[ok] - vv[ok]]
    rd = lambda x: x if isinstance(x, np.ndarray) else g.read(x)
    for n in g.j["nodes"]:
        if n.get("name") in bodies and "mesh" in n:
            for pr in g.j["meshes"][n["mesh"]]["primitives"]:
                v = rd(pr["attributes"]["POSITION"]).astype(float); tr = g.read(pr["indices"]).reshape(-1, 3)
                pts = (v[tr][:, None] * bary[None, :, :, None]).sum(2).reshape(-1, 3)
                q = f.inv(pts[:, :2]) + (0, T); i = (q[:, 0] / st).round().astype(int); j = (q[:, 1] / st).round().astype(int); m = (i >= 0) & (i < gw) & (j >= 0) & (j < gh)
                np.maximum.at(zf0, (j[m], i[m]), pts[m, 2]); np.minimum.at(zb0, (j[m], i[m]), pts[m, 2])
    ins = zf0 > -5; ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)); hf = cv2.dilate(zf0, ker); hb = cv2.erode(zb0, ker); ins = hf > -5
    hb = np.minimum(hb, cv2.erode(hb, np.ones((61, 1), np.uint8)))  # chỗ sọ hở (Snow): lấy mặt sau của sọ ở hàng trên, dưới để tóc phủ qua lỗ
    oval = np.int32(L[OVAL[:21], :2] + (0, T)); U = A.copy(); cv2.fillPoly(U, [cv2.convexHull(oval)], 1)
    hm = cv2.resize(ins.astype(np.uint8), (W, HH), interpolation=cv2.INTER_NEAREST); ey = int(L[33, 1]) + T; sk = hm.copy(); hm[:, :int(L[234, 0])] = 0; hm[:, int(L[454, 0]):] = 0; U[:ey] |= hm[:ey]
    U[ey - 5:int(L[[234, 454], 1].max()) + T + 25, int(L[234, 0]):int(L[454, 0])] = 1  # đoạn giữa mắt và tai, để mặt sau của tóc ngắn không hở
    A0 = A.copy(); top = A.argmax(0); cols = np.nonzero(A.any(0))[0]
    for x in cols: A[:top[x], x] |= sk[:top[x], x]  # sọ nhô cao hơn tóc trong ảnh: phủ thêm tóc để không lộ da
    med = np.median(rgb[A0 > 0], axis=0); w = cv2.GaussianBlur(A0.astype(np.float32), (0, 0), 30)[..., None]
    fill = cv2.GaussianBlur(rgb * A0[..., None], (0, 0), 30) / (w + 1e-4); k = np.clip(w * 4, 0, 1); fill = fill * k + med * (1 - k)
    noise = cv2.GaussianBlur(np.random.default_rng(1).standard_normal((HH, W)).astype(np.float32), (0, 0), sigmaX=1.6, sigmaY=45); fill = fill * (1 + 0.2 * noise[..., None] / noise.std())  # sợi tóc giả cho vùng ảnh không thấy
    tex = np.where(A0[..., None] > 0, rgb, fill).clip(0, 255).astype(np.uint8)
    rng = np.random.default_rng(2); warp_ = cv2.GaussianBlur(rng.standard_normal((HH, W)).astype(np.float32), (0, 0), 40); warp_ /= warp_.std()
    xx = np.arange(W, dtype=np.float32)[None] + 14 * warp_; clump = 0.5 + 0.5 * np.cos(2 * np.pi * xx / 62)  # lọn tóc mặt sau, ảnh không thấy nên dựng theo quy tắc
    back = np.zeros((HH, W, 3), np.float32)
    lum_ = rgb.mean(-1)
    for y in range(HH):  # mặt sau: màu chính của từng hàng tóc, không lấy vùng tối sát mặt
        xs = np.nonzero(A0[y])[0]
        back[y] = np.median(rgb[y, xs][lum_[y, xs] >= np.median(lum_[y, xs])], axis=0) if len(xs) > 3 else med
    back = cv2.GaussianBlur(back, (0, 0), sigmaX=1, sigmaY=24) * (1 + 0.16 * noise[..., None] / noise.std()) * (0.82 + 0.26 * clump[..., None])
    n, lab, stt, _ = cv2.connectedComponentsWithStats(1 - U)
    for c in range(1, n):  # lấp lỗ bên trong
        x, y, w_, h_ = stt[c, :4]
        if x > 0 and y > 0 and x + w_ < W and y + h_ < HH: U[lab == c] = 1
    bot = max(np.nonzero(A.any(1))[0].max(), int(L[13, 1]) + T); U[bot + 1:] = 0  # tóc ngắn: mặt sau dừng ở đuôi tóc, nhưng luôn che hết gáy (sọ Snow hở phía dưới tóc)
    Af = A.copy(); cv2.fillPoly(Af, [cv2.convexHull(oval)], 1); nf, labf, stf, _ = cv2.connectedComponentsWithStats(1 - Af)
    for c in range(1, nf):  # lấp lỗ bên trong, chỉ còn đường viền ngoài của khối tóc
        x, y, w_, h_ = stf[c, :4]
        if x > 0 and y > 0 and x + w_ < W and y + h_ < HH: Af[labf == c] = 1
    fe = (1 - sstep(0, 26, cv2.distanceTransform(Af, cv2.DIST_L2, 5)))[..., None]  # rìa ngoài mặt trước chuyển về màu mặt sau, hai mặt gặp nhau cùng màu nên không lộ đường ranh
    tex = np.concatenate([(tex * (1 - fe) + back * fe).clip(0, 255).astype(np.uint8), back.clip(0, 255).astype(np.uint8)], 1)
    samp = lambda M: M[gy.clip(0, HH - 1), gx.clip(0, W - 1)]
    Dn = samp(cv2.distanceTransform(U, cv2.DIST_L2, 5)); DA = samp(cv2.distanceTransform(A, cv2.DIST_L2, 5))
    chin = L[152, 1] + T; R = 0.05 + 0.08 * (1 - sstep(chin - 40, chin + 40, gy.astype(float)))
    b = R * np.sqrt(1 - (1 - np.clip(Dn * f.s / R, 0, 1)) ** 2); zm = float(np.median((hf + hb)[ins & (gy < chin)]) / 2); off = 0.008
    zf = (zm + b).astype(np.float32); zb = (zm - b).astype(np.float32)
    for sg in (6, 3, 1.5):
        zf[ins] = np.maximum(zf[ins], hf[ins] + off); zb[ins] = np.minimum(zb[ins], hb[ins] - off)
        zf = cv2.GaussianBlur(zf, (0, 0), sg); zb = cv2.GaussianBlur(zb, (0, 0), sg)
    gray = rgb.mean(-1); wa = cv2.GaussianBlur(A0.astype(np.float32), (0, 0), 16) + 1e-4; rel = cv2.GaussianBlur(gray * A0, (0, 0), 2.5) / (cv2.GaussianBlur(A0.astype(np.float32), (0, 0), 2.5) + 1e-4) - cv2.GaussianBlur(gray * A0, (0, 0), 16) / wa
    rel = np.clip(samp(rel) / (rel[A0 > 0].std() + 1e-4), -2, 2).astype(np.float32); zf = zf + 0.0034 * rel * sstep(2, 22, DA).astype(np.float32)  # lọn tóc nổi theo nếp sáng tối trong ảnh
    zb = zb - 0.007 * samp(clump) * sstep(2, 26, Dn).astype(np.float32)
    zn = zf.copy()  # bóng sáng tính trên mặt chưa chui xuống da, tránh viền tối quanh mặt
    wi = cv2.GaussianBlur(ins.astype(np.float32), (0, 0), 3) + 1e-4; hs = cv2.GaussianBlur(np.where(ins, hf, 0).astype(np.float32), (0, 0), 3) / wi
    t = sstep(0, 16, DA); face = ins & (Dn > 14); zf = np.where(face, hs - 0.002 + (zf - hs + 0.002) * t, zf); zf = cv2.GaussianBlur(zf, (0, 0), 1.2)  # mép tóc trên da chui xuống, không hở
    e = sstep(0, 9, Dn); zf = zm + (zf - zm) * e; zb = zm + (zb - zm) * e  # hai lớp gặp nhau ở viền ngoài
    cover = ins & ((t >= 1) | (Dn <= 14)); zf = np.where(cover, np.maximum(zf, hf + 0.003), zf); zb = np.where(ins & (Dn > 3), np.minimum(zb, hb - 0.003), zb)  # tai không xuyên qua tóc
    P, UV, Nn, tri = [], [], [], []
    for M, z, sgn in ((A, zf, 1), (U, zb, -1)):
        Bl = cv2.GaussianBlur(M.astype(np.float32), (0, 0), 2.2); mask = Bl[(gy[:-1, :-1] + 1).clip(0, HH - 1), (gx[:-1, :-1] + 1).clip(0, W - 1)] > 0.5
        cnt = np.zeros(gx.shape, int); cnt[:-1, :-1] += mask; cnt[1:, :-1] += mask; cnt[:-1, 1:] += mask; cnt[1:, 1:] += mask; use = cnt > 0
        x = gx.astype(float); y = gy.astype(float); bd = use & (cnt < 4)
        for _ in range(4):  # kéo mép về viền trơn
            xi = x[bd].round().astype(int).clip(1, W - 2); yi = y[bd].round().astype(int).clip(1, HH - 2)
            dx = (Bl[yi, xi + 1] - Bl[yi, xi - 1]) / 2; dy = (Bl[yi + 1, xi] - Bl[yi - 1, xi]) / 2; s_ = np.clip((0.5 - Bl[yi, xi]) / (dx ** 2 + dy ** 2 + 1e-5), -2, 2); x[bd] += s_ * dx; y[bd] += s_ * dy
        z = z.copy(); z[bd & (Dn < 7)] = zm  # hai lớp dùng chung đường viền, không nứt
        wxy = f.px(x[use], y[use] - T); dz = 1 / (st * f.s); zs = zn if sgn > 0 else z; dzx = cv2.Sobel(zs, cv2.CV_32F, 1, 0, ksize=3) / 8 * dz; dzy = -cv2.Sobel(zs, cv2.CV_32F, 0, 1, ksize=3) / 8 * dz
        nr = np.stack([-dzx * sgn, -dzy * sgn, np.full(z.shape, float(sgn), np.float32)], -1)[use]; nr /= np.linalg.norm(nr, axis=1, keepdims=True)
        idm = -np.ones(gx.shape, int); b0 = sum(len(p) for p in P); idm[use] = b0 + np.arange(use.sum())
        P.append(np.c_[wxy, z[use]]); Nn.append(nr); UV.append(np.c_[(x[use] / W + (0 if sgn > 0 else 1)) / 2, y[use] / HH])
        j, i = np.nonzero(mask); a, b_, c, d = idm[j, i], idm[j, i + 1], idm[j + 1, i + 1], idm[j + 1, i]
        tri.append(np.r_[np.c_[a, c, b_], np.c_[a, d, c]] if sgn > 0 else np.r_[np.c_[a, b_, c], np.c_[a, c, d]])
    P = np.concatenate(P); nid = next(n["mesh"] for n in g.j["nodes"] if n.get("name") == node and "mesh" in n); old = g.j["meshes"][nid]["primitives"][0]
    op = rd(old["attributes"]["POSITION"]).astype(float); J = g.read(old["attributes"]["JOINTS_0"]); Wt = g.read(old["attributes"]["WEIGHTS_0"]).astype("<f4")
    sub = np.arange(0, len(op), max(1, len(op) // 4000)); near = sub[np.concatenate([((P[k:k + 3000, None] - op[sub][None]) ** 2).sum(-1).argmin(1) for k in range(0, len(P), 3000)])]
    Nn = np.concatenate(Nn); UV = np.concatenate(UV); fr, bk = tri; lx = P[fr, 0].mean(1) < 0
    def sub(t):  # mỗi mảng tóc một phần riêng để sửa, thay từng mảng về sau
        u, inv = np.unique(t, return_inverse=True)
        return {"attributes": {"POSITION": P[u].astype("<f4"), "NORMAL": Nn[u].astype("<f4"), "TEXCOORD_0": UV[u].astype("<f4"), "JOINTS_0": J[near[u]], "WEIGHTS_0": Wt[near[u]]}, "indices": inv.reshape(-1, 1).astype("<u4"), "material": old["material"]}
    g.j["meshes"][nid]["primitives"] = [sub(t) for t in (fr[lx], fr[~lx], bk) if len(t)]; g.j["meshes"][nid].pop("weights", None); g.j["meshes"][nid].pop("extras", None)
    mt = g.j["materials"][old["material"]]; mt.pop("normalTexture", None); mt["doubleSided"] = True; mt["pbrMetallicRoughness"]["roughnessFactor"] = 0.95;  # tóc bóng (0.55) phản chiếu môi trường rất sáng khi nhìn nghiêng, thành dải sáng giống đường ranh giữa hai mặt tóc; mt["pbrMetallicRoughness"]["metallicFactor"] = 0.0
    mt["emissiveTexture"] = {"index": mt["pbrMetallicRoughness"]["baseColorTexture"]["index"]}; mt["emissiveFactor"] = [0.42] * 3  # ảnh đã có sẵn sáng tối, cần bớt phụ thuộc đèn để không tối hơn ảnh
    g.set_image(g.image_of(mt["name"]), Image.fromarray(tex))


def paint_face(g, f, face_png, tex_img, lin, srgb, skin, lip=None, beard=None, beard_col=None):
    """Sắc da theo ảnh: má hồng, vùng sáng tối đo trên face.png (chỉ phần tần số thấp) rồi đổ lên ảnh da. Gầm mũi được làm dịu."""
    a = np.asarray(face_png.convert("RGBA"), np.float32); m = (a[..., 3] > 200).astype(np.float32); s_ = 22
    w = cv2.GaussianBlur(m, (0, 0), s_) + 1e-4; low = cv2.GaussianBlur(a[..., :3] * m[..., None], (0, 0), s_) / w[..., None]
    med = np.median(a[m > 0][:, :3], axis=0); cx = int(f.inv(np.array([[0.0, f.chin]]))[0, 0]); ratio = low / med; ratio[:, :] = (ratio + cv2.warpAffine(ratio[:, ::-1], np.float32([[1, 0, 2 * cx - ratio.shape[1]], [0, 1, 0]]), ratio.shape[1::-1], borderMode=cv2.BORDER_REPLICATE)) / 2  # ánh sáng trong ảnh lệch một bên: lấy trung bình hai bên
    lum = ratio.mean(-1, keepdims=True); ratio = np.clip(1 + (ratio - lum), 0.965, 1.04); conf = sstep(0.35, 0.9, cv2.GaussianBlur(m, (0, 0), 20))
    head = next(x for x in g.j["meshes"] if any(g.j["materials"][p["material"]]["name"] == "head" for p in x["primitives"])); pr = next(p for p in head["primitives"] if g.j["materials"][p["material"]]["name"] == "head")
    v = pr["attributes"]["POSITION"]; v = (v if isinstance(v, np.ndarray) else g.read(v)).astype(float); uv = g.read(pr["attributes"]["TEXCOORD_0"]); tr = g.read(pr["indices"]).reshape(-1, 3)
    q = f.inv(v[:, :2]); i = q[:, 0].round().astype(int).clip(0, a.shape[1] - 1); j = q[:, 1].round().astype(int).clip(0, a.shape[0] - 1)
    front = v[:, 2] > 0.06; col = ratio[j, i]; cf = (conf[j, i] * front).astype(np.float32)
    nose = sstep(0.034, 0.018, np.linalg.norm(v[:, :2] - (f.tgt[2] + (0, 0.002)), axis=1))
    W, H = tex_img.size; RW = 1024; px_ = np.int32(uv * RW); ring = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
    lm = np.zeros(len(v), np.float32)
    if lip is not None:  # Snow không có vùng môi riêng: tô theo đường viền môi sau khi nắn
        pg = np.float32(f.tgt[ring]); lm = np.array([cv2.pointPolygonTest(pg, (float(x), float(y)), False) >= 0 for x, y in v[:, :2]], np.float32) * (v[:, 2] > 0.1)
    map_l = np.zeros((RW, RW), np.float32); map_b = np.zeros((RW, RW), np.float32); bv = beard[j, i] * (v[:, 2] > 0.05) if beard is not None else None
    map_r = np.ones((RW, RW, 3), np.float32); map_c = np.zeros((RW, RW), np.float32); map_n = np.zeros((RW, RW), np.float32)
    for t in tr:
        c = cf[t].mean()
        if c > 0.01 or nose[t].mean() > 0.01:
            p = px_[t]; cv2.fillConvexPoly(map_r, p, col[t].mean(0).tolist()); cv2.fillConvexPoly(map_c, p, float(c)); cv2.fillConvexPoly(map_n, p, float(nose[t].mean()))
        if lm[t].max() > 0: cv2.fillConvexPoly(map_l, px_[t], float(lm[t].mean()))
        if bv is not None and bv[t].max() > 0.02: cv2.fillConvexPoly(map_b, px_[t], float(bv[t].mean()))
    map_r = cv2.GaussianBlur(map_r, (0, 0), 4); map_c = cv2.GaussianBlur(map_c, (0, 0), 4); map_n = cv2.GaussianBlur(map_n, (0, 0), 3)
    map_r = cv2.resize(map_r, (W, H)); map_c = cv2.resize(map_c, (W, H))[..., None]; map_n = cv2.resize(map_n, (W, H))[..., None]
    t = lin(np.asarray(tex_img, np.float32) / 255); r = map_r ** 2.2
    t = t * (1 - map_n * 0.8) + lin(skin / 255)[None, None] * 0.82 * map_n * 0.8  # lỗ mũi vẽ sẵn của nền đậm hơn ảnh Pixar
    t = t * (1 - map_c + map_c * r)
    if lip is not None:
        m = cv2.resize(cv2.GaussianBlur(map_l, (0, 0), 3), (W, H))[..., None] * 0.75; t = t * (1 - m) + lin(lip / 255)[None, None] * m
    if beard is not None:
        m = (cv2.resize(cv2.GaussianBlur(map_b, (0, 0), 2), (W, H))[..., None] * 1.5).clip(0, 0.96); t = t * (1 - m) + lin(beard_col / 255)[None, None] * m
    return t


def head_part(g):
    head = next(m for m in g.j["meshes"] if any(g.j["materials"][p["material"]]["name"] == "head" for p in m["primitives"]))
    return head, next(p for p in head["primitives"] if g.j["materials"][p["material"]]["name"] == "head")


def add_to_head(g, P, N, T, idx, w, name, rgb):
    """Thêm một phần vào lưới đầu. Mỗi điểm mượn chuyển động của các điểm da gần nó, nên tự đi theo mọi biểu cảm."""
    head, hp = head_part(g); rd = lambda x: x if isinstance(x, np.ndarray) else g.read(x)
    tg = [{"POSITION": (rd(t["POSITION"]).astype(float)[idx] * w[..., None]).sum(1).astype("<f4")} for t in hp.get("targets", [])]
    g.j["materials"].append({"name": name, "doubleSided": True, "pbrMetallicRoughness": {"baseColorFactor": [float(x) for x in rgb] + [1.0], "metallicFactor": 0.0, "roughnessFactor": 0.7}})
    pr = {"attributes": {"POSITION": P.astype("<f4"), "NORMAL": N.astype("<f4"), "JOINTS_0": g.read(hp["attributes"]["JOINTS_0"])[idx[:, 0]], "WEIGHTS_0": g.read(hp["attributes"]["WEIGHTS_0"])[idx[:, 0]].astype("<f4")}, "indices": T.astype("<u4").reshape(-1, 1), "material": len(g.j["materials"]) - 1}
    if tg: pr["targets"] = tg
    head["primitives"].append(pr)


def nearest(P, X, k=4):
    """k điểm da gần nhất và trọng số theo khoảng cách."""
    d = np.concatenate([((P[i:i + 2000, None] - X[None]) ** 2).sum(-1) for i in range(0, len(P), 2000)]); idx = np.argsort(d, 1)[:, :k]
    w = 1 / (np.take_along_axis(d, idx, 1) + 1e-8); return idx, w / w.sum(1, keepdims=True)


def brows(g, f, a, lin):
    """Lông mày: dựng mới đúng hình trong ảnh, ôm sát da. Khối gốc của Snow góc cạnh, nắn không ra đầu thuôn."""
    al = a[..., 3] > 128
    if al.sum() < 300: return False
    core = (a[..., 3] > 175).astype(np.float32)  # bỏ lông tơ ở mép ảnh cắt, nếu không mày dày và lởm chởm
    k = 3; H = cv2.GaussianBlur(cv2.resize(core, None, fx=k, fy=k, interpolation=cv2.INTER_LINEAR), (0, 0), 5); st = 3
    gy, gx = np.mgrid[0:H.shape[0] // st, 0:H.shape[1] // st] * st; C = H[(gy[:-1, :-1] + 1), (gx[:-1, :-1] + 1)] > 0.5
    idx = -np.ones(gx.shape, int); V = []; Q = []
    for y, x in zip(*np.nonzero(C)):
        q = []
        for yy, xx in ((y, x), (y, x + 1), (y + 1, x + 1), (y + 1, x)):
            if idx[yy, xx] < 0: idx[yy, xx] = len(V); V.append((gx[yy, xx], gy[yy, xx]))
            q.append(idx[yy, xx])
        Q.append(q)
    V = np.array(V, float); Q = np.array(Q); E = {e for p, q, r, t in Q for e in ((p, q), (q, r), (r, t), (t, p))}
    bd = np.unique(np.array([e for e in E if (e[1], e[0]) not in E]))
    def sam(M, x, y):
        i, j = x.round().astype(int).clip(1, M.shape[1] - 2), y.round().astype(int).clip(1, M.shape[0] - 2); return i, j, M[j, i]
    for _ in range(4):  # kéo mép lưới về đúng đường viền trơn
        i, j, v = sam(H, V[bd, 0], V[bd, 1]); dx = (H[j, i + 1] - H[j, i - 1]) / 2; dy = (H[j + 1, i] - H[j - 1, i]) / 2
        s_ = np.clip((0.5 - v) / (dx ** 2 + dy ** 2 + 1e-5), -2, 2); V[bd, 0] += s_ * dx; V[bd, 1] += s_ * dy
    W = f.px(V[:, 0] / k, V[:, 1] / k); head, hp = head_part(g); rd = lambda x: x if isinstance(x, np.ndarray) else g.read(x)
    hv = rd(hp["attributes"]["POSITION"]).astype(float); hn = rd(hp["attributes"]["NORMAL"]).astype(float); tri = g.read(hp["indices"]).reshape(-1, 3)
    uu, vv = np.meshgrid(np.linspace(0, 1, 6), np.linspace(0, 1, 6)); ok = uu + vv <= 1; bary = np.c_[uu[ok], vv[ok], 1 - uu[ok] - vv[ok]]; pts = (hv[tri][:, None] * bary[None, :, :, None]).sum(2).reshape(-1, 3)
    pts = pts[pts[:, 2] > 0.04]; q = f.inv(pts[:, :2]); Z = np.full((a.shape[0], a.shape[1]), -9.0, np.float32); i = q[:, 0].round().astype(int); j = q[:, 1].round().astype(int); m = (i >= 0) & (i < Z.shape[1]) & (j >= 0) & (j < Z.shape[0])
    np.maximum.at(Z, (j[m], i[m]), pts[m, 2]); Z = cv2.dilate(Z, np.ones((3, 3), np.uint8)); ins = (Z > -5).astype(np.float32); Z = cv2.GaussianBlur(np.where(Z > -5, Z, 0), (0, 0), 2) / (cv2.GaussianBlur(ins, (0, 0), 2) + 1e-4)
    D = cv2.distanceTransform((H > 0.5).astype(np.uint8), cv2.DIST_L2, 5); i, j, d = sam(D, V[:, 0], V[:, 1]); z = Z[(V[:, 1] / k).round().astype(int).clip(0, Z.shape[0] - 1), (V[:, 0] / k).round().astype(int).clip(0, Z.shape[1] - 1)]
    P = np.c_[W, z + 0.0006 + 0.0022 * np.sqrt(np.clip(d / max(d.max(), 1e-3), 0, 1))]; front = np.nonzero(hv[:, 2] > 0.05)[0]; ix, w = nearest(P, hv[front]); ix = front[ix]
    N = (hn[ix] * w[..., None]).sum(1); N /= np.linalg.norm(N, axis=1, keepdims=True).clip(1e-6)
    B = cv2.GaussianBlur(H, (0, 0), 3); i, j, _ = sam(B, V[:, 0], V[:, 1]); i = i.clip(2, B.shape[1] - 3); j = j.clip(2, B.shape[0] - 3); N[:, 0] -= (B[j, i + 2] - B[j, i - 2]) * 1.2; N[:, 1] += (B[j + 2, i] - B[j - 2, i]) * 1.2; N /= np.linalg.norm(N, axis=1, keepdims=True).clip(1e-6)
    add_to_head(g, P, N, Q[:, [0, 2, 1, 0, 3, 2]].reshape(-1, 3), ix, w, "brows_fit", lin(np.median(a[al & (a[..., 3] > 200)][:, :3], axis=0) / 255))
    for n in g.j["nodes"]:
        if n.get("name") == "eyebrows": n.pop("mesh", None); n.pop("skin", None)
    return True


def lashes(g, f, parts, lin):
    """Lông mi: dải bám đúng viền mí trên của lưới đầu, nên khi chớp mắt đi cùng mí. Độ dày đo trên ảnh."""
    head, hp = head_part(g); rd = lambda x: x if isinstance(x, np.ndarray) else g.read(x)
    hv = rd(hp["attributes"]["POSITION"]).astype(float); tri = g.read(hp["indices"]).reshape(-1, 3); seen = {}
    for a, b, c in tri.tolist():
        for e in ((a, b), (b, c), (c, a)): seen[tuple(sorted(e))] = seen.get(tuple(sorted(e)), 0) + 1
    adj = {}
    for (a, b), n in seen.items():
        if n == 1: adj.setdefault(a, []).append(b); adj.setdefault(b, []).append(a)
    loops = []; left = set(adj)
    while left:  # các vòng viền hở của lưới
        st = [left.pop()]; c = list(st)
        while st:
            for q in adj[st.pop()]:
                if q in left: left.remove(q); st.append(q); c.append(q)
        loops.append(c)
    masks = [p[..., 3] > 128 for p in parts]
    col = np.concatenate([p[m][:, :3] for p, m in zip(parts, masks)]); col = np.percentile(col[col.mean(1) <= np.percentile(col.mean(1), 25)], 40, axis=0) * 0.4  # mép ảnh cắt lẫn màu da: chỉ lấy phần tối nhất
    h_px = np.median(np.concatenate([m.sum(0)[m.sum(0) > 0] for m in masks])); L0 = float(np.clip(h_px * f.s * 2.2, 0.0032, 0.0056))
    P, N, T, ix, wt = [], [], [], [], []
    for sd, (ca, cb) in ((-1, (33, 133)), (1, (263, 362))):
        cn = (f.tgt[ca] + f.tgt[cb]) / 2; hw = np.linalg.norm(f.tgt[ca] - f.tgt[cb]) / 2
        cand = [c for c in loops if len(c) >= 8 and np.linalg.norm(hv[c][:, :2].mean(0) - cn) < hw and hv[c, 2].mean() > 0.06]
        if not cand: continue
        rim = np.array(max(cand, key=lambda c: hv[c, 2].mean())); r = hv[rim]; a, b = r[r[:, 0].argmin()], r[r[:, 0].argmax()]
        up = rim[r[:, 1] >= a[1] + (r[:, 0] - a[0]) / (b[0] - a[0]) * (b[1] - a[1]) - 1e-4]; up = up[np.argsort(hv[up, 0])]  # nửa trên của vòng mí
        n = 8; S = (len(up) - 1) * n + 1; kk = np.arange(S) / n; i0 = np.minimum(kk.astype(int), len(up) - 2); t = kk - i0; p = hv[up[i0]] * (1 - t[:, None]) + hv[up[i0 + 1]] * t[:, None]
        for _ in range(12): p[1:-1] = (p[:-2] + 2 * p[1:-1] + p[2:]) / 4  # làm trơn đường mí, lưới gốc ít điểm
        u = (p[:, 0] - p[:, 0].min()) / np.ptp(p[:, 0]); u = u if sd > 0 else 1 - u  # u: 0 ở khoé trong, 1 ở đuôi mắt
        keep = u > 0.06; p, u, a0, a1, t = p[keep], u[keep], up[i0[keep]], up[i0[keep] + 1], t[keep]
        L = L0 * np.sin(np.pi * np.clip(u * 0.92 + 0.08, 0, 1)) ** 0.55
        d = np.c_[sd * (0.1 + 0.55 * u), np.full(len(u), 0.8), np.full(len(u), 0.45)]; d /= np.linalg.norm(d, axis=1, keepdims=True); b0 = sum(len(q) for q in P)
        root = p.copy(); near_ = hv[np.abs(hv[:, 0] - p[:, 0, None]).min(0) < 0.004]  # mí dày che mất viền lưới: đặt chân mi lên mặt ngoài của mí
        root[:, 2] = np.array([near_[np.hypot(near_[:, 0] - x, near_[:, 1] - y) < 0.0035, 2].max(initial=z) for x, y, z in p]) + 0.0005
        for _ in range(10): root[1:-1, 2] = (root[:-2, 2] + 2 * root[1:-1, 2] + root[2:, 2]) / 4
        P.append(np.r_[root, root + d * L[:, None]]); N.append(np.tile((0.0, 0.45, 0.9), (2 * len(p), 1)))
        k_ = np.arange(len(p) - 1) + b0; m_ = len(p); T.append(np.r_[np.c_[k_, k_ + 1, k_ + 1 + m_], np.c_[k_, k_ + 1 + m_, k_ + m_]])
        pair = np.c_[a0, a1]; w_ = np.c_[1 - t, t]; ix.append(np.r_[pair, pair]); wt.append(np.r_[w_, w_])
    if not P: return False
    add_to_head(g, np.concatenate(P), np.concatenate(N), np.concatenate(T), np.concatenate(ix), np.concatenate(wt), "lashes_fit", lin(col / 255))
    return True
