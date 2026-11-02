# Báo cáo so sánh bản cũ (output_nu) và bản sửa (output_nu2): tóc nhìn bên/sau và mi khi chớp. Ảnh nhúng sẵn trong file, gửi đi vẫn thấy ảnh.
# Dùng: python female_report2.py     đọc output_nu2/_cmp (ảnh chụp bằng demo/shoot.html, tên cu_<id>_<góc>.jpg và moi_<id>_<góc>.jpg), ghi output_nu/bao_cao_2.html.
import base64, glob, io, os
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CMP = os.path.join(ROOT, "output_nu2", "_cmp")
PARTS = os.path.join(os.path.dirname(ROOT), "generative ai", "results", "pixar_parts_seg3_clean", "assembled")
NOTED = ("0021", "0002", "0044")  # các ảnh người dùng đã nêu trong đánh giá


def b64(im, n=300):
    buf = io.BytesIO(); im.convert("RGB").resize((n, n), Image.LANCZOS).save(buf, "JPEG", quality=82)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def fig(path, cap):
    return f'<figure><img src="{b64(Image.open(path))}" alt="{cap}"><figcaption>{cap}</figcaption></figure>'


ids = sorted({os.path.basename(f)[4:8] for f in glob.glob(os.path.join(CMP, "moi_*_front.jpg"))}, key=lambda i: (i not in NOTED, i))
cards = []
for i in ids:
    a = Image.open(os.path.join(PARTS, i + ".png")).convert("RGBA"); bg = Image.new("RGBA", a.size, "white"); bg.alpha_composite(a)
    c = lambda t, v: os.path.join(CMP, f"{t}_{i}_{v}.jpg")
    mark = " <small>(đã nêu trong đánh giá)</small>" if i in NOTED else ""
    cards.append(
        f'<article><h2>{i}{mark}</h2>'
        f'<div class="p"><figure><img src="{b64(bg)}" alt="Ảnh Pixar {i}"><figcaption>Ảnh Pixar</figcaption></figure>{fig(c("moi", "front"), "Bản mới, chính diện")}</div>'
        f'<h3>Tóc nhìn bên và sau</h3><div class="p">{fig(c("cu", "back"), "Cũ")}{fig(c("moi", "back"), "Mới")}</div>'
        f'<h3>Chớp mắt</h3><div class="p">{fig(c("cu", "blink"), "Cũ")}{fig(c("moi", "blink"), "Mới")}</div>'
        f'<span><a href="http://localhost:8766/demo/view.html?m=../output_nu2/{i}.glb" target="_blank">Xem bản mới xoay được</a> · '
        f'<a href="http://localhost:8766/demo/view.html?m=../output_nu/{i}.glb" target="_blank">bản cũ</a></span></article>')
html = f'''<!doctype html><html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>So sánh bản cũ và bản sửa</title>
<style>
:root{{--bg:#eef2f6;--card:#fff;--fg:#17293a;--muted:#55697b;--line:#cfdce6;--link:#1f6d94}}
@media (prefers-color-scheme:dark){{:root{{--bg:#121b24;--card:#1b2835;--fg:#e8f0f6;--muted:#a3b5c4;--line:#2f4152;--link:#6cbde4}}}}
body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif}}main{{max-width:1120px;margin:auto;padding:28px 16px}}
h1{{margin:0 0 6px;font-size:1.8rem}}.note{{color:var(--muted);margin:0 0 12px;max-width:80ch}}.note b{{color:var(--fg)}}.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:14px;margin-top:18px}}
article{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px;display:grid;gap:8px}}h2{{margin:0;font-size:1rem}}h2 small{{color:var(--muted);font-weight:400}}h3{{margin:6px 0 0;font-size:.9rem;color:var(--muted)}}.p{{display:grid;grid-template-columns:1fr 1fr;gap:6px}}figure{{margin:0}}img{{width:100%;display:block;border-radius:6px;background:#fff}}figcaption{{font-size:.8rem;color:var(--muted);margin-top:3px}}a{{color:var(--link);font-weight:600}}
</style></head><body><main><h1>So sánh bản cũ và bản đã sửa</h1>
<p class="note">{len(cards)} nhân vật nữ. Mỗi thẻ có ảnh Pixar, bản mới chính diện, rồi hai cặp so sánh <b>Cũ | Mới</b>: tóc nhìn bên và sau, và chớp mắt. Bản cũ là <code>bao_cao.html</code> (thư mục <code>output_nu</code>), bản mới nằm ở <code>output_nu2</code>. Để mở link xem model xoay được, chạy file mo_xem_model.bat trong thư mục này trước và giữ cửa sổ đó mở.</p>
<p class="note"><b>Đã sửa:</b> (1) tóc: nửa trước và nửa sau giờ gặp nhau cùng màu nên hết đường ranh, và bỏ viền da còn sót ở mép ảnh tóc. (2) mi: khi chớp, mi nằm trên mí đã khép thay vì chui xuống dưới da, và dẹp lại sát đường mí.</p>
<div class="grid">{"".join(cards)}</div></main></body></html>'''
out = os.path.join(ROOT, "output_nu", "bao_cao_2.html")
open(out, "w", encoding="utf-8").write(html); print(len(cards), os.path.getsize(out) // 1024, "KB")
