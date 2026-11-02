# Báo cáo một file: ảnh Pixar + ảnh chụp thẳng model + link xem model. Ảnh nhúng sẵn trong file, gửi đi vẫn thấy ảnh.
# Dùng: python female_report.py     đọc output_nu/_shots, ghi output_nu/bao_cao.html. Link xem cần chạy output_nu/mo_xem_model.bat.
import base64, glob, io, os
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "output_nu")
PARTS = os.path.join(os.path.dirname(ROOT), "generative ai", "results", "pixar_parts_seg3_clean", "assembled")


def b64(im):
    buf = io.BytesIO(); im.convert("RGB").resize((340, 340), Image.LANCZOS).save(buf, "JPEG", quality=84)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


cards = []
for f in sorted(glob.glob(os.path.join(OUT, "_shots", "*.png"))):
    i = os.path.basename(f)[:4]; a = Image.open(os.path.join(PARTS, i + ".png")).convert("RGBA"); bg = Image.new("RGBA", a.size, "white"); bg.alpha_composite(a)
    url = f"http://localhost:8766/demo/view.html?m=../output_nu/{i}.glb"
    cards.append(f'<article><h2>{i}</h2><div class="p"><figure><img src="{b64(bg)}" alt="Ảnh Pixar {i}"><figcaption>Ảnh Pixar</figcaption></figure><figure><img src="{b64(Image.open(f))}" alt="Model 3D {i}, chụp chính diện"><figcaption>Model 3D, chính diện</figcaption></figure></div><a href="{url}" target="_blank">Xem model 3D xoay được</a></article>')
html = f'''<!doctype html><html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Model nữ từ ảnh Pixar</title>
<style>
:root{{--bg:#eef2f6;--card:#fff;--fg:#17293a;--muted:#55697b;--line:#cfdce6;--link:#1f6d94}}
@media (prefers-color-scheme:dark){{:root{{--bg:#121b24;--card:#1b2835;--fg:#e8f0f6;--muted:#a3b5c4;--line:#2f4152;--link:#6cbde4}}}}
body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif}}main{{max-width:1120px;margin:auto;padding:28px 16px}}
h1{{margin:0 0 6px;font-size:1.8rem}}.note{{color:var(--muted);margin:0 0 20px;max-width:70ch}}.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));gap:14px}}
article{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px;display:grid;gap:8px}}h2{{margin:0;font-size:1rem}}.p{{display:grid;grid-template-columns:1fr 1fr;gap:6px}}figure{{margin:0}}img{{width:100%;display:block;border-radius:6px;background:#fff}}figcaption{{font-size:.8rem;color:var(--muted);margin-top:3px}}a{{color:var(--link);font-weight:600}}
</style></head><body><main><h1>Model nữ từ ảnh Pixar</h1>
<p class="note">{len(cards)} nhân vật. Để mở link xem model, chạy file mo_xem_model.bat trong thư mục này trước và giữ cửa sổ đó mở.</p>
<div class="grid">{"".join(cards)}</div></main></body></html>'''
open(os.path.join(OUT, "bao_cao.html"), "w", encoding="utf-8").write(html)
open(os.path.join(OUT, "mo_xem_model.bat"), "w", encoding="ascii").write('@echo off\r\ncd /d "%~dp0.."\r\npython -m http.server 8766\r\n')
print(len(cards), os.path.getsize(os.path.join(OUT, "bao_cao.html")) // 1024, "KB")
