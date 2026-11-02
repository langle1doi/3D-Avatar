# Ảnh Pixar → avatar 3D

Hai hướng đã làm, mỗi hướng có hướng dẫn chạy riêng bên dưới. Tiến độ, hạn chế và việc kế tiếp: [BAO_CAO_LEAD.md](BAO_CAO_LEAD.md).

| | Hướng A: nhân vật gốc, chỉnh theo ảnh | Hướng B: AI sinh mô hình từ ảnh |
|---|---|---|
| Cách làm | Lấy nhân vật Rain/Snow có sẵn xương, nắn mặt/tóc/kính theo ảnh | Hunyuan3D-2 + texture trên Kaggle T4 |
| Xương + biểu cảm (cười, chớp) | **Có** (68 xương, 25 blendshape ARKit) | Chưa (việc kế tiếp: rig bằng UniRig) |
| Giống ảnh | Vừa | **Tốt nhất** trong các model đã thử |
| Cần GPU | Không (chạy máy thường, khoảng 16 giây/nhân vật) | Có (Kaggle T4 miễn phí, khoảng 45 giây/ảnh) |

**Luồng:** ảnh thật → (Bước 1, FLUX trên Kaggle) → ảnh Pixar → (Bước 2) Hướng A hoặc Hướng B → avatar 3D.

## Cấu trúc
```
buoc1_flux/  (real2pixar_flux.ipynb, real2pixar_front.py, api/ mẫu Kaggle API, tham_khao_qwen/)
huong_A/  blender/ (nhân vật nền)  tools/ (code chính)  demo/view.html  output_nu2/ (28 nữ, bản mới)
          output_nu/ (bản cũ + bao_cao*.html)  segment_tham_khao/  requirements.txt  mo_xem_model.bat
huong_B/  kaggle/ (hunyuan_tex.ipynb, run.sh, run_tex.py, run_unirig.sh, api/ = mẫu cho Kaggle API)  so_sanh/ (kết quả 6 ảnh + hunyuan.html)
```

---

# Bước 1: ảnh thật → ảnh Pixar (FLUX trên Kaggle)

Dùng FLUX.1 Kontext [dev] + một LoRA đã huấn luyện (prompt `zqx style`, 100 cặp ảnh, rank 16). Gồm hai lượt: (1) xoay mặt nhìn thẳng, (2) đổi sang phong cách Pixar. Chạy trên Kaggle GPU T4, khoảng 105 giây mỗi ảnh cho lượt Pixar (số đo từ lần chạy của chủ repo; lượt xoay mặt thêm khoảng bằng đó, chưa đo).

### Chuẩn bị chung (làm một lần, dùng cho cả Bước 1 và Hướng B)
1. **Tài khoản Kaggle** đã xác minh số điện thoại (để bật GPU và Internet).
2. **Hugging Face:** FLUX.1 Kontext là model "gated". Đăng nhập huggingface.co, vào https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev và bấm đồng ý điều khoản (cần được duyệt), rồi tạo token loại **Read** ở Settings → Access Tokens.
3. **Đưa token HF cho Kaggle:** trong notebook, mở **Add-ons → Secrets**, thêm secret tên đúng `HF_TOKEN`, rồi bật cho notebook. (Cách dự phòng, hợp với dòng lệnh: tạo dataset **private** có một file `hf_token.txt` chứa token và Add Data vào notebook; notebook tự đọc.)
4. **Kaggle API (nếu muốn chạy bằng dòng lệnh):**
   - `pip install kaggle`
   - Kaggle → ảnh đại diện → **Settings → API → Create New Token**. Đưa token cho CLI bằng biến môi trường `KAGGLE_API_TOKEN` (PowerShell: `$env:KAGGLE_API_TOKEN = "..."`; bash: `export KAGGLE_API_TOKEN=...`), hoặc ghi vào file `~/.kaggle/access_token` (Windows: `C:\Users\<tên>\.kaggle\access_token`), hoặc dùng file `kaggle.json` đặt ở `~/.kaggle/kaggle.json`.
   - Kiểm tra: `kaggle datasets list --mine` không báo lỗi xác thực.
   - **Token Kaggle và token HF là mật khẩu: không commit, không gửi ai.** Mỗi người dùng token của tài khoản mình.
   - Mẫu metadata nằm ở `buoc1_flux/api/` (và `huong_B/kaggle/api/`): thay `TEN_KAGGLE_CUA_BAN` bằng username Kaggle của bạn (phần sau `kaggle.com/` ở trang hồ sơ).

### Dữ liệu cần có
- **Ảnh thật:** ảnh chân dung của bạn (tên file = mã kết quả, `.jpg` hoặc `.png`). Ảnh người thật nên chỉ giữ riêng, không đưa lên repo công khai.
- **Checkpoint LoRA `trig100.pt`** (105MB, **chủ repo gửi riêng**, không nằm trong git vì vượt giới hạn 100MB/file của GitHub): chủ repo upload thành dataset Kaggle rồi chia sẻ cho partner (`kaggle datasets` có thể đặt public, hoặc partner tự upload bản nhận được ở dạng file). Tên file phải đúng `trig100.pt`.
  Đây là file `final.pt` của lần huấn luyện `flux_train_100_trig` (100 cặp, 500 bước, prompt `zqx style`); tôi đối chiếu qua tên và thông số trong `experiment.json`, không có chữ ký nào xác nhận thêm.

### Cách 1.1: bấm trên web Kaggle
1. **Datasets → New Dataset**: một dataset chứa ảnh thật, một dataset chứa `trig100.pt`.
2. **Code → New Notebook → File → Import Notebook**, chọn `buoc1_flux/real2pixar_flux.ipynb`.
3. **Add Data** hai dataset ở bước 1. **Session options**: Accelerator = GPU T4, Internet = On. **Add-ons → Secrets**: bật `HF_TOKEN`.
4. **Run All.** Lần đầu cài thư viện và tải FLUX (vài GB) mất khá lâu.
5. Lấy ảnh ở tab **Output**: `real2pixar/pixar/<tên>.png` (ảnh Pixar), `real2pixar/front/` (ảnh đã xoay thẳng), `real2pixar/grid.jpg` (gốc | xoay thẳng | Pixar).

### Cách 1.2: bằng Kaggle API (làm trong `buoc1_flux/api`)
1. Thay `TEN_KAGGLE_CUA_BAN` trong 3 file metadata.
2. Chép ảnh thật vào `dataset_anh/`, chép `trig100.pt` vào `dataset_ckpt/`, rồi:
   ```
   kaggle datasets create -p dataset_anh
   kaggle datasets create -p dataset_ckpt
   ```
   (File `trig100.pt` 105MB, upload mất vài phút. Đợi `kaggle datasets status TEN/real-portraits` và `TEN/pixar-trig-ckpt` báo `ready`.)
3. Đẩy và chạy: `kaggle kernels push -p kernel`.
4. **Bật `HF_TOKEN` cho notebook** (Secrets, xem chuẩn bị chung). Secrets không đặt được bằng API, nên lần đầu có thể phải mở notebook trên web, bật secret rồi chạy lại; hoặc dùng cách dự phòng `hf_token.txt` bằng dataset private và thêm dataset đó vào `dataset_sources` trong `kernel-metadata.json`.
5. Theo dõi: `kaggle kernels status TEN/real2pixar-flux` (tới `COMPLETE`), rồi tải: `kaggle kernels output TEN/real2pixar-flux -p ket_qua_buoc1`.
Ảnh Pixar nằm trong `ket_qua_buoc1/real2pixar/pixar/`. Đây là **đầu vào của Bước 2**.

Chưa kiểm chứng: tôi chưa chạy notebook `real2pixar_flux.ipynb` và các lệnh `kaggle` này; script bên trong (`real2pixar_front.py`) là bản chủ repo đã dùng, notebook là bản tôi gộp lại cho ảnh bất kỳ. Lần đầu hãy xem log từng ô.

Ghi chú: ở các lần chạy chính (50 ảnh) chủ repo dùng Qwen-Image-Edit-2509 để xoay mặt thay cho FLUX (mặt thẳng ổn hơn, nhưng khoảng 13 phút mỗi ảnh trên T4). Code và notebook nằm ở `buoc1_flux/tham_khao_qwen/`, gắn với dataset riêng của chủ repo, chưa đóng gói lại. Cách 1.1 và 1.2 dùng FLUX cho cả hai lượt, nên ảnh Pixar có thể khác nhẹ so với bản dùng Qwen.
Hai lượt dùng chung một prompt cố định; checkpoint chỉ hợp prompt `zqx style` (xem `generative ai/docs/experiment_log.md`, Experiment 10 đến 12, nếu cần biết lý do).
Cách huấn luyện lại LoRA (nếu cần) nằm ở thư mục `generative ai` của chủ repo (`scripts/flux_kontext.py train`, `scripts/flux_zeroshot_pixar.py` tạo ảnh mẫu "teacher"); chưa đóng gói trong repo này.

---

# Bước 2: ảnh Pixar → avatar 3D (hai hướng)

Đầu vào: ảnh Pixar từ Bước 1.

## Hướng A, cách 1: tái tạo kết quả của chủ repo (đã chạy thử, ra đúng như bản gốc)

Cần: Python 3.11 và dữ liệu đầu vào `pixar_parts_seg3_clean.zip` (197MB, **chủ repo gửi riêng**, không nằm trong git).

1. Cài thư viện:
   ```
   cd huong_A
   pip install -r requirements.txt
   ```
2. Giải nén `pixar_parts_seg3_clean.zip` vào một thư mục, ví dụ `D:\du_lieu\pixar_parts_seg3_clean` (bên trong có `assembled/` và `parts/`).
3. Đặt biến môi trường trỏ tới thư mục đó (PowerShell: `$env:AVATAR_PARTS = "D:\du_lieu\pixar_parts_seg3_clean"`; bash: `export AVATAR_PARTS=...`).
4. Tạo avatar:
   ```
   cd tools
   python make_avatar.py 0002 --out ../output_moi          # một nhân vật
   python make_avatar.py --all --out ../output_moi         # cả 50 (nữ dùng khung Rain, nam dùng Snow)
   ```
   Kết quả: `huong_A/output_moi/<mã>.glb`. Nữ so được với `output_nu2/`. Thử trên `0002`: ra file cùng kích thước (8.506.364 byte) với `output_nu2/0002.glb`, một số byte lệch nhỏ; hình xem bằng mắt chưa so từng chi tiết.
   Lần đầu chạy có thể in vài dòng cảnh báo TensorFlow/MediaPipe, bỏ qua được.
5. Xem kết quả:
   ```
   cd huong_A
   python -m http.server 8766
   ```
   Mở `http://localhost:8766/demo/view.html?m=../output_moi/0002.glb`. Trang này tải three.js từ CDN nên cần có mạng. Xem so sánh cũ/mới: `http://localhost:8766/output_nu/bao_cao_2.html`.

## Hướng A, cách 2: ảnh Pixar mới (chưa đóng gói, chưa chạy thử)

`make_avatar.py` không đọc một ảnh đơn lẻ mà đọc ảnh **đã cắt thành từng phần** (face, hair, glasses, eyes, mouth, iris...) trong `parts/<mã>/` và ảnh ghép `assembled/<mã>.png`. Bước cắt chạy trên Kaggle GPU (SegFormer, MediaPipe, SAM 2, ViTMatte) và các script gốc nằm ở `huong_A/segment_tham_khao/`. Chúng viết cho đúng 50 ảnh của chủ repo (có lệnh `assert len(src) == 50`), nên cần sửa trước khi dùng cho ảnh mới. Phần này **chưa được đóng gói thành một lệnh**. Nếu cần, nhờ chủ repo hoặc làm tiếp mục này. Sau khi có thư mục `parts/<mã>/` và `assembled/<mã>.png` mới, làm lại bước 3 và 4 của cách 1.

---

## Hướng B: Hunyuan3D-2 + texture trên Kaggle (ảnh Pixar bất kỳ)

Cần: tài khoản Kaggle (xem "Chuẩn bị chung" ở Bước 1). Không cần cài model trên máy. Model Hunyuan3D-2 tự tải từ Hugging Face khi notebook chạy.
Chuẩn bị: gom ảnh Pixar (.jpg/.png, một người đứng một mình, script tự xoá nền) vào một thư mục. **Tên file = tên file kết quả.** Chọn một trong hai cách.

### Cách B1: bấm trên web Kaggle (không cần cài gì)
1. **Datasets → New Dataset**, upload các ảnh, đặt tên `pixar-3d-inputs` (private).
2. **Code → New Notebook → File → Import Notebook**, chọn `huong_B/kaggle/hunyuan_tex.ipynb`.
3. Panel phải: **Add Data** chọn dataset ở bước 1. **Session options**: Accelerator = GPU T4, Internet = **On**.
4. **Run All.** Lần đầu mất vài phút (biên dịch rasterizer, tải model), sau đó khoảng 45 giây mỗi ảnh.
5. Lấy `h3d_tex_out/<tên ảnh>.glb` ở tab **Output** của notebook.

### Cách B2: bằng Kaggle API (dòng lệnh)
Làm cả quá trình mà không mở web. Đây là cách chủ repo dùng.

**Cài Kaggle API và lấy token:** xem "Chuẩn bị chung" ở Bước 1.

**Chạy (làm trong `huong_B/kaggle/api`):**
1. Thay `TEN_KAGGLE_CUA_BAN` bằng username Kaggle của bạn trong `dataset/dataset-metadata.json` và `kernel/kernel-metadata.json` (username là phần sau `kaggle.com/` ở trang hồ sơ).
2. Chép ảnh Pixar vào thư mục `dataset/` (cạnh `dataset-metadata.json`), rồi tạo dataset:
   ```
   kaggle datasets create -p dataset
   ```
   Muốn thêm ảnh sau này: `kaggle datasets version -p dataset -m "them anh"`. Đợi dataset xử lý xong: `kaggle datasets status TEN/pixar-3d-inputs` báo `ready`.
3. Đẩy và chạy notebook (tự bật GPU và Internet theo `kernel-metadata.json`):
   ```
   kaggle kernels push -p kernel
   ```
4. Theo dõi tới khi `COMPLETE` (khoảng 5 đến 10 phút lần đầu):
   ```
   kaggle kernels status TEN/pixar-hunyuan-tex
   ```
5. Tải kết quả:
   ```
   kaggle kernels output TEN/pixar-hunyuan-tex -p ../../so_sanh/ket_qua
   ```
   File `.glb` nằm trong `h3d_tex_out/`. Nếu lỗi 429 thì thử lại sau ít phút, hoặc tải ở tab Output trên web.

Ghi chú: GPU Kaggle miễn phí giới hạn theo tuần, tối đa 2 phiên cùng lúc. Một số tài khoản bị hạn chế `kernels push`; nếu báo lỗi quyền thì dùng cách B1. Tôi chưa chạy thử các lệnh `kaggle` này trong phiên chuẩn bị repo.

### Xem kết quả
Chép các `.glb` vào `huong_B/so_sanh/` (đặt tên `h3t_<tên ảnh>.glb`), chạy `python -m http.server 8766` trong `huong_B`, mở `http://localhost:8766/so_sanh/hunyuan.html` (trang này đang liệt kê 6 ảnh mẫu của chủ repo; thêm khối mới theo mẫu trong file).

Kết quả mẫu sẵn trong repo: `huong_B/so_sanh/h3t_*.glb` và ảnh đầu vào `in_*.png`. Muốn kiểm tra pipeline có ra giống bản gốc không, dùng 6 ảnh `in_*.png` làm dataset. Tôi chưa chạy lại notebook sau khi sửa để nhận ảnh bất kỳ (bản gốc ghi cứng tên 6 ảnh), nên lần đầu hãy xem log ở ô cuối.
`run_unirig.sh` dựng UniRig để rig mesh (dùng ở thử nghiệm TripoSR), chưa chạy trên mesh Hunyuan3D-2.

---

## Tải nhân vật Rain / Snow (Blender Studio) và nướng ra `.glb`

**Chỉ cần làm nếu muốn tạo lại hoặc sửa nhân vật nền.** Nếu chỉ chạy Hướng A cách 1 thì bỏ qua: file nền đã nướng sẵn trong `huong_A/blender/out/` (`rain_rpm.glb`, `snow_rpm.glb`, texture, điểm mốc).

1. Tải (miễn phí, CC-BY):
   - Rain: https://studio.blender.org/characters/rain/ (nút Download). Chủ repo dùng `rain_v3.2.blend` kèm thư mục `textures/`.
   - Snow: https://studio.blender.org/characters/snow/ (bản v4). Chủ repo dùng `snow_v4.2.blend` kèm `textures/`.
   Giải nén, giữ nguyên file `.blend` cạnh thư mục `textures/` (blend tham chiếu texture theo đường dẫn tương đối). Nếu trang đã đổi tên file hoặc phiên bản thì tải bản mới nhất; chưa kiểm chứng rig bản mới chạy được với `rain_bake.py`.
2. Cài Blender: https://www.blender.org/download/ (chủ repo dùng 4.5; bản khác chưa thử).
3. Nướng (chạy trong `huong_A/blender`; `blender` phải có trong PATH, hoặc dùng đường dẫn đầy đủ tới `blender.exe`):
   ```
   blender -b "D:\rain\rain_v3.2.blend" --python rain_bake.py -- out/rain_rpm.glb rain
   blender -b "D:\snow\snow_v4.2.blend" --python rain_bake.py -- out/snow_rpm.glb snow
   ```
   Ra `out/<tên>_rpm.glb`, `out/<tên>_rpm.json` và thư mục `out/<tên>_tex/`. Tôi chưa chạy lại hai lệnh này (file nền trong repo là bản chủ repo đã nướng).
4. Điểm mốc khuôn mặt (`*_marks.npy`) tạo bằng `tools/base_marks.py`; file đã có sẵn trong `out/`.

Ghi công bắt buộc khi dùng: "Rain Rig (CC) Blender Foundation | studio.blender.org" và "Snow Rig (CC) Blender Foundation | studio.blender.org" (code đã ghi vào từng `.glb`).

---

## Chưa có trong repo
- Checkpoint LoRA `trig100.pt` (105MB) cho Bước 1: chủ repo gửi riêng.
- Thư mục `pixar_parts_seg3_clean.zip` (197MB) cho Hướng A.
- `blender/rain_bake.py` cần Blender và nhân vật gốc Rain (tải ở studio.blender.org). Chỉ chạy một lần để tạo `rain_rpm.glb`, file đó đã có sẵn trong `huong_A/blender/out/`.
https://drive.google.com/drive/folders/1gHIP0pb0gK-OtXTAT7Znh4ozWef4wnh9?usp=sharing
