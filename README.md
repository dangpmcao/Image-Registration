# Image Registration

Baseline so sánh đăng ký ảnh biến dạng bằng B-spline tối ưu riêng cho từng cặp và CNN dự đoán trường biến dạng. Mã dùng chung và mã của từng phương pháp được tách riêng; việc train CNN đầy đủ dự kiến chạy trên server.

## Cấu trúc

```text
Image-Registration/
├── b-spline/       # B-spline FFD 2D và 3D
├── cnn/            # Mô hình CNN 2D/3D và vòng train
├── data/           # Loader, chia tập, hình học, đánh giá, download.md
├── demo/           # CLI đăng ký một cặp; chỗ mở rộng demo sau này
├── results/        # <dataset>/cnn/ và <dataset>/b-spline/
├── requirement.txt
├── run_demo.py
├── train.py
├── README.md
└── .gitignore
```

`results/` hiện chỉ giữ thư mục rỗng. Mỗi lần chạy sẽ ghi `metrics.json`, `field.npz`, `preview.png` vào `results/<dataset>/<method>/<split>_seed<seed>/<pair>/<settings>/`. Checkpoint CNN nằm trong `results/<dataset>/cnn/<run-name>/`. Các checkpoint và kết quả smoke test trước đây đã được xóa vì chúng không đại diện cho mô hình được train đầy đủ.

## Dataset

Dữ liệu gốc nằm trong `../dataset/`; xem [data/download.md](data/download.md) để biết nguồn và cách tải. Các tên dataset đã có adapter:

`mnist`, `oasis_2d`, `acdc`, `sunnybrook`, `sunnybrook_2d`, `lungct`, `nlst2023`, `oasis_3d`, `abdomenctct`, `tcia_breast_nact`, `ixi`.

Sunnybrook 2D và 3D dùng chung một thư mục dữ liệu. Mỗi adapter tạo cặp train/val/test theo cách xác định và trả ảnh 2D hoặc 3D cùng lưới fixed. NLST dùng tập test giữ lại nội bộ vì bản dữ liệu tại máy không có ảnh test chính thức. Sunnybrook 3D và breast DCE chưa có nhãn voxel đầy đủ trong loader; mask LungCT/NLST là vùng đánh giá, không phải segmentation. IXI đa phương thức, nên NCC hiện chỉ là baseline kỹ thuật; NMI chưa được triển khai. DIR-Lab, FIRE và ANHIR chưa có dữ liệu tại máy.

## Chạy

Cài dependency vào môi trường Python của bạn:

```powershell
python -m pip install -r requirement.txt
```

Chạy B-spline trên một cặp:

```powershell
python run_demo.py --dataset oasis_2d --method b-spline
python run_demo.py --dataset acdc --method b-spline --split val --pair-index 0
```

Train CNN trên server có GPU, rồi dùng checkpoint để chạy suy luận:

```powershell
python train.py --dataset acdc --device cuda --run-name acdc_v1
python run_demo.py --dataset acdc --method cnn --checkpoint results/acdc/cnn/acdc_v1/best.pt
```

`--root` cho phép trỏ tới thư mục dữ liệu khác; `--output` đổi thư mục kết quả. Với ảnh 3D, `--max-dim` quyết định độ phân giải lưới làm việc. Dùng `--help` ở từng lệnh để xem các tham số còn lại.

Đây là baseline một tầng CNN và B-spline FFD. Chưa có affine, NMI, DLIR nhiều tầng, TRE hoặc tái lập chính xác kết quả bài báo. Không nên xem kết quả từ checkpoint chỉ train vài bước là đánh giá mô hình.
