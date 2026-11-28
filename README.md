# Image Registration

Đăng ký ảnh biến dạng bằng B-spline tối ưu cho từng cặp và CNN dự đoán phép biến đổi.
Hai phương pháp theo paper được gọi riêng: **Rueckert 1999** và **DLIR của de Vos**.
Baseline 2D/3D cũ vẫn giữ nguyên cách gọi để so sánh và ablation.

Đối chiếu thuật toán, các lựa chọn số học và giới hạn tái lập nằm trong
[paper/IMPLEMENTATION.md](paper/IMPLEMENTATION.md), dựa trên hai PDF gốc trong `paper/`.

## Cấu trúc

```text
Image-Registration/
├── b-spline/       # Rueckert affine + NMI + FFD; baseline FFD 2D/3D
├── cnn/            # DLIR nhiều tầng, spline interpolation, data protocol, train; baseline
├── data/           # Loader, chia tập, hình học, đánh giá, download.md
├── demo/           # CLI đăng ký một cặp; chỗ mở rộng demo sau này
├── paper/          # Hai PDF gốc và đối chiếu implementation
├── scripts/slurm/  # Job tải dữ liệu và template train DLIR
├── tests/          # Kiểm tra thuật toán, gradient, checkpoint và hình học
├── results/        # <dataset>/cnn/ và <dataset>/b-spline/
├── requirement.txt
├── run_demo.py
├── train.py
├── README.md
└── .gitignore
```

Mỗi lần chạy ghi `metrics.json`, `field.npz`, `preview.png` vào
`results/<dataset>/<method>/<split>_seed<seed>/<pair>/<settings>/`.
Checkpoint DLIR nằm trong `results/<dataset>/dlir/<run-name>/`; checkpoint baseline
nằm trong `results/<dataset>/cnn/<run-name>/`. Smoke test lưu ở `/tmp`, không phải
kết quả mô hình train đầy đủ.

## Dataset

Dữ liệu gốc nằm trong `../dataset/`; xem [data/download.md](data/download.md) để biết nguồn và cách tải. Các tên dataset đã có adapter:

`mnist`, `oasis_2d`, `acdc`, `sunnybrook`, `sunnybrook_2d`, `lungct`, `nlst2023`, `oasis_3d`, `abdomenctct`, `tcia_breast_nact`, `ixi`.

Sunnybrook 2D/3D dùng chung dữ liệu. Baseline dùng released train/val/test parts;
DLIR cardiac dùng ba folds, tất cả cine phases để train, ED/ES hai chiều để test.
Contours Sunnybrook được rasterize thành nhãn kèm mask các lát được annotate.
Breast DCE chưa có segmentation voxel trong loader. Mask LungCT/NLST là ROI,
không phải nhãn giải phẫu để tính Dice. IXI đa phương thức: NCC chỉ là baseline kỹ
thuật; có thể chọn Rueckert/NMI để thử phương pháp dùng similarity khác.
Learn2Reg NLST2023 không phải cohort NLST gốc của de Vos. DIR-Lab chưa có adapter
và protocol leave-one-out; không tuyên bố tái lập thí nghiệm đó.

## Chạy phương pháp theo paper

Chạy các lệnh từ `Image-Registration/`. Cài dependencies:

Cài dependency vào môi trường Python của bạn:

```bash
python -m pip install -r requirement.txt
```

Rueckert: affine 12 tham số, cubic FFD nhiều mức, NMI và bending weight 0.01.
Ví dụ smoke trên ACDC (chuyển phương pháp sang dataset khác):

```bash
python run_demo.py --dataset acdc --method rueckert --device cpu \
    --final-spacing-mm 10 --rueckert-levels 3 --max-dim 128
```

Trên breast NACT phải thêm `--roi Z0 Z1 Y0 Y1 X0 X1` với tọa độ voxel `(z,y,x)`
đã kiểm tra trên ảnh; chạy riêng từng breast. `--max-dim` giới hạn độ phân giải
pyramid; cấu hình này không mặc định là độ phân giải gốc của paper.

DLIR cardiac: train các tầng tuần tự, đóng băng tầng trước, loss `-NCC + 0.05*bending`.
Lệnh đầy đủ dưới đây chỉ chạy trong allocation có GPU:

```bash
python train.py --architecture dlir --dataset sunnybrook \
    --preset sunnybrook-multi --fold 1 --run-name multi_fold1_seed42 \
    --steps-per-stage 10000 --device cuda

python run_demo.py --dataset sunnybrook --method dlir --split test \
    --checkpoint results/sunnybrook/dlir/multi_fold1_seed42/best.pt --device cuda
```

`sunnybrook-single` là ablation một tầng; `--bending 0` là ablation bỏ bending.
`chest-ct` cấu hình affine + ba tầng deformable theo Table 3, trên dataset địa
phương LungCT/NLST2023. `custom` nhận JSON stage list qua `--stages-config`.
Không gọi các dataset thay thế là tái lập cohort/thí nghiệm gốc.

Submit ba cardiac folds bằng Slurm (mỗi task 1 GPU, chạy lần lượt):

```bash
cd /mnt/disk1/backup_user/dang.cpm/Optimization
mkdir -p Image-Registration/results/slurm_logs
sbatch --output="$PWD/Image-Registration/results/slurm_logs/dlir_%A_%a.log" \
    Image-Registration/scripts/slurm/train_dlir.sbatch "$PWD"
```

Template này chưa được submit khi sửa code. Để resume cùng run/config, thêm
`--resume` sau `"$PWD"`. Job giữ GPU do Slurm cấp, không gán device vật lý.
Checkpoint lưu optimizer, stage/step, RNG, pair manifest, cấu hình và hash hai paper
cùng source code. `best.pt` chỉ dùng các tầng đã train, kể cả nếu dừng giữa campaign.

## Baseline cũ

B-spline FFD dùng NCC hoặc SSD, không có affine:

```bash
python run_demo.py --dataset oasis_2d --method b-spline
python run_demo.py --dataset acdc --method b-spline --split val --pair-index 0
```

CNN một tầng:

```bash
python train.py --architecture baseline --dataset acdc --device cuda --run-name acdc_v1
python run_demo.py --dataset acdc --method cnn --checkpoint results/acdc/cnn/acdc_v1/best.pt
```

`--root` cho phép trỏ tới thư mục dữ liệu khác; `--output` đổi thư mục kết quả. Với ảnh 3D, `--max-dim` quyết định độ phân giải lưới làm việc. Dùng `--help` ở từng lệnh để xem các tham số còn lại.

## Kiểm tra

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m unittest discover -s tests -v
```

Smoke train CPU trên Sunnybrook thật, với crop và batch nhỏ (khác full protocol):

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python train.py --architecture dlir \
    --dataset sunnybrook --preset sunnybrook-multi --fold 1 --run-name smoke \
    --steps-per-stage 1 --batch-size 1 --val-pairs 1 --val-every 1 \
    --crop-shape 6 24 24 --device cpu --output /tmp/registration-smoke
```

Không diễn giải checkpoint train vài bước thành chất lượng mô hình hoặc kết quả paper.
