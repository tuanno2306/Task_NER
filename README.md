# Drug NER

Nhận diện thực thể thuốc trong DDI Corpus bằng embedding BiomedBERT, Flair và FastText đóng băng, kết hợp Attention → BiLSTM (32 hidden units) → CRF.

## 1. Cài đặt

Mở terminal trong thư mục `task_NER`, sau đó cài các thư viện:

```bash
python -m pip install torch scikit-learn seqeval tqdm pytorch-crf transformers flair fasttext-wheel huggingface_hub
```

Lần chạy đầu cần Internet để tải các embedding pretrained. FastText có dung lượng lớn; dù đã có checkpoint, chương trình vẫn cần các tài nguyên embedding để khởi tạo model.

## 2. Cấu trúc dự án

```text
task_NER/
├── data/
│   ├── Train/
│   │   ├── DrugBank/
│   │   └── MedLine/
│   └── Test/
│       └── Test for DrugNER task/
├── models/
│   └── best _no_grad_32.pt
├── NOTEBOOK/
├── data_setup.py       # Đọc XML, tạo nhãn BIO và DataLoader
├── model_builder.py    # Embedding, Attention, BiLSTM và CRF
├── engine.py           # Huấn luyện và đánh giá
├── utils.py            # Seed và checkpoint
└── train.py            # Điểm chạy chương trình
```

Chương trình tự tách 10% tài liệu XML của từng nguồn trong `Train` làm tập dev. Tập test sử dụng thư mục `Test for DrugNER task`.

## 3. Đánh giá model có sẵn

```bash
python train.py --eval-only
```

Mặc định sử dụng `models/best _no_grad_32.pt`. Để chọn checkpoint khác:

```bash
python train.py --eval-only --checkpoint "models/best_modular.pt"
```

Kết quả gồm loss, precision, recall và F1 theo strict IOB2 ở mức thực thể. Chế độ này vẫn cần cả dữ liệu train và test theo cấu trúc trên.

## 4. Huấn luyện

```bash
python train.py
```

Mặc định: 10 epochs, batch size 32, learning rate 0.001; dừng sớm sau 3 epochs không cải thiện dev F1. Chương trình tự chọn CUDA nếu khả dụng, nếu không sẽ dùng CPU.

Ví dụ tùy chỉnh:

```bash
python train.py --epochs 20 --batch-size 16 --lr 0.001 --patience 5
```

Chạy trên CPU:

```bash
python train.py --device cpu
```

Checkpoint tốt nhất theo dev F1 được lưu vào `models/best_modular.pt`, sau đó được nạp lại để đánh giá trên test. Lệnh mặc định không ghi đè trọng số gốc `best _no_grad_32.pt`; các lần train tiếp theo sẽ ghi đè `best_modular.pt`.

## 5. Kết quả và tùy chọn

Các báo cáo được lưu trong `outputs/`:

- `split_manifest.json`: danh sách tài liệu của từng tập.
- `annotation_report.json`: báo cáo căn chỉnh annotation với token.
- `history.json`: lịch sử huấn luyện, chỉ tạo khi train.
- `test_metrics.json`: kết quả đánh giá test.

Nếu đã có FastText `model.bin`, chỉ định đường dẫn để sử dụng:

```bash
python train.py --eval-only --fasttext-path "/duong/dan/model.bin" --no-download-fasttext
```

Xem toàn bộ tùy chọn:

```bash
python train.py --help
```
