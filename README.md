# Nguyên mẫu GRU và FedAvg bằng NumPy

Đây là kiểm tra kỹ thuật vòng học liên kết trên **chuỗi đặc trưng mô phỏng**, không phải phần mềm nhận diện hành vi lợn từ video. Mã được xây dựng với sự hỗ trợ của AI.

## Cài đặt và chạy

Yêu cầu Python 3 và NumPy. Trong môi trường Python của người dùng:

```text
python -m pip install -r requirements.txt
python run_experiment.py --out results
```

Lệnh tái lập tại máy đã thực hiện thử nghiệm (PowerShell):

```powershell
& 'C:/Users/Admin/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' -X utf8 run_experiment.py --out results
```

## Thành phần đã cài đặt

- Sinh 3 dạng chuỗi: ổn định, trôi tuyến tính và dao động; chiếu qua ma trận 4 x 8 cố định rồi tanh.
- GRU 12 chiều ẩn, đầu phân loại 3 lớp, lan truyền ngược qua thời gian, SGD và giới hạn chuẩn gradient.
- Ba máy khách có tỷ lệ lớp khác nhau; mỗi vòng học một epoch rồi FedAvg có trọng số theo số mẫu.
- Đối chứng học riêng từng máy khách và học tập trung với cùng khởi tạo, số epoch và cùng tập kiểm tra.
- 3 hạt giống (random seed) cố định; xuất kết quả JSON, lịch sử loss CSV và trọng số NPZ.
- Kiểm tra gradient bằng sai phân hữu hạn tại một tọa độ của mỗi tensor, trung bình có trọng số, từ chối sai kích thước, giảm loss trên batch nhỏ, bất biến bộ chiếu và không có chuỗi trùng train/test.

## Đọc kết quả

`results/results.json` chứa cấu hình, môi trường, kiểm tra, kết quả từng lần chạy và trung bình/độ lệch chuẩn mẫu. `learning_curves.csv` chứa loss huấn luyện; tập kiểm tra chỉ được đánh giá sau 20 vòng, không dùng chọn cấu hình. Mọi lớp đều là tín hiệu nhân tạo, không tương ứng trực tiếp với nằm/đi lại/khác của lợn.

Số byte là tổng dung lượng mảng float64 được trao đổi theo thiết kế mỗi vòng, **không phải lưu lượng mạng thực đo**. Không tính đóng gói, giao thức hay phân phối ma trận chiếu ban đầu. Thời gian đo bằng đồng hồ đơn điệu trên một máy, các máy khách chạy tuần tự; không phản ánh thời gian trên thiết bị biên độc lập. Mỗi mô hình local chỉ thấy 90 mẫu, còn centralized/FedAvg tổng cộng 270 mẫu; khác biệt này là một nguyên nhân của chênh lệch hiệu năng, không chứng minh ưu thế thuật toán đơn thuần.

Chưa cài đặt: xử lý video, YOLO, ByteTrack, MobileNetV3, Flower, FedProx, dịch vụ mạng/TLS, MQTT/ThingsBoard hay triển khai đa thiết bị. Không có cam kết riêng tư. Kết quả này chỉ chứng minh đường tính toán của nguyên mẫu hoạt động trong bài toán nhân tạo dễ kiểm soát.
