import time
from pycoral.adapters.common import input_size
from pycoral.adapters.detect import get_objects
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter, run_inference
from common import SVG
import gstreamer
import cv2
import os

# Đường dẫn mô hình và nhãn
MODEL_PATH = "/home/mendel/Lan_test_3/tf2_ssd_mobilenet_v2_coco17_ptq_edgetpu.tflite"
LABEL_PATH = "/home/mendel/Lan_test_3/labels.txt"
OUTPUT_DIR = "output_images"

# Tạo thư mục lưu ảnh nếu chưa tồn tại
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Tải nhãn
labels = read_label_file(LABEL_PATH)

# Tạo interpreter và chuẩn bị
interpreter = make_interpreter(MODEL_PATH)
interpreter.allocate_tensors()
inference_size = input_size(interpreter)

# Mở camera
camera = cv2.VideoCapture(1)  # Sử dụng camera kết nối
if not camera.isOpened():
    print("Không thể mở camera!")
    exit()

print("Bắt đầu lưu ảnh, nhấn 'q' để thoát.")

# Đếm số lượng ảnh đã lưu
saved_images = 0
max_images = 10

while saved_images < max_images:
    try:
        ret, frame = camera.read()
        if not ret:
            print("Không nhận được hình ảnh từ camera!")
            break

        # Chuẩn bị ảnh đầu vào
        input_tensor = cv2.resize(frame, inference_size)  # Resize ảnh về (300, 300)
        input_tensor = cv2.cvtColor(input_tensor, cv2.COLOR_BGR2RGB)
        input_tensor = input_tensor.astype('uint8')  # Đảm bảo định dạng chính xác

        # Thực hiện suy luận
        run_inference(interpreter, input_tensor.tobytes())

        # Trích xuất dữ liệu đầu ra
        boxes = interpreter.tensor(interpreter.get_output_details()[0]['index'])()[0]
        class_ids = interpreter.tensor(interpreter.get_output_details()[1]['index'])()[0]
        scores = interpreter.tensor(interpreter.get_output_details()[2]['index'])()[0]
        count = int(interpreter.tensor(interpreter.get_output_details()[3]['index'])()[0].item())  # Chuyển đổi chính xác
        
        # In giá trị đầu ra
        print("Boxes:", boxes)
        print("Class IDs:", class_ids)
        print("Scores:", scores)
        print("Count:", interpreter.tensor(interpreter.get_output_details()[3]['index'])())

        # Xử lý các dự đoán
        for i in range(count):
            if scores[i] > 0.5 and int(class_ids[i]) == 43:  # Chỉ xử lý nhãn "bottle"
                bbox = boxes[i]
                x_min, y_min, x_max, y_max = (
                    int(bbox[1] * frame.shape[1]),
                    int(bbox[0] * frame.shape[0]),
                    int(bbox[3] * frame.shape[1]),
                    int(bbox[2] * frame.shape[0]),
                )

                # Vẽ bounding box
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                # Hiển thị nhãn và xác suất
                label = f"{labels[int(class_ids[i])]}: {scores[i]:.2f}"
                cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Lưu ảnh
        output_path = os.path.join(OUTPUT_DIR, f"frame_{saved_images:03d}.jpg")
        cv2.imwrite(output_path, frame)
        print(f"Ảnh đã lưu: {output_path}")
        saved_images += 1

        # Chờ 5 giây trước khi chụp ảnh tiếp theo
        time.sleep(5)

    except Exception as e:
        print(f"Lỗi không xác định: {e}")
        break

# Giải phóng tài nguyên
camera.release()
print("Đã hoàn thành lưu ảnh.")
