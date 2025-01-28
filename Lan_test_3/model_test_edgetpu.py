import time
from pycoral.adapters.common import input_size
from pycoral.adapters.detect import get_objects
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter, run_inference
import cv2
import os
import numpy as np

PWD = "/home/mendel/CoralCom/Lan_test_3/"

MODEL_PATH = PWD + "tf2_ssd_mobilenet_v2_coco17_ptq_edgetpu.tflite"
LABEL_PATH = PWD + "labels.txt"
OUTPUT_DIR = "output_images"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

labels = read_label_file(LABEL_PATH)

interpreter = make_interpreter(MODEL_PATH)
interpreter.allocate_tensors()
inference_size = input_size(interpreter)
print("inference size: ", inference_size)

VIDEO_SOURCE = "http://192.168.0.169:8080/stream"

#camera = cv2.VideoCapture(1)
camera = cv2.VideoCapture(VIDEO_SOURCE)
if not camera.isOpened():
    exit()

saved_images = 0
max_images = 10

while saved_images < max_images:
    try:
        ret, frame = camera.read()
        if not ret:
            break

        input_tensor = cv2.resize(frame, inference_size)
        input_tensor = cv2.cvtColor(input_tensor, cv2.COLOR_BGR2RGB)
        input_tensor = input_tensor.astype('uint8')
        input_tensor = np.ascontiguousarray(input_tensor)

        run_inference(interpreter, input_tensor.tobytes())

        #for output_detail in interpreter.get_output_details():
        #    print("Output tensor index:", output_detail['index'])
        #    print("Shape:", output_detail['shape'])
        #    print("Data type:", output_detail['dtype'])

        #boxes = interpreter.tensor(interpreter.get_output_details()[0]['index'])()[0]
        #class_ids = interpreter.tensor(interpreter.get_output_details()[1]['index'])()[0]
        #scores = interpreter.tensor(interpreter.get_output_details()[2]['index'])()[0]
        #count = int(interpreter.tensor(interpreter.get_output_details()[3]['index'])()[0].item())

        output_details = interpreter.get_output_details()

        for detail in output_details:
            tensor = interpreter.tensor(detail['index'])()
            print(f"Index {detail['index']} - Shape: {tensor.shape}, Data: {tensor}")


        #scores = interpreter.tensor(output_details[0]['index'])()[0]  # Shape: [20]
        #boxes = interpreter.tensor(output_details[1]['index'])()[0]   # Shape: [20, 4]
        #class_ids = interpreter.tensor(output_details[2]['index'])()[0]  # Shape: [20]
        #count_tensor = interpreter.tensor(output_details[3]['index'])()  # Shape: [1]

        #count_tensor = interpreter.tensor(output_details[3]['index'])()  # Shape: [1]
        scores = np.ascontiguousarray(interpreter.tensor(output_details[0]['index'])()[0])  # Shape: [20]
        boxes = np.ascontiguousarray(interpreter.tensor(output_details[1]['index'])()[0])   # Shape: [20, 4]
        class_ids = np.ascontiguousarray(interpreter.tensor(output_details[2]['index'])()[0])  # Shape: [20]
        #count_tensor = np.ascontiguousarray(interpreter.tensor(output_details[3]['index'])())  # Shape: [1]
        #count = int(count_tensor[0])  # Extract the single value

        count_tensor = interpreter.tensor(output_details[3]['index'])()  # Shape: [1]
        #count = count_tensor.item()  # Use .item() to extract the scalar
        #count = count_tensor.tolist()[0]  # Safely extract the scalar value
       
        # Determine count dynamically based on scores above a threshold
        count = sum(score > 0.5 for score in scores)  # Replace 0.5 with your threshold
        print(f"Detection count (dynamic): {count}")


        print("Scores:", scores.shape, scores)
        print("Boxes:", boxes.shape, boxes)
        print("Class IDs:", class_ids.shape, class_ids)
        print("Count Tensor:", count_tensor.shape, count_tensor)

        # Convert detection count to scalar
        #if count_tensor.size == 1:
        #    continue
            #count = int(count_tensor.flat[0])  # Use `.flat[0]` to access the scalar value
        #else:
        #    raise ValueError(f"Unexpected tensor size for detection count: {count_tensor.shape}")

        for i in range(count):
            if scores[i] > 0.5 and int(class_ids[i]) == 43:
                bbox = boxes[i]
                x_min, y_min, x_max, y_max = (
                    int(bbox[1] * frame.shape[1]),
                    int(bbox[0] * frame.shape[0]),
                    int(bbox[3] * frame.shape[1]),
                    int(bbox[2] * frame.shape[0]),
                )

                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                label = f"{labels[int(class_ids[i])]}: {scores[i]:.2f}"
                cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                print(f"Class ID: {class_id}, Score: {score}, Bounding Box: {bbox}")

        output_path = os.path.join(OUTPUT_DIR, f"frame_{saved_images:03d}.jpg")
        cv2.imwrite(output_path, frame)
        saved_images += 1

        time.sleep(5)

    except Exception as e:
        print(e)
        break

camera.release()
