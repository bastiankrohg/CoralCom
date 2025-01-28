import cv2
import socket
import json
from inference_edgetpu import (
    load_labels,
    initialize_interpreter,
    run_inference_on_frame,
)
from pycoral.adapters.common import input_size
from pycoral.adapters.detect import get_objects
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter, run_inference


VIDEO_SOURCE = "http://192.168.0.169:8080/stream"

class VisionSystem:
    def __init__(self, model_path, label_path, udp_ip, udp_port):
        self.labels = load_labels(label_path)
        self.interpreter = initialize_interpreter(model_path)
        self.inference_size = input_size(self.interpreter)

        # UDP settings
        self.udp_ip = udp_ip
        self.udp_port = udp_port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # Camera setup
        #self.camera = cv2.VideoCapture(0)
        self.camera = cv2.VideoCapture(VIDEO_SOURCE)
        if not self.camera.isOpened():
            raise Exception("Failed to open camera.")

    def run_inference(self):
        ret, frame = self.camera.read()
        if not ret:
            raise Exception("Failed to read frame from camera.")

        # Resize frame to model input size
        input_frame = cv2.resize(frame, self.inference_size)
        input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)

        # Run inference
        detections = run_inference_on_frame(self.interpreter, input_frame, self.labels)
        return frame, detections

    def send_results(self, detections):
        payload = json.dumps(detections)
        self.sock.sendto(payload.encode(), (self.udp_ip, self.udp_port))
        print(f"Sent: {payload}")

    def annotate_frame(self, frame, detections):
        for detection in detections:
            bbox = detection["bbox"]
            class_id = detection["class_id"]
            score = detection["score"]

            ymin, xmin, ymax, xmax = bbox
            start_point = (int(xmin * frame.shape[1]), int(ymin * frame.shape[0]))
            end_point = (int(xmax * frame.shape[1]), int(ymax * frame.shape[0]))
            color = (0, 255, 0)
            thickness = 2

            cv2.rectangle(frame, start_point, end_point, color, thickness)
            label = f"{detection['label']}: {score:.2f}"
            cv2.putText(frame, label, (start_point[0], start_point[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)
        return frame

    def start(self):
        try:
            while True:
                frame, detections = self.run_inference()
                self.send_results(detections)
                annotated_frame = self.annotate_frame(frame, detections)
                #cv2.imshow("Annotated Stream", annotated_frame)
                annotated_frame = self.annotate_frame(frame, detections)
                output_path = f"output_frame.jpg"  # Overwrites the same file
                cv2.imwrite(output_path, annotated_frame)            

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            self.camera.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    MODEL_PATH = "Lan_test_3/tf2_ssd_mobilenet_v2_coco17_ptq_edgetpu.tflite"
    LABEL_PATH = "Lan_test_3/labels.txt"
    UDP_IP = "192.168.1.100"
    UDP_PORT = 5005

    vision_system = VisionSystem(MODEL_PATH, LABEL_PATH, UDP_IP, UDP_PORT)
    vision_system.start()
