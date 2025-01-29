import cv2
import socket
import json
import argparse
from inference_edgetpu import (
    load_labels,
    initialize_interpreter,
    run_inference_on_frame,
)
from pycoral.adapters.common import input_size
from pycoral.adapters.detect import get_objects
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter, run_inference

from flask_streamer import FlaskMJPEGStreamer

VIDEO_SOURCE = "http://192.168.0.169:8080/stream"

class VisionSystem:
    def __init__(self, model_path, label_path, udp_ip, udp_port, enable_stream=False, stream_host="192.168.0.169", stream_port=5000):
        self.labels = load_labels(label_path)
        self.interpreter = initialize_interpreter(model_path)
        #self.inference_size = (300, 300) 
        self.inference_size = (640, 480)

        # UDP settings
        self.udp_ip = udp_ip
        self.udp_port = udp_port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # Camera setup
        self.camera = cv2.VideoCapture("http://192.168.0.169:8080/stream")  # Adjust to your video source
        if not self.camera.isOpened():
            raise Exception("Failed to open camera.")

        # Streaming setup
        self.enable_stream = enable_stream
        self.stream_writer = None
        if self.enable_stream:
            self.stream_writer = cv2.VideoWriter(
                f'appsrc ! videoconvert ! x264enc tune=zerolatency bitrate=500 speed-preset=superfast ! rtph264pay ! udpsink host={stream_host} port={stream_port}',
                cv2.CAP_GSTREAMER,
                0,
                30,
                self.inference_size,
                True,
            )
            if not self.stream_writer.isOpened():
                raise Exception("Failed to initialize GStreamer stream.")

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

    def start(self, enable_stream=False, stream_port=8081):
        """ 
        Run inference and optionally start an MJPEG stream.

        Args:
            enable_stream (bool): Whether to start the Flask-based MJPEG streamer.
            stream_port (int): The port for the MJPEG stream (if enabled).
        """
        streamer = None
        try:
            if enable_stream:
                streamer = FlaskMJPEGStreamer(self, port=stream_port)
                streamer.start_stream()
                print(f"Flask MJPEG stream available at http://{streamer.host}:{stream_port}/stream")

            while True:
                frame, detections = self.run_inference()
                self.send_results(detections)

                # Save frame to disk for debugging
                output_path = f"output_frame.jpg"
                cv2.imwrite(output_path, frame)
        except KeyboardInterrupt:
            print("Shutting down VisionSystem...")
        finally:
            self.camera.release()
            if streamer:    
                streamer.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run vision system with optional MJPEG streaming.")
    parser.add_argument("--stream", action="store_true", help="Enable MJPEG streaming of annotated frames.")
    parser.add_argument("--stream_port", type=int, default=8081, help="Port for MJPEG stream (default: 8081).")
    args = parser.parse_args()

    MODEL_PATH = "Lan_test_3/tf2_ssd_mobilenet_v2_coco17_ptq_edgetpu.tflite"
    LABEL_PATH = "Lan_test_3/labels.txt"
    UDP_IP = "192.168.1.100"
    #UDP_PORT = 5005
    UDP_PORT = 60010
    
    vision_system = VisionSystem(MODEL_PATH, LABEL_PATH, UDP_IP, UDP_PORT)
    vision_system.start(enable_stream=args.stream, stream_port=args.stream_port)