from flask import Flask, Response
import threading
import cv2
import time

class FlaskMJPEGStreamer:
    def __init__(self, vision_system, host="0.0.0.0", port=8081):
        """
        Initialize the Flask-based MJPEG streamer.

        Args:
            vision_system: An instance of VisionSystem to get annotated frames.
            host (str): The host address to bind the Flask server. Default is "0.0.0.0".
            port (int): The port to serve the Flask application. Default is 8081.
        """
        self.vision_system = vision_system
        self.host = host
        self.port = port
        self.app = Flask(__name__)
        self.current_frame = None
        self.stop_thread = False

        # Define the video stream route
        @self.app.route('/stream')
        def stream():
            return Response(self.generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

    def start_stream(self):
        """
        Start the Flask server in a separate thread.
        """
        threading.Thread(target=self.run_flask, daemon=True).start()

    def run_flask(self):
        """
        Run the Flask application.
        """
        self.app.run(host=self.host, port=self.port, threaded=True)

    def generate_frames(self):
        """
        Generate annotated frames to serve as an MJPEG stream.
        """
        while not self.stop_thread:
            # Get the next frame and detections
            frame, detections = self.vision_system.run_inference()

            # Annotate the frame
            annotated_frame = self.vision_system.annotate_frame(frame, detections)

            # Convert to JPEG format
            _, jpeg = cv2.imencode('.jpg', annotated_frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

            # Limit frame rate
            time.sleep(0.05)

    def stop(self):
        """
        Stop the Flask streamer.
        """
        self.stop_thread = True
        print("Stopping Flask MJPEG streamer...")