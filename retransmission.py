import socket
import threading
import time
import json
import logging
from queue import Queue

class TelemetryRetransmission:
    def __init__(self, rover_ip, rover_port, earth_ip, earth_port, log_file="telemetry_coral.log", buffer_size=500, resend_interval=1):
        """
        Initializes the retransmission system.

        :param rover_ip: IP address to listen to the rover's telemetry.
        :param rover_port: Port to listen to the rover's telemetry.
        :param earth_ip: IP address to send data to Earth (Mission Control).
        :param earth_port: Port to send data to Earth (Mission Control).
        :param log_file: File to log telemetry data.
        :param buffer_size: Number of telemetry updates to keep in memory.
        :param resend_interval: Time in seconds to retry sending failed data.
        """
        self.rover_ip = rover_ip
        self.rover_port = rover_port
        self.earth_ip = earth_ip
        self.earth_port = earth_port
        self.log_file = log_file
        self.buffer_size = buffer_size
        self.resend_interval = resend_interval

        self.telemetry_buffer = Queue(maxsize=self.buffer_size)
        self.link_status = {"to_rover": True, "to_earth": True}
        self.stop_event = threading.Event()

        # Set up logging
        logging.basicConfig(level=logging.INFO, filename=self.log_file, filemode='a',
                            format='%(asctime)s - %(message)s')

        # UDP Socket for receiving telemetry from the rover
        self.rover_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.rover_socket.bind((self.rover_ip, self.rover_port))

        # TCP Socket for sending telemetry to Earth
        self.earth_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.earth_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    def start(self):
        """Start the telemetry retransmission system."""
        print("Starting telemetry retransmission system...")

        self.reception_thread = threading.Thread(target=self.receive_telemetry, daemon=True)
        self.retransmission_thread = threading.Thread(target=self.retransmit_telemetry, daemon=True)

        self.reception_thread.start()
        self.retransmission_thread.start()

    def stop(self):
        """Stop the telemetry retransmission system."""
        print("Stopping telemetry retransmission system...")
        self.stop_event.set()
        self.rover_socket.close()
        self.earth_socket.close()

    def receive_telemetry(self):
        """Receive telemetry data from the rover."""
        print("Listening for telemetry from the rover...")
        while not self.stop_event.is_set():
            try:
                data, addr = self.rover_socket.recvfrom(1024)
                telemetry = json.loads(data.decode('utf-8'))

                # Add telemetry to buffer
                if self.telemetry_buffer.full():
                    self.telemetry_buffer.get()  # Discard the oldest telemetry

                self.telemetry_buffer.put(telemetry)

                # Log the telemetry
                logging.info(f"Received telemetry: {telemetry}")

                self.link_status["to_rover"] = True
            except (socket.error, json.JSONDecodeError) as e:
                print(f"Error receiving telemetry: {e}")
                self.link_status["to_rover"] = False

    def retransmit_telemetry(self):
        """Retransmit telemetry data to Earth."""
        print("Retransmitting telemetry to Earth...")
        try:
            self.earth_socket.connect((self.earth_ip, self.earth_port))
            while not self.stop_event.is_set():
                if not self.telemetry_buffer.empty():
                    telemetry = self.telemetry_buffer.get()
                    try:
                        self.earth_socket.sendall(json.dumps(telemetry).encode('utf-8'))
                        logging.info(f"Sent telemetry to Earth: {telemetry}")
                        self.link_status["to_earth"] = True
                    except socket.error as e:
                        print(f"Error sending telemetry to Earth: {e}")
                        self.link_status["to_earth"] = False

                        # Retry logic
                        time.sleep(self.resend_interval)
                        self.telemetry_buffer.put(telemetry)  # Requeue telemetry for retry
                else:
                    time.sleep(0.1)  # Prevent busy waiting
        except socket.error as e:
            print(f"Error connecting to Earth socket: {e}")
            self.link_status["to_earth"] = False

    def get_link_status(self):
        """Return the current link statuses."""
        return self.link_status

if __name__ == "__main__":
    # Define configuration
    ROVER_IP = "0.0.0.0"  # Listen on all interfaces
    ROVER_PORT = 50055
    EARTH_IP = "127.0.0.1"  # Replace with the actual mission control IP
    EARTH_PORT = 60066

    telemetry_retransmission = TelemetryRetransmission(
        rover_ip=ROVER_IP,
        rover_port=ROVER_PORT,
        earth_ip=EARTH_IP,
        earth_port=EARTH_PORT
    )

    try:
        telemetry_retransmission.start()
        while True:
            time.sleep(1)  # Keep the main thread alive
    except KeyboardInterrupt:
        print("Shutting down...")
        telemetry_retransmission.stop()
