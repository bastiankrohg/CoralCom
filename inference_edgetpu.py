import time
from pycoral.adapters.common import input_size
from pycoral.adapters.detect import get_objects
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter, run_inference
import numpy as np


def load_labels(label_path):
    """Load labels from a file."""
    return read_label_file(label_path)


def initialize_interpreter(model_path):
    """Initialize and allocate tensors for the model."""
    interpreter = make_interpreter(model_path)
    interpreter.allocate_tensors()
    return interpreter


def run_inference_on_frame(interpreter, input_frame, labels, threshold=0.5):
    """
    Run inference on a single input frame.

    Args:
        interpreter: The TensorFlow Lite interpreter.
        input_frame: The frame resized to model's input size.
        labels: Dictionary of labels.
        threshold: Confidence threshold for filtering detections.

    Returns:
        List of detection dictionaries with class_id, bbox, and score.
    """
    input_tensor = input_frame.astype("uint8")
    input_tensor = np.ascontiguousarray(input_tensor)

    # Perform inference
    run_inference(interpreter, input_tensor.tobytes())

    # Extract output tensors
    output_details = interpreter.get_output_details()
    scores = np.ascontiguousarray(interpreter.tensor(output_details[0]['index'])()[0])  # Shape: [N]
    boxes = np.ascontiguousarray(interpreter.tensor(output_details[1]['index'])()[0])   # Shape: [N, 4]
    class_ids = np.ascontiguousarray(interpreter.tensor(output_details[2]['index'])()[0])  # Shape: [N]

    # Debug tensor shapes and content
    #print(f"Scores shape: {scores.shape}")
    #print(f"Boxes shape: {boxes.shape}")
    #print(f"Class IDs shape: {class_ids.shape}")
    #print(f"Raw scores data: {scores}")
    #print(f"Raw class_ids data: {class_ids}")

    # Handle empty outputs (no detections)
    if scores.size == 0 or boxes.size == 0 or class_ids.size == 0:
        print("No detections found.")
        return []

    # Filter detections based on threshold and ensure index bounds
    detections = []
    for i in range(min(len(scores), len(class_ids), len(boxes))):
        if scores[i] > threshold:
            bbox = boxes[i].tolist()  # Convert bounding box to list
            class_id = int(class_ids[i]) if i < len(class_ids) else -1  # Handle missing class IDs
            detections.append({
                "class_id": class_id,
                "bbox": bbox,
                "score": float(scores[i]),
                "label": labels.get(class_id, "Unknown"),
            })

    return detections
