
import numpy as np
import scipy.special # For softmax
from coco_classes import COCO_CLASSES # Ensure COCO_CLASSES is imported
import onnxruntime as ort
from PIL import Image
import time

# from rfdetr.assets.coco_classes import COCO_CLASSES # Not strictly needed here, but doesn't hurt

# Load the ONNX model
session = ort.InferenceSession("rf_detr_nano.onnx")

# Prepare input image
image = Image.open("pizza.png").convert("RGB")
image = image.resize((384, 384))  # Resize to model's input resolution
image_array = np.array(image).astype(np.float32) / 255.0

# Normalize
mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
image_array = (image_array - mean) / std

# Convert to NCHW format
image_array = np.transpose(image_array, (2, 0, 1))
image_array = np.expand_dims(image_array, axis=0)

# Run inference
start_time = time.time()
outputs = session.run(None, {"input": image_array})
end_time = time.time()

print(f"Inference Time: {(end_time - start_time) * 1000:.2f} ms")
# Assign outputs to distinct, descriptive variable names and make them global
#global onnx_boxes, onnx_raw_logits
onnx_boxes = outputs[0]
onnx_raw_logits = outputs[1]

# Assuming `onnx_raw_logits` is available from the previous ONNX inference cell
# `onnx_raw_logits` here represents the raw logits (batch_size, num_queries, num_classes + 1)

# Get the 2D array of logits (remove batch dimension)
raw_logits = onnx_raw_logits[0]

# Apply softmax to logits to get probabilities
probabilities = scipy.special.softmax(raw_logits, axis=-1)

# Extract class IDs and confidence scores, excluding the 'no object' class (last column)
class_ids_all = np.argmax(probabilities[:, :-1], axis=-1)
confidences_all = np.max(probabilities[:, :-1], axis=-1)

# Combine class_ids and confidences into a list of tuples
detector_outputs = []
for i in range(len(class_ids_all)):
    detector_outputs.append((class_ids_all[i], confidences_all[i]))

# Sort by confidence in descending order
detector_outputs.sort(key=lambda x: x[1], reverse=True)

print("Top 5 Detection IDs and their Confidences:")
for i, (class_id_val, confidence_val) in enumerate(detector_outputs[:5]):
    # Use COCO_CLASSES directly, which is 1-indexed
    class_name = COCO_CLASSES.get(class_id_val, f'Unknown Class {class_id_val}')
    print(f"  {i+1}. Class Name: {class_name}, ID: {class_id_val}, Confidence: {confidence_val:.4f}")
