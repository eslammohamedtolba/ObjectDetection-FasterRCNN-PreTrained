# Load dependencies
from flask import Flask, request, render_template
import numpy as np
import cv2
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import base64

# Create Flask application
app = Flask(__name__, template_folder='templates', static_folder='static', static_url_path='/')

# Load model and set model to the evaluation or prediction mode
model = fasterrcnn_resnet50_fpn(pretrained=True, progress=False)
model.eval()

# COCO dataset class names
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Create predict page route
@app.route('/', methods = ['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Get the uploaded image
        file = request.files['image']
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        batch = T.ToTensor()(image).unsqueeze(0)

        batch_prediction = model(batch)  
        image_prediction = batch_prediction[0]  # First image prediction

        # Extract prediction details
        boxes = image_prediction['boxes'].detach().cpu().numpy()
        scores = image_prediction['scores'].detach().cpu().numpy()
        labels = image_prediction['labels'].detach().cpu().numpy()
        # Filter out low-confidence predictions with threshold = 0.5
        confidence_threshold = 0.5
        high_conf_indices = scores >= confidence_threshold
        boxes = boxes[high_conf_indices]
        scores = scores[high_conf_indices]
        labels = labels[high_conf_indices]

        # Map labels to class names
        label_names = [COCO_INSTANCE_CATEGORY_NAMES[label] for label in labels]
        # Prepare labels and scores for display
        labels_scores = [f"Label: {label_name}, Score: {score:.2f}" for label_name, score in zip(label_names, scores)]

        # Generate random colors for the bounding boxes (in BGR format)
        colors = [tuple(np.random.randint(0, 255, size=3).tolist()) for _ in range(len(boxes))]
        # Draw bounding boxes on the image
        for box, color in zip(boxes, colors):
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # Convert the image back to RGB for display
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Encode the image to base64
        buffer = cv2.imencode('.png', image_rgb)[1]
        image_base64 = base64.b64encode(buffer).decode('utf-8')

        # Render the result
        return render_template('index.html', image=image_base64, labels_scores=labels_scores)
    
    return render_template('index.html')

