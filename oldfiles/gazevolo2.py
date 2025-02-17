import argparse
import os
import cv2
import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from PIL import ImageFont
import numpy as np
from mivolo.predictor import Predictor
from timm.utils import setup_default_logging
from helpers import visualize_all, get_parser  # Import from helpers.py
import warnings
warnings.filterwarnings("ignore")

def main():
    parser = get_parser()
    setup_default_logging()
    args = parser.parse_args()
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    device = args.device

    # Load MiVOLO predictor
    print("loading MiVOLO Model..")

    predictor = Predictor(args)
    img_cv2 = cv2.imread(args.input)
    img = Image.open(args.input)
    width, height = img.size

    # Perform detection and age/gender estimation with MiVOLO
    detected_objects, _ = predictor.recognize(img_cv2)
    print("Predicting w/ YOLO & MiVOLO Model..")

    #Extract face/body bounding boxes and indices from yolo_results
    yolo_results = detected_objects.yolo_results
    boxes = yolo_results.boxes.xyxy.cpu().numpy()
    classes = yolo_results.boxes.cls.cpu().numpy()

    #face_class_indices = [0, 1]  # 0: 'person', 1: 'face'
    face_class_indices = [0]  # 0: 'person', 1: 'face'
    face_indices = np.where(np.isin(classes, face_class_indices))[0]
    bboxes = boxes[face_indices]

    # Get ages and genders for faces
    ages = [detected_objects.ages[i] for i in face_indices]
    genders = [detected_objects.genders[i] for i in face_indices]

    # Load Gazelle model
    print("loading Gazelle Model..")
    model, transform = torch.hub.load('fkryan/gazelle', 'gazelle_dinov2_vitb14_inout', source='github')
    model.eval()
    model.to(device)

    # Prepare Gazelle input
    img_tensor = transform(img).unsqueeze(0).to(device)
    norm_bboxes = [[bbox / np.array([width, height, width, height]) for bbox in bboxes]]
    gazelle_input = {
        "images": img_tensor,
        "bboxes": norm_bboxes
    }

    # Predict!
    with torch.no_grad():
        output = model(gazelle_input)
    print("Predicting w/ Gazelle Model with YOLO body..")

    # Prepare data for visualization
    ages_list = ages
    genders_list = genders
    inout_scores = output['inout'][0] if output['inout'] is not None else None

    # Visualize results using visualize_all
    print(f'Detected boxes: {len(norm_bboxes[0])}')
    overlay_image = visualize_all(
        img,
        output['heatmap'][0],
        norm_bboxes[0],
        inout_scores,
        ages_list,
        genders_list,
        inout_thresh=0.5
    )

    print('saving as image...')
    # Save or display the result
    if args.output:
        os.makedirs(args.output, exist_ok=True)
        output_path = os.path.join(args.output, os.path.basename(args.input))
        overlay_image.convert("RGB").save(output_path)
        print(f"Result saved to {output_path}")
    else:
        plt.imshow(overlay_image)
        plt.axis('off')
        plt.show()

if __name__ == "__main__":
    main()