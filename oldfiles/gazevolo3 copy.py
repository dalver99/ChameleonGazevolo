import argparse
import os
import cv2
import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from PIL import ImageFont
import numpy as np
from mivolo.predictor import Predictor  # MiVOLO specific
from timm.utils import setup_default_logging  # Logging for timm
from helpers import visualize_all, get_parser  # Import helpers
import warnings
warnings.filterwarnings("ignore")

# List to store bounding box points
box_points = []
roi_selected = False


def select_roi(event, x, y, flags, param):
    """
    Mouse callback function to capture ROI corners.
    """
    global box_points, roi_selected

    # Left mouse click to store first point
    if event == cv2.EVENT_LBUTTONDOWN and len(box_points) < 2:
        box_points.append((x, y))
        print(f"Point {len(box_points)} selected: {x}, {y}")

    # If two points are selected, ROI is finalized
    if len(box_points) == 2:
        roi_selected = True


def main():
    global box_points, roi_selected

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

    # Read the input image in OpenCV and display it
    img_cv2 = cv2.imread(args.input)
    if img_cv2 is None:
        print(f"Could not load image: {args.input}")
        return

    img = Image.open(args.input)
    width, height = img.size

    print("Click to select ROI (2 points: top-left and bottom-right)...")
    cv2.namedWindow('Select ROI')
    cv2.setMouseCallback('Select ROI', select_roi)

    while not roi_selected:
        temp_img = img_cv2.copy()
        if len(box_points) == 1:
            cv2.circle(temp_img, box_points[0], 5, (0, 255, 0), -1)  # Draw the first point
        elif len(box_points) == 2:
            cv2.rectangle(temp_img, box_points[0], box_points[1], (0, 255, 0), 2)  # Draw rectangle
        cv2.imshow('Select ROI', temp_img)
        key = cv2.waitKey(1) & 0xFF

        # Press 'r' to reset the bounding box
        if key == ord('r'):
            box_points = []
            roi_selected = False
            print("ROI reset. Start selecting again.")

        # Press 'q' to quit selection
        elif key == ord('q'):
            print("Exiting ROI selection.")
            cv2.destroyAllWindows()
            return

    cv2.destroyAllWindows()
    print(f"ROI Selected: {box_points}")

    # Extract the face/body region using the selected ROI
    bboxes = [box_points[0][0], box_points[0][1], box_points[1][0], box_points[1][1]]
    bboxes = np.array(bboxes).reshape(1, 4)  # Convert to expected format (1, 4)

    # Perform detection and age/gender estimation
    print("Predicting w/ YOLO & MiVOLO Model..")
    detected_objects, _ = predictor.recognize(img_cv2)

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
    print("Predicting w/ Gazelle Model with selected ROI..")

    # Visualize results using visualize_all
    inout_scores = output['inout'][0] if output['inout'] is not None else None
    overlay_image = visualize_all(
        img,
        output['heatmap'][0],
        norm_bboxes[0],
        None,
        None,
        None,
        inout_thresh=0.5
    )

    print('Saving as image...')
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