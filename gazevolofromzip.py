import os
import cv2
import torch
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from zipfile import ZipFile
from io import BytesIO
from mivolo.predictor import Predictor
from timm.utils import setup_default_logging
from helpers import visualize_all, get_parser_zip, select_roi  # Import from helpers.py
import warnings
warnings.filterwarnings("ignore")

#zip 파일을 열고, 한 장에서 roi를 특정하면 반복적으로 돌아가며 데이터를 txt파일에 저장
def main():
    setup_default_logging()

    #zip파일에서 이미지만 버퍼
    zip_file_name = 'frames.zip'

    #
    with ZipFile(zip_file_name, 'r') as zip_file:
        file_names = zip_file.namelist()
        images = []
        image_file_names = []
        for file_name in file_names:
            if file_name.endswith('.png') or file_name.endswith('.jpg') or file_name.endswith('.jpeg'):
                data = zip_file.read(file_name)
                image = Image.open(BytesIO(data))
                images.append(image)
                image_file_names.append(file_name)   # update the name of image file
    print(f'{len(images)} images imported!')

    roi_state = {'box_points': [], 'roi_selected': False}

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    parser = get_parser_zip()
    args = parser.parse_args()
    device = args.device

    #첫 이미지 임포트, 미지정시 구역 지정, 지정시 등록
    img = images[0]
    img_cv2 = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    width, height = img.size
#    roi1 = args.roi1
#    roi2 = args.roi2
    roi1 = -1
    roi2 = -1

    if roi1 == -1 and roi2 == -1:
        print("Click to select ROI (2 points: top-left and bottom-right)...")
        cv2.namedWindow('Select ROI')
        cv2.setMouseCallback('Select ROI', select_roi)

        cv2.namedWindow('Select ROI')
        cv2.setMouseCallback('Select ROI', select_roi, roi_state)
    else:
        roi_state['box_points'].append(roi1)
        roi_state['box_points'].append(roi2)
        roi_state['roi_selected'] = True


    #ROI Select..
    while not roi_state['roi_selected']:
        temp_img = img_cv2.copy()
        # Draw the selected points and rectangle dynamically
        if len(roi_state['box_points']) == 1:
            cv2.circle(temp_img, roi_state['box_points'][0], 5, (0, 255, 0), -1)  # Draw the first point
        elif len(roi_state['box_points']) == 2:
            cv2.rectangle(temp_img, roi_state['box_points'][0], roi_state['box_points'][1], (0, 255, 0), 2)  # Draw rectangle
        cv2.imshow('Select ROI', temp_img)
        key = cv2.waitKey(1) & 0xFF
        # Press 'r' to reset the bounding box
        if key == ord('r'):
            roi_state['box_points'] = []
            roi_state['roi_selected'] = False
            print("ROI reset. Start selecting again.")
        # Press 'q' to quit selection
        elif key == ord('q'):
            print("Exiting ROI selection.")
            cv2.destroyAllWindows()
            return

    cv2.destroyAllWindows()
    print(f"ROI Selected: {roi_state['box_points']}")    

#각 이미지에 대해 반복
    for i in range (len(images)):
        img = images[i]
        raw_file_name = os.path.splitext(os.path.basename(image_file_names[i]))[0]  # updated line
        img_cv2 = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        # Load MiVOLO predictor
        print("loading MiVOLO Model..")
        predictor = Predictor(args)

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

        if len(face_indices) == 0:
            print('No bounding boxes detected by YOLO. Skipping to the next image.')
            continue  # Skipping to the next iteration in outer loop


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
        print("Predicting w/ Gazelle Model with YOLO body..")
        with torch.no_grad():
            output = model(gazelle_input)

        # Prepare data for visualization
        ages_list = ages
        genders_list = genders
        inout_scores = output['inout'][0] if output['inout'] is not None else None

        # Visualize results using visualize_all
        print(f'Detected boxes: {len(norm_bboxes[0])}')
        overlay_image = visualize_all(
            roi_state['box_points'],
            img,
            output['heatmap'][0],
            norm_bboxes[0],
            raw_file_name,
            inout_scores,
            ages_list,
            genders_list,
            inout_thresh=0.5,
        )

        print('saving as image...')
        # Save or display the result
        if args.output:
            os.makedirs(args.output, exist_ok=True)
            output_path = os.path.join(args.output, f"{raw_file_name}.png")
            overlay_image.convert("RGB").save(output_path)
            print(f"Result saved to {output_path}")
        else:
            plt.imshow(overlay_image)
            plt.axis('off')
            plt.show()

if __name__ == "__main__":
    main()