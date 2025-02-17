import argparse
import logging
import os
import cv2
import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from retinaface import RetinaFace
import numpy as np
from mivolo.data.data_reader import InputType, get_input_type
from mivolo.predictor import Predictor
from timm.utils import setup_default_logging

def get_parser():
    parser = argparse.ArgumentParser(description="PyTorch Gazelle and MiVOLO Inference")
    parser.add_argument("--input", type=str, default=None, required=True, help="image file or folder with images")
    parser.add_argument("--output", type=str, default=None, required=True, help="folder for output results")
    parser.add_argument("--detector-weights", type=str, default=None, required=True, help="Detector weights (YOLOv8).")
    parser.add_argument("--checkpoint", default="", type=str, required=True, help="path to mivolo checkpoint")
    parser.add_argument("--with-persons", action="store_true", default=False, help="If set model will run with persons, if available")
    parser.add_argument("--disable-faces", action="store_true", default=False, help="If set model will use only persons if available")
    parser.add_argument("--draw", action="store_true", default=False, help="If set, resulted images will be drawn")
    parser.add_argument("--device", default="cuda", type=str, help="Device (accelerator) to use.")
    return parser

def main():
    parser = get_parser()
    setup_default_logging()
    args = parser.parse_args()
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    # load Gazelle model
    device = args.device
    model, transform = torch.hub.load('fkryan/gazelle', 'gazelle_dinov2_vitb14_inout')
    model.eval()
    model.to(device)

    # load an input image
    image = Image.open(args.input)
    width, height = image.size

    # detect faces
    resp = RetinaFace.detect_faces(np.array(image))
    bboxes = [resp[key]['facial_area'] for key in resp.keys()]

    # prepare gazelle input
    img_tensor = transform(image).unsqueeze(0).to(device)
    norm_bboxes = [[np.array(bbox) / np.array([width, height, width, height]) for bbox in bboxes]]

    input = {
        "images": img_tensor,
        "bboxes": norm_bboxes
    }

    with torch.no_grad():
        output = model(input)

    # load MiVOLO predictor
    predictor = Predictor(args)
    img = cv2.imread(args.input)
    detected_objects, _ = predictor.recognize(img)

    # visualize predicted results
    draw = ImageDraw.Draw(image)

    # Visualize MiVOLO results
    # Process faces
    bboxes_faces, inds_faces = detected_objects.get_bboxes_inds('face')
    for bbox, ind in zip(bboxes_faces, inds_faces):
        xmin, ymin, xmax, ymax = bbox
        draw.rectangle([xmin, ymin, xmax, ymax], outline="lime", width=int(min(width, height) * 0.01))
        text = f"Age: {detected_objects.ages[ind]}, Sex: {detected_objects.genders[ind]}"
        draw.text((xmin, ymax + 10), text, fill="lime")

    # Process persons (if you want to include them)
    bboxes_persons, inds_persons = detected_objects.get_bboxes_inds('person')
    for bbox, ind in zip(bboxes_persons, inds_persons):
        xmin, ymin, xmax, ymax = bbox
        draw.rectangle([xmin, ymin, xmax, ymax], outline="cyan", width=int(min(width, height) * 0.01))
        text = f"Age: {detected_objects.ages[ind]}, Sex: {detected_objects.genders[ind]}"
        draw.text((xmin, ymax + 10), text, fill="cyan")

    # Visualize Gazelle results
    for i in range(len(bboxes)):
        img_person_heatmap = output['heatmap'][0][i]
        if model.inout:
            img_person_inout = output['inout'][0][i]

        # visualize heatmap
        if isinstance(img_person_heatmap, torch.Tensor):
            img_person_heatmap = img_person_heatmap.detach().cpu().numpy()
        heatmap = Image.fromarray((img_person_heatmap * 255).astype(np.uint8)).resize(image.size, Image.Resampling.BILINEAR)
        heatmap = plt.cm.jet(np.array(heatmap) / 255.)
        heatmap = (heatmap[:, :, :3] * 255).astype(np.uint8)
        heatmap = Image.fromarray(heatmap).convert("RGBA")
        heatmap.putalpha(90)
        image = Image.alpha_composite(image.convert("RGBA"), heatmap)

    # Save or display the final image
    image.save(os.path.join(args.output, 'result.png'))
    plt.imshow(image)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()