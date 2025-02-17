import argparse
import os
import cv2
import requests
import base64
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
import time
import mysql.connector
import warnings
warnings.filterwarnings("ignore")



def upload_gaze(raw_file_name, age, gender, gazed):
    cnx = mysql.connector.connect(user='bbig09', password='bigproject_back09',
                              host='big09db.mysql.database.azure.com',
                              database='big092')
    cursor = cnx.cursor()
    frame_id_query = "SELECT id from frame where name = %s"
    cursor.execute(frame_id_query, (raw_file_name,))
    row = cursor.fetchone()
    # print(f'Got {row[0]} as frame_id..')

    if row is not None:
        frame_id = row[0]
        insert_gaze = "INSERT INTO ad_gaze(frame_id, pre_age, pre_gender, gazed) VALUES (%s, %s, %s, %s)"
        cursor.execute(insert_gaze, (frame_id, age, gender, gazed))
        cnx.commit()


def select_roi(event, x, y, flags, param):
    """
    Mouse callback function to capture ROI corners.
    :param event: Mouse event type
    :param x, y: Mouse click coordinates
    :param flags: Event flags
    :param param: A shared structure (e.g., dictionary) to store ROI state
    """
    state = param  # The shared state passed via `param`

    # Left mouse button down: Store points
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(state['box_points']) < 2:  # Add up to two points
            state['box_points'].append((x, y))
            print(f"Point {len(state['box_points'])} selected: ({x}, {y})")

        # If two points are selected, finalize ROI
        if len(state['box_points']) == 2:
            state['roi_selected'] = True
            print("ROI finalized.")

    # Right mouse button down: Reset the selection
    elif event == cv2.EVENT_RBUTTONDOWN:
        state['box_points'] = []
        state['roi_selected'] = False
        print("ROI reset. Start selecting again.")

def visualize_all(ad_area, pil_image, heatmaps, bboxes, raw_file_name, inout_scores=None, ages=None, genders=None, inout_thresh=0.5):
    from PIL import Image, ImageDraw, ImageFont
    colors = ['lime', 'cyan', 'fuchsia', 'yellow']
    overlay_image = pil_image.convert("RGBA")
    draw = ImageDraw.Draw(overlay_image)
    width, height = pil_image.size

    #AD area show
    draw.rectangle( 
            [ad_area[0][0], ad_area[1][1], ad_area[1][0], ad_area[0][1]],
            outline='tomato',
            width=int(min(width, height) * 0.01)
        )
    ad_font_size = max(36, int(min(width, height) * 0.06))

    try:
        font3 = ImageFont.truetype("arial.ttf", ad_font_size)
    except IOError:
        font3 = ImageFont.load_default()
    draw.multiline_text(((ad_area[0][0]+ad_area[1][0])/2-80, (ad_area[0][1]+ad_area[1][1])/2), 'AD AREA', fill='black', font=font3)

    #For Each Box..
    for i in range(len(bboxes)):
        gazed = False
        bbox = bboxes[i]
        xmin, ymin, xmax, ymax = bbox
        color = colors[i % len(colors)]
        draw.rectangle(
            [xmin * width, ymin * height, xmax * width, ymax * height],
            outline=color,
            width=int(min(width, height) * 0.005)
        )

        # Prepare text information
        text_lines = []
        if ages is not None and genders is not None:
            age = ages[i]
            gender = genders[i]
            text_lines.append(f"Age: {age:.1f}, Gender: {gender}")

        if inout_scores is not None:
            inout_score = inout_scores[i]
            text_lines.append(f"in-frame: {inout_score:.2f}")

        text = "\n".join(text_lines)
        # Adjust font size
        font_size = max(12, int(min(width, height) * 0.02))
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except IOError:
            font = ImageFont.load_default()
        text_x = xmin * width
        text_y = ymax * height - 60
        draw.multiline_text((text_x, text_y), text, fill=color, font=font)

        mid_font_size = max(24, int(min(width, height) * 0.04))
        try:
            font2 = ImageFont.truetype("arial.ttf", mid_font_size)
        except IOError:
            font2 = ImageFont.load_default()

        draw.multiline_text(((xmin+xmax)*width/2, (ymin+ymax)*height/2), str(i), fill=color, font=font2)


        # Visualize gaze if inout score is above threshold AND gaze is within AD box.
        if inout_scores is not None and inout_scores[i] > inout_thresh:
            heatmap = heatmaps[i]
            heatmap_np = heatmap.detach().cpu().numpy()
            max_index = np.unravel_index(np.argmax(heatmap_np), heatmap_np.shape)
            gaze_target_x = max_index[1] / heatmap_np.shape[1] * width
            gaze_target_y = max_index[0] / heatmap_np.shape[0] * height
            bbox_center_x = ((xmin + xmax) / 2) * width
            bbox_center_y = ((ymin + ymax) / 2) * height

            if gaze_target_x > ad_area[0][0] and gaze_target_x < ad_area[1][0] and gaze_target_y > ad_area[1][1] and gaze_target_y < ad_area[0][1]:
                print(f'Gaze Target for box[{i}]: x:{gaze_target_x}, y:{gaze_target_y}')
                print(f'Gender for box[{i}]: {genders[i]}')
                print(f'Age for box[{i}]: {ages[i]}')

                # Draw gaze point and line
                radius = int(0.01 * min(width, height))
                draw.ellipse(
                    [(gaze_target_x - radius, gaze_target_y - radius), (gaze_target_x + radius, gaze_target_y + radius)],
                    fill=color
                )
                draw.line(
                    [(bbox_center_x, bbox_center_y), (gaze_target_x, gaze_target_y)],
                    fill=color,
                    width=int(0.005 * min(width, height))
                )
                #print(f'gaze_target_x: {gaze_target_x}')
                #print(f'gaze_target_y: {gaze_target_y}')
                gazed = True
            else:
                print(f'Gaze Target for box[{i}]: x:{gaze_target_x}, y:{gaze_target_y}')
                print('Not looking at ad!!')

        # Uplaod to ad_gaze
        # upload_gaze(raw_file_name, age, gender, gazed)

    return overlay_image

def get_parser():
    parser = argparse.ArgumentParser(description="PyTorch Gazelle and MiVOLO Inference")
    parser.add_argument("--input", type=str, required=True, help="Image file or folder with images")
    parser.add_argument("--output", type=str, default = 'output', help="Folder for output results")
    parser.add_argument("--detector-weights", type=str, default="models/yolov8x_person_face.pt", help="Detector weights (YOLOv8).")
    parser.add_argument("--checkpoint", default="models/mivolo_imbd.pth.tar", type=str, help="Path to MiVOLO checkpoint")
    parser.add_argument("--with-persons", action="store_true", default=True, help="If set, model will run with persons, if available")
    parser.add_argument("--disable-faces", action="store_true", default=False, help="If set, model will use only persons if available")
    parser.add_argument("--draw", action="store_true", default=True, help="If set, resulting images will be drawn")
    parser.add_argument("--device", default="cuda", type=str, help="Device (accelerator) to use.")
    parser.add_argument("--roi1", type=float, default=-1, help="coord for left low")
    parser.add_argument("--roi2", type=float, default=-1, help="coord for top right")
    return parser

def get_parser_zip():
    parser = argparse.ArgumentParser(description="PyTorch Gazelle and MiVOLO Inference")
    parser.add_argument("--output", type=str, default = 'output', help="Folder for output results")
    parser.add_argument("--detector-weights", type=str, default="models/yolov8x_person_face.pt", help="Detector weights (YOLOv8).")
    parser.add_argument("--checkpoint", default="models/mivolo_imbd.pth.tar", type=str, help="Path to MiVOLO checkpoint")
    parser.add_argument("--with-persons", action="store_true", default=True, help="If set, model will run with persons, if available")
    parser.add_argument("--disable-faces", action="store_true", default=False, help="If set, model will use only persons if available")
    parser.add_argument("--draw", action="store_true", default=True, help="If set, resulting images will be drawn")
    parser.add_argument("--device", default="cuda", type=str, help="Device (accelerator) to use.")
    parser.add_argument("--roi1", type=float, default=-1, help="coord for left low")
    parser.add_argument("--roi2", type=float, default=-1, help="coord for top right")
    return parser

def upload_to_imgur(img, retry_count = 3):
    #IMGUR_CLIENT_ID = "76b478adb423c26"
    IMGUR_CLIENT_ID = "c48138d7019a1af"
    
    url = "https://api.imgur.com/3/image"
    headers = {"Authorization": f"Client-ID {IMGUR_CLIENT_ID}",
                   "User-Agent": "My Python App"
}

    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    img_byte = buffered.getvalue()
    encoded_string = base64.b64encode(img_byte).decode('utf-8')

    data = {"image": encoded_string}

    #response = requests.post(url, headers=headers, data=data) #Safer
    response = requests.post(url, headers=headers, data=data, verify=False)

    if response.status_code == 200:
        print(f'Success! Image uploaded to: {response.json()["data"]["link"]}')
        return response.json()["data"]["link"]
    elif retry_count > 0:
        print('Upload failed. Retrying...')
        time.sleep(5)  # Let's wait for 5 seconds before retrying
        return upload_to_imgur(img, retry_count - 1)
    else:
        print(f'Failed to upload image: {response.content}')
        return None