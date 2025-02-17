import argparse
import torch
from PIL import Image
import json
import os
import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from gazelle.model import get_gazelle_model
from gazelle.model import GazeLLE
from gazelle.backbone import DinoV2Backbone

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="./data/gazefollow")
parser.add_argument("--model_name", type=str, default="gazelle_dinov2_vitl14_inout")
parser.add_argument("--ckpt_path", type=str, default="./checkpoints/gazelle_dinov2_vitl14_inout.pt")
parser.add_argument("--batch_size", type=int, default=128)
args = parser.parse_args()

class GazeFollow(torch.utils.data.Dataset):
    def __init__(self, path, img_transform):
        self.images = json.load(open(os.path.join(path, "test_preprocessed.json"), "rb"))
        self.path = path
        self.transform = img_transform

    def __getitem__(self, idx):
        item = self.images[idx]
        image = self.transform(Image.open(os.path.join(self.path, item['path'])).convert("RGB"))
        height = item['height']
        width = item['width']
        bboxes = [head['bbox_norm'] for head in item['heads']]
        gazex = [head['gazex_norm'] for head in item['heads']]
        gazey = [head['gazey_norm'] for head in item['heads']]

        return image, bboxes, gazex, gazey, height, width

    def __len__(self):
        return len(self.images)
    
def collate(batch):
    images, bboxes, gazex, gazey, height, width = zip(*batch)
    return torch.stack(images), list(bboxes), list(gazex), list(gazey), list(height), list(width)

# GazeFollow calculates AUC using original image size with GT (x,y) coordinates set to 1 and everything else as 0
# References:
    # https://github.com/ejcgt/attention-target-detection/blob/acd264a3c9e6002b71244dea8c1873e5c5818500/eval_on_gazefollow.py#L78
    # https://github.com/ejcgt/attention-target-detection/blob/acd264a3c9e6002b71244dea8c1873e5c5818500/utils/imutils.py#L67
    # https://github.com/ejcgt/attention-target-detection/blob/acd264a3c9e6002b71244dea8c1873e5c5818500/utils/evaluation.py#L7
def gazefollow_auc(heatmap, gt_gazex, gt_gazey, height, width):
    target_map = np.zeros((height, width))
    for point in zip(gt_gazex, gt_gazey):
        if point[0] >= 0:
            x, y = map(int, [point[0]*float(width), point[1]*float(height)])
            x = min(x, width - 1)
            y = min(y, height - 1)
            target_map[y, x] = 1
    resized_heatmap = torch.nn.functional.interpolate(heatmap.unsqueeze(dim=0).unsqueeze(dim=0), (height, width), mode='bilinear').squeeze()
    auc = roc_auc_score(target_map.flatten(), resized_heatmap.cpu().flatten())
    
    return auc

# Reference: https://github.com/ejcgt/attention-target-detection/blob/acd264a3c9e6002b71244dea8c1873e5c5818500/eval_on_gazefollow.py#L81
def gazefollow_l2(heatmap, gt_gazex, gt_gazey):
    argmax = heatmap.flatten().argmax().item()
    pred_y, pred_x = np.unravel_index(argmax, (64, 64))
    pred_x = pred_x / 64.
    pred_y = pred_y / 64.

    gazex = np.array(gt_gazex)
    gazey = np.array(gt_gazey)

    avg_l2 = np.sqrt((pred_x - gazex.mean())**2 + (pred_y - gazey.mean())**2)
    all_l2s = np.sqrt((pred_x - gazex)**2 + (pred_y - gazey)**2)
    min_l2 = all_l2s.min().item()

    return avg_l2, min_l2


@torch.no_grad()
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Running on {}".format(device))

    model, transform = get_gazelle_model(args.model_name)
    model.load_gazelle_state_dict(torch.load(args.ckpt_path, weights_only=True))
    model.to(device)
    model.eval()

    dataset = GazeFollow(args.data_path, transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate)

    aucs = []
    min_l2s = []
    avg_l2s = []

    for _, (images, bboxes, gazex, gazey, height, width) in tqdm(enumerate(dataloader), desc="Evaluating", total=len(dataloader)):
        preds = model.forward({"images": images.to(device), "bboxes": bboxes})
        
        # eval each instance (head)
        for i in range(images.shape[0]): # per image
            for j in range(len(bboxes[i])): # per head
                auc = gazefollow_auc(preds['heatmap'][i][j], gazex[i][j], gazey[i][j], height[i], width[i])
                avg_l2, min_l2 = gazefollow_l2(preds['heatmap'][i][j], gazex[i][j], gazey[i][j])
                aucs.append(auc)
                avg_l2s.append(avg_l2)
                min_l2s.append(min_l2)
    
    print("AUC: {}".format(np.array(aucs).mean()))
    print("Avg L2: {}".format(np.array(avg_l2s).mean()))
    print("Min L2: {}".format(np.array(min_l2s).mean()))

        
if __name__ == "__main__":
    main()