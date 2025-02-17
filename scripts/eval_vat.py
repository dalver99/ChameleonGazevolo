import argparse
import torch
from PIL import Image
import json
import os
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm

from gazelle.model import get_gazelle_model

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="./data/videoattentiontarget")
parser.add_argument("--model_name", type=str, default="gazelle_dinov2_vitl14_inout")
parser.add_argument("--ckpt_path", type=str, default="./checkpoints/gazelle_dinov2_vitl14_inout.pt")
parser.add_argument("--batch_size", type=int, default=64)
args = parser.parse_args()

class VideoAttentionTarget(torch.utils.data.Dataset):
    def __init__(self, path, img_transform):
        self.sequences = json.load(open(os.path.join(path, "test_preprocessed.json"), "rb"))
        self.frames = []
        for i in range(len(self.sequences)):
            for j in range(len(self.sequences[i]['frames'])):
                self.frames.append((i, j))
        self.path = path
        self.transform = img_transform

    def __getitem__(self, idx):
        seq_idx, frame_idx = self.frames[idx]
        seq = self.sequences[seq_idx]
        frame = seq['frames'][frame_idx]
        image = self.transform(Image.open(os.path.join(self.path, frame['path'])).convert("RGB"))
        bboxes = [head['bbox_norm'] for head in frame['heads']]
        gazex = [head['gazex_norm'] for head in frame['heads']]
        gazey = [head['gazey_norm'] for head in frame['heads']]
        inout = [head['inout'] for head in frame['heads']]

        return image, bboxes, gazex, gazey, inout

    def __len__(self):
        return len(self.frames)
    
def collate(batch):
    images, bboxes, gazex, gazey, inout = zip(*batch)
    return torch.stack(images), list(bboxes), list(gazex), list(gazey), list(inout)

# VideoAttentionTarget calculates AUC on 64x64 heatmap, defining a rectangular tolerance region of 6*(sigma=3) + 1 (uses 2D Gaussian code but binary thresholds > 0 resulting in rectangle)
# References:
    # https://github.com/ejcgt/attention-target-detection/blob/acd264a3c9e6002b71244dea8c1873e5c5818500/eval_on_videoatttarget.py#L106
    # https://github.com/ejcgt/attention-target-detection/blob/acd264a3c9e6002b71244dea8c1873e5c5818500/utils/imutils.py#L31
def vat_auc(heatmap, gt_gazex, gt_gazey):
    res = 64
    sigma = 3
    assert heatmap.shape[0] == res and heatmap.shape[1] == res
    target_map = np.zeros((res, res))
    gazex = gt_gazex * res
    gazey = gt_gazey * res
    ul = [max(0, int(gazex - 3 * sigma)), max(0, int(gazey - 3 * sigma))]
    br = [min(int(gazex + 3 * sigma + 1), res-1), min(int(gazey + 3 * sigma + 1), res-1)]
    target_map[ul[1]:br[1], ul[0]:br[0]] = 1
    auc = roc_auc_score(target_map.flatten(), heatmap.cpu().flatten())
    return auc

# Reference: https://github.com/ejcgt/attention-target-detection/blob/acd264a3c9e6002b71244dea8c1873e5c5818500/eval_on_videoatttarget.py#L118
def vat_l2(heatmap, gt_gazex, gt_gazey):
    argmax = heatmap.flatten().argmax().item()
    pred_y, pred_x = np.unravel_index(argmax, (64, 64))
    pred_x = pred_x / 64.
    pred_y = pred_y / 64.

    l2 = np.sqrt((pred_x - gt_gazex)**2 + (pred_y - gt_gazey)**2)

    return l2


@torch.no_grad()
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Running on {}".format(device))

    model, transform = get_gazelle_model(args.model_name)
    model.load_gazelle_state_dict(torch.load(args.ckpt_path, weights_only=True))
    model.to(device)
    model.eval()

    dataset = VideoAttentionTarget(args.data_path, transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate)

    aucs = []
    l2s = []
    inout_preds = []
    inout_gts = []

    for _, (images, bboxes, gazex, gazey, inout) in tqdm(enumerate(dataloader), desc="Evaluating", total=len(dataloader)):
        preds = model.forward({"images": images.to(device), "bboxes": bboxes})
        
        # eval each instance (head)
        for i in range(images.shape[0]): # per image
            for j in range(len(bboxes[i])): # per head
                if inout[i][j] == 1: # in frame
                    auc = vat_auc(preds['heatmap'][i][j], gazex[i][j][0], gazey[i][j][0])
                    l2 = vat_l2(preds['heatmap'][i][j], gazex[i][j][0], gazey[i][j][0])
                    aucs.append(auc)
                    l2s.append(l2)
                inout_preds.append(preds['inout'][i][j].item())
                inout_gts.append(inout[i][j])

    
    print("AUC: {}".format(np.array(aucs).mean()))
    print("Avg L2: {}".format(np.array(l2s).mean()))
    print("Inout AP: {}".format(average_precision_score(inout_gts, inout_preds)))

        
if __name__ == "__main__":
    main()