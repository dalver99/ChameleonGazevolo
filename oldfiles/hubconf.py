dependencies = ['torch', 'timm']

import torch
from gazelle.model import get_gazelle_model

def gazelle_dinov2_vitb14():
    model, transform = get_gazelle_model('gazelle_dinov2_vitb14')
    ckpt_path = "https://github.com/fkryan/gazelle/releases/download/v1.0.0/gazelle_dinov2_vitb14_hub.pt"
    model.load_gazelle_state_dict(torch.hub.load_state_dict_from_url(ckpt_path))
    return model, transform

def gazelle_dinov2_vitl14():
    model, transform = get_gazelle_model('gazelle_dinov2_vitl14')
    ckpt_path = "https://github.com/fkryan/gazelle/releases/download/v1.0.0/gazelle_dinov2_vitl14.pt"
    model.load_gazelle_state_dict(torch.hub.load_state_dict_from_url(ckpt_path))
    return model, transform

def gazelle_dinov2_vitb14_inout():
    model, transform = get_gazelle_model('gazelle_dinov2_vitb14_inout')
    ckpt_path = "https://github.com/fkryan/gazelle/releases/download/v1.0.0/gazelle_dinov2_vitb14_inout.pt"
    model.load_gazelle_state_dict(torch.hub.load_state_dict_from_url(ckpt_path))
    return model, transform

def gazelle_dinov2_vitl14_inout():
    model, transform = get_gazelle_model('gazelle_dinov2_vitl14_inout')
    ckpt_path = "https://github.com/fkryan/gazelle/releases/download/v1.0.0/gazelle_dinov2_vitl14_inout.pt"
    model.load_gazelle_state_dict(torch.hub.load_state_dict_from_url(ckpt_path))
    return model, transform
