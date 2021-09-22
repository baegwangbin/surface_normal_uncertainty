import os
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F


# convert arg line to args
def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield str(arg)


# unnormalize image
__imagenet_stats = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
def unnormalize(img_in):
    img_out = np.zeros(img_in.shape)
    for ich in range(3):
        img_out[:, :, ich] = img_in[:, :, ich] * __imagenet_stats['std'][ich]
        img_out[:, :, ich] += __imagenet_stats['mean'][ich]
    img_out = (img_out * 255).astype(np.uint8)
    return img_out


# kappa to exp error (only applicable to AngMF distribution)
def kappa_to_alpha(pred_kappa):
    alpha = ((2 * pred_kappa) / ((pred_kappa ** 2.0) + 1)) \
            + ((np.exp(- pred_kappa * np.pi) * np.pi) / (1 + np.exp(- pred_kappa * np.pi)))
    alpha = np.degrees(alpha)
    return alpha


# concatenate images
def concat_image(image_path_list, concat_image_path):
    imgs = [Image.open(i).convert("RGB").resize((640, 480), resample=Image.BILINEAR) for i in image_path_list]
    imgs_list = []
    for i in range(len(imgs)):
        img = imgs[i]
        imgs_list.append(np.asarray(img))

        H, W, _ = np.asarray(img).shape
        imgs_list.append(255 * np.ones((H, 20, 3)).astype('uint8'))

    imgs_comb = np.hstack(imgs_list[:-1])
    imgs_comb = Image.fromarray(imgs_comb)
    imgs_comb.save(concat_image_path)



# load model
def load_checkpoint(fpath, model):
    ckpt = torch.load(fpath, map_location='cpu')['model']

    load_dict = {}
    for k, v in ckpt.items():
        if k.startswith('module.'):
            k_ = k.replace('module.', '')
            load_dict[k_] = v
        else:
            load_dict[k] = v

    model.load_state_dict(load_dict)
    return model
