import glob
import numpy as np
from PIL import Image

import torch
import torch.utils.data.distributed
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class CustomLoader(object):
    def __init__(self, args, fldr_path):
        self.testing_samples = CustomLoadPreprocess(args, fldr_path)
        self.data = DataLoader(self.testing_samples, 1,
                               shuffle=False,
                               num_workers=1,
                               pin_memory=False)


class CustomLoadPreprocess(Dataset):
    def __init__(self, args, fldr_path):
        self.fldr_path = fldr_path
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.filenames = glob.glob(self.fldr_path + '/*.png') + glob.glob(self.fldr_path + '/*.jpg')
        self.input_height = args.input_height
        self.input_width = args.input_width

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = self.filenames[idx]
        img = Image.open(img_path).convert("RGB").resize(size=(self.input_width, self.input_height), resample=Image.BILINEAR)
        img = np.array(img).astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)
        img = self.normalize(img)

        img_name = img_path.split('/')[-1]
        img_name = img_name.split('.png')[0] if '.png' in img_name else img_name.split('.jpg')[0]

        sample = {'img': img,
                  'img_name': img_name}

        return sample
