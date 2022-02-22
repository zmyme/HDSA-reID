import torch
import torch.utils.data
import os
import cv2
import numpy as np
import json
import math
from torchvision.transforms import ColorJitter
import torchvision.transforms.functional as F
import cv2


class HalfCrop(object):
    """Half Crop Augmentation
    random crop bottom half of a pedestrian image (i.e. waist to foot)
    perform well for occluded reid
    Args:
        prob(float): probability to perform half crop
        keep_range(list): in height dimension, keep range
    """

    def __init__(self, prob=0.5,  keep_range=(0.50, 1.5)):
        self.prob = prob
        self.keep_range = keep_range

    def __call__(self, img):
        '''
        Args:
            img(np.array): image
        '''
        do_aug = torch.rand(1).item() < self.prob
        if do_aug:
            ratio = torch.rand(1).item() * (self.keep_range[1] - self.keep_range[0]) + self.keep_range[0]
            _, h, w = img.shape
            tw = w
            th = int(h * ratio)
            img = F.crop(img, 0, 0, th, tw)
            img = F.resize(img, [h, w], F.InterpolationMode.BILINEAR)
            return img
        else:
            return img

class RandomErasing():
    """
    the module must be used after image normalization, it expects img mean = 0, and std = 1.0
    """
    def __init__(self, prob=0.5, sl=0.02, sh=0.4, rl=0.3):
        self.prob = prob
        self.sl = sl # size low: minimum proportion of erased area
        self.sh = sh # size high: maximum proportion of erased area
        self.rl = rl # ratio low: the minmum of the ration of width and height, rl=min(w/h, h/w)
    def __call__(self, img):
        # the img is expected of shape c * h * w
        _, img_h, img_w = img.shape
        img_sz = img_h * img_w
        # control prob:
        if torch.rand(1).item() > self.prob:
            return img

        patience = 100
        for _ in range(patience):
            # generate erased size (sl, sh)
            esz = torch.rand(1).item() * (self.sh - self.sl) + self.sl # esz = (h * w)/img_sz
            aspect_ratio = torch.rand(1).item() * (1/self.rl - self.rl) + self.rl # aspect_ratio = h/w
            h = math.sqrt(esz * img_sz * aspect_ratio)
            w = h / aspect_ratio
            h, w = int(h), int(w)

            # find the area to erase.
            max_x = img_w - w
            max_y = img_h - h
            if max_x <= 1 or max_y <= 1:
                continue
            x = torch.randint(1, max_x, (1,)).item()
            y = torch.randint(1, max_y, (1,)).item()

            # erase.
            num_channel = img.shape[0]
            randpatch = torch.rand(num_channel, h, w)
            randpatch = randpatch - 0.5
            randpatch = randpatch / randpatch.std()
            img[:, y:y+h, x:x+w] = randpatch
            return img

        return img

class RandomCrop():
    def __init__(self, min_x, min_y, max_x, max_y):
        self.min_x = min_x
        self.min_y = min_y
        self.max_x = max_x
        self.max_y = max_y

    def __call__(self, img):
        # image should be of shape h * w * c
        h, w, _ = img.shape
        offset_x = self.min_x + torch.rand(1).item() * (self.max_x-self.min_x)
        offset_y = self.min_y + torch.rand(1).item() * (self.max_y-self.min_y)
        x_flag = 1 if torch.rand(1).item() > 0.5 else -1
        y_flag = 1 if torch.rand(1).item() > 0.5 else -1
        offset_x *= x_flag
        offset_y *= y_flag
        y1, x1, y2, x2 = [offset_y, offset_x, h + offset_y, w + offset_x]
        def clamp(value, min_value, max_value):
            value = min_value if value < min_value else value
            value = max_value if value > max_value else value
            return value
        x1, x2 = [int(clamp(v, 0, w)) for v in [x1, x2]]
        y1, y2 = [int(clamp(v, 0, h)) for v in [y1, y2]]
        # print('(x1, x2, y1, y2) = ({0}, {1}, {2}, {3}'.format(x1, x2, y1, y2))
        img = img[y1:y2, x1:x2, :]
        # print('img.shape =', img.shape)
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
        return img

 # img shape: c * h * w
def normalize_image(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    _, h, w = img.shape
    mean = np.broadcast_to(mean, (h, w, 3)).transpose(2, 0, 1)
    std = np.broadcast_to(std, (h, w, 3)).transpose(2, 0, 1)
    img = (img - mean)/std
    return img

def inverse_normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    _, h, w = img.shape # 3 * h * w
    mean = np.broadcast_to(mean, (h, w, 3)).transpose(2, 0, 1)
    std = np.broadcast_to(std, (h, w, 3)).transpose(2, 0, 1)
    img = img * std + mean
    return img

class ReIDDataSet(torch.utils.data.Dataset):
    def __init__(self,
        dataset_root='.',
        img_root = "original",
        list_path='list',
        img_size=(128, 384),
        random_filp=False,
        random_erase=False,
        random_crop=False,
        color_jitter=False,
        half_crop = False
    ):
        self.dataset_root = dataset_root
        self.img_root = os.path.join(self.dataset_root, img_root)
        self.list_path = os.path.join(self.dataset_root, list_path)
        self.lists = None
        self.img_size = tuple(img_size)
        # for data augmentation.
        self.random_filp = random_filp
        self.random_erase = random_erase
        self.random_crop = random_crop
        self.color_jitter = color_jitter
        self.half_crop = half_crop
        self.re = RandomErasing(prob=0.5)
        self.rc = RandomCrop(5, 10, 20, 40)
        self.cj = ColorJitter(brightness=0.25, contrast=0.15, saturation=0.25, hue=0)
        self.hc = HalfCrop(prob=0.5, keep_range=[0.5, 1.5])
        # necessary_outputs
        print('Data augmentation method:')
        print('    random_filp:', self.random_filp)
        print('    random_erase:', self.random_erase)
        print('    random_crop:', self.random_crop)
        print('    color_jitter:', self.color_jitter)
        print('    half_crop:', self.half_crop)
        self.load(self.list_path)

    def load(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            print('loading', path)
            self.lists = json.loads(f.read())
            print('num_samples:', len(self.lists))

    def __len__(self):
        return len(self.lists)

    def __getitem__(self, index):
        relative_path, people_id, camera_id = self.lists[index]
        path = os.path.join(self.img_root,relative_path)
        img = self.load_image(path)
        img = self.process_image(img)
        img = img.astype(np.float32)
        return img, people_id, camera_id, relative_path

    def load_image(self, path):
        # load image as size self.size
        image = cv2.imread(path)
        image = cv2.resize(image, self.img_size, interpolation=cv2.INTER_CUBIC)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.random_crop:
            image = self.rc(image)
        # transfer image size from h*w*c to c*h*w
        image = image.transpose(2, 0, 1)
        # convert pixel from 0~255 to 0~1
        image = image / 255.0
        image = image.astype(np.float32)
        image = torch.from_numpy(image)
        if self.color_jitter:
            image = self.cj(image)
        if self.half_crop:
            image = self.hc(image)
        image = image.numpy()
        image = normalize_image(image)
        return image

    def process_image(self, img):
        if self.random_filp:
            ratio = torch.rand(1).item()
            if ratio > 0.5:
                img = np.flip(img, 2).copy()
        if self.random_erase:
            img = self.re(img)
        return img

if __name__ == '__main__':
    from ToolBox.netimshow import imshow, waitKey, configure
    configure('115.156.214.96:12345', token="zmyimshow")

    def print_stat(name, value):
        min_value = value.min()
        max_value = value.max()
        mean_value = value.mean()
        abs_mean = np.abs(value).mean()
        std_value = value.std()
        print('[{0}] min: {1} | max: {2} | mean: {3} | std: {4} | absmean: {5}'.format(name, min_value, max_value, mean_value, std_value, abs_mean))
    from resources import get_dataloader
    from config import conf
    dataloader = get_dataloader(conf, names='query')
    dataset = dataloader.dataset
    for img, people_id, camera_id, path in dataset:
        print_stat('img', img)
        img = inverse_normalize(img)
        img = img.transpose(1, 2, 0)
        img = (img * 255).astype(np.uint8)
        print('==================================')
        print('img.shape =', img.shape)
        print('people_id:', people_id)
        print('camera_id:', camera_id)
        print('path:', path)
        imshow("sample", img, waitKey=0)