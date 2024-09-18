import logging
import numpy as np
import torch
import json
import cv2
from PIL import Image
from functools import lru_cache
from functools import partial
from itertools import repeat
from multiprocessing import Pool
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm



def load_image(filename):
    ext = splitext(filename)[1]
    if ext == '.npy':
        return Image.fromarray(np.load(filename))
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy())
    else:
        return Image.open(filename)


def unique_mask_values(idx, mask_dir, mask_suffix):
    mask_file = list(mask_dir.glob(idx + mask_suffix + '.*'))[0]
    mask = np.asarray(load_image(mask_file))
    if mask.ndim == 2:
        return np.unique(mask)
    elif mask.ndim == 3:
        mask = mask.reshape(-1, mask.shape[-1])
        return np.unique(mask, axis=0)
    else:
        raise ValueError(f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}')


class BasicDataset(Dataset):
    def __init__(self, images_dir: str, mask_dir: str, scale: float = 1.0, mask_suffix: str = ''):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if isfile(join(images_dir, file)) and not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')

        logging.info(f'Creating dataset with {len(self.ids)} examples')
        logging.info('Scanning mask files to determine unique values')
        with Pool() as p:
            unique = list(tqdm(
                p.imap(partial(unique_mask_values, mask_dir=self.mask_dir, mask_suffix=self.mask_suffix), self.ids),
                total=len(self.ids)
            ))

        self.mask_values = list(sorted(np.unique(np.concatenate(unique), axis=0).tolist()))
        logging.info(f'Unique mask values: {self.mask_values}')

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(mask_values, pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img = np.asarray(pil_img)

        if is_mask:
            mask = np.zeros((newH, newW), dtype=np.int64)
            for i, v in enumerate(mask_values):
                if img.ndim == 2:
                    mask[img == v] = i
                else:
                    mask[(img == v).all(-1)] = i

            return mask

        else:
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            else:
                img = img.transpose((2, 0, 1))

            if (img > 1).any():
                img = img / 255.0

            return img

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.mask_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        mask = load_image(mask_file[0])
        img = load_image(img_file[0])
        # print("type of img: {}".format(type(img)))
        # print("type of mask: {}".format(type(mask)))
        # print("img_file[0]: {}".format(img_file[0]))
        # print("mask_file[0]: {}".format(mask_file[0]))
        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(self.mask_values, img, self.scale, is_mask=False)
        mask = self.preprocess(self.mask_values, mask, self.scale, is_mask=True)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, images_dir, mask_dir, scale=1):
        super().__init__(images_dir, mask_dir, scale, mask_suffix='_mask')


class JsonDataset(Dataset):
    def __init__(self, image_dir_path, json_file_path, scale: float = 1.0):
        self.img_dir_path = Path(image_dir_path)
        self.json_file_path = Path(json_file_path)
        self.scale = scale
        with open(json_file_path, 'r') as json_file:
            self.json_data = json.load(json_file)
            self.idx = len(self.json_data)
            # print("json len: {}".format(len(json_data)))
            img_amount = [file for file in listdir(self.img_dir_path) if isfile(join(self.img_dir_path, file)) and file.endswith('.jpg')]
            # print("img amount: {}".format(len(img_amount)))
          



    def __len__(self):
        return self.idx
    
    @staticmethod
    def preprocess(mask_values, pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img = np.asarray(pil_img)

        if is_mask:
            mask = np.zeros((newH, newW), dtype=np.int64)
            for i, v in enumerate(mask_values):
                if img.ndim == 2:
                    mask[img == v] = i
                else:
                    mask[(img == v).all(-1)] = i

            return mask

        else:
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            else:
                img = img.transpose((2, 0, 1))

            if (img > 1).any():
                img = img / 255.0

            return img

    def __getitem__(self, index):
        # print(type(index))
        # print("index: {}".format(index))

        # self.json_data[index]
        image_name = self.json_data[index]['image']
        image_path = self.img_dir_path.joinpath(image_name)
        self.image = load_image(image_path)
        self.mask = np.zeros(self.image.size, dtype=np.uint8)
        # Json datas
        label = self.json_data[index]['annotations'][0]['label']
        x = self.json_data[index]['annotations'][0]['coordinates']['x']
        y = self.json_data[index]['annotations'][0]['coordinates']['y']
        width = self.json_data[index]['annotations'][0]['coordinates']['width']
        height = self.json_data[index]['annotations'][0]['coordinates']['height']

        centre = (int(x), int(y))
        radius = int(np.sqrt((width / 2) ** 2 + (height / 2) ** 2))
        cv2.circle(self.mask, centre, radius, (255), thickness=-1)  # thickness=-1 代表填滿圓形
        # self.image = Image.fromarray(self.image)
        self.mask = Image.fromarray(self.mask)

        
        # for item in json_data:
        #     image_name = item['image']
           
        # # 創建一個空白的mask，尺寸與影像相同
            

        #     # 遍歷每個標註項目
        #     for annotation in item['annotations']:
        #         label = annotation['label']
        #         x = annotation['coordinates']['x']
        #         y = annotation['coordinates']['y']
        #         width = annotation['coordinates']['width']
        #         height = annotation['coordinates']['height']

        #         # 繪製圓形遮罩在mask上
        #         centre = (int(x), int(y))
        #         radius = int(np.sqrt((width / 2) ** 2 + (height / 2) ** 2))
        #         cv2.circle(self.mask, centre, radius, (255), thickness=-1)  # thickness=-1 代表填滿圓形
        #         self.image = Image.fromarray(self.image)
        #         self.mask = Image.fromarray(self.mask)

        # # Image.fromarray(np.uint8(num_img))

        # # return super().__getitem__(index)
        # # torch.as_tensor(img.copy()).float().contiguous(),
        self.mask = self.preprocess(self.mask, self.mask, self.scale, is_mask=False)
        self.image = self.preprocess(self.mask, self.image, self.scale, is_mask=False)

        return {
            'image': torch.as_tensor(self.image.copy()).float().contiguous(),
            'mask' : torch.as_tensor(self.mask.copy()).float().contiguous()
        }