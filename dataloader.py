import os
import cv2
import torch
import numpy as np
from torch.utils import data
import random
from utils.img_utils import generate_random_crop_pos, random_crop_pad_to_shape
from BaseDataset import BaseDataset


def random_mirror(img, gt=None):
    if random.random() >= 0.5:
        img = cv2.flip(img, 1)
        if gt is not None:
            gt = cv2.flip(gt, 1)

    return img, gt

def normalize(img, mean, std):
    img = img.astype(np.float32) / 255.0
    img = img - mean
    img = img / std

    return img


def random_scale(img, gt=None, scales=None):
    scale = random.choice(scales)
    sh = int(img.shape[0] * scale)
    sw = int(img.shape[1] * scale)
    img = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_LINEAR)
    if gt is not None:
        gt = cv2.resize(gt, (sw, sh), interpolation=cv2.INTER_NEAREST)

    return img, gt, scale

class TrainPre(object):
    def __init__(self, img_mean, img_std, config):
        self.img_mean = img_mean
        self.img_std = img_std
        self.config = config

    def __call__(self, img, gt=None):
        # gt = gt - 1     # label 0 is invalid, this operation transfers label 0 to label 255
        img, gt = random_mirror(img, gt)
        if self.config.train_scale_array is not None:
            img, gt, scale = random_scale(img, gt, self.config.train_scale_array)

        img = normalize(img, self.img_mean, self.img_std)

        crop_size = (self.config.image_height, self.config.image_width)
        crop_pos = generate_random_crop_pos(img.shape[:2], crop_size)

        p_img, _ = random_crop_pad_to_shape(img, crop_pos, crop_size, 0)
        if gt is not None:
            p_gt, _ = random_crop_pad_to_shape(gt, crop_pos, crop_size, 255)
        else:
            p_gt = None

        p_img = p_img.transpose(2, 0, 1)

        extra_dict = {}

        return p_img, p_gt, extra_dict

class ValPre(object):
    def __call__(self, img, gt):
        # gt = gt - 1
        extra_dict = {}
        return img, gt, extra_dict

def get_train_loader_uni(engine, dataset, train_source, config,
                        unsupervised=False, collate_fn=None):

    data_setting = {'img_root': config.img_root_folder,
                    'gt_root': config.gt_root_folder,
                    'train_source': train_source,
                    'eval_source': config.eval_source}
    train_preprocess = TrainPre(config.image_mean, config.image_std, config)

    if 'selftrain' in train_source:
        train_dataset = dataset(data_setting, "train", train_preprocess, config.tot_samples, unsupervised=unsupervised,
        dataset_name=config.name)
    else:
        train_dataset = dataset(data_setting, "train", train_preprocess,
                                config.max_samples, unsupervised=unsupervised, dataset_name=config.name)

    train_sampler = None
    is_shuffle = True
    batch_size = config.batch_size

    if engine.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset)
        batch_size = config.batch_size // engine.world_size
        is_shuffle = False

    train_loader = data.DataLoader(train_dataset,
                                   batch_size=batch_size,
                                   num_workers=config.num_workers,
                                   drop_last=True,
                                   shuffle=is_shuffle,
                                   pin_memory=True,
                                   sampler=train_sampler,
                                   collate_fn=collate_fn)

    return train_loader, train_sampler


class Dataset_uni(BaseDataset):
    def __init__(self, setting, split_name, preprocess=None,
                 file_length=None, training=True, unsupervised=False, pseudo_label=False, dataset_name="VOC"):
        self.istraining = training
        self.unsupervised = unsupervised
        super().__init__(setting, split_name, preprocess, file_length)
        self._split_name = split_name
        self._img_path = setting['img_root']
        self._gt_path = setting['gt_root']
        self._train_source = setting['train_source']
        self._eval_source = setting['eval_source']
        self._file_names = self._get_file_names(split_name)
        self._file_length = file_length
        self.preprocess = preprocess
        self.pseudo_label = pseudo_label
        self.dataset_name = dataset_name

    def __getitem__(self, index):
        if self._file_length is not None:
            names = self._construct_new_file_names(self._file_length)[index]
        else:
            names = self._file_names[index]

        sp = names.strip().split('\t')
        if not self.istraining:
            img_path = os.path.join(self._img_path, 'images', 'val', sp[0])
            gt_path = os.path.join(self._gt_path, 'labels', 'val', sp[0])
            if self.dataset_name == 'city_4':
                gt_path = gt_path[:-4] + '_gtFine' + ".png"
            elif not os.path.isfile(gt_path):
                gt_path = gt_path[:-4] + ".png"
        else:
            img_path = os.path.join(self._img_path, 'images', 'train', sp[0])
            gt_path = os.path.join(self._gt_path, 'labels', 'train', sp[0])
            if self.dataset_name == 'city_4':
                gt_path = gt_path[:-4] + '_gtFine' + ".png"

        if not self.unsupervised:
            img, gt = self._fetch_data(img_path, gt_path)
        else:
            img, gt = self._fetch_data(img_path, None)

        if gt is not None:
            gt = np.uint8(gt)

        img = img[:, :, ::-1]
        if self.preprocess is not None:
            img, gt, extra_dict = self.preprocess(img, gt)

        if self._split_name in ['train', 'trainval', 'train_aug', 'trainval_aug']:
            img = torch.from_numpy(np.ascontiguousarray(img)).float()
            if gt is not None:
                gt = torch.from_numpy(np.ascontiguousarray(gt)).long()

            if self.preprocess is not None and extra_dict is not None:
                for k, v in extra_dict.items():
                    extra_dict[k] = torch.from_numpy(np.ascontiguousarray(v))
                    if 'label' in k:
                        extra_dict[k] = extra_dict[k].long()
                    if 'img' in k:
                        extra_dict[k] = extra_dict[k].float()

        output_dict = dict(data=img, fn=str(names),
                           n=len(self._file_names), id=index)
        if gt is not None:
            extra_dict['label'] = gt

        if self.preprocess is not None and extra_dict is not None:
            output_dict.update(**extra_dict)

        return output_dict

    def _fetch_data(self, img_path, gt_path=None, dtype=None):
        if img_path.endswith(".jpg") or img_path.endswith(".png"):
            img = self._open_image(img_path)
        else:
            img = self._open_image(img_path + ".jpg")

        if gt_path is not None:
            if gt_path.endswith(".jpg") or gt_path.endswith(".png"):
                gt = self._open_image(gt_path, cv2.IMREAD_GRAYSCALE, dtype=dtype)
                if self.dataset_name == 'fish' or self.dataset_name == 'disk':
                    gt = self.preprocess_gt(gt)
            else:
                gt = self._open_image(gt_path + ".png", cv2.IMREAD_GRAYSCALE, dtype=dtype)
                if self.dataset_name == 'fish' or self.dataset_name == 'disk':
                    gt = self.preprocess_gt(gt)
            return img, gt
        return img, None

    @staticmethod
    def preprocess_gt(gt):
        gt[gt==255] = 1
        return gt

    def get_class_colors(self, *args):
        def uint82bin(n, count=8):
            """returns the binary of integer n, count refers to amount of bits"""
            return ''.join([str((n >> y) & 1) for y in range(count - 1, -1, -1)])

        N = 4 if self.dataset_name == 'city_4' else 2
        cmap = np.zeros((N, 3), dtype=np.uint8)
        for i in range(N):
            r, g, b = 0, 0, 0
            id = i
            for j in range(7):
                str_id = uint82bin(id)
                r = r ^ (np.uint8(str_id[-1]) << (7 - j))
                g = g ^ (np.uint8(str_id[-2]) << (7 - j))
                b = b ^ (np.uint8(str_id[-3]) << (7 - j))
                id = id >> 3
            cmap[i, 0] = r
            cmap[i, 1] = g
            cmap[i, 2] = b
        class_colors = cmap.tolist()
        return class_colors

    def get_class_names(self, *args):
        if self.dataset_name == 'city_4':
            return ['fence', 'vegetation', 'car', 'truck']
        else:
            return ['background', 'object']

    @classmethod
    def transform_label(cls, pred, name):
        label = np.zeros(pred.shape)
        ids = np.unique(pred)
        for id in ids:
            label[np.where(pred == id)] = cls.trans_labels[id]

        new_name = (name.split('.')[0]).split('_')[:-1]
        new_name = '_'.join(new_name) + '.png'

        return label, new_name

    def _get_file_names(self, split_name, train_extra=False):
        # assert split_name in ['train', 'val']
        source = self._train_source
        if not self.istraining:
            file_names = [it for it in os.listdir(self._eval_source + os.path.sep + 'images' + os.path.sep + 'val')
                            if it.endswith(".jpg") or it.endswith(".png")]
            return file_names

        file_names = []
        with open(source) as f:
            files = f.readlines()

        for item in files:
            name = item.strip()
            file_names.append(name)
        return file_names