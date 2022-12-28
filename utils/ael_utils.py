import torch
from torch.nn import functional as F
import numpy as np
import random
from skimage.measure import label, regionprops
import torch.nn as nn

def dynamic_copy_paste(images_sup, paste_imgs, labels_sup, paste_labels, query_cat):

    compose_imgs = []
    compose_labels = []
    for idx in range(images_sup.shape[0]):
        paste_label = paste_labels[idx]
        image_sup = images_sup[idx]
        label_sup = labels_sup[idx]
        if torch.sum(paste_label) == 0:
            compose_imgs.append(image_sup.unsqueeze(0))
            compose_labels.append(label_sup.unsqueeze(0))
        else:
            paste_img = paste_imgs[idx]
            alpha = torch.zeros_like(paste_label).int()
            for cat in query_cat:
                alpha = alpha.__or__((paste_label==cat).int())
            alpha = (alpha > 0).int()
            compose_img = (1-alpha)*image_sup + alpha * paste_img
            compose_label = (1-alpha)*label_sup + alpha * paste_label
            compose_imgs.append(compose_img.unsqueeze(0))
            compose_labels.append(compose_label.unsqueeze(0))
    compose_imgs = torch.cat(compose_imgs,dim=0)
    compose_labels = torch.cat(compose_labels,dim=0)
    return compose_imgs, compose_labels


def sample_from_bank(cutmix_bank, conf, smooth=False):
    # cutmix_bank [num_classes, len(dataset)]
    classes = [i for i in range(cutmix_bank.shape[0])]
    if len(classes) > 2:
        conf = (1 - conf).numpy()
        if smooth:
            conf = conf**(1/3)
        conf = np.exp(conf)/np.sum(np.exp(conf))
        class_id = np.random.choice(classes, p=conf)
    else:
        class_id = 1 # consider 0 - background, 1 as object

    sample_bank = torch.nonzero(cutmix_bank[class_id])
    if len(sample_bank)>0:
        sample_id = random.choice(sample_bank)
    else:
        sample_id = random.randint(0, cutmix_bank.shape[1]-1)
    return sample_id, class_id

def generate_cutmix_mask(pred, sample_cat, area_thresh=0.0001, no_pad=False, no_slim=False, num_classes=2):
    h, w = pred.shape[0], pred.shape[1]
    valid_mask = np.zeros((h,w))
    values = np.unique(pred)
    assert len(values) <= num_classes
    if not sample_cat in values:
        rectangles = init_cutmix(h)
    else:
        rectangles = generate_cutmix(pred, sample_cat, area_thresh,no_pad=no_pad, no_slim=no_slim)
    y0, x0, y1, x1 = rectangles
    valid_mask[int(y0):int(y1), int(x0):int(x1)] = 1
    valid_mask = torch.from_numpy(valid_mask).long().cuda()

    return valid_mask

def init_cutmix(crop_size):
    h = crop_size
    w = crop_size
    n_masks = 1
    prop_range = 0.5
    mask_props = np.random.uniform(prop_range, prop_range, size=(n_masks, 1))
    y_props = np.exp(np.random.uniform(low=0.0, high=1.0, size=(n_masks, 1)) * np.log(mask_props))
    x_props = mask_props / y_props
    sizes = np.round(np.stack([y_props, x_props], axis=2) * np.array((h,w))[None, None, :])
    positions = np.round((np.array((h,w))-sizes) * np.random.uniform(low=0.0, high=1.0, size=sizes.shape))
    rectangles = np.append(positions, positions+sizes, axis=2)[0,0]
    return rectangles

def generate_cutmix(pred, cat, area_thresh, no_pad=False, no_slim=False):
    h = pred.shape[0]
    #print('h',h)
    area_all = h**2
    prop = (pred==cat)*1
    pred = label(pred)
    prop = regionprops(pred)
    values = np.unique(pred)[1:]
    random.shuffle(values)

    flag = 0
    for value in values:
        if np.sum(pred == value) > area_thresh*area_all:
            flag=1
            break
    if flag == 1:
        rectangles = prop[value-1].bbox
        #area = prop[value-1].area
        area = (rectangles[2]-rectangles[0])*(rectangles[3]-rectangles[1])
        if area >= 0.5*area_all and not no_slim:
            rectangles = sliming_bbox(rectangles, h)
        elif area < 0.5*area_all and not no_pad:
            rectangles = padding_bbox_new(rectangles, h)
        else:
            pass
    else:
        rectangles = init_cutmix(h)
    return rectangles

def padding_bbox_new(rectangles, size):
    area = 0.5 * (size ** 2)
    y0, x0, y1, x1 = rectangles
    h = y1 - y0
    w = x1 - x0
    new_h = int(size*(np.exp(np.random.uniform(low=0.0, high=1.0, size=(1)) * np.log(0.5))))
    new_w = int(area/new_h)
    delta_h = new_h - h
    delta_w = new_w - w
    y_ratio = y0/(size-y1+1)
    x_ratio = x0/(size-x1+1)
    x1 = min(x1+int(delta_w*(1/(1+x_ratio))), size)
    x0 = max(x0-int(delta_w*(x_ratio/(1+x_ratio))), 0)
    y1 = min(y1+int(delta_h*(1/(1+y_ratio))), size)
    y0 = max(y0-int(delta_h*(y_ratio/(1+y_ratio))), 0)
    return [y0, x0, y1, x1]

def sliming_bbox(rectangles, size):
    area = 0.5 * (size ** 2)
    y0, x0, y1, x1 = rectangles
    h = y1 - y0
    w = x1 - x0
    lower_h = int(area/w)
    if lower_h > h:
        print('wrong')
        new_h = h
    else:
        new_h = random.randint(lower_h, h)
    new_w = int(area/new_h)
    if new_w > w:
        print('wrong')
        new_w = w - 1
    delta_h = h - new_h
    delta_w = w - new_w
    prob = random.random()
    if prob > 0.5:
        y1 = max(random.randint(y1 - delta_h, y1), y0)
        y0 = max(y1 - new_h, y0)
    else:
        y0 = min(random.randint(y0, y0 + delta_h), y1)
        y1 = min(y0 + new_h, y1)
    prob = random.random()
    if prob > 0.5:
        x1 = max(random.randint(x1 - delta_w, x1), x0)
        x0 = max(x1 - new_w, x0)
    else:
        x0 = min(random.randint(x0, x0 + delta_w), x1)
        x1 = min(x0 + new_w, x1)
    return [y0, x0, y1, x1]

def update_cutmix_bank(cutmix_bank, preds_teacher_unsup_1, preds_teacher_unsup_2, img_id, sample_id, area_thresh=0.0001):
    # cutmix_bank [num_classes, len(dataset)]
    # preds_teacher_unsup [2,num_classes,h,w]
    area_all = preds_teacher_unsup_1.shape[-1]**2
    pred1 = preds_teacher_unsup_1.max(0)[1]   # (h,w)
    pred2 = preds_teacher_unsup_2.max(0)[1]   # (h,w)
    values1 = torch.unique(pred1)
    values2 = torch.unique(pred2)
    # for img1
    for idx in range(cutmix_bank.shape[0]):
        if idx not in values1:
            cutmix_bank[idx][img_id] = 0
        elif torch.sum(pred1==idx) < area_thresh*area_all:
            cutmix_bank[idx][img_id] = 0
        else:
            cutmix_bank[idx][img_id] = 1
    # for img2
    for idx in range(cutmix_bank.shape[0]):
        if idx not in values2:
            cutmix_bank[idx][sample_id] = 0
        elif torch.sum(pred2==idx) < area_thresh*area_all:
            cutmix_bank[idx][sample_id] = 0
        else:
            cutmix_bank[idx][sample_id] = 1

    return cutmix_bank

def cal_category_confidence(preds_student_sup, preds_student_unsup, gt, num_classes):
    category_confidence = torch.zeros(num_classes).type(torch.float32)
    preds_student_sup = F.softmax(preds_student_sup, dim=1)
    preds_student_unsup = F.softmax(preds_student_unsup, dim=1)
    for ind in range(num_classes):
        cat_mask_sup_gt = (gt == ind)
        if torch.sum(cat_mask_sup_gt) == 0:
            value = 0
        else:
            conf_map_sup = preds_student_sup[:, ind, :, :]
            value = torch.sum(conf_map_sup*cat_mask_sup_gt)/(torch.sum(cat_mask_sup_gt)+1e-12)
        category_confidence[ind] = value

    return category_confidence

def get_criterion(config, cons=False):
    cfg_criterion = config.criterion
    aux_weight = config.aux_loss['loss_weight']
    ignore_index = 255
    if cfg_criterion['type'] == 'ohem':
        criterion = CriterionOhem(aux_weight, ignore_index=ignore_index,
                                  **cfg_criterion['kwargs'])
    else:
        criterion = Criterion(aux_weight, ignore_index=ignore_index, **cfg_criterion['kwargs'])
    if cons:
        gamma = config.criterion['cons']['gamma']
        sample = config.criterion['cons'].get('sample', False)
        gamma2 = config.criterion['cons'].get('gamma2',1)
        criterion = Criterion_cons(gamma, sample=sample, gamma2=gamma2,
                                   ignore_index=ignore_index, **cfg_criterion['kwargs'])
    return criterion

class Criterion(nn.Module):
    def __init__(self, aux_weight, ignore_index=255, thresh=0.7, min_kept=100000,use_weight=False):
        super(Criterion, self).__init__()
        self._aux_weight = aux_weight
        self._ignore_index = ignore_index
        self.use_weight = use_weight
        if not use_weight:
            self._criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
        else:
            weights = torch.FloatTensor(
                [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0]).cuda()
            self._criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
            self._criterion1 = nn.CrossEntropyLoss(ignore_index=ignore_index, weight=weights)

    def forward(self, preds, target, aux):
        h, w = target.size(1), target.size(2)
        if self._aux_weight > 0:  # require aux loss
            main_pred = preds
            aux_pred = aux
            main_h, main_w = main_pred.size(2), main_pred.size(3)
            aux_h, aux_w = aux_pred.size(2), aux_pred.size(3)
            assert main_h == aux_h and main_w == aux_w and main_h == h and main_w == w
            if self.use_weight:
                loss1 = self._criterion(main_pred, target) + self._criterion1(main_pred, target)
            else:
                loss1 = self._criterion(main_pred, target)
            loss2 = self._criterion(aux_pred, target)
            loss = loss1 + self._aux_weight * loss2
        else:
            pred_h, pred_w = preds.size(2), preds.size(3)
            assert pred_h == h and pred_w == w
            loss = self._criterion(preds, target)
        return loss

class Criterion_cons(nn.Module):
    def __init__(self, gamma, sample=False, gamma2=1,
                ignore_index=255, thresh=0.7, min_kept=100000, use_weight=False):
        super(Criterion_cons, self).__init__()
        self.gamma = gamma
        self.gamma2 = float(gamma2)
        self._ignore_index = ignore_index
        self.sample = sample
        self._criterion = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')

    def forward(self, preds, conf, gt, dcp_criterion=None):
        #conf = F.softmax(conf)
        ce_loss = self._criterion(preds,gt)
        conf = torch.pow(conf,self.gamma)

        if self.sample:
            dcp_criterion = 1 - dcp_criterion
            dcp_criterion = dcp_criterion / (torch.max(dcp_criterion)+1e-12)
            dcp_criterion = torch.pow(dcp_criterion, self.gamma2)
            pred_map = preds.max(1)[1].float()

            sample_map = torch.zeros_like(pred_map).float()
            h, w = pred_map.shape[-2], pred_map.shape[-1]

            for idx in range(len(dcp_criterion)):
                prob = 1 - dcp_criterion[idx]
                rand_map = torch.rand(h, w).cuda()*(pred_map == idx)
                rand_map = (rand_map>prob)*1.0
                sample_map += rand_map
            conf = conf * (sample_map)
        conf = conf/(conf.sum() + 1e-12)

        loss = conf * ce_loss
        return loss.sum()

class CriterionOhem(nn.Module):
    def __init__(self, aux_weight, thresh=0.7, min_kept=100000,  ignore_index=255, use_weight=False):
        super(CriterionOhem, self).__init__()
        self._aux_weight = aux_weight
        self._criterion1 = OhemCrossEntropy2dTensor(ignore_index, thresh, min_kept,use_weight)
        self._criterion2 = OhemCrossEntropy2dTensor(ignore_index, thresh, min_kept)

    def forward(self, preds, target, aux):
        h, w = target.size(1), target.size(2)
        if self._aux_weight > 0:  # require aux loss
            main_pred = preds
            aux_pred = aux
            main_h, main_w = main_pred.size(2), main_pred.size(3)
            aux_h, aux_w = aux_pred.size(2), aux_pred.size(3)

            loss1 = self._criterion1(main_pred, target)
            loss2 = self._criterion2(aux_pred, target)
            loss = loss1 + self._aux_weight * loss2
        else:
            pred_h, pred_w = preds.size(2), preds.size(3)
            assert pred_h == h and pred_w == w
            loss = self._criterion1(preds, target)
        return loss

class OhemCrossEntropy2dTensor(nn.Module):
    """
        Ohem Cross Entropy Tensor Version
    """
    def __init__(self, ignore_index=255, thresh=0.7, min_kept=256,
                 use_weight=False, reduce=False):
        super(OhemCrossEntropy2dTensor, self).__init__()
        self.ignore_index = ignore_index
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        if use_weight:
            weight = torch.FloatTensor(
                [0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754, 1.0489,
                 0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037, 1.0865, 1.0955,
                 1.0865, 1.1529, 1.0507]).cuda()

            self.criterion = torch.nn.CrossEntropyLoss(reduction="elementwise_mean",
                                                       weight=weight,
                                                       ignore_index=ignore_index)
        elif reduce:
            self.criterion = torch.nn.CrossEntropyLoss(reduction="none",
                                                       ignore_index=ignore_index)
        else:
            self.criterion = torch.nn.CrossEntropyLoss(reduction="elementwise_mean",
                                                       ignore_index=ignore_index)

    def forward(self, pred, target):
        b, c, h, w = pred.size()
        target = target.view(-1)
        valid_mask = target.ne(self.ignore_index)
        target = target * valid_mask.long()
        num_valid = valid_mask.sum()

        prob = F.softmax(pred, dim=1)
        prob = (prob.transpose(0, 1)).reshape(c, -1)

        if self.min_kept > num_valid:
            pass
        elif num_valid > 0:
            prob = prob.masked_fill_(~valid_mask, 1)
            mask_prob = prob[
                target, torch.arange(len(target), dtype=torch.long)]
            threshold = self.thresh
            if self.min_kept > 0:
                _, index = mask_prob.sort()
                threshold_index = index[min(len(index), self.min_kept) - 1]
                if mask_prob[threshold_index] > self.thresh:
                    threshold = mask_prob[threshold_index]
                kept_mask = mask_prob.le(threshold)
                target = target * kept_mask.long()
                valid_mask = valid_mask * kept_mask

        target = target.masked_fill_(~valid_mask, self.ignore_index)
        target = target.view(b, h, w)

        return self.criterion(pred, target)