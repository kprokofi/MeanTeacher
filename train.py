from __future__ import division
import os.path as osp
import os
import sys
import time
import argparse
import math
from tqdm import tqdm
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import numpy as np

from configs import get_config_voc, get_config_city, get_config_fish, get_config_water, get_config_disk, get_config_kvasir, get_config_city_4, get_config_voc_person
from dataloader import get_train_loader_uni
from dataloader_city import get_train_loader_city
from dataloader_voc import get_train_loader_voc
from dataloader_city import CityScape
from dataloader_voc import VOC
from dataloader import Dataset_uni, ValPre
from network import Network
from utils.init_func import init_weight, group_weight
from utils.contrastive_loss import compute_contra_memobank_loss
from engine.lr_policy import WarmUpPolyLR
from utils.pyt_utils import parse_devices
from utils.ael_utils import dynamic_copy_paste, sample_from_bank, generate_cutmix_mask, update_cutmix_bank, cal_category_confidence, get_criterion
from engine.engine import Engine
from seg_opr.loss_opr import SigmoidFocalLoss, ProbOhemCrossEntropy2d
from eval import SegEvaluator
# from seg_opr.sync_bn import DataParallelModel, Reduce, BatchNorm2d
import mask_gen
from custom_collate import SegCollate
from tensorboardX import SummaryWriter

from apex.parallel import DistributedDataParallel, SyncBatchNorm

try:
    from azureml.core import Run
    azure = True
    run = Run.get_context()
except:
    azure = False

parser = argparse.ArgumentParser()
parser.add_argument('--dev', default='1', type=str)
parser.add_argument('--dataset', default='VOC', type=str)

if os.getenv('debug') is not None:
    is_debug = os.environ['debug']
else:
    is_debug = False

def set_random_seed(seed, deterministic=False, use_rank_shift=False, rank=0):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
        rank_shift (bool): Whether to add rank number to the random seed to
            have different random seed in different threads. Default: False.
    """
    if use_rank_shift:
        seed += rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def prepare_data(engine, dataset, config, collate_fn):
    train_loader_0, train_sampler = get_train_loader(engine, dataset, train_source=config.train_source, \
                                                   unsupervised=False, collate_fn=collate_fn, config=config)
    train_loader_1, train_sampler = get_train_loader(engine, dataset, train_source=config.train_source, \
                                                   unsupervised=False, collate_fn=collate_fn, config=config)
    unsupervised_train_loader_0, unsupervised_train_sampler_0 = get_train_loader(engine, dataset, \
                train_source=config.unsup_source, unsupervised=True, collate_fn=mask_collate_fn, config=config)
    unsupervised_train_loader_1, unsupervised_train_sampler_1 = get_train_loader(engine, dataset, \
                train_source=config.unsup_source, unsupervised=True, collate_fn=collate_fn, config=config)

    return (train_loader_0, train_loader_1, train_sampler, unsupervised_train_loader_0,
            unsupervised_train_sampler_0, unsupervised_train_loader_1,
            unsupervised_train_sampler_1)


def label_onehot(inputs, num_segments):
    batch_size, im_h, im_w = inputs.shape
    outputs = torch.zeros((num_segments, batch_size, im_h, im_w)).cuda()

    inputs_temp = inputs.clone()
    inputs_temp[inputs == 255] = 0
    outputs.scatter_(0, inputs_temp.unsqueeze(1), 1.0)
    outputs[:, inputs == 255] = 0

    return outputs.permute(1, 0, 2, 3)


with Engine(custom_parser=parser) as engine:
    args = parser.parse_args()
    all_dev = parse_devices(args.dev)

    if args.dataset == 'VOC':
        dataset = VOC
        config = get_config_voc()
        get_train_loader = get_train_loader_voc
    elif args.dataset == 'voc_person':
        dataset = Dataset_uni
        config = get_config_voc_person()
        get_train_loader = get_train_loader_uni
    elif args.dataset == 'city_4':
        dataset = Dataset_uni
        config = get_config_city_4()
        get_train_loader = get_train_loader_uni
    elif args.dataset == 'VOC':
        dataset = VOC
        config = get_config_voc()
        get_train_loader = get_train_loader_voc
    elif args.dataset == 'city':
        dataset = CityScape
        config = get_config_city()
        get_train_loader = get_train_loader_city
    elif args.dataset == 'fish':
        dataset = Dataset_uni
        config = get_config_fish()
        get_train_loader = get_train_loader_uni
    elif args.dataset == 'water':
        dataset = Dataset_uni
        config = get_config_water()
        get_train_loader = get_train_loader_uni
    elif args.dataset == 'kvasir':
        dataset = Dataset_uni
        config = get_config_kvasir()
        get_train_loader = get_train_loader_uni
    elif args.dataset == 'disk':
        dataset = Dataset_uni
        config = get_config_disk()
        get_train_loader = get_train_loader_uni
    else:
        raise

    set_random_seed(config.seed, True, True,  engine.local_rank)
    # build class-wise memory bank
    memobank = []
    queue_ptrlis = []
    queue_size = []
    for i in range(config.num_classes):
        memobank.append([torch.zeros(0, 256)])
        queue_size.append(30000)
        queue_ptrlis.append(torch.zeros(1, dtype=torch.long))
    queue_size[0] = 50000


    # # build prototype
    # prototype = torch.zeros(
    #     (
    #         config["net"]["num_classes"],
    #         config["trainer"]["contrastive"]["num_queries"],
    #         1,
    #         256,
    #     )
    # ).cuda()

    mask_generator = mask_gen.BoxMaskGenerator(prop_range=config.cutmix_mask_prop_range, n_boxes=config.cutmix_boxmask_n_boxes,
                                            random_aspect_ratio=not config.cutmix_boxmask_fixed_aspect_ratio,
                                            prop_by_area=not config.cutmix_boxmask_by_size, within_bounds=not config.cutmix_boxmask_outside_bounds,
                                            invert=not config.cutmix_boxmask_no_invert)

    add_mask_params_to_batch = mask_gen.AddMaskParamsToBatch(
        mask_generator
    )
    collate_fn = SegCollate()
    mask_collate_fn = SegCollate(batch_aug_fn=add_mask_params_to_batch)

    train_loader_0, train_loader_1, train_sampler, unsupervised_train_loader_0, \
    unsupervised_train_sampler_0, unsupervised_train_loader_1, \
    unsupervised_train_sampler_1 = prepare_data(engine, dataset, config, collate_fn)

    if config.consistency_acm or config.consistency_acp:
        class_criterion = torch.rand(config.num_classes).type(torch.float32)
        cutmix_bank = torch.zeros(config.num_classes, unsupervised_train_loader_0.dataset.__len__()).cuda()
        class_momentum = 0.999
        all_cat = [i for i in range(config.num_classes)]
        ignore_cat = config.ignore_cat
        target_cat = list(set(all_cat)-set(ignore_cat))
        num_cat = config.number_cat
        area_thresh = config.area_thresh
        no_pad = True
        no_slim = True
        area_thresh2 = config.area_thresh2

    if engine.distributed and (engine.local_rank == 0):
        tb_dir = config.tb_dir + '/{}'.format(time.strftime("%b%d_%d-%H-%M", time.localtime()))
        generate_tb_dir = config.tb_dir + '/tb'
        logger = SummaryWriter(log_dir=tb_dir)
        engine.link_tb(tb_dir, generate_tb_dir)

    # config network and criterion
    criterion = get_criterion(config) # try to change criterion from Ohem to ordinary CE
    if config.consistency_acm or config.consistency_acp:
        criterion_csst = get_criterion(config, cons=True)
    else:
        criterion_csst = torch.nn.CrossEntropyLoss(ignore_index=255)

    if engine.distributed:
        BatchNorm2d = SyncBatchNorm

    model = Network(config.num_classes, criterion=criterion,
                    pretrained_model=config.pretrained_model,
                    norm_layer=BatchNorm2d, config=config, start_update=config.start_unsupervised_training)
    init_weight(model.branch1.business_layer, nn.init.kaiming_normal_,
                BatchNorm2d, config.bn_eps, config.bn_momentum,
                mode='fan_in', nonlinearity='relu')
    init_weight(model.branch2.business_layer, nn.init.kaiming_normal_,
                BatchNorm2d, config.bn_eps, config.bn_momentum,
                mode='fan_in', nonlinearity='relu')

    # group weight and config optimizer
    base_lr = config.lr
    if engine.distributed:
        base_lr = config.lr * engine.world_size

    params_list_l = []
    params_list_l = group_weight(params_list_l, model.branch1.backbone,
                               BatchNorm2d, base_lr)
    for module in model.branch1.business_layer:
        params_list_l = group_weight(params_list_l, module, BatchNorm2d,
                                   base_lr)

    optimizer_l = torch.optim.SGD(params_list_l,
                                lr=base_lr,
                                momentum=config.momentum,
                                weight_decay=config.weight_decay)

    params_list_r = []
    params_list_r = group_weight(params_list_r, model.branch2.backbone,
                               BatchNorm2d, 0.0)
    for module in model.branch2.business_layer:
        params_list_r = group_weight(params_list_r, module, BatchNorm2d,
                                   0.0)        # head lr * 10



    # config lr policy
    total_iteration = config.nepochs * config.niters_per_epoch
    lr_policy = WarmUpPolyLR(base_lr, config.lr_power, total_iteration, config.niters_per_epoch * config.warm_up_epoch)

    if engine.distributed:
        print('distributed !!')
        if torch.cuda.is_available():
            model.cuda()
            model = DistributedDataParallel(model)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = DataParallelModel(model, device_ids=engine.devices)
        model.to(device)

    engine.register_state(dataloader=train_loader_0, model=model,
                          optimizer_l=optimizer_l)
    if engine.continue_state_object:
        engine.restore_checkpoint()

    print('begin train')

    sum_loss_sup = 0
    sum_loss_sup_t = 0
    sum_csst = 0
    sum_contra = 0
    global_index = 0
    global_time = 0
    global_confidence = [0 for _ in range(config.num_classes)]
    model.train()

    for epoch in range(engine.state.epoch, config.nepochs):
        if engine.distributed:
            train_sampler.set_epoch(epoch)
        bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'

        if is_debug:
            pbar = tqdm(range(10), file=sys.stdout, bar_format=bar_format)
        else:
            pbar = tqdm(range(config.niters_per_epoch), file=sys.stdout, bar_format=bar_format)

        dataloader_0 = iter(train_loader_0)
        dataloader_1 = iter(train_loader_1)
        unsupervised_dataloader_0 = iter(unsupervised_train_loader_0)
        unsupervised_dataloader_1 = iter(unsupervised_train_loader_1)

        ''' supervised part '''
        start_time = time.time()

        for idx in pbar:
            global_index += 1
            optimizer_l.zero_grad()
            engine.update_iteration(epoch, idx)

            if (config.consistency_acm or config.consistency_acp) and config.num_classes > 2:
                conf = 1 - class_criterion
                conf = conf[target_cat]
                conf = (conf**0.5).numpy()
                conf = np.exp(conf)/np.sum(np.exp(conf))
                query_cat = []
                for rc_idx in range(num_cat):
                    query_cat.append(np.random.choice(target_cat, p=conf))
                query_cat = list(set(query_cat))
            else:
                # 0 - background (discard), 1 - object (always choose)
                conf = [0., 1.]
                query_cat = [1]

            minibatch_0 = dataloader_0.next()
            minibatch_1 = dataloader_1.next()
            unsup_minibatch_0 = unsupervised_dataloader_0.next()
            unsup_minibatch_1 = unsupervised_dataloader_1.next()
            labeled_imgs = minibatch_0['data']
            gts = minibatch_0['label']
            paste_imgs = minibatch_1['data']
            paste_gts = minibatch_1['label']
            unsup_imgs_0 = unsup_minibatch_0['data']
            unsup_imgs_1 = unsup_minibatch_1['data']
            img_id_0 = unsup_minibatch_0['id']
            img_id_1 = unsup_minibatch_1['id']
            mask_params = unsup_minibatch_0['mask_params']
            labeled_imgs = labeled_imgs.cuda(non_blocking=True)
            gts = gts.cuda(non_blocking=True)
            unsup_imgs_0 = unsup_imgs_0.cuda(non_blocking=True)
            unsup_imgs_1 = unsup_imgs_1.cuda(non_blocking=True)
            mask_params = mask_params.cuda(non_blocking=True)

            current_idx = epoch * config.niters_per_epoch + idx

            ### ACP pre-processing ###
            if config.consistency_acp and (epoch >= config.start_unsupervised_training):
                paste_imgs = paste_imgs.cuda()
                paste_gts = paste_gts.long().cuda()
                labeled_imgs, gts = dynamic_copy_paste(labeled_imgs, paste_imgs, gts, paste_gts, query_cat)
                del paste_imgs, paste_gts
            else:
                del paste_imgs, paste_gts

            ### Supervised inference ###
            s_rep_sup, s_sup_pred, aux_pred = model(labeled_imgs, step=1, cur_iter=epoch, return_rep=True, return_aux=True, update=True)

            with torch.no_grad():
                t_rep_sup, t_sup_pred, t_aux_pred = model(labeled_imgs, step=2, return_rep=True, no_upscale=True, return_aux=True)
                t_sup_pred_large = F.interpolate(t_sup_pred, size=gts.shape[1:], mode='bilinear', align_corners=True)
                t_conf_sup_pred_large = F.softmax(t_sup_pred_large, dim=1)
                t_logits_sup_pred_large, t_labels_sup_large = torch.max(t_conf_sup_pred_large, dim=1)
                t_labels_sup_large = t_labels_sup_large.long()

                # drop pixels with high entropy
                drop_percent = config.drop_percent
                percent_unreliable = (100 - drop_percent) * (1 - epoch /config.nepochs)
                drop_percent = 100 - percent_unreliable
                batch_size, num_class, h, w = t_sup_pred_large.shape

                entropy = -torch.sum(t_conf_sup_pred_large * torch.log(t_conf_sup_pred_large + 1e-10), dim=1)

                thresh = np.percentile(
                    entropy[t_labels_sup_large != 255].detach().cpu().numpy().flatten(), drop_percent
                )
                thresh_mask = entropy.ge(thresh).bool() * (t_labels_sup_large != 255).bool()

                t_labels_sup_large[thresh_mask] = 255
                weight_sup = batch_size * h * w / torch.sum(t_labels_sup_large != 255)

            ### Unsupervised inference ###
            if epoch >= config.start_unsupervised_training:

                ### ACM pre-processing ###
                if config.consistency_acm:
                    prob_im = random.random()
                    if prob_im > 0.5:
                        image_unsup = unsup_imgs_0
                        img_id = img_id_0
                    else:
                        image_unsup = unsup_imgs_1
                        img_id = img_id_1
                    # TODO maybe we can add loop here and draw different samples - not just one for all images.
                    image_unsup = image_unsup.cuda()
                    # sample id with target class
                    image_unsup2 = []
                    sample_id_bank = []
                    for _ in range(batch_size):
                        sample_id, sample_cat = sample_from_bank(cutmix_bank, class_criterion)
                        sample_id_bank.append(sample_id)
                        image_unsup2.append(unsupervised_dataloader_0._dataset.__getitem__(index=sample_id)['data'].unsqueeze(0))
                    image_unsup2 = torch.cat(image_unsup2)
                    image_unsup2 = image_unsup2.cuda()
                    # forward on this data
                    t_preds_unsup_1_large = model(image_unsup, step=2)
                    t_preds_unsup_2_large = model(image_unsup2, step=2)

                    labels_teacher_unsup_target_cat = torch.max(t_preds_unsup_2_large, dim=1)[1].cpu().numpy()
                    # generate cutmix mask relying on the $preds_teacher_unsup$
                    valid_mask_mix = []
                    for label_map in labels_teacher_unsup_target_cat:
                        valid_mask_mix.append(generate_cutmix_mask(label_map, sample_cat, area_thresh, no_pad=no_pad, no_slim=no_slim, num_classes=config.num_classes).unsqueeze(0))
                    valid_mask_mix = torch.cat(valid_mask_mix).unsqueeze(1)
                    valid_mask_mix = valid_mask_mix.cuda()
                    # actual mix the images
                    unsup_imgs_mixed = image_unsup * (1 - valid_mask_mix) + image_unsup2 * valid_mask_mix
                    #update cutmix bank for each image accordingly
                    for (sam_id, im_id) in zip(sample_id_bank, img_id):
                        cutmix_bank = update_cutmix_bank(cutmix_bank, t_preds_unsup_1_large, t_preds_unsup_2_large, im_id, sam_id, area_thresh2)
                    # mix the teacher labels
                    t_unsup_pred_mixed = t_preds_unsup_1_large * (1-valid_mask_mix) + t_preds_unsup_2_large * valid_mask_mix
                    t_unsup_prob_mixed = F.softmax(t_unsup_pred_mixed, dim=1)
                    t_unsup_logits_mixed, t_unsup_labels_mixed = torch.max(t_unsup_prob_mixed, dim=1)
                    t_unsup_labels_mixed = t_unsup_labels_mixed.long()

                else:
                    with torch.no_grad():
                        # unsupervised cutmix inference
                        t_unsup_pred_0_large = model(unsup_imgs_0, step=2, return_rep=False)
                        t_unsup_pred_1_large = model(unsup_imgs_1, step=2, return_rep=False)

                    # unsupervised loss on model/branch#1
                    batch_mix_masks = mask_params
                    unsup_imgs_mixed = unsup_imgs_0 * (1 - batch_mix_masks) + unsup_imgs_1 * batch_mix_masks

                    t_unsup_pred_mixed = t_unsup_pred_0_large * (1 - batch_mix_masks) + t_unsup_pred_1_large * batch_mix_masks
                    t_unsup_prob_mixed = F.softmax(t_unsup_pred_mixed, dim=1)
                    t_unsup_logits_mixed, t_unsup_labels_mixed = torch.max(t_unsup_prob_mixed, dim=1)
                    t_unsup_labels_mixed = t_unsup_labels_mixed.long()

                # unsupervised student cutmix inference
                s_rep_unsup, s_unsup_pred = model(unsup_imgs_mixed, step=1, cur_iter=epoch, return_rep=True, update=False)

                s_pred_all = torch.cat([s_sup_pred, s_unsup_pred])
                s_rep_all = torch.cat([s_rep_sup, s_rep_unsup])

                ### Mean Teacher loss ###
                ### Filter loss ###
                batch_size, num_class, h, w = t_unsup_pred_mixed.shape
                with torch.no_grad():
                    # drop pixels with high entropy
                    entropy = -torch.sum(t_unsup_prob_mixed * torch.log(t_unsup_prob_mixed + 1e-10), dim=1)

                    thresh = np.percentile(
                        entropy[t_unsup_labels_mixed != 255].detach().cpu().numpy().flatten(), drop_percent
                    )
                    thresh_mask = entropy.ge(thresh).bool() * (t_unsup_labels_mixed != 255).bool()

                    t_unsup_labels_mixed[thresh_mask] = 255
                    weight_unsup = batch_size * h * w / torch.sum(t_unsup_labels_mixed != 255)

                if config.consistency_acm or config.consistency_acp:
                    loss_consistency1 = criterion_csst(s_sup_pred, t_logits_sup_pred_large, t_labels_sup_large, class_criterion) / engine.world_size
                    loss_consistency2 = criterion_csst(s_unsup_pred, t_unsup_logits_mixed, t_unsup_labels_mixed, class_criterion) / engine.world_size
                    csst_loss = (loss_consistency1 + loss_consistency2) * config.unsup_weight
                else:
                    ### unsup loss ###
                    csst_loss = weight_unsup * criterion_csst(s_unsup_pred, t_unsup_labels_mixed)

                dist.all_reduce(csst_loss, dist.ReduceOp.SUM)

                with torch.no_grad():
                    if config.consistency_acm or config.consistency_acp:
                        category_entropy = cal_category_confidence(s_sup_pred.detach(), s_unsup_pred.detach(), gts.detach(), config.num_classes)
                        # perform momentum update
                        class_criterion = class_criterion * class_momentum + category_entropy * (1 - class_momentum)

                if config.use_contrastive_learning:
                    prob_sup_teacher =  F.softmax(t_sup_pred, dim=1)
                    with torch.no_grad():
                        # get the representations from teacher rep head
                        t_rep_unsup, t_unsup_mixed = model(unsup_imgs_mixed, step=2, return_rep=True, no_upscale=True)
                    t_rep_all = torch.cat([t_rep_sup, t_rep_unsup])

                    ### Contrastive loss ###
                    alpha_t = config.low_entropy_threshold * (
                        1 - epoch / config.nepochs
                    )

                    with torch.no_grad():
                        entropy = -torch.sum(t_unsup_prob_mixed * torch.log(t_unsup_prob_mixed + 1e-10), dim=1)

                        low_thresh = np.percentile(
                            entropy[t_unsup_labels_mixed != 255].cpu().numpy().flatten(), alpha_t
                        )
                        low_entropy_mask = (
                            entropy.le(low_thresh).float() * (t_unsup_labels_mixed != 255).bool()
                        )

                        high_thresh = np.percentile(
                            entropy[t_unsup_labels_mixed != 255].cpu().numpy().flatten(),
                            100 - alpha_t,
                        )
                        high_entropy_mask = (
                            entropy.ge(high_thresh).float() * (t_unsup_labels_mixed != 255).bool()
                        )

                        low_mask_all = torch.cat(
                            (
                                (gts.unsqueeze(1) != 255).float(),
                                low_entropy_mask.unsqueeze(1),
                            )
                        )

                        low_mask_all = F.interpolate(
                            low_mask_all, size=s_rep_all.shape[2:], mode="nearest"
                        )
                        # down sample

                        high_mask_all = torch.cat(
                            (
                                (gts.unsqueeze(1) != 255).float(),
                                high_entropy_mask.unsqueeze(1),
                            )
                        )

                        high_mask_all = F.interpolate(
                            high_mask_all, size=s_rep_all.shape[2:], mode="nearest"
                        )  # down sample

                        # down sample and concat
                        gts_small = F.interpolate(
                            label_onehot(gts, config.num_classes),
                            size=s_rep_all.shape[2:],
                            mode="nearest",
                        )
                        label_u_small = F.interpolate(
                            label_onehot(t_unsup_labels_mixed, config.num_classes),
                            size=s_rep_all.shape[2:],
                            mode="nearest",
                        )

                    # if not config_contra.get("anchor_ema", False): # delete if not needed
                    new_keys, contra_loss = compute_contra_memobank_loss(
                        s_rep_all,
                        gts_small.long(),
                        label_u_small.long(),
                        prob_sup_teacher.detach(),
                        t_unsup_mixed.detach(),
                        low_mask_all,
                        high_mask_all,
                        config,
                        memobank,
                        queue_ptrlis,
                        queue_size,
                        t_rep_all.detach(),
                    )

                    dist.all_reduce(contra_loss)
                    contra_loss = contra_loss / engine.world_size
                    contra_loss = contra_loss * config.unsup_contra_weight

                else:
                    contra_loss = 0 * s_rep_sup.sum()

            else:
                csst_loss = 0 * s_rep_sup.sum()
                contra_loss = 0 * s_rep_sup.sum()

            ### Supervised loss For Student ###
            loss_sup = criterion(s_sup_pred, gts, aux_pred)
            dist.all_reduce(loss_sup, dist.ReduceOp.SUM)
            loss_sup = loss_sup / engine.world_size

            ### Supervised loss For Teacher. No Backward. Just for the record ###
            with torch.no_grad():
                loss_sup_t = criterion(t_sup_pred_large, gts, t_aux_pred)
                dist.all_reduce(loss_sup_t, dist.ReduceOp.SUM)
                loss_sup_t = loss_sup_t / engine.world_size


            lr = lr_policy.get_lr(current_idx)

            optimizer_l.param_groups[0]['lr'] = lr
            optimizer_l.param_groups[1]['lr'] = lr
            for i in range(2, len(optimizer_l.param_groups)):
                optimizer_l.param_groups[i]['lr'] = lr

            loss = loss_sup + csst_loss + contra_loss
            loss.backward()
            optimizer_l.step()      # only the student model need to be updated by SGD.

            sum_loss_sup += loss_sup.item()
            sum_loss_sup_t += loss_sup_t.item()
            if epoch >= config.start_unsupervised_training:
                sum_csst += csst_loss.item()
                if config.use_contrastive_learning:
                    sum_contra += contra_loss.item()
                if config.consistency_acm or config.consistency_acp:
                    global_confidence = [gc + cr for gc,cr in zip(global_confidence, class_criterion)]

            print_str = 'Epoch{}/{}'.format(epoch, config.nepochs) \
                        + ' Iter{}/{}:'.format(idx + 1, config.niters_per_epoch) \
                        + ' lr=%.2e' % lr \
                        + ' loss_sup=%.2f' % (sum_loss_sup / global_index) \
                        + ' loss_sup_t=%.2f' % (sum_loss_sup_t / global_index)

            if (epoch >= config.start_unsupervised_training):
                print_str_2 = ' loss_csst=%.4f' % (sum_csst / global_index) \
                                + ' loss_contra=%.4f' % (sum_contra / global_index)
                print_str = print_str + print_str_2

            if (config.consistency_acm or config.consistency_acp):
                print_str_3 = f' confidence_per_class {[gc / global_index for gc in global_confidence]}'
                print_str = print_str + print_str_3

            pbar.set_description(print_str, refresh=False)

        end_time = time.time()
        global_time += end_time - start_time

        if engine.distributed and (engine.local_rank == 0):
            logger.add_scalar('train_loss_sup', sum_loss_sup / len(pbar), epoch)
            logger.add_scalar('train_loss_sup_t', sum_loss_sup_t / len(pbar), epoch)
            logger.add_scalar('train_loss_csst', sum_csst / len(pbar), epoch)
            logger.add_scalar('train_loss_contrast', sum_contra / len(pbar), epoch)

        if azure and engine.local_rank == 0:
            run.log(name='Supervised Training Loss', value=sum_loss_sup / len(pbar))
            run.log(name='Supervised Training Loss Teacher', value=sum_loss_sup_t / len(pbar))
            run.log(name='Unsupervised Training Loss CSST', value=sum_csst / len(pbar))
            run.log(name='Unsupervised Training Loss Contrastive', value=sum_contra / len(pbar))

        if (epoch % config.snapshot_iter == 0) or (epoch == config.nepochs - 1):
            if engine.distributed and (engine.local_rank == 0):
                engine.save_and_link_checkpoint(config.snapshot_dir,
                                                config.log_dir,
                                                config.log_dir_link)
            elif not engine.distributed:
                engine.save_and_link_checkpoint(config.snapshot_dir,
                                                config.log_dir,
                                                config.log_dir_link)

    print("OVERALL TRAINING TIME: ", global_time / config.nepochs)

