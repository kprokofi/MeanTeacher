from __future__ import division
import os.path as osp
import os
import sys
import time
import argparse
import math
from tqdm import tqdm

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
from utils.contrastive_loss import compute_contra_memobank_loss, get_consit_criterion
from engine.lr_policy import WarmUpPolyLR
from utils.pyt_utils import parse_devices
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

os.environ['MASTER_PORT'] = '169711'

if os.getenv('debug') is not None:
    is_debug = os.environ['debug']
else:
    is_debug = False

def prepare_data(engine, dataset, config, collate_fn):
    train_loader, train_sampler = get_train_loader(engine, dataset, train_source=config.train_source, \
                                                   unsupervised=False, collate_fn=collate_fn, config=config)
    unsupervised_train_loader_0, unsupervised_train_sampler_0 = get_train_loader(engine, dataset, \
                train_source=config.unsup_source, unsupervised=True, collate_fn=mask_collate_fn, config=config)
    unsupervised_train_loader_1, unsupervised_train_sampler_1 = get_train_loader(engine, dataset, \
                train_source=config.unsup_source, unsupervised=True, collate_fn=collate_fn, config=config)

    return (train_loader, train_sampler, unsupervised_train_loader_0,
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

    cudnn.benchmark = True

    seed = config.seed
    if engine.distributed:
        seed = engine.local_rank
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

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

    train_loader, train_sampler, unsupervised_train_loader_0, \
    unsupervised_train_sampler_0, unsupervised_train_loader_1, \
    unsupervised_train_sampler_1 = prepare_data(engine, dataset, config, collate_fn)

    if config.consistency:
        class_criterion = torch.rand(3, config.num_classes).type(torch.float32)
        cutmix_bank = torch.zeros(config.num_classes, unsupervised_train_loader_0.dataset.__len__()).cuda()
        class_momentum = 0.999
        all_cat = [i for i in range(config.num_classes)]
        ignore_cat = [0]
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
    criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=255)
    if config.consistency:
        sample = config.samples
        criterion_csst = get_consit_criterion(config, cons=True)
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
                                   base_lr)        # head lr * 10

    optimizer_l = torch.optim.SGD(params_list_l,
                                lr=base_lr,
                                momentum=config.momentum,
                                weight_decay=config.weight_decay)

    params_list_r = []
    params_list_r = group_weight(params_list_r, model.branch2.backbone,
                               BatchNorm2d, base_lr)
    for module in model.branch2.business_layer:
        params_list_r = group_weight(params_list_r, module, BatchNorm2d,
                                   base_lr)        # head lr * 10



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

    engine.register_state(dataloader=train_loader, model=model,
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
    model.train()

    for epoch in range(engine.state.epoch, config.nepochs):
        if engine.distributed:
            train_sampler.set_epoch(epoch)
        bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'

        if is_debug:
            pbar = tqdm(range(10), file=sys.stdout, bar_format=bar_format)
        else:
            pbar = tqdm(range(config.niters_per_epoch), file=sys.stdout, bar_format=bar_format)

        dataloader = iter(train_loader)
        unsupervised_dataloader_0 = iter(unsupervised_train_loader_0)
        unsupervised_dataloader_1 = iter(unsupervised_train_loader_1)

        if config.consistency:
            conf = 1 - class_criterion[0]
            conf = conf[target_cat]
            conf = (conf**0.5).numpy()
            conf_print = np.exp(conf)/np.sum(np.exp(conf))
            if engine.local_rank == 0:
                print('epoch [',epoch,': ]', 'sample_rate_target_class_conf', conf_print)
                print('epoch [',epoch,': ]', 'criterion_per_class' ,class_criterion[0])
                print('epoch [',epoch,': ]', 'sample_rate_per_class_conf' ,(1-class_criterion[0])/(torch.max(1-class_criterion[0])+1e-12))
            query_cat = []
            for rc_idx in range(num_cat):
                query_cat.append(np.random.choice(target_cat, p=conf))
            query_cat = list(set(query_cat))

        ''' supervised part '''
        start_time = time.time()
        for idx in pbar:
            global_index += 1
            optimizer_l.zero_grad()
            engine.update_iteration(epoch, idx)

            minibatch = dataloader.next()
            unsup_minibatch_0 = unsupervised_dataloader_0.next()
            unsup_minibatch_1 = unsupervised_dataloader_1.next()
            labeled_imgs,  paste_imgs = minibatch['data']
            gts, paste_gts = minibatch['label']
            unsup_imgs_0 = unsup_minibatch_0['data']
            unsup_imgs_1 = unsup_minibatch_1['data']
            mask_params = unsup_minibatch_0['mask_params']
            labeled_imgs = labeled_imgs.cuda(non_blocking=True)
            gts = gts.cuda(non_blocking=True)
            unsup_imgs_0 = unsup_imgs_0.cuda(non_blocking=True)
            unsup_imgs_1 = unsup_imgs_1.cuda(non_blocking=True)
            mask_params = mask_params.cuda(non_blocking=True)

            if paste_imgs:
                paste_imgs = paste_imgs.cuda()
                paste_gts = paste_gts.long().cuda() # TO DO HERE
                labeled_imgs, gts = dynamic_copy_paste(images_sup, labels_sup, paste_img, paste_label, query_cat)
                del paste_img, paste_label

            # unsupervised loss on model/branch#1
            batch_mix_masks = mask_params
            unsup_imgs_mixed = unsup_imgs_0 * (1 - batch_mix_masks) + unsup_imgs_1 * batch_mix_masks
            current_idx = epoch * config.niters_per_epoch + idx
            # supervised inference
            s_rep_sup, s_sup_pred = model(labeled_imgs, step=1, cur_iter=epoch, return_rep=True)
            batch_mix_masks = batch_mix_masks.reshape((batch_mix_masks.shape[0], unsup_imgs_0.shape[2],  unsup_imgs_0.shape[3]))
            with torch.no_grad():
                t_rep_sup, t_sup_pred = model(labeled_imgs, step=2, return_rep=True, no_upscale=True)

            if epoch >= config.start_unsupervised_training:
                with torch.no_grad():
                    # unsupervised cutmix inference
                    t_unsup_pred_0 = model(unsup_imgs_0, generate_pseudo=True, step=2)
                    t_unsup_pred_1 = model(unsup_imgs_1, generate_pseudo=True, step=2)

                logits_t_unsup_label_0, t_unsup_label_0 = torch.max(t_unsup_pred_0, dim=1)
                logits_t_unsup_label_1, t_unsup_label_1 = torch.max(t_unsup_pred_1, dim=1)

                unsup_labels_mixed = t_unsup_label_0 * (1 - batch_mix_masks) + t_unsup_label_1 * batch_mix_masks
                unsup_labels_mixed = unsup_labels_mixed.long()
                # unsupervised student cutmix inference
                s_rep_unsup, s_unsup_pred = model(unsup_imgs_mixed, step=1, cur_iter=epoch, return_rep=True)

                # unsupervised teacher cutmix inference
                with torch.no_grad():
                    t_rep_unsup, t_unsup_pred = model(unsup_imgs_mixed, step=2, return_rep=True, no_upscale=True)
                    t_unsup_pred_large = F.interpolate(t_unsup_pred, size=gts.shape[1:], mode='bilinear', align_corners=True)
                    prob_unsup_teacher_large = F.softmax(t_unsup_pred_large, dim=1)

                prob_sup_teacher =  F.softmax(t_sup_pred, dim=1)
                prob_unsup_teacher = F.softmax(t_unsup_pred, dim=1)

                s_pred_all = torch.cat([s_sup_pred, s_unsup_pred])
                s_rep_all = torch.cat([s_rep_sup, s_rep_unsup])
                t_rep_all = torch.cat([t_rep_sup, t_rep_unsup])

                ### Mean Teacher loss ###
                ### Filter loss ###
                drop_percent = config.drop_percent
                percent_unreliable = (100 - drop_percent) * (1 - epoch /config.nepochs)
                drop_percent = 100 - percent_unreliable
                batch_size, num_class, h, w = s_unsup_pred.shape
                with torch.no_grad():
                    # drop pixels with high entropy
                    entropy = -torch.sum(prob_unsup_teacher_large * torch.log(prob_unsup_teacher_large + 1e-10), dim=1)

                    thresh = np.percentile(
                        entropy[unsup_labels_mixed != 255].detach().cpu().numpy().flatten(), drop_percent
                    )
                    thresh_mask = entropy.ge(thresh).bool() * (unsup_labels_mixed != 255).bool()

                    unsup_labels_mixed[thresh_mask] = 255
                    weight = batch_size * h * w / torch.sum(unsup_labels_mixed != 255)

                CE_l = F.cross_entropy(s_unsup_pred, unsup_labels_mixed, ignore_index=255)
                csst_loss = weight * CE_l
                # csst_loss = criterion_csst(s_unsup_pred, unsup_labels_mixed)

                dist.all_reduce(csst_loss, dist.ReduceOp.SUM)
                csst_loss = csst_loss / engine.world_size
                csst_loss = csst_loss * config.unsup_weight

                if config.use_contrastive_learning:
                    ### Contrastive loss ###
                    alpha_t = config.low_entropy_threshold * (
                        1 - epoch / config.nepochs
                    )

                    with torch.no_grad():
                        prob = torch.softmax(t_unsup_pred_large, dim=1) # teacher inference -> get probs
                        entropy = -torch.sum(prob * torch.log(prob + 1e-10), dim=1)

                        low_thresh = np.percentile(
                            entropy[unsup_labels_mixed != 255].cpu().numpy().flatten(), alpha_t
                        )
                        low_entropy_mask = (
                            entropy.le(low_thresh).float() * (unsup_labels_mixed != 255).bool()
                        )

                        high_thresh = np.percentile(
                            entropy[unsup_labels_mixed != 255].cpu().numpy().flatten(),
                            100 - alpha_t,
                        )
                        high_entropy_mask = (
                            entropy.ge(high_thresh).float() * (unsup_labels_mixed != 255).bool()
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

                        # if config_contra.get("negative_high_entropy", True):
                        high_mask_all = torch.cat(
                            (
                                (gts.unsqueeze(1) != 255).float(),
                                high_entropy_mask.unsqueeze(1),
                            )
                        )
                        # else:
                        #     contra_flag += " low"
                        #     high_mask_all = torch.cat(
                        #         (
                        #             (gts.unsqueeze(1) != 255).float(),
                        #             torch.ones(logits_u_aug.shape)
                        #             .float()
                        #             .unsqueeze(1)
                        #             .cuda(),
                        #         ),
                        #     )
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
                            label_onehot(unsup_labels_mixed, config.num_classes),
                            size=s_rep_all.shape[2:],
                            mode="nearest",
                        )

                    # if not config_contra.get("anchor_ema", False): # delete if not needed
                    new_keys, contra_loss = compute_contra_memobank_loss(
                        s_rep_all,
                        gts_small.long(),
                        label_u_small.long(),
                        prob_sup_teacher.detach(),
                        prob_unsup_teacher.detach(),
                        low_mask_all,
                        high_mask_all,
                        config,
                        memobank,
                        queue_ptrlis,
                        queue_size,
                        t_rep_all.detach(),
                    )
                    # else:
                    #     prototype, new_keys, contra_loss = compute_contra_memobank_loss(
                    #         rep_all,
                    #         gts_small.long(),
                    #         label_u_small.long(),
                    #         prob_l_teacher.detach(),
                    #         prob_u_teacher.detach(),
                    #         low_mask_all,
                    #         high_mask_all,
                    #         cfg_contra,
                    #         memobank,
                    #         queue_ptrlis,
                    #         queue_size,
                    #         rep_all_teacher.detach(),
                    #         prototype,
                    #     )

                    dist.all_reduce(contra_loss)
                    contra_loss = contra_loss / engine.world_size
                    contra_loss = contra_loss * config.unsup_contra_weight

                else:
                    contra_loss = 0 * s_rep_sup.sum()

            else:
                csst_loss = unsup_loss = 0 * s_rep_sup.sum()
                contra_loss = 0 * s_rep_sup.sum()

            ### Supervised loss For Student ###
            loss_sup = criterion(s_sup_pred, gts)
            dist.all_reduce(loss_sup, dist.ReduceOp.SUM)
            loss_sup = loss_sup / engine.world_size

            ### Supervised loss For Teacher. No Backward ###
            with torch.no_grad():
                t_sup_pred_large = F.interpolate(t_sup_pred, size=gts.shape[1:], mode='bilinear', align_corners=True)
                loss_sup_t = criterion(t_sup_pred_large, gts)
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

            print_str = 'Epoch{}/{}'.format(epoch, config.nepochs) \
                        + ' Iter{}/{}:'.format(idx + 1, config.niters_per_epoch) \
                        + ' lr=%.2e' % lr \
                        + ' loss_sup=%.2f' % (sum_loss_sup / global_index) \
                        + ' loss_sup_t=%.2f' % (sum_loss_sup_t / global_index) \
                        + ' loss_csst=%.4f' % (sum_csst / global_index) \
                        + ' loss_contra=%.4f' % (sum_contra / global_index)

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

