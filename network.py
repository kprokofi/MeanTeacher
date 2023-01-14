# encoding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from functools import partial
from einops import rearrange, repeat

from base_model import resnet50
from utils.proto_utils import distributed_sinkhorn, momentum_update, l2_normalize, ProjectionHead, trunc_normal_

class Network(nn.Module):
    def __init__(self, num_classes, criterion, norm_layer, pretrained_model=None, config=None, start_update=1):
        super(Network, self).__init__()
        # student network
        self.branch1 = SingleNetwork(num_classes, criterion, norm_layer, pretrained_model, config=config, use_prototypes=config.protoseg.use_prototypes)
        # teacher network
        self.branch2 = SingleNetwork(num_classes, criterion, norm_layer, pretrained_model, config=config, use_prototypes=config.protoseg.use_prototypes, is_teacher=True)
        self.config = config
        self.start_update = start_update
        # detach the teacher model
        for param in self.branch2.parameters():
            param.detach_()

    def forward(self, data, step=1, cur_iter=None, no_upscale=False,
                    update=False, gt_semantic_seg=None,
                    pretrain_prototype=False):

        if not self.training:
            pred1 = self.branch1(data, no_upscale)
            return pred1

        # copy the parameters from teacher to student
        if cur_iter == self.start_update:
            for t_param, s_param in zip(self.branch2.parameters(), self.branch1.parameters()):
                t_param.data.copy_(s_param.data)

        if step == 1:
            s_out = self.branch1(data, no_upscale, gt_semantic_seg=gt_semantic_seg, pretrain_prototype=pretrain_prototype)
            if cur_iter >= self.start_update and update:
                self._update_ema_variables(self.config.ema_decay, cur_iter)
            return s_out

        if step == 2:
            with torch.no_grad():
                t_out = self.branch2(data, no_upscale)
            return t_out

    def _update_ema_variables(self, ema_decay, cur_step):
        # for name, t_param in self.branch2.state_dict().items():
        #     s_param = self.branch1.state_dict()[name]
        #     t_param.data = t_param.data * ema_decay + (1 - ema_decay) * s_param.data
        for t_param, s_param in zip(self.branch2.parameters(), self.branch1.parameters()):
            t_param.data = t_param.data * ema_decay + (1 - ema_decay) * s_param.data


class Aux_Module(nn.Module):
    def __init__(self, in_planes, num_classes=19, norm_layer=nn.BatchNorm2d):
        super(Aux_Module, self).__init__()

        norm_layer = norm_layer
        self.aux = nn.Sequential(
                nn.Conv2d(in_planes, 256, kernel_size=3, stride=1, padding=1),
                norm_layer(256),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.1),
                nn.Conv2d(256, num_classes, kernel_size=1, stride=1, padding=0, bias=True))

    def forward(self, x):
        res = self.aux(x)
        return res


class SingleNetwork(nn.Module):
    def __init__(self, num_classes, criterion, norm_layer, pretrained_model=None, config=None, use_prototypes=False, is_teacher=False):
        super(SingleNetwork, self).__init__()
        self.config = config
        self.backbone = resnet50(pretrained_model, norm_layer=norm_layer,
                                  bn_eps=self.config.bn_eps,
                                  bn_momentum=self.config.bn_momentum,
                                  deep_stem=True, stem_width=64)
        self.dilate = 2
        for m in self.backbone.layer4.children():
            m.apply(partial(self._nostride_dilate, dilate=self.dilate))
            self.dilate *= 2

        self.use_prototype = use_prototypes
        self.teacher = is_teacher
        # self.head = Head(num_classes, norm_layer, self.config.bn_momentum)
        if self.use_prototype:
            in_proto_channels = 512
            #### PROTO init ####
            self.gamma = config.protoseg['gamma']
            self.num_prototype = config.protoseg['num_prototype']
            self.pretrain_prototype = config.protoseg['pretrain_prototype']
            self.num_classes = config.num_classes
            self.prototypes = nn.Parameter(torch.zeros(self.num_classes, self.num_prototype, in_proto_channels),
                                    requires_grad=False)
            trunc_normal_(self.prototypes, std=0.02)
            self.avg_pool = nn.AdaptiveAvgPool2d(256)
            self.map_convs = nn.ModuleList([
                    nn.Conv2d(256, 128, 1, bias=False),
                    nn.Conv2d(512, 128, 1, bias=False),
                    nn.Conv2d(1024, 128, 1, bias=False),
                    nn.Conv2d(2048, 128, 1, bias=False)
                ])
            self.map_bn = norm_layer(256 * 4)
            # self.proto_head = nn.Sequential(
            #     nn.Conv2d(in_proto_channels, in_proto_channels, kernel_size=3, stride=1, padding=1),
            #     norm_layer(in_proto_channels, momentum=self.config.bn_momentum),
            #     nn.ReLU(inplace=True),
            #     nn.Dropout2d(0.10)
            # )

            self.proj_head = ProjectionHead(in_proto_channels, in_proto_channels)
            self.feat_norm = nn.LayerNorm(in_proto_channels)
            self.mask_norm = nn.LayerNorm(self.num_classes)

        self.head = dec_deeplabv3_plus(num_classes, norm_layer, self.config.bn_momentum)
        self.business_layer = []
        self.business_layer.append(self.head)
        self.criterion = criterion
        cfg_aux = config.aux_loss
        self.use_aux = cfg_aux['use_auxloss']
        if cfg_aux['use_auxloss']:
            self.auxor = Aux_Module(cfg_aux['aux_plane'], num_classes, norm_layer)

        self.classifier = nn.Conv2d(256, num_classes, kernel_size=1, bias=True)
        self.business_layer.append(self.classifier)


    def forward(self, data, no_upscale=False, gt_semantic_seg=None, pretrain_prototype=False):
        h, w = data.shape[-1], data.shape[-2]
        blocks = self.backbone(data)
        x1,x2,x3,x4 = blocks
        _h, _w = x1.shape[-1], x1.shape[-2]
        v3plus_feature = self.head(blocks) # main head

        if self.use_aux:
            # feat1 used as dsn loss as default, f1 is layer2's output as default
            pred_aux = self.auxor(x3)
            pred_aux = F.upsample(input=pred_aux, size=(h, w), mode='bilinear', align_corners=True)
            v3plus_feature['aux'] = pred_aux

        if self.use_prototype and not self.teacher:
            ### protototype learning ###
            # feat1 = self.map_convs[0](x1)
            # feat2 = F.interpolate(self.map_convs[1](x2), size=(_h, _w), mode="bilinear", align_corners=True)
            # feat3 = F.interpolate(self.map_convs[2](x3), size=(_h, _w), mode="bilinear", align_corners=True)
            # feat4 = F.interpolate(self.map_convs[3](x4), size=(_h, _w), mode="bilinear", align_corners=True)
            # feats = torch.cat([feat1, feat2, feat3, feat4], 1)
            feats = v3plus_feature['aspp_out']
            # c = self.proto_head(feats)
            c = self.proj_head(feats)
            _c = rearrange(c, 'b c h w -> (b h w) c')
            _c = self.feat_norm(_c)
            _c = l2_normalize(_c)
            self.prototypes.data.copy_(l2_normalize(self.prototypes))

            # n: h*w, k: num_class, m: num_prototype
            masks = torch.einsum('nd,kmd->nmk', _c, self.prototypes)

            out_seg = torch.amax(masks, dim=1)
            out_seg = self.mask_norm(out_seg)
            out_seg = rearrange(out_seg, "(b h w) k -> b k h w", b=feats.shape[0], h=feats.shape[2])

            if not pretrain_prototype and gt_semantic_seg is not None:
                gt_seg = F.interpolate(gt_semantic_seg.float(), size=feats.size()[2:], mode='nearest').view(-1)
                contrast_logits, contrast_target = self.prototype_learning(_c, out_seg, gt_seg, masks)
                v3plus_feature.update({'seg_out': out_seg, 'proto_logits': contrast_logits, 'proto_targets': contrast_target})

            # else:
            #     if not self.training:
            #         return  F.interpolate(out_seg, size=(h, w), mode='bilinear', align_corners=True)

        # pred = self.classifier(v3plus_feature)
        if not no_upscale:
            v3plus_feature["pred"] = F.interpolate(v3plus_feature["pred"], size=(h, w), mode='bilinear', align_corners=True)
            if self.use_prototype and not self.teacher and self.training:
                v3plus_feature['seg_out'] = F.interpolate(v3plus_feature["seg_out"], size=(h, w), mode='bilinear', align_corners=True)

        if self.training:
            return v3plus_feature

        return v3plus_feature["pred"]

    # @staticmethod
    def _nostride_dilate(self, m, dilate):
        if isinstance(m, nn.Conv2d):
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def prototype_learning(self, _c, out_seg, gt_seg, masks):
        pred_seg = torch.max(out_seg, 1)[1]
        mask = (gt_seg == pred_seg.view(-1))

        cosine_similarity = torch.mm(_c, self.prototypes.view(-1, self.prototypes.shape[-1]).t())

        proto_logits = cosine_similarity
        proto_target = gt_seg.clone().float()

        # clustering for each class
        protos = self.prototypes.data.clone()
        for k in range(self.num_classes):
            init_q = masks[..., k]
            init_q = init_q[gt_seg == k, ...]
            if init_q.shape[0] == 0:
                continue

            q, indexs = distributed_sinkhorn(init_q)

            m_k = mask[gt_seg == k]

            c_k = _c[gt_seg == k, ...]

            m_k_tile = repeat(m_k, 'n -> n tile', tile=self.num_prototype)

            m_q = q * m_k_tile  # n x self.num_prototype

            c_k_tile = repeat(m_k, 'n -> n tile', tile=c_k.shape[-1])

            c_q = c_k * c_k_tile  # n x embedding_dim

            f = m_q.transpose(0, 1) @ c_q  # self.num_prototype x embedding_dim

            n = torch.sum(m_q, dim=0)

            if torch.sum(n) > 0:
                f = F.normalize(f, p=2, dim=-1)

                new_value = momentum_update(old_value=protos[k, n != 0, :], new_value=f[n != 0, :],
                                            momentum=self.gamma, debug=False)
                protos[k, n != 0, :] = new_value

            proto_target[gt_seg == k] = indexs.float() + (self.num_prototype * k)

        self.prototypes = nn.Parameter(l2_normalize(protos),
                                       requires_grad=False)

        if dist.is_available() and dist.is_initialized():
            protos = self.prototypes.data.clone()
            dist.all_reduce(protos.div_(dist.get_world_size()))
            self.prototypes = nn.Parameter(protos, requires_grad=False)

        return proto_logits, proto_target

class ASPP(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 dilation_rates=(12, 24, 36),
                 hidden_channels=256,
                 norm_act=nn.BatchNorm2d,
                 pooling_size=None):
        super(ASPP, self).__init__()
        self.pooling_size = pooling_size

        self.map_convs = nn.ModuleList([
            nn.Conv2d(in_channels, hidden_channels, 1, bias=False),
            nn.Conv2d(in_channels, hidden_channels, 3, bias=False, dilation=dilation_rates[0],
                      padding=dilation_rates[0]),
            nn.Conv2d(in_channels, hidden_channels, 3, bias=False, dilation=dilation_rates[1],
                      padding=dilation_rates[1]),
            nn.Conv2d(in_channels, hidden_channels, 3, bias=False, dilation=dilation_rates[2],
                      padding=dilation_rates[2])
        ])

        self.map_bn = norm_act(hidden_channels * 4)

        self.global_pooling_conv = nn.Conv2d(in_channels, hidden_channels, 1, bias=False)
        self.global_pooling_bn = norm_act(hidden_channels)

        self.red_conv = nn.Conv2d(hidden_channels * 4, out_channels, 1, bias=False)
        self.pool_red_conv = nn.Conv2d(hidden_channels, out_channels, 1, bias=False)
        self.red_bn = norm_act(out_channels)
        self.out_planes = out_channels

        self.leak_relu = nn.LeakyReLU()

    def forward(self, x):
        # Map convolutions
        out = torch.cat([m(x) for m in self.map_convs], dim=1)
        out = self.map_bn(out)
        out = self.leak_relu(out)       # add activation layer
        out = self.red_conv(out)

        # Global pooling
        pool = self._global_pooling(x)
        pool = self.global_pooling_conv(pool)
        pool = self.global_pooling_bn(pool)

        pool = self.leak_relu(pool)  # add activation layer

        pool = self.pool_red_conv(pool)
        if self.training or self.pooling_size is None:
            pool = pool.repeat(1, 1, x.size(2), x.size(3))

        out += pool
        out = self.red_bn(out)
        out = self.leak_relu(out)  # add activation layer
        return out

    def _global_pooling(self, x):
        if self.training or self.pooling_size is None:
            pool = x.view(x.size(0), x.size(1), -1).mean(dim=-1)
            pool = pool.view(x.size(0), x.size(1), 1, 1)
        else:
            pooling_size = (min(try_index(self.pooling_size, 0), x.shape[2]),
                            min(try_index(self.pooling_size, 1), x.shape[3]))
            padding = (
                (pooling_size[1] - 1) // 2,
                (pooling_size[1] - 1) // 2 if pooling_size[1] % 2 == 1 else (pooling_size[1] - 1) // 2 + 1,
                (pooling_size[0] - 1) // 2,
                (pooling_size[0] - 1) // 2 if pooling_size[0] % 2 == 1 else (pooling_size[0] - 1) // 2 + 1
            )

            pool = nn.functional.avg_pool2d(x, pooling_size, stride=1)
            pool = nn.functional.pad(pool, pad=padding, mode="replicate")
        return pool

class ASPP2(nn.Module):
    """
    Reference:
        Chen, Liang-Chieh, et al. *"Rethinking Atrous Convolution for Semantic Image Segmentation."*
    """

    def __init__(
        self, in_planes, inner_planes=256, norm_layer=nn.BatchNorm2d, dilations=(12, 24, 36)
    ):
        super(ASPP2, self).__init__()

        norm_layer = norm_layer
        self.conv1 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(
                in_planes,
                inner_planes,
                kernel_size=1,
                padding=0,
                dilation=1,
                bias=False,
            ),
            norm_layer(inner_planes),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_planes,
                inner_planes,
                kernel_size=1,
                padding=0,
                dilation=1,
                bias=False,
            ),
            norm_layer(inner_planes),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_planes,
                inner_planes,
                kernel_size=3,
                padding=dilations[0],
                dilation=dilations[0],
                bias=False,
            ),
            norm_layer(inner_planes),
            nn.ReLU(inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_planes,
                inner_planes,
                kernel_size=3,
                padding=dilations[1],
                dilation=dilations[1],
                bias=False,
            ),
            norm_layer(inner_planes),
            nn.ReLU(inplace=True),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(
                in_planes,
                inner_planes,
                kernel_size=3,
                padding=dilations[2],
                dilation=dilations[2],
                bias=False,
            ),
            norm_layer(inner_planes),
            nn.ReLU(inplace=True),
        )

        self.out_planes = (len(dilations) + 2) * inner_planes

    def get_outplanes(self):
        return self.out_planes

    def forward(self, x):
        _, _, h, w = x.size()
        feat1 = F.interpolate(
            self.conv1(x), size=(h, w), mode="bilinear", align_corners=True
        )
        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        feat5 = self.conv5(x)
        aspp_out = torch.cat((feat1, feat2, feat3, feat4, feat5), 1)
        return aspp_out

class Head(nn.Module):
    def __init__(self, classify_classes, norm_act=nn.BatchNorm2d, bn_momentum=0.0003):
        super(Head, self).__init__()

        self.classify_classes = classify_classes
        self.aspp = ASPP(2048, 256, [6, 12, 18], norm_act=norm_act)

        self.reduce = nn.Sequential(
            nn.Conv2d(256, 48, 1, bias=False),
            norm_act(48, momentum=bn_momentum),
            nn.ReLU(),
        )
        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       norm_act(256, momentum=bn_momentum),
                                       nn.ReLU(),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       norm_act(256, momentum=bn_momentum),
                                       nn.ReLU(),
                                       )

    def forward(self, f_list):
        f = f_list[-1]
        f = self.aspp(f)

        low_level_features = f_list[0]
        low_h, low_w = low_level_features.size(2), low_level_features.size(3)
        low_level_features = self.reduce(low_level_features)

        f = F.interpolate(f, size=(low_h, low_w), mode='bilinear', align_corners=True)
        f = torch.cat((f, low_level_features), dim=1)
        f = self.last_conv(f)

        return f

class dec_deeplabv3_plus(nn.Module):
    def __init__(
        self,
        num_classes=19,
        norm_layer = nn.BatchNorm2d,
        bn_momentum = 0.0001,
        in_planes=2048,
        inner_planes=256,
        dilations=(12, 24, 36),
        rep_head=True,
    ):
        super(dec_deeplabv3_plus, self).__init__()

        self.rep_head = rep_head

        self.low_conv = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1),
            norm_layer(256, momentum=bn_momentum),
            nn.ReLU(inplace=True)
        )

        self.aspp = ASPP2(
            in_planes, inner_planes=inner_planes, norm_layer=norm_layer, dilations=dilations
        )

        self.head = nn.Sequential(
            nn.Conv2d(
                self.aspp.out_planes,
                256,
                kernel_size=3,
                padding=1,
                dilation=1,
                bias=False,
            ),
            norm_layer(256, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
        )

        self.pre_classifier = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=True),
            norm_layer(256, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            norm_layer(256, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )
        self.classifier = nn.Conv2d(256, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        self.rep_classifier = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        x1, x2, x3, x4 = x
        aspp_out = self.aspp(x4)
        low_feat = self.low_conv(x1)
        aspp_out = self.head(aspp_out)
        h, w = low_feat.size()[-2:]
        aspp_out = F.interpolate(
            aspp_out, size=(h, w), mode="bilinear", align_corners=True
        )

        aspp_out = torch.cat((low_feat, aspp_out), dim=1)
        aspp_out = self.pre_classifier(aspp_out)
        res = {"pred": self.classifier(aspp_out)}
        res["aspp_out"] = aspp_out
        res["rep"] = self.rep_classifier(aspp_out)

        return res


if __name__ == '__main__':
    model = Network(40, criterion=nn.CrossEntropyLoss(),
                    pretrained_model=None,
                    norm_layer=nn.BatchNorm2d)
    left = torch.randn(2, 3, 128, 128)
    right = torch.randn(2, 3, 128, 128)

    print(model.backbone)

    out = model(left)
    print(out.shape)
