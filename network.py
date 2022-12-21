# encoding: utf-8

import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from collections import OrderedDict
from base_model import resnet50

class Network(nn.Module):
    def __init__(self, num_classes, criterion, norm_layer, pretrained_model=None, config=None, start_update=1):
        super(Network, self).__init__()
        # student network
        self.branch1 = SingleNetwork(num_classes, criterion, norm_layer, pretrained_model, config=config)
        # teacher network
        self.branch2 = SingleNetwork(num_classes, criterion, norm_layer, pretrained_model, config=config)
        self.config = config
        self.start_update = start_update
        # detach the teacher model
        for param in self.branch2.parameters():
            param.detach_()

    def forward(self, data, step=1, cur_iter=None, generate_pseudo=False,
                    return_rep=False, no_upscale=False, return_aux=False, update=False):

        if not self.training:
            pred1 = self.branch1(data, no_upscale, return_aux)
            return pred1

        if generate_pseudo:
            with torch.no_grad():
                t_rep, t_out, _ = self.branch2(data, no_upscale, return_aux)
            if return_rep:
                return t_rep, t_out
            return t_out

        # copy the parameters from teacher to student
        if cur_iter == self.start_update:
            for t_param, s_param in zip(self.branch2.parameters(), self.branch1.parameters()):
                t_param.data.copy_(s_param.data)

        if step == 1:
            s_feature, s_out, pred_aux = self.branch1(data, no_upscale, return_aux)
            if cur_iter >= self.start_update and update:
                self._update_ema_variables(self.config.ema_decay, cur_iter)
            if return_rep and not return_aux:
                return s_feature, s_out
            if return_rep and return_aux:
                return s_feature, s_out, pred_aux
            if return_aux:
                return s_out, pred_aux

            return s_out

        if step == 2:
            with torch.no_grad():
                t_feature, t_out, pred_aux = self.branch2(data, no_upscale, return_aux)
                if return_rep and not return_aux:
                    return t_feature, t_out
                if return_rep and return_aux:
                    return t_feature, t_out, pred_aux
                if return_aux:
                    return t_out, pred_aux
                return t_out

    def _update_ema_variables(self, ema_decay, cur_step):
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
    def __init__(self, num_classes, criterion, norm_layer, pretrained_model=None, config=None):
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

        # self.head = Head(num_classes, norm_layer, self.config.bn_momentum)
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


    def forward(self, data, no_upscale=False, return_aux=False):
        h, w = data.shape[-1], data.shape[-2]
        blocks = self.backbone(data)
        x1,x2,x3,x4 = blocks
        v3plus_feature = self.head(blocks)

        if self.use_aux:
            # feat1 used as dsn loss as default, f1 is layer2's output as default
            pred_aux = self.auxor(x3)
            pred_aux = F.upsample(input=pred_aux, size=(h, w), mode='bilinear', align_corners=True)

        # v3plus_feature = self.head(blocks)
        # pred = self.classifier(v3plus_feature)
        pred = v3plus_feature["pred"]
        if "rep" in v3plus_feature:
            rep = v3plus_feature["rep"]

        if not no_upscale:
            pred = F.interpolate(pred, size=(h, w), mode='bilinear', align_corners=True)

        if return_aux and self.training:
            return rep, pred, pred_aux
        elif self.training:
            return rep, pred, None
        return pred

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
            nn.Conv2d(256, 256, kernel_size=1), norm_layer(256, momentum=bn_momentum), nn.ReLU(inplace=True)
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
