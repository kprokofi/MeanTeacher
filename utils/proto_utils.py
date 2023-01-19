import torch
import torch.nn.functional as F
import torch.nn as nn
import math
import warnings
from sklearn.cluster import KMeans
import random
import numpy as np
from tqdm import tqdm

def distributed_sinkhorn(out, sinkhorn_iterations=3, epsilon=0.05):
    L = torch.exp(out / epsilon).t() # K x B
    B = L.shape[1]
    K = L.shape[0]

    # make the matrix sums to 1
    sum_L = torch.sum(L)
    L /= sum_L

    for _ in range(sinkhorn_iterations):
        L /= torch.sum(L, dim=1, keepdim=True)
        L /= K

        L /= torch.sum(L, dim=0, keepdim=True)
        L /= B

    L *= B
    L = L.t()

    indexs = torch.argmax(L, dim=1)
    # L = torch.nn.functional.one_hot(indexs, num_classes=L.shape[1]).float()
    L = F.gumbel_softmax(L, tau=0.5, hard=True)

    return L, indexs

def momentum_update(old_value, new_value, momentum, debug=False):
    update = momentum * old_value + (1 - momentum) * new_value
    if debug:
        print("old prot: {:.3f} x |{:.3f}|, new val: {:.3f} x |{:.3f}|, result= |{:.3f}|".format(
            momentum, torch.norm(old_value, p=2), (1 - momentum), torch.norm(new_value, p=2),
            torch.norm(update, p=2)))
    return update


def l2_normalize(x):
    return F.normalize(x, p=2, dim=-1)

def init_prototypes(network, labeled_dataloader, config):
    # extract_fatures
    print("Starting features extraction for the prototypes initialization ....")
    trh = int(0.01 * 512 * 512)
    low_trh = int(0.15 * 512 * 512)
    feat_dim = 512
    features_bank = {i:[] for i in range(config.num_classes)}
    class_factor = {i:False for i in range(config.num_classes)}
    count_feat_bank = {i:0 for i in range(config.num_classes)}
    output_weigths = torch.zeros(config.num_classes, config.protoseg["num_prototype"], feat_dim, requires_grad=False, device="cuda")
    kmeans = KMeans(n_clusters=config.protoseg["num_prototype"], random_state=0, n_init=5)
    len_data = len(labeled_dataloader._dataset)
    k_s = len_data
    num_images_to_extract = random.sample(list(range(len_data)), k=k_s)
    for b in tqdm(num_images_to_extract):
        if all(class_factor.values()):
            break
        data = labeled_dataloader._dataset.__getitem__(index=b)
        labeled_images = data['data'].unsqueeze(0).cuda(non_blocking=True)
        labels = data['label'].cuda(non_blocking=True)
        features = network(labeled_images, init_prototypes=True)
        features = F.interpolate(features, size=(labeled_images.shape[-2], labeled_images.shape[-1]), mode='bilinear', align_corners=True)
        features = features[0].reshape(feat_dim, -1)
        for k in range(config.num_classes):
            if count_feat_bank[k] >= 200000:
                class_factor[k] = True
                continue
            k_mask = labels == k
            area = torch.sum(k_mask)
            if area < trh:
                continue
            elif area < low_trh:
                updated_mask = k_mask
            else:
                updated_mask = torch.rand(k_mask.shape) * k_mask.detach().cpu()
                perc = np.percentile(updated_mask[updated_mask > 0].reshape(-1), 80)
                updated_mask = updated_mask > perc

            updated_mask = updated_mask.reshape(1, -1).repeat(feat_dim, 1)
            k_features = features[updated_mask].view(feat_dim, -1).t().detach().cpu() # [ 512, -1 ] -> [ -1, 512 ]
            features_bank[k].append(k_features)
            count_feat_bank[k] += k_features.shape[0]

    for k, f in features_bank.items():
        new_f = torch.cat(f, 0).numpy() # stack all features for corresponding pixels [ -1, 512 ]
        n_s = min(100000, len(new_f))
        indices_to_sample = sorted(random.sample(range(len(new_f)), n_s))
        new_f = new_f[indices_to_sample]
        print(f"Starting K-Mean for the {k} class ....")
        kmeans.fit(new_f)
        output_weigths[k] = torch.tensor(kmeans.cluster_centers_, requires_grad=False, device="cuda")

    network.module.branch1.prototypes = nn.Parameter(output_weigths, requires_grad=False)
    network.module.branch1.cuda()

class ProjectionHead(nn.Module):
    def __init__(self, dim_in, proj_dim=256):
        super(ProjectionHead, self).__init__()

        self.proj = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_in, proj_dim, 1))

    def forward(self, x):
        return l2_normalize(self.proj(x))


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)