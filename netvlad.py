import torch
import torch.nn as nn
import torch.nn.functional as torch_func

class BaseModule(nn.Module):
    def __init__(self):
        super(BaseModule, self).__init__()
        self.enable_summary = True

def GetStdOrthnormalMatrix(h, w):
    randmat = torch.rand(h, w)
    u, s, v = torch.svd(randmat)
    # v.shape = w * h
    return v

class vlad_core_euc(BaseModule):
    def __init__(self, fg_num=32, bg_num=0, dim=64, alpha=10.0, static=False, residual=True, norm_input=True, norm_intra='l2', norm_init=True, learnw=False, zero_bg=False):
        super().__init__()
        self.fg_num = fg_num
        self.bg_num = bg_num
        self.dim = dim
        self.alpha = alpha
        self.static = static
        self.enable_summary = True
        self.residual = residual
        self.norm_input = norm_input
        self.norm_intra = norm_intra
        self.norm_init = norm_init
        self.learnw = learnw
        self.zero_bg = zero_bg

        self.num_clusters = self.fg_num + self.bg_num
        self.fg_len = self.dim * self.fg_num
        if not static:
            random_normalize = torch.rand(dim, self.num_clusters) * 2.0 - 1.0
            if self.norm_init:
                random_normalize = torch_func.normalize(random_normalize, p=2, dim=0)
                # random_normalize = GetStdOrthnormalMatrix(self.num_clusters, self.dim)
            self.centroids = nn.Parameter(random_normalize)
        else:
            self.centroids = nn.Parameter(GetStdOrthnormalMatrix(self.num_clusters, self.dim))
        self.output_size = self.fg_num * self.dim
        self.supress = None
        self.mask = None

        if self.learnw:
            self.weights = nn.Parameter(torch.ones(fg_num))
        else:
            self.weights = None
    
    def extra_repr(self) -> str:
        keywords = {
            "fg_num": self.fg_num,
            "bg_num": self.bg_num,
            "dim": self.dim,
            "alpha": self.alpha,
            "static": self.static,
            "residual": self.residual,
            "norm_input": self.norm_input,
            "norm_intra": self.norm_intra,
            "norm_init": self.norm_init,
            "learnw": self.learnw,
            "zero_bg": self.zero_bg
        }
        info = ', '.join(['{0}={1}'.format(key, value) for key, value in keywords.items()])
        return info

    def forward(self, x):
        """
        x: b * n * d
        background: None or b * n
        """
        # b : batchsize n: num_datapoints c: num_centers d: input datapoint dimension
        # x.shape: b * n * d
        b, n, d = x.size()
        c = self.num_clusters
        x_norm = x
        if self.norm_input:
            x_norm = torch_func.normalize(x, p=2, dim=2) # b * n * d
        self.x = x_norm

        centroids = self.centroids # d * c

        if self.zero_bg:
            centroids = torch.cat([centroids, torch.zeros(d, 1).type(x.type())], dim=1) # d * c + 1
            c = c + 1

        # calculate pair-wise distance & generate soft assign.
        x_norm_for_assign = x_norm
        dot = x_norm_for_assign.bmm(centroids.repeat(b, 1, 1)) # b * n * c
        c_len = (centroids**2).sum(dim=0) # c
        c_len = c_len.repeat(b, n, 1) # b * n * c
        x_len = (x_norm_for_assign**2).sum(dim=2, keepdim=True).repeat(1, 1, c)
        distance = x_len + c_len - dot * 2 # ||x - y||^2 = ||x||^2 + ||y^2|| - 2x^Ty
        assign = torch_func.softmax(-distance*self.alpha, dim=2) # b * n * c
        self.assign = assign
        # print('assign', assign.sum(dim=0).sum(dim=0))

        # generate vlad.
        x_norm = x_norm.permute(0, 2, 1).contiguous() # b * d * n
        x_sum = x_norm.bmm(assign)
        if self.residual:
            c_weight = assign.sum(dim=1) # b * c
            c_weight = torch.diag_embed(c_weight) # b * c * c
            c_sum = centroids.repeat(b, 1, 1).bmm(c_weight)
            vlad = x_sum - c_sum
        else:
            vlad = x_sum

        # flatten vlad vector
        vlad = vlad.permute(0, 2, 1).contiguous() # b * c * d
        if self.norm_intra == 'l2':
            vlad = torch_func.normalize(vlad, p=2, dim=2) # intra normalization
        elif self.norm_intra == 'w':
            # vlad.shape = b * c * d
            # weight.shape = b * c
            c_weight = assign.sum(dim=1) # b * c
            c_weight = c_weight.unsqueeze(2).repeat(1, 1, d) # b * c * d
            vlad = vlad / c_weight # normalize.

        if self.supress is not None:
            if self.mask is None:
                mask = torch.ones_like(vlad[0])
                mask[self.supress] = 0
                self.mask = mask
            mask = self.mask.repeat(b, 1, 1)
            vlad = vlad * mask
        
        # vlad.shape = b * c * d
        if self.weights is not None:
            weights = torch_func.softmax(self.weights, dim=0) * self.fg_num
            # print("weights: ", weights)
            weights = weights.repeat(b, d, 1).permute(0, 2, 1).contiguous()
            vlad = vlad * weights

        vlad = vlad.view(b, -1) # flatten vlad vector
        vlad = torch_func.normalize(vlad, p=2, dim=1) # final l2 normalization.


        # divide foreground and background.
        self.fg = vlad[:, :self.fg_len]
        self.bg = vlad[:, self.fg_len:]
        return self.fg

class vlad_core_cos(nn.Module):
    def __init__(self, fg_num=32, bg_num=4, dim=64, alpha=10.0, **kwargs):
        super().__init__()
        self.fg_num = fg_num
        self.bg_num = bg_num
        self.dim = dim
        self.alpha = alpha
        self.enable_summary = True

        self.num_clusters = self.fg_num + self.bg_num
        self.fg_len = self.dim * self.fg_num
        self.centroids = nn.Parameter((torch.rand(dim, self.num_clusters)*2.0-1.0))
        self.output_size = self.fg_num * self.dim
    
    def extra_repr(self) -> str:
        info = 'fg_num={0}, bg_num={1}, dim={2}, alpha={3}'.format(
            self.fg_num, self.bg_num, self.dim, self.alpha
        )
        return info

    def forward(self, x):
        # b : batchsize n: num_datapoints c: num_centers d: input datapoint dimension
        # x.shape: b * n * d
        b, n, d = x.size()
        c = self.num_clusters
        bndc = [b, n, d, c]
        # normalize inpust and centroids alongside feature vector dim.
        x_norm = torch_func.normalize(x, p=2, dim=2) # b * n * d
        c_norm = torch_func.normalize(self.centroids, p=2, dim=0) # d * c
        # calculate soft assign.
        cos_sim = x_norm.bmm(c_norm.repeat(b, 1, 1)) # b * n * c
        assign = torch_func.softmax(cos_sim*self.alpha, dim=2)
        self.assign = assign # record.
        # generate vlad vector. this time, we calculate residual with orthnormal decompose direction.
        x_norm = x_norm.permute(0, 2, 1).contiguous() # b * d * n
        weighted_sum = x_norm.bmm(assign) # b * d * c
        self_product = (cos_sim * assign).sum(dim=1) # b * c
        self_product_extend = torch.diag_embed(self_product) # b * c * c
        projection = c_norm.repeat(b, 1, 1).bmm(self_product_extend) # b * d * c
        vlad = weighted_sum - projection
        # generate vlad vector
        vlad = vlad.permute(0, 2, 1).contiguous() # b * c * d
        vlad = torch_func.normalize(vlad, p=2, dim=2) # intra normalization
        vlad = vlad.view(b, d*c) # flatten vlad vector
        vlad = torch_func.normalize(vlad, p=2, dim=1) # final l2 normalization.
        # divide foreground and background.
        # divide foreground and background.
        self.fg = vlad[:, :self.fg_len]
        self.bg = vlad[:, self.fg_len:]
        return self.fg

class Bottleneck(nn.Module):
    def __init__(self, in_channel, out_channel=None, bottleneck=None, use_bn=False, use_relu=False):
        super().__init__()
        if out_channel is None:
            out_channel = in_channel
        if bottleneck is None:
            bottleneck = in_channel//4
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.bottleneck = bottleneck
        layers = [
            nn.Conv2d(in_channel, bottleneck, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(bottleneck),
            nn.ReLU(),
            nn.Conv2d(bottleneck, bottleneck, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(bottleneck),
            nn.ReLU(),
            nn.Conv2d(bottleneck, out_channel, kernel_size=(1, 1), stride=(1, 1), bias=False),
        ]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_channel))
        if use_relu:
            layers.append(nn.ReLU(inplace=True))
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.layers(x)
        return x

class netvlad_2d(nn.Module):
    def __init__(self, core='euc', **core_kwargs):
        super().__init__()
        cores = {
            "euc": vlad_core_euc,
            "cos": vlad_core_cos
        }
        self.core = core
        self.vlad = cores[core](**core_kwargs)
        self.output_size = self.vlad.output_size

    def get_assign(self, detach=True, cpu=True, tonp=True):
        # process data first.
        if tonp is True:
            detach, cpu = True, True
        assign = self.vlad.assign # b * hw * c

        if detach:
            assign = assign.detach()
        if cpu:
            assign = assign.cpu()
        
        b, _, c = assign.shape
        h, w = self.size
        assign = assign.view(b, h, w, c)
        assign = assign.permute(0, 3, 1, 2)

        if tonp:
            assign = assign.numpy()
        
        return assign
        

    def forward(self, x, raw_x = None):
        # x.shape = b * c * h * w
        self.x = x # for backup and get featuremap.
        b, c, h, w = x.shape
        self.size = (h, w) # record feature map size incase you need it.
        x = x.view(b, c, -1).permute(0, 2, 1) # b * hw * c
        x = self.vlad(x)
        self.bg = self.vlad.bg
        return x

class fakevlad(nn.Module):
    def __init__(self, in_channels, use_bn=True, fg_num=32, bg_num=0, dim=64, alpha=10.0):
        super().__init__()
        self.output_size = fg_num * dim
        self.weight_reg = Bottleneck(in_channels, fg_num)
        self.alpha = alpha
    
    def get_assign(self, detach=True, cpu=True, tonp=True):
        assign = self.assign # b * c * n
        b, c, n = assign.shape
        assign = assign.reshape(b, c, *self.size) # b * c * h * w

        if detach:
            assign = assign.detach()
        if cpu:
            assign = assign.cpu()
        if tonp:
            assign = assign.numpy()
        return assign

    def forward(self, x, x_ori):
        # c: channel | n: num_maps.
        # x.shape = b * c * h * w
        b, c, h, w = x.shape
        self.size = (h, w)
        weights = self.weight_reg(x_ori) # b * n * h * w
        # print('x.shape =', x.shape)
        x = x.view(*x.shape[:2], -1) # b * c * hw
        weights = weights.view(*weights.shape[:2], -1) # b * n * hw
        weights = torch_func.softmax(weights * self.alpha, dim=1) # b * n * hw
        self.assign = weights
        x = x.permute(0, 2, 1).contiguous() # b * hw * c
        x = torch_func.normalize(x, p=2, dim=2)
        x = weights.bmm(x) # b * n * c
        x = torch_func.normalize(x, p=2, dim=2)
        x = x.view(b, -1) # b * nc
        return x

class vlad_wrap(nn.Module):
    def __init__(self, in_channels=None, use_bn=True, core='euc', **core_kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.downsample = None
        dim = core_kwargs['dim']
        if in_channels != dim:
            self.downsample = nn.Conv2d(in_channels, dim, kernel_size=(1, 1), bias=False)
        
        if core in ['cos', 'euc']:
            self.vlad = netvlad_2d(core=core, **core_kwargs)
        elif core == 'fake':
            self.vlad = fakevlad(in_channels=in_channels, **core_kwargs)

        self.bn = None
        if use_bn:
            self.bn = nn.BatchNorm1d(self.vlad.output_size, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.output_size = self.vlad.output_size
    def get_assign(self, *args, **kwargs):
        return self.vlad.get_assign(*args, **kwargs)
    def get_features(self):
        return self.features
    def forward(self, x):
        x_ori = x
        if self.downsample is not None:
            x = self.downsample(x)
        self.features = x
        x = self.vlad(x, x_ori)
        if self.bn is not None:
            x = self.bn(x)
        return x

class multi_vlad_wrap(nn.Module):
    def __init__(self, in_channels:list=None, use_bn=True, core='euc', fg_num=32, bg_num=0, dim=64, alpha=10.0):
        super().__init__()
        cores = {
            "euc": vlad_core_euc,
            "cos": vlad_core_cos
        }
        # process in_channels.
        self.downsamples = None
        if not all([channel == dim for channel in in_channels]):
            self.downsamples = nn.ModuleList()
            for channel in in_channels:
                self.downsamples.append(nn.Conv2d(channel, dim, kernel_size=(1, 1), bias=False))
        self.vlad = cores[core](fg_num=fg_num, bg_num=bg_num, dim=dim, alpha=alpha) # takes b * n * d
        if use_bn:
            self.bn = nn.BatchNorm1d(self.vlad.output_size, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.output_size = self.vlad.output_size
    def forward(self, features):
        if self.downsamples is not None:
            downsampled_features = []
            for feature, downsample in zip(features, self.downsamples):
                feature = downsample(feature)
                downsampled_features.append(feature)
            features = downsampled_features
        # features.shape: [b * dim * h * w] * num_features
        # reshape.
        features_1d = []
        for feature in features:
            b, d, h, w = feature.shape
            feature = feature.view(b, d, -1).permute(0, 2, 1).contiguous() # b * n * d
            features_1d.append(feature)
        features_1d = torch.cat(features_1d, dim=1) # b * (n1 + n2 +...) * d
        features = self.vlad(features_1d)
        if self.bn is not None:
            features = self.bn(features)
        return features
