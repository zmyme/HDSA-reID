from copy import deepcopy
import torch
import torch.nn as nn
from torch.nn.modules.linear import Linear
from torchvision import models
import torch.nn.functional as torch_func

import netvlad
import hrnet

from utils import BaseModule, filt_modules

"""
The defination of an reid model:
image -> backbone -> heads -> final_preds
"""

def remove_conv_stride(net):
    conv2ds = filt_modules(net, 'Conv2d')
    for conv in conv2ds:
        conv.stride = (1, 1)

class BaseModule(nn.Module):
    def __init__(self):
        super(BaseModule, self).__init__()
        self.enable_summary = True

class EmptyModule(nn.Module):
    def __init__(self) -> None:
        super(EmptyModule, self).__init__()
    def forward(self, x):
        return x

class backbone_shufflenet(BaseModule):
    def __init__(self, pretrained=True, **kwargs):
        super().__init__()
        self.pretrained = pretrained
        self.model = models.shufflenet_v2_x0_5(pretrained=pretrained)
        self.out_channels = [1024]
        self.model.stage4[0].branch1[0] = nn.Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=96, bias=False)
        self.model.stage4[0].branch2[3] = nn.Conv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=96, bias=False)
    
    def get_frozen_layers(self):
        return [self.model.conv1]
    
    def __repr__(self):
        info = '{class_name}(pretrained={0})'.format(
            self.pretrained,
            class_name = self.__class__.__name__
        )
        return info
    
    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.maxpool(x)
        x = self.model.stage2(x)
        x = self.model.stage3(x)
        x = self.model.stage4(x)
        x = self.model.conv5(x)
        return [x]

class backbone_dense121(BaseModule):
    def __init__(self, pretrained=True, **kwargs):
        super().__init__()
        self.pretrained = pretrained
        self.model = models.densenet121(pretrained=pretrained).features
        self.model.transition3.pool = EmptyModule()
        self.out_channels = [1024]
    
    def get_frozen_layers(self):
        frozen_layers = [
            self.model.conv0,
            self.model.norm0,
            self.model.relu0,
            self.model.pool0,
            self.model.denseblock1,
            self.model.transition1,
        ]
        return frozen_layers
    
    def __repr__(self):
        info = '{class_name}(pretrained={0})'.format(
            self.pretrained,
            class_name = self.__class__.__name__
        )
        return info
    
    def forward(self, x):
        x = self.model.conv0(x)
        x = self.model.norm0(x)
        x = self.model.relu0(x)
        x = self.model.pool0(x)
        x = self.model.denseblock1(x)
        x = self.model.transition1(x)
        x = self.model.denseblock2(x)
        x = self.model.transition2(x)
        x = self.model.denseblock3(x)
        x = self.model.transition3(x)
        x = self.model.denseblock4(x)
        x = self.model.norm5(x)
        return [x]

class backbone_hrnet(BaseModule):
    def __init__(self, name='w32c', features=4, remove_ds=4, pretrained=True, cat=False) -> None:
        super().__init__()
        self.name = name
        self.features = features
        self.remove_ds = remove_ds
        self.pretrained = pretrained
        self.cat = cat
        self.base = hrnet.get_cls_net(name=name, pretrain=pretrained)
        print('cat =', cat)
        out_channels = [128, 256, 512, 1024]
        self.out_channels = [out_channels[f-1] for f in self.features]
        if cat:
            if cat == 'raw':
                self.out_channels = [64]
                self.upsamples = nn.ModuleList([
                    nn.Sequential(nn.Conv2d(128, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)),
                    nn.Sequential(nn.UpsamplingBilinear2d(scale_factor=2), nn.Conv2d(256, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)),
                    nn.Sequential(nn.UpsamplingBilinear2d(scale_factor=4), nn.Conv2d(512, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)),
                    nn.Sequential(nn.UpsamplingBilinear2d(scale_factor=8), nn.Conv2d(1024, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)),
                ])
            elif cat == 'fuse':
                self.out_channels = [128 * 4]
                self.upsamples = nn.ModuleList([
                    nn.Sequential(nn.UpsamplingBilinear2d(scale_factor=2), nn.Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)),
                    nn.Sequential(nn.UpsamplingBilinear2d(scale_factor=4), nn.Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)),
                    nn.Sequential(nn.UpsamplingBilinear2d(scale_factor=8), nn.Conv2d(1024, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)),
                ])
            elif cat == 'direct':
                self.out_channels = [sum(out_channels)]
                self.upsamples = nn.ModuleList([
                    nn.UpsamplingBilinear2d(scale_factor=2),
                    nn.UpsamplingBilinear2d(scale_factor=4),
                    nn.UpsamplingBilinear2d(scale_factor=8),
                ])
            else:
                raise NotImplementedError('Concatenation Mode', cat, 'is not Impelenmented.')

    def __repr__(self):
        info = '{class_name}(name={0}, features={1}, remove_ds={2}, pretrained={3})'.format(
            self.name,
            self.features,
            self.remove_ds,
            self.pretrained,
            class_name = self.__class__.__name__
        )
        return info
    def get_frozen_layers(self):
        frozen_layers = [
            self.base.conv1,
            self.base.bn1,
            self.base.relu,
            self.base.conv2,
            self.base.bn2,
            self.base.relu,
        ]
        return frozen_layers
    def forward(self, x):
        outputs = self.base(x)
        if self.cat == 'raw':
            outputs[0] = self.upsamples[0](outputs[0])
            outputs[1] = self.upsamples[1](outputs[1])
            outputs[2] = self.upsamples[2](outputs[2])
            outputs[3] = self.upsamples[3](outputs[3])
            outputs = [torch.cat(outputs, dim=1)]
        elif self.cat == 'fuse':
            outputs[1] = self.upsamples[0](outputs[1])
            outputs[2] = self.upsamples[1](outputs[2])
            outputs[3] = self.upsamples[2](outputs[3])
            outputs = [torch.cat(outputs, dim=1)]
        elif self.cat == 'direct':
            outputs[1] = self.upsamples[0](outputs[1])
            outputs[2] = self.upsamples[1](outputs[2])
            outputs[3] = self.upsamples[2](outputs[3])
            outputs = [torch.cat(outputs, dim=1)]
        else:
            # process features.
            new_outputs = []
            for feature in self.features:
                new_outputs.append(outputs[feature - 1])
            outputs = new_outputs
        return outputs


class backbone_resnet(BaseModule):
    def __init__(self, name='resnet50', features=4, remove_ds=4, pretrained=True, cat=False) -> None:
        super().__init__()
        self.name = name
        self.features = features
        self.remove_ds = remove_ds
        self.pretrained = pretrained
        self.cat = cat
        avaliable_names = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
        if name not in avaliable_names:
            raise ValueError('model name for resnet can only be choosen from [{0}]'.format('|'.join(avaliable_names)))
        # get model.
        model = getattr(models, name)(pretrained=pretrained)
        # construct layers
        layer0 = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool
        )
        layers = [layer0, model.layer1, model.layer2, model.layer3, model.layer4]
        # remove downsamples.
        if remove_ds is not None:
            if type(remove_ds) is not list:
                remove_ds = [remove_ds]
            for layer in remove_ds:
                remove_conv_stride(layers[layer])
        # convert reserved layers
        self.single = False
        if type(features) is not list:
            features = [features]
            self.single = True
        self.features = features
        max_layer = max(features)
        self.layers = nn.ModuleList(layers[:max_layer + 1])
        # build output channels.
        channels_config = {
            "resnet18": [64, 64, 128, 256, 512],
            "resnet34": [64, 64, 128, 256, 512],
            "resnet50": [64, 256, 512, 1024, 2048],
            "resnet101": [64, 256, 512, 1024, 2048],
            "resnet152": [64, 256, 512, 1024, 2048],
        }
        channels = channels_config[name]
        channels = [channels[layer] for layer in self.features]
        if self.single:
            channels = channels[0]
        self.out_channels = channels

        if cat == 'fuse':
            self.out_channels = [256 * 4]
            self.upsamples = nn.ModuleList([
                nn.Sequential(nn.UpsamplingBilinear2d(scale_factor=2), nn.Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)),
                nn.Sequential(nn.UpsamplingBilinear2d(scale_factor=4), nn.Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)),
                nn.Sequential(nn.UpsamplingBilinear2d(scale_factor=4), nn.Conv2d(2048, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)),
            ])
    
    def __repr__(self):
        info = '{class_name}(name={0}, features={1}, remove_ds={2}, pretrained={3})'.format(
            self.name,
            self.features,
            self.remove_ds,
            self.pretrained,
            class_name = self.__class__.__name__
        )
        return info
    def get_frozen_layers(self):
        return self.layers[0], self.layers[1]
    def forward(self, x):
        features = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i in self.features:
                features.append(x)
        if self.cat == 'fuse':
            h, w = features[0].shape[2:]
            features[1] = self.upsamples[0](features[1])
            features[2] = self.upsamples[1](features[2])
            features[3] = self.upsamples[2](features[3])
            features = [torch.cat(features, dim=1)]
        if self.single:
            features = features[0]
        return features

class GroupLinear(BaseModule):
    def __init__(self, in_features:int, out_features:int, n_groups:int, mode:str='shared', bias=False, flatten=False) -> None:
        """
        acts like batched linear
        Input:
        - tensor, shape=b * d | b * n * d.
            - shape = b * d: b=batchsize, d=in_features*n_groups
            - shape = b * n * d: b=batchsize, n=n_groups, d=in_features
        Args:
        - mode:str, either be shared or independent
        - bias: whether use bias or not.
        - flatten: flatten output to b * nd
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_groups = n_groups
        self.mode = mode
        self.flatten = flatten
        self.bias = bias
        if mode == 'shared':
            self.linear = nn.Linear(in_features, out_features, bias)
        elif mode == 'independent':
            self.linear = nn.ModuleList()
            for _ in range(n_groups):
                self.linear.append(nn.Linear(in_features, out_features, bias))
    
    def __repr__(self) -> str:
        info = '{name}(in_features={0}, out_features={1}, n_groups={2}, mode={3}, bias={4}, flatten={5})'.format(
            self.in_features, self.out_features, self.n_groups, self.mode, self.bias, self.flatten, name=self.__class__.__name__
        )
        return info
    
    def forward(self, x):
        b = x.shape[0]
        if len(x.shape) == 2:
            # x.shape = b * nd
            x = x.view(b, self.n_groups, self.in_features) # b * n * d
        elif len(x.shape) != 3:
            raise ValueError('x.shape is expected for b * d or b * n * d')
        # now, x.shape = b * n_groups * in_features
        outputs = []
        if self.mode == 'shared':
            for i in range(self.n_groups):
                outputs.append(self.linear(x[:, i, :]))
        elif self.mode == 'independent':
            for i in range(self.n_groups):
                outputs.append(self.linear[i](x[:, i, :]))
        outputs = torch.stack(outputs, dim=1) # b * n * out_features
        if self.flatten:
            outputs = outputs.view(b, -1)
        return outputs

class PCBWrapper(BaseModule):
    def __init__(self) -> None:
        super().__init__()
    def forward(self, x):
        # x.shape  = b * c * h * w
        b, c, h, w = x.shape
        x = x.view(-1, c, 6, h//6, w).permute(2, 0, 1, 3, 4) # 6 * b * 2048 * 4 * 8
        x = x.contiguous() # 6 * b * 2048 * (4*8)
        # print('x.shape =', x.shape)
        x = list(x)
        return x

class PCBPooling(BaseModule):
    def __init__(self, in_channel, strip=6) -> None:
        super().__init__()
        self.in_channel = in_channel
        self.strip = strip
        self.output_size = [in_channel] * self.strip
    def forward(self, x):
        # x.shape  = b * c * h * w
        b, c, h, w = x.shape
        x = x.view(-1, c, self.strip, h//self.strip, w).permute(2, 0, 1, 3, 4) # 6 * b * 2048 * 4 * 8
        x = x.view(*x.shape[:-2], -1)
        x = x.mean(dim=-1)
        x = list(x)
        return x

class AvgPooling(BaseModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
    def forward(self, x):
        x = self.pool(x)
        x = x.view(x.shape[0], -1)
        return x

# g = GroupLinear()

class MultiFeatureHead(BaseModule):
    def __init__(self, in_channel, pooling:dict=None, transform:dict=None, classifier:dict=None, **kwargs) -> None:
        super().__init__()
        self.in_channel = in_channel
        # prepare pooling.
        self.pooling = None
        pooling_size = None
        if type(pooling) is str:
            pooling = {
                "name": pooling, 
            }
        pooling_name = pooling.pop('name')
        pooling_kwargs = pooling
        if pooling_name == 'avg':
            self.pooling = AvgPooling()
            pooling_size = in_channel
        elif pooling_name == 'vlad':
            self.pooling = netvlad.vlad_wrap(in_channel, **pooling_kwargs)
            pooling_size = self.pooling.output_size
        elif pooling_name == 'pcb':
            self.pooling = PCBPooling(in_channel, strip=pooling_kwargs.get('strip', 6))
            pooling_size = self.pooling.output_size
        else:
            raise ValueError('Unknown pooling method.')
        # prepare transform.
        if transform is None:
            transform = {}
        transtype = transform.get('type', None)
        transdim = transform.get('dim', 2048)
        self.transform = None
        if transtype is not None:
            if type(pooling_size) is list:
                self.transform = nn.ModuleList()
                for size in pooling_size:
                    self.transform.append(nn.Sequential(
                        nn.Linear(size, transdim),
                        nn.BatchNorm1d(transdim, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
                    ))
                pooling_size = [transdim] * len(pooling_size)
            else:
                self.transform = nn.Sequential(
                    nn.Linear(pooling_size, transdim),
                    nn.BatchNorm1d(transdim, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
                )
                pooling_size = transdim
        # prepare classifier.
        if classifier is None:
            classifier = {}
        classnum = classifier.get('classnum')
        if type(pooling_size) is list:
            self.classifier = nn.ModuleList()
            for size in pooling_size:
                self.classifier.append(nn.Sequential(
                    nn.ReLU(),
                    nn.Linear(size, classnum)
                ))
        else:
            self.classifier = nn.Sequential(
                nn.ReLU(),
                nn.Linear(size, classnum)
            )
    def forward(self, x):
        x = self.pooling(x)
        if type(x) is list:
            if self.transform is not None:
                x = [trans(feat) for feat, trans in zip(x, self.transform)]
            if self.training:
                x = [classifier(feat) for feat, classifier in zip(x, self.classifier)]
            else:
                x = torch.cat(x, dim=1)
        else:
            if self.transform is not None:
                x = self.transform(x)
            if self.training:
                x = self.classifier(x)
        if self.training:
            return x
        else:
            return {'trans': x}

class SimpleHead(BaseModule):
    """
    pooling -> linear -> classifier
    conf = {
        "in_channel": int,
        "pooling": {
            "name"： str,
            **kwargs
        },
        "linear": None | out_featurs:int
    }
    """
    def __init__(self, in_channel, pooling:dict=None, transform:dict=None, classifier:dict=None, triplet:bool=False, test_kind=None):
        super().__init__()
        self.in_channel = in_channel
        """
        build pooling layer.
        """
        self.pooling = None
        pooling_size = None
        if type(pooling) is str:
            pooling = {
                "name": pooling, 
            }
        pooling_name = pooling.pop('name')
        pooling_kwargs = pooling
        if pooling_name == 'avg':
            self.pooling = nn.AdaptiveAvgPool2d((1, 1))
            pooling_size = in_channel
        elif pooling_name == 'vlad':
            self.pooling = netvlad.vlad_wrap(in_channel, **pooling_kwargs)
            pooling_size = self.pooling.output_size
        elif pooling_name == 'fvlad':
            self.pooling = netvlad.multi_vlad_wrap(in_channel, **pooling_kwargs)
            pooling_size = self.pooling.output_size
        else:
            raise ValueError('Unknown pooling method.')
        """
        build transform layer.
        a transform layer must takes a input from the pooling layer, and output a feature of specified feature length of "transform"
        """
        self.transform = None
        if transform is None:
            transform = {}
        transtype = transform.get('type', None)
        transdim = transform.get('dim', 2048)
        if transtype is not None:
            translayer = None
            if transtype == 'linear':
                translayer = nn.Linear(pooling_size, transdim)
            elif transtype in ['group_indp', 'group_shared']:
                mode = {
                    "group_indp": "independent",
                    "group_shared": "shared"
                }[transtype]
                fg_num = pooling_kwargs['fg_num']
                translayer = GroupLinear(pooling_size//fg_num, transdim//fg_num, fg_num, mode=mode, flatten=True)
            self.transform = nn.Sequential(
                translayer,
                nn.BatchNorm1d(transdim, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
            )
        """
        build classifier.
        """
        if classifier is None:
            classifier = {}
        classnum = classifier.get('classnum')
        clsfeatures = classifier.get('feature', {}) # should be a list.
        self.clsfeatures = clsfeatures
        features_size = {
            'pool': pooling_size,
            'avgpool': self.in_channel,
            'trans': transdim
        }
        self.classifiers = nn.ModuleDict()
        self.cls_sources = {}
        enabled = clsfeatures.pop('enabled', 'trans').split(',')
        for name in enabled:
            fconf = clsfeatures[name]
            src = fconf.get('src', 'trans')
            use_relu = fconf.get('relu', True)
            clstype = fconf.get('type', 'linear')
            layers = []
            if use_relu:
                layers.append(nn.ReLU())
            if clstype == 'linear':
                layers.append(nn.Linear(features_size[name], classnum))
            elif clstype == 'group':
                dim = pooling_kwargs.get('dim', 64)
                clusters = pooling_kwargs.get('fg_num', 32)
                # layers.append(GroupLinear(dim, dim, clusters, mode='independent', bias=False, flatten=True))
                # layers.append(nn.ReLU())
                layers.append(GroupLinear(dim, classnum, clusters, mode='independent', bias=False, flatten=False))
                # this would output b * clusters * class_num
            else:
                raise ValueError('Unrecognized classifier type {0}'.format(type))
            classifier = nn.Sequential(*layers)
            self.classifiers[name] = classifier
            self.cls_sources[name] = src
        self.triplet = triplet
        self.test_kind = test_kind.split(',')
    
    def forward(self, x): # x.shape = b * c * h * w
        features_avgpool = x.view(*x.shape[:2], -1).mean(dim=-1)
        x = self.pooling(x)
        x = x.view(x.shape[0], -1) # b * d
        features_pooled = x
        if self.transform is not None:
            x = self.transform(x)
        features_transformed = x
        avaliable_features = {
            "avgpool": features_avgpool,
            "pool": features_pooled,
            "trans": features_transformed,
        }
        
        scores = []
        if self.training:
            for name in self.classifiers:
                feature = avaliable_features[self.cls_sources[name]]
                classifier = self.classifiers[name]
                score = classifier(feature)
                scores.append(score)
                avaliable_features['score_{0}'.format(name)] = score

        if self.triplet is True and self.training:
            return scores, features_transformed
        elif not self.training:
            features = {key:avaliable_features[key] for key in self.test_kind}
            return features
        else:
            return scores

class ReIDSimpleModel(BaseModule):
    def __init__(self, backbone:dict = None, head:dict=None, fuse=None, pcb=False) -> None:
        """
        fuse: reuse | new. which features to use. reuse will collect features from previous vlads, while new use the new one.
        """
        super().__init__()
        # create backbone
        if backbone is None:
            backbone = {}
        backbones = {
            "dense121": backbone_dense121,
            "shufflenet": backbone_shufflenet,
            "hrnet": backbone_hrnet,
            "resnet": backbone_resnet
        }
        backbone_types = {
            "dense121": "dense121",
            "shufflenet": "shufflenet"
        }
        backbone_type = None
        backbone_name = backbone.get('name', 'w32c')
        if backbone_name in backbone_types:
            backbone_type = backbone_types[backbone_name]
        else:
            backbone_type = 'resnet' if backbone.get('name', 'w32c').startswith('resnet') else 'hrnet'
        backbone_class = backbones[backbone_type]
        features = backbone.get('features', 4)
        if type(features) is not list:
            features = [features]
            backbone['features'] = features
        self.backbone = backbone_class(**backbone)
        channel_nums = self.backbone.out_channels
        self.pcb = None
        if pcb:
            self.pcb = PCBWrapper()
            channel_nums = channel_nums * 6

        # create heads for each featuremap.
        if head is None:
            head = {}
        headtype = head.pop('type', 'simple')
        head_meta = {
            "simple": SimpleHead,
            "mf": MultiFeatureHead
        }[headtype]
        self.heads = nn.ModuleList()
        for channel in channel_nums:
            this_head = head_meta(channel, **deepcopy(head))
            self.heads.append(this_head)
        self.num_outputs = len(channel_nums)
        # if you want a fused head, this head will feed all the heatmaps.
        self.fuse_head = None
        self.fuse = fuse
        if fuse is not None:
            this_head = deepcopy(head)
            this_head['pooling']['name'] = 'fvlad'
            in_channels = channel_nums if fuse == 'new' else [this_head['pooling']['dim']] * len(self.backbone.out_channels)
            self.fuse_head = SimpleHead(in_channels, **this_head)
            self.num_outputs += 1
            self.vlad_blocks = filt_modules(self.heads, 'vlad_wrap')
        self.triplet = head.get('triplet', False)
    
    def forward(self, x):
        # backbone forward
        features = self.backbone(x)
        self.features = features
        if self.pcb is not None:
            if len(features) != 1:
                raise ValueError('PCB only takes single feature map!')
            features = self.pcb(features[0])
        # process each head.
        outputs = []
        for feature, head in zip(features, self.heads):
            output = head(feature)
            outputs.append(output)
        # process fuse head
        if self.fuse_head is not None:
            if self.fuse == 'reuse':
                features = [vlad.features for vlad in self.vlad_blocks]
            fused_output = self.fuse_head(features)
            outputs.append(fused_output)
        return outputs
