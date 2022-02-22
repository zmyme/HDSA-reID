from collections import defaultdict
import numpy as np
from utils import LineDisplayer, save_arrays
import torch
import torch.nn.functional as torch_func
from resources import get_dataloader, get_network
import time
from utils import filt_modules
import torch.nn as nn

def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long().cuda()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip


def cos_simlarity(vec1, vec2):
    """
    vec.shape = [b, dim]
    out.shape = [b]
    """
    vec1 = torch_func.normalize(vec1, p=2, dim=1)
    vec2 = torch_func.normalize(vec2, p=2, dim=1)
    cos_sim = (vec1 * vec2).sum(dim=1)
    return cos_sim # b

def collect_features(outputs):
    """
    outputs: list of {key:value(shape=[b, dim])}
    retuen: {key:[len = num_features, value=[len=b, value=single feature]]}
    """
    collected = defaultdict(list)
    for output in outputs:
        for key in output:
            collected[key].append(list(output[key].detach().cpu().numpy()))
    return collected

def mean_network_outputs(output1, output2):
    outputs = []
    beliefs = []
    for out1, out2 in zip(output1, output2):
        out = {key:(out1[key] + out2[key])/2.0 for key in out1}
        belief = {key:cos_simlarity(out1[key], out2[key]) for key in out1}
        outputs.append(out)
        beliefs.append(belief)
    return collect_features(outputs), collect_features(beliefs)



def calculate_single(net, dataloader, with_assign=False):
    # prepare necessary resources
    displayer = LineDisplayer()
    start_time = time.time()
    num_features = net.num_outputs
    vlads = filt_modules(net, 'vlad_wrap')
    # values to record.
    def generate_feature_structure(num=num_features):
        return [[] for _ in range(num)]
    features = defaultdict(generate_feature_structure)
    beliefs = defaultdict(generate_feature_structure)
    assigns = generate_feature_structure() # single assign.shape = b * c * h * w
    labels = []
    cameras = []
    paths = []

    finished = 0
    with torch.no_grad():
        net = net.train(False)
        for data in dataloader:
            image, label, camera, path = data
            image = image.cuda().float()

            # record labels, cameras and paths.
            labels += label.numpy().tolist()
            cameras += camera.numpy().tolist()
            paths += list(path)

            # calculate image features
            image_flipped = fliplr(image)
            out1 = net(image)
            # obtain netvlad assigns.
            if with_assign:
                for i, vlad in enumerate(vlads):
                    assign = vlad.get_assign(tonp=True) # b * c * h * w
                    assign = list(assign)
                    assigns[i] += assign
            out2 = net(image_flipped)
            this_features, this_beliefs = mean_network_outputs(out1, out2)

            for key in this_features: # for different features name.
                for i, (cur_f, cur_belief) in enumerate(zip(this_features[key], this_beliefs[key])): # for different heads.
                    features[key][i] += cur_f
                    beliefs[key][i] += cur_belief
            # update tracking status.
            finished += image.shape[0]
            displayer.disp('Progress:', str(finished))
    end_time = time.time()
    displayer.disp('Finished in {0} s.'.format(end_time - start_time))
    displayer.newline()
    
    # convert types.
    features = {key:[np.asarray(o, dtype=np.float32) for o in features[key]] for key in features}
    beliefs = {key:[np.asarray(o, dtype=np.float32).reshape(1, -1) for o in beliefs[key]] for key in beliefs}
    labels = np.asarray(labels, dtype=np.int32).reshape(1, -1)
    cameras = np.asarray(cameras, dtype=np.int32).reshape(1, -1)
    if with_assign:
        assigns = [np.asarray(a) for a in assigns]

    mat = {}
    for key in features:
        for i, f in enumerate(features[key]):
            mat['feature_{name}_{idx}'.format(name=key, idx=i)] = f
    for key in beliefs:
        for i, b in enumerate(beliefs[key]):    
            mat['belief_{name}_{idx}'.format(name=key, idx=i)] = b

    if with_assign:
        for i, a in enumerate(assigns):
            mat['assign_{0}'.format(i)] = a
    mat['labels'] = labels
    mat['cameras'] = cameras
    mat['paths'] = paths
    # print summary info.
    for key, value in mat.items():
        this_type = type(value)
        shape = None
        length = None
        if hasattr(value, 'shape'):
            shape = value.shape
        elif hasattr(value, '__len__'):
            length = len(value)
        info = '{0}: {1},'.format(key, this_type)
        if shape is not None:
            info += ' shape={0}'.format(shape)
        elif length is not None:
            info += ' length={0}'.format(length)
        print(info)
    return mat

def calculate_all(net:nn.Module, conf:dict)->list:
    """
    calculate fetaures in loaders.
    requires the following config region:
    conf = {
        "calf": {
            "loaders": "query,gallery", # which loader to calculate, seperated by ','
            "assign": False # whether calculate assign
        }
    }
    return: a list of the calculated map.
    """
    loader_string = conf['calf']['loaders']
    with_assign = conf['calf']['assign']
    names = loader_string.split(',')
    flag = conf.get('flag', None)

    mats = []
    dataloaders = get_dataloader(conf, names)
    for name, loader in zip(names, dataloaders):
        mat = calculate_single(net, loader, with_assign=with_assign)
        mats.append(mat)
        if flag is not None:
            name = flag + '_' + name
        save_arrays(conf['paths']['params'] + '/{0}'.format(name), mat, backend='numpy', hint=True)
    return mats

if __name__ == '__main__':
    from config import conf
    net = get_network(conf)
    calculate_all(net, conf)
