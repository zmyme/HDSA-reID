from copy import deepcopy
from typing import Any, Union
import torch
import torch.nn as nn
import os
import utils

from model import ReIDSimpleModel
from dataset import ReIDDataSet
import loss


# args handles args that tou want to pass. either be a None or list of same length as names
def build_objects(meta:Any, conf:dict, names:list, check_nonexist:bool=True):
    common = conf.get('_common', {})
    objects = []
    for name in names:
        special = conf.get(name, {})
        if check_nonexist and name not in conf:
            msg = 'Warning: specical conf not found, fallback using common configuration\n'
            msg += '    Note: if you want to remove this warning message, add an empty configuration or set check_nonexist to False'
            print(msg)
        args = special.pop('_args', [])
        kwargs = deepcopy(common)
        kwargs.update(special)
        this_obj = meta(*args, **kwargs)
        objects.append(this_obj)
    return objects

def get_dataloader(conf, names='train'):
    # process names.
    single = True if type(names) is str else False
    if type(names) is not list:
        names = [names]

    # extract conf area.
    conf = conf['dataconf']
    setconf = conf['set']
    loaderconf = conf['loader']

    # create dataset.
    datasets = build_objects(ReIDDataSet, setconf, names, check_nonexist=True)
    
    # create dataloader.
    for name, dataset in zip(names, datasets):
        if name not in loaderconf:
            loaderconf[name] = {}
        loaderconf[name]['_args'] = [dataset]
    dataloaders = build_objects(torch.utils.data.DataLoader, loaderconf, names, check_nonexist=False)
    if single:
        dataloaders = dataloaders[0]
    return dataloaders

def resume(net:nn.Module, conf):
    def remove_params_from_state_dict(state_dict, filters):
        def remove_keyword(key, keyword=None):
            if keyword in key:
                return True
            else:
                return False
        # convert filters.
        if type(filters) is not list:
            filters = [filters]
        for i, f in enumerate(filters):
            if type(f) is str:
                f = lambda key,keyword=f:remove_keyword(key,keyword)
                filters[i] = f
            elif callable(f):
                pass
            else:
                raise TypeError('filter must be a callable or str, instead of', type(f))
        new_state_dict = {}
        for key in state_dict:
            remove = any([f(key) for f in filters])
            if not remove:
                new_state_dict[key] = state_dict[key]
            else:
                print('removing key', key)
        return new_state_dict
    if conf['resume'] is None:
        return net
    strict = conf.get('strict_resume', True)
    no_fc_classifier = conf.get('no_fc_classifier', False)
    resume_dir = conf['paths']['checkpoint']
    resume_file = conf['resume']
    resume_path = os.path.join(resume_dir, resume_file)
    state_dict = torch.load(resume_path)
    keywords2remove = []

    if no_fc_classifier:
        keywords2remove.append('transform')
        keywords2remove.append('pooling.bn')
        strict = False
    if not strict:
        keywords2remove.append('classifiers')
    
    state_dict = remove_params_from_state_dict(state_dict, keywords2remove)
    print('Loading model from {0} with strict = {1}'.format(resume_path, strict))
    net.load_state_dict(state_dict, strict=strict)
    
    return net


def get_network(conf):
    modelconf = conf['model']
    net = ReIDSimpleModel(**modelconf)
    supress = conf.get('supress', None)
    if supress is not None:
        vlads = utils.filt_modules(net, 'vlad_core_euc')
        for vlad, sup in zip(vlads, supress):
            vlad.supress = sup

    net = net.cuda()
    net = resume(net, conf)
    if conf['training_mode']:
        print(net)
    print('num_params:', utils.count_params(net))
    resources = conf.get('resources', {})
    resources['net'] = net
    conf['resources'] = resources
    return net

def get_criterion(conf):
    criterion = loss.ReIDLoss(conf['resources']['net'], **conf['loss'])
    conf['resources']['criterion'] = criterion
    return criterion


if __name__ == '__main__':
    from config import conf
    dataloader = get_dataloader(conf, 'train')
    for data in dataloader:
        imgs, labels, cameras, paths = data
        print('labels:', labels)
        print('cameras', cameras)
        # print('labels:', '|'.join(labels.numpy().tolist()))
        # print('cameras', '|'.join(cameras.numpy().tolist()))
        print('paths:')
        print('\n'.join(paths))
        input('>>> ')