import os
import sys
import time

from gpuutil import auto_set

from logger import logger
import utils
from configuration import ConfigurationParser

from utils import set_benchmark, set_deterministic

# /data/data3/zhangmingyang/reid
server_conf = {
    # "root": "path/to/your/dataset/root"
    "root": "/data/data3/zhangmingyang/reid"
}

def sync_from(path):
    def sync(conf, path=path):
        conf = utils.DictAdvancedAccess(conf)
        return conf[path]
    return sync

dataset_defaults = {
    'cuhk03np-l': {
        'dirname': 'cuhk03-np-labeled',
        'classnum': 767,
        'wd': 5e-3
    },
    'cuhk03np-d': {
        'dirname': 'cuhk03-np-detected',
        'classnum': 767,
        'wd': 5e-3
    },
    'market1501': {
        'dirname': 'market1501',
        'classnum': 751,
        'wd': 2e-3
    },
    'dukemtmc': {
        'dirname': 'dukemtmc',
        'classnum': 702,
        'wd': 2e-3
    },
    'msmt17': {
        'dirname': 'msmt17_v2',
        'classnum': 1041,
        'wd': 2e-3
    },
    'huawei': {
        'dirname': 'huawei',
        'classnum': 1041,
        'wd': 2e-3
    },
    "occduke": {
        'dirname': 'occluded_duke',
        'classnum': 702,
        'wd': 2e-3
    },
    "occreid": {
        'dirname': 'occluded_reid',
        'classnum': 200,
        'wd': 2e-3,
    },
    "occreidsp": {
        'dirname': 'occluded_reid',
        'classnum': 100,
        'wd': 5e-3,
        "list": {
            "train": "list/train_sp.json",
            "query": "list/query_sp.json",
            "gallery": "list/gallery_sp.json"
        }
    }
}

conf = {
    "dataset": "cuhk03np-d",
    "imgsize": (128, 384),
    "gpu": "auto:1",
    "seed": None,
    "num_epoch": 60,
    "root": None,
    "main_name": sys.argv[0],
    "raw_args": ' '.join(sys.argv),
    "resume": None,
    "flag": None,
    "summary": 'default',
    "paths": { # autobackup  checkpoint  list  logs  original  params  samples  saved_model  visual
        "autobackup": None,
        "checkpoint": None,
        "list": None,
        "logs": None,
        "original": None,
        "params": None,
        "samples": None,
        "saved_model": None,
        "visual": None
    },
    "dataconf": {
        "set": {
            "_common": {
                "dataset_root": None,
                "img_root": "original",
                "list_path": None,
                "img_size": (128, 384), # w * h
                "random_filp": False,
                "random_erase": False,
                "random_crop": False,
                "color_jitter": False,
            },
            "train": {
                "list_path": "list/train_all.json",
                "random_erase": True,
                "random_filp": True
            },
            "val": {
                "list_path": "list/train_part2.json",
            },
            "query": {
                "list_path": "list/query.json",
            },
            "gallery": {
                "list_path": "list/gallery.json",
            }
        },
        "loader": {
            "_common": {
                "num_workers": 8,
                "batch_size": 32,
                "shuffle": False,
                "drop_last": False
            },
            "train": {
                "shuffle": True,
                "drop_last": True
            },
            "val": {},
            "query": {},
            "gallery": {}
        }
    },
    "model": {
        "backbone": {
            "name": "resnet50",
            "features": [1, 2, 3, 4],
            "remove_ds": 4,
            "pretrained": True,
        },
        "head": {
            "pooling": {
                "_parser": "default_loader:name",
                "name": "vlad",
                "_defaults": {
                    "vlad": {
                        "use_bn": True,
                        "core": "euc",
                        "fg_num": 32,
                        "bg_num": 0,
                        "dim": 64,
                        "alpha": 10.0
                    }
                }
            },
            "transform": {
                "type": "linear", # linear | group
                "dim": 2048, # output dim.
            },
            "classifier": {
                "classnum": None,
                "feature": {
                    "enabled": "trans",
                    "trans": {
                        "src": "trans",
                        "type": "linear",
                        "relu": True,
                    },
                    "pool": {
                        "src": "pool",
                        "type": "linear",
                        "relu": False,
                    },
                    "gpool": {
                        "src": "pool",
                        "type": "group",
                        "relu": False,
                    },
                    "avgpool": {
                        "src": "avgpool",
                        "type": "linear",
                        "relu": False
                    }
                }
            },
            "test_kind": "trans",
        },
        "fuse": None,
    },
    "supress": None,
    "optim": {
        "lr": 0.1,
        "wd": 5e-3,
    },
    "scheduler": {
        # gamma=0.1, location=[30, 50], warm_epoch=2, pretrain_groups=[0]
        "gamma": 0.1,
        "location": [30, 50],
        "warm_epoch": 2,
        "pretrain_groups": [0],
    },
    "loss": {
        "cls": "ce", # ce|sce
    },
    # for test
    "test": {
        "from": "calf",
    },
    # for calculate features
    "calf": {
        "loaders": "query,gallery",
        "assign": False
    },
    # for convert_features
    "covf": {
        "type": 'cat', # [0.1, 1.0, 1.3, 1.1]
        "args": {
            "weights": [0.1, 1.0, 1.3, 1.1],
            "id": -1,
        },
        "flag": None,
        "refine": False
    },
    "mine": {
        "k": 8,
        "escape": False
    }
}

parse_metas = {
    "model.head.pooling": {"short": "pooling"},
    "model.backbone": {"short": "backbone"},
    "model.head": {"short": "head"},
}

confmgr = ConfigurationParser(name="Reid")
args = confmgr.parse_cmd(metas=parse_metas)

def merge_dataset_defaults(conf:dict, args:dict, defaults:dict):
    dataset = args.pop('dataset', conf['dataset'])
    default = defaults[dataset]
    class_num = args.get('classnum', default['classnum'])
    updater = {
        "dataset": dataset,
        "classnum": class_num,
        "optim": {
            "wd": default['wd']
        },
        "model": {
            "head": {
                "classifier": {
                    "classnum": class_num
                }
            }
        },
        "root": os.path.join(server_conf['root'], default['dirname'])
    }
    # update paths.
    updater['paths'] = {key:os.path.join(updater['root'], key) for key in conf['paths']}
    for path in updater['paths']:
        utils.ensure_dir_exist(updater['paths'][path])
    updater['dataconf'] = {
        "set": {
            "_common": {
                "dataset_root": updater['root'],
            }
        }
    }
    list_path = default.get('list', {})
    for key in list_path:
        if key not in updater['dataconf']['set']:
            updater['dataconf']['set'][key] = {}
        updater['dataconf']['set'][key]['list_path'] = list_path[key]
    conf = confmgr.merge(conf, updater)
    return conf

conf = merge_dataset_defaults(conf, args, dataset_defaults)

########## parser defination ##########
def default_loader(conf, identifier):
    identifier = conf[identifier]
    defaults = conf.pop('_defaults')[identifier]
    defaults.update(conf)
    return defaults

parsers = {
    "default_loader": default_loader,
}

def apply_parser(conf):
    parser = conf.pop('_parser', None)
    if parser is not None:
        # parser parser and args.
        parser = parser.split(':', maxsplit=1)
        args = []
        if len(parser) > 1:
            parser, args = parser
            args = args.split('|')
        parser = parsers[parser]
        conf = parser(conf, *args)
    for key, value in conf.items():
        if type(value) is dict:
            conf[key] = apply_parser(value)
    return conf
conf = apply_parser(conf)
conf = confmgr.merge(conf, args)

# generate summary.
conf['summary'] = conf['model']['head']['pooling']['name']

if 'train' in conf['main_name']:
    conf['training_mode'] = True
else:
    conf['training_mode'] = False

logname = '{0}-{1}-{2}.log'.format(sys.argv[0], conf['summary'], time.strftime("%Y%m%d-%H%M%S", time.localtime()))
logfile = os.path.join(conf['paths']['logs'], logname)
logger.setfile(logfile)
logger.log('Syncing stdout to file', logfile)

if conf['training_mode']:
    print(utils.pretty_dict(conf))


# autoset gpu.
blacklist = []
gpu = conf['gpu']
if gpu[:4] == 'auto':
    num = 1
    if len(gpu) > 5:
        try:
            num = int(gpu[5:])
        except:
            raise ValueError('When auto setting gpu, format must be auto:int, eg. auto:2 if you want to use two gpu, but you give {0}'.format(gpu))
    auto_set(num, blacklist=blacklist)
elif gpu == 'none':
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu

# set deterministic
if conf['seed'] is not None:
    print('setting deterministic mode with seed', conf['seed'])
    set_deterministic(conf['seed'])
else:
    set_benchmark()
