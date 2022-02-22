import torch
import time
import sys
from copy import deepcopy
import random
import numpy as np
import os
from scipy.io import loadmat, savemat

class BaseModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.enable_summary = True

def filt_modules(net, f, mode='any', no_exception=True):
    def wrap_filter(f):
        def wrapper(m):
            try:
                return f(m)
            except Exception:
                return False
        return wrapper
    filters = []
    if type(f) is not list:
        f = [f]
    for sub in f:
        if type(sub) is str:
            filters.append(lambda m, name=sub:m.__class__.__name__ == name)
        elif callable(sub):
            filters.append(sub)
        else:
            raise ValueError('filter must be callable or string, while got {0}'.format(type(sub)))
    if mode == 'any':
        mode = any
    elif mode == 'all':
        mode = all
    else:
        raise ValueError('mode can either be any or all, while got {0}'.format(mode))
    ans = []
    if no_exception:
        filters = [wrap_filter(f) for f in filters]
    for m in net.modules():
        if mode(sub_f(m) for sub_f in filters):
            ans.append(m)
    return ans

table_chars = {
    "lt": '\u250c',
    'rt': '\u2510',
    'lb': '\u2514',
    'rb': '\u2518',
    'c': '\u253c',
    'l': '\u251c',
    'r': '\u2524',
    't': '\u252c',
    'b': '\u2534',
    'h': '\u2500',
    'v': '\u2502',
}

class TableDrawer():
    def __init__(self, names=None, aligns=None, widths=None, output=None, mute=False):
        if output is None:
            output = sys.stdout
        self.output = output
        self.mute = mute
        self.aligns = []
        self.widths = []
        self.cols = []
        self.indexs = {}
        self.justifiers = {
            "c": lambda s, w:s.center(w),
            "l": lambda s, w:s.ljust(w),
            "r": lambda s, w:s.rjust(w),
        }
        if names is not None:
            self.configure(names)
    """
    rows should be a list of dicts.
    rows = [
        {
            "name": str,
            "width": int,
            "align": str, c, l, r for center, left and right.
        }
    ]
    """
    def configure(self, names, aligns=None, widths=None):
        num_cols = len(names)
        if type(aligns) is str:
            if len(aligns) == 1:
                aligns = aligns * num_cols
            aligns = list(aligns)
        if type(widths) is int:
            widths = [widths] * num_cols
        if len(widths) != num_cols:
            raise ValueError('widths must have same length as names, but len(widths)={0} while len(names)={1}'.format(len(widths), num_cols))
        if len(aligns) != num_cols:
            raise ValueError('aligns must have same length as names, but len(aligns)={0} while len(names)={1}'.format(len(aligns), num_cols))

        self.rows = names
        self.aligns = aligns
        self.widths = widths
        self.indexs = {name:index for index, name in enumerate(names)}
    def modconf(self, name, align=None, width=None):
        if name not in self.indexs:
            raise ValueError('The column {0} do not exist.'.format(name))
        index = self.indexs[name]
        if align is not None:
            self.aligns[index] = None
        if width is not None:
            self.aligns[index] = width
    def draw(self, msg):
        if not self.mute:
            self.output.write(msg)
        return msg
    """
    row can be a list of columns or a dict of {key=colname:value=value}
    """
    def drawrow(self, row):
        if type(row) is dict:
            dictrow = row
            row = []
            for colname in self.rows:
                if colname in dictrow:
                    row.append(dictrow[colname])
                else:
                    row.append("")
        padded = []
        for col, align, width in zip(row, self.aligns, self.widths):
            if len(col) > width:
                col = col[:width]
            else:
                col = self.justifiers[align](col, width)
            padded.append(col)
        row = table_chars['v'].join(padded)
        row = '{0}{1}{0}'.format(table_chars['v'], row)
        return self.draw(row)
    def drawline(self, mode='middle'):
        connectchars = {
            "first": table_chars['t'],
            "end": table_chars['b'],
            "middle": table_chars['c']
        }
        leftchars = {
            "first": table_chars['lt'],
            "end": table_chars['lb'],
            "middle": table_chars["l"]
        }
        rightchars = {
            "first": table_chars['rt'],
            "end": table_chars['rb'],
            "middle": table_chars["r"]
        }
        connector = connectchars[mode]
        leftch = leftchars[mode]
        rightch = rightchars[mode]
        padded = [table_chars['h'] * w for w in self.widths]
        row = connector.join(padded)
        row = '{0}{1}{2}'.format(leftch, row, rightch)
        return self.draw(row)
    def drawheader(self, upperline=True, bottomline=True):
        msgs = []
        original_state = self.mute
        self.mute = True
        if upperline:
            msgs.append(self.drawline(mode='first'))
        msgs.append(self.drawrow(self.rows))
        if bottomline:
            msgs.append(self.drawline(mode='middle'))
        msg = '\n'.join(msgs) + '\n'
        self.mute = original_state
        return self.draw(msg)


class ValueTrackerBase():
    def __init__(self) -> None:
        self._counter = 0
        pass
    def count(self):
        return self._counter
    def feed(self, value):
        self._counter += 1
    def get(self):
        pass
    def clear(self):
        self._counter = 0

class ValueTrackerSum(ValueTrackerBase):
    def __init__(self) -> None:
        super().__init__()
        self.sum = 0
    def feed(self, value):
        super().feed(value)
        self.sum += value
    def get(self):
        return self.sum
    def clear(self):
        super().clear()
        self.sum = 0

class ValueTrackerAvg(ValueTrackerSum):
    def __init__(self) -> None:
        super().__init__()
    def get(self):
        return super().get() / self.count()

class ValueTrackerLatest(ValueTrackerBase):
    def __init__(self) -> None:
        super().__init__()
        self.value = None
    def feed(self, value):
        super().feed(value)
        self.value = value
    def get(self):
        return self.value
    def clear(self):
        super().clear()
        self.value = None


class ValueSummarizer():
    def __init__(self) -> None:
        self.conf:list = None
        self.index = None
        self.registed_trackers = {
            "sum": ValueTrackerSum,
            "avg": ValueTrackerAvg,
            "latest": ValueTrackerLatest
        }
        self.trackers:list = None
    def make_default(self, num_rows):
        self.conf = []
        for i in range(num_rows):
            conf = {
                "name": "row_{0}".format(i),
                "tracker": "latest"
            }
            self.conf.append(conf)
    def generate(self):
        self.index = {}
        self.trackers = []
        for i, conf in enumerate(self.conf):
            name = conf['name']
            tracker = conf['tracker']
            self.index[name] = i
            tracker = self.registed_trackers[tracker]()
            self.trackers.append(tracker)
        
    def config(self, rows):
        self.conf = rows
        self.generate()
    
    def feed(self, values:list=None, kwvalues:dict=None):
        """
        values: a list of values to update, order is given in config.
        kwvalues: feed values via given name, should be a dict.
        Note: we do not check if you feed two same value in values and kwvalues.
        """
        if values is None:
            values = []
        if kwvalues is None:
            kwvalues = {}
        for i, value in enumerate(values):
            # print('value:', value)
            self.trackers[i].feed(value)
        for key, value in kwvalues.items():
            idx = self.index[key]
            self.trackers[idx].feed(value)
    
    def get(self, mode):
        """
        Args:
            - mode:str: either be list or dict.
        """
        if mode == 'list':
            values = [tracker.get() for tracker in self.trackers]
        elif mode == 'dict':
            values = {conf['name']:tracker.get() for conf, tracker in zip(self.conf, self.trackers)}
        else:
            raise ValueError('mode should either be list or dict, instead of', mode)
        return values
    
    def clear(self):
        for tracker in self.trackers:
            tracker.clear()

class TrainTableSummarizer():
    def __init__(self, num_batches, output = None) -> None:
        if output is None:
            output = sys.stdout
        self.output = output
        self.conf = None # should be a list.
        self.default_conf = {
            "name": 'Unnamed',
            "mode": "avg",
            "width": 6,
            "align": "c",
        }
        self.tracker = ValueSummarizer()
        self.drawer = TableDrawer(output=None, mute=True)
        self.epochid = 0
        self.batchid = 0
        self.num_batches = num_batches
        self.epoch_start_time = None
        self.dispinterval = 1.0
        self.lastdisp = 0
    def make_default_conf(self, num_rows)->list:
        self.conf = [deepcopy(self.default_conf) for i in range(num_rows)]
    def config(self, rows:list, rowids=None):
        """
        config the summarizer.
        row:dict = {
            name: str,
            mode: str, sum|avg|max|min|latest, correspond to pre-defined value tracker.
            width: width.
            align: center|left|right
        }
        rowids:Union(None, list), contros which row to update.
        """
        if self.conf is None:
            self.make_default_conf(len(rows))
        if rowids is None:
            rowids = list(range(len(rows)))
        for row, rowid in zip(rows, rowids):
            prev = self.conf[rowid]
            prev.update(row)
    def config_single(self, rows, target='name', rowids=None):
        if target not in self.default_conf:
            raise ValueError('Target {0} is not found. Existed target: {1}'.format(target, '|'.join(self.default_conf)))
        if self.conf is None:
            self.make_default_conf(len(rows))
        if rowids is None:
            rowids = list(range(len(self.conf)))
        for rowid, updater in zip(rowids, rows):
            self.conf[rowid][target] = updater
    
    def append_default_rows(self):
        epoch = deepcopy(self.default_conf)
        epoch.update({
            "name": "epoch",
            "mode": "latest",
            "width": 5
        })
        time_used = deepcopy(self.default_conf)
        time_used.update({
            "name": "time",
            "mode": "latest",
            "width": 5
        })
        self.conf = [epoch] + self.conf + [time_used]
        # print('self.conf:', self.conf)
    
    def get_internal_values(self, mode:str="batch"):
        """
        """
        values = {
            "epoch": self.batchid*100 / self.num_batches if mode == "batch" else self.epochid,
            "time": time.time() - self.epoch_start_time
        }
        return values
        
    def start(self):
        self.append_default_rows()
        # initialize trackers.
        tracker_conf = []
        for conf in self.conf:
            this_tracker = {
                "name": conf["name"],
                "tracker": conf["mode"]
            }
            tracker_conf.append(this_tracker)
        self.tracker.config(tracker_conf)
        # config drawer and draw header.
        display_rows = [conf['name'] for conf in self.conf]
        widths = [conf['width'] for conf in self.conf]
        aligns = [conf['align'] for conf in self.conf]
        self.drawer.configure(display_rows, widths=widths, aligns=aligns)
        self.output.write(self.drawer.drawheader())
    
    def finish(self):
        self.output.write(self.drawer.drawline(mode="end") + '\n')
    
    def summary_epoch(self, additional:dict=None):
        self.epochid += 1
        
        if additional is None:
            additional = {}
        internal = self.get_internal_values(mode="epoch")
        internal.update(additional)
        self.tracker.feed(kwvalues=internal)

        self.realdisp()
        self.output.write('\n')

        self.batchid = 0
        self.epoch_start_time = None
        self.tracker.clear()
    
    def feed_batch(self, info:dict = None):
        self.batchid += 1
        if self.epoch_start_time is None:
            self.epoch_start_time = time.time()
        if info is None:
            info = {}
        internal_values = self.get_internal_values(mode="batch")
        info.update(internal_values)
        self.tracker.feed(kwvalues=info)
        self.dispbatch()
    
    def realdisp(self):
        # collect stat.
        stat = self.tracker.get(mode='list')
        stat = [str(s) for s in stat]
        rowinfo = self.drawer.drawrow(stat)
        self.output.write(rowinfo + '\r')
    
    def dispbatch(self):
        t = time.time()
        if t - self.lastdisp > self.dispinterval:
            self.realdisp()
            self.lastdisp = t

class lr_manager():
    def __init__(self, optimizer, gamma=0.1, location=None, show_info=True, warm_epoch=2, pretrain_groups=None, warm_mode='linear_freeze'):
        if pretrain_groups is None:
            pretrain_groups = [0]
        if location is None:
            location = []
        self.optimizer = optimizer
        self.gamma = gamma
        self.location = location
        self.steps = 0
        self.show_info = show_info
        self.warm_epoch = warm_epoch
        if self.warm_epoch > 0:
            self.warm_up = True
            self.original_lrs = []
            for param_group in self.optimizer.param_groups:
                self.original_lrs.append(param_group['lr'])
            if self.show_info:
                print('Original learning rate is backed up as', self.original_lrs)
        else:
            self.warm_up = False
        self.pretrain_groups = pretrain_groups
        self.warm_mode = warm_mode

    def update_lr(self):
        group_id = 0
        if self.show_info:
            print('changing learning rate:')
        for param_group in self.optimizer.param_groups:
            group_id += 1
            if self.show_info:
                print('param group', group_id, ':', param_group['lr'], end = ' ')
            param_group['lr'] *= self.gamma
            if self.show_info:
                print('->', param_group['lr'])

    def step(self):
        self.steps += 1
        if self.warm_up:
            num_group = len(self.original_lrs)
            if self.steps <= self.warm_epoch:
                lr_decrease_rate = 1.0
                if 'linear' in self.warm_mode:
                    lr_decrease_rate = self.steps * 1.0 / (self.warm_epoch + 1.0)
                this_group_lr = []
                for i in range(num_group):
                    if i in self.pretrain_groups and 'freeze' in self.warm_mode:
                        this_group_lr.append(0.0)
                    else:
                        this_group_lr.append(self.original_lrs[i] * lr_decrease_rate)
                if self.show_info:
                    print('learning rate in this epoch is set to', this_group_lr)
                for i in range(num_group):
                    self.optimizer.param_groups[i]['lr'] = this_group_lr[i]
            if self.steps == self.warm_epoch + 1:
                if self.show_info:
                    print('recover original learning rate to', self.original_lrs, 'warm up process finished.')
                for i in range(num_group):
                    self.optimizer.param_groups[i]['lr'] = self.original_lrs[i]
        if self.steps - 1 in self.location:
            self.update_lr()

class LossManager():
    def __init__(self):
        self.weights = {}
        self.current = {}
        self.history = {} # history is a name:[epoch[iteration]] dict.
        self.current_epoch = {}
        self.record = False
    # losses is a dict of name:value, name is the loss name, value is its weight.
    def config(self, **conf):
        self.weights = conf
        for key in self.weights:
            self.history[key] = [[]]
        self.initialize_epoch()
    def initialize_epoch(self):
        self.current_epoch = {"sum": {}, "num_batch": 0}
        for key in self.weights:
            self.current_epoch["sum"][key] = 0
    """
    feed a list of losses in the order of lossname=lossvalue, return the overall loss.
    """
    def feed(self, losses):
        self.current = {}
        for key in self.weights:
            current_loss = 0.0
            if key in losses:
                current_loss = losses[key]
            self.current[key] = current_loss # this should be a tensor.
            current_loss = float(current_loss) # convert to float for sum and record.
            self.current_epoch["sum"][key] += current_loss
            if self.record:
                self.history[key][-1].append(current_loss)
        return self.get_overall_loss()
    def summarize_epoch(self, fmt, delemeter, post_process=lambda x:x):
        # supported keywords in format: sum, num, avg
        # build summarize dict.
        summarize = {}
        num = self.current_epoch["num_batch"]
        for key in self.weights:
            sum = self.current_epoch["sum"][key]
            avg = sum/num
            summarize[key] = {
                "sum": sum,
                "avg": avg,
                "num": num
            }
        items = [post_process(fmt.format(**summarize[key])) for key in summarize]
        description = delemeter.join(items)
        return description
    # this will tell the manager the epoch is over, it will return the description for this epoch.
    def epoch_finish(self, *args, **kwargs):
        for key in self.history:
            self.history[key].append([])
        description = self.summarize_epoch(*args, **kwargs)
        self.initialize_epoch()
        return description
    def get_overall_loss(self):
        return sum([self.current[key] * self.weights[key] for key in self.weights])
    def backward(self, *args, **kwargs):
        overall = self.get_overall_loss()
        return overall.backward(*args, **kwargs)

def clean_split(string, sep=' '):
    subs = string.split(sep)
    subs = [sub.strip() for sub in subs]
    subs = [sub for sub in subs if sub != '']
    return subs

class DictAdvancedAccess():
    def __init__(self, data:dict=None, sep='.', auto_create=True) -> None:
        if data is None:
            data = {}
        self._data = data
        self.sep = sep
        self.auto_create = auto_create
    
    @staticmethod
    def parse(data:dict, path:str, sep='.', auto_create=False):
        # print('parsing', path)
        paths = clean_split(path, sep=sep)
        if len(paths) == 0:
            return data, None
        node = data
        # print('paths:', paths)
        for key in paths[:-1]:
            if key not in node and auto_create:
                node[key] = {}
                # print('creating key', key)
            if key in node:
                node = node[key]
            else:
                raise KeyError('Unknown key {0}, avaliable keys: {1}'.format(key, list(node.keys())))
        entry = paths[-1]
        return node, entry
    @staticmethod
    def get(data:dict, path:str, sep='.'):
        node, entry = DictAdvancedAccess.parse(data, path, sep, auto_create=False)
        if entry is not None:
            node = node[entry]
        return node
    @staticmethod
    def set(data:dict, value:any, path:str, sep='.', auto_create=True):
        node, entry = DictAdvancedAccess.parse(data, path, sep, auto_create=auto_create)
        if entry is not None:
            node[entry] = value
        else:
            node = value
        return node
    def keys(self):
        return self._data.keys()
    def __getitem__(self, index):
        return self.get(self._data, index, sep=self.sep)
    def __setitem__(self, index, value):
        self.set(self._data, value, index, sep=self.sep, auto_create=self.auto_create)
    def __contains__(self, index):
        try:
            father, entry = self.parse(self._data, index, sep=self.sep, auto_create=False)
            if entry in father or entry is None:
                return True
            else:
                return False
        except KeyError:
            return False
    def __iter__(self):
        for key in self._data:
            yield key
    def items(self):
        return self._data.items()
    def __delitem__(self, index):
        node, entry = self.parse(self._data, index, sep=self.sep, auto_create=False)
        del node[entry]

def set_deterministic(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def set_benchmark():
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def count_params(net):
    params = list(net.parameters())
    num_params = 0
    for p in params:
        this_param = 1
        this_size = p.size()
        for dim in this_size:
            this_param *= dim
        num_params += this_param
    return num_params

def ensure_dir_exist(path, show=True):
    # replace all \\ to /
    while '\\' in path:
        path = path.replace('\\', '/')
    current = ''
    while path != '':
        seppos = path.find('/')
        if seppos == -1:
            current += path
            path = ''
        else:
            current += path[:seppos+1]
            path = path[seppos+1:]
        if not os.path.isdir(current):
            print('creating', current)
            os.mkdir(current)

def pretty_dict(d, indent=4, depth=0):
    info = ''
    list_like_types = [list, tuple]
    base_types = [int, float]
    if type(d) is dict:
        this = []
        for k in d:
            item = pretty_dict(d[k], indent=indent, depth = depth + 1)
            this.append(' '*indent*depth + '{0}: '.format(k) + item)
        if len(this) != 0:
            info = '\n' + '\n'.join(this)
        else:
            info = '{}'
    elif type(d) in list_like_types:
        d = list(d)
        if all([type(item) in base_types for item in d]):
            info = str(d)
        else:
            this = []
            for sub in d:
                item = pretty_dict(sub, indent=indent, depth=depth+1)
                this.append(' '*indent*depth + '- ' + item)
            info = '\n' + '\n'.join(this)
    else:
        info = str(d)
    return info

class LineDisplayer():
    def __init__(self):
        self.lastlen = 0 # lastlen is used for 
    def newline(self):
        self.lastlen = 0
        print('\n', end = '')
    def disp(self, *args, sep=' ', end='\r', file=None, flush=False):
        if file is None:
            file = sys.stdout
        args = [str(arg) for arg in args]
        info = sep.join(args)
        info += ' '*(self.lastlen - len(info))
        self.lastlen = len(info)
        print(info, end='\r', file=file, flush=flush)

def save_arrays(path:str, arrs:dict, backend:str='numpy', hint=False):
    """
    save a dict of arrays, the given dict should have depth of 1.
    Args:
        path: string of the file to save, no need to provied extension.
        arrs: a dict containing the arrays to save.
        backend: method to save, should be numpy | matlab
    return: None
    """
    if backend == 'numpy':
        path += '.npz'
        np.savez(path, **arrs)
    elif backend == 'matlab':
        path += '.mat'
        savemat(path, arrs)
    else:
        raise ValueError('Unrecognized backend, should be numpy | matlab')
    if hint:
        print('Arrays successfully saved to', path)

def load_arrays(path:str, backend:str='numpy', hint=False)->dict:
    """
    load a dict of arrays
    """
    arrs = None
    if backend == 'numpy':
        path += '.npz'
        arrs = np.load(path)
        arrs = dict(arrs)
    elif backend == 'matlab':
        path += '.mat'
        arrs = loadmat(path)
        arrs = dict(arrs)
    else:
        raise ValueError('Unrecognized backend, should be numpy | matlab')
    if hint:
        print('Arrays successfully loaded from', path)
    return arrs
