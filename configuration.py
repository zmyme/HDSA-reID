import sys

import yaml
import utils
from io import StringIO

def str2bool(string):
    positive = ['true',
        't',
        'y',
        'yes',
        '1',
        'correct',
        'accept',
        'positive'
    ]
    if string.lower() in positive:
        return True
    else:
        return False

def str2list(string):
    items = [p.strip() for p in string.split(',')]
    items = [p for p in items if p != '']
    return items
def str2tuple(string):
    return tuple(str2list(string))

class ConfigurationParser():
    type_mapping = {
        bool:str2bool,
        list:str2list,
        tuple:str2tuple
    }
    def __init__(self, name=None):
        if name is None:
            name = 'Configuration Parser'
        self.name = name
        self.conf = {}
        self.shortcuts = {}
    
    @staticmethod
    def parse_args(source=None):
        if source is None:
            source = sys.argv[1:]
        args = {}
        for arg in source:
            arg = arg.split('=', maxsplit=1)
            if len(arg) < 2:
                raise ValueError('Commandline argument format: name=value, however, no \'=\' detected in {0}'.format(arg[0]))
            path, value = arg
            args[path] = value
        return args

    @staticmethod
    def parse_cmd(metas, source=None):
        # get shortcuts from metas.
        shortcuts = {}
        for entry in metas:
            meta = metas[entry]
            if 'short' in meta:
                shortcuts[meta['short']] = entry
        # print('===================================================')
        # print('shortcuts:', shortcuts)
        # parse args from cmd.
        args = ConfigurationParser.parse_args(source=source)
        # process shortcuts.
        new_agrs = {}
        for key, value in args.items():
            # print('key before translate:', key)
            key = key.split('.')
            if key[0] in shortcuts:
                key[0] = shortcuts[key[0]]
            key = '.'.join(key)
            # print('key after translate:', key)
            new_agrs[key] = value
        args = new_agrs
        # print('args:', args)
        # convert args to corresponding type.
        for entry in args:
            value = args[entry]
            if entry in metas and 'type' in metas[entry]:
                value = metas[entry]['type'](value)
            else:
                value = yaml.safe_load(StringIO(value))
            args[entry] = value
        # build tree.
        parsed = utils.DictAdvancedAccess()
        for key in args:
            parsed[key] = args[key]
        # print('parsed_args:', parsed._data)
        # print('===================================================')
        return parsed._data
    
    @staticmethod
    def merge(src, updater):
        for key in updater:
            if key in src and type(src[key]) is dict:
                src[key] = ConfigurationParser.merge(src[key], updater[key])
            else:
                src[key] = updater[key]
        return src

    @staticmethod      
    def merge_cmdargss(target, metas, source=None):
        args = ConfigurationParser.parse_cmd(metas, source)
        return ConfigurationParser.merge(target, args)


if __name__ == '__main__':
    mgr = ConfigurationParser()
    mgr.regist('test1', default=None, shortcut='t1', dtype=int)
    mgr.regist('test2', default=2, shortcut='t2', dtype=int)
    print(mgr.parse())
    print(mgr.to_str_args())
    mgr.regist('test3', default=3, shortcut='t3', dtype=int)
    mgr.regist('test4', default=4, shortcut='t4', dtype=int)
    mgr.regist('test5', default="hello", shortcut='t5')
    mgr.regist('test6', default=True, shortcut='t6')
    print(mgr.parse())
    print(mgr.to_str_args())
