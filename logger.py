import os
import sys
import time
import platform
import threading

import terminal

"""
This is a class that broadcast the write and flush operations of 
one stream in to a set of given destinations. It also provided a
src args that stores the source stream since you may want that a
while later.
"""
class StreamBroadcaster():
    def __init__(self, src=None, destinations = None):
        self.configure(src, destinations)
    
    def configure(self, src=None, destinations = None):
        self.src = src
        if destinations is None:
            destinations = []
        if type(destinations) is not None:
            destinations = [destinations]
        self.destinations = destinations
        if not self.check_write_stream(self.destinations):
            raise TypeError('Stream must have write method, while not all destinations provided has that.')
    
    @staticmethod
    def check_write_stream(streams):
        check_pass = True
        for stream in streams:
            if not hasattr(stream, 'write'):
                check_pass = False
                break
        return check_pass
    def write(self, msg):
        for dst in self.destinations:
            dst.write(msg)
    
    def flush(self):
        for dst in self.destinations:
            if hasattr(dst, "flush"):
                dst.flush()

def find_all(string, ch):
    indexs = []
    index = string.find(ch)
    offset = 0
    while index != -1:
        indexs.append(index+offset)
        offset += index + 1
        string = string[index+1:]
        index = string.find(ch)
    return indexs

# this could split multi characters but each characters should only have length of 1
def multi_splits(string, split_chars, reserve=True):
    if type(split_chars) is not list:
        split_chars = [split_chars]
    splits = []
    indexs = []
    for ch in split_chars:
        indexs += find_all(string, ch)
    indexs.sort()
    start = 0
    for index in indexs:
        if start < index:
            splits.append(string[start:index])
        if reserve:
            splits.append(string[index:index+1])
        start = index + 1
    if start < len(string):
        splits.append(string[start:])
    return splits

class SubLogger():
    def __init__(self, meta, defaults):
        self.meta = meta
        self.defaults = defaults
    def __getattr__(self, attr):
        default = {}
        if attr in self.defaults:
            default = self.defaults[attr]
        def caller(*args, **kwargs):
            for key in default:
                if key not in kwargs:
                    kwargs[key] = default[key]
            getattr(self.meta, attr)(*args, **kwargs)
        return caller
                    


"""
what is important about an log method:
1. level
2. color
"""
class Logger():
    """
    logfile: the file that the log is stored.
    redirect_stdio: if set to true, the logger will replace sys.stdout
    minlevel: log information that is smaller than this value will not be displayed.
    """
    def __init__(self, logfile=None, redirect_stdio=True, default_type="info", minlevel=0):
        self.lock = threading.Lock()
        self.redirect = redirect_stdio
        self.default_type = default_type
        self.minlevel = minlevel
        self.redirect_stdio = redirect_stdio
        self.destinations = [sys.stdout]
        if logfile is not None:
            self.setfile(logfile)
        self.logconfs = {
            "debug": {
                "name": "D",
                "level": 0,
                "color": "green",
            },
            "info": {
                "name": "I",
                "level": 1,
                "color": "white",
            },
            "warning": {
                "name": "W",
                "level": 2,
                "color": "yellow",
            },
            "error": {
                "name": "E",
                "level": 3,
                "color": "red",
            },
        }
        self.is_newline = True
        if self.redirect_stdio:
            sys.stdout = self.get_sublogger(kind="info")
            sys.stderr = self.get_sublogger(kind="error")
    
    @staticmethod
    def getlogfile(file):
        if type(file) is str:
            file = open(file, 'w+', encoding='utf-8')
        return file
    
    def setfile(self, file):
        self.file = self.getlogfile(file)
        self.destinations.append(self.file)
    
    def get_sublogger(self, kind):
        return SubLogger(self,
            defaults={
                "log": {"kind": kind},
                "write": {
                    "name": self.logconfs[kind]["name"],
                    "color": self.logconfs[kind]["color"]
                }
            }
        )
    
    def __getattr__(self, attr):
        if attr not in self.logconfs:
            raise AttributeError('Class {0} has no attribute {1} and {1} do not exist in logconf'.format(__class__.__name__, attr))
        def logmethod(*args, **kwargs):
            if 'kind' not in kwargs:
                kwargs['kind'] = attr
            self.log(*args, **kwargs)
        return logmethod
    
    def log(self, *args, kind='info', sep=' ', end='\n'):
        conf = self.logconfs[kind]
        name = conf["name"]
        level = conf["level"]
        color = conf["color"]
        if level < self.minlevel:
            return
        args = [str(arg) for arg in args]
        message = sep.join(args) + end
        self.write(message, name=name, color=color)
    
    def write(self, msg, name=None, color=None):
        prefix_info = {
            "name": name,
            "time": time.strftime("%H:%M:%S", time.localtime())
        }
        prefix = '[{name} {time}] '.format(**prefix_info)
        # set color.
        if color is not None:
            self.basic_write(terminal.TerminalStyle(color=color))
        # split lines.
        msgs = multi_splits(msg, ['\n', '\r'])
        for msg in msgs:
            if msg == '\n' or msg == '\r':
                if self.is_newline:
                    msg = prefix + msg
                self.basic_write(msg)
                self.is_newline = True
                continue
            if self.is_newline:
                msg = prefix + msg
            self.basic_write(msg)
            self.is_newline = False
        if color is not None:
            color = self.basic_write(terminal.TerminalStyle())
    
    def flush(self):
        for dst in self.destinations:
            if hasattr(dst, "flush"):
                dst.flush()
    
    def basic_write(self, msg):
        for destinations in self.destinations:
            destinations.write(msg)

logger = Logger()

if __name__ == '__main__':
    logger = Logger('test.log')
    logger.debug('debug info')
    logger.info('info info')
    logger.warning('warning info')
    logger.error('error info')
    logger.debug('debug info')
    logger.info('info info')
    logger.warning('warning info')
    logger.error('error info')
    logger.debug('debug info')
    logger.info('info info')
    logger.warning('warning info')
    logger.error('error info')
    print('hello, world')
    print('he\n\nllo, wor\nld')
    print('hello, world', end='')
    print(' this is', end='')
    print(' a test', end='\r')
    print('23333')
    print('12345\b12345')
    print('Testing progress bar')
    for epoch in range(2):
        for i in range(100):
            print('currently process: {0}%'.format(i), end='\r')
            time.sleep(0.02)
        print('')
    print('Test finished!')
    string = 'ab'
    print(string)
    print(multi_splits(string, ['a', 'b'], reserve=True))
    print(multi_splits(string, ['a', 'b'], reserve=False))