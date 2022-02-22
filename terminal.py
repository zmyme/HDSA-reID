
TerminalEscape = {
    "mode": {
        "default": 0,
        "highlight": 1,
        "underline": 4,
        "blink": 5,
        "invwhite": 7,
        "invisible": 8,
    },
    "color": {
        "black" : 30,
        "red" : 31,
        "green" : 32,
        "yellow" : 33,
        "blue" : 34,
        "purple" : 35,
        "cyan" : 36,
        "white" : 37,
    },
    "background": {
        "black": 40,
        "red": 41,
        "green": 42,
        "yellow": 43,
        "blue": 44,
        "purple": 45,
        "cyan": 46,
        "white": 47,
    },
    "cursor": {
        "up": '\033[{n}A',
        "down": '\033[{n}B',
        "right": '\033[{n}C',
        "left": '\033[{n}D',
        "setpos": '\033[{y}:{x}H',
        "savepos": '\033[s',
        "recpos": '\033[u',
        "hide": '\033[?25l',
        "show": '\033[?25h',
    },
    "others": {
        "cls": '\033[2J',
        "cll": '\033[K'
    }
}

def TerminalStyle(mode="default", color=None, background=None):
    fmt = str(TerminalEscape['mode'][mode])
    if background is not None:
        fmt += ';{0}'.format(TerminalEscape['background'][background])
    if color is not None:
        fmt += ';{0}'.format(TerminalEscape['color'][color])
    fmt = '\033[{0}m'.format(fmt)
    return fmt

def CursorControl():
    def up(n):
        return TerminalEscape['cursor']['up'].format(n=n)
    def down(n):
        return TerminalEscape['cursor']['down'].format(n=n)
    def left(n):
        return TerminalEscape['cursor']['left'].format(n=n)
    def right(n):
        return TerminalEscape['cursor']['right'].format(n=n)
    def moveto(y, x):
        return TerminalEscape['cursor']['setpos'].format(y=y, x=x) 
    def hide():
        return TerminalEscape['cursor']['hide']
    def show():
        return TerminalEscape['cursor']['show']
    
if __name__ == '__main__':
    # print(BgColor["black"])
    pass
