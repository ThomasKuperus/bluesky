''' Stack Command implementation. '''
import inspect

from bluesky.stack.parser import Parameter, getnextarg


class Command:
    ''' Stack command object. '''

    # Dictionary with all command objects
    cmddict = dict()

    def __init__(self, func, parent=None, **kwargs):
        # Get function object if it's decorated as static or classmethod
        func = func.__func__ if isinstance(func, (staticmethod, classmethod)) \
            else func
        self.name = kwargs.get('name', func.__name__).upper()
        self.help = inspect.cleandoc(kwargs.get('help', ''))
        self.brief = kwargs.get('brief', '')
        self.aliases = kwargs.get('aliases', tuple())
        self.impl = ''
        self.valid = True
        self.annotations = get_annot(kwargs.get('annotations', ''))
        self.params = list()
        self.parent = parent
        self.callback = func

        # When a parent is passed this function is a subcommand
        if parent is None:
            Command.cmddict[self.name] = self
            for alias in self.aliases:
                Command.cmddict[alias] = self
        elif isinstance(parent, CommandGroup):
            parent.subcmds[self.name] = self
            for alias in self.aliases:
                parent.subcmds[alias] = self

        # Store reference to self for function
        if not inspect.ismethod(func):
            func.__stack_cmd__ = self

    def __call__(self, argstring):
        args = []
        param = None
        # Use callback-specified parameter parsers to generate param list from strings
        for param in self.params:
            result = param(argstring)
            argstring = result[-1]
            args.extend(result[:-1])

        # Parse repeating final args
        while argstring:
            if param is None or not param.gobble:
                msg = f'{self.name} takes {len(self.params)} argument'
                if len(self.params) > 1:
                    msg += 's'
                count = len(self.params)
                while argstring:
                    _, argstring = getnextarg(argstring)
                    count += 1
                msg += f', but {count} were given'
                raise TypeError(msg)
            arg, argstring = param(argstring)
            args.append(arg)

        # Call callback function with parsed parameters
        ret = self.callback(*args)
        return ret

    def __repr__(self):
        if self.valid:
            return f'<Stack Command {self.name}, callback={self.callback}>'
        return f'<Stack Command {self.name} (invalid), callback=unbound method {self.callback}'

    def notimplemented(self, *args, **kwargs):
        ''' Stub for stack functions based on Entity methods, when a currently
            selected Entity implementation doesn't provide the stack function of
            this command.
        '''
        impl = self.impl or inspect.getmodule(self.callback).__name__
        return False, f'The current {self.name} implementation doesn\'t' +\
            f'provide this function (function was originally declared in {impl})'

    @property
    def callback(self):
        ''' Callback pointing to the actual function that implements this
            stack command.
        '''
        return self._callback

    @callback.setter
    def callback(self, function):
        self._callback = function
        spec = inspect.signature(function)
        # Check if this is an unbound class/instance method
        self.valid = spec.parameters.get('self') is None and \
            spec.parameters.get('cls') is None

        if self.valid:
            # Store implementation origin
            if not self.impl:
                # Check if this is a bound (class or object) method
                if inspect.ismethod(function):
                    if inspect.isclass(function.__self__):
                        self.impl = function.__self__.__name__
                    else:
                        self.impl = function.__self__.__class__.__name__

            self.brief = self.brief or (
                self.name + ' ' + ','.join(spec.parameters))
            self.help = self.help or inspect.cleandoc(
                inspect.getdoc(function) or '')
            paramspecs = list(spec.parameters.values())
            if self.annotations:
                self.params = list()
                pos = 0
                for annot in self.annotations:
                    if annot == '...' and not self.params[-1].gobble:
                        raise IndexError('Repeating arguments (...) given for function'
                                         ' not ending in starred (variable-length) argument')

                    param = Parameter(paramspecs[pos], annot)
                    if param:
                        pos = min(pos + param.size(), len(paramspecs) - 1)
                        self.params.append(param)
                if len(self.params) > len(paramspecs) and not self.params[-1].gobble:
                    raise IndexError(f'More annotations given than function '
                                     f'{self.callback.__name__} has arguments.')
            else:
                self.params = [p for p in map(Parameter, paramspecs) if p]

    def helptext(self):
        ''' Return complete help text. '''
        return f'{self.help}\nUsage:\n{self.brief}'

    def brieftext(self):
        ''' Return the brief usage text. '''
        return self.brief


class CommandGroup(Command):
    ''' Stack command group object.
        Command groups can have subcommands.
    '''
    def __init__(self, func, parent=None, **kwargs):
        super().__init__(func, parent, **kwargs)
        self.subcmds = dict()

        # Give the function a method to add subcommands
        func.subcommand = lambda fun=None, **kwargs: command(
            fun, parent=self, **kwargs)

    def __call__(self, strargs):
        # First check subcommand
        if strargs:
            subcmd = self.subcmds.get(strargs[0], None)
            if subcmd:
                return subcmd(strargs[1:])
        return super().__call__(strargs)

    def helptext(self):
        ''' Return complete help text. '''
        msg = f'{self.help}\nUsage:\n{self.brief}'
        for subcmd in self.subcmds.values():
            msg += f'\n{self.name} {subcmd.brief}'
        return msg

    def brieftext(self):
        ''' Return the brief usage text. '''
        msg = self.brief
        for subcmd in self.subcmds.values():
            msg += f'\n{self.name} {subcmd.brief}'
        return msg

def commandgroup(fun=None, parent=None, **kwargs):
    ''' BlueSky stack command decorator for command groups.

        Functions decorated with this decorator become available as stack
        functions, and have the ability to have subcommands.
    '''
    return command(fun, CommandGroup, parent, **kwargs)


def command(fun=None, cls=Command, parent=None, **kwargs):
    ''' BlueSky stack command decorator.

        Functions decorated with this decorator become available as stack
        functions.
    '''
    def deco(fun):
        # Construct the stack command object, but return the original function
        cls(fun, parent, **kwargs)
        return fun
    # Allow both @command and @command(args)
    return deco if fun is None else deco(fun)


def append_commands(newcommands, syndict=None):
    """ Append additional functions to the stack command dictionary """
    for name, (brief, annotations, fun, hlp) in newcommands.items():
        if syndict:
            aliases = (k for k,v in syndict.items() if v == name)
        else:
            aliases = tuple()
        # Use the command decorator function to register each new command
        command(fun, name=name, annotations=annotations, brief=brief, help=hlp, aliases=aliases)


def remove_commands(commands):
    """ Remove functions from the stack """
    for cmd in commands:
        Command.cmddict.pop(cmd)


def get_commands():
    """ Return the stack dictionary of commands. """
    return Command.cmddict


def get_annot(annotations):
    ''' Get annotations from string, or tuple/list. '''
    if isinstance(annotations, (tuple, list)):
        return tuple(annotations)
    # Assume it is a comma-separated string
    argtypes = []
    argisopt = []

    # Process and reduce annotation string from left to right
    # First cut at square brackets, then take separate argument types
    while annotations:
        opt = annotations[0] == "["
        cut = (annotations.find("]") if opt else annotations.find(
            "[") if "[" in annotations else len(annotations))

        types = annotations[:cut].strip("[,]").split(",")
        argtypes += types
        argisopt += [opt or t == "..." for t in types]
        annotations = annotations[cut:].lstrip(",]")

    return tuple(argtypes)
