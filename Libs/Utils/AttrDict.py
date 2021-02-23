import collections
from six import iteritems, iterkeys

def update_dict(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update_dict(d.get(k, AttrDict()), v)
        else:
            d[k] = v
    return d


class AttrDict(dict):
    """
        A dictionary that provides attribute-style access.
        Adapted from https://github.com/Infinidat/munch
    """

    def update(self, *args):
        """
            update current config with new args
            /!\ order: first args will be the most important and overwrite the one afters
        """
        args = list(args)
        for idx in range(len(args)-1, -1, -1):
            if args[idx] is None:
                args[idx] = AttrDict()
            elif not isinstance(args[idx], AttrDict):
                args[idx] = AttrDict.fromDict(args[idx])

            update_dict(self, args[idx])

    # only called if k not found in normal places
    def __getattr__(self, k):
        try:
            # Throws exception if not in prototype chain
            return object.__getattribute__(self, k)
        except AttributeError:
            try:
                return self[k]
            except KeyError:
                raise AttributeError(k)

    def __setattr__(self, k, v):
        """
            Sets attribute k if it exists, otherwise sets key k.
        """
        try:
            # Throws exception if not in prototype chain
            object.__getattribute__(self, k)
        except AttributeError:
            try:
                self[k] = v
            except:
                raise AttributeError(k)
        else:
            object.__setattr__(self, k, v)

    def __delattr__(self, k):
        """
            Deletes attribute k if it exists, otherwise deletes key k.
        """
        try:
            # Throws exception if not in prototype chain
            object.__getattribute__(self, k)
        except AttributeError:
            try:
                del self[k]
            except KeyError:
                raise AttributeError(k)
        else:
            object.__delattr__(self, k)

    def toDict(self):
        """
            Recursively converts a munch back into a dictionary.
        """
        return unmunchify(self)

    @property
    def __dict__(self):
        return self.toDict()

    def __repr__(self):
        """
            Invertible* string-form of a Munch.
        """
        return '{0}({1})'.format(self.__class__.__name__, dict.__repr__(self))

    def __dir__(self):
        return list(iterkeys(self))

    def __getstate__(self):
        """
            Implement a serializable interface used for pickling.
            See https://docs.python.org/3.6/library/pickle.html.
        """
        return {k: v for k, v in self.items()}

    def __setstate__(self, state):
        """
            Implement a serializable interface used for pickling.
            See https://docs.python.org/3.6/library/pickle.html.
        """
        self.clear()
        self.update(state)

    __members__ = __dir__  # for python2.x compatibility

    @classmethod
    def fromDict(cls, d):
        """
            Recursively transforms a dictionary into a Munch via copy.
        """
        return munchify(d, cls)

    def copy(self):
        return type(self).fromDict(self)


def munchify(x, factory=AttrDict):
    """
        Recursively transforms a dictionary into a Munch via copy.
    """
    if isinstance(x, dict):
        return factory((k, munchify(v, factory)) for k, v in iteritems(x))
    elif isinstance(x, (list, tuple)):
        return type(x)(munchify(v, factory) for v in x)
    else:
        return x


def unmunchify(x):
    """
        Recursively converts a Munch into a dictionary.
    """
    if isinstance(x, dict):
        return dict((k, unmunchify(v)) for k, v in iteritems(x))
    elif isinstance(x, (list, tuple)):
        return type(x)(unmunchify(v) for v in x)
    else:
        return x
