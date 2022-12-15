
def inject(cls, callbacks=[]):
    """ Runtime dependency injector, takes a class and an array of callables """

    def raise_callbacks(self, *args, **kwargs):
        for f in callbacks:
            f(self, *args, **kwargs)
        return

    # Overload methods at runtime
    for f in callbacks:
        setattr(cls, f.__name__, f)

    setattr(cls, "raise_callbacks", raise_callbacks)

    return cls


def main():
    pass


if __name__ == "__main__":
    main()
