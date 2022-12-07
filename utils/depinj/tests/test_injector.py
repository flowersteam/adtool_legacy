from leaf.tests.test_leaf import DummyModule
from depinj.tests.testdeps import *
from depinj.injector import inject


def test_inject(capsys):
    """ NOTE: pytest is used to capture stderr """
    inject(DummyModule, callbacks=[logger, hw])
    a = DummyModule("hello")
    assert a.logger
    assert a.hw
    assert a.raise_callbacks

    a.hw()
    expected_out = "hello world\n"
    captured = capsys.readouterr()
    assert captured.out == expected_out

    a.logger()
    expected_out = str(a.__dict__) + "\n"
    captured = capsys.readouterr()
    assert captured.out == expected_out

    a.raise_callbacks()
    expected_out = str(a.__dict__) + "\nhello world\n"
    captured = capsys.readouterr()
    assert captured.out == expected_out

    return


def test_inject_args(capsys):
    inject(DummyModule, callbacks=[logger, hw])
    a = DummyModule("hello")

    a.logger("arg0", kw0="kw0")
    expected_out = str(a.__dict__) + "\narg0\n" + "kw0, kw0\n"
    captured = capsys.readouterr()
    assert captured.out == expected_out


if __name__ == "__main__":
    test_inject()
