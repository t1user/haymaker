from ib_tools.misc import Counter


def test_Counter():
    c = Counter()
    num = c()
    assert isinstance(num, str)
    assert num.endswith("00000")


def test_Counter_increments_by_one():
    c = Counter()
    num = c()
    num1 = c()
    assert int(num1[-1]) - int(num[-1]) == 1


def test_Counter_doesnt_duplicate_on_reinstantiation():
    c = Counter()
    num = c()
    d = Counter()
    num1 = d()
    assert num != num1
