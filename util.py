import sys
import inspect
import heapq
import random
import signal
from functools import cmp_to_key

class Stack:
    def __init__(self):
        self.list = []

    def push(self, item):
        self.list.append(item)

    def pop(self):
        return self.list.pop()

    def isEmpty(self):
        return len(self.list) == 0

class Queue:
    def __init__(self):
        self.list = []

    def push(self, item):
        self.list.insert(0, item)

    def pop(self):
        return self.list.pop()

    def isEmpty(self):
        return len(self.list) == 0

class PriorityQueue:
    def __init__(self):
        self.heap = []

    def push(self, item, priority):
        pair = (priority, item)
        heapq.heappush(self.heap, pair)

    def pop(self):
        priority, item = heapq.heappop(self.heap)
        return item

    def isEmpty(self):
        return len(self.heap) == 0

class PriorityQueueWithFunction(PriorityQueue):
    def __init__(self, priorityFunction):
        self.priorityFunction = priorityFunction
        super().__init__()

    def push(self, item):
        PriorityQueue.push(self, item, self.priorityFunction(item))

def manhattanDistance(xy1, xy2):
    return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])

class Counter(dict):
    def __getitem__(self, idx):
        self.setdefault(idx, 0)
        return dict.__getitem__(self, idx)

    def incrementAll(self, keys, count):
        for key in keys:
            self[key] += count

    def argMax(self):
        if len(self) == 0:
            return None
        max_index = max(self.items(), key=lambda x: x[1])[0]
        return max_index

    def totalCount(self):
        return sum(self.values())

    def normalize(self):
        total = float(self.totalCount())
        if total == 0:
            return
        for key in list(self.keys()):
            self[key] /= total

    def divideAll(self, divisor):
        divisor = float(divisor)
        for key in self:
            self[key] /= divisor

    def copy(self):
        return Counter(dict.copy(self))

    def __mul__(self, y):
        sum = 0
        for key in self:
            if key in y:
                sum += self[key] * y[key]
        return sum

    def __add__(self, y):
        addend = Counter()
        for key, value in self.items():
            addend[key] = value + y.get(key, 0)
        for key, value in y.items():
            if key not in self:
                addend[key] = value
        return addend

    def __sub__(self, y):
        subend = Counter()
        for key, value in self.items():
            subend[key] = value - y.get(key, 0)
        for key, value in y.items():
            if key not in self:
                subend[key] = -value
        return subend

def raiseNotDefined():
    print("Method not implemented: %s" % inspect.stack()[1][3])
    sys.exit(1)

def normalize(vectorOrCounter):
    normalizedCounter = Counter()
    if isinstance(vectorOrCounter, Counter):
        counter = vectorOrCounter
        total = float(counter.totalCount())
        if total == 0:
            return counter
        for key in counter:
            value = counter[key]
            normalizedCounter[key] = value / total
        return normalizedCounter
    else:
        vector = vectorOrCounter
        total = float(sum(vector))
        if total == 0:
            return vector
        return [el / total for el in vector]

def flipCoin(p):
    r = random.random()
    return r < p

def sign(x):
    if x >= 0:
        return 1
    else:
        return -1

def pause():
    print("<Press enter/return to continue>")
    input()

class TimeoutFunctionException(Exception):
    """Exception to raise on a timeout"""
    pass

class TimeoutFunction:
    def __init__(self, function, timeout):
        self.timeout = timeout
        self.function = function

    def handle_timeout(self, signum, frame):
        raise TimeoutFunctionException()

    def __call__(self, *args):
        if not hasattr(signal, 'SIGALRM'):
            return self.function(*args)
        old = signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.timeout)
        try:
            result = self.function(*args)
        finally:
            signal.signal(signal.SIGALRM, old)
            signal.alarm(0)
        return result
