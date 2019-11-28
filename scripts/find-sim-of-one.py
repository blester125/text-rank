import sys
import argparse
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--max", type=int, default=100)
parser.add_argument("--min_degree", type=int, default=1)
parser.add_argument("--max_degree", type=int, default=6)
args = parser.parse_args()

# We start fitting at 3 because 2 is weird and is larger than the value at 3
# Out side of this the function is monotonically increasing
x = np.arange(3, args.max)
y = x / (np.log(x) + np.log(x))


best_degree = None
best_res = sys.maxsize
best_z = None
for degree in range(args.min_degree, args.max_degree + 1):
    z, res, *_ = np.polyfit(x, y, deg=degree, full=True)
    if res < best_res:
        best_degree = degree
        best_z = z
        best_res = res

p = np.poly1d(best_z)

one_should_be = p(1)
one_denom = 1 / one_should_be

print(f"The value for similarity we want for exact match of sentences of length one is {one_should_be}")
print(f"The denominator when each sentence is length 1 should be {one_denom}")

try:
    import matplotlib.pyplot as plt
    plt.plot(x, y, label='real')
    plt.plot(np.arange(0, args.max), p(np.arange(0, args.max)), '--', label="fit")
    plt.xlim(0, args.max)

    plt.title(f"Best fit with degree={best_degree}, residual={best_res}, One should use {one_denom} as the denominator")

    plt.show()
except ImportError:
    pass
