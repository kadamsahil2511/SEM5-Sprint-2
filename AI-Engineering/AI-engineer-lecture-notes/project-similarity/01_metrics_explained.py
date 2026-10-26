"""01 — The three similarity metrics, hand-rolled in NumPy.

Teaching goal: dot product, Euclidean distance and cosine similarity are each
ONE line of NumPy. Four tiny 3-D vectors are enough to show where they
agree and where they disagree. No ML libraries, no formulas to memorize:
read the code, read the table, read the three observations at the bottom.

Run:  python 01_metrics_explained.py
"""

from itertools import combinations

import numpy as np


def dot_product(a, b) -> float:
    """Multiply element-wise, add up. Big = same direction AND big magnitudes."""
    return float(np.sum(a * b))


def euclidean_distance(a, b) -> float:
    """Straight-line distance between the two arrow tips. Small = close."""
    return float(np.sqrt(np.sum((a - b) ** 2)))


def cosine_similarity(a, b) -> float:
    """Angle only: dot product divided by both lengths. 1 = same, 0 = perpendicular, -1 = opposite."""
    return float(np.sum(a * b) / (np.sqrt(np.sum(a * a)) * np.sqrt(np.sum(b * b))))


vectors = {
    "v1": np.array([1.0, 0.0, 0.0]),   # unit, along x
    "v2": np.array([2.0, 0.0, 0.0]),   # same direction, 2x magnitude
    "v3": np.array([0.0, 1.0, 0.0]),   # unit, perpendicular to v1
    "v4": np.array([-1.0, 0.0, 0.0]),  # unit, opposite of v1
}

print("THE VECTORS")
for name, v in vectors.items():
    print(f"  {name} = {v.tolist()}   length = {np.sqrt(np.sum(v * v)):.2f}")

print("\nALL PAIRS")
header = f"  {'pair':<10}{'dot':>10}{'euclidean':>12}{'cosine':>10}"
print(header)
print("  " + "-" * (len(header) - 2))
for (na, a), (nb, b) in combinations(vectors.items(), 2):
    print(f"  {na + ' vs ' + nb:<10}"
          f"{dot_product(a, b):>10.2f}"
          f"{euclidean_distance(a, b):>12.2f}"
          f"{cosine_similarity(a, b):>10.2f}")

print("\nWHAT TO NOTICE")
print("  1. v1 vs v2: SAME DIRECTION but dot=2 (big!), euclidean=1, cosine=1.")
print("     → dot product rewards magnitude, cosine ignores it.")
print("  2. v1 vs v3: perpendicular. dot=0, euclidean=1.41, cosine=0.")
print("     → all three agree on \"unrelated\".")
print("  3. v1 vs v4: opposite. dot=-1, euclidean=2, cosine=-1.")
print("     → cosine shows the DIRECTION (opposite); euclidean just shows \"far\".")
