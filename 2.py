# Stage 2
import functools
from queue import Queue  # in python 3+
# import Queue # in python 2.7

# PROBLEM 1: Sort version numbers in ascending order, where version numbers are of
# the form x.y.z.
# 3 layers to a version: major, minor, revision. Idea is to compare version
# numbers by interpreting them as lists, and then comparing element-wise.
depth = 3

# A base comparator function for two version numbers, given as arrays of length 3.


def compare_lists(l1, l2):
    if len(l1) == 0 and len(l2) == 0:
        return 0
    return l1[0] - l2[0] if l1[0] != l2[0] else compare_lists(l1[1:], l2[1:])


def compare(s1, s2):
    l1, l2 = [int(el) for el in s1.split(".")], [int(el)
                                                 for el in s2.split(".")]
    # Pad the lists with 0s, e.g. "2" -> "[2, 0, 0]"
    l1 += [0] * (depth - len(l1))
    l2 += [0] * (depth - len(l2))
    res = compare_lists(l1, l2)
    if res != 0:
        return res
    # If compare_lists returns 0, then the two versions are the same as lists,
    # e.g. "2", "2.0", and "2.0.0". In this case, the shorter string comes first
    return len(s1) - len(s2)


def sort_versions(l):
    # Using custom comparator as a key.
    return sorted(l, key=functools.cmp_to_key(compare))


def test_q1():
    print(sort_versions(["1.1.2", "1.0", "1.3.3", "1.0.12", "1.0.2"]))

# PROBLEM 2: Find the least number of moves for a knight to travel between
# two squares on a chessboard.

# Helper functions for representing chessboard positions as tuples of ints.


def coord_to_num(c):
    return 8*c[0] + c[1]


def num_to_coord(n):
    return (n // 8, n % 8)

# Manually define the knight's next possible moves


def next_moves(c):
    i, j = c
    moves = [(i-2, j-1), (i-2, j+1), (i-1, j-2), (i-1, j+2),
             (i+1, j-2), (i+1, j+2), (i+2, j-1), (i+2, j+1)]
    return [m for m in moves if min(m) >= 0 and max(m) <= 7]

# Use a BFS to compute the least number of moves required to move between
# two squares.


def knight_moves(src, dest):
    if src == dest:
        return 0
    parent = [-1] * 64
    parent[src] = src
    q = Queue()
    q.put(src)
    # BFS implementation using queue
    while not q.empty():
        curr = q.get()
        if curr == dest:
            l = 0
            while curr != src:
                l += 1
                curr = parent[curr]
            return l
        next_tiles = []
        for tile in [coord_to_num(c) for c in next_moves(num_to_coord(curr))]:
            if parent[tile] == -1:
                next_tiles.append(tile)
        for next in next_tiles:
            parent[next] = curr
            q.put(next)
            