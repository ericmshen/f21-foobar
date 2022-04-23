# Stage 5

# A description of the problem can be viewed here:
# https://randomds.com/2022/03/11/google-foobar-challenge-level-5-expanding-nebula/.
# This site also contains a solution different from the one presented here.
# The essence of this problem is playing around with a simple cellular automaton,
# the aim being finding previous steps given a current state.

# Helper functions to visualize the problem.
def draw_grid(grid):
    for row in grid:
        for elem in row:
            if (elem):
                print("■"),
            else:
                print("□"),
        print(" ")
    print(" ")


def update_cell(grid, i, j):
    return [grid[i][j], grid[i][j+1], grid[i+1][j], grid[i+1][j+1]].count(True) == 1


def next_grid(grid):
    w, h = len(grid[0]), len(grid)
    new_grid = [[False] * (w-1) for _ in range(h-1)]
    for i in range(h-1):
        for j in range(w-1):
            new_grid[i][j] = update_cell(grid, i, j)
    return new_grid


# Examples given on the site.
# 11567; 4 x 10
ex1 = [[True, True, False, True, False, True, False, True, True, False], [True, True, False, False, False, False, True, True, True, False], [
    True, True, False, False, False, False, False, False, False, True], [False, True, False, False, False, False, True, True, False, False]]
# 4; 3 x 3
ex2 = [[True, False, True], [False, True, False], [True, False, True]]
ex2_prev = [[False, False, True, False], [False, True, False, False], [
    True, False, False, False], [False, False, False, True]]
# 254; 5 x 8
ex3 = [[True, False, True, False, False, True, True, True], [True, False, True, False, False, False, True, False], [True, True, True, False,
                                                                                                                    False, False, True, False], [True, False, True, False, False, False, True, False], [True, False, True, False, False, True, True, True]]

# Test preliminary functions for visualization:
# draw_grid(ex1)
# print(" ")
# draw_grid(ex2)
# print(" ")
# draw_grid(ex3)
# print(" ")

# grid = ex3
# for i in range(4):
#     draw_grid(grid)
#     print(" ")
#     grid = next_grid(grid)

target = [[False], [True], [False]]


def rep(arr):
    return int("".join(str(i) for i in arr), 2)


prev_true = [[[1, 0], [0, 0]], [[0, 1], [0, 0]],
             [[0, 0], [1, 0]], [[0, 0], [0, 1]]]


prev_false = [[[0, 0], [0, 0]], [[1, 1], [0, 0]], [[1, 0], [1, 0]], [[0, 1], [0, 1]], [[0, 0], [1, 1]], [[0, 1], [
    1, 0]], [[1, 0], [0, 1]], [[0, 1], [1, 1]], [[1, 0], [1, 1]], [[1, 1], [0, 1]], [[1, 1], [1, 0]], [[1, 1], [1, 1]]]


prev_squares = [prev_false, prev_true]


def prev_rows(row):
    prevs = prev_squares[row[0]]
    for elem in row[1:]:
        updated_prevs = []
        prev_elem_sqs = prev_squares[elem]
        for p in prevs:
            for sq in prev_elem_sqs:
                if p[0][-1] == sq[0][0] and p[1][-1] == sq[1][0]:
                    updated_prevs.append(
                        [p[0] + [sq[0][1]], p[1] + [sq[1][1]]])
        if not updated_prevs:
            return []
        prevs = updated_prevs
    return [(rep(p[0]), rep(p[1])) for p in prevs]

# Test helper function:
# for prev in prev_rows([1, 0, 1]):
#     draw_grid(prev)


def summary(row):
    prevs = prev_rows(row)
    d = dict()
    for k in list(set(p[1] for p in prevs)):
        d[k] = [p[0] for p in prevs if p[1] == k]
    return d


def counts(row):
    prevs = prev_rows(row)
    c = dict()
    for p in prevs:
        if p[1] not in c.keys():
            c[p[1]] = 1
        else:
            c[p[1]] += 1
    return c


def solution(g):
    grid = []
    h, w = len(g), len(g[0])
    for i in range(w):
        row = []
        for j in range(h):
            row.append(int(g[j][i]))
        grid.append(row)
    ctr = counts(grid[0])
    # print(ctr)
    for row in grid[1:]:
        s = summary(row)
        updated_ctr = dict()
        for k, arr in s.viewitems():
            updated_ctr[k] = sum([ctr[i] for i in arr if i in ctr.keys()])
        # print(updated_ctr)
        ctr = updated_ctr
    return sum(ctr.values())


def test():
    print(prev_rows([1, 0, 1]))
    print(len(prev_rows([1, 0, 1])))
    print(summary(prev_rows([1, 0, 1])))
    print(prev_rows([0, 1, 0]))
    print(len(prev_rows([0, 1, 0])))
    print(summary(prev_rows([0, 1, 0])))
    print(prev_rows([0, 1, 1, 1, 0, 1, 0, 0, 0]))
    print(len(prev_rows([0, 1, 1, 1, 0, 1, 0, 0, 0])))

    print(solution(ex1))
    print(solution(ex2))
    print(solution(ex3))

    for i in range(100):
        arr = [int(c) for c in bin(i * 100000)[2:].zfill(9)][:10]
        print(len(prev_rows(arr)))
