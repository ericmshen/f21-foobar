# Stage 4
import time
import fractions
import math
import itertools


# PROBLEM 1
# Like the problems in Stage 3, the descriptions become longer. Here is
# a description of this problem: https://codereview.stackexchange.com/questions/264310/google-foobar-distract-the-trainers.

# This solution draws from the Blossom algorithm. A good writeup I found post-completion
# is https://yifan.lu/2017/09/13/foobar-blossoms-and-isomorphism/. The actual
# algorithm code was adapted from Prof. Eppstein at UC Irvine:
# https://www.ics.uci.edu/~eppstein/PADS/CardinalityMatching.py.

# Some code to understand the problem by simulating the game.
def play_game(a, b):
    print(a, b)
    prevs = []
    while a != b:
        if (a, b) in prevs:
            return True  # "Looped"
        prevs.append((a, b))
        if a < b:
            a, b = 2*a, b-a
        else:
            a, b = a-b, 2*b
        print(a, b)
    return False  # "Done"


print(play_game(3, 11))


def game_result(a, b):
  # Assumes a < b.
    prevs = []
    while a != b:
        if (a, b) in prevs:
            return True
        prevs.append((a, b))
        a, b = min(2*a, b-a), max(2*a, b-a)
    return False

# symmetric: if (i, j) terminates then so will (j, i)
# and if (i, j) infinite loops then so will (j, i)


def creates_loop(a, b):
    # No game to play if bananas are the same
    if a == b:
        return False
    # Divide by GCD, so numbers are in simplest terms
    g = fractions.gcd(a, b)
    s, l = min(a, b)/g, max(a, b)/g
    # Game terminates when sum is a power of 2
    if (s+l & (s+l-1)) == 0:
        return False
    return True


def test_q1_preliminaries():
    print("Starting")
    for i in range(1, 100):
        for j in range(i+1, 200):
            if game_result(i, j) != creates_loop(i, j):
                print("FAILED:", i, j)
            # if not game_result(i, j):
                # print(i, j)
                # print("(" + str(i) + ", " + str(j) + "), sum " + str(i+j) + " factor " + str(float(j)/float(i)))
                # sums.append(i+j)
    print("Finished")

# Represent players in banana_list starting from 0 as nodes.
# If two players at indices i, j can be matched up to infinite loop, there is
# an edge between them. This forms an undirected graph.

# We use an adjacency matrix, where indices [i][j] and [j][i] is a bool
# indicating if there is an edge, i.e. if i, j can be put into a loop. Matrices
# work well for dense graphs (which this is).
# Now need a max matching on this graph.


def generate_pair_matrix(banana_list):
    l = len(banana_list)
    # mat = [[False] * l for _ in range(l)]
    mat = [[] for _ in range(l)]
    for i in range(l):
        for j in range(i):
            res = creates_loop(banana_list[i], banana_list[j])
            # mat[i][j] = res
            # mat[j][i] = res
            if res:
                mat[i].append(j)
                mat[j].append(i)
    return mat


# Test performance (to ensure time limit is not exceeded).
begin = time.time()

m = generate_pair_matrix([1, 7, 3, 21, 13, 19])

end1 = time.time()


def print_matrix(m):
    for r in m:
        print(r)


end2 = time.time()

print("Time to generate matrix:", str(end1-begin))
print("Time to print matrix:", str(end2-end1))


# The implementation of the Blossom algorithm.
class UnionFind:
    # Each unionFind object is a family of disjoint sets supporting the methods:
    # - X[obj] returns the name of the set that obj is in.
    #   Each set is named by an arbitrary member. If the item is not part of
    #   a set in X, a new singleton set is added. Path compression is also
    #   applied if possible to shorten later access times.
    # - X.union(item1, item2, ...) merges the sets containing each item.
    #   If any item is not yet part of a set in X, it is added to the union.
    # These disjoint sets are represented by a dictionary with a key-value
    # pair for each object: self.parents[object] iterated until we get to an
    # unchanging value is the name of the set. In essence, self.parents serves
    # as the "parent" pointer for each element when viewed as a node (using the)
    # conventional UnionFind visualization).

    def __init__(self):
        # weights[obj]: the "depth" or "rank" of the current union-find set that
        # obj is part of. Used when merging sets.
        self.weights = {}
        # parents: as described above, a way to access the parent of each object.
        self.parents = {}

    def __getitem__(self, object):
        # check for previously unknown object
        if object not in self.parents:
            # initialize a new set
            self.parents[object] = object
            self.weights[object] = 1
            return object

        # follow parent "pointers" until we get to the root
        # (which is its own parent)
        path = [object]
        root = self.parents[object]
        while root != path[-1]:
            path.append(root)
            root = self.parents[root]

        # apply path compression
        for ancestor in path:
            self.parents[ancestor] = root
        return root

    def __iter__(self):
        return iter(self.parents)

    def union(self, *objects):
        # use __getitem__ as above to instantiate each object if needed
        roots = [self[x] for x in objects]
        # get the root which has larger weight to add to
        heaviest = max([(self.weights[r], r) for r in roots])[1]
        for r in roots:
            if r != heaviest:
                # update the roots
                self.weights[heaviest] += self.weights[r]
                self.parents[r] = heaviest


def matching(G):
    vertices = [i for i in range(len(G))]
    matching = {}

    def augment():
        # Search for a single augmenting path. Return true if one is found.
        #
        # leader: union-find structure; the leader of a blossom is one
        # of its vertices (not necessarily topmost), and leader[v] always
        # points to the leader of the largest blossom containing v
        #
        # S: dictionary of blossoms at even levels of the structure tree.
        # Dictionary keys are names of blossoms (as returned by the union-find
        # data structure) and values are the structure tree parent of the blossom
        # (a T-node, or the top vertex if the blossom is a root of a structure tree).
        #
        # T: dictionary of vertices at odd levels of the structure tree.
        # Dictionary keys are the vertices; T[x] is a vertex with an unmatched
        # edge to x.  To find the parent in the structure tree, use leader[T[x]].
        #
        # unexplored: collection of unexplored vertices within blossoms of S
        #
        # base: if x was originally a T-vertex, but becomes part of a blossom,
        # base[t] will be the pair (v,w) at the base of the blossom, where v and t
        # are on the same side of the blossom and w is on the other side.

        leader = UnionFind()
        S = {}
        T = {}
        unexplored = []
        base = {}

        def blossom(v, w, a):
            # Create a new blossom from edge v-w with common ancestor a.
            def findSide(v, w):
                path = [leader[v]]
                b = (v, w)   # new base for all T nodes found on the path
                while path[-1] != a:
                    tnode = S[path[-1]]
                    path.append(tnode)
                    base[tnode] = b
                    unexplored.append(tnode)
                    path.append(leader[T[tnode]])
                return path

            a = leader[a]
            path1, path2 = findSide(v, w), findSide(w, v)
            leader.union(*path1)
            leader.union(*path2)
            S[leader[a]] = S[a]  # update structure tree

        topless = object()  # should be unequal to any graph vertex

        def alternatingPath(start, goal=topless):
            # Return sequence of vertices on alternating path from start to goal.
            # The goal must be a T node along the path from the start to
            # the root of the structure tree. If goal is omitted, we find
            # an alternating path to the structure tree root.
            path = []
            while 1:
                while start in T:
                    v, w = base[start]
                    vs = alternatingPath(v, start)
                    vs.reverse()
                    path += vs
                    start = w
                path.append(start)
                if start not in matching:
                    return path     # reached top of structure tree, done!
                tnode = matching[start]
                path.append(tnode)
                if tnode == goal:
                    return path     # finished recursive subpath
                start = T[tnode]

        def alternate(v):
            # Unmatch v by alternating the path to the root of its structure tree.
            path = alternatingPath(v)
            path.reverse()
            for i in range(0, len(path)-1, 2):
                matching[path[i]] = path[i+1]
                matching[path[i+1]] = path[i]

        def addMatch(v, w):
            # Given an S-S edge v, w connecting disjoint trees in the forest,
            # find the augmenting and augment the matching to add the edge (v, w).
            alternate(v)
            alternate(w)
            matching[v] = w
            matching[w] = v

        def ss(v, w):
            # Handle detection of an S-S edge in augmenting path search.
            # Like augment(), returns true iff the matching size was increased.

            if leader[v] == leader[w]:
                return False        # self-loop within blossom, ignore

            # parallel search up two branches of structure tree
            # until we find a common ancestor of v and w
            path1, head1 = {}, v
            path2, head2 = {}, w

            def step(path, head):
                head = leader[head]
                parent = leader[S[head]]
                if parent == head:
                    return head     # found root of structure tree
                path[head] = parent
                path[parent] = leader[T[parent]]
                return path[parent]

            while True:
                head1 = step(path1, head1)
                head2 = step(path2, head2)
                if head1 == head2:
                    blossom(v, w, head1)
                    return False
                if leader[S[head1]] == head1 and leader[S[head2]] == head2:
                    addMatch(v, w)
                    return True
                if head1 in path2:
                    blossom(v, w, head1)
                    return False
                if head2 in path1:
                    blossom(v, w, head2)
                    return False

        # Start of main augmenting path search code.
        for v in vertices:
            if v not in matching:
                S[v] = v
                unexplored.append(v)

        current = 0     # index into unexplored, in FIFO order so we get short paths
        while current < len(unexplored):
            v = unexplored[current]
            current += 1

            for w in G[v]:
                if leader[w] in S:  # S-S edge: blossom or augmenting path
                    if ss(v, w):
                        return True

                elif w not in T:    # previously unexplored node, add as T-node
                    T[w] = v
                    u = matching[w]
                    if leader[u] not in S:
                        S[u] = w    # and add its match as an S-node
                        unexplored.append(u)

        return False    # ran out of graph without finding an augmenting path

    # augment the matching until it is maximum
    while augment():
        pass

    return matching


def solution(banana_list):
    G = generate_pair_matrix(banana_list)
    m = matching(G)
    # print(m)
    return len(banana_list) - len(m)


def test_q1():
    G = []
    for i in range(100):
        G.append([j for j in range(100) if j != i])
    print(G)
    print(matching(G))
    print(len(matching(G)))

    print(solution([1, 1]))
    print(solution([1, 7, 3, 21, 13, 19]))
    print(solution([1]))

# PROBLEM 2
# The problem description can be found at
# https://stackoverflow.com/questions/46898131/google-foobar-free-the-bunny-prisoners-clarification.

# E.g. (5, 3): 5 total subsets, require choice of 3
# range is 0...9, 10 numbers, and length of each subset is 6
# individual subsets are listed in increasing order.

# Prelimary testing code.
# d1 = [[0, 1, 2, 3, 4, 5], [0, 1, 2, 6, 7, 8], [0, 3, 4, 6, 7, 9], [1, 3, 5, 6, 8, 9], [2, 4, 5, 7, 8, 9]]
# l1 = len(d1)
# c1 = 0

# # any combination of 3 subsets covers the entire range 0...9
# for i in range(l1-2):
#     for j in range(i+1, l1-1):
#         for k in range(j+1, l1):
#             c1 += 1
#             print(list(sorted(set(d1[i]) | set(d1[j]) | set(d1[k]))))

# # but any combination of 2 subsets does not
# for i in range(l1-1):
#     for j in range(i+1, l1):
#         print(list(sorted(set(d1[i]) | set(d1[j]))))

# print(c1)

# d2 = [[0, 1, 2, 3], [0, 1, 2, 4], [0, 1, 3, 4], [0, 2, 3, 4], [1, 2, 3, 4]]
# l2 = len(d2)
# c2 = 0
# for i in range(l1-1):
#     for j in range(i+1, l1):
#         c2 += 1
#         print(list(sorted(set(d2[i]) | set(d2[j]))))

# print(c2)

# d3 = [[0, 1], [0, 2], [1, 3], [2, 4], [3, 4]]
# l3 = len(d3)
# c3 = 0
# for i in range(l3):
#     s = set()
#     for j in [k for k in range(l3) if k != i]:
#         s |= set(d3[j])
#     print(sorted(list(s)))
#     c3 += 1

# print(c3)


def test_optimal_partition(subsets, k):
    if k == 0:
        return
    assert (len(subsets) >= k)
    total = set()
    for i in range(k):
        total |= set(subsets[i])
    print(sorted(list(total)))
    valid_combs = itertools.combinations(subsets, k)
    for comb in valid_combs:
        union = set()
        for subset in comb:
            union |= set(subset)
        if union != total:
            print("FAILED", comb)
            return
    invalid_combs = itertools.combinations(subsets, k-1)
    for comb in invalid_combs:
        union = set()
        for subset in comb:
            union |= set(subset)
        if union == total:
            print("FAILED", comb)
            return
    print("PASSED")

# subsets = [[1], [2], [3], [4], [5]]
# combs = itertools.combinations(subsets, 3)
# for comb in combs:
#     print(comb)


def nCr(n, r):
    f = math.factorial
    return f(n) // f(r) // f(n-r)

# print(nCr(5, 3))
# print(nCr(10, 4))


def find_optimal_partition(num_bunnies, num_required):
    assert(num_bunnies >= num_required)
    if num_required == 0:
        return [[] for _ in range(num_bunnies)]
    if num_required == 1:
        return [[0] for _ in range(num_bunnies)]
    if num_required == num_bunnies:
        return [[i] for i in range(num_bunnies)]
    freq = num_bunnies - num_required + 1
    result = [[] for _ in range(num_bunnies)]
    count = 0
    for comb in itertools.combinations(range(num_bunnies), freq):
        for ind in comb:
            result[ind].append(count)
        count += 1
    return result


def test_q2():
    sixThree = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                [0, 1, 2, 3, 4, 5, 10, 11, 12, 13],
                [0, 1, 2, 6, 7, 8, 10, 11, 12, 14],
                [0, 3, 4, 6, 7, 9, 10, 11, 13, 14],
                [1, 3, 5, 6, 8, 9, 10, 12, 13, 14],
                [2, 4, 5, 7, 8, 9, 11, 12, 13, 14]]

    sixFour = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
               [0, 1, 2, 3, 10, 11, 12, 13, 14, 15],
               [0, 4, 5, 6, 10, 11, 12, 16, 17, 18],
               [1, 4, 7, 8, 10, 13, 14, 16, 17, 19],
               [2, 5, 7, 9, 11, 13, 15, 16, 18, 19],
               [3, 6, 8, 9, 12, 14, 15, 17, 18, 19]]

    fourThree = [[0, 1, 2], [0, 3, 4], [1, 3, 5], [2, 4, 5]]
    fiveFour = [[0, 1, 2, 3], [0, 4, 5, 6], [
        1, 4, 7, 8], [2, 5, 7, 9], [3, 6, 8, 9]]
    sixFive = [[0, 1, 2, 3, 4], [0, 5, 6, 7, 8], [1, 5, 9, 10, 11],
               [2, 6, 9, 12, 13], [3, 7, 10, 12, 14], [4, 8, 11, 13, 14]]

    test_optimal_partition(sixThree, 3)
    test_optimal_partition(sixFour, 4)
    test_optimal_partition(fourThree, 3)
    test_optimal_partition(fiveFour, 4)
    test_optimal_partition(sixFive, 5)

    assert(find_optimal_partition(4, 3) == fourThree)
    assert(find_optimal_partition(5, 4) == fiveFour)
    assert(find_optimal_partition(6, 3) == sixThree)
    assert(find_optimal_partition(6, 4) == sixFour)
    assert(find_optimal_partition(6, 5) == sixFive)

    for i in range(1, 7):
        for j in range(i+1):
            print(i, j)
            print(find_optimal_partition(i, j))
            test_optimal_partition(find_optimal_partition(i, j), j)

    print(find_optimal_partition(9, 4))
    test_optimal_partition(find_optimal_partition(9, 4), 4)
    print(find_optimal_partition(9, 5))
    test_optimal_partition(find_optimal_partition(9, 5), 5)
