# Stage 3
import math

# PROBLEM 1: Count the number of lucky triples in a list. A lucky triple
# is defined as three numbers in increasing positions in a list where
# the next number divides the previous one, i.e. i < j < k, lst[i] divides
# lst[j], and lst[j] divides lst[k].


def count_lucky_triples(l):
    length = len(l)
    if length < 3:
        return 0
    lucky_triples = 0
    # Create a list of lists, which record the indices of factors
    # for the jth element in the list that come before that element.
    factor_indices = [[] for _ in range(length)]
    for i in range(1, length):
        for j in range(i):
            if l[i] % l[j] == 0:
                factor_indices[i].append(j)
    # Now iterate through the 3rd element and beyond in the list. For
    # each element, check all of its factors (indices j), and add the
    # number of factors that come before that factor. Each corresponds
    # to a unique lucky triple.
    for i in range(2, length):
        for j in factor_indices[i]:
            lucky_triples += len(factor_indices[j])
    return lucky_triples


def test_q1():
    print(count_lucky_triples([1, 1, 1]))


# PROBLEM 2: See https://github.com/ivanseed/google-foobar-help/blob/master/challenges/bomb_baby/bomb_baby.md
# for a description. This solution was, however, developed independently of
# whatever may have been written online about the problem.
def bombs(x, y):
    larger, smaller = max(int(x), int(y)), min(int(x), int(y))
    count = 0
    while (larger != 1 or smaller != 1):
        if larger < 1 or smaller < 1:
            return "impossible"
        if smaller == 1:
            return str(count + larger - 1)
        count += larger // smaller
        larger, smaller = max(larger % smaller, smaller), min(
            larger % smaller, smaller)
    return str(count)


def test_q2():
    print(bombs("2", "1"))
    print(bombs("5", "1"))
    print(bombs("4", "7"))
    print(bombs("2", "4"))


# PROBLEM 3: See https://stackoverflow.com/questions/68429960/google-foobar-fuel-injection-perfection
# for a description. This solution was also developed independently.
def naive(n):
    p = int(n)
    if p == 1:
        return 0
    if p % 2 == 0:
        return naive(p/2) + 1
    return min(naive(p-1), naive(p+1)) + 1


def naive_better(n):
    p = int(n)
    if p == 1:
        return 0
    if math.log(p, 2).is_integer():
        return int(math.log(p, 2))
    if p % 2 == 0:
        return naive(p / 2) + 1
    return min(naive(p+1), naive(p-1)) + 1


def fuel_pellet_steps(n):
    p = int(n)
    count = 0
    while p != 1:
        count += 1
        if p % 2 == 0:
            p >>= 1
        elif p % 4 == 1 or p == 3:
            p -= 1
        else:
            p += 1
    return count


def test_q3():
    print(fuel_pellet_steps("15"))
    print("Beginning")
    for i in range(1, 512):
        # print(i, naive_better(str(i)), fuel_pellet_steps(str(i)))
        if naive_better(str(i)) != fuel_pellet_steps(str(i)):
            print("FAILED:", str(i), naive_better(
                str(i)), fuel_pellet_steps(str(i)))
    print("All done!")
