# Stage 1
# Find the number of times a substring repeats in a string.
# Given multiple possible substrings, proceed with calculations
# for the substring that repeats the most times.

def substring_check(s):
    n = len(s)
    # Check all possible lengths of substrings, with smaller lengths
    # preferred (as that leads to more repetitions).
    for l in range(1, n // 2 + 1):
        if n % l == 0:
            substrings = [s[i:i+l] for i in range(0, n, l)]
            # Check if substring is uniquely repeated
            if len(set(substrings)) == 1:
                return n // l
    # Otherwise, the only substring is the entire string itself.
    return 1


def test():
    print(substring_check("abcabcabcabc"))


test()
