from fractions import gcd
import random
def phi(b):
    """The Phi function, by Euler, tells how many numbers below 'b' do not share a common factor. Used in rsa functions."""
    ans = 0
    for i in range(b):
        if gcd(i, b) == 1:
            ans += 1
    return ans
