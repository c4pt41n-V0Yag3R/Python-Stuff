#m^e mod n = c
#c^d mod n = m
#m^phi(n) mod n == 1 where m & n dont share a common factor
#since 1^k = 1, m^(k * phi(n)) mod n == 1, too.
#since 1*m = m, m* (m^(k * phi(n)) mod n) == m
#^^^^simplifies to m^(k * phi(n) + 1) mod n == m
import math
from fractions import gcd
import random
i = 1
j = 1
t = 1
def is_prime(a):
    return all(a % i for i in range(2, a))
while True:
    p1 = random.randrange(100.0000)#gens the 1st random prime
    if is_prime(p1):
        if p1 == 0 or p1 == 1:
            i+=1
            continue
        else:
            print("First Random Prime Found on attempt "+str(i)+": "+str(p1))
            break
    i+=1
while True:
    p2 = random.randrange(100.0000)#gens the second random prime
    if is_prime(p2):
        if p2 == 0 or p2 == 1:
            j+=1
            continue
        else:
            print("Second Random Prime Found on attempt "+str(j)+": "+str(p2))
            break
    j+=1
n = p1 * p2
print("n = p1 * p2 = "+str(n))
phi_n = (p1 - 1) * (p2 - 1)#phi(n) = how many numbers below n share no factors w/ n. Given Definition of a prime, phi(any_prime_num) is always any_prime_num - 1.
print("phi_n = (p1 - 1) * (p2 - 1) = "+str(phi_n))
while True:
    e = random.randrange(10)#gens the 3rd random prime
    if e % 2 != 0:
        if gcd(phi_n, e) > 1:
            t+=1
            continue
        if e == p1 or e == p2:
            t+=1
            continue
        else:
            print("Public Random Prime(is e)Found on attempt "+str(t)+": "+str(e))
            break
k = random.randrange(e)
print("num used to find d(is k): "+str(k))
d = (k * phi_n + 1)/e
print("PRIVATE key(is d): "+str(d))
#pub_key = [n, e]
#priv_key = [d, k, p1, p2, phi_n]
m = input("Type an int: ")
if gcd(n, int(m)) == 0:
    quit()
c = math.fmod((int(m)**e), n)
print("Encrypted: "+str(c))
u = math.fmod((c**d), n)
print("Decrypted: "+str(u))
if m == u:
    print("Successful!!")
else:
    print("Unsuccessful....")
