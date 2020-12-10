



var = [0, 0, 0]
len_recs = len(var)
granularity = 5
temp = 0

u = 1 / ( len_recs * granularity ) 
i = granularity * len_recs

print(2.33 % 2)

'''
for r in range(0, len_recs):
    for r2 in range(0, len_recs):
        for it in range(0, i**(r+1)):
            var[r2] += u if it % i**r == 0 else 0
            #temp+=1
            print('|{:.3f}|{:.3f}|{:.3f}|'.format(var[0], var[1], var[2]))
'''
from math import factorial # python math library

i = 5               # i is the lexicographic index (counting starts from 0)
n = 3               # n is the length of the permutation
p = range(1, n + 1) # p is a list from 1 to n

for k in range(1, n + 1): # k goes from 1 to n
    f = factorial(n - k)  # compute factorial once per iteration
    d = i // f            # use integer division (like division + floor)
    print(p[d]),          # print permuted number with trailing space
    p.remove(p[d])        # delete p[d] from p
    i = i % f             # reduce i to its remainder
        
  

    

    