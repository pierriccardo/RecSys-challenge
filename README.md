
'''

print(urm.indptr[1])
print(urm.indptr[1+1])

print(urm.tocsr())

urm is a sparse matrix in this format 
we have a tuple (user, item) data

if we want the item which a user interacted with
we need the user

user = 1

s = urm.indptr[user_id] 

s will point at the first row in which
user 1 appear, this row: (1, 2665)     1.0

e = urm.indptr[user_id + 1] is where 
the next user interactions list begin 

so we want to cut the item seen just from
user 1 and we do:

seen = urm.indices[s:e]

(0, 19467)    1.0
(1, 2665)     1.0
(1, 7494)     1.0
(1, 17068)    1.0
(1, 17723)    1.0
(1, 18131)    1.0
(1, 20146)    1.0
(2, 19337)    1.0
(2, 21181)    1.0
(3, 18736)    1.0
(3, 23037)    1.0

user_id = 3

a = urm.indptr[user_id]
b = urm.indptr[user_id+1]
seen = urm.indices[a:b]

print(seen)

https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html
'''

r1 r2

var = [0 0]
[0 0]
[0.1, 0]
+0.1
... 
[0.9, 0]
[0.0, 0.1]
[0.1, 0.1]
[0.2, 0.1]
...
[0.9, 0.1]
[0.0, 0.2]
[0.0, 0.2]
[0.0, 0.2]
[0.0, 0.2]

1
___
len(recs) * granularity

1
__     = 0.1
2 * 5

a 

1/(1*5) = 0.2

a: 0.0, 0.2, 0.4, 0.6, 0.8, 1.0

1/(2*5)

i = 1 (1 iterazione = 1/u)
0.0 -> 1.0 
a:0.1,0.2,0.3,0.4,0.5,...,0.9, 1.0, 0.1 - i  
b:.......................,0.0, 0.0, 0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,

a 0.1.....1.0,1.1....2.1....3.1.....
b 10*10
c 10*10*10

i=0
a -> update every iteration 1
a += u 
a - i
a:0.1,0.2,0.3,0.4,0.5,...,0.9, 1.0,


i 0 -> len(recs) 

----------------

granularity = 5
len(recs) = 2
u = 1/(len(recs) * granularity) = 0.1
i = granularity * len(recs)

temp = 0

var = [0, 0]

for r in range(len(recs)):
    for it in range(i):
        var[r] += u if temp % i**r == 0 else 0

