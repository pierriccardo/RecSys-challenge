
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

