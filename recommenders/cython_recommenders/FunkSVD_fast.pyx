import numpy as np
import time
from tqdm import tqdm

from libc.stdlib cimport rand, srand, RAND_MAX

# These can be used to remove checks done at runtime (e.g. null pointers etc). Be careful as they can introduce errors
# For example cdivision performs the C division which can result in undesired integer divisions where
# floats are instead required
import cython
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.overflowcheck(False)

def fit(urm, epochs, steps_per_epoch, num_factors, learning_rate, regularization):

    urm_coo = urm.tocoo()
    n_users, n_items = urm_coo.shape
    cdef int n_interactions = urm.nnz

    cdef int sample_num, sample_index, user_id, item_id, factor_index
    cdef double rating, predicted_rating, prediction_error

    #cdef int num_factors = 10 #n_factors
    #cdef double learning_rate = 1e-4 #lr
    #cdef double regularization = 1e-5 #regularization

    cdef int[:] urm_coo_row = urm_coo.row
    cdef int[:] urm_coo_col = urm_coo.col
    cdef double[:] urm_coo_data = urm_coo.data

    cdef double[:,:] user_factors = np.random.random((n_users, num_factors))
    cdef double[:,:] item_factors = np.random.random((n_items, num_factors))
    cdef double H_i, W_u
    cdef double item_factors_update, user_factors_update

    cdef double loss = 0.0
    cdef long start_time = time.time()

    for n_epoch in range(epochs):

        loss = 0.0
        start_time = time.time()

        for sample_num in tqdm(range(steps_per_epoch)):

            # Randomly pick sample
            sample_index = rand() % n_interactions

            user_id = urm_coo_row[sample_index]
            item_id = urm_coo_col[sample_index]
            rating = urm_coo_data[sample_index]

            # Compute prediction
            predicted_rating = 0.0

            for factor_index in range(num_factors):
                predicted_rating += user_factors[user_id, factor_index] * item_factors[item_id, factor_index]

            # Compute prediction error, or gradient
            prediction_error = rating - predicted_rating
            loss += prediction_error**2

            # Copy original value to avoid messing up the updates
            for factor_index in range(num_factors):

                H_i = item_factors[item_id,factor_index]
                W_u = user_factors[user_id,factor_index]

                user_factors_update = prediction_error * H_i - regularization * W_u
                item_factors_update = prediction_error * W_u - regularization * H_i

                user_factors[user_id,factor_index] += learning_rate * user_factors_update
                item_factors[item_id,factor_index] += learning_rate * item_factors_update

        print("|Epoch: {} | loss: {:.2f} |".format(n_epoch, loss/(sample_num)))

    return np.array(user_factors), np.array(item_factors)