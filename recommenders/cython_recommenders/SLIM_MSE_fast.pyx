import numpy as np
import time
from tqdm import tqdm
from libc.stdlib cimport rand, srand, RAND_MAX
from scipy import sparse

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
def train(urm, learning_rate_input, epochs, n_samples):

    urm_coo = urm.tocoo()
    cdef int n_items = urm.shape[1]
    cdef int n_interactions = urm.nnz
    cdef long samples = n_samples
    cdef int[:] urm_coo_row = urm_coo.row
    cdef int[:] urm_coo_col = urm_coo.col
    cdef double[:] urm_coo_data = urm_coo.data
    cdef int[:] urm_indices = urm.indices
    cdef int[:] urm_indptr = urm.indptr
    cdef double[:] urm_data = urm.data

    cdef double[:,:] item_item_S = np.zeros((n_items, n_items), dtype = np.float)
    cdef double learning_rate = learning_rate_input
    cdef double loss = 0.0
    cdef long start_time
    cdef double true_rating, predicted_rating, prediction_error, profile_rating
    cdef int start_profile, end_profile
    cdef int index, sample_num, user_id, item_id, profile_item_id

    for n_epoch in range(epochs):

        loss = 0.0
        ts = time.time()

        for sample_num in range(samples):

            # Randomly pick sample
            index = rand() % n_interactions

            user_id = urm_coo_row[index]
            item_id = urm_coo_col[index]
            true_rating = urm_coo_data[index]

            # Compute prediction
            start_profile = urm_indptr[user_id]
            end_profile = urm_indptr[user_id+1]
            predicted_rating = 0.0

            for index in range(start_profile, end_profile):
                profile_item_id = urm_indices[index]
                profile_rating = urm_data[index]
                predicted_rating += item_item_S[profile_item_id,item_id] * profile_rating

            # Compute prediction error, or gradient
            prediction_error = true_rating - predicted_rating
            loss += prediction_error**2

            # Update model, in this case the similarity
            for index in range(start_profile, end_profile):
                profile_item_id = urm_indices[index]
                profile_rating = urm_data[index]
                item_item_S[profile_item_id,item_id] += learning_rate * prediction_error * profile_rating

        #print("|Epoch: {} | loss: {:.4f} | Time: {:.2f} |".format(n_epoch, loss/(sample_num), time.time() - ts))
        
        
    return sparse.csr_matrix(sparse.coo_matrix(item_item_S))
