"""
Static methods for learning the mixing weights given two components.
The learning is done with using a fixed-parameters EM algorithm, where the distribution parameters are
not updated in the M-Step but just the \pi probability. The E-Step is as usual.

This is not the cleanest code in the world but it works for now.

Author: Moshe Lichman
"""
from __future__ import division
import numpy as np
import time

from scipy.stats import dirichlet as dirch

from utils import log_utils as log
from utils.helpers import normalize_mat_row, convert_sparse_to_coo, col_vector


def _learn_mix_mult(alpha, mem_mult, mf_mult, val_data, num_em_iter=100, tol=0.00001):
    """
    Learning the mixing weights for mixture of two multinomials. Each observation is considered as a data point
    and the mixing weights (\pi) are learned using all the points.

    NOTE: In order for the algorithm to work, there can be no location that can get 0 probability by both the mem_mult
    and the mf_mult. In my runs, I use MPE to estimate the mf_mult while using MLE for the mum_mul. That way the mf_mult
    has no 0 values.


     INPUT:
    -------
        1. alpha:       <float / (2, ) ndarray>   Dirichlet prior for the pi learning. If <float> is given it is treated
                                                  as a flat prior. Has to be bigger than 1.
        2. mem_mult:    <(I, L) ndarray>    each row is the multinomial parameter according to the "self" data
        3. mf_mult:     <(I, L) ndarray>    each row is the multinomial parameter according to the matrix factorization
        4. val_data:    <(N, 3) ndarray>    each row is [ind_id, loc_id, counts]
        5. num_em_iter: <int>               number of em iterations
        6. tol:         <float>             convergence threshold

     OUTPUT:
    --------
        1. pi:  <(2, ) ndarray>     mixing weights.

     RAISE:
    -------
        1. ValueError:
                a. alphas are not bigger than 1
                b. the multinomial's rows don't sum to 1
                c. There is a location with both mults 0 (see NOTE)

    """
    if np.any(alpha <= 1):
        raise ValueError('alpha values have to be bigger than 1')

    if np.any(np.abs(np.sum(mem_mult, axis=1) - 1) > 0.001):
        raise ValueError('mem_mult param is not a multinomial -- all rows must sum to 1')

    if np.any(np.abs(np.sum(mf_mult, axis=1) - 1) > 0.001):
        raise ValueError('mf_mult param is not a multinomial -- all rows must sum to 1')

    if type(alpha) == float or type(alpha) == int:
        alpha = np.array([alpha, alpha])

    # Creating responsibility matrix and initializing it hard assignment on random
    log_like_tracker = [-np.inf]
    pi = np.array([0.5, 0.5])
    start = time.time()
    for em_iter in xrange(1, num_em_iter + 1):
        # Evey 5 iteration we will compute the posterior log probability to see if we converged.
        if em_iter % 5 == 0:
            data_log_like = pi[0] * mem_mult[val_data[:, 0].astype(int), val_data[:, 1].astype(int)] + \
                            pi[1] * mf_mult[val_data[:, 0].astype(int), val_data[:, 1].astype(int)]

            # The data likelihood was computed for each location, but it should be in the power of the number
            # of observations there, or a product in the log space.
            data_likelihood = np.log(data_log_like) * val_data[:, 2]

            prior_probability = dirch.logpdf(pi, alpha=alpha)
            log_likelihood = np.mean(data_likelihood + prior_probability)

            if np.abs(log_likelihood - log_like_tracker[-1]) < tol:
                log.info('[iter %d] [Reached convergence.]' % em_iter)
                print "log_likelihood:", log_like_tracker
                break

            log.debug('[iter %d] [Liklihood: [%.4f -> %.4f]]' % (em_iter, log_like_tracker[-1], log_likelihood))
            log_like_tracker.append(log_likelihood)

        # E-Step
        resp = [pi[0] * mem_mult[val_data[:, 0].astype(int), val_data[:, 1].astype(int)],
                pi[1] * mf_mult[val_data[:, 0].astype(int), val_data[:, 1].astype(int)]]

        if np.all(resp == 0):
            raise ValueError('0 mix probability')

        resp = np.array(resp).T
        resp = normalize_mat_row(resp)

        # M-Step. Only on the \pi with Dirichlet prior alpha > 1
        pi = np.sum(resp * col_vector(val_data[:, 2]), axis=0)
        print pi
        pi += alpha - 1
        pi /= np.sum(pi)

    total_time = time.time() - start
    log.debug('Finished EM. Total time = %d secs -- %.3f per iteration' % (total_time, total_time / em_iter))

    return pi


def learn_mix_mult_on_individual(alpha, mem_mult, mf_mult, val_mat, num_em_iter=10000, tol=0.00001):
    """
    Learning the mixing weights for mixture of two multinomials. Each individual learns mixing weights.

    NOTE: In order for the algorithm to work, there can be no location that can get 0 probability by both the mem_mult
    and the mf_mult. In my runs, I use MPE to estimate the mf_mult while using MLE for the mum_mul. That way the mf_mult
    has no 0 values.

     INPUT:
    -------
        1. alpha:       <float / (2, ) ndarray>   Dirichlet prior for the pi learning. If <float> is given it is treated
                                                  as a flat prior. Has to be bigger than 1.
        2. mem_mult:    <(I, L) ndarray>    each row is the multinomial parameter according to the "self" data
        3. mf_mult:     <(I, L) ndarray>    each row is the multinomial parameter according to the matrix factorization
        4. val_mat:     <(I, L) ndarray>    counts matrix to optimize on
        5. num_em_iter: <int>               number of em iterations
        6. tol:         <float>             convergence threshold

     OUTPUT:
    --------
        1. pis:  <(I, 2) ndarray>     each row is mixing weights for the i'th individual

     RAISE:
    -------
        1. ValueError:
                a. alphas are not bigger than 1
                b. the multinomial's rows don't sum to 1
                c. There is a location with both mults 0 (see NOTE)
    """
    I = mem_mult.shape[0]
    pis = np.zeros([I, 2])

    start = time.time()
    for i in range(I):
        if i % 200 == 0:
            log.info('Em for individual %d out of %d' % (i + 1, I))

        # The way the global em is implemented, allows me to simply call it with the i_val_data and it will only
        # compute the \pi as a function of that user.
        i_val_data = convert_sparse_to_coo(val_mat[i])

        # The learning method treats the multinomials as matrices. So I have to wrap it in an array.
        # All the rows in i_val_data are going to be 0 because I'm coverting a single row_vector.
        i_mem_mult = np.array([mem_mult[i]])
        i_mf_mult = np.array([mf_mult[i]])

        i_pi = _learn_mix_mult(alpha, i_mem_mult, i_mf_mult, i_val_data, num_em_iter, tol)

        pis[i] = i_pi

    total_time = time.time() - start
    log.info('Finished EM on users. Total time = %d secs -- %.3f per user' % (total_time, total_time / I))
    return pis


def learn_mix_mult_global(alpha, mem_mult, mf_mult, val_mat, num_em_iter=100000, tol=0.0001):
    """
    Learning the mixing weights for mixture of two multinomials globally for all users. Each observation is a point in
    model.

    NOTE: In order for the algorithm to work, there can be no location that can get 0 probability by both the mem_mult
    and the mf_mult. In my runs, I use MPE to estimate the mf_mult while using MLE for the mum_mul. That way the mf_mult
    has no 0 values.

     INPUT:
    -------
        1. alpha:       <float / (2, ) ndarray>   Dirichlet prior for the pi learning. If <float> is given it is treated
                                                  as a flat prior. Has to be bigger than 1.
        2. mem_mult:    <(I, L) ndarray>    each row is the multinomial parameter according to the "self" data
        3. mf_mult:     <(I, L) ndarray>    each row is the multinomial parameter according to the matrix factorization
        4. val_mat:     <(I, L) ndarray>    counts matrix to optimize on
        5. num_em_iter: <int>               number of em iterations
        6. tol:         <float>             convergence threshold

     OUTPUT:
    --------
        1. pis:  <(I, 2) ndarray>     each row is mixing weights for the i'th individual

     RAISE:
    -------
        1. ValueError:
                a. alphas are not bigger than 1
                b. the multinomial's rows don't sum to 1
                c. There is a location with both mults 0 (see NOTE)
    """
    log.info('Learning global mixing weights for all points')
    start = time.time()
    pi = _learn_mix_mult(alpha, mem_mult, mf_mult, convert_sparse_to_coo(val_mat), num_em_iter, tol)
    total_time = time.time() - start
    log.info('Finished EM on all data. Total time = %d secs' % total_time)

    return pi



