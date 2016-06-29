"""
Different approaches for matrix factorization.
The methods should all be accessed through the factory method 'factorize_matrix'.

For a list of available methods run print_methods() or just look at the _mfs_factory dictionary.

Author: Moshe Lichman
"""
from __future__ import division
import numpy as np
import abc
import time

from utils import log_utils as log
from utils import file_utils as fu

from os.path import join
from scipy.sparse.linalg import svds
from sklearn.decomposition import NMF as sk_NMF


class _MFS(object):
    """
    Template for a matrix factorization class.
    """
    @abc.abstractmethod
    def get_factorized_mat(self, data, dim, area):
        raise NotImplementedError


class _NMF(_MFS):
    """
    The latent space is found using Poission matrix factorization.

    Currently we're using the most out-of-the-box run of NMF. This could be investigated further.
    """
    def get_factorized_mat(self, data, dim, area):
        log.info('Running sklearn NMF')
        start = time.time()

        model = sk_NMF(n_components=dim, init='random', random_state=0)
        W = model.fit_transform(data.toarray())     # It can't run on the sparse representation :(
        H = model.components_
        mf = np.dot(W, H)

        log.info('Factorizing took %d seconds' % (time.time() - start))
        return mf


class _HBNMF(_MFS):
    """
    Hierarchical Bayes Poisson MF. We use the code that was published with the paper.
    The factorization is done in the c++ code and here we simply load the two matrices and combine
    them into the mf.

    NOTE: IOError will be raised if the files are not in their root directory.
    """

    def get_factorized_mat(self, data, dim, area):
        log.info('Loading hierarchical bayes NMF')

        start = time.time()
        I, L = data.shape

        assert area is not None

        root_dir = '/home/disij/projects/hbpf/data/%s/hier_nmf' % area

        htheta = fu.load_np_txt(join(root_dir, 'htheta.tsv'), delimiter='\t')
        htheta = self._fix_projection(htheta, I, dim)

        hbeta = fu.load_np_txt(join(root_dir, 'hbeta.tsv'), delimiter='\t')
        hbeta = self._fix_projection(hbeta, L, dim)

        mf = htheta.dot(hbeta.T)

        log.info('Factorizing took %d seconds' % (time.time() - start))
        return mf


    def _fix_projection(self, mat, items, dim):
        """
        Both matrices needs to be fixed first. In the c++ code they ignore locations with
        no data and they change the entire projection.

         INPUT:
        -------
            1. mat:     <( <= items, dim + 2) ndarray>   htheta or hbeta. There could be less rows than items because
                                                         in the cpp code if a location didn't have data in training they
                                                         remove it. I don't.
                                                         Each row is [their_id, my_id, [factor_values]]
            2. items:    <int>                           number of individual or location.
            3. dims:     <int>                           number of hidden latent space.

         OUTPUT:
        --------
            1. fixed:   <(items, dim) ndarray>           fixed projection matrix
        """
        fixed = np.zeros([items, dim])
        my_ids = mat[:, 1] - 1          # Because Disi added 1 to my 0'based projection
    	values = mat[:, 2:]

        for i in range(mat.shape[0]):
            fixed[my_ids[i]] = values[i]

        return fixed


class _SVD(_MFS):
    """
    The latent space is found using "greedy" svd, finding the largest dim components.
    """
    def get_factorized_mat(self, data, dim, area):
        log.info('Running numpy svds')
        start = time.time()

        # For SVD we need to remove the mean from the data.
        tmp = np.copy(np.array(data.toarray()))
        m = np.mean(tmp, axis=0)
        tmp -= m

        u, s, v = svds(tmp, dim)
        W = u
        H = np.dot(np.diag(s), v)
        mf = np.dot(W, H)
        mf += m
        log.info('Factorizing took %d seconds' % (time.time() - start))

        return mf


class _Memory(_MFS):
    """
    No latent space, this is memory bases. Smoothed by column.
    """
    def get_factorized_mat(self, data, dim, area):
        temp = np.array(data.toarray())
        alpha = 0.01 # alpha is smoothing parameter
        temp = temp * (1-alpha) + np.mean(temp,axis = 0)*alpha       
        return temp


"""
*******************************************************************************************
                                    FACTORY METHODS
*******************************************************************************************
"""
_mfs_factory = {'hbnmf': _HBNMF, 'svd': _SVD, 'nmf': _NMF, 'memory': _Memory}


def print_methods():
    """
    Prints the available methods.
    """
    log.info('Available MF methods: %s' % list(_mfs_factory.keys()))


def factorize_matrix(method, data, dim=None, area=None):
    """
    Find latent space representation for the counts data and return the factorized matrix.
    The factorized matrix is a product of W * H' where W and H  ar the latent space representation of the
    individuals and location respectively.

     INPUT:
    -------
        1. method:      <string>            matrix factorization method.
        2. data:        <(I, L) csr_mat>    counts matrix.
        3. dim:         <int>               number of hidden latent dimensions.

     OUTPUT:
    --------
        1. mf:      <(I, L) ndarray>    matrix factorization for the data.

     RAISE:
    -------
        1. NotImplementedError:     Method was not implemented (not in _
    """
    method = method.lower()
    if method not in _mfs_factory:
        raise NotImplementedError('Method %s not implemented. Use %s instead.' % (method, list(_mfs_factory.keys())))

    mf = _mfs_factory[method]()
    return mf.get_factorized_mat(data, dim, area)

