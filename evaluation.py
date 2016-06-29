"""
Running all versions of MF as a stand along component.
The evaluation is done using erank per user and for points.

All methods should be accessed through the factory at the bottom of this file.

Author: Moshe Lichman
"""
from __future__ import division
import numpy as np
import mfs
import abc
from mix_learning import learn_mix_mult_global, learn_mix_mult_on_individual

from utils import log_utils as log
from utils.objecitves import obj_func
from utils.helpers import normalize_mat_row, col_vector

from pandas import DataFrame


class _Evaluation(object):
    """
    Abstract class for performing ranking evaluation.
    """
    @abc.abstractmethod
    def evaluate(self, train, val, test, dim, area):
        """
        Performing ranking evaluation on different models.

         INPUT:
        -------
            1. train:   <(I, L) csr_mat>    sparse counts matrix. Rows are individuals, columns are locations.
            2. val:     <(I, L) csr_mat>    sparse counts matrix. Rows are individuals, columns are locations.
            3. test:    <(I, L) csr_mat>    sparse counts matrix. Rows are individuals, columns are locations.
            4. dim:     <int>               number of dimension for matrix factorization
            5. area:    <string>            area for testing.

         OUTPUT:
        --------
            1. results:     dict>  each key, val pair is method name and [indiv ranking, points ranking] array.
                                   For example: {'svd': [0.56, 0.75], 'nmf': [0.87, 0.54]}
        """
        raise NotImplementedError


    @staticmethod
    def _compute_erank(test, comp_a, comp_b=None, pi=None):
        """
        Computes the ranking for the test data. This could work with one component (single component evaluation) or
        with two and mixing weights. The mixing weights can be global or for each individual.

         INPUT:
        -------
            1. test:          <(I, L) csr_mat>    sparse counts matrix. Rows are individuals, columns are locations.
            2. comp_a:        <(I, L) ndarray>    each row is a score for the i'th individual.
            3. comp_b:        <(I, L) ndarray>    each row is a score for the i'th individual.
            3. pi:            <(2, ) or (2, I)>   mixing weights, global or for each user.

         OUTPUT:
        --------
            1. ranking:     <(2, ) tuple>   avg. per individual and avg. across all points
        """
        if pi is None:
            scores = comp_a
        elif len(pi.shape) == 1:
            scores = pi[0] * comp_a + pi[1] * comp_b
        else:
            scores = col_vector(pi[:, 0]) * comp_a + col_vector(pi[:, 1]) * comp_b

        return [obj_func['ind_erank'](scores, test), obj_func['p_erank'](scores, test)]


    @staticmethod
    def _train_mfs(mf_types, train, dim, area):
        """
        Wrapper function for training multiple MFs.

         INPUT:
        -------
            1. mf_types:    <list>             strings for the trained methods.
            2. train:       <(I, L) csr_mat>   sparse counts matrix. Rows are individuals, columns are locations.
            3. dim:         <int>              number of dimension for the MFs
            4. area:        <string>           tested area -- the hb_nmf needs it.

         OUTPUT:
        --------
            1. trained_mfs:     <list>      trained MFs in the order given in mf_types
        """
        trained_mfs = []
        for mf in mf_types:
            log.info('Training %s' % mf)
            trained_mfs.append(mfs.factorize_matrix(mf, train, dim, area))

        return trained_mfs

    @staticmethod
    def pretty_print(results):
        """
        Printing the results using pandas DataFrame

         INPUT:
        -------
            1. results:     <dict>  each key, val pair is method name and [indiv ranking, points ranking] array.
                                    For example: {'svd': [0.56, 0.75], 'nmf': [0.87, 0.54]}
        """
        x = []
        col_names = ['avg. indiv', 'avg. points']
        row_names = []
        for name, scores in results.items():
            row_names.append(name)
            x.append(scores)

        x = np.array(x)
        print(DataFrame(x, columns=col_names, index=row_names))


class _EvaluateSingle(_Evaluation):
    """
    Using a single scoring method instead of creating a mixture model.
    """
    def __init__(self):
        log.info('Evaluating ranking on a single component')

    def evaluate(self, train, val, test, dim, area):
        # There is no mixing weights optimization in this code.
        # Therefore the val can be added to train.
        eval_train = train + val

        svd_scores, nmf_scores, hb_nmf_scores, mem_scores = self._train_mfs(['svd', 'nmf', 'hbnmf', 'memory'],
                                                                            eval_train, dim, area)
        gt_scores = self._train_mfs(['memory'],test, dim, area)

        log.info('Evaluating SVD')
        svd_erank = self._compute_erank(test, svd_scores)

        log.info('Evaluating sklearn NMF')
        nmf_erank = self._compute_erank(test, nmf_scores)

        log.info('Evaluating Hierarchical Bayes NMF')
        hb_nmf_erank = self._compute_erank(test, hb_nmf_scores)

        log.info('Evaluating memory')
        mem_erank = self._compute_erank(test, mem_scores)

        log.info('Evaluating ground truth')
        gt_erank = self._compute_erank(test, gt_scores)

        results = {'MEMORY': mem_erank, 'SVD': svd_erank, 'NMF': nmf_erank, 'HBPF': hb_nmf_erank, 'GROUNDTRUTH': gt_erank}
        self.pretty_print(results)

        return results


class _EvaluateEmGlobal(_Evaluation):
    """
    Mixing memory with either NMF or HB_NMF. The mixing weights are learned globally for all users.
    """
    def __init__(self):
        log.info('Evaluating ranking with global learned mixing weights')

    def evaluate(self, train, val, test, dim, area):
        log.info('Learning Memory, NMF and hb NMF mfs on train only for mixing weights optimization')
        nmf_scores, hb_nmf_scores, mem_scores = self._train_mfs(['nmf', 'hbnmf', 'memory'], train, dim, area)

        log.info('Learning mix for MEM and NMF')
        mem_mult = normalize_mat_row(mem_scores)
        nmf_mult = normalize_mat_row(nmf_scores + 0.001)   # Small flat prior to avoid 0.

        pi_mem_nmf = learn_mix_mult_global(1.1, mem_mult, nmf_mult, val)

        log.info('Learning mix for MEM and hb NMF')
        hb_nmf_mult = normalize_mat_row(hb_nmf_scores + 0.001)  # Small flat prior to avoid 0.
        pi_mem_hb_nmf = learn_mix_mult_global(1.1, mem_mult, hb_nmf_mult, val)

        log.info('Learning Memory NMF and hier NMF mfs on train+val for evaluation')
        eval_train = train + val
        nmf_scores, hb_nmf_scores, mem_scores = self._train_mfs(['nmf', 'hbnmf', 'memory'], eval_train, dim, area)

        # The flat prior won't change the ranking so there's no need to add it here.
        log.info('Evaluating memory with NMF')
        mem_nmf_erank = self._compute_erank(test, mem_scores, nmf_scores, pi_mem_nmf)

        log.info('Evaluating memory with hb_NMF')
        mem_hb_nmf_erank = self._compute_erank(test, mem_scores, hb_nmf_scores, pi_mem_hb_nmf)

        results = {'mem_nmf': mem_nmf_erank, 'mem_hb_nmf': mem_hb_nmf_erank}
        self.pretty_print(results)

        return results


class _EvaluateEmIndiv(_Evaluation):
    """
    Mixing memory with either NMF or HB_NMF. The mixing weights are learned for each individual separately.
    """
    def __init__(self):
        log.info('Evaluating ranking with individual learned mixing weights')

    def evaluate(self, train, val, test, dim, area):
        log.info('Learning Memory, NMF and hb NMF mfs on train only for mixing weights optimization')
        nmf_scores, hb_nmf_scores, mem_scores = self._train_mfs(['nmf', 'hbnmf', 'memory'], train, dim, area)

        log.info('Learning mix for MEM and NMF')
        mem_mult = normalize_mat_row(mem_scores)
        nmf_mult = normalize_mat_row(nmf_scores + 0.001)   # Small flat prior to avoid 0.
        pis_mem_nmf = learn_mix_mult_on_individual(1.1, mem_mult, nmf_mult, val)

        log.info('Learning mix for MEM and hb NMF')
        hb_nmf_mult = normalize_mat_row(hb_nmf_scores + 0.001)  # Small flat prior to avoid 0.
        pis_mem_hb_nmf = learn_mix_mult_on_individual(1.1, mem_mult, hb_nmf_mult, val)

        log.info('Learning Memory NMF and hier NMF mfs on train+val for evaluation')
        eval_train = train + val
        nmf_scores, hb_nmf_scores, mem_scores = self._train_mfs(['nmf', 'hbnmf', 'memory'], eval_train, dim, area)

        # The flat prior won't change the ranking so there's no need to add it here.
        log.info('Evaluating memory with NMF')
        mem_nmf_erank = self._compute_erank(test, mem_scores, nmf_scores, pis_mem_nmf)

        log.info('Evaluating memory with hb_NMF')
        mem_hb_nmf_erank = self._compute_erank(test, mem_scores, hb_nmf_scores, pis_mem_hb_nmf)

        results = {'mem_nmf': mem_nmf_erank, 'mem_hb_nmf': mem_hb_nmf_erank}
        self.pretty_print(results)

        return results

"""
******************************************************************************
                                METHOD FACTORY
******************************************************************************
"""
method_factory = {'single': _EvaluateSingle, 'mix_global': _EvaluateEmGlobal, 'mix_indiv': _EvaluateEmIndiv}
