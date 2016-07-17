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
import collections

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
    def _compute_logp(multi,test):
        """
        Returns the logP of the test data.

         INPUT:
        -------
            1. user_mult:   <(L, )> ndarray>    probabilities. Sums to 1.
            2. test:        <(N_te, ) ndarray>  test points.

         OUTPUT:
        --------
            1. logP:    <(N_te, ) ndarray>     logP of the test points
        """
        return [obj_func['ind_logp'](multi, test), obj_func['p_logp'](multi, test)]



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
            #log.info('Training %s' % mf)
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
        for name, scores in sorted(results.items()):
            row_names.append(name)
            x.append(scores)

        x = np.array(x)
        print(DataFrame(x, columns=col_names, index=row_names))

class _NonSmoothing(_Evaluation):
    
    def __init__(self):
        log.info('Evaluating ranking on a train+val without smoothing')

    def evaluate(self, train, val, test, dim, area):
        eval_train = train + val

        gt_scores = self._train_mfs(['memory'],test, dim, area)[0]
        mem_scores = self._train_mfs(['memory'],eval_train, dim, area)[0]
        popularity_scores = self._train_mfs(['popularity'],eval_train,dim,area)[0]

        log.info('Evaluating popularity')
        popularity_erank = self._compute_logp(test, popularity_scores)

        log.info('Evaluating memory')
        mem_erank = self._compute_logp(test, mem_scores)

        log.info('Evaluating ground truth')
        gt_erank = self._compute_logp(test, gt_scores)


        results = {'MEMORY': mem_erank, 'GROUNDTRUTH': gt_erank, 'POPULARITY': popularity_erank}
        self.pretty_print(results)

        return results           

class _SmoothedMem(_Evaluation):
    
    def __init__(self):
        log.info('Evaluating ranking on a train+val with smoothing')

    def evaluate(self, train, val, test, dim, area):
        eval_train = train + val

        s_mem_scores = self._train_mfs(['s_memory'],eval_train, dim, area)[0]
        log.info('Evaluating smoothed memory')
        s_mem_erank = self._compute_logp(test, s_mem_scores)

        results = {'S_MEMORY': s_mem_erank}
        self.pretty_print(results)

        return results 

class _GridSearch(_Evaluation):
    
    def __init__(self):
        log.info('Evaluating ranking with global gridsearched mixing weights')

    def evaluate(self, train, val, test, dim, area):

        ALPHA = [0,0.001,0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.99,0.999,1]
        mem_scores = self._train_mfs(['memory'],train, dim, area)[0]
        popularity_scores = self._train_mfs(['popularity'],train,dim,area)[0]

        mem_mult = normalize_mat_row(mem_scores)
        popularity_mult = normalize_mat_row(popularity_scores)

        log.info('Mem and popularity learnt from training data; searching alpha')
        results_val = dict()
        results_test = dict()
        for alpha in ALPHA:
            log.info('Ranking when alpha is %.2f' % alpha)
            scores = alpha * mem_mult + (1-alpha)*popularity_mult
            erank_val = self._compute_logp(val, scores)
            erank_test = self._compute_logp(test, scores)
            results_val['%.2f' % alpha] = erank_val
            results_test['%.2f' % alpha] = erank_test
        log.info('Erank on validation data')
        self.pretty_print(results_val)
        log.info('Erank on test data')
        self.pretty_print(results_test)

        eval_train = train + val
        mem_scores = self._train_mfs(['memory'],eval_train, dim, area)[0]
        popularity_scores = self._train_mfs(['popularity'],eval_train,dim,area)[0]

        mem_mult = normalize_mat_row(mem_scores)
        popularity_mult = normalize_mat_row(popularity_scores)

        log.info('Mem and popularity learnt from training and val data; searching alpha')
        results_val = dict()
        results_test = dict()
        for alpha in ALPHA:
            log.info('Ranking when alpha is %.2f' % alpha)
            scores = alpha * mem_mult + (1-alpha)*popularity_mult
            erank_val = self._compute_logp(val, scores)
            erank_test = self._compute_logp(test, scores)
            results_val['%.2f' % alpha] = erank_val
            results_test['%.2f' % alpha] = erank_test
        log.info('Erank on validation data')
        self.pretty_print(results_val)
        log.info('Erank on test data')
        self.pretty_print(results_test)


class _EmGlobal(_Evaluation):
    """
    Mixing memory with either NMF or HB_NMF. The mixing weights are learned globally for all users.
    """
    def __init__(self):
        log.info('Evaluating ranking with global learned mixing weights')

    def evaluate(self, train, val, test, dim, area):
        mem_scores = self._train_mfs(['memory'],train, dim, area)[0]
        popularity_scores = self._train_mfs(['popularity'],train,dim,area)[0]

        mem_mult = normalize_mat_row(mem_scores)
        popularity_mult = normalize_mat_row(popularity_scores+0.001)

        pi_mem_pop = learn_mix_mult_global(1.1, mem_mult, popularity_mult, val)
        log.info('Global mixing weight is %f and %f' % (pi_mem_pop[0],pi_mem_pop[1]))
        print sum((pi_mem_pop).astype(float))

        # The flat prior won't change the ranking so there's no need to add it here.
        log.info('Evaluating memory with popularity')
        mem_pop_erank = self._compute_logp(test, mem_mult, popularity_mult, pi_mem_pop)

        results = {'MEMORY+POPULARITY': mem_pop_erank}
        self.pretty_print(results)

        return results


class _EmIndiv(_Evaluation):
    """
    Mixing memory with either NMF or HB_NMF. The mixing weights are learned for each individual separately.
    """
    def __init__(self):
        log.info('Evaluating ranking with individual learned mixing weights')

    def evaluate(self, train, val, test, dim, area):
        mem_scores = self._train_mfs(['memory'],train, dim, area)[0]
        popularity_scores = self._train_mfs(['popularity'],train,dim,area)[0]

        mem_mult = normalize_mat_row(mem_scores)
        popularity_mult = normalize_mat_row(popularity_scores+0.001)

        pi_mem_pop = learn_mix_mult_on_individual(1.1, mem_mult, popularity_mult, val)

        # The flat prior won't change the ranking so there's no need to add it here.
        log.info('Evaluating memory with popularity')
        mem_pop_erank = self._compute_logp(test, mem_mult, popularity_mult, pi_mem_pop)

        results = {'MEMORY+POPULARITY': mem_pop_erank}
        self.pretty_print(results)

        return results


"""
******************************************************************************
                                METHOD FACTORY
******************************************************************************
"""
method_factory = {'s_mem': _SmoothedMem,'non_smoothing': _NonSmoothing, 'gridsearch': _GridSearch, 'mix_indiv': _EmIndiv, 'mix_global': _EmGlobal,}
