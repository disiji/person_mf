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
from scipy.sparse import coo_matrix
from sklearn import metrics

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
    def _compute_erank_logp(test, comp_a, comp_b=None, pi=None):
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

        return [obj_func['ind_erank'](scores, test), obj_func['p_erank'](scores, test),obj_func['ind_logp'](scores, test), obj_func['p_logp'](scores, test)]

    @staticmethod
    def _compute_logp_point(test, comp_a, comp_b=None, pi=None):
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

        return obj_func['p_logp'](scores, test)


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
        col_names = ['avg. indiv Erank', 'avg. points Erank','avg. indiv logP','avg. points logP']
        row_names = []
        for name, scores in sorted(results.items()):
            row_names.append(name)
            x.append(scores)

        x = np.array(x)
        print(DataFrame(x, columns=col_names, index=row_names))

class _Smoothing(_Evaluation):
    
    def __init__(self):
        log.info('Mem and popularity learnt from training data; searching alpha on validation set')


    def evaluate(self, train, val, test, dim, area):

        def logP(score_mat,test):
            logp_p = np.zeros(test.sum())
            logp_indiv = np.zeros(test.shape[0])
            test_data = coo_matrix(test)

            temp = score_mat / np.sum(score_mat)
            idx = 0
            for i,j,v in zip(test_data.row, test_data.col, test_data.data):
                logp_p[int(idx):int(idx+v)] = np.log(temp[i,j])
                idx += v

            temp = normalize_mat_row(score_mat)
            for i,j,v in zip(test_data.row, test_data.col, test_data.data):
                logp_indiv[i] += v * np.log(temp[i,j])

            n_train = np.array([int(test.sum(axis=1)[i][0]) for i in range(I)])
            logp_indiv /= n_train

            return logp_p,logp_indiv

        ALPHA =  np.arange(0.5,1.05,0.1)

        mem_scores = self._train_mfs(['memory'],train, dim, area)[0]
        popularity_scores = self._train_mfs(['popularity'],train,dim,area)[0]+0.0001

        mem_mult = normalize_mat_row(mem_scores)
        popularity_mult = normalize_mat_row(popularity_scores)

        N = int(np.sum(mem_scores))
        I,L = train.shape
        
        results = dict()
        headers = ['EM global','EM indiv','S_mem','Dirichlet','Translation']
        logP_p = DataFrame(np.zeros((int(test.sum()),5)), columns=headers)
        logP_indiv = DataFrame(np.zeros((I,5)), columns=headers)
        mix_alpha = DataFrame(np.zeros((I,5)), columns=headers)


        log.info('#####learning statistical translation model#######')
        log.info('computing sparse mutual information')
        # # binary = (mem_scores>0)*1#I*L
        # # count_1d = np.sum(binary,axis = 0)#1*L
        # # count_2d = np.dot(binary.T,binary)#L*L
        # # P_1d_1 = count_1d/I
        # # P_1d_0 = 1 - P_1d_1
        # # P_2d_1_1 = count_2d/I
        # # P_2d_1_0 = (np.asmatrix(count_1d).T - count_2d)/I
        # # P_2d_0_1 = P_2d_1_0.T
        # # P_2d_0_0 = 1 - P_2d_1_1 - P_2d_1_0 - P_2d_0_1
        # # MI = np.zeros((L,L))
        # # temp = P_2d_0_0/(np.outer(P_1d_0,P_1d_0))
        # # temp[temp==0]=1
        # # MI += np.multiply(P_2d_0_0,np.log(temp))
        # # temp = P_2d_0_1/(np.outer(P_1d_0,P_1d_1))
        # # temp[temp==0]=1
        # # MI += np.multiply(P_2d_0_1,np.log(temp))
        # # temp = P_2d_1_0/(np.outer(P_1d_1,P_1d_0))
        # # temp[temp==0]=1
        # # MI += np.multiply(P_2d_1_0,np.log(temp))
        # # temp = P_2d_1_1/(np.outer(P_1d_1,P_1d_1))
        # # temp[temp==0]=1
        # # MI += np.multiply(P_2d_1_1,np.log(temp))
        # # MI = np.array([[
        # #     (0 if P_2d_1_1[u,w]==0 else P_2d_1_1[u,w]*np.log(P_2d_1_1[u,w]/P_1d_1[u]/P_1d_1[w]))
        # #     + (0 if P_2d_1_0[u,w]==0 else P_2d_1_0[u,w]*np.log(P_2d_1_0[u,w]/P_1d_1[u]/P_1d_0[w]))
        # #     + (0 if P_2d_0_1[u,w]==0 else P_2d_0_1[u,w]*np.log(P_2d_0_1[u,w]/P_1d_0[u]/P_1d_1[w]))
        # #     + (0 if P_2d_0_0[u,w]==0 else P_2d_0_0[u,w]*np.log(P_2d_0_0[u,w]/P_1d_0[u]/P_1d_0[w]))
        # #     for w in range(L)] for u in range(L)])
        
        binary = (train>0)*1#I*L
        count_1d = binary.sum(axis = 0)#1*L
        count_2d = np.dot(binary.T,binary)#L*L
        P_1d = count_1d/I # exists zeros
        P_2d = count_2d/I
        temp = P_2d/np.outer(P_1d,P_1d)
        temp[ ~ np.isfinite( temp )]= 1 # zero / zero = zero
        temp[temp==0]=1 # avoid log_zero
        PPMI = np.log2(temp)
        PPMI[PPMI<0] = 0

 
        k = 50
        idx = np.array([[j for j in np.asarray(PPMI[i].argsort().T).reshape(-1)[-k:][::-1] 
                         if PPMI[i,j]>0] for i in range(L)])
        for u in range(L):
            if u not in idx[u]:
                idx[u].append(u)

        binary = (np.array(train.toarray())>0)*1#I*L
        MI = np.zeros((L,L))
        from sklearn import metrics
        for u in range(L):
            for w in idx[u]:
                MI[u,w] = metrics.mutual_info_score(None, None, 
                        contingency=np.histogram2d(binary[:,u], binary[:,w])[0])

        MI = normalize_mat_row(MI)
        MI[ ~ np.isfinite( MI )]= 0
        ##########and self transition probability########
        log.info('gridsearching on validation set (can be optimized)')
        val_result = dict()
        for alpha in ALPHA:
            for mu in ALPHA:
                pref = np.zeros((I,L))
                trans = MI*(1-alpha)+np.identity(L)*alpha
                for i in range(I):
                    pref[i] = np.sum(trans * mem_mult[i][:, np.newaxis], axis=0)
                temp = pref * mu + popularity_mult*(1-mu)
                val_result[(alpha,mu)] =  self._compute_logp_point(val, temp)
        #####choose alpha and mu that achieves best avg. point logP 
        alpha,mu = max(val_result, key=val_result.get)
        temp = MI*(1-alpha)+np.identity(L)*alpha
        stm_scores = np.dot(mem_mult , temp.T) * mu + popularity_mult*(1-mu)
        log.info('Evaluating MI based translation model')
        stm_result = self._compute_erank_logp(test, stm_scores)
        results['Translation'] = stm_result
        print("alpha and mu:",alpha, mu)
        #####record results and mixture parameters########
        logP_p['Translation'],logP_indiv['Translation'] = logP(stm_scores,test)
        mix_alpha['Translation'] = np.zeros(I)+mu*alpha      


        log.info('#############learning EM global#################')
        pi_mem_pop = learn_mix_mult_global(1.1, mem_mult, popularity_mult, val)
        log.info('Global mixing weight is %f and %f' % (pi_mem_pop[0],pi_mem_pop[1]))
        log.info('Evaluating EM global')


        em_global_scores = pi_mem_pop[0] * mem_mult + pi_mem_pop[1] * popularity_mult
        EM_global_result = self._compute_erank_logp(test, em_global_scores)
        results['EM global'] = EM_global_result
        logP_p['EM global'],logP_indiv['EM global'] = logP(em_global_scores,test)
        mix_alpha['EM global'] = pi_mem_pop[0]+np.zeros(I)



        log.info('#############learning EM individual##############')
        pi_mem_pop = learn_mix_mult_on_individual(1.1, mem_mult, popularity_mult, val)
        log.info('Evaluating EM indiv')

        em_indiv_scores = col_vector(pi_mem_pop[:, 0]) * mem_mult + col_vector(pi_mem_pop[:, 1]) * popularity_mult
        EM_indiv_result = self._compute_erank_logp(test, mem_mult, popularity_mult, pi_mem_pop)
        results['EM indiv'] = EM_indiv_result
        logP_p['EM indiv'],logP_indiv['EM indiv'] = logP(em_indiv_scores,test)
        mix_alpha['EM indiv'] = pi_mem_pop[:,0]         



        log.info('#############learning S_memory###################')
        log.info('gridsearching on validation set')
        val_result = dict()
        for alpha in ALPHA:
            temp = mem_scores * alpha + popularity_scores*(1-alpha)
            val_result[alpha] =  self._compute_logp_point(val, temp)
        #####choose alpha that achieves best avg. point logP 
        alpha = max(val_result, key=val_result.get)
        print('alpha:',alpha)
        s_mem_scores = mem_scores * alpha + popularity_scores*(1-alpha)
        log.info('Evaluating smoothed memory')
        s_mem_result = self._compute_erank_logp(test, s_mem_scores)
        results['S_Mem'] = s_mem_result

        n_train = np.array([int(train.sum(axis=1)[i][0]) for i in range(I)])
        temp = n_train.mean()
        logP_p['S_mem'],logP_indiv['S_mem'] = logP(s_mem_scores,test)
        mix_alpha['S_mem'] = alpha*n_train/(alpha*n_train+(1-alpha)*temp)



        log.info('############learning with Dirichlet prior#############')
        log.info('gridsearching on validation set')
        val_result = dict()
        for alpha in ALPHA:
            temp = mem_scores + popularity_mult*alpha*N/I
            val_result[alpha] =  self._compute_logp_point(val, temp)
        #####choose alpha that achieves best avg. point logP 
        alpha = max(val_result, key=val_result.get)
        print('alpha:',alpha)
        dirichlet_scores = mem_scores + popularity_mult*alpha*N/I
        log.info('Evaluating with Dirichlet prior')
        dirichlet_result = self._compute_erank_logp(test, dirichlet_scores)
        results['Dirichlet'] = dirichlet_result

        n_train = np.array([int(train.sum(axis=1)[i][0]) for i in range(I)])
        logP_p['Dirichlet'],logP_indiv['Dirichlet'] = logP(dirichlet_scores,test)
        mix_alpha['Dirichlet'] = n_train/(n_train+alpha*N/I)

        self.pretty_print(results)
        return logP_p,logP_indiv,mix_alpha






"""
******************************************************************************
                                METHOD FACTORY
******************************************************************************
"""
method_factory = {'smoothing':_Smoothing}