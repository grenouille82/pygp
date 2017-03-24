'''
Created on Jul 2, 2012

@author: marcel
'''
import numpy as np
import upgeo.util.metric as metric
import upgeo.util.stats as stats

from abc import ABCMeta, abstractmethod
from upgeo.util.array import count_unique
from upgeo.util.exception import StateError
from upgeo.eval.cv import NBagKFold, RandKFold

class Experiment(object):
    
    __metaclass__ = ABCMeta
    
    @abstractmethod
    def eval(self, algo):
        return None
    
class RegressionExperiment(Experiment):
    
    __slots__ = ()
    
    def __init__(self):
        '''
        '''
    
    def eval(self, algo):
        '''
        '''

class CVRegressionExperiment(Experiment):

    __slots__ = ('_X',
                 '_y',
                 '_nfolds',
                 '_filter',
                 '_filter_mask',
                 '_seed' 
                 )
    
    def __init__(self, X, y, nfolds=10, filter=None, filter_mask=None, seed=None):
        self._X = X
        self._y = y
        self._nfolds = nfolds
        self._filter = filter
        self._filter_mask = filter_mask
        self._seed = seed        
    
    def eval(self, algo):
        '''
        @todo: - check that the algo isnt a type of multitask learner
        '''
        hyperparams = algo.hyperparams
        
        X = self._X
        y = self._y
        
        n = X.shape[0]
        k = self._nfolds
        
        scores = np.empty(k)
        weights = np.empty(k)
        
        loo = RandKFold(n, k, self._seed)
        i = 0
        for train, test in loo:
            Xtrain, ytrain, Xtest, ytest = self.__prepare_eval_data(X, y, train, test)
            
            algo.hyperparams = hyperparams
            algo.fit(Xtrain, ytrain)
            print 'hyperparams = {0}'.format(np.exp(algo.hyperparams))
            yhat = algo.predict(Xtest)
            
            mse = metric.mspe(ytest, yhat)
            scores[i] = mse
            weights[i] = len(ytrain)
            i = i+1
            print 'fold[{0}]: {1}'.format(i, mse)
            
        algo.hyperparams = hyperparams
        
        print scores
        mse = stats.mean(scores, weights)
        err = stats.stddev(scores, weights)
        
        return mse, err
    
    def __prepare_eval_data(self, X, y, train, test):
        Xtr, ytr = X[train], y[train]
        Xte, yte = X[test], y[test]
        
        if self._filter is not None:
            Ztr = np.c_[Xtr, ytr]
            Zte = np.c_[Xte, yte]
            
            k = Ztr.shape[1]
            mask = self._filter_mask
            if mask is None:
                mask = np.arange(k)
            
            Ztr[:,mask] = self._filter.apply(Ztr[:,mask])
            Zte[:,mask] = self._filter.apply(Zte[:,mask], True)
            
            Xtr = Ztr[:,0:(k-1)]
            ytr = Ztr[:,k-1]
            Xte = Zte[:,0:(k-1)]
            yte = Zte[:,k-1]
                
        return Xtr, ytr, Xte, yte


class TLRegressionExperiment(Experiment):
    
    __slots__ = ()
    
    def __init__(self):
        '''
        '''
    
    def eval(self, algo):
        '''
        '''

class BagTLRegressionExperiment(Experiment):
    
    __slots__ = ('_Xp', 
                 '_yp',
                 '_Xs',
                 '_ys',
                 '_itask',
                 '_bag',
                 '_reps',
                 '_seed',
                 '_bsize')
    
    def __init__(self, Xp, yp, Xs, ys, itask, bag, bsize, fold_size, reps=5, seed=None):
        '''
        Parameters:
            Xp:       Covariates of size n_p x d for the primary task.
            yp:       Target values of size n_p x 1 for the primary task.
            Xs:       Covariates of size n_s x d for the secondary tasks.
            ys:       Target values of size n_s x 1 for the secondary tasks.
            itask:    Task indices for the secondary task.
            bag:      Array of bags for the primary task.
            bsize:    Determines the size of each bag. Bags with a size below this value 
                      are automatically choosen as test data, otherwise bags are shrinked
                      to size of bsize, if they are used as training data.
            fold_size: 
            rep:      Number of repetition for the experiment.
            seed:
            
        '''
    
    def eval(self, algo):
        '''
        '''
        Xp = self._Xp
        yp = self._yp
        Xs = self._Xs
        ys = self._ys
        itask = self._itask
        
        reps = self._reps
        seed = self._seed
        
        #construct a fixed set of test data, used for all repititions
        fold_size = self._fold_size
        bsize = self._bsize
        bag = self._bag
        ubag, counts = count_unique(bag)
        test_idx = np.in1d(bag, ubag[counts < bsize])
        rbag = ubag[counts >= bsize]
        if len(rbag) < fold_size:
            raise StateError('size of remaining bags are smaller than fold size')
        elif len(rbag) > fold_size:
            #chose randomly a number of the remaining bags below the fold_size
            rand = np.random.mtrand.RandomState(seed)    
            perm = rand.permutation(len(rbag))
            test_idx = np.logical_or(test_idx, np.in1d(bag, rbag[perm[:len(rbag)-fold_size]]))
        train_idx = np.logical_or(test_idx)
        Xtest = Xp[test_idx]
        ytest = yp[test_idx]
        Xp = Xp[train_idx]
        yp = yp[train_idx]
        bag = bag[train_idx]
        
        #shrink the size of each training bag to the given bsize
        train_idx = np.zeros(len(Xp), dtype=np.bool)
        ubag, counts = count_unique(bag)
        for i in xrange(len(ubag)):
            perm = rand.permutation(counts[i])
            train_idx[np.nonzero(bag==ubag[i])[0][perm[:bsize]]] = True
        Xp = Xp[train_idx]
        yp = yp[train_idx]
        bag = bag[train_idx]
        
        reseed = rand.int(0)
        mse = np.zeros(reps)
        loo = NBagKFold(bag, fold_size, reps, reseed)
        i = 0
        for train, _ in loo:
            algo.fit(Xp[train], yp[train], Xs, ys, itask)
            yfit = algo.predict(Xtest)            
            
            mse[i] = metric.mspe(ytest, yfit)
            i += 1
        
        mean_mse = np.mean(mse)
        std_mse = np.std(mse)
        return mean_mse, std_mse
