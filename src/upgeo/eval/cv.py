'''
Created on Jul 13, 2011

@author: marcel
'''

import numpy as np
from upgeo.util.array import count_unique

class RandKFold(object):
    '''
    classdocs
    '''

    __slots__ = ('_n',
                 '_k',
                 '_seed')

    def __init__(self, n, k=10, seed=None):
        '''
        Constructor
        '''
        if k < 1: 
            raise ValueError('k must be greater 0.')
        if k > n: 
            raise ValueError('k cannot be below the number of samples n') 
        
        self._n = n
        self._k = k
        self._seed = seed
        
    def __iter__(self):
        n = self._n
        k = self._k 
        
        rand = np.random.mtrand.RandomState(self._seed)    
        perm = rand.permutation(n)
        
        fold_size = np.ones(k) * np.ceil(n/k) #todo: check for casting to float
        fold_size[0:n%k] += 1
        
        cursor = 0
        for i in xrange(k):
            test_idx = np.zeros(n, dtype=bool)
            test_idx[perm[cursor:cursor+fold_size[i]]] = True
            train_idx = np.logical_not(test_idx)
            
            cursor += fold_size[i]
            
            yield train_idx, test_idx
            
class PSampleKFold(object):
    '''
    classdocs
    '''

    __slots__ = ('_n',
                 '_k',
                 '_p',
                 '_seed')

    def __init__(self, n, p, k=10, seed=None):
        '''
        Constructor
        '''
        if k < 1: 
            raise ValueError('k must be greater 0.')
        if k > n: 
            raise ValueError('k cannot be below the number of samples n')
        if p >= n:
            raise ValueError('p must be smaller than n')
        
        self._n = n
        self._k = k
        self._p = p
        self._seed = seed
        
    def __iter__(self):
        n = self._n
        k = self._k 
        p = self._p
        
        rand = np.random.mtrand.RandomState(self._seed)    
        perm = rand.permutation(n)
        
        #step_size = n/p #todo: check for casting to float
        #print step_size
        
        
        step_size = np.ones(k) * np.ceil(n/k) #todo: check for casting to float
        step_size[0:n%k] += 1
        cursor = 0
        for i in xrange(k):
            #print p
            train_idx = np.zeros(n, dtype=bool)
            if cursor+p < n+1:
                train_idx[perm[cursor:cursor+p]] = True
            else:
                train_idx[perm[cursor:]] = True
                
                train_idx[perm[:(cursor+p)-n]] = True
                print (cursor+p)-n
            test_idx = np.zeros(n, dtype=bool)
            test_idx = np.logical_not(train_idx)
            
            cursor += step_size[i]
            print 'cursor={0}'.format(cursor)
            
            yield train_idx, test_idx


class BagFold(object):
    
    __slots__ = ('_x')
    
    def __init__(self, x):
        '''
        @todo: check dimension
        '''
        self._x = np.asarray(x)
    
    def __iter__(self):
        '''
        '''
        x = self._x
        n = len(x)
        x_unique = np.unique(x) 
        
        for bag in x_unique:
            test_idx = np.zeros(n, dtype=bool)
            test_idx[x == bag] = True
            train_idx = np.logical_not(test_idx)
            
            yield train_idx, test_idx
            
class BagKFold(object):
    __slots__ = ('_bag',
                 '_k',
                 '_seed',
                 '_n',
                 '_uvalues')

    def __init__(self, bag, k=10, seed=None):
        bag = np.asarray(bag) 
        if bag.ndim != 1:
            raise TypeError('bag must be a 1-dim array.')
        if k < 1: 
            raise ValueError('k must be greater 0.')
        
        uvalues = np.unique(bag)
        n = len(uvalues)
        if k > n:
            raise ValueError('k is larger than unique values of the bag.')
        
        self._bag = bag
        self._k = k
        self._seed = seed
        
        self._n = n
        self._uvalues = uvalues
        
    def __iter__(self):
        n = self._n
        k = self._k 
        bags = self._bags
        uvalues = self._uvalues
        
        rand = np.random.mtrand.RandomState(self._seed)    
        perm = rand.permutation(n)
        
        fold_size = np.ones(k) * np.ceil(n/k) #todo: check for casting to float
        fold_size[0:n%k] += 1
        
        cursor = 0
        for i in xrange(k):
            test_bags = np.zeros(n, dtype=bool)
            test_bags[perm[cursor:cursor+fold_size[i]]] = True
            
            test_idx = np.in1d(bags, uvalues[test_bags])
            train_idx = np.logical_not(test_idx)
            
            cursor += fold_size[i]
            
            yield train_idx, test_idx
            
class NBagKFold(object):
    '''
    '''
    def __init__(self, bag, n, k=10, seed=None):
        '''
        '''
        
        
if __name__ == '__main__':
    loo = PSampleKFold(88, 9, 10)
    samples = np.array([])
    for train, test in loo:
        print "TRAIN:", np.nonzero(train), "TEST:", np.nonzero(test)
        samples = np.r_[samples, np.nonzero(train)[0]]
        
    print count_unique(samples)