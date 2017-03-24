'''
Created on Mar 8, 2011

@author: marcel
'''

class StateError(Exception):
    pass

class NotInitializedError(StateError):
    pass

class NotFittedError(StateError):
    pass