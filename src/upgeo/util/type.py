'''
Created on Jul 31, 2012

@author: marcel
'''

def enum(*names):
    ##assert names, "Empty enums are not supported" # <- Don't like empty enums? Uncomment!

    class EnumClass(object):
        __slots__ = names
        def __iter__(self): return iter(constants)
        def __len__(self):  return len(constants)
        def __getitem__(self, i):   return constants[i]
        def __repr__(self): return 'Enum' + str(names)
        def __str__(self):  return 'enum ' + str(constants)

    class EnumValue(object):
        __slots__ = ('__value')
        def __init__(self, value): self.__value = value
        
        value = property(lambda self: self.__value)
        type = property(lambda self: EnumType)
        
        def __hash__(self): return hash(self.__value)
        
        def __cmp__(self, other):
            # C fans might want to remove the following assertion
            # to make all enums comparable by ordinal value {;))
            #assert self.type is other.type, "Only values from the same enum are comparable"
            return cmp(self.__value, other.__value)
        
        def __invert__(self):   return constants[maximum - self.__value]
        def __nonzero__(self):  return bool(self.__value)
        def __repr__(self):     return str(names[self.__value])
    
    maximum = len(names) - 1
    constants = [None] * len(names)
    for i, each in enumerate(names):
        val = EnumValue(i)
        setattr(EnumClass, each, val)
        constants[i] = val
    constants = tuple(constants)
    EnumType = EnumClass()
    
    return EnumType

