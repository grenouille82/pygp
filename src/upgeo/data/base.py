'''
Created on Jul 2, 2012

@author: marcel
'''

'''
Created on Mar 14, 2011

@author: marcel

@todo: - implement summary statistics for dataset and multidataset
'''
import numpy as np
import os


from abc import ABCMeta, abstractmethod

from upgeo.util.array import rec_names_at, rec_disjoin, rec2csv, csv2rec, rec_drop_fields, rec_append_fields, rec_join,\
    rec_index_of
from _abcoll import Container

__numerical_types = {}

__catergorical_types = {}

class Dataset(object):
    '''
    @todo:
    '''
    #__metaclass__ = ABCMeta
    
    __slots__ = ('_data', '_fields', '_size', '_ndim', '_targets')
    
    def __init__(self, data, targets=None):
        if not isinstance(data, np.recarray):
            raise TypeError('data must be a type of np.recarray')
        
        self._fields = data.dtype
        self._data = data
        self._size = data.size
        self._ndim = len(self._fields)
        self._targets = targets
    
    def _get_fields(self):
        '''
        Returns all fields as dtype.
        '''
        return self._fields
    
    fields = property(fget=_get_fields)
    
    def field_type(self, key):
        '''
        Distinguish between a categorical or numerical attribute for an
        attribute specified by key, whereas key could be the index or the
        name of the attribute.
        '''
        return 'numerical' if self._fields[key].name in __numerical_types else 'categorical'
    
    def _get_field_types(self):
        '''
        Returns all attribute types of the attributes as tuple. The attribute 
        type is categorical or numerical.
        '''
        return  tuple(self.field_type(i) for i in xrange(self._ndim))
    
    field_types = property(fget=_get_field_types)
    
    def isnumerical(self, key):
        '''
        Returns true, iff the key-specified attribute a numerical type, 
        otherwise false. whereas key could be the index or the name of 
        the attribute.
        '''
        return self._fields[key].name in __numerical_types
    
    def iscategorical(self, key):
        '''
        Returns true, iff the key-specified attribute a categorical type, 
        otherwise false. whereas key could be the index or the name of 
        the attribute.
        '''
        return self._fields[key].name in __catergorical_types
    
    def has_targets(self):
        '''
        Returns true, iff target attributes are specified, otherwise false.
        '''
        return self._targets != None
    
    def _get_targets(self):
        '''
        Returns the specified indices of the target attribute.
        '''
        return self._targets
    
    def _set_targets(self, fields):
        '''
        Sets the specified target attributes.
        @todo:
        -distinguish between numeric and string targets
        '''
        value = np.asarray(fields)
        if np.issubdtype(value.dtype, 'string'):
            value = rec_index_of(self._data, value) 
        #if np.any([value<0, value>self.ndim-1]):
        #    raise ValueError('Some targets are not contained in dataset') 
        self._targets = value
    
    def _del_targets(self):
        '''
        Delete the target attributes of the dataset.
        '''
        self._targets = None
    
    targets = property(fget=_get_targets, fset=_set_targets, fdel=_del_targets)
    
    def toarray(self, dtype=None):
        '''
        Returns an array representation of the underlying data. The type of the
        returned array can be set by the variable dtype. If the variable not 
        specified the type of array is automatically determined. (See for further
        information to the numpy specification)
        
        '''
        return np.array(self._data.tolist(), dtype)
    
    def target_data(self, dtype=None):
        '''
        Returns all target data as numpy array. If no target data was specified 
        an empty numpy array is returned. The type of the returned array can be 
        set by the variable dtype. If the variable not specified the type of 
        array is automatically determined. (See for further information to the 
        numpy specification)
        '''
        if self._targets == None:
            data = np.array([], dtype)
        else:
            data = self.toarray()
            data = np.squeeze(np.asarray(data[:, self._targets], dtype))
        return data
        
    def covariate_data(self, dtype=None):
        '''
        Returns all covariates data as numpy array. If no target data was specified 
        an empty numpy array is returned. Covariates are these attributes which are
        not specified as target attributes. The type of the returned array can be 
        set by the variable dtype. If the variable not specified the type of 
        array is automatically determined. (See for further information to the 
        numpy specification)
        '''
        if self._targets == None:
            data = self.toarray(dtype)
        else:
            i = np.ones(self._ndim, dtype=bool)
            i[self._targets] = 0
            data = self.toarray()
            data = np.squeeze(np.asarray(data[:, i], dtype))
        return data
    
    def data_by_fields(self, fields, dtype=None):
        '''
        '''
        fields = np.asarray(fields)
        if fields.dtype == int:
            fields = rec_names_at(self._data, fields)
            
        records = rec_drop_fields(self._data, fields)
        return np.array(records.tolist(), dtype)
    
    def __getitem__(self, k):
        if isinstance(k, slice):
            indices = k.indices(self._size)
            return [self._data[i] for i in range(*indices)]
        elif isinstance(k, int):
            return self._data[k]
        elif isinstance(k, str):
            return self._data[k]
        elif isinstance(k, np.ndarray) or isinstance(k, list):
            return self._data[k]
        else:
            raise TypeError('index must be an integer, not {0}'.format(type(k)))
        
    def __iter__(self):
        return iter(self._data)
    
    def __len__(self):
        return self._size
    
    def _get_data(self):
        return self._data
    
    data = property(fget=_get_data)

    def _get_size(self):
        return self._size
    
    size = property(fget=_get_size)
    
    def _get_ndim(self):
        return self._ndim
     
    ndim = property(fget=_get_ndim)

class MultitaskDataset(object):
    '''
    '''
    #__metaclass__ = ABCMeta
    
    __slots__ = ('_task_data', '_task_descriptions', '_task_identifiers', '_ntasks', 
                 '_data_fields', '_description_fields', '_targets')
    
    def __init__(self, data, descriptions=None, identifiers=None):
        ''' 
        '''
        if not isinstance(data, Container):
            raise TypeError("data must be a subtype of Container")
        if len(data) == 0:
            raise AttributeError("data must be non-empty.")
        
        self._task_data = tuple(data)
        self._data_fields = data[0].fields
        self._ntasks = len(self._task_data)
        
        self._task_descriptions = None
        if descriptions != None:
            if not isinstance(descriptions, Container):
                raise TypeError("descriptions must be a subtype of Container")
            if len(descriptions) != self._ntasks:
                raise AttributeError("number of task descriptions must "+ 
                                     "be equal to task number")
            self._task_descriptions = tuple(descriptions)
            self._description_fields = descriptions[0].dtype
        
        self._task_identifiers = None
        if identifiers != None:     
            if not isinstance(identifiers, Container):
                raise TypeError("descriptions must be a subtype of Container")
            if len(identifiers) != self._ntasks:
                raise AttributeError("number of task identifiers must "+ 
                                     "be equal to task number")
            self._task_identifiers = identifiers
    
    def task_dataset(self, i):
        '''
        Returns the corresponding dataset of the i-th task.
        '''
        #self._range_check(i)
        return self._task_data[i]
    
    def has_descriptions(self):
        return self._task_descriptions != None
        
    def task_description(self, i):
        '''
        Returns the description data of the i-th task.
        '''
        return self._task_descriptions[i] if self.has_descriptions() else None
    
    def has_identifiers(self):
        return self._task_identifiers != None
    
    def task_id(self, i):
        '''
        Returns the task identifier of the i-th task.
        '''
        return self._task_identifiers[i] if self.has_identifiers() else None
    
    def __getitem__(self, index):
        '''
        '''
        if isinstance(index, slice):
            data = self.task_dataset(index)
            descr = self.task_description(index) #BUG: exception occurs if description is not defined
            ret = zip(data, descr)
        elif isinstance(index, Container):
            ret = []
            for i in index:
                ret.append((self.task_dataset(i), self.task_description(i)))
        elif isinstance(index, int):
            ret = (self.task_dataset(index), self.task_description(index))
        else:
            raise TypeError('invalid index type: %s' % type(index))
        
        return ret
        
    def __iter__(self):
        '''
        '''
        for i in xrange(self._ntasks):
            yield self[i]
    
    def __len__(self):
        '''
        '''
        return self._ntask

    def _get_ntasks(self):
        '''
        '''
        return self._ntasks
    
    ntasks = property(fget=_get_ntasks)
    
    def _get_data_fields(self):
        return self._data_fields
    
    data_fields = property(fget=_get_data_fields)
    
    def _get_description_fields(self):
        return self._description_fields
    
    description_fields = property(fget=_get_description_fields)

    def data_field_type(self, i):
        return 'numerical' if self._data_fields[i] in __numerical_types else 'categorical'
    
    def _get_data_field_types(self):
        n = len(self._data_fields)
        return  tuple(self.data_field_type(i) for i in range(n))
    
    data_field_type = property(fget=_get_data_field_types)
    
    def description_field_type(self, i):
        return 'numerical' if self._description_fields[i] in __numerical_types else 'categorical'
    
    def _get_description_field_types(self):
        n = len(self._description_fields)
        return  tuple(self.description_field_type(i) for i in range(n))
    
    description_field_type = property(fget=_get_description_field_types)
    
    def _get_task_data(self):
        return self._task_data
    
    task_data = property(fget=_get_task_data)
    
    def _get_task_descriptions(self):
        return self._task_descriptions
    
    task_descriptions = property(fget=_get_task_descriptions)
    
    def _get_task_identifiers(self):
        return self._task_identifiers
    
    task_identifiers = property(fget=_get_task_identifiers)
    
    def has_targets(self):
        return self._targets != None
    
    def _get_targets(self):
        return self._targets
    
    def _set_targets(self, fields):
        '''
        @todo:
        -distinguish between numeric and string targets
        '''
        fields = np.asarray(fields)
        self._targets = fields
        for data in self._task_data:
            data.targets = fields
    
    def _del_targets(self):
        self._targets = None
        for data in self._task_data:
            del data.targets
    
    targets = property(fget=_get_targets, fset=_set_targets, fdel=_del_targets)

class DataSource(object):
    __metaclass__ = ABCMeta
    
    __slots__ = ('_filename', '_dataset', '_excluded_fields')

    @staticmethod
    def create_datasource(filename, dataset=None):
        '''
        @todo: 
        - validate filename
        - add arg list for specific datasources
        '''
        #if not os.path.isfile(filename):
        #    raise IOError("file: '%s' does not exist." % filename)
        
        file_ext = os.path.splitext(filename)[1].lower()
        if file_ext in set(['.csv', '.dat', '.txt']):
            ds = CSVDataSource(filename, dataset)
        else:
            raise IOError('Unknown file type: %s.' % file_ext)
        
        return ds
    
    def __init__(self, filename, dataset=None):
        self._filename = filename
        self._dataset = dataset
        self._excluded_fields = frozenset([])
    
    @abstractmethod
    def load(self):
        pass
    
    @abstractmethod
    def save(self):
        pass
    
    def _get_filename(self):
        return self._filename
    
    filename = property(fget=_get_filename)
    
    def _get_dataset(self):
        return self._dataset
    
    dataset = property(fget=_get_dataset)
    
    def _get_excluded_fields(self):
        return self._excluded_fields
    
    def _set_excluded_fields(self, fields):
        self._excluded_fields = frozenset(fields)
        
    def _del_excluded_fields(self):
        self._excluded_fields = frozenset()
        
    excluded_fields = property(fget=_get_excluded_fields, 
                               fset=_set_excluded_fields, 
                               fdel=_del_excluded_fields)
        
class CSVDataSource(DataSource):

    __slots__ = ('delimiter')
    
    def __init__(self, filename, dataset=None):
        DataSource.__init__(self, filename, dataset)
        self.delimiter = ','
        
    def load(self):
        records = csv2rec(self.filename, delimiter=self.delimiter, missingd={'-1':-2})
        print records
        records = rec_drop_fields(records, self._excluded_fields)
        self._dataset = Dataset(records)
        return self._dataset
    
    def save(self):
        if self.dataset == None:
            raise AttributeError('dataset to be stored is not specified')
        
        records = self.dataset.data
        records = rec_drop_fields(records, self._excluded_fields)
        rec2csv(records, self.filename, delimiter=self.delimiter)
    
class MultitaskDataSource(object):
    __metaclass__ = ABCMeta
    
    __slots__ = ('_filename', '_dataset')

    @staticmethod
    def create_datasource(filename, dataset=None):
        if not os.path.isfile(filename):
            raise IOError("file: '%s' does not exist." % filename)
        
        file_ext = os.path.splitext(filename)[1].lower()   
        if file_ext in set(['.csv', '.dat', '.txt']):
            ds = CSVMultitaskDataSource(filename, dataset)
        else:
            raise IOError('Unknown file type: %s.' % file_ext)
        
        return ds
        pass

    def __init__(self, filename, dataset=None):
        self._filename = filename
        self._dataset = dataset
    
    @abstractmethod
    def load(self):
        pass
    
    @abstractmethod
    def save(self):
        pass

    def _get_filename(self):
        return self._filename
    
    filename = property(fget=_get_filename)
    
    def _get_dataset(self):
        return self._dataset
    
    dataset = property(fget=_get_dataset)
    
    def _get_excluded_fields(self):
        return self._excluded_fields
    
    def _set_excluded_fields(self, fields):
        self._excluded_fields = frozenset(fields)
        
    def _del_excluded_fields(self):
        self._excluded_fields = frozenset()
        
    excluded_fields = property(fget=_get_excluded_fields, 
                               fset=_set_excluded_fields, 
                               fdel=_del_excluded_fields)

    
class CSVMultitaskDataSource(MultitaskDataSource):
    
    __slots__ = ('delimiter', '_task_key', '_task_fields', '_task_id_field', '_task_ids')
    
    def __init__(self, filename, dataset=None):
        MultitaskDataSource.__init__(self, filename, dataset)
        self.delimiter = ','
        self._task_key = 'task_id'
        self._task_fields = frozenset([])
        self._task_id_field = self.task_key
        self._task_ids = None

    def _get_task_key(self):
        return self._task_key
        
    def _set_task_key(self, key):
        if type(key) != str:
            raise TypeError('key must be of type str, not %s' % type(key))
        self._task_key = key
    
    def _del_task_key(self):
        self._task_key = 'task_id'
        
    task_key = property(fget=_get_task_key, fset=_set_task_key, fdel=_del_task_key)
    
    def _get_task_fields(self):
        return self._task_fields
    
    def _set_task_fields(self, fields):
        if not isinstance(fields, Container):
            fields = [str(fields)]
        else:
            fields = [str(element) for element in fields]
        
        self._task_fields = tuple(fields)
    
    def _del_task_fields(self):
        self._task_fields = tuple([])
        
    task_fields = property(fget=_get_task_fields, fset=_set_task_fields, fdel=_del_task_fields)
    
    def _get_task_id_field(self):
        return self._task_id_field
        
    def _set_task_id_field(self, field):
        if type(field) != str:
            raise TypeError('key must be of type str, not %s' % type(field))
        self._task_id_field = field
    
    def _del_task_id_field(self):
        self._task_id_field = self.task_key    
        
    task_id_field = property(fget=_get_task_id_field, 
                             fset=_set_task_id_field,
                             fdel=_del_task_id_field)
    
    def load(self):
        '''     
        '''
        records = csv2rec(self.filename, delimiter=self.delimiter)
        
        data_fields = list(set(records.dtype.names)-set(self.task_fields))
        task_fields = list(self._task_fields)
        print data_fields
        print task_fields
        (data, descr) = rec_disjoin(records, self.task_key, data_fields, 
                                    task_fields)
        
        self._task_ids = descr[self.task_key]
        print 'descr'
        print descr[self.task_key]
        print data[self.task_key]
        n = len(self._task_ids)
        task_data = np.empty(n, dtype=object)
        for i in xrange(n):
            task_data[i] = data[data[self.task_key] == self._task_ids[i]]
            task_data[i] = Dataset(rec_drop_fields(task_data[i], self.task_key))
        
        if not self.task_key in self.task_fields:
            task_descr = rec_drop_fields(descr, self.task_key)
        else:
            task_descr = descr
        
        task_id = None
        if self.task_id_field != None and self.task_id_field != self.task_key:
            task_id = task_descr[self.task_id_field]
            task_descr = rec_drop_fields(task_descr, self.task_id_field)
        
        print 'task_id'
        print task_id
        print task_descr
        
        self._dataset = MultitaskDataset(task_data, task_descr, task_id)
        return self._dataset
    
    def save(self):
        '''
        '''
        if self.dataset == None:
            raise Exception()
        
        dataset = self.dataset
        data_fields = dataset.data_fields
        descr_fields = dataset.descr_fields
        
        #determine the number of tasks and the complete number of datapoints
        m = self.dataset.ntasks
        n = 0
        for i in xrange(m):
            n += dataset.task_data(i).size
        
        data_ids = np.empty(n, dtype=np.str)
        descr_ids = self._task_ids if self._task_ids != None else np.array(xrange(1,m+1))
        data_records = np.empty(n, dtype=data_fields).view(np.recarray)
        descr_records = np.empty(n, dtype=descr_fields).view(np.recarray)
        
        offset = 0
        for i in xrange(m):
            task_dataset = dataset.task_data(i)
            k = task_dataset.size
            
            data_ids[offset:offset+k] = descr_ids[i]
            data_records[offset:offset+k] = task_dataset.data
        
            offset += k
        
        data_records = rec_append_fields(data_records, self.task_key, data_ids)
        descr_records = rec_append_fields(descr_records, self.task_key, descr_ids)
        if self.task_description_field != self.task_key:
            descr_records = rec_append_fields(descr_records, 
                                              self.task_description_field, 
                                              dataset.task_identifiers)
            
        
        records = rec_join(self.task_key, data_records, descr_records)
        rec2csv(records, self.filename, delimiter=self.delimiter)
        

def dataset2array(dataset):
    '''
    @todo: - parametrizable by the requested fields to be extracted
    '''
    
    if isinstance(dataset, Dataset):
        return __arrays_from_dataset(dataset)
    elif isinstance(dataset, MultitaskDataset):
        return __arrays_from_multidataset(dataset)
    else:
        TypeError('Unknown type of dataset!')
    
def __arrays_from_dataset(dataset, fields=None):
    '''
    '''
    X = dataset.covariate_data()
    y = dataset.target_data()
    return (X, y)

def __arrays_from_multidataset(dataset, data_fields=None, task_fields=None):
    '''
    @todo: -return empty z array, if task description is not defined
    '''
    n = dataset.ntasks
    X = [0]*n #covariate data
    Y = [0]*n #target data
    Z = [0]*n #task data
    for i in xrange(n):
        task_dataset, task_descr = dataset[i]
        X[i], Y[i] = __arrays_from_dataset(task_dataset, data_fields)
        Z[i] = np.array(task_descr.tolist())
    
    return (X,Y,Z)
        


if __name__ == '__main__':
    ds = DataSource.create_datasource("/home/marcel/datasets/debug/flat_dataset_nan.csv")
    dataset = ds.load()
    print dataset.data
    ds = DataSource.create_datasource("/home/marcel/datasets/debug/flat_dataset_nan_copy.csv", dataset)
    ds.save()
    
    ds = MultitaskDataSource.create_datasource("/home/marcel/datasets/debug/multi_dataset.csv")
    ds.task_fields = ['g']
    multi_dataset = ds.load()
    print multi_dataset.task_descriptions
    print multi_dataset.task_identifiers
    for (task_dataset, meta_data) in multi_dataset:
        print task_dataset.data
        print meta_data

