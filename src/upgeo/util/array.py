'''
Created on Mar 10, 2011

@author: marcel
'''

import numpy as np
import csv, warnings, copy, os, operator

from matplotlib import cbook
from collections import OrderedDict


def unique(ar, return_index=False, return_inverse=False):
    """
    Find the unique elements of an array.

    Returns the sorted unique elements of an array. There are two optional
    outputs in addition to the unique elements: the indices of the input array
    that give the unique values, and the indices of the unique array that
    reconstruct the input array.

    Parameters
    ----------
    ar : array_like
        Input array. This will be flattened if it is not already 1-D.
    return_index : bool, optional
        If True, also return the indices of `ar` that result in the unique
        array.
    return_inverse : bool, optional
        If True, also return the indices of the unique array that can be used
        to reconstruct `ar`.

    Returns
    -------
    unique : ndarray
        The sorted unique values.
    unique_indices : ndarray, optional
        The indices of the first occurrences of the unique values in the
        (flattened) original array. Only provided if `return_index` is True.
    unique_inverse : ndarray, optional
        The indices to reconstruct the (flattened) original array from the
        unique array. Only provided if `return_inverse` is True.

    See Also
    --------
    numpy.lib.arraysetops : Module with a number of other functions for
                            performing set operations on arrays.

    Examples
    --------
    >>> np.unique([1, 1, 2, 2, 3, 3])
    array([1, 2, 3])
    >>> a = np.array([[1, 1], [2, 3]])
    >>> np.unique(a)
    array([1, 2, 3])

    Return the indices of the original array that give the unique values:

    >>> a = np.array(['a', 'b', 'b', 'c', 'a'])
    >>> u, indices = np.unique(a, return_index=True)
    >>> u
    array(['a', 'b', 'c'],
           dtype='|S1')
    >>> indices
    array([0, 1, 3])
    >>> a[indices]
    array(['a', 'b', 'c'],
           dtype='|S1')

    Reconstruct the input array from the unique values:

    >>> a = np.array([1, 2, 6, 4, 2, 3, 2])
    >>> u, indices = np.unique(a, return_inverse=True)
    >>> u
    array([1, 2, 3, 4, 6])
    >>> indices
    array([0, 1, 4, 3, 1, 2, 1])
    >>> u[indices]
    array([1, 2, 6, 4, 2, 3, 2])

    """
    try:
        ar = ar.flatten()
    except AttributeError:
        if not return_inverse and not return_index:
            items = sorted(set(ar))
            return np.asarray(items)
        else:
            ar = np.asanyarray(ar).flatten()

    if ar.size == 0:
        if return_inverse and return_index:
            return ar, np.empty(0, np.bool), np.empty(0, np.bool)
        elif return_inverse or return_index:
            return ar, np.empty(0, np.bool)
        else:
            return ar

    if return_inverse or return_index:
        if return_index:
            perm = ar.argsort(kind='mergesort')
        else:
            perm = ar.argsort()
        aux = ar[perm]
        flag = np.concatenate(([True], aux[1:] != aux[:-1]))
        if return_inverse:
            iflag = np.cumsum(flag) - 1
            iperm = perm.argsort()
            if return_index:
                return aux[flag], perm[flag], iflag[iperm]
            else:
                return aux[flag], iflag[iperm]
        else:
            return aux[flag], perm[flag]

    else:
        ar.sort()
        flag = np.concatenate(([True], ar[1:] != ar[:-1]))
        return ar[flag]


def invweight(a):
    '''
    '''
    a = np.asarray(a).ravel()
    n = len(a)
    w = np.empty(n)
    if n > 0:
        w[0] = 1
        for i in xrange(1,n):
            w[i] = w[i-1]*(1-a[i-1])
    return w

def weight(a):
    a = np.asarray(a).ravel()
    n = len(a)
    w = np.empty(n)
    if n > 0:
        iw = invweight(a)
        for i in xrange(n):
            w[i] = a[i]*iw[i]
    return w
    

def isempty(a):
    '''
    '''
    return a == None or np.asarray(a).size == 0
    

def count_unique(a):
    '''
    the more elegant way does not work because np.unique did not return
    the index of the fist occurence, instead we are using bincount on the
    inverse indices.
    
    u, indices = np.unique(np.sort(a), return_index=True)
    counts = np.diff(np.r_[indices, len(a)])
    '''
    a = np.ravel(a)
    
    u, i = np.unique(np.sort(a), return_inverse=True)
    counts = np.bincount(i)
    return u, counts

def sqix_(*args):
    """
    Construct an open mesh from multiple sequences.

    """
    out = []
    n = len(args)
    
    #determine the number of arguments with more than 1 value
    k = 0
    for i in range(n):
        seq = np.atleast_1d(args[i])
        if seq.size > 1:
            k += 1
        if (seq.ndim != 1):
            raise ValueError, "Cross index must be 1 dimensional"
        if issubclass(seq.dtype.type, np.bool_):
            seq = seq.nonzero()[0]
        out.append(seq)
    
    baseshape = [1]*k
    j = 0
    for i in range(n):
        seq = out[i]
        m = len(seq)
        if m > 1:
            baseshape[j] = len(seq)
            out[i] = seq.reshape(tuple(baseshape))
            baseshape[j] = 1
            j += 1
    return tuple(out)

def unique2d(a, return_index=False, transpose=False):
    '''
    Returns all unique rows or columns of a 2-dimensional matrix.
    
    @todo: - implement a more general version
           - use a similarity threshold
           - code optimization
    '''
    a = np.atleast_2d(a)
    if a.ndim != 2:
        raise ValueError('dimension of a cannot be greater than 2.')
    
    if transpose == True:
        a = a.T
    
    dt = a.dtype
    n,d = a.shape
    
    keepers = {}
    for i in xrange(n):
        key = tuple(a[i].tolist())
        if not key in keepers:
            keepers[key] = i
    
    m = len(keepers)
    unique = np.empty([m,d], dt)
    unique_index = np.empty(m, 'int32')
    i = 0
    for key, value in keepers.iteritems():
        unique[i] = np.asarray(key, dt)
        unique_index[i] = value
        i += 1
            
    return unique if return_index == False else (unique, unique_index)

def rec_index_of(rec, names):
    '''
    '''
    if not isinstance(rec, np.recarray):
        raise TypeError('rec must be an instance of numpy.recarray')
    
    names = np.asarray(names).ravel()
    attr_dict = dict([(name, i) for i,name in enumerate(rec.dtype.names)])
    indices = [attr_dict[name] for name in names]
    return indices
    
def rec_names_at(rec, idx):
    if not isinstance(rec, np.recarray):
        raise TypeError('rec must be an instance of numpy.recarray')
    
    idx = np.asarray(idx).ravel()
    names = [rec.dtype.names[i] for i in idx]
    return names 

def rec_disjoin(rec, key, field1, field2=[]):
    '''
    Disjoin a record array on single column key into two record arrays, where
    the seperation is specified by the fields. The key field is attached at 
    the of both record arrays.
    '''
    if not isinstance(rec, np.recarray):
        raise TypeError('rec must be an instance of numpy.recarray')
    
    newrec1 = rec[field1 + [key]]
    newrec2 = rec[field2 + [key]]

    #key_rec_map = dict()
    key_rec_map = OrderedDict()
    for row in newrec2:
        if row[key] not in key_rec_map:
            key_rec_map[row[key]] = row.tolist()
    
    newrec2 = np.rec.array(key_rec_map.values(), dtype=newrec2.dtype) 
    return (newrec1, newrec2)

def rec_append_fields(rec, names, arrs, dtypes=None):
    """
    Return a new record array with field names populated with data
    from arrays in *arrs*.  If appending a single field, then *names*,
    *arrs* and *dtypes* do not have to be lists. They can just be the
    values themselves.
    """
    if (not cbook.is_string_like(names) and cbook.iterable(names) \
            and len(names) and cbook.is_string_like(names[0])):
        if len(names) != len(arrs):
            raise ValueError, "number of arrays do not match number of names"
    else: # we have only 1 name and 1 array
        names = [names]
        arrs = [arrs]
    arrs = map(np.asarray, arrs)
    if dtypes is None:
        dtypes = [a.dtype for a in arrs]
    elif not cbook.iterable(dtypes):
        dtypes = [dtypes]
    if len(arrs) != len(dtypes):
        if len(dtypes) == 1:
            dtypes = dtypes * len(arrs)
        else:
            raise ValueError, "dtypes must be None, a single dtype or a list"

    newdtype = np.dtype(rec.dtype.descr + zip(names, dtypes))
    newrec = np.recarray(rec.shape, dtype=newdtype)
    for field in rec.dtype.fields:
        newrec[field] = rec[field]
    for name, arr in zip(names, arrs):
        newrec[name] = arr
    return newrec


def rec_drop_fields(rec, names):
    """
    Return a new numpy record array with fields in *names* dropped.
    """
    if type(names) == str:
        names = [names]
    

    names = set(names)
    Nr = len(rec)

    newdtype = np.dtype([(name, rec.dtype[name]) for name in rec.dtype.names
                       if name not in names])

    newrec = np.recarray(rec.shape, dtype=newdtype)
    for field in newdtype.names:
        newrec[field] = rec[field]

    return newrec

def rec_keep_fields(rec, names):
    """
    Return a new numpy record array with only fields listed in names
    """

    if cbook.is_string_like(names):
        names = names.split(',')

    arrays = []
    for name in names:
        arrays.append(rec[name])

    return np.rec.fromarrays(arrays, names=names)



def rec_groupby(r, groupby, stats):
    """
    *r* is a numpy record array

    *groupby* is a sequence of record array attribute names that
    together form the grouping key.  eg ('date', 'productcode')

    *stats* is a sequence of (*attr*, *func*, *outname*) tuples which
    will call ``x = func(attr)`` and assign *x* to the record array
    output with attribute *outname*.  For example::

      stats = ( ('sales', len, 'numsales'), ('sales', np.mean, 'avgsale') )

    Return record array has *dtype* names for each attribute name in
    the the *groupby* argument, with the associated group values, and
    for each outname name in the *stats* argument, with the associated
    stat summary output.
    """
    # build a dictionary from groupby keys-> list of indices into r with
    # those keys
    rowd = dict()
    for i, row in enumerate(r):
        key = tuple([row[attr] for attr in groupby])
        rowd.setdefault(key, []).append(i)

    # sort the output by groupby keys
    keys = rowd.keys()
    keys.sort()

    rows = []
    for key in keys:
        row = list(key)
        # get the indices for this groupby key
        ind = rowd[key]
        thisr = r[ind]
        # call each stat function for this groupby slice
        row.extend([func(thisr[attr]) for attr, func, outname in stats])
        rows.append(row)

    # build the output record array with groupby and outname attributes
    attrs, funcs, outnames = zip(*stats)
    names = list(groupby)
    names.extend(outnames)
    return np.rec.fromrecords(rows, names=names)



def rec_summarize(r, summaryfuncs):
    """
    *r* is a numpy record array

    *summaryfuncs* is a list of (*attr*, *func*, *outname*) tuples
    which will apply *func* to the the array *r*[attr] and assign the
    output to a new attribute name *outname*.  The returned record
    array is identical to *r*, with extra arrays for each element in
    *summaryfuncs*.

    """

    names = list(r.dtype.names)
    arrays = [r[name] for name in names]

    for attr, func, outname in summaryfuncs:
        names.append(outname)
        arrays.append(np.asarray(func(r[attr])))

    return np.rec.fromarrays(arrays, names=names)


def rec_join(key, r1, r2, jointype='inner', defaults=None, r1postfix='1', r2postfix='2'):
    """
    Join record arrays *r1* and *r2* on *key*; *key* is a tuple of
    field names -- if *key* is a string it is assumed to be a single
    attribute name. If *r1* and *r2* have equal values on all the keys
    in the *key* tuple, then their fields will be merged into a new
    record array containing the intersection of the fields of *r1* and
    *r2*.

    *r1* (also *r2*) must not have any duplicate keys.

    The *jointype* keyword can be 'inner', 'outer', 'leftouter'.  To
    do a rightouter join just reverse *r1* and *r2*.

    The *defaults* keyword is a dictionary filled with
    ``{column_name:default_value}`` pairs.

    The keywords *r1postfix* and *r2postfix* are postfixed to column names
    (other than keys) that are both in *r1* and *r2*.
    """

    if cbook.is_string_like(key):
        key = (key, )

    for name in key:
        if name not in r1.dtype.names:
            raise ValueError('r1 does not have key field %s'%name)
        if name not in r2.dtype.names:
            raise ValueError('r2 does not have key field %s'%name)

    def makekey(row):
        return tuple([row[name] for name in key])

    r1d = dict([(makekey(row),i) for i,row in enumerate(r1)])
    r2d = dict([(makekey(row),i) for i,row in enumerate(r2)])

    r1keys = set(r1d.keys())
    r2keys = set(r2d.keys())

    common_keys = r1keys & r2keys

    r1ind = np.array([r1d[k] for k in common_keys])
    r2ind = np.array([r2d[k] for k in common_keys])

    common_len = len(common_keys)
    left_len = right_len = 0
    if jointype == "outer" or jointype == "leftouter":
        left_keys = r1keys.difference(r2keys)
        left_ind = np.array([r1d[k] for k in left_keys])
        left_len = len(left_ind)
    if jointype == "outer":
        right_keys = r2keys.difference(r1keys)
        right_ind = np.array([r2d[k] for k in right_keys])
        right_len = len(right_ind)

    def key_desc(name):
        'if name is a string key, use the larger size of r1 or r2 before merging'
        dt1 = r1.dtype[name]
        if dt1.type != np.string_:
            return (name, dt1.descr[0][1])

        dt2 = r1.dtype[name]
        assert dt2==dt1
        if dt1.num>dt2.num:
            return (name, dt1.descr[0][1])
        else:
            return (name, dt2.descr[0][1])


    keydesc = [key_desc(name) for name in key]

    def mapped_r1field(name):
        """
        The column name in *newrec* that corresponds to the column in *r1*.
        """
        if name in key or name not in r2.dtype.names: return name
        else: return name + r1postfix

    def mapped_r2field(name):
        """
        The column name in *newrec* that corresponds to the column in *r2*.
        """
        if name in key or name not in r1.dtype.names: return name
        else: return name + r2postfix

    r1desc = [(mapped_r1field(desc[0]), desc[1]) for desc in r1.dtype.descr if desc[0] not in key]
    r2desc = [(mapped_r2field(desc[0]), desc[1]) for desc in r2.dtype.descr if desc[0] not in key]
    newdtype = np.dtype(keydesc + r1desc + r2desc)

    newrec = np.recarray((common_len + left_len + right_len,), dtype=newdtype)

    if defaults is not None:
        for thiskey in defaults:
            if thiskey not in newdtype.names:
                warnings.warn('rec_join defaults key="%s" not in new dtype names "%s"'%(
                    thiskey, newdtype.names))

    for name in newdtype.names:
        dt = newdtype[name]
        if dt.kind in ('f', 'i'):
            newrec[name] = 0

    if jointype != 'inner' and defaults is not None: # fill in the defaults enmasse
        newrec_fields = newrec.dtype.fields.keys()
        for k, v in defaults.items():
            if k in newrec_fields:
                newrec[k] = v

    for field in r1.dtype.names:
        newfield = mapped_r1field(field)
        if common_len:
            newrec[newfield][:common_len] = r1[field][r1ind]
        if (jointype == "outer" or jointype == "leftouter") and left_len:
            newrec[newfield][common_len:(common_len+left_len)] = r1[field][left_ind]

    for field in r2.dtype.names:
        newfield = mapped_r2field(field)
        if field not in key and common_len:
            newrec[newfield][:common_len] = r2[field][r2ind]
        if jointype == "outer" and right_len:
            newrec[newfield][-right_len:] = r2[field][right_ind]

    newrec.sort(order=key)

    return newrec

def recs_join(key, name, recs, jointype='outer', missing=0., postfixes=None):
    """
    Join a sequence of record arrays on single column key.

    This function only joins a single column of the multiple record arrays

    *key*
      is the column name that acts as a key

    *name*
      is the name of the column that we want to join

    *recs*
      is a list of record arrays to join

    *jointype*
      is a string 'inner' or 'outer'

    *missing*
      is what any missing field is replaced by

    *postfixes*
      if not None, a len recs sequence of postfixes

    returns a record array with columns [rowkey, name0, name1, ... namen-1].
    or if postfixes [PF0, PF1, ..., PFN-1] are supplied,
      [rowkey, namePF0, namePF1, ... namePFN-1].

    Example::

      r = recs_join("date", "close", recs=[r0, r1], missing=0.)

    """
    results = []
    aligned_iters = cbook.align_iterators(operator.attrgetter(key), *[iter(r) for r in recs])

    def extract(r):
        if r is None: return missing
        else: return r[name]


    if jointype == "outer":
        for rowkey, row in aligned_iters:
            results.append([rowkey] + map(extract, row))
    elif jointype == "inner":
        for rowkey, row in aligned_iters:
            if None not in row: # throw out any Nones
                results.append([rowkey] + map(extract, row))

    if postfixes is None:
        postfixes = ['%d'%i for i in range(len(recs))]
    names = ",".join([key] + ["%s%s" % (name, postfix) for postfix in postfixes])
    return np.rec.fromrecords(results, names=names)


def csv2rec(fname, comments='#', skiprows=0, checkrows=0, delimiter=',',
            converterd=None, names=None, missing='', missingd=None,
            use_mrecords=False):
    """
    Load data from comma/space/tab delimited file in *fname* into a
    numpy record array and return the record array.

    If *names* is *None*, a header row is required to automatically
    assign the recarray names.  The headers will be lower cased,
    spaces will be converted to underscores, and illegal attribute
    name characters removed.  If *names* is not *None*, it is a
    sequence of names to use for the column names.  In this case, it
    is assumed there is no header row.


    - *fname*: can be a filename or a file handle.  Support for gzipped
      files is automatic, if the filename ends in '.gz'

    - *comments*: the character used to indicate the start of a comment
      in the file

    - *skiprows*: is the number of rows from the top to skip

    - *checkrows*: is the number of rows to check to validate the column
      data type.  When set to zero all rows are validated.

    - *converterd*: if not *None*, is a dictionary mapping column number or
      munged column name to a converter function.

    - *names*: if not None, is a list of header names.  In this case, no
      header will be read from the file

    - *missingd* is a dictionary mapping munged column names to field values
      which signify that the field does not contain actual data and should
      be masked, e.g. '0000-00-00' or 'unused'

    - *missing*: a string whose value signals a missing field regardless of
      the column it appears in

    - *use_mrecords*: if True, return an mrecords.fromrecords record array if any of the data are missing

      If no rows are found, *None* is returned -- see :file:`examples/loadrec.py`
    """

    if converterd is None:
        converterd = dict()

    if missingd is None:
        missingd = {}

    import dateutil.parser
    import datetime
    parsedate = dateutil.parser.parse


    fh = cbook.to_filehandle(fname)


    class FH:
        """
        For space-delimited files, we want different behavior than
        comma or tab.  Generally, we want multiple spaces to be
        treated as a single separator, whereas with comma and tab we
        want multiple commas to return multiple (empty) fields.  The
        join/strip trick below effects this.
        """
        def __init__(self, fh):
            self.fh = fh

        def close(self):
            self.fh.close()

        def seek(self, arg):
            self.fh.seek(arg)

        def fix(self, s):
            return ' '.join(s.split())


        def next(self):
            return self.fix(self.fh.next())

        def __iter__(self):
            for line in self.fh:
                yield self.fix(line)

    if delimiter==' ':
        fh = FH(fh)

    reader = csv.reader(fh, delimiter=delimiter)
    def process_skiprows(reader):
        if skiprows:
            for i, row in enumerate(reader):
                if i>=(skiprows-1): break

        return fh, reader

    process_skiprows(reader)

    def ismissing(name, val):
        "Should the value val in column name be masked?"

        if val == missing or val == missingd.get(name) or val == '':
            return True
        else:
            return False

    def with_default_value(func, default):
        def newfunc(name, val):
            if ismissing(name, val):
                return default
            else:
                return func(val)
        return newfunc


    def mybool(x):
        if x=='True': return True
        elif x=='False': return False
        else: raise ValueError('invalid bool')

    dateparser = dateutil.parser.parse
    mydateparser = with_default_value(dateparser, datetime.date(1,1,1))
    myfloat = with_default_value(float, np.nan)
    myint = with_default_value(int, np.NaN)
    mystr = with_default_value(str, '')
    mybool = with_default_value(mybool, None)

    def mydate(x):
        # try and return a date object
        d = dateparser(x)

        if d.hour>0 or d.minute>0 or d.second>0:
            raise ValueError('not a date')
        return d.date()
    mydate = with_default_value(mydate, datetime.date(1,1,1))

    def get_func(name, item, func):
        # promote functions in this order
        funcmap = {mybool:myint,myint:myfloat, myfloat:mydate, mydate:mydateparser, mydateparser:mystr}
        try: func(name, item)
        except:
            if func==mystr:
                raise ValueError('Could not find a working conversion function')
            else: return get_func(name, item, funcmap[func])    # recurse
        else: return func


    # map column names that clash with builtins -- TODO - extend this list
    itemd = {
        'return' : 'return_',
        'file' : 'file_',
        'print' : 'print_',
        }

    def get_converters(reader):

        converters = None
        for i, row in enumerate(reader):
            if i==0:
                converters = [mybool]*len(row)
            if checkrows and i>checkrows:
                break
            #print i, len(names), len(row)
            #print 'converters', zip(converters, row)
            for j, (name, item) in enumerate(zip(names, row)):
                func = converterd.get(j)
                if func is None:
                    func = converterd.get(name)
                if func is None:
                    #if not item.strip(): continue
                    func = converters[j]
                    if len(item.strip()):
                        func = get_func(name, item, func)
                else:
                    # how should we handle custom converters and defaults?
                    func = with_default_value(func, None)
                converters[j] = func
        return converters

    # Get header and remove invalid characters
    needheader = names is None

    if needheader:
        for row in reader:
            #print 'csv2rec', row
            if len(row) and row[0].startswith(comments):
                continue
            headers = row
            break

        # remove these chars
        delete = set("""~!@#$%^&*()-=+~\|]}[{';: /?.>,<""")
        delete.add('"')

        names = []
        seen = dict()
        for i, item in enumerate(headers):
            item = item.strip().lower().replace(' ', '_')
            item = ''.join([c for c in item if c not in delete])
            if not len(item):
                item = 'column%d'%i

            item = itemd.get(item, item)
            cnt = seen.get(item, 0)
            if cnt>0:
                names.append(item + '_%d'%cnt)
            else:
                names.append(item)
            seen[item] = cnt+1

    else:
        if cbook.is_string_like(names):
            names = [n.strip() for n in names.split(',')]

    # get the converter functions by inspecting checkrows
    converters = get_converters(reader)
    if converters is None:
        raise ValueError('Could not find any valid data in CSV file')

    # reset the reader and start over
    fh.seek(0)
    reader = csv.reader(fh, delimiter=delimiter)
    process_skiprows(reader)

    if needheader:
        while 1:
            # skip past any comments and consume one line of column header
            row = reader.next()
            if len(row) and row[0].startswith(comments):
                continue
            break

    # iterate over the remaining rows and convert the data to date
    # objects, ints, or floats as approriate
    rows = []
    rowmasks = []
    for i, row in enumerate(reader):
        if not len(row): continue
        if row[0].startswith(comments): continue
        rows.append([func(name, val) for func, name, val in zip(converters, names, row)])
        rowmasks.append([ismissing(name, val) for name, val in zip(names, row)])
    fh.close()

    if not len(rows):
        return None

    if use_mrecords and np.any(rowmasks):
        try: from numpy.ma import mrecords
        except ImportError:
            raise RuntimeError('numpy 1.05 or later is required for masked array support')
        else:
            r = mrecords.fromrecords(rows, names=names, mask=rowmasks)
    else:
        r = np.rec.fromrecords(rows, names=names)
    return r


# a series of classes for describing the format intentions of various rec views
class FormatObj:
    def tostr(self, x):
        return self.toval(x)

    def toval(self, x):
        return str(x)

    def fromstr(self, s):
        return s

class FormatString(FormatObj):
    def tostr(self, x):
        val = repr(x)
        return val[1:-1]

#class FormatString(FormatObj):
#    def tostr(self, x):
#        return '"%r"'%self.toval(x)



class FormatFormatStr(FormatObj):
    def __init__(self, fmt):
        self.fmt = fmt

    def tostr(self, x):
        if x is None: return 'None'
        return self.fmt%self.toval(x)




class FormatFloat(FormatFormatStr):
    def __init__(self, precision=4, scale=1.):
        FormatFormatStr.__init__(self, '%%1.%df'%precision)
        self.precision = precision
        self.scale = scale

    def toval(self, x):
        if x is not None:
            x = x * self.scale
        return x

    def fromstr(self, s):
        return float(s)/self.scale


class FormatInt(FormatObj):

    def tostr(self, x):
        return '%d'%int(x)

    def toval(self, x):
        return int(x)

    def fromstr(self, s):
        return int(s)

class FormatBool(FormatObj):


    def toval(self, x):
        return str(x)

    def fromstr(self, s):
        return bool(s)

class FormatPercent(FormatFloat):
    def __init__(self, precision=4):
        FormatFloat.__init__(self, precision, scale=100.)

class FormatThousands(FormatFloat):
    def __init__(self, precision=4):
        FormatFloat.__init__(self, precision, scale=1e-3)


class FormatMillions(FormatFloat):
    def __init__(self, precision=4):
        FormatFloat.__init__(self, precision, scale=1e-6)


class FormatDate(FormatObj):
    def __init__(self, fmt):
        self.fmt = fmt

    def toval(self, x):
        if x is None: return 'None'
        return x.strftime(self.fmt)

    def fromstr(self, x):
        import dateutil.parser
        return dateutil.parser.parse(x).date()

class FormatDatetime(FormatDate):
    def __init__(self, fmt='%Y-%m-%d %H:%M:%S'):
        FormatDate.__init__(self, fmt)

    def fromstr(self, x):
        import dateutil.parser
        return dateutil.parser.parse(x)




defaultformatd = {
    np.bool_ : FormatBool(),
    np.int16 : FormatInt(),
    np.int32 : FormatInt(),
    np.int64 : FormatInt(),
    np.float32 : FormatFloat(),
    np.float64 : FormatFloat(),
    np.object_ : FormatObj(),
    np.string_ : FormatString(),
    }

def get_formatd(r, formatd=None):
    'build a formatd guaranteed to have a key for every dtype name'
    if formatd is None:
        formatd = dict()

    for i, name in enumerate(r.dtype.names):
        dt = r.dtype[name]
        format = formatd.get(name)
        if format is None:
            format = defaultformatd.get(dt.type, FormatObj())
        formatd[name] = format
    return formatd

def csvformat_factory(format):
    format = copy.deepcopy(format)
    if isinstance(format, FormatFloat):
        format.scale = 1. # override scaling for storage
        format.fmt = '%r'
    return format

def rec2txt(r, header=None, padding=3, precision=3, fields=None):
    """
    Returns a textual representation of a record array.

    *r*: numpy recarray

    *header*: list of column headers

    *padding*: space between each column

    *precision*: number of decimal places to use for floats.
        Set to an integer to apply to all floats.  Set to a
        list of integers to apply precision individually.
        Precision for non-floats is simply ignored.

    *fields* : if not None, a list of field names to print.  fields
    can be a list of strings like ['field1', 'field2'] or a single
    comma separated string like 'field1,field2'

    Example::

      precision=[0,2,3]

    Output::

      ID    Price   Return
      ABC   12.54    0.234
      XYZ    6.32   -0.076
    """

    if fields is not None:
        r = rec_keep_fields(r, fields)

    if cbook.is_numlike(precision):
        precision = [precision]*len(r.dtype)

    def get_type(item,atype=int):
        tdict = {None:int, int:float, float:str}
        try: atype(str(item))
        except: return get_type(item,tdict[atype])
        return atype

    def get_justify(colname, column, precision):
        ntype = type(column[0])

        if ntype==np.str or ntype==np.str_ or ntype==np.string0 or ntype==np.string_:
            length = max(len(colname),column.itemsize)
            return 0, length+padding, "%s" # left justify

        if ntype==np.int or ntype==np.int16 or ntype==np.int32 or ntype==np.int64 or ntype==np.int8 or ntype==np.int_:
            length = max(len(colname),np.max(map(len,map(str,column))))
            return 1, length+padding, "%d" # right justify

        # JDH: my powerbook does not have np.float96 using np 1.3.0
        """
        In [2]: np.__version__
        Out[2]: '1.3.0.dev5948'

        In [3]: !uname -a
        Darwin Macintosh-5.local 9.4.0 Darwin Kernel Version 9.4.0: Mon Jun  9 19:30:53 PDT 2008; root:xnu-1228.5.20~1/RELEASE_I386 i386 i386

        In [4]: np.float96
        ---------------------------------------------------------------------------
        AttributeError                            Traceback (most recent call la
        """
        if ntype==np.float or ntype==np.float32 or ntype==np.float64 or (hasattr(np, 'float96') and (ntype==np.float96)) or ntype==np.float_:
            fmt = "%." + str(precision) + "f"
            length = max(len(colname),np.max(map(len,map(lambda x:fmt%x,column))))
            return 1, length+padding, fmt   # right justify

        return 0, max(len(colname),np.max(map(len,map(str,column))))+padding, "%s"

    if header is None:
        header = r.dtype.names

    justify_pad_prec = [get_justify(header[i],r.__getitem__(colname),precision[i]) for i, colname in enumerate(r.dtype.names)]

    justify_pad_prec_spacer = []
    for i in range(len(justify_pad_prec)):
        just,pad,prec = justify_pad_prec[i]
        if i == 0:
            justify_pad_prec_spacer.append((just,pad,prec,0))
        else:
            pjust,ppad,pprec = justify_pad_prec[i-1]
            if pjust == 0 and just == 1:
                justify_pad_prec_spacer.append((just,pad-padding,prec,0))
            elif pjust == 1 and just == 0:
                justify_pad_prec_spacer.append((just,pad,prec,padding))
            else:
                justify_pad_prec_spacer.append((just,pad,prec,0))

    def format(item, just_pad_prec_spacer):
        just, pad, prec, spacer = just_pad_prec_spacer
        if just == 0:
            return spacer*' ' + str(item).ljust(pad)
        else:
            if get_type(item) == float:
                item = (prec%float(item))
            elif get_type(item) == int:
                item = (prec%int(item))

            return item.rjust(pad)

    textl = []
    textl.append(''.join([format(colitem,justify_pad_prec_spacer[j]) for j, colitem in enumerate(header)]))
    for i, row in enumerate(r):
        textl.append(''.join([format(colitem,justify_pad_prec_spacer[j]) for j, colitem in enumerate(row)]))
        if i==0:
            textl[0] = textl[0].rstrip()

    text = os.linesep.join(textl)
    return text



def rec2csv(r, fname, delimiter=',', formatd=None, missing='',
            missingd=None, withheader=True):
    """
    Save the data from numpy recarray *r* into a
    comma-/space-/tab-delimited file.  The record array dtype names
    will be used for column headers.

    *fname*: can be a filename or a file handle.  Support for gzipped
      files is automatic, if the filename ends in '.gz'

    *withheader*: if withheader is False, do not write the attribute
      names in the first row

    for formatd type FormatFloat, we override the precision to store
    full precision floats in the CSV file


    .. seealso::

        :func:`csv2rec`
            For information about *missing* and *missingd*, which can
            be used to fill in masked values into your CSV file.
    """

    if missingd is None:
        missingd = dict()

    def with_mask(func):
        def newfunc(val, mask, mval):
            if mask:
                return mval
            else:
                return func(val)
        return newfunc

    if r.ndim != 1:
        raise ValueError('rec2csv only operates on 1 dimensional recarrays')

    formatd = get_formatd(r, formatd)
    funcs = []
    for i, name in enumerate(r.dtype.names):
        funcs.append(with_mask(csvformat_factory(formatd[name]).tostr))

    fh, opened = cbook.to_filehandle(fname, 'wb', return_opened=True)
    writer = csv.writer(fh, delimiter=delimiter)
    header = r.dtype.names
    if withheader:
        writer.writerow(header)

    # Our list of specials for missing values
    mvals = []
    for name in header:
        mvals.append(missingd.get(name, missing))

    ismasked = False
    if len(r):
        row = r[0]
        ismasked = hasattr(row, '_fieldmask')

    for row in r:
        if ismasked:
            row, rowmask = row.item(), row._fieldmask.item()
        else:
            rowmask = [False] * len(row)
        writer.writerow([func(val, mask, mval) for func, val, mask, mval
                         in zip(funcs, row, rowmask, mvals)])
    if opened:
        fh.close()

if __name__ == '__main__':
    a = np.asarray([0.10, 0.34, 0.843])
    print a
    print invweight(a)
    print weight(a)
    print np.sum(weight(a))