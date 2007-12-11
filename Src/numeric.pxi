# Pyrex include file for access to Numeric/NumPy arrays
#
# This file is part of the Crystallographic Data Toolkit and
# distributed under the CeCILL-C licence.See the file LICENCE
# for the full text of this licence.
#
# Written by Konrad Hinsen.
#

cdef extern from "Scientific/arrayobject.h": 

    void import_array()

    cdef enum Pyarray_TYPES:
        PyArray_CHAR, PyArray_UBYTE, PyArray_SBYTE,
        PyArray_SHORT, PyArray_USHORT, 
        PyArray_INT, PyArray_UINT, 
        PyArray_LONG,
        PyArray_FLOAT, PyArray_DOUBLE, 
        PyArray_CFLOAT, PyArray_CDOUBLE,
        PyArray_OBJECT,
        PyArray_NTYPES, PyArray_NOTYPE

    struct PyArray_Descr: 
        int type_num, elsize 
        char type 

    ctypedef struct PyArrayObject:
        char *data 
        int nd 
        int *dimensions, *strides 
        PyArray_Descr *descr 
        int flags

    ctypedef class Scientific.N.array_type [object PyArrayObject]: 
        cdef char *data 
        cdef int nd 
        cdef int *dimensions, *strides 
        cdef object base 
        cdef PyArray_Descr *descr 
        cdef int flags

    object PyArray_FromDims(int n_dimensions, int dimensions[], int item_type)
    object PyArray_FromDimsAndData(int n_dimensions, int dimensions[],
                                   int item_type, char *data)
    int PyArray_ISCONTIGUOUS(array_type a)

import_array()
