# Read MTZ files using libmtz from CCP4.
#
# This file is part of the Crystallographic Data Toolkit and
# distributed under the CeCILL-C licence. See the file LICENCE
# for the full text of this licence.
#
# Written by Konrad Hinsen.
#

include "numeric.pxi"

from Scientific import N
import copy
import os

cdef extern from "cmtzlib.h":

    ctypedef struct SYMGRP:
        int spcgrp            # spacegroup number
        char *spcgrpname      # spacegroup name
        int nsym              # number of symmetry operations
        float sym[192][4][4]  # symmetry operations 
                              # (translations in [*][3])
        int nsymp             # number of primitive symmetry ops.
        char symtyp           # lattice type (P,A,B,C,I,F,R)
        char *pgname          # pointgroup name

    ctypedef struct MTZCOL:
        char label[31]        # column name as given by user
        char type[3]          # column type
        int active            # whether column in active list
        unsigned int source   # column index in input file
        float min             # minimum data element
        float max             # maximum data element
        float *ref            # data array

    ctypedef struct MTZSET:
        int setid             # Dataset id
        char dname[65]        # Dataset name
        float wavelength      # Dataset wavelength
        int ncol              # number of columns
        MTZCOL **col          # columns

    ctypedef struct MTZXTAL:
        int xtalid            # Crystal id
        char xname[65]        # Crystal name
        char pname[65]        # Project name
        float cell[6]         # Crystal cell
        float resmin          # Low resolution limit
        float resmax          # High resolution limit
        int nset              # number of datasets
        MTZSET **set          # datasets

    ctypedef struct MNF:
        char amnf[4]          # character representation of "missing number"
        float fmnf            # value used for "missing number"

    ctypedef struct MTZ:
        char *title           # title of mtz structure
        int nxtal             # number of crystals
        MNF mnf               # value of missing number flag
        MTZXTAL **xtal        # crystals
        int nref              # total number of reflections
        SYMGRP mtzsymm        # symmetry information

    MTZ *MtzGet(char *logname, int read_refs)
    int MtzFree(MTZ *mtz)
    int ccp4_ismnf(MTZ *mtz, float datum)


cdef class MTZColumn:

    cdef public object mtz_dataset
    cdef public object mtz_crystal
    cdef public object mtz_file
    cdef MTZ *mtz_data
    cdef MTZCOL *mtz_col
    cdef is_int

    property label:
        def __get__(self):
            return self.mtz_col.label

    property type:
        def __get__(self):
            return self.mtz_col.type

    property range:
        def __get__(self):
            if self.is_int:
                return (int(self.mtz_col.min), int(self.mtz_col.max))
            else:
                return (self.mtz_col.min, self.mtz_col.max)

    property values:
        def __get__(self):
            cdef int dims[1]
            cdef object array
            dims[0] = self.mtz_file.number_of_reflections
            array = PyArray_FromDimsAndData(1, dims, PyArray_FLOAT,
                                            <char *>self.mtz_col.ref)
            if self.is_int:
                return array.astype(N.Int)
            else:
                return copy.deepcopy(array)

    property missing_number_flag:
        def __get__(self):
            cdef int dims[1]
            cdef array_type array
            cdef unsigned char *data
            cdef int i
            dims[0] = self.mtz_file.number_of_reflections
            array = PyArray_FromDims(1, dims, PyArray_UBYTE)
            data = <unsigned char *>array.data
            for 0 <= i < dims[0]:
                data[i] = ccp4_ismnf(self.mtz_data, self.mtz_col.ref[i])
            return <object>array


    def __init__(self, object mtz_dataset):
        self.mtz_dataset = mtz_dataset
        self.mtz_crystal = self.mtz_dataset.mtz_crystal
        self.mtz_file = self.mtz_crystal.mtz_file

    cdef _setData(self, MTZ *mtz_data, MTZCOL *col):
        self.mtz_data = mtz_data
        self.mtz_col = col
        # types H, B, Y, I are integer types
        self.is_int = self.mtz_col.type[0] == 72 \
                          or self.mtz_col.type[0] == 66 \
                          or self.mtz_col.type[0] == 73 \
                          or self.mtz_col.type[0] == 89

    def __dealloc__(self):
        self.mtz_dataset = None
        self.mtz_crystal = None
        self.mtz_file = None


cdef class MTZDataset:

    cdef public object mtz_crystal
    cdef public object mtz_file
    cdef public object columns
    cdef MTZ *mtz_data
    cdef MTZSET *mtz_set

    property id:
        def __get__(self):
            return self.mtz_set.setid

    property name:
        def __get__(self):
            return self.mtz_set.dname

    property wavelength:
        def __get__(self):
            return self.mtz_set.wavelength

    def __init__(self, object mtz_crystal):
        self.mtz_crystal = mtz_crystal
        self.mtz_file = mtz_crystal.mtz_file

    cdef _setData(self, MTZ *mtz_data, MTZSET *set):
        cdef int i
        cdef MTZColumn column
        self.mtz_data = mtz_data
        self.mtz_set = set
        self.columns = {}
        for 0 <= i < self.mtz_set.ncol:
            column = MTZColumn(self)
            column._setData(self.mtz_data, self.mtz_set.col[i])
            self.columns[column.label] = column

    def __dealloc__(self):
        self.mtz_crystal = None
        self.mtz_file = None


cdef class MTZCrystal:

    cdef public object mtz_file
    cdef public object datasets
    cdef public object cell
    cdef MTZ *mtz_data
    cdef MTZXTAL *mtz_xtal

    property id:
        def __get__(self):
            return self.mtz_xtal.xtalid

    property name:
        def __get__(self):
            return self.mtz_xtal.xname

    property project_name:
        def __get__(self):
            return self.mtz_xtal.pname

    property resolution_range:
        def __get__(self):
            return (self.mtz_xtal.resmin, self.mtz_xtal.resmax)

    def __init__(self, object mtz_file):
        self.mtz_file = mtz_file

    cdef _setData(self, MTZ *mtz_data, MTZXTAL *xtal):
        cdef int i
        cdef MTZDataset dataset
        cdef int dims[1]
        cdef object array
        self.mtz_data = mtz_data
        self.mtz_xtal = xtal
        self.datasets = []
        for 0 <= i < self.mtz_xtal.nset:
            dataset = MTZDataset(self)
            dataset._setData(self.mtz_data, self.mtz_xtal.set[i])
            self.datasets.append(dataset)
        dims[0] = 6
        array = PyArray_FromDimsAndData(1, dims, PyArray_FLOAT,
                                        <char *>self.mtz_xtal.cell)
        self.cell = copy.deepcopy(array)

    def __dealloc__(self):
        self.mtz_file = None


cdef class MTZSymmetry:

    cdef public object mtz_file
    cdef public object symmetry_operations
    cdef SYMGRP *mtz_symgrp

    property space_group_number:
        def __get__(self):
            return self.mtz_symgrp.spcgrp

    property space_group_name:
        def __get__(self):
            return self.mtz_symgrp.spcgrpname

    property point_group_name:
        def __get__(self):
            return self.mtz_symgrp.pgname

    def __init__(self, object mtz_file):
        self.mtz_file = mtz_file

    cdef _setData(self, SYMGRP *mtz_symgrp):
        cdef int dims[2]
        cdef object array
        cdef object rotation
        cdef object translation
        cdef int i
        self.mtz_symgrp = mtz_symgrp
        self.symmetry_operations = []
        dims[0] = 4
        dims[1] = 4
        for 0 <= i < self.mtz_symgrp.nsym:
            array = PyArray_FromDimsAndData(2, dims, PyArray_FLOAT,
                                            <char *>self.mtz_symgrp.sym[i])
            rotation = copy.deepcopy(array[:3, :3])
            translation = copy.deepcopy(array[:3, 3])
            self.symmetry_operations.append((rotation, translation))

    def __dealloc__(self):
        self.mtz_file = None


cdef class MTZFile:

    """
    Structure factor file in MTZ (CCP4) format
    """

    cdef MTZ *mtz_data
    cdef public object crystals
    cdef public object symmetry

    property title:
        def __get__(self):
            return self.mtz_data.title

    property number_of_reflections:
        def __get__(self):
            return self.mtz_data.nref


    def __init__(self, filename, mode='r'):
        """
        @param filename: the name of the MTZ file
        @type filename: C{str}
        @param mode: the file access mode
                     (only read mode is implemented at the moment)
        @type mode: C{str}
        """
        cdef int i
        cdef MTZCrystal crystal
        cdef MTZSymmetry symmetry

        if mode != 'r':
            raise IOError("MTZ output not yet implemented")

        full_filename = os.path.expanduser(filename)
        self.mtz_data = MtzGet(full_filename, 1)
        if self.mtz_data == NULL:
            raise IOError("Couldn't open file " + filename)

        symmetry = MTZSymmetry(self)
        symmetry._setData(&self.mtz_data.mtzsymm)
        self.symmetry = <object>symmetry

        self.crystals = []
        for 0 <= i < self.mtz_data.nxtal:
            crystal = MTZCrystal(self)
            crystal._setData(self.mtz_data, self.mtz_data.xtal[i])
            self.crystals.append(crystal)

    def __dealloc__(self):
        if self.mtz_data != NULL:
            MtzFree(self.mtz_data)
