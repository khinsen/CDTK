# Reflections
#
# This file is part of the Crystallographic Data Toolkit and
# distributed under the CeCILL-C licence. See the file LICENCE
# for the full text of this licence.
#

"""
Reflections

A ReflectionSet object represents the reflections that are observed
in a crystallographic experiment. It contains reflections that lie
in a spherical shell in reciprocal space covering a specific
resolution range. Iteration over a ReflectionSet yields a minimal
set of Reflection objects from which all other reflections can be
obtained by symmetry criteria. Indexation with a tuple of Miller
indices (h, k, l) returns the corresponding Reflection object.

A Reflection object represents a single reflection in a
ReflectionSet}. Reflection objects are used as indices to
CDTK.ReflectionData.ReflectionData objects.

Data defined per reflection is handled by the module
CDTK.ReflectionData.

.. moduleauthor:: Konrad Hinsen <konrad.hinsen@cnrs-orleans.fr>

"""

from CDTK.Crystal import Crystal
from Scientific import N
import numpy as np

# Some expensive computations that are likely to be reused are cached.
import weakref
_cache = weakref.WeakKeyDictionary()

#
# A Reflection object stores Miller indices and a reference to the
# ReflectionSet to which it belongs, plus some bookkeeping information.
#
class Reflection(object):

    """
    Reflection within a ReflectionSet

    Applications obtain Reflection objects from ReflectionSet objects,
    but should not attempt to create their own ones.
    """

    def __init__(self, h, k, l, crystal, index):
        """
        :param h: the first Miller index
        :type h: int
        :param k: the second Miller index
        :type k: int
        :param l: the third Miller index
        :type l: int
        :param crystal: the crystal to which the reflection belongs
        :type crystal: CDTK.Crystal.Crystal
        :param index: the corresponding index into the list of
                      minimal reflections of the reflection set.
                      The index is None for systematic absences.
        :type index: int
        """
        self.h = h
        self.k = k
        self.l = l
        self.crystal = crystal
        self.index = index
        self.phase_factor = 1.
        self.sf_conjugate = False
        self._n_symmetry_equivalents = None

    @property
    def array(self):
        return N.array([self.h, self.k, self.l])

    @property
    def n_symmetry_equivalents(self):
        if self._n_symmetry_equivalents is None:
            self._n_symmetry_equivalents = len(self.symmetryEquivalents())
        return self._n_symmetry_equivalents
    
    def __repr__(self):
        return "Reflection(%d, %d, %d)" % (self.h, self.k, self.l)

    def __eq__(self, other):
        return self.h == other.h and self.k == other.k and self.l == other.l
        
    def __ne__(self, other):
        return self.h != other.h or self.k != other.k or self.l != other.l

    def __gt__(self, other):
        if self.h != other.h:
            return self.h > other.h
        if self.k != other.k:
            return self.k > other.k
        if self.l != other.l:
            return self.l > other.l
        return False

    def __lt__(self, other):
        return (not self.__eq__(other)) and (not self.__gt__(other))

    def __ge__(self, other):
        return self.__eq__(other) or self.__gt__(other)

    def __le__(self, other):
        return self.__eq__(other) or (not self.__gt__(other))

    def __hash__(self):
        return 400*self.h + 20*self.k + self.l

    def sVector(self, cell=None):
        """
        :param cell: the unit cell for which the scattering vector is
                     calculated. Defaults to the unit cell of the
                     reflection set.
        :type cell: CDTK.Crystal.UnitCell
        :return: the scattering vector of the reflection
        :rtype: Scientific.Geometry.Vector
        """
        if cell is None:
            cell = self.crystal.cell
        r1, r2, r3 = cell.reciprocalBasisVectors()
        return self.h*r1 + self.k*r2 + self.l*r3

    def qVector(self, cell=None):
        """
        :param cell: the unit cell for which the scattering vector is
                     calculated. Defaults to the unit cell of the
                     reflection set.
        :type cell: CDTK.Crystal.UnitCell
        :return: the scattering vector of the reflection multiplied by 2pi
        :rtype: Scientific.Geometry.Vector
        """
        return 2.*N.pi*self.sVector(cell)

    def resolution(self):
        """
        :return: the resolution of the reflection
        :rtype: float
        """
        return 1./self.sVector().length()

    def isSystematicAbsence(self):
        """
        :return: True if the reflection is systematically absent
                 due to symmetry
        :rtype: bool
        """
        hkl = N.array([self.h, self.k, self.l])
        pf = {}
        for rot, tn, td in self.crystal.space_group.transformations:
            hkl_rot = tuple(N.dot(N.transpose(rot), hkl))
            t = (tn*1.)/td
            pf[hkl_rot] = pf.get(hkl_rot, 0.) + N.exp(2j*N.pi*N.dot(hkl, t))
        for z in pf.values():
            if abs(z) > 1.e-12:
                return False
        return True

    def symmetryEquivalents(self):
        """
        :return: a set of all reflections that are equivalent due to
                 space group symmetry operations or due to the general
                 centrosymmetry of reciprocal space in the absence of
                 anomalous scattering
        :rtype: set
        """
        c = self.crystal
        sg = c.space_group
        ri = self.index
        equivalents, phases = sg.symmetryEquivalentMillerIndices(self.array)
        centric = self.isCentric(equivalents)
        unique_reflections = set()
        for (h, k, l), p in zip(equivalents, phases):
            r = Reflection(h, k, l, c, ri)
            r.phase_factor = p
            unique_reflections.add(r)
            if not centric:
                r = Reflection(-h, -k, -l, c, ri)
                r.phase_factor = p
                r.sf_conjugate = True
                unique_reflections.add(r)
        n = len(unique_reflections)
        for r in unique_reflections:
            r._n_symmetry_equivalents = n
        return unique_reflections

    def symmetryFactor(self):
        """
        :return: the symmetry factor used in the normalization of structure
                 factors or intensities. It is equal to the number of
                 space group symmetry operations that map a reflection
                 to itself.
        :rtype: int
        """
        sg = self.crystal.space_group
        equivalents = sg.symmetryEquivalentMillerIndices(self.array)[0]
        return N.int_sum(N.alltrue(N.array(equivalents) == self.array, axis=1))

    def isCentric(self, _equivalents=None):
        """
        :return: True if the reflection is centric (i.e. equivalent to
                 the reflection (-h, -k, -l) by space group symmetry
                 operations)
        :rtype: bool
        """
        sg = self.crystal.space_group
        if _equivalents is None:
            _equivalents = sg.symmetryEquivalentMillerIndices(self.array)[0]
        return N.int_sum(N.alltrue(N.array(_equivalents) == -self.array,
                                   axis=1)) > 0

#
# ReflectionSelector is a mix-in class that contains the methods for
# reflection selection shared between ReflectionSet and ReflectionSubset.
#
class ReflectionSelector(object):

    def select(self, condition):
        """
        :param condition: a function that returns True for each reflection
                          to be selected.
        :type condition: function taking a Reflection argument
                         and returning bool
        :return: the subset of the reflections that satisfy the condition
        :rtype: ReflectionSubset
        """
        return ReflectionSubset(self, [r for r in self if condition(r)])

    def randomlyAssignedSubsets(self, fractions):
        """
        Partition the reflections into several subsets of
        given (approximate) sizes by a random choice algorithm.

        :param fractions: a sequence of fractions (between 0. and 1.)
                          that specify the size that each subset should have
        :type fractions: sequence of int
        :return: subsets of approximately the requested sizes
        :rtype: sequence of ReflectionSubset
        """
        if N.sum(fractions) > 1.:
            raise ValueError("Sum of fractions > 1")
        import random
        fractions = N.add.accumulate(fractions)
        subsets = []
        for i in range(len(fractions)):
            subsets.append([])
        for r in self:
            index = N.sum(random.uniform(0., 1.) > fractions)
            if index < len(subsets):
                subsets[index].append(r)
        return [ReflectionSubset(self, s) for s in subsets]

    def resolutionShells(self, shells):
        """
        Partition the reflections into resolution shells.

        :param shells: the resolution shell specification, either a sequence of
                       s values delimiting the shells (one more value than
                       there will be shells), or an integer indicating the
                       number of shells into which the total resolution range
                       will be divided.
        :type shells:  int or sequence of float
        :return: the resolution shells, each of which has its average
                 s value stored in the attribute 's_avg'. The average is
                 None if the subset is empty.
        :rtype: sequence of ReflectionSubset
        """
        if isinstance(shells, int):
            assert shells > 0
            s_min, s_max = self.sRange()
            nshells = shells
            shells = s_min + N.arange(nshells+1)*((s_max-s_min)/nshells)
            shells[0] *= 0.99
            shells[-1] *= 1.01
        else:
            shells = N.array(shells)
            nshells = len(shells)-1
            assert nshells > 0
            assert ((shells[1:]-shells[:-1]) > 0.).all()
        reflections = [[] for i in range(nshells)]
        s_sum = N.zeros((nshells,), N.Float)
        for reflection in self:
            s = reflection.sVector().length()
            n = N.sum(s >= shells)-1
            if n >= 0 and n < nshells:
                reflections[n].append(reflection)
                s_sum[n] += s
        subsets = []
        for i in range(nshells):
            subset = ReflectionSubset(self, reflections[i])
            subset.s_min = shells[i]
            subset.s_max = shells[i+1]
            subset.s_middle = 0.5*(shells[i]+shells[i+1])
            if reflections[i]:
                subset.s_avg = s_sum[i]/len(reflections[i])
            else:
                subset.s_avg = None
            subsets.append(subset)
        return subsets

#
# A ReflectionSet represents all possible reflections for a given crystal
# within a given resolution range and manages a minimal list of reflections
# from which all others can be constructed by symmetry operations.
# Reflection amplitudes and intensities are stored separately in
# ReflectionData objects.
#
# Iteration over a ReflectionSet yields the elements of the minimal list.
# Indexation by (h, k, l) tuples gives access to arbitrary Miller indices.
#
class ReflectionSet(ReflectionSelector):

    """
    Set of reflections observed in an experiment
    """

    def __init__(self, cell, space_group,
                 max_resolution=None, min_resolution=None,
                 compact=True):
        """
        :param cell: the unit cell of the crystal
        :type cell: CDTK.Crystal.UnitCell
        :param space_group: the space group of the crystal
        :type space_group: CDTK.SpaceGroups.SpaceGroup
        :param max_resolution: the upper limit of the resolution range.
                               If not None, all reflections in the
                               specified resolution range are added to the
                               reflection set. If None, the reflection set
                               is initially empty.
        :type max_resolution: float
        :param min_resolution: the lower limit of the resolution range.
                               If None, there is no lower limit and the
                               reflection (0, 0, 0) is included in the set.
        :type min_resolution: float
        :param compact: if True, only the reflections in an asymmetric unit
                        are stored explicitly. Retrieving a reflection for
                        a given set of Miller indices can be slow, because
                        the symmetry operations of the space group must be
                        tried one by one. If False, symmetry-related reflections
                        are stored explicitly. This takes more memory, but
                        access by Miller indices is fast. There is no
                        performance difference for iteration over a
                        reflection set, which is always an iteration over an
                        asymmetric unit.
        :type compact: Bool
        """
        self.cell = cell
        self.space_group = space_group
        self.compact = compact
        self.crystal = Crystal(cell, space_group)
        self.minimal_reflection_list = []
        self.reflection_map = {}
        self.systematic_absences = set()
        self.s_min = None
        self.s_max = None
        self.completeness_range = (None, None)
        if max_resolution is not None:
            self.fillResolutionSphere(max_resolution, min_resolution)
        self.frozen = None

    def freeze(self):
        if self.frozen is None:
            self.frozen = FrozenReflectionSet(self)
        return self.frozen

    def addReflection(self, h, k, l):
        """
        Adds the reflection (h, k, l) and all reflections that are equivalent
        by symmetry to the reflection set.

        :param h: the first Miller index
        :type h: int
        :param k: the second Miller index
        :type k: int
        :param l: the third Miller index
        :type l: int
        """
        self.frozen = None
        hkl = Reflection(h, k, l, self.crystal,
                         len(self.minimal_reflection_list))
        if self.compact:

            equivalents = list(hkl.symmetryEquivalents())
            equivalents.sort()
            hkl = equivalents[-1]
            key = (hkl.h, hkl.k, hkl.l)
            if self.reflection_map.has_key(key):
                return
            self.reflection_map[key] = hkl
            hkl.phase_factor = 1.
            hkl.sf_conjugate = False
            if hkl.isSystematicAbsence():
                self.systematic_absences.add(hkl)
                hkl.index = None
            else:
                self.minimal_reflection_list.append(hkl)

        else:

            if self.reflection_map.has_key((hkl.h, hkl.k, hkl.l)):
                return
            equivalents = list(hkl.symmetryEquivalents())
            equivalents.sort()
            hkl = equivalents[-1]
            for r in equivalents:
                r.phase_factor /= hkl.phase_factor
                self.reflection_map[(r.h, r.k, r.l)] = r
            if hkl.sf_conjugate:
                for r in equivalents:
                    r.sf_conjugate = not r.sf_conjugate
                    r.phase_factor = N.conjugate(r.phase_factor)
            if hkl.isSystematicAbsence():
                self.systematic_absences.add(hkl)
                for r in equivalents:
                    r.index = None
            else:
                self.minimal_reflection_list.append(hkl)

        s = hkl.sVector().length()
        if self.s_min is None:
            self.s_min = s
        else:
            self.s_min = min(s, self.s_min)
        if self.s_max is None:
            self.s_max = s
        else:
            self.s_max = max(s, self.s_max)

    def fillResolutionSphere(self, max_resolution=None, min_resolution=None):
        """
        Add all reflections in the spherical shell in reciprocal space
        specified by the resolution range to the reflection set.

        :param max_resolution: the upper limit of the resolution range.
                               If both max_resolution and min_resolution
                               are None, use the resolution range of
                               the currenly present reflections.
        :type max_resolution: float
        :param min_resolution: the lower limit of the resolution range.
                               If None, there is no lower limit and the
                               reflection (0, 0, 0) is included in the set.
        :type min_resolution: float
        """
        if max_resolution is None:
            max_resolution = 1./self.s_max
            min_resolution = 1./self.s_min
        max_inv_sq_resolution = 1.00001/max_resolution**2
        if min_resolution is None:
            min_inv_sq_resolution = 0.
        else:
            min_inv_sq_resolution = (1.-0.00001)/min_resolution**2
        if self.isComplete(N.sqrt(min_inv_sq_resolution),
                           N.sqrt(max_inv_sq_resolution)):
            return
        r1, r2, r3 = self.cell.reciprocalBasisVectors()
        h_max = int(N.sqrt(max_inv_sq_resolution/(r1*r1)))
        k_max = int(N.sqrt(max_inv_sq_resolution/(r2*r2)))
        l_max = int(N.sqrt(max_inv_sq_resolution/(r3*r3)))
        for h in range(-h_max, h_max+1):
            s1 = h*r1
            for k in range(-k_max, k_max+1):
                s2 = k*r2
                for l in range(-l_max, l_max+1):
                    s3 = l*r3
                    s = s1+s2+s3
                    if min_inv_sq_resolution <= s*s \
                           <= max_inv_sq_resolution:
                        self.addReflection(h, k, l)
        self.minimal_reflection_list.sort()
        self.completeness_range = (N.sqrt(min_inv_sq_resolution),
                                   N.sqrt(max_inv_sq_resolution))

    def isComplete(self, s_min=None, s_max=None):
        """
        :param s_min: the lower limit of the range of scattering vector
                      lengths to check. Defaults to the lowest-resolution
                      reflection.
        :type s_min: float
        :param s_max: the upper limit of the range of scattering vector
                      lengths to check. Defaults to the highest-resolution
                      reflection.
        :type s_max: float
        :return: True if the ReflectionSet is known to contain all
                 reflections in the given range, False if the
                 completeness cannot be guaranteed
        :rtype: bool
        """
        c_min, c_max = self.completeness_range
        if c_min is None or c_max is None:
            return False
        if s_min is None: s_min = self.s_min
        if s_max is None: s_max = self.s_max
        return s_min >= c_min and s_max <= c_max

    def sRange(self):
        """
        :return: a tuple (s_min, s_max) containing the range of
                 scattering vector lengths in the reflection set
        :rtype: tuple of float
        """
        if not self.reflection_map:
            raise ValueError("Empty ReflectionSet")
        return self.s_min, self.s_max

    def resolutionRange(self):
        """
        :return: a tuple (r_min, r_max) containing the range of
                 resolutions in the reflection set
        :rtype: tuple of float
        :raise ZeroDivisionError: if the upper resolution limit is infinite
        """
        if not self.reflection_map:
            raise ValueError("Empty ReflectionSet")
        return 1./self.s_max, 1./self.s_min

    def maxHKL(self):
        """
        :return: the highest absolute values of the Miller indices in the
                 reflection set
        :rtype: tuple
        """
        if self.compact:
            max_hkl = N.zeros((3,), N.Int)
            for r in self.reflection_map.values():
                hkl = [(re.h, re.k, re.l) for re in r.symmetryEquivalents()]
                max_hkl = N.maximum(max_hkl, N.maximum.reduce(N.array(hkl)))
            return tuple(max_hkl)
        else:
            return tuple(N.maximum.reduce(N.array(self.reflection_map.keys())))

    def __iter__(self):
        """
        :return: a generator yielding the elements of the minimal
                 reflection set
        :rtype: generator
        """
        for r in self.minimal_reflection_list:
            yield r

    def __len__(self):
        """
        :return: the number of reflections in the reflection set
        :rtype: int
        """
        return len(self.minimal_reflection_list)

    def __getitem__(self, item):
        """
        :param item: a set of Miller indices (h, k, l)
        :type item: tuple of int
        :return: the corresponding reflection object
        :rtype: CDTK.Reflections.Reflection
        :raise KeyError: if the requested reflection is not part of the set
        """
        try:
            return self.reflection_map[item]
        except KeyError:
            if self.compact:
                hkl = N.array(item)
                for hkl in self.crystal.space_group.\
                                symmetryEquivalentMillerIndices(hkl)[0]:
                    for sign in [1., -1.]:
                        try:
                            r = self.reflection_map[tuple(sign*hkl)]
                            for re in r.symmetryEquivalents():
                                if (re.h, re.k, re.l) == item:
                                    return re
                        except KeyError:
                            pass
            raise KeyError(item)

    def hasReflection(self, h, k, l):
        """
        :param h: the first Miller index
        :type h: int
        :param k: the second Miller index
        :type k: int
        :param l: the third Miller index
        :type l: int
        :return: True if there is a corresponding reflection
        :rtype: boolean
        """
        item = (h, k, l)
        if self.reflection_map.has_key(item):
            return True
        if self.compact:
            hkl = N.array(item)
            for hkl in self.crystal.space_group.\
                          symmetryEquivalentMillerIndices(hkl)[0]:
                for sign in [1., -1.]:
                    try:
                        r = self.reflection_map[tuple(sign*hkl)]
                        for re in r.symmetryEquivalents():
                            if (re.h, re.k, re.l) == item:
                                return True
                    except KeyError:
                        pass
        return False

    def getReflection(self, h, k, l):
        """
        Return the reflection object corresponding to the given Miller indices.
        If the reflection is not yet part of the reflection set, it is added
        first and all its symmetry equivalents are added as well.

        :param h: the first Miller index
        :type h: int
        :param k: the second Miller index
        :type k: int
        :param l: the third Miller index
        :type l: int
        :return: the corresponding reflection object
        :rtype: CDTK.Reflections.Reflection
        """
        try:
            return self[(h, k, l)]
        except KeyError:
            self.addReflection(h, k, l)
            return self[(h, k, l)]

    def totalReflectionCount(self):
        """
        :return: the total number of reflections (explicitly stored
                 reflections plus symmetry equivalents)
        :rtype: int
        """
        if self.compact:
            return sum([r.n_symmetry_equivalents
                        for r in self.reflection_map.values()])
        else:
            return len(self.reflection_map)
        
    def intersection(self, other):
        """
        Return a new ReflectionSet that is the intersection of self
        with other.

        :param other: a second reflectionset
        :type other: CDTK.Reflections.ReflectionSet
        """
        new =  ReflectionSet(self.cell, self.space_group,
                             None, None, self.compact)
        for r in self:
            if other.hasReflection(r.h, r.k, r.l):
                new.addReflection(r.h, r.k, r.l)
        return new

    # When pickling, store only a minimal information set in order to
    # reduce pickle file size and CPU time. The lost information is
    # restored after unpickling.

    def __getstate__(self):
        reflections = [(r.h, r.k, r.l, r.index)
                       for r in self.minimal_reflection_list]
        absences = [(r.h, r.k, r.l)
                    for r in self.systematic_absences]
        return (tuple(self.cell.basisVectors()),
                self.space_group.number,
                self.s_min, self.s_max, self.compact,
                self.completeness_range,
                N.array(reflections), N.array(absences))

    def __setstate__(self, state):
        from CDTK.SpaceGroups import space_groups
        from CDTK.Crystal import UnitCell
        cell_basis, space_group_number, \
                    self.s_min, self.s_max, self.compact, \
                    self.completeness_range, \
                    reflections, absences = state
        self.cell = UnitCell(*cell_basis)
        self.space_group = space_groups[space_group_number]
        self.crystal = Crystal(self.cell, self.space_group)
        self.minimal_reflection_list = []
        self.reflection_map = {}
        self.systematic_absences = set()
        for h, k, l, index in reflections:
            r = Reflection(h, k, l, self.crystal, index)
            for re in r.symmetryEquivalents():
                hkl = (re.h, re.k, re.l)
                if not self.compact:
                    self.reflection_map[hkl] = re
                if hkl == (h, k, l):
                    self.minimal_reflection_list.append(re)
                    if self.compact:
                        self.reflection_map[hkl] = re
                        break
        for h, k, l in absences:
            r = Reflection(h, k, l, self.crystal, None)
            for re in r.symmetryEquivalents():
                hkl = (re.h, re.k, re.l)
                if not self.compact:
                    self.reflection_map[hkl] = re
                if hkl == (h, k, l):
                    self.systematic_absences.add(re)
                    if self.compact:
                        self.reflection_map[(h, k, l)] = re
                        break

    # Use the same minimal state for equality checks
    def __eq__(self, other):
        for v1, v2 in zip(self.__getstate__(), other.__getstate__()):
            if type(v1) is N.array_type:
                if (v1 != v2).any():
                    return False
            else:
                if v1 != v2:
                    return False
        return True

    def storeHDF5(self, parent_group, path):
        """
        :param parent_group: HDF5 group in which the dataset is created
        :type parent_group: h5py.Group
        :param path: the path from parent_group to the dataset
                     for the ReflectionSet.
        :type : str
        :return: the HDF5 dataset and the index permutation corresponding
                 to the sort order used in the file
        :rtype: (h5py.Dataset, np.ndarray)
        """
        import h5py
        import numpy as np
        rs = np.array([(r.h, r.k, r.l, r.index)
                       for r in self.minimal_reflection_list],
                      dtype=[('h', np.int32), ('k', np.int32), ('l', np.int32),
                             ('index', np.int32)])
        si = N.argsort(rs["index"])
        # indices should run from 0 to n-1
        assert rs["index"][si[-1]] == len(rs)-1
        rs = N.take(rs[['h', 'k', 'l']], si)

        # Sort Miller indices and create the inverse index vector
        # for the rearrangement which is needed for converting
        # ReflectionData arrays.
        si = np.argsort(rs, order=('h', 'k', 'l'))
        rs = np.take(rs, si)
        sinv = np.argsort(si)
        assert (np.take(si, sinv) == np.arange(len(si))).all()

        dataset = parent_group.require_dataset(path, shape=rs.shape,
                                               dtype=rs.dtype, exact=True)
        dataset[...] = rs
        a1, a2, a3 = self.cell.basisVectors()
        dataset.attrs['a'] = a1.length()
        dataset.attrs['b'] = a2.length()
        dataset.attrs['c'] = a3.length()
        dataset.attrs['alpha'] = a2.angle(a3)
        dataset.attrs['beta'] = a1.angle(a3)
        dataset.attrs['gamma'] = a1.angle(a2)
        dataset.attrs['space_group'] = self.space_group.number
        dataset.attrs['DATA_MODEL'] = 'CDTK'
        dataset.attrs['DATA_MODEL_MAJOR_VERSION'] = 0
        dataset.attrs['DATA_MODEL_MINOR_VERSION'] = 1
        dataset.attrs['DATA_CLASS'] = 'ReflectionSet'
        return dataset, sinv

    @classmethod
    def fromHDF5(cls, store, dataset):
        if dataset.attrs['DATA_MODEL'] != 'CDTK' \
           or dataset.attrs['DATA_MODEL_MAJOR_VERSION'] > 0 \
           or dataset.attrs['DATA_MODEL_MINOR_VERSION'] > 1 \
           or dataset.attrs['DATA_CLASS'] != 'ReflectionSet':
            raise ValueError("HDF5 dataset does not contain a ReflectionSet")
        from CDTK.SpaceGroups import space_groups
        from CDTK.Crystal import UnitCell
        space_group = space_groups[dataset.attrs['space_group']]
        cell = UnitCell(dataset.attrs['a'],
                        dataset.attrs['b'],
                        dataset.attrs['c'],
                        dataset.attrs['alpha'],
                        dataset.attrs['beta'],
                        dataset.attrs['gamma'])
        self = cls(cell, space_group)
        for h, k, l in dataset:
            self.addReflection(h, k, l)
        return self

#
# A FrozenReflectionSet is a ReflectionSet that cannot be modified.
# Its reflections are stored in an array for compactness and rapid access.
#
class FrozenReflectionSet(ReflectionSet):

    def __init__(self, reflection_set):
        self.cell = reflection_set.cell
        self.space_group = reflection_set.space_group
        self.crystal = reflection_set.crystal
        self.s_min = reflection_set.s_min
        self.s_max = reflection_set.s_max
        self.completeness_range = reflection_set.completeness_range
        self.compact = True

        rs = reflection_set.minimal_reflection_list
        self.reflections = np.zeros((len(rs),), dtype=self._dtype)
        for r in rs:
            self.reflections[r.index] = (r.h, r.k, r.l)

        rs = reflection_set.systematic_absences
        self.systematic_absences = np.zeros((len(rs),), dtype=self._dtype)
        for index, r in enumerate(rs):
            self.systematic_absences[index] = (r.h, r.k, r.l)

    _dtype = np.dtype([('h', np.int32),
                       ('k', np.int32),
                       ('l', np.int32)])

    def freeze(self):
        return self

    def addReflection(self, h, k, l):
        raise ValueError("Can't modify FrozenReflectionSet")

    def fillResolutionSphere(self, max_resolution=None, min_resolution=None):
        raise ValueError("Can't modify FrozenReflectionSet")

    def sRange(self):
        if len(self.reflections) == 0:
            raise ValueError("Empty ReflectionSet")
        return self.s_min, self.s_max

    def resolutionRange(self):
        if len(self.reflections) == 0:
            raise ValueError("Empty ReflectionSet")
        return 1./self.s_max, 1./self.s_min

    def maxHKL(self):
        if len(self.systematic_absences) == 0:
            max_hkl = np.zeros((3,), np.int)
        else:
            max_hkl = np.array([self.systematic_absences['h'].max(),
                                self.systematic_absences['k'].max(),
                                self.systematic_absences['l'].max()])
        for r in self:
            hkl = [(re.h, re.k, re.l) for re in r.symmetryEquivalents()]
            max_hkl = N.maximum(max_hkl, N.maximum.reduce(N.array(hkl)))
        return tuple(max_hkl)

    def __iter__(self):
        for i, (h, k, l) in enumerate(self.reflections):
            yield Reflection(h, k, l, self.crystal, i)

    def __len__(self):
        return len(self.reflections)

    def __getitem__(self, item):
        hkl = np.array(item, dtype=self._dtype)
        test = self.reflections == hkl
        if test.any():
            index = np.repeat(np.arange(len(self.reflections)), test)[0]
            return Reflection(hkl['h'], hkl['k'], hkl['l'], self.crystal, index)
        if (self.systematic_absences == hkl).any():
            return Reflection(hkl['h'], hkl['k'], hkl['l'], self.crystal, None)

        for hkl in self.crystal.space_group.\
                        symmetryEquivalentMillerIndices(np.array(item))[0]:
            for sign in [1., -1.]:
                hkl_sym = np.array(tuple(sign*hkl), dtype=self._dtype)
                test = self.reflections == hkl_sym
                if test.any():
                    index = np.repeat(np.arange(len(self.reflections)),
                                      test)[0]
                    r = Reflection(hkl_sym['h'], hkl_sym['k'], hkl_sym['l'],
                                   self.crystal, index)
                    for re in r.symmetryEquivalents():
                        if (re.h, re.k, re.l) == item:
                            return re
                if (self.systematic_absences == hkl_sym).any():
                    return Reflection(hkl_sym['h'], hkl_sym['k'], hkl_sym['l'],
                                      self.crystal, None)

        raise KeyError(item)

    def hasReflection(self, h, k, l):
        try:
            _ = self[(h, k, l)]
            return True
        except KeyError:
            hkl = np.array((h, k, l), dtype=self._dtype)
            return (self.systematic_absences == hkl).any()

    def totalReflectionCount(self):
        return sum([r.n_symmetry_equivalents for r in self],
                   len(self.systematic_absences))
        
    def __getstate__(self):
        return (tuple(self.cell.basisVectors()),
                self.space_group.number,
                self.s_min, self.s_max,
                self.completeness_range,
                self.reflections,
                self.systematic_absences)

    def __setstate__(self, state):
        from CDTK.SpaceGroups import space_groups
        from CDTK.Crystal import UnitCell
        cell_basis, space_group_number, \
                    self.s_min, self.s_max, \
                    self.completeness_range, \
                    self.reflections, \
                    self.systematic_absences = state
        self.cell = UnitCell(*cell_basis)
        self.space_group = space_groups[space_group_number]
        self.crystal = Crystal(self.cell, self.space_group)

    def sVectorArray(self, cell=None):
        """
        :param cell: a unit cell, which defaults to the unit cell for
                     which the reflection set is defined.
        :type cell: CDTK.Crystal.UnitCell
        :return: an array containing the s vectors for all reflections
        :rtype: N.array
        """
        global _cache
        if cell is None:
            cell = self.cell
        cached_data = _cache.get(self, {})
        try:
            return cached_data[('sVectorArray', id(cell))]
        except KeyError:
            pass

        r1, r2, r3 = cell.reciprocalBasisVectors()
        sv = self.reflections['h'][:, np.newaxis]*r1.array + \
             self.reflections['k'][:, np.newaxis]*r2.array + \
             self.reflections['l'][:, np.newaxis]*r3.array

        cached_data[('sVectorArray', id(cell))] = sv
        _cache[self] = cached_data
        return sv

    def sVectorArrayAndPhasesForASU(self, cell=None):
        """
        Calculates the transformed s vectors and phases that are used
        for calculating the structure factor from the atoms in the
        asymmetric unit.

        :param cell: a unit cell, which defaults to the unit cell for
            which the reflection set is defined.
        :type cell: CDTK.Crystal.UnitCell
        :returns: a tuple (s, p), where s is an array containing the s vectors
            for all reflections and space group operations and p is an
            array with the corresponding phases
        :rtype: Scientific.N.array_type
        """
        global _cache
        if cell is None:
            cell = self.cell
        cached_data = _cache.get(self, {})
        try:
            return cached_data[('sVectorArrayAndPhasesForASU', id(cell))]
        except KeyError:
            pass

        sg = self.space_group
        ntrans = len(sg)
        nr = len(self.reflections)
        sv = N.zeros((ntrans, nr, 3), N.Float)
        p = N.zeros((ntrans, nr), N.Complex)
        twopii = 2.j*N.pi
        r1, r2, r3 = cell.reciprocalBasisVectors()
        r1 = r1.array
        r2 = r2.array
        r3 = r3.array
        for index, r in enumerate(self.reflections):
            hkl_list = sg.symmetryEquivalentMillerIndices(r.view('3i4'))[0]
            rh, rk, rl = r
            for i in range(ntrans):
                h, k, l = hkl_list[i]
                sv[i, index] = h*r1+k*r2+l*r3
                tr_num, tr_den = sg.transformations[i][1:]
                st = rh*float(tr_num[0])/float(tr_den[0]) \
                     + rk*float(tr_num[1])/float(tr_den[1]) \
                     + rl*float(tr_num[2])/float(tr_den[2])
                p[i, index] = N.exp(twopii*st)

        cached_data[('sVectorArrayAndPhasesForASU', id(cell))] = (sv, p)
        _cache[self] = cached_data
        return sv, p

    def symmetryAndCentricityArrays(self):
        """
        :return: an array containing the symmetry factors and centricity flags
                 for all reflections
        :rtype: N.array
        """
        sm = N.zeros((len(self.reflections), 2), N.Int)
        for r in self:
            sm[r.index, 0] = r.symmetryFactor()
            sm[r.index, 1] = r.isCentric()
        return sm

    def storeHDF5(self, parent_group, path):
        import h5py
        import numpy as np

        # Sort Miller indices and create the inverse index vector
        # for the rearrangement which is needed for converting
        # ReflectionData arrays.
        rs = self.reflections
        si = np.argsort(rs, order=('h', 'k', 'l'))
        rs = np.take(rs, si)
        sinv = np.argsort(si)
        assert (np.take(si, sinv) == np.arange(len(si))).all()

        dataset = parent_group.require_dataset(path, shape=rs.shape,
                                               dtype=rs.dtype, exact=True)
        dataset[...] = rs
        a1, a2, a3 = self.cell.basisVectors()
        dataset.attrs['a'] = a1.length()
        dataset.attrs['b'] = a2.length()
        dataset.attrs['c'] = a3.length()
        dataset.attrs['alpha'] = a2.angle(a3)
        dataset.attrs['beta'] = a1.angle(a3)
        dataset.attrs['gamma'] = a1.angle(a2)
        dataset.attrs['space_group'] = self.space_group.number
        dataset.attrs['systematic_absences'] = self.systematic_absences
        dataset.attrs['DATA_MODEL'] = 'CDTK'
        dataset.attrs['DATA_MODEL_MAJOR_VERSION'] = 0
        dataset.attrs['DATA_MODEL_MINOR_VERSION'] = 1
        dataset.attrs['DATA_CLASS'] = 'ReflectionSet'
        return dataset, sinv

    @classmethod
    def fromHDF5(cls, store, dataset):
        if dataset.attrs['DATA_MODEL'] != 'CDTK' \
           or dataset.attrs['DATA_MODEL_MAJOR_VERSION'] > 0 \
           or dataset.attrs['DATA_MODEL_MINOR_VERSION'] > 1 \
           or dataset.attrs['DATA_CLASS'] != 'ReflectionSet':
            raise ValueError("HDF5 dataset does not contain a ReflectionSet")
        from CDTK.SpaceGroups import space_groups
        from CDTK.Crystal import UnitCell
        space_group = space_groups[dataset.attrs['space_group']]
        cell = UnitCell(dataset.attrs['a'],
                        dataset.attrs['b'],
                        dataset.attrs['c'],
                        dataset.attrs['alpha'],
                        dataset.attrs['beta'],
                        dataset.attrs['gamma'])
        self = cls(ReflectionSet(cell, space_group))
        self.reflections = dataset[...]
        self.systematic_absences = dataset.attrs['systematic_absences']
        return self

#
# A ReflectionSubset object is an iterator over a subset of a
# ReflectionSet.
#
class ReflectionSubset(ReflectionSelector):

    """
    Iterator over a subset of reflections
    """

    def __init__(self, reflection_set, reflection_list):
        """
        :param reflection_set: complete reflection set
        :type reflection_set: ReflectionSet
        :param reflection_list: the reflections to be included in the subset
        :type reflection_list: list
        """
        self.reflection_set = reflection_set
        self.reflection_list = reflection_list

    def __len__(self):
        return len(self.reflection_list)

    def __iter__(self):
        """
        :return: a generator yielding the reflections of the subset
        :rtype: generator
        """
        for r in self.reflection_list:
            yield r

    def sRange(self):
        """
        :return: a tuple (s_min, s_max) containing the range of
                 scattering vector lengths in the reflection set
        :rtype: tuple of float
        """
        s = [r.sVector().length() for r in self]
        return min(s), max(s)

    def resolutionRange(self):
        """
        :return: a tuple (r_min, r_max) containing the range of
                 resolutions in the reflection set
        :rtype: tuple of float
        :raise ZeroDivisionError: if the upper resolution limit is infinite
        """
        s_min, s_max = self.sRange()
        return 1./s_max, 1./s_min

    def hasReflection(self, h, k, l):
        """
        :param h: the first Miller index
        :type h: int
        :param k: the second Miller index
        :type k: int
        :param l: the third Miller index
        :type l: int
        :return: True if there is a corresponding reflection
        :rtype: boolean
        """
        try:
            r = self.reflection_set[(h, k, l)]
        except KeyError:
            return False
        return r in self.reflection_list
