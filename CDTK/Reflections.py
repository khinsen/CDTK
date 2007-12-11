# Reflections and data defined on reflections
#
# This file is part of the Crystallographic Data Toolkit and
# distributed under the CeCILL-C licence. See the file LICENCE
# for the full text of this licence.
#
# Written by Konrad Hinsen.
#

"""
Reflections and reflection data

A L{ReflectionSet} object represents the reflections that are observed
in a crystallographic experiment. It contains reflections that lie
in a spherical shell in reciprocal space covering a specific
resolution range. Iteration over a L{ReflectionSet} yields a minimal
set of L{Reflection} objects from which all other reflections can be
obtained by symmetry criteria. Indexation with a tuple of Miller
indices (h, k, l) returns the corresponding L{Reflection} object.

A L{Reflection} object represents a single reflection in a
L{ReflectionSet}. L{Reflection} objects are used as indices to
L{ReflectionData} objects.

The subclasses of L{ReflectionData} represent the various kind of data
that can be defined for each reflection in a L{ReflectionSet}. Only
the values for the minimal reflection set is stored explicitly, values
for other reflections are reconstructed by applying symmetry operations.

There are two subclasses of L{ReflectionData} for representing experimental
data:

  - L{ExperimentalIntensities}
  - L{ExperimentalAmplitudes}

There are three subclasses of L{ReflectionData} for representing data
calculated from a model:

  - L{StructureFactor}
  - L{ModelIntensities}
  - L{ModelAmplitudes}

The classes for experimental data store standard deviations in addition to the
values themselves and can handle missing data points. The classes for
model data provide routines for the calculation of structure factors
from an atomic model.
"""

from Scientific import N, LA
from Scientific.Geometry import Tensor
from CDTK import Units
import copy

#
# A Reflection object stores Miller indices and a reference to the
# ReflectionSet to which it belongs, plus some bookkeeping information.
#
class Reflection(object):

    """
    Reflection within a L{ReflectionSet}

    Applications obtain Reflection objects from ReflectionSet objects,
    but should not attempt to create their own ones.
    """

    def __init__(self, h, k, l, reflection_set, index):
        """
        @param h: the first Miller index
        @type h: C{int}
        @param k: the second Miller index
        @type k: C{int}
        @param l: the third Miller index
        @type l: C{int}
        @param reflection_set: the reflection set to which
                               the reflection belongs
        @type reflection_set: L{ReflectionSet}
        @param index: the corresponding index into the list of
                      minimal reflections of the reflection set.
                      The index is C{None} for systematic absences.
        @type index: C{int}
        """
        self.h = h
        self.k = k
        self.l = l
        self.reflection_set = reflection_set
        self.index = index
        self.phase_factor = 1.
        self.sf_conjugate = False
        self.n_symmetry_equivalents = None

    def _getarray(self):
        return N.array([self.h, self.k, self.l])
    array = property(_getarray)

    def __repr__(self):
        return "Reflection(%d, %d, %d)" % (self.h, self.k, self.l)

    def __eq__(self, other):
        return self.h == other.h and self.k == other.k and self.l == other.l
        
    def __ne__(self, other):
        return self.h != other.h or self.k != other.k or self.l != other.l

    def __gt__(self, other):
        pref1 = 4*(self.h >= 0) + 2*(self.k >= 0) + (self.l >= 0)
        pref2 = 4*(other.h >= 0) + 2*(other.k >= 0) + (other.l >= 0)
        return pref1 > pref2 \
               or self.h > other.h or self.k > other.k or self.l > other.l

    def __lt__(self, other):
        pref1 = 4*(self.h >= 0) + 2*(self.k >= 0) + (self.l >= 0)
        pref2 = 4*(other.h >= 0) + 2*(other.k >= 0) + (other.l >= 0)
        return pref1 < pref2 \
               or self.h < other.h or self.k < other.k or self.l < other.l

    def __ge__(self, other):
        return self.__eq__(other) or self.__gt__(other)

    def __le__(self, other):
        return self.__eq__(other) or self.__lt__(other)

    def __hash__(self):
        return 400*self.h + 20*self.k + self.l

    def sVector(self, cell=None):
        """
        @param cell: the unit cell for which the scattering vector is
                     calculated. Defaults to the unit cell of the
                     reflection set.
        @type cell: L{CDTK.Crystal.UnitCell}
        @return: the scattering vector of the reflection
        @rtype: C{Scientific.Geometry.Vector}
        """
        if cell is None:
            cell = self.reflection_set.cell
        r1, r2, r3 = cell.reciprocalBasisVectors()
        return self.h*r1 + self.k*r2 + self.l*r3

    def qVector(self, cell=None):
        """
        @param cell: the unit cell for which the scattering vector is
                     calculated. Defaults to the unit cell of the
                     reflection set.
        @type cell: L{CDTK.Crystal.UnitCell}
        @return: the scattering vector of the reflection multiplied by 2pi
        @rtype: C{Scientific.Geometry.Vector}
        """
        return 2.*N.pi*self.sVector(cell)

    def resolution(self):
        """
        @return: the resolution of the reflection
        @rtype: C{float}
        """
        return 1./self.sVector().length()

    def isSystematicAbsence(self):
        """
        @return: C{True} if the reflection is systematically absent
                 due to symmetry
        @rtype: C{bool}
        """
        hkl = N.array([self.h, self.k, self.l])
        pf = {}
        for rot, tn, td in self.reflection_set.space_group.transformations:
            hkl_rot = tuple(N.dot(N.transpose(rot), hkl))
            t = (tn*1.)/td
            pf[hkl_rot] = pf.get(hkl_rot, 0.) + N.exp(2j*N.pi*N.dot(hkl, t))
        for z in pf.values():
            if abs(z) > 1.e-12:
                return False
        return True

    def symmetryEquivalents(self):
        """
        @return: a set of all reflections that are equivalent due to
                 space group symmetry operations or due to the general
                 centrosymmetry of reciprocal space in the absence of
                 anomalous scattering
        @rtype: C{set}
        """
        rs = self.reflection_set
        sg = rs.space_group
        ri = self.index
        unique_reflections = set()
        equivalents, phases = sg.symmetryEquivalentMillerIndices(self.array)
        for (h, k, l), p in zip(equivalents, phases):
            r = Reflection(h, k, l, rs, ri)
            r.phase_factor = p
            unique_reflections.add(r)
            r = Reflection(-h, -k, -l, rs, ri)
            r.phase_factor = p
            r.sf_conjugate = True
            unique_reflections.add(r)
        n = len(unique_reflections)
        for r in unique_reflections:
            r.n_symmetry_equivalents = n
        return unique_reflections

    def symmetryFactor(self):
        """
        @return: the symmetry factor used in the normalization of structure
                 factors or intensities. It is equal to the number of
                 space group symmetry operations that map a reflection
                 to itself.
        @rtype: C{int}
        """
        sg = self.reflection_set.space_group
        equivalents = sg.symmetryEquivalentMillerIndices(self.array)[0]
        return N.int_sum(N.alltrue(N.array(equivalents) == self.array, axis=1))

    def isCentric(self):
        """
        @return: C{True} if the reflection is centric (i.e. equivalent to
                 the reflection (-h, -k, -l) by space group symmetry
                 operations)
        @rtype: C{bool}
        """
        sg = self.reflection_set.space_group
        equivalents = sg.symmetryEquivalentMillerIndices(self.array)[0]
        return N.int_sum(N.alltrue(N.array(equivalents) == -self.array,
                                   axis=1)) > 0
        
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
class ReflectionSet(object):

    """
    Set of reflections observed in an experiment
    """

    def __init__(self, cell, space_group,
                 max_resolution=None, min_resolution=None):
        """
        @param cell: the unit cell of the crystal
        @type cell: L{CDTK.Crystal.UnitCell}
        @param space_group: the space group of the crystal
        @type space_group: L{CDTK.SpaceGroups.SpaceGroup}
        @param max_resolution: the upper limit of the resolution range.
                               If not None, all reflections in the
                               specified resolution range are added to the
                               reflection set. If None, the reflection set
                               is initially empty.
        @type max_resolution: C{float}
        @param min_resolution: the lower limit of the resolution range.
                               If None, there is no lower limit and the
                               reflection (0, 0, 0) is included in the set.
        @type min_resolution: C{float}
        """
        self.cell = cell
        self.space_group = space_group
        self.minimal_reflection_list = []
        self.reflection_map = {}
        self.systematic_absences = set()
        self.s_min = None
        self.s_max = None
        if max_resolution is not None:
            self.fillResolutionSphere(max_resolution, min_resolution)

    def addReflection(self, h, k, l):
        """
        Adds the reflection (h, k, l) and all reflections that are equivalent
        by symmetry to the reflection set.

        @param h: the first Miller index
        @type h: C{int}
        @param k: the second Miller index
        @type k: C{int}
        @param l: the third Miller index
        @type l: C{int}
        """
        hkl = Reflection(h, k, l, self,
                         len(self.minimal_reflection_list))
        if self.reflection_map.has_key((hkl.h, hkl.k, hkl.l)):
            return
        equivalents = list(hkl.symmetryEquivalents())
        equivalents.sort()
        for r in equivalents:
            self.reflection_map[(r.h, r.k, r.l)] = r
        hkl = equivalents[-1]
        for r in equivalents:
            r.phase_factor /= hkl.phase_factor
        if hkl.sf_conjugate:
            for r in equivalents:
                r.sf_conjugate = not r.sf_conjugate
                r.phase_factor = N.conjugate(r.phase_factor)
        if hkl.isSystematicAbsence():
            for r in equivalents:
                r.index = None
                self.systematic_absences.add(r)
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

    def fillResolutionSphere(self, max_resolution, min_resolution=None):
        """
        Add all reflections in the spherical shell in reciprocal space
        specified by the resolution range to the reflection set.

        @param max_resolution: the upper limit of the resolution range.
        @type max_resolution: C{float}
        @param min_resolution: the lower limit of the resolution range.
                               If None, there is no lower limit and the
                               reflection (0, 0, 0) is included in the set.
        @type min_resolution: C{float}
        """
        max_inv_sq_resolution = 1.00001/max_resolution**2
        if min_resolution is None:
            min_inv_sq_resolution = 0.
        else:
            min_inv_sq_resolution = (1.-0.00001)/min_resolution**2
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

    def sRange(self):
        """
        @return: a tuple (s_min, s_max) containing the range of
                 scattering vector lengths in the reflection set
        @rtype: C{tuple} of C{float}
        """
        if not self.reflection_map:
            raise ValueError("Empty ReflectionSet")
        return self.s_min, self.s_max

    def resolutionRange(self):
        """
        @return: a tuple (r_min, r_max) containing the range of
                 resolutions in the reflection set
        @rtype: C{tuple} of C{float}
        @raise ZeroDivisionError: if the upper resolution limit is infinite
        """
        if not self.reflection_map:
            raise ValueError("Empty ReflectionSet")
        return 1./self.s_max, 1./self.s_min

    def maxHKL(self):
        """
        @return: the highest absolute values of the Miller indices in the
                 reflection set
        @rtype: C{tuple}
        """
        return tuple(N.maximum.reduce(N.array(self.reflection_map.keys())))
        
    def __iter__(self):
        """
        @return: a generator yielding the elements of the minimal
                 reflection set
        @rtype: generator
        """
        for r in self.minimal_reflection_list:
            yield r

    def __len__(self):
        """
        @return: the number of reflections in the reflection set
        @rtype: C{int}
        """
        return len(self.reflection_map)

    def __getitem__(self, item):
        """
        @param item: a set of Miller indices (h, k, l)
        @type item: C{tuple} of C{int}
        @return: the corresponding reflection object
        @rtype: L{CDTK.Reflections.Reflection}
        @raise KeyError: if the requested reflection is not part of the set
        """
        return self.reflection_map[item]

    def getReflection(self, hkl):
        """
        Return the reflection object corresponding to the given Miller indices.
        If the reflection is not yet part of the reflection set, it is added
        first and all its symmetry equivalents are added as well.

        @param hkl: a set of Miller indices (h, k, l)
        @type hkl: C{tuple} of C{int}
        @return: the corresponding reflection object
        @rtype: L{CDTK.Reflections.Reflection}
        """
        try:
            return self.reflection_map[hkl]
        except KeyError:
            h, k, l = hkl
            self.addReflection(h, k, l)
            return self.reflection_map[hkl]

    # When pickling, store only a minimal information set in order to
    # reduce pickle file size and CPU time. The lost information is
    # restored after unpickling.

    def __getstate__(self):
        reflections = [(r.h, r.k, r.l, r.index)
                       for r in self.minimal_reflection_list]
        absences = [(r.h, r.k, r.l)
                    for r in self.systematic_absences]
        return (tuple(self.cell.basis),
                self.space_group.number,
                self.s_min, self.s_max,
                N.array(reflections), N.array(absences))

    def __setstate__(self, state):
        from CDTK.SpaceGroups import space_groups
        from CDTK.Crystal import UnitCell
        cell_basis, space_group_number, \
                    self.s_min, self.s_max, \
                    reflections, absences = state
        self.cell = UnitCell(*cell_basis)
        self.space_group = space_groups[space_group_number]
        self.minimal_reflection_list = []
        self.reflection_map = {}
        self.systematic_absences = set()
        for h, k, l, index in reflections:
            r = Reflection(h, k, l, self, index)
            for re in r.symmetryEquivalents():
                hkl = (re.h, re.k, re.l)
                if hkl == (h, k, l):
                    r = re
                self.reflection_map[hkl] = re
            self.minimal_reflection_list.append(r)
        for h, k, l in absences:
            r = Reflection(h, k, l, self, None)
            self.systematic_absences.add(r)
            self.reflection_map[(h, k, l)] = r

#
# ReflectionData and its subclasses describe data defined per reflection,
# such as structure factors or intensities.
#
# There are "experimental" and "model" classes, the former storing
# variances and missing-value information in addition to the basic data.
# There is also a distinction between "amplitude" and "intensity" classes,
# because the operations on them are different. In the "model" and
# "amplitude" category, there are two classes: StructureFactor stores
# amplitude and phase information, whereas ModelAmplitudes stores only
# the amplitudes.
#
class ReflectionData(object):

    """
    Base class for data defined per reflection

    The subclasses for use by applications are
    L{ExperimentalIntensities}, L{ExperimentalAmplitudes},
    L{StructureFactor}, L{ModelIntensities}, and
    L{ModelAmplitudes}.
    """

    def __init__(self, reflection_set):
        self.reflection_set = reflection_set
        self.number_of_reflections = \
                 len(self.reflection_set.minimal_reflection_list)

    def __getitem__(self, reflection):
        """
        @param reflection: a reflection
        @type reflection: L{Reflection}
        @return: the data value for the given reflection
        @rtype: C{float} or C{complex}
        """
        index = reflection.index
        if index is None: # systematic absence
            return self.absent_value
        return self.array[index]

    def __setitem__(self, reflection, value):
        """
        @param reflection: a reflection
        @type reflection: L{Reflection}
        @param value: the new data value for the given reflection
        """
        index = reflection.index
        if index is None: # systematic absence
            raise ValueError("Cannot set value: "
                             "reflection is absent due to symmetry")
        self.array[index] = value

    def __iter__(self):
        """
        @return: a generator yielding (reflection, value) tuples where
                 reflection iterates over the elements of the minimal
                 reflection set
        @rtype: generator
        """
        for r in self.reflection_set:
            yield (r, self[r])

    def __len__(self):
        """
        @return: the number of data values, equal to the number of reflections
                 in the minimal reflection set
        @rtype: C{int}
        """
        return self.number_of_reflections

    def __add__(self, other):
        """
        @param other: reflection data of the same type
        @type other: C{type(self)}
        @return: the reflection-by-reflection sum of the two data sets
        @rtype: C{type(self)}
        """
        assert self.__class__ is other.__class__
        assert self.reflection_set is other.reflection_set
        result = self.__class__(self.reflection_set)
        self.__add_op__(other, result)
        return result

    def __iadd__(self, other):
        """
        @param other: reflection data of the same type
        @type other: C{type(self)}
        """
        assert self.__class__ is other.__class__
        assert self.reflection_set is other.reflection_set
        self.__iadd_op__(other)
        return self

    def __sub__(self, other):
        """
        @param other: reflection data of the same type
        @type other: C{type(self)}
        @return: the reflection-by-reflection difference of the two data sets
        @rtype: C{type(self)}
        """
        assert self.__class__ is other.__class__
        assert self.reflection_set is other.reflection_set
        result = self.__class__(self.reflection_set)
        self.__sub_op__(other, result)
        return result

    def __isub__(self, other):
        """
        @param other: reflection data of the same type
        @type other: C{type(self)}
        """
        assert self.__class__ is other.__class__
        assert self.reflection_set is other.reflection_set
        self.__isub_op__(other)
        return self

    def __add_op__(self, other, result):
        result.array[:] = self.array[:]+other.array[:]
        
    def __iadd_op__(self, other):
        self.array += other.array
        
    def __sub_op__(self, other, result):
        result.array[:] = self.array[:]-other.array[:]

    def __isub_op__(self, other):
        self.array -= other.array

    def __mul__(self, other):
        """
        @param other: a scalar or reflection data of the same type
        @type other: C{float} or C{type(self)}
        @return: the reflection-by-reflection product of the two data sets
        @rtype: C{type(self)}
        """
        result = self.__class__(self.reflection_set)
        if isinstance(other, ReflectionData):
            self.__mul_op__(other, result)
        else:
            self.__mul_scalar_op__(other, result)
        return result

    __rmul__ = __mul__

    def __mul_op__(self, other, result):
        other_array = other.array
        if len(other_array.shape) == 2:
            other_array = other_array[:, 0]
        result.array[:] = self.array*other_array

    def __mul_scalar_op__(self, other, result):
        result.array[:] = self.array*other

    def __imul__(self, other):
        """
        @param other: a scalar or reflection data of the same type
        @type other: C{float} or C{type(self)}
        """
        if isinstance(other, ReflectionData):
            self.__imul_op__(other)
        else:
            self.__imul_scalar_op__(other)
        return self

    def __imul_op__(self, other):
        other_array = other.array
        if len(other_array.shape) == 2:
            other_array = other_array[:, 0]
        self.array *= other_array

    def __imul_scalar_op__(self, other):
        self.array *= other

    def __div__(self, other):
        """
        @param other: a scalar or reflection data of the same type
        @type other: C{float} or C{type(self)}
        @return: the reflection-by-reflection quotient of the two data sets
        @rtype: C{type(self)}
        """
        result = self.__class__(self.reflection_set)
        if isinstance(other, ReflectionData):
            self.__div_op__(other, result)
        else:
            self.__div_scalar_op__(other, result)
        return result

    def __div_op__(self, other, result):
        other_array = other.array
        if len(other_array.shape) == 2:
            other_array = other_array[:, 0]
        result.array[:] = self.array/other_array

    def __div_scalar_op__(self, other, result):
        result.array[:] = self.array/other

    def __idiv__(self, other):
        """
        @param other: a scalar or reflection data of the same type
        @type other: C{float} or C{type(self)}
        """
        if isinstance(other, ReflectionData):
            self.__idiv_op__(other)
        else:
            self.__idiv_scalar_op__(other)
        return self

    def __idiv_op__(self, other):
        other_array = other.array
        if len(other_array.shape) == 2:
            other_array = other_array[:, 0]
        self.array /= other_array

    def __idiv_scalar_op__(self, other):
        self.array /= other

    def _debyeWallerFactor(self, adp_or_scalar):
        twopisq = -2.*N.pi**2
        sv = N.zeros((self.number_of_reflections, 3), N.Float)
        for r in self.reflection_set:
            sv[r.index] = r.sVector(self.reflection_set.cell).array
        if isinstance(adp_or_scalar, float):
            dwf = N.exp(twopisq*adp_or_scalar*N.sum(sv*sv, axis=-1))
        else:
            dwf = N.exp(twopisq*N.sum(N.dot(sv, adp_or_scalar.array)*sv,
                                      axis=-1))
        return dwf

    def writeToVMDScript(self, filename):
        """
        Writes a VMD script containing the values as map data in
        reciprocal space.

        @param filename: the name of the VMD script
        @type filename: C{str}
        """
        hmax, kmax, lmax = self.reflection_set.maxHKL()
        array = N.zeros((2*hmax+1, 2*kmax+1, 2*lmax+1), N.Float)
        for r_asu in self.reflection_set:
            for r in r_asu.symmetryEquivalents():
                value = self[r]
                if value is not None:
                    array[r.h+hmax, r.k+kmax, r.l+lmax] = N.absolute(value)
        factor = 1./N.maximum.reduce(N.ravel(array))
        r1, r2, r3 = self.reflection_set.cell.reciprocalBasisVectors()

        vmd_script = file(filename, 'w')
        vmd_script.write('mol new\n')
        vmd_script.write('mol volume top "%s" \\\n' % self.__class__.__name__)
        vmd_script.write('  {%f %f %f} \\\n' %
                         tuple(-(hmax*r1+kmax*r2+lmax*r3)*Units.Ang))
        vmd_script.write('  {%f %f %f} \\\n' % tuple((2*hmax+1)*r1*Units.Ang))
        vmd_script.write('  {%f %f %f} \\\n' % tuple((2*kmax+1)*r2*Units.Ang))
        vmd_script.write('  {%f %f %f} \\\n' % tuple((2*lmax+1)*r3*Units.Ang))
        vmd_script.write('  %d %d %d \\\n' % array.shape)
        vmd_script.write('  {')
        for iz in range(array.shape[2]):
            for iy in range(array.shape[1]):
                for ix in range(array.shape[0]):
                    vmd_script.write(str(factor*array[ix, iy, iz]) + ' ')
        vmd_script.write('}\n')
        vmd_script.write('mol addrep top\nmol modstyle 0 top isosurface\n')
        vmd_script.close()


class ExperimentalReflectionData(ReflectionData):

    """
    Reflection data of experimental origin

    Experimental data has standard deviations on the values and
    can have missing values.
    """

    def __init__(self, reflection_set):
        ReflectionData.__init__(self, reflection_set)
        self.data_available = N.zeros((self.number_of_reflections,), N.Int0)

    def __getitem__(self, reflection):
        index = reflection.index
        if index is None: # systematic absence
            return self.absent_value
        if self.data_available[index]:
            return self.array[index, 0]
        else:
            return None

    def __iter__(self):
        for r in self.reflection_set:
            value = self[r]
            if value is not None:
                yield (r, value)

    def __setitem__(self, reflection, value_sigma):
        """
        @param reflection: a reflection
        @type reflection: L{Reflection}
        @param value_sigma: the new data value and standard deviations
                            for the given reflection
        @type value_sigma: sequence of length 2
        """
        index = reflection.index
        if index is None: # systematic absence
            raise ValueError("Cannot set value: "
                             "reflection is absent due to symmetry")
        self.data_available[index] = True
        self.array[index, 0] = value_sigma[0]
        self.array[index, 1] = value_sigma[1]

    def __len__(self):
        """
        @return: the number of data values, equal to the number of reflections
                 in the minimal reflection set minus the number of missing
                 values
        @rtype: C{int}
        """
        return N.int_sum(self.data_available)

    def setFromArrays(self, h, k, l, value, sigma, missing=None):
        """
        Sets data values and standard deviations for many reflections
        from information stored in multiple arrays, usually read from
        a file.

        @param h: the array of the first Miller indices
        @type h: C{Scientific.N.array_type}
        @param k: the array of the second Miller indices
        @type k: C{Scientific.N.array_type}
        @param l: the array of the third Miller indices
        @type l: C{Scientific.N.array_type}
        @param value: the array of data values for each reflection
        @type value: C{Scientific.N.array_type}
        @param sigma: the array of standard deviation values for each reflection
        @type sigma: C{Scientific.N.array_type}
        @param missing: the array of missing value flags, or None if there
                        are no missing values
        @type missing: C{Scientific.N.array_type} or C{NoneType}
        @raise AssertionError: if the input arrays have different lengths
        @raise KeyError: if at least one (h, k, l) set corresponds to a
                         reflection that is not in the reflection set
        """
        n = len(h)
        assert len(k) == n
        assert len(l) == n
        assert len(value) == n
        assert len(sigma) == n
        if missing is not None:
            assert len(missing) == n
        for i in range(n):
            if missing is None or missing[i] == 0:
                r = self.reflection_set[(h[i], k[i], l[i])]
                self.array[r.index, 0] = value[i]
                self.array[r.index, 1] = sigma[i]
                self.data_available[r.index] = True

    def __add_op__(self, other, result):
        result.data_available[:] = self.data_available*other.data_available
        result.array[:] = self.array[:]+other.array[:]
        
    def __iadd_op__(self, other):
        self.data_available *= other.data_available
        self.array += other.array
        
    def __sub_op__(self, other, result):
        result.data_available[:] = self.data_available*other.data_available
        result.array[:, 0] = self.array[:, 0]-other.array[:, 0]
        result.array[:, 1] = self.array[:, 1]+other.array[:, 1]

    def __isub_op__(self, other):
        self.data_available *= other.data_available
        self.array[:, 0] -= other.array[:, 0]
        self.array[:, 1] += other.array[:, 1]

    def __mul_op__(self, other, result):
        other_array = other.array
        if len(other_array.shape) == 2:
            other_array = other_array[:, 0]
        result.data_available[:] = self.data_available
        if hasattr(other, 'data_available'):
            result.data_available *= other.data_available
        result.array[:] = self.array*other_array[:, N.NewAxis]

    def __imul_op__(self, other):
        other_array = other.array
        if len(other_array.shape) == 2:
            other_array = other_array[:, 0]
        if hasattr(other, 'data_available'):
            self.data_available *= other.data_available
        self.array *= other_array[:, N.NewAxis]

    def __div_op__(self, other, result):
        other_array = other.array
        if len(other_array.shape) == 2:
            other_array = other_array[:, 0]
        result.data_available[:] = self.data_available
        if hasattr(other, 'data_available'):
            result.data_available *= other.data_available
            result.array[:] = self.array / \
                              (other_array+(1-other.data_available)) \
                              [:, N.NewAxis]
        else:
            result.array[:] = self.array/other_array[:, N.NewAxis]

    def __idiv_op__(self, other):
        other_array = other.array
        if len(other_array.shape) == 2:
            other_array = other_array[:, 0]
        if hasattr(other, 'data_available'):
            self.data_available *= other.data_available
            self.array /= (other_array+(1-other.data_available))[:, N.NewAxis]
        else:
            self.array /= other_array[:, N.NewAxis]


class AmplitudeData(object):

    """
    Mix-in class for L{ReflectionData} subclasses representing amplitude
    values
    """

    def rFactor(self, other):
        """
        @param other: reflection data containing amplitudes or structure factors
        @return: the R factor between the two data sets
        @rtype: C{float}
        """
        assert isinstance(other, AmplitudeData)
        sum_self = 0.
        sum_diff = 0.
        for r in self.reflection_set:
            f_self = self[r]
            f_other = other[r]
            if f_self is None or f_other is None:
                continue
            f_self = abs(f_self)
            f_other = abs(f_other)
            sum_self += f_self
            sum_diff += abs(f_self-f_other)
        return sum_diff/sum_self

    def rFactorWithScale(self, other):
        """
        @param other: reflection data containing amplitudes or structure factors
        @return: a tuple (R, scale) where scale is the scale factor that must
                 be applied to other to minimize the R factor and R is the
                 R factor obtained with this scale factor
        @rtype: C{float}
        """
        assert isinstance(other, AmplitudeData)
        sum_self = 0.
        sum_other = 0.
        for r in self.reflection_set:
            f_self = self[r]
            f_other = other[r]
            if f_self is None or f_other is None:
                continue
            f_self = abs(f_self)
            f_other = abs(f_other)
            sum_self += f_self*f_other
            sum_other += f_other*f_other
        scale = sum_self/sum_other
        sum_self = 0.
        sum_diff = 0.
        for r in self.reflection_set:
            f_self = self[r]
            f_other = other[r]
            if f_self is None or f_other is None:
                continue
            f_self = abs(f_self)
            f_other = abs(f_other)
            sum_self += f_self
            sum_diff += abs(f_self-scale*f_other)
        return sum_diff/sum_self, scale

    def rFactorByResolution(self, other, nbins = 50):
        """
        @param other: reflection data containing amplitudes or structure factors
        @param nbins: the number of intervals into which the resolution range
                      is divided before calculating the R factor for each
                      interval
        @type nbins: C{int}
        @return: the R factor between the two data sets for each interval
        @rtype: C{Scientific.Functions.InterpolatingFunction}
        """
        from Scientific.Functions.Interpolation import InterpolatingFunction
        assert isinstance(other, AmplitudeData)
        s_min, s_max = self.reflection_set.sRange()
        bin_width = (s_max-s_min)/nbins
        sum_self = N.zeros((nbins,), N.Float)
        sum_diff = N.zeros((nbins,), N.Float)
        for r in self.reflection_set:
            f_self = self[r]
            f_other = other[r]
            if f_self is None or f_other is None:
                continue
            f_self = abs(f_self)
            f_other = abs(f_other)
            s = r.sVector().length()
            bin = min(nbins-1, int((s-s_min)/bin_width))
            sum_self[bin] += f_self
            sum_diff[bin] += abs(f_self-f_other)
        s = s_min + bin_width*(N.arange(nbins)+0.5)
        return InterpolatingFunction((s,), sum_diff/(sum_self+(sum_self == 0.)))

    def intensities(self):
        """
        @return: a reflection data set containing the reflection intensities
        @rtype: L{ReflectionData} and L{IntensityData}
        """
        return ModelIntensities(self.reflection_set,
                                (self.array[:]*N.conjugate(self.array[:])).real)

    def applyDebyeWallerFactor(self, adp_or_scalar):
        """
        @param adp_or_scalar: a symmetric ADP tensor or a scalar
                              position fluctuation value
        @type adp_or_scalar: C{Scientific.Geometry.Tensor} or C{float}
        @return: a new reflection data set containing the amplitudes
                 multiplied by the Debye-Waller factor defined by
                 adp_or_scalar
        @rtype: C{type(self)}
        """
        dwf = self._debyeWallerFactor(adp_or_scalar)
        return self.__class__(self.reflection_set, self.array*dwf)

    def scaleTo(self, other, iterations=0):
        """
        Performs a non-linear fit of self multiplied by a Debye-Waller
        factor and a overall global factor to other. The fit parameters
        are the global factor and the ADP tensor of the Debye-Waller
        factor.

        The initial values for the fit parameters are obtained by taking
        the logarithm of the two amplitude sets, which makes the fit
        problem linear. These initial values can be improved in a specified
        number of Gauss-Newton iterations.

        @param other: another amplitude data set
        @param other: L{AmplitudeData}
        @param iterations: the number of Gauss-Newton iterations
        @type iterations: C{int}
        @return: a tuple (scaled_amplitudes, k, u), where scaled_amplitudes
                 is the fitted amplitude data set, k is the global factor,
                 and u is the ADP tensor of the fitted Debye-Waller factor
        @rtype: C{tuple}
        """
        assert isinstance(other, AmplitudeData)
        twopisq = -2.*N.pi**2
        mat = []
        rhs = []
        for r in self.reflection_set:
            a1 = other[r]
            a2 = self[r]
            if a1 is None or a2 is None:
                continue
            rhs.append(N.log(abs(a1)/abs(a2)))
            sx, sy, sz = r.sVector()
            mat.append([1., twopisq*sx*sx, twopisq*sy*sy, twopisq*sz*sz,
                        2.*twopisq*sy*sz, 2.*twopisq*sx*sz, 2.*twopisq*sx*sy])
        fit = LA.linear_least_squares(N.array(mat), N.array(rhs))
        log_k, uxx, uyy, uzz, uyz, uxz, uxy = fit[0]
        k = N.exp(log_k)
        u = Tensor(N.array([[uxx, uxy, uxz],
                            [uxy, uyy, uyz],
                            [uxz, uyz, uzz]]))
        ev, rot = u.diagonalization()
        if N.sum(ev) >= 0.:
            u = Tensor(N.dot(N.transpose(rot.array)*N.maximum(ev, 0.),
                             rot.array))
        else:
            u = Tensor(N.dot(N.transpose(rot.array)*N.minimum(ev, 0.),
                             rot.array))

        while iterations > 0:
            iterations -= 1
            # Gauss-Newton iteration
            mat = []
            rhs = []
            for r in self.reflection_set:
                a1 = other[r]
                a2 = self[r]
                if a1 is None or a2 is None:
                    continue
                a1 = abs(a1)
                a2 = abs(a2)
                s = r.sVector()
                sx, sy, sz = s
                dwf = N.exp(twopisq*(s*(u*s)))
                rhs.append(a1-k*dwf*a2)
                f = k*twopisq*a2*dwf
                mat.append([a2*dwf, f*sx*sx, f*sy*sy, f*sz*sz,
                            2.*f*sy*sz, 2.*f*sx*sz, 2.*f*sx*sy])
            fit = LA.linear_least_squares(N.array(mat), N.array(rhs))
            dk, duxx, duyy, duzz, duyz, duxz, duxy = fit[0]
            k += dk
            u += Tensor(N.array([[duxx, duxy, duxz],
                                 [duxy, duyy, duyz],
                                 [duxz, duyz, duzz]]))
            ev, rot = u.diagonalization()
            if N.sum(ev) >= 0.:
                u = Tensor(N.dot(N.transpose(rot.array)*N.maximum(ev, 0.),
                                 rot.array))
            else:
                u = Tensor(N.dot(N.transpose(rot.array)*N.minimum(ev, 0.),
                                 rot.array))

        return k*self.applyDebyeWallerFactor(u), k, u


class IntensityData(object):

    """
    Mix-in class for L{ReflectionData} subclasses representing intensity
    values
    """

    def isotropicAverage(self, nbins = 50):
        """
        @param nbins: the number of intervals into which the resolution range
                      is divided before averaging the intensities within each
                      interval
        @type nbins: C{int}
        @return: the averaged intensities for each resolution interval
        @rtype: C{Scientific.Functions.InterpolatingFunction}
        """
        from Scientific.Functions.Interpolation import InterpolatingFunction
        s_min, s_max = self.reflection_set.sRange()
        bin_width = (s_max-s_min)/nbins
        reflection_count = N.zeros((nbins,), N.Int)
        intensity_sum = N.zeros((nbins,), N.Float)
        for reflection, intensity in self:
            s = reflection.sVector().length()
            bin = min(nbins-1, int((s-s_min)/bin_width))
            n = reflection.n_symmetry_equivalents
            reflection_count[bin] += n
            intensity_sum[bin] += n*intensity
        intensity_average = intensity_sum / \
                            (reflection_count + (reflection_count==0))
        s = s_min + bin_width*(N.arange(nbins)+0.5)
        return InterpolatingFunction((s,), intensity_average)

    def wilsonPlot(self, nbins=50):
        """
        Returns the logarithm of the isotropic average as a function of the
        square of the scattering vector length. This is a Wilson plot only
        if applied to normalized intensity data.

        @param nbins: the number of intervals into which the resolution range
                      is divided before averaging the intensities within each
                      interval
        @type nbins: C{int}
        @return: the logarithm of the averaged intensities
        @rtype: C{Scientific.Functions.InterpolatingFunction}
        """
        from Scientific.Functions.Interpolation import InterpolatingFunction
        av = self.isotropicAverage(nbins)
        s = av.axes[0]
        intensity_average = av.values
        return InterpolatingFunction((s*s,), N.log(intensity_average))

    def amplitudes(self):
        """
        @return: the structure factor amplitudes corresponding to the
                 reflection intensities
        @rtype: L{ReflectionData} and L{AmplitudeData}
        """
        return ModelAmplitudes(self.reflection_set, N.sqrt(self.array))

    def applyDebyeWallerFactor(self, adp_or_scalar):
        """
        @param adp_or_scalar: a symmetric ADP tensor or a scalar
                              position fluctuation value
        @type adp_or_scalar: C{Scientific.Geometry.Tensor} or C{float}
        @return: a new reflection data set containing the intensities
                 multiplied by the Debye-Waller factor defined by
                 adp_or_scalar
        @rtype: C{type(self)}
        """
        dwf = self._debyeWallerFactor(adp_or_scalar)
        return self.__class__(self.reflection_set, self.array*(dwf**2))

    def scaleTo(self, other, iterations=0):
        """
        Performs a non-linear fit of self multiplied by a Debye-Waller
        factor and a overall global factor to other. The fit parameters
        are the global factor and the ADP tensor of the Debye-Waller
        factor.

        The initial values for the fit parameters are obtained by taking
        the logarithm of the two amplitude sets, which makes the fit
        problem linear. These initial values can be improved in a specified
        number of Gauss-Newton iterations.

        @param other: another amplitude data set
        @param other: L{AmplitudeData}
        @param iterations: the number of Gauss-Newton iterations
        @type iterations: C{int}
        @return: a tuple (scaled_amplitudes, k, u), where scaled_amplitudes
                 is the fitted amplitude data set, k is the global factor,
                 and u is the ADP tensor of the fitted Debye-Waller factor
        @rtype: C{tuple}
        """
        a, k, u = self.amplitudes().scaleTo(other.amplitudes(), iterations)
        return a.intensities(), k, u

    def normalize(self, atom_count):
        """
        Calculate normalized intensities, defined as the real intensities
        divided by the intensities for a system containing the same
        atoms but with all positions distributed uniformly over the
        unit cell.

        @param atom_count: a dictionary mapping chemical element symbols
                           to the number of atoms of that element in the
                           system
        @type atom_count: C{dict}
        @return: the normalized intensities
        @rtype: L{IntensityData}
        """
        i_random = ModelIntensities(self.reflection_set)
        i_random.calculateFromUniformAtomDistribution(atom_count)
        i_random, k, u = i_random.scaleTo(self)
        return self/i_random


class StructureFactor(ReflectionData, AmplitudeData):

    """
    Structure factor values (amplitudes and phases) calculated from a model

    Structure factors can be calculated from 1) an MMTK universe,
    2) a sequence of data describing the atoms in the unit cell,
    3) a sequence of data describing the atoms in the asymmetric unit, or
    4) from an electron density map by Fourier transform.
    """

    def __init__(self, reflection_set, data=None):
        """
        @param reflection_set: the reflection set for which the structure
                               factor is defined
        @type reflection_set: L{ReflectionSet}
        @param data: an optional array storing the data values, of the
                     right shape and type (complex). If C{None}, a new array
                     containing zero values is created.
        @type data: C{Scientific.N.array_type}
        """
        ReflectionData.__init__(self, reflection_set)
        if data is None:
            self.array = N.zeros((self.number_of_reflections,), N.Complex)
        else:
            assert(data.shape == (self.number_of_reflections,))
            self.array = data
        self.absent_value = 0j

    def __getitem__(self, reflection):
        index = reflection.index
        if index is None: # systematic absence
            return self.absent_value
        if reflection.sf_conjugate:
            return N.conjugate(self.array[index]*reflection.phase_factor)
        else:
            return self.array[index]*reflection.phase_factor

    def __setitem__(self, reflection, value):
        index = reflection.index
        if index is None: # systematic absence
            raise ValueError("Cannot set value: "
                             "reflection is absent due to symmetry")
        if reflection.sf_conjugate:
            self.array[index] = N.conjugate(value)/reflection.phase_factor
        else:
            self.array[index] = value/reflection.phase_factor

    def setFromArrays(self, h, k, l, amplitude, phase):
        """
        Sets data values and standard deviations for many reflections
        from information stored in multiple arrays, usually read from
        a file.

        @param h: the array of the first Miller indices
        @type h: C{Scientific.N.array_type}
        @param k: the array of the second Miller indices
        @type k: C{Scientific.N.array_type}
        @param l: the array of the third Miller indices
        @type l: C{Scientific.N.array_type}
        @param amplitude: the array of amplitude values for each reflection
        @type amplitude: C{Scientific.N.array_type}
        @param phase: the array of phase values for each reflection
        @type phase: C{Scientific.N.array_type}
        @raise AssertionError: if the input arrays have different lengths
        @raise KeyError: if at least one (h, k, l) set corresponds to a
                         reflection that is not in the reflection set
        """
        n = len(h)
        assert len(k) == n
        assert len(l) == n
        assert len(amplitude) == n
        assert len(phase) == n
        for i in range(n):
            r = self.reflection_set[(h[i], k[i], l[i])]
            f = amplitude[i]*N.exp(1j*phase[i])
            if r.sf_conjugate:
                self.array[r.index] = N.conjugate(f)/r.phase_factor
            else:
                self.array[r.index] = f/r.phase_factor

    def calculateFromUniverse(self, universe, adps=None, conf=None):
        """
        Calculate the structure factor from a periodic MMTK universe
        representing a unit cell.

        @param universe: the MMTK universe
        @type universe: C{MMTK.Periodic3DUniverse}
        @param adps: the anisotropic displacement parameters for all atoms
        @type adps: C{MMTK.ParticleTensor}
        @param conf: a configuration for the universe, defaults to the
                     current configuration
        @type conf: C{MMTK.Configuration}
        """
        from AtomicScatteringFactors import atomic_scattering_factors
        from CDTK_sfcalc import sfTerm
        if conf is None:
            conf = universe.configuration()

        cell = universe.__class__()
        cell.setCellParameters(conf.cell_parameters)
        sv = N.zeros((self.number_of_reflections, 3), N.Float)
        for r in self.reflection_set:
            sv[r.index] = r.sVector(cell).array
        ssq = N.sum(sv*sv, axis=-1)

        f_atom = {}
        for atom in universe.atomList():
            key = atom.symbol
            if f_atom.has_key(key):
                continue
            a, b = atomic_scattering_factors[key.lower()]
            f_atom[key] = N.sum(a[:, N.NewAxis] \
                                * N.exp(-b[:, N.NewAxis]*ssq[N.NewAxis, :]))

        self.array[:] = 0j
        for atom in universe.atomList():
            if adps is None:
                sfTerm(self.array, sv, f_atom[atom.symbol],
                       conf[atom].array, sv, 0., 0)
            else:
                sfTerm(self.array, sv, f_atom[atom.symbol],
                       conf[atom].array, adps[atom].array, 0., 2)

    def calculateFromUnitCellAtoms(self, atom_iterator, cell=None):
        """
        @param atom_iterator: an iterator or sequence that yields
                              for each atom in the unit cell a
                              tuple of (chemical element,
                              position vector, position fluctuation,
                              occupancy). The position fluctuation
                              can be a symmetric tensor (ADP tensor)
                              or a scalar (implicitly multiplied by
                              the unit tensor).
        @type atom_iterator: iterable
        @param cell: a unit cell, which defaults to the unit cell for
                     which the map object is defined. If a different
                     unit cell is given, the map is calculated for
                     this cell in fractional coordinates and converted
                     to Cartesian coordinates using the unit cell of
                     the map object. This is meaningful only if the two
                     unit cells are very similar, such as for unit cells
                     corresponding to different steps in a constant-pressure
                     Molecular Dynamics simulation.
        @type cell: L{CDTK.Crystal.UnitCell}
        """
        from AtomicScatteringFactors import atomic_scattering_factors
        from CDTK_sfcalc import sfTerm
        sv = N.zeros((self.number_of_reflections, 3), N.Float)
        for r in self.reflection_set:
            sv[r.index] = r.sVector(cell).array
        ssq = N.sum(sv*sv, axis=-1)
        self.array[:] = 0j
        twopii = 2.j*N.pi
        twopisq = -2.*N.pi**2
        for element, position, adp, occupancy in atom_iterator:
            a, b = atomic_scattering_factors[element.lower()]
            f_atom = occupancy * \
                     N.sum(a[:, N.NewAxis]
                           * N.exp(-b[:, N.NewAxis]*ssq[N.NewAxis, :]))
            if adp is None:
                sfTerm(self.array, sv, f_atom, position.array, sv, 0., 0)
            elif isinstance(adp, float):
                sfTerm(self.array, sv, f_atom, position.array, sv, adp, 1)
            else:
                sfTerm(self.array, sv, f_atom, position.array, adp.array, 0., 2)

    def calculateFromAsymmetricUnitAtoms(self, atom_iterator, cell=None):
        """
        @param atom_iterator: an iterator or sequence that yields
                              for each atom in the asymmetric unit a
                              tuple of (chemical element,
                              position vector, position fluctuation,
                              occupancy). The position fluctuation
                              can be a symmetric tensor (ADP tensor)
                              or a scalar (implicitly multiplied by
                              the unit tensor).
        @type atom_iterator: iterable
        @param cell: a unit cell, which defaults to the unit cell for
                     which the map object is defined. If a different
                     unit cell is given, the map is calculated for
                     this cell in fractional coordinates and converted
                     to Cartesian coordinates using the unit cell of
                     the map object. This is meaningful only if the two
                     unit cells are very similar, such as for unit cells
                     corresponding to different steps in a constant-pressure
                     Molecular Dynamics simulation.
        @type cell: L{CDTK.Crystal.UnitCell}
        """
        from AtomicScatteringFactors import atomic_scattering_factors
        if cell is None:
            cell = self.reflection_set.cell
        twopii = 2.j*N.pi
        twopisq = -2.*N.pi**2
        sg = self.reflection_set.space_group
        ntrans = len(sg)
        sv = N.zeros((ntrans, self.number_of_reflections, 3), N.Float)
        p = N.zeros((ntrans, self.number_of_reflections), N.Complex)
        r1, r2, r3 = cell.reciprocalBasisVectors()
        for r in self.reflection_set:
            hkl_list = sg.symmetryEquivalentMillerIndices(r.array)[0]
            for i in range(ntrans):
                h, k, l = hkl_list[i]
                sv[i, r.index] = (h*r1+k*r2+l*r3).array
                tr_num, tr_den = sg.transformations[i][1:]
                st = r.h*float(tr_num[0])/float(tr_den[0]) \
                     + r.k*float(tr_num[1])/float(tr_den[1]) \
                     + r.l*float(tr_num[2])/float(tr_den[2])
                p[i, r.index] = N.exp(twopii*st)
        ssq = N.sum(sv[0]*sv[0], axis=-1)
        self.array[:] = 0j
        for element, position, adp, occupancy in atom_iterator:
            a, b = atomic_scattering_factors[element.lower()]
            f_atom = N.sum(a[:, N.NewAxis]
                           * N.exp(-b[:, N.NewAxis]*ssq[N.NewAxis, :]))
            if adp is None:
                dwf = 1.
            elif isinstance(adp, float):
                dwf = N.exp(twopisq*adp*ssq)
            else:
                dwf = None
            for i in range(ntrans):
                if isinstance(adp, Tensor):
                    dwf = N.exp(twopisq*N.sum(N.dot(sv[i], adp.array)*sv[i],
                                              axis=-1))
                self.array += occupancy*p[i]*f_atom*dwf \
                                * N.exp(twopii*(N.dot(sv[i], position.array)))


    def calculateFromElectronDensityMap(self, density_map):
        """
        @param density_map: an electronic density map
        @type density_map: L{CDTK.Maps.ElectronDensityMap}
        """
        from CDTK_sf_fft import map_to_sf
        m_fc = self.reflection_set.cell.fractionalToCartesianMatrix()
        det_m_fc = LA.determinant(m_fc)
        map_to_sf(density_map.array, self, det_m_fc)
        

class ModelAmplitudes(ReflectionData,
                      AmplitudeData):

    """
    Structure factor amplitudes (without phases) calculated from a model

    ModelAmplitudes are the result of a conversion of L{ModelIntensities}
    to amplitudes.
    """

    def __init__(self, reflection_set, data=None):
        """
        @param reflection_set: the reflection set for which the amplitudes
                               are defined
        @type reflection_set: L{ReflectionSet}
        @param data: an optional array storing the data values, of the
                     right shape and type (float). If C{None}, a new array
                     containing zero values is created.
        @type data: C{Scientific.N.array_type}
        """
        ReflectionData.__init__(self, reflection_set)
        if data is None:
            self.array = N.zeros((self.number_of_reflections,), N.Float)
        else:
            assert(data.shape == (self.number_of_reflections,))
            self.array = data
        self.absent_value = 0.


class ModelIntensities(ReflectionData,
                       IntensityData):

    """
    Reflection intensities calculated from a model

    ModelIntensities are usually created by converting L{StructureFactor}
    objects to intensities.
    """

    def __init__(self, reflection_set, data=None):
        """
        @param reflection_set: the reflection set for which the intensities
                               are defined
        @type reflection_set: L{ReflectionSet}
        @param data: an optional array storing the data values, of the
                     right shape and type (float). If C{None}, a new array
                     containing zero values is created.
        @type data: C{Scientific.N.array_type}
        """
        ReflectionData.__init__(self, reflection_set)
        if data is None:
            self.array = N.zeros((self.number_of_reflections,), N.Float)
        else:
            assert(data.shape == (self.number_of_reflections,))
            self.array = data
        self.absent_value = 0.

    def calculateFromUniformAtomDistribution(self, atom_count):
        """
        Calculate the intensities for a system containing a given combination
        of atoms whose positions are assumed to be distributed uniformly
        over the unit cell.

        @param atom_count: a dictionary mapping chemical element symbols
                           to the number of atoms of that element in the
                           system
        @type atom_count: C{dict}
        """
        from AtomicScatteringFactors import atomic_scattering_factors
        sv = N.zeros((self.number_of_reflections, 3), N.Float)
        epsilon = N.zeros((self.number_of_reflections,), N.Int)
        for r in self.reflection_set:
            sv[r.index] = r.sVector().array
            epsilon[r.index] = r.symmetryFactor()
        ssq = N.sum(sv*sv, axis=-1)
        sum_f_sq = N.zeros((self.number_of_reflections,), N.Float)
        for element, count in atom_count.items():
            a, b = atomic_scattering_factors[element.lower()]
            f_atom = N.sum(a[:, N.NewAxis]
                           * N.exp(-b[:, N.NewAxis]*ssq[N.NewAxis, :]))
            sum_f_sq += count*f_atom*f_atom
        self.array[:] = epsilon*sum_f_sq


class ExperimentalAmplitudes(ExperimentalReflectionData,
                             AmplitudeData):

    """
    Structure factor amplitudes (without phases) from experiment
    """

    def __init__(self, reflection_set, data=None):
        ExperimentalReflectionData.__init__(self, reflection_set)
        if data is None:
            self.array = N.zeros((self.number_of_reflections, 2), N.Float)
        else:
            assert(data.shape == (self.number_of_reflections, 2))
            self.array = data
        self.absent_value = N.zeros((2,), N.Float)

    def intensities(self):
        intensities = ExperimentalIntensities(self.reflection_set)
        intensities.data_available[:] = self.data_available
        intensities.array[:, 0] = self.array[:, 0]*self.array[:, 0]
        intensities.array[:, 1] = 2.*self.array[:, 0]*self.array[:, 1]
        return intensities

    def applyDebyeWallerFactor(self, adp_or_scalar):
        dwf = self._debyeWallerFactor(adp_or_scalar)
        result = self.__class__(self.reflection_set, self.array*dwf)
        result.data_available[:] = self.data_available
        return result


class ExperimentalIntensities(ExperimentalReflectionData,
                              IntensityData):

    """
    Reflection intensities from experiment
    """

    def __init__(self, reflection_set, data=None):
        ExperimentalReflectionData.__init__(self, reflection_set)
        if data is None:
            self.array = N.zeros((self.number_of_reflections, 2), N.Float)
        else:
            assert(data.shape == (self.number_of_reflections, 2))
            self.array = data
        self.absent_value = N.zeros((2,), N.Float)

    def amplitudes(self):
        amplitudes = ExperimentalAmplitudes(self.reflection_set)
        amplitudes.data_available[:] = self.data_available
        for i in range(self.number_of_reflections):
            if self.array[i, 0] > 0.:
                a = N.sqrt(self.array[i, 0])
                amplitudes.array[i, 0] = a
                amplitudes.array[i, 1] = 0.5*self.array[i, 1]/a
            elif self.array[i, 0] == 0.:
                amplitudes.array[i, 0] = 0.
                amplitudes.array[i, 1] = N.sqrt(self.array[i, 1])
            else:
                amplitudes.data_available[i] = False
                amplitudes.array[i, :] = 0.
        return amplitudes

    def applyDebyeWallerFactor(self, adp_or_scalar):
        dwf = self._debyeWallerFactor(adp_or_scalar)
        result = self.__class__(self.reflection_set, self.array*(dwf**2))
        result.data_available[:] = self.data_available
        return result
