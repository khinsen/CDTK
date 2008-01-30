# Reflections
#
# This file is part of the Crystallographic Data Toolkit and
# distributed under the CeCILL-C licence. See the file LICENCE
# for the full text of this licence.
#
# Written by Konrad Hinsen.
#

"""
Reflections

A L{ReflectionSet} object represents the reflections that are observed
in a crystallographic experiment. It contains reflections that lie
in a spherical shell in reciprocal space covering a specific
resolution range. Iteration over a L{ReflectionSet} yields a minimal
set of L{Reflection} objects from which all other reflections can be
obtained by symmetry criteria. Indexation with a tuple of Miller
indices (h, k, l) returns the corresponding L{Reflection} object.

A L{Reflection} object represents a single reflection in a
L{ReflectionSet}. L{Reflection} objects are used as indices to
L{CDTK.ReflectionData.ReflectionData} objects.

Data defined per reflection is handled by the module
L{CDTK.ReflectionData}.
"""

from Scientific import N

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

    def sVectorArray(self, cell=None):
        """
        @param cell: a unit cell, which defaults to the unit cell for
                     which the reflection set is defined.
        @type cell: L{CDTK.Crystal.UnitCell}
        @return: an array containing the s vectors for all reflections
        @rtype: C{N.array}
        """
        sv = N.zeros((len(self.minimal_reflection_list), 3), N.Float)
        for r in self:
            sv[r.index] = r.sVector(cell).array
        return sv

    def sVectorArrayAndPhasesForASU(self, cell=None):
        """
        Calculates the transformed s vectors and phases that are used
        for calculating the structure factor from the atoms in the
        asymmetric unit.
        @param cell: a unit cell, which defaults to the unit cell for
                     which the reflection set is defined.
        @type cell: L{CDTK.Crystal.UnitCell}
        @return: a tuple (s, p), where s is an array containing the s vectors
                 for all reflections and space group operations and p is an
                 array with the corresponding phases
        @rtype: C{N.array}
        """
        if cell is None:
            cell = self.cell
        sg = self.space_group
        ntrans = len(sg)
        sv = N.zeros((ntrans, len(self.minimal_reflection_list), 3), N.Float)
        p = N.zeros((ntrans, len(self.minimal_reflection_list)), N.Complex)
        twopii = 2.j*N.pi
        r1, r2, r3 = cell.reciprocalBasisVectors()
        for r in self:
            hkl_list = sg.symmetryEquivalentMillerIndices(r.array)[0]
            for i in range(ntrans):
                h, k, l = hkl_list[i]
                sv[i, r.index] = (h*r1+k*r2+l*r3).array
                tr_num, tr_den = sg.transformations[i][1:]
                st = r.h*float(tr_num[0])/float(tr_den[0]) \
                     + r.k*float(tr_num[1])/float(tr_den[1]) \
                     + r.l*float(tr_num[2])/float(tr_den[2])
                p[i, r.index] = N.exp(twopii*st)
        return sv, p

        sv = N.zeros((len(self.minimal_reflection_list), 3), N.Float)
        for r in self:
            sv[r.index] = r.sVector(cell).array

    def randomlyAssignedSubsets(self, fractions):
        """
        Partition the reflection set into several subsets of
        given (approximate) sizes by a random choice algorithm.

        @param fractions: a sequence of fractions (between 0. and 1.)
                          that specify the size that each subset should have
        @type fractions: sequence of C{int}
        @return: subsets of approximately the requested sizes
        @rtype: sequence of L{ReflectionSubset}
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

#
# A ReflectionSubset object is an iterator over a subset of a
# ReflectionSet.
#
class ReflectionSubset(object):

    """
    Iterator over a subset of reflections
    """

    def __init__(self, reflection_set, reflection_list):
        """
        @param reflection_set: complete reflection set
        @type reflection_set: L{ReflectionSet}
        @param reflection_list: the reflections to be included in the subset
        @type reflection_list: C{list}
        """
        self.reflection_set = reflection_set
        self.reflection_list = reflection_list

    def __len__(self):
        return len(self.reflection_list)

    def __iter__(self):
        """
        @return: a generator yielding the reflections of the subset
        @rtype: generator
        """
        for r in self.reflection_list:
            yield r

#
# A ResolutionShell object is an iterator over the reflections in
# a given resolution shell.
#
class ResolutionShell(ReflectionSubset):

    """
    Iterator over reflections in a resolution shell
    """

    def __init__(self, reflection_set, min_resolution, max_resolution):
        """
        @param reflection_set: complete reflection set
        @type reflection_set: L{ReflectionSet}
        @param min_resolution: the lower limit of the resolution range
        @type min_resolution: C{float}
        @param max_resolution: the upper limit of the resolution range
        @type max_resolution: C{float}
        """
        subset = [r for r in reflection_set
                  if min_resolution <= r.resolution() < max_resolution]
        ReflectionSubset.__init__(self, reflection_set, subset)

