from Scientific import N

#
# A Reflection object stores Miller indices and a reference to the
# ReflectionSet to which it belongs, plus some bookkeeping information.
#
class Reflection(object):

    def __init__(self, h, k, l, reflection_set, index):
        self.h = h
        self.k = k
        self.l = l
        self.reflection_set = reflection_set
        self.index = index
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

    def sVector(self):
        r1, r2, r3 = self.reflection_set.cell.reciprocal_basis
        return self.h*r1 + self.k*r2 + self.l*r3

    def qVector(self):
        return 2.*N.pi*self.sVector()

    def resolution(self):
        return 1./self.sVector().length()

    def isSystematicAbsence(self):
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
        rs = self.reflection_set
        ri = self.index
        equivalents = rs.space_group.symmetryEquivalentMillerIndices(self.array)
        equivalents.extend([-hkl for hkl in equivalents])
        unique_reflections = \
              set([Reflection(h, k, l, rs, ri) for h, k, l in equivalents])
        n = len(unique_reflections)
        for r in unique_reflections:
            r.n_symmetry_equivalents = n
        return unique_reflections

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

    def __init__(self, cell, space_group,
                 max_resolution=None, min_resolution=None):
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
        hkl = Reflection(h, k, l, self,
                         len(self.minimal_reflection_list))
        if self.reflection_map.has_key((hkl.h, hkl.k, hkl.l)):
            return
        equivalents = list(hkl.symmetryEquivalents())
        equivalents.sort()
        for r in equivalents:
            self.reflection_map[(r.h, r.k, r.l)] = r
        hkl = equivalents[-1]
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
        max_inv_sq_resolution = 1.00001/max_resolution**2
        if min_resolution is None:
            min_inv_sq_resolution = 0.
        else:
            min_inv_sq_resolution = (1.-0.00001)/min_resolution**2
        r1, r2, r3 = self.cell.reciprocal_basis
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

    def resolutionRange(self):
        if not self.reflection_map:
            raise ValueError("Empty ReflectionSet")
        return 1./self.s_max, 1./self.s_min

    def __iter__(self):
        for r in self.minimal_reflection_list:
            yield r

    def __len__(self):
        return len(self.reflection_map)

    def __getitem__(self, item):
        return self.reflection_map[item]

    def getReflection(self, hkl):
        try:
            return self.reflection_map[hkl]
        except KeyError:
            h, k, l = hkl
            self.addReflection(h, k, l)
            return self.reflection_map[hkl]


class ReflectionData(object):

    def __init__(self, reflection_set):
        self.reflection_set = reflection_set
        self.number_of_reflections = \
                 len(self.reflection_set.minimal_reflection_list)

    def __getitem__(self, reflection):
        index = reflection.index
        if index is None: # systematic absence
            return self.absent_value
        return self.array[index]

    def __iter__(self):
        for r in self.reflection_set:
            yield (r, self[r])


class ExperimentalReflectionData(ReflectionData):

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
        index = reflection.index
        if index is None: # systematic absence
            raise ValueError("Cannot set value: "
                             "reflection is absent due to symmetry")
        self.data_available[index] = True
        self.array[index, 0] = value_sigma[0]
        self.array[index, 1] = value_sigma[1]

    def setFromArrays(self, h, k, l, value, sigma, missing=None):
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


class AmplitudeData(object):

    def rFactor(self, other):
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
        sum_self = 0.
        sum_other = 0.
        for r in self.reflection_set:
            f_self = self[r]
            f_other = other[r]
            if f_self is None or f_other is None:
                continue
            f_self = abs(f_self)
            f_other = abs(f_other)
            sum_self += f_self
            sum_other += f_other
        scale = sum_self/sum_other
        sum_diff = 0.
        for r in self.reflection_set:
            f_self = self[r]
            f_other = other[r]
            if f_self is None or f_other is None:
                continue
            f_self = abs(f_self)
            f_other = abs(f_other)
            sum_diff += abs(f_self-scale*f_other)
        return sum_diff/sum_self, scale


class StructureFactor(ReflectionData, AmplitudeData):

    def __init__(self, reflection_set):
        ReflectionData.__init__(self, reflection_set)
        self.array = N.zeros((self.number_of_reflections,), N.Complex)
        self.absent_value = 0j

    def __setitem__(self, reflection, value):
        index = reflection.index
        if index is None: # systematic absence
            raise ValueError("Cannot set value: "
                             "reflection is absent due to symmetry")
        self.array[index] = value

    def setFromArrays(self, h, k, l, modulus, phase):
        n = len(h)
        assert len(k) == n
        assert len(l) == n
        assert len(modulus) == n
        assert len(phase) == n
        for i in range(n):
            r = self.reflection_set[(h[i], k[i], l[i])]
            self.array[r.index] = modulus[i]*N.exp(1j*phase[i])


class ExperimentalAmplitudes(ExperimentalReflectionData,
                             AmplitudeData):

    def __init__(self, reflection_set):
        ExperimentalReflectionData.__init__(self, reflection_set)
        self.array = N.zeros((self.number_of_reflections, 2), N.Float)
        self.absent_value = N.zeros((2,), N.Float)

    def convertToIntensities(self):
        intensities = ExperimentalIntensities(self.reflection_set)
        intensities.data_available[:] = self.data_available
        intensities.array[:, 0] = self.array[:, 0]*self.array[:, 0]
        intensities.array[:, 1] = 2.*self.array[:, 0]*self.array[:, 1]
        return intensities


class ExperimentalIntensities(ExperimentalReflectionData):

    def __init__(self, reflection_set):
        ExperimentalReflectionData.__init__(self, reflection_set)
        self.array = N.zeros((self.number_of_reflections, 2), N.Float)
        self.absent_value = N.zeros((2,), N.Float)

    def convertToAmplitudes(self):
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
