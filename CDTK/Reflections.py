from Scientific import N, LA
from Scientific.Geometry import Tensor
from CDTK import Units
import copy

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
        if cell is None:
            cell = self.reflection_set.cell
        r1, r2, r3 = cell.reciprocalBasisVectors()
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
        sg = self.reflection_set.space_group
        equivalents = sg.symmetryEquivalentMillerIndices(self.array)[0]
        return N.sum(N.alltrue(N.array(equivalents) == self.array, axis=1))

    def isCentric(self):
        sg = self.reflection_set.space_group
        equivalents = sg.symmetryEquivalentMillerIndices(self.array)[0]
        return N.sum(N.alltrue(N.array(equivalents) == -self.array, axis=1)) > 0
        
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
        if not self.reflection_map:
            raise ValueError("Empty ReflectionSet")
        return self.s_min, self.s_max

    def resolutionRange(self):
        if not self.reflection_map:
            raise ValueError("Empty ReflectionSet")
        return 1./self.s_max, 1./self.s_min

    def maxHKL(self):
        return tuple(N.maximum.reduce(N.array(self.reflection_map.keys())))
        
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

    def __init__(self, reflection_set):
        self.reflection_set = reflection_set
        self.number_of_reflections = \
                 len(self.reflection_set.minimal_reflection_list)

    def __getitem__(self, reflection):
        index = reflection.index
        if index is None: # systematic absence
            return self.absent_value
        return self.array[index]

    def __setitem__(self, reflection, value):
        index = reflection.index
        if index is None: # systematic absence
            raise ValueError("Cannot set value: "
                             "reflection is absent due to symmetry")
        self.array[index] = value

    def __iter__(self):
        for r in self.reflection_set:
            yield (r, self[r])

    def __len__(self):
        return self.number_of_reflections

    def __add__(self, other):
        assert self.__class__ is other.__class__
        assert self.reflection_set is other.reflection_set
        result = self.__class__(self.reflection_set)
        self.__add_op__(other, result)
        return result

    def __iadd__(self, other):
        assert self.__class__ is other.__class__
        assert self.reflection_set is other.reflection_set
        self.__iadd_op__(other)
        return self

    def __sub__(self, other):
        assert self.__class__ is other.__class__
        assert self.reflection_set is other.reflection_set
        result = self.__class__(self.reflection_set)
        self.__sub_op__(other, result)
        return result

    def __isub__(self, other):
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

    def rFactor(self, other):
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
        return ModelIntensities(self.reflection_set,
                                (self.array[:]*N.conjugate(self.array[:])).real)

    def applyDebyeWallerFactor(self, adp_or_scalar):
        dwf = self._debyeWallerFactor(adp_or_scalar)
        return self.__class__(self.reflection_set, self.array*dwf)

    def scaleTo(self, other, iterations=0):
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

    def isotropicAverage(self, nbins = 50):
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
        from Scientific.Functions.Interpolation import InterpolatingFunction
        av = self.isotropicAverage(nbins)
        s = av.axes[0]
        intensity_average = av.values
        return InterpolatingFunction((s*s,), N.log(intensity_average))

    def applyDebyeWallerFactor(self, adp_or_scalar):
        dwf = self._debyeWallerFactor(adp_or_scalar)
        return self.__class__(self.reflection_set, self.array*(dwf**2))

    def scaleTo(self, other, iterations=0):
        a, k, u = self.amplitudes().scaleTo(other.amplitudes(), iterations)
        return a.intensities(), k, u

    def normalize(self, atom_count):
        i_random = ModelIntensities(self.reflection_set)
        i_random.calculateFromUniformAtomDistribution(atom_count)
        i_random, k, u = i_random.scaleTo(self)
        return self/i_random


class StructureFactor(ReflectionData, AmplitudeData):

    def __init__(self, reflection_set, data=None):
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

    def setFromArrays(self, h, k, l, modulus, phase):
        n = len(h)
        assert len(k) == n
        assert len(l) == n
        assert len(modulus) == n
        assert len(phase) == n
        for i in range(n):
            r = self.reflection_set[(h[i], k[i], l[i])]
            f = modulus[i]*N.exp(1j*phase[i])
            if r.sf_conjugate:
                self.array[r.index] = N.conjugate(f)/r.phase_factor
            else:
                self.array[r.index] = f/r.phase_factor

    def calculateFromUniverse(self, universe, adps=None, conf=None):
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
        from CDTK_sf_fft import map_to_sf
        m_fc = self.reflection_set.cell.fractionalToCartesianMatrix()
        det_m_fc = LA.determinant(m_fc)
        map_to_sf(density_map.array, self, det_m_fc)
        

class ModelAmplitudes(ReflectionData,
                      AmplitudeData):

    def __init__(self, reflection_set, data=None):
        ReflectionData.__init__(self, reflection_set)
        if data is None:
            self.array = N.zeros((self.number_of_reflections,), N.Float)
        else:
            assert(data.shape == (self.number_of_reflections,))
            self.array = data
        self.absent_value = 0.


class ModelIntensities(ReflectionData,
                       IntensityData):

    def __init__(self, reflection_set, data=None):
        ReflectionData.__init__(self, reflection_set)
        if data is None:
            self.array = N.zeros((self.number_of_reflections,), N.Float)
        else:
            assert(data.shape == (self.number_of_reflections,))
            self.array = data
        self.absent_value = 0.

    def amplitudes(self):
        return ModelAmplitudes(self.reflection_set, N.sqrt(self.array))

    def calculateFromUniformAtomDistribution(self, atom_count):
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
