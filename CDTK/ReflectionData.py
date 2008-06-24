# Data defined on reflections
#
# This file is part of the Crystallographic Data Toolkit and
# distributed under the CeCILL-C licence. See the file LICENCE
# for the full text of this licence.
#
# Written by Konrad Hinsen.
#

"""
Reflection data classes

The subclasses of L{ReflectionData} represent the various kind of data
that can be defined for each reflection in a
L{CDTK.ReflectionSet.ReflectionSet}. Only the values for the minimal
reflection set is stored explicitly, values for other reflections are
reconstructed by applying symmetry operations.

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
from CDTK import Units
from CDTK.Utility import SymmetricTensor
from Scientific.Geometry import Tensor

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
        self.number_of_reflections = len(self.reflection_set)

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
        sv = self.reflection_set.sVectorArray()
        if isinstance(adp_or_scalar, float):
            dwf = N.exp(twopisq*adp_or_scalar*N.sum(sv*sv, axis=-1))
        else:
            dwf = N.exp(twopisq*N.sum(N.dot(sv, adp_or_scalar.array2d)*sv,
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

    def completeness(self, nbins = 1, s_range = (None, None)):
        """
        @param nbins: the number of intervals into which the s range
                      is divided
        @type nbins: C{int}
        @param s_range: the range of s values for which the average
                        is calculated. The minimum and/or maximum
                        value can be C{None}, in which case it is replaced
                        by the lower/upper limit of the resolution range of
                        the reflection set.
        @type s_range: C{(float, float)}
        @return: the fraction of reflections within the resolution interval
                 for which observations are available
        @rtype: C{Scientific.N.array}
        """
        s_min, s_max = s_range
        if s_min is None or s_max is None:
            s1, s2 = self.reflection_set.sRange()
            if s_min is None: s_min = 0.99*s1
            if s_max is None: s_max = 1.01*s2
        bin_width = (s_max-s_min)/nbins
        reflection_count = N.zeros((nbins,), N.Int)
        observed_reflection_count = N.zeros((nbins,), N.Float)
        for reflection in self.reflection_set:
            s = reflection.sVector().length()
            bin = int((s-s_min)/bin_width)
            if bin >= 0 and bin < nbins:
                n = reflection.n_symmetry_equivalents
                reflection_count[bin] += n
                if self.data_available[reflection.index]:
                    observed_reflection_count[bin] += n
        completeness = observed_reflection_count / \
                            (reflection_count + (reflection_count==0))
        return completeness

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

    def rFactor(self, other, subset = None):
        """
        @param other: reflection data containing amplitudes or structure factors
        @param subset: a reflection subset over which the R factor is
                       calculated. The default value of None selects
                       the complete reflection set of the data.
        @return: the R factor between the two data sets
        @rtype: C{float}
        """
        assert isinstance(other, AmplitudeData)
        if subset is None:
            subset = self.reflection_set
        sum_self = 0.
        sum_diff = 0.
        for r in subset:
            f_self = self[r]
            f_other = other[r]
            if f_self is None or f_other is None:
                continue
            f_self = abs(f_self)
            f_other = abs(f_other)
            sum_self += f_self
            sum_diff += abs(f_self-f_other)
        return sum_diff/sum_self

    def rFactorWithScale(self, other, subset=None):
        """
        @param other: reflection data containing amplitudes or structure factors
        @param subset: a reflection subset over which the R factor is
                       calculated. The default value of None selects
                       the complete reflection set of the data.
        @return: a tuple (R, scale) where scale is the scale factor that must
                 be applied to other to minimize the R factor and R is the
                 R factor obtained with this scale factor
        @rtype: C{float}
        """
        assert isinstance(other, AmplitudeData)
        if subset is None:
            subset = self.reflection_set
        sum_self = 0.
        sum_other = 0.
        for r in subset:
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
        for r in subset:
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

    def amplitudes(self):
        """
        @return: self
        @rtype: L{ReflectionData} and L{AmplitudeData}
        """
        return self

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
        @type adp_or_scalar: L{CDTK.Utility.SymmetricTensor} or C{float}
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
        u = SymmetricTensor(uxx, uyy, uzz, uyz, uxz, uxy).makeDefinite()

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
            u += SymmetricTensor(duxx, duyy, duzz, duyz, duxz, duxy)
            u = u.makeDefinite()

        return k*self.applyDebyeWallerFactor(u), k, u


class IntensityData(object):

    """
    Mix-in class for L{ReflectionData} subclasses representing intensity
    values
    """

    def isotropicAverage(self, nbins = 50, s_range = (None, None)):
        """
        @param nbins: the number of intervals into which the s range
                      is divided before averaging the intensities within each
                      interval
        @type nbins: C{int}
        @param s_range: the range of s values for which the average
                        is calculated. The minimum and/or maximum
                        value can be C{None}, in which case it is replaced
                        by the lower/upper limit of the resolution range of
                        the reflection set.
        @type s_range: C{(float, float)}
        @return: the averaged intensities for each resolution interval
        @rtype: C{Scientific.Functions.InterpolatingFunction}
        """
        from Scientific.Functions.Interpolation import InterpolatingFunction
        s_min, s_max = s_range
        if s_min is None or s_max is None:
            s1, s2 = self.reflection_set.sRange()
            if s_min is None: s_min = 0.99*s1
            if s_max is None: s_max = 1.01*s2
        bin_width = (s_max-s_min)/nbins
        reflection_count = N.zeros((nbins,), N.Int)
        intensity_sum = N.zeros((nbins,), N.Float)
        for reflection, intensity in self:
            s = reflection.sVector().length()
            bin = int((s-s_min)/bin_width)
            if bin >= 0 and bin < nbins:
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

    def intensities(self):
        """
        @return: self
        @rtype: L{ReflectionData} and L{IntensityData}
        """
        return self

    def applyDebyeWallerFactor(self, adp_or_scalar):
        """
        @param adp_or_scalar: a symmetric ADP tensor or a scalar
                              position fluctuation value
        @type adp_or_scalar: L{CDTK.Utility.SymmetricTensor} or C{float}
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
        @type reflection_set: L{CDTK.ReflectionSet.ReflectionSet}
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
        sv = self.reflection_set.sVectorArray(cell)
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
                sfTerm(self.array, sv, f_atom[atom.symbol], conf[atom].array,
                       SymmetricTensor(adps[atom]).array, 0., 2)

    def calculateFromUnitCellAtoms(self, atom_iterator, cell=None):
        """
        @param atom_iterator: an iterator or sequence that yields
                              for each atom in the unit cell a
                              tuple of (atom_id, chemical element,
                              position vector, position fluctuation,
                              occupancy). The position fluctuation
                              can be a symmetric tensor (ADP tensor)
                              or a scalar (implicitly multiplied by
                              the unit tensor).
        @type atom_iterator: iterable
        @param cell: a unit cell, which defaults to the unit cell for
                     which the reflection set is defined.
        @type cell: L{CDTK.Crystal.UnitCell}
        """
        from AtomicScatteringFactors import atomic_scattering_factors
        from CDTK_sfcalc import sfTerm
        sv = self.reflection_set.sVectorArray(cell)
        ssq = N.sum(sv*sv, axis=-1)
        self.array[:] = 0j
        twopii = 2.j*N.pi
        twopisq = -2.*N.pi**2
        for atom_id, element, position, adp, occupancy in atom_iterator:
            a, b = atomic_scattering_factors[element.lower()]
            f_atom = occupancy * \
                     N.sum(a[:, N.NewAxis]
                           * N.exp(-b[:, N.NewAxis]*ssq[N.NewAxis, :]))
            if adp is None:
                sfTerm(self.array, sv, f_atom, position.array, sv, 0., 0)
            elif isinstance(adp, float):
                sfTerm(self.array, sv, f_atom, position.array, sv, adp, 1)
            elif isinstance(adp, SymmetricTensor):
                sfTerm(self.array, sv, f_atom, position.array, adp.array, 0., 2)
            else: # assume rank-2 tensor object
                sfTerm(self.array, sv, f_atom, position.array,
                       SymmetricTensor(adp.array).array, 0., 2)

    def calculateFromAsymmetricUnitAtoms(self, atom_iterator, cell=None):
        """
        @param atom_iterator: an iterator or sequence that yields
                              for each atom in the asymmetric unit a
                              tuple of (atom_id, chemical element,
                              position vector, position fluctuation,
                              occupancy). The position fluctuation
                              can be a symmetric tensor (ADP tensor)
                              or a scalar (implicitly multiplied by
                              the unit tensor).
        @type atom_iterator: iterable
        @param cell: a unit cell, which defaults to the unit cell for
                     which the reflection set is defined.
        @type cell: L{CDTK.Crystal.UnitCell}
        """
        from AtomicScatteringFactors import atomic_scattering_factors
        twopii = 2.j*N.pi
        twopisq = -2.*N.pi**2
        sv, p = self.reflection_set.sVectorArrayAndPhasesForASU(cell)
        ssq = N.sum(sv[0]*sv[0], axis=-1)
        self.array[:] = 0j
        for atom_id, element, position, adp, occupancy in atom_iterator:
            a, b = atomic_scattering_factors[element.lower()]
            f_atom = N.sum(a[:, N.NewAxis]
                           * N.exp(-b[:, N.NewAxis]*ssq[N.NewAxis, :]))
            if adp is None:
                dwf = 1.
            elif isinstance(adp, float):
                dwf = N.exp(twopisq*adp*ssq)
            else:
                dwf = None
            for i in range(len(p)):
                if isinstance(adp, SymmetricTensor):
                    dwf = N.exp(twopisq*N.sum(N.dot(sv[i], adp.array2d)*sv[i],
                                              axis=-1))
                elif isinstance(adp, Tensor):
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
        @type reflection_set: L{CDTK.ReflectionSet.ReflectionSet}
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
        @type reflection_set: L{CDTK.ReflectionSet.ReflectionSet}
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
        sv = self.reflection_set.sVectorArray()
        ssq = N.sum(sv*sv, axis=-1)
        epsilon = N.zeros((self.number_of_reflections,), N.Int)
        for r in self.reflection_set:
            epsilon[r.index] = r.symmetryFactor()
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
