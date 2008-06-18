# Structure refinement
#
# This file is part of the Crystallographic Data Toolkit and
# distributed under the CeCILL-C licence. See the file LICENCE
# for the full text of this licence.
#
# Written by Konrad Hinsen.
#

"""
Basic structure refinement support

The class L{RefinementEngine} calculates a target function as well as
its derivatives with respect to the model parameters (positions and
ADPs). There is no minimization algorithm and no support for
restraints of any kind. Two subclasses implement a least-squares
target and a maximum-likelihood target.
"""

#
# Enable distributed computing support if specified by environment variable.
#
import os
import copy
_distributed = os.environ.get("CDTK_DISTRIBUTED_REFINEMENT_TASKS", 0) > 0
del os
if _distributed:
    from CDTK.DistributedComputations import _evaluateModel_distributed, \
         _distributed_refinement_cleanup

from CDTK.Reflections import ResolutionShell
from CDTK.Utility import SymmetricTensor
from CDTK_math import I1divI0, logI0, logcosh
from Scientific.Functions.Interpolation import InterpolatingFunction
from Scientific.Geometry import Vector, Tensor
from Scientific import N

#
# A RefinementEngine represents a model (whose parameters can be
# updated) and a target function to be minimized.
#
class RefinementEngine(object):

    """
    A RefinementEngine describes a crystallographic model whose parameters
    can be modified. It also defines a target function whose global minimum
    corresponds to the ideal refined state.
    """

    def __init__(self, exp_amplitudes, asu_iterator, working_subset=None,
                 validation_subset=None):
        """
        @param exp_amplitudes: experimental structure factor amplitudes
        @type exp_amplitudes: L{CDTK.ReflectionData.ExperimentalAmplitudes}
        @param asu_iterator: an iterator or sequence that yields
                             for each atom in the asymmetric unit a
                             tuple of (id, chemical element,
                             position vector, ADP tensor,
                             occupancy). The id object is an arbitrary object
                             that uniquely identifies an atom; it is used
                             later for updating or retrieving the parameters
                             of this atom.
        @type asu_iterator: iterator
        @param working_subset: a subset of the reflections used in the
                               calculation of the target function. If None,
                               all reflections are used.
        @type working_subset: L{CDTK.Reflection.ReflectionSubset}
        @param validation_subset: a subset of the reflections used for
                                  validation. If None, all reflections are used.
        @type validation_subset: L{CDTK.Reflection.ReflectionSubset}
        """
        # Store the atomic model
        ids = []
        id_dict = {}
        elements = []
        positions = []
        adps = []
        occupancies = []
        for id, element, position, adp, occupancy in asu_iterator:
            ids.append(id)
            id_dict[id] = len(id_dict)
            elements.append(element.lower())
            positions.append(position.array)
            adps.append(SymmetricTensor(adp).array)
            occupancies.append(occupancy)
        self.ids = ids
        self.id_dict = id_dict
        self.elements = elements
        self.positions = N.array(positions)
        self.adps = N.array(adps)
        self.occupancies = N.array(occupancies)
        self.natoms = len(self.elements)

        # Define masks for the working and validation subsets
        working_set = 0*exp_amplitudes.data_available
        if working_subset is None:
            working_set[:] = 1
        else:
            for r in working_subset:
                working_set[r.index] = 1
        validation_set = 0*exp_amplitudes.data_available
        if validation_subset is None:
            validation_set[:] = 1
        else:
            for r in validation_subset:
                validation_set[r.index] = 1

        # Choose the reflections to keep
        mask = N.logical_and(exp_amplitudes.data_available,
                             N.logical_or(working_set, validation_set))
        self.nreflections = int(N.sum(mask))
        self.working_set = N.repeat(working_set, mask)
        self.nwreflections = N.int_sum(self.working_set)
        self.validation_set = N.repeat(validation_set, mask)
        self.nvreflections = N.int_sum(self.validation_set)
        
        # Store the experimental structure factor amplitudes
        self.reflection_set = exp_amplitudes.reflection_set
        self.exp_amplitudes = N.repeat(exp_amplitudes.array[:, 0], mask)
        self.exp_sigmas = N.repeat(exp_amplitudes.array[:, 1], mask)
        if N.minimum.reduce(self.exp_sigmas) <= 0.:
            raise ValueError("Experimental sigmas must be positive")
        self.exp_sigmas_sq = self.exp_sigmas*self.exp_sigmas
        self.working_exp_amplitudes = \
                N.repeat(self.exp_amplitudes, self.working_set)
        self.validation_exp_amplitudes = \
                N.repeat(self.exp_amplitudes, self.validation_set)
        self.scale = None

        # Precompute arrays that are used frequently later
        self._precomputeArrays(mask)
        
        # Mark the internal state as invalid because the model amplitudes
        # have not yet been calculated
        self.state_valid = False

    def _precomputeArrays(self, mask):
        # S vectors, their squared length, and the phase factors for
        # structure calculation.
        sv, p = self.reflection_set.sVectorArrayAndPhasesForASU()
        self.sv = N.repeat(sv, mask, axis=1)
        self.p = N.repeat(p, mask, axis=1)
        self.ssq = N.add.reduce(self.sv[0]*self.sv[0], axis=1)
        # Symmetry factors and centricity for all reflections
        sm = self.reflection_set.symmetryAndCentricityArrays()
        self.epsilon = N.repeat(sm[:, 0], mask)
        self.centric = N.repeat(sm[:, 1], mask)
        self.working_centric_indices = \
               N.array(N.repeat(N.arange(self.nreflections),
                                N.logical_and(self.working_set,
                                              self.centric)),
                       N.Int32)
        self.working_acentric_indices = \
               N.array(N.repeat(N.arange(self.nreflections),
                                N.logical_and(self.working_set,
                                              1-self.centric)),
                       N.Int32)
        # Atomic scattering factors for the atoms in the model
        from CDTK.AtomicScatteringFactors import atomic_scattering_factors
        e_indices = {}
        for i in range(self.natoms):
            e_indices.setdefault(self.elements[i], len(e_indices))
        self.element_indices = N.array([e_indices[self.elements[i]]
                                        for i in range(self.natoms)],
                                       N.Int32)
        f_atom = N.zeros((len(e_indices), self.nreflections), N.Float)
        for i in range(self.natoms):
            a, b = atomic_scattering_factors[self.elements[i]]
            f_atom[self.element_indices[i], :] = \
                  N.sum(a[:, N.NewAxis] * N.exp(-b[:, N.NewAxis]
                                                * self.ssq[N.NewAxis, :]))
        self.f_atom = f_atom

    def getPosition(self, atom_id):
        """
        @param atom_id: id of the atom whose position is requested
        @return: the position of the atom
        @rtype: C{Scientific.Geometry.Vector}
        """
        return Vector(self.positions[self.id_dict[atom_id]])

    def getADP(self, atom_id):
        """
        @param atom_id: id of the atom whose ADP tensor is requested
        @return: the ADP tensor of the atom
        @rtype: C{Scientific.Geometry.Tensor}
        """
        return SymmetricTensor(self.adps[self.id_dict[atom_id]])

    def getOccupancy(self, atom_id):
        """
        @param atom_id: id of the atom whose occupancy is requested
        @return: the occupancy the atom
        @rtype: C{float}
        """
        return self.occupancies[self.id_dict[atom_id]]

    def setPosition(self, atom_id, position):
        """
        @param atom_id: id of the atom whose position is changed
        @param position: the new position for the atom
        @type position: C{Scientific.Geometry.Vector}
        """
        self.positions[self.id_dict[atom_id]] = position.array
        self.state_valid = False

    def setADP(self, atom_id, adp):
        """
        @param atom_id: id of the atom whose ADP tensor is changed
        @param adp: the new ADP tensor for the atom
        @type adp: C{CDTK_symtensor.SymmetricTensor}
        """
        self.adps[self.id_dict[atom_id]] = adp.array
        self.state_valid = False

    def setOccupancy(self, atom_id, occupancy):
        """
        @param atom_id: id of the atom whose position is changed
        @param occupancy: the new position for the atom
        @type occupancy: C{float}
        """
        assert occupancy >= 0. and occupancy <= 1.
        self.occupancies[self.id_dict[atom_id]] = occupancy
        self.state_valid = False

    def updateInternalState(self):
        """
        Recalculate all internally stored data that depends on the
        model parameters.
        """
        if not self.state_valid:
            self._updateInternalState()
        self.state_valid = True

    def _updateInternalState(self):
        self._calculateModelAmplitudes()

    def _evaluateModel(self, sf, pd, adpd, deriv):
        from CDTK_sfcalc import sfDeriv
        dummy_array = N.zeros((0,), N.Int)
        if sf is None: sf = dummy_array
        if pd is None: pd = dummy_array
        if adpd is None: adpd = dummy_array
        if deriv is None:
            deriv = dummy_array
            sf_in = dummy_array
            a_in = dummy_array
        else:
            sf_in = self.structure_factor
            a_in = self.model_amplitudes
        sfDeriv(self.element_indices, self.f_atom, self.positions,
                self.adps, self.occupancies, self.sv, self.p,
                sf, pd, adpd, deriv, sf_in, a_in)

    if _distributed:
        _evaluateModel = _evaluateModel_distributed
        _distribution_initialized = False
        __del__ = _distributed_refinement_cleanup
        def __getstate__(self):
            state = copy.copy(self.__dict__)
            try:
                del state['_distribution_initialized']
            except KeyError:
                pass
            return state

    def _evaluateModel_python(self, sf, pd, adpd, deriv):
        # This is the first implementation of _evaluateModel()
        # in pure Python. It is left as a documentation and for testing.
        twopii = 2.j*N.pi
        twopisq = -2.*N.pi**2
        for i in range(self.natoms):
            f_atom = self.f_atom[self.element_indices[i]]
            adp = SymmetricTensor(self.adps[i]).array2d
            for j in range(len(self.p)):
                sv = self.sv[j]
                dwf = N.exp(twopisq*N.sum(N.dot(sv, adp)*sv, axis=-1))
                pf = N.exp(twopii*(N.dot(sv, self.positions[i])))
                sfi = self.occupancies[i]*self.p[j]*f_atom * dwf * pf
                if sf is not None:
                    sf[:] += sfi
                if pd is not None:
                    pd[i] += N.sum(((N.conjugate(twopii*sfi)
                                     * self.structure_factor).real * deriv
                                     / self.model_amplitudes)[:, N.NewAxis]
                                     * sv)
                if adpd is not None:
                    ssq = N.transpose([sv[:, 0]**2, sv[:, 1]**2, sv[:, 2]**2,
                                       2.*sv[:,1]*sv[:,2],
                                       2.*sv[:,0]*sv[:,2],
                                       2.*sv[:,0]*sv[:,1]])
                    adpd[i] += N.sum(((N.conjugate(sfi)
                                       * self.structure_factor).real * deriv
                                       / self.model_amplitudes)[:, N.NewAxis]
                                       * ssq)

    def _calculateModelAmplitudes(self):
        # Calculate the structure factor amplitudes for the model using the
        # current parameters.
        sf = N.zeros(self.ssq.shape, N.Complex)
        self._evaluateModel(sf, None, None, None)
        self.structure_factor = sf
        self.model_amplitudes = N.absolute(sf)
        self.working_model_amplitudes = \
                N.repeat(self.model_amplitudes, self.working_set)
        self.validation_model_amplitudes = \
                N.repeat(self.model_amplitudes, self.validation_set)

    def rFactors(self):
        """
        @return: the R factors for the working and the validation subset
        @rtype: C{(float, float)}
        """
        self.updateInternalState()
        scale = N.sum(self.working_model_amplitudes
                      * self.working_exp_amplitudes) \
                  / N.sum(self.working_model_amplitudes**2)
        r_work = N.sum(N.fabs(scale*self.working_model_amplitudes
                              - self.working_exp_amplitudes)) \
                   / N.sum(self.working_exp_amplitudes)
        r_free = N.sum(N.fabs(scale*self.validation_model_amplitudes
                              - self.validation_exp_amplitudes)) \
                   / N.sum(self.validation_exp_amplitudes)
        return r_work, r_free

    def targetFunction(self):
        """
        Calculate the target function of the refinement (the function whose
        global minimum corresponds to the ideal refined model).
        @return: target
        @rtype: C{float}
        """
        return self.targetFunctionAndAmplitudeDerivatives()[0]

    def targetFunctionAndAmplitudeDerivatives(self):
        """
        Calculate the target function of the refinement (the function whose
        global minimum corresponds to the ideal refined model) and its
        derivatives with respect to the structure factor amplitudes
        of the model.
        @return: (target, derivatives)
        @rtype: (C{float}, C{N.array_type}) 
        """
        return NotImplementedError

    def targetFunctionAndPositionDerivatives(self):
        """
        Calculate the target function and its derivatives with respect to the
        atomic position parameters of the model.
        @return: (target, derivatives)
        @rtype: (C{float}, C{N.array_type}) 
        """
        target, deriv = self.targetFunctionAndAmplitudeDerivatives()
        pd = N.zeros(self.positions.shape, N.Float)
        self._evaluateModel(None, pd, None, deriv)
        return target, AtomPositionDataArray(self, pd)

    def targetFunctionAndADPDerivatives(self):
        """
        Calculate the target function and its derivatives with respect to the
        six elements of the ADP tensor of all atoms.
        @return: (target, derivatives)
        @rtype: (C{float}, C{N.array_type}) 
        """
        target, deriv = self.targetFunctionAndAmplitudeDerivatives()
        adpd = N.zeros(self.adps.shape, N.Float)
        self._evaluateModel(None, None, adpd, deriv)
        return target, AtomDataArray(self, -2.*N.pi**2*adpd)

    def targetFunctionScaleFactorDerivative(self, f):
        """
        Calculate the derivative of the target function with respect to the
        scale factor of the model amplitudes.
        @param f: the scale factor value
        @type f: C{float}
        @return: scale factor derivative
        @rtype: C{float}
        """
        self.scale = f
        target, deriv = self.targetFunctionAndAmplitudeDerivatives()
        return N.sum(N.repeat(deriv, self.working_set) * \
                     self.working_model_amplitudes)

#
# Thin wrapper around arrays representing per-atom positions, ADPs,
# or derivatives. They permit indexing with atom id objects.
#
class AtomDataArray(object):
    
    def __init__(self, refinement_engine, array):
        self.re = refinement_engine
        self.array = array

    def __getitem__(self, atom_id):
        return self.array[self.re.id_dict[atom_id]]

class AtomPositionDataArray(AtomDataArray):

    def __getitem__(self, atom_id):
        return Vector(self.array[self.re.id_dict[atom_id]])

    
#
# RefinementEngine with a least-squares-likelihood target function
#
class LeastSquaresRefinementEngine(RefinementEngine):

    """
    A RefinementEngine whose target function is the sum over all reflections
    of the squared deviation of the model structure factor amplitudes from
    the experimental ones, after multiplication of the model amplitudes by
    an optimized scale factor.
    """

    def __init__(self, *args, **kw_args):
        RefinementEngine.__init__(self, *args, **kw_args)
        self.weights = N.ones((self.nreflections,), N.Float)
        self.calculateWeights()
        self.working_weights = N.repeat(self.weights, self.working_set)
        self.validation_weights = N.repeat(self.weights, self.validation_set)
        wweights = N.repeat(self.working_weights, self.working_weights > 0.)
        self.constant_term = -0.5*N.sum(N.log(2.*N.pi/wweights))

    def calculateWeights(self):
        """
        Calculate the statistical weight of each reflection in the
        array self.weights. Subclasses can override this method to
        implement different weighting schemes. The default implementation
        sets the weight to 1/sigma**2.
        """
        self.weights[:] = 1./self.exp_sigmas_sq

    def targetFunction(self):
        self.updateInternalState()
        me = self.working_model_amplitudes*self.working_exp_amplitudes
        mm = self.working_model_amplitudes**2
        scale = N.sum(self.working_weights*me)/N.sum(self.working_weights*mm)
        self.scale = scale
        df = scale*self.working_model_amplitudes - self.working_exp_amplitudes
        return (0.5*N.sum(self.working_weights*df*df) + self.constant_term) \
               / self.nwreflections

    def targetFunctionAndAmplitudeDerivatives(self):
        self.updateInternalState()
        me = self.working_model_amplitudes*self.working_exp_amplitudes
        s_me = N.sum(self.working_weights*me)
        mm = self.working_model_amplitudes**2
        s_mm = N.sum(self.working_weights*mm)
        scale = s_me/s_mm
        self.scale = scale
        df = scale*self.working_model_amplitudes - self.working_exp_amplitudes
        sum_sq = N.sum(self.working_weights*df*df)
        df = scale*self.model_amplitudes - self.exp_amplitudes
        sderiv = self.weights*self.exp_amplitudes/s_mm \
                 - 2.*self.weights*self.model_amplitudes*s_me/s_mm**2
        deriv = 2.*scale*self.weights*df + \
                N.sum(N.repeat(2.*self.weights*df*self.model_amplitudes,
                               self.working_set)) \
                 * sderiv
        return (0.5*sum_sq+self.constant_term)/self.nwreflections, \
               0.5*deriv*self.working_set/self.nwreflections

#
# RefinementEngines with a maximum-likelihood target function
#
class MLRefinementEngine(RefinementEngine):

    """
    A RefinementEngine whose target function is the negative logarithm of
    the likelihood for an error model assuming uncorrelated structure factor
    coefficients and Gaussian errors on the structure factor amplitudes
    given by the experimental sigma values.
    """

    def __init__(self, exp_amplitudes, asu_iterator, working_subset=None,
                 validation_subset=None, undefined_atom_count = {}):
        RefinementEngine.__init__(self, exp_amplitudes, asu_iterator,
                                  working_subset, validation_subset)
        self._initializeAlphaBeta(undefined_atom_count)
        self._precomputeModelIndependentTerms()

    def _initializeAlphaBeta(self, undefined_atom_count):
        from AtomicScatteringFactors import atomic_scattering_factors
        self.alpha = N.ones((self.nreflections,), N.Float)
        self.beta = N.zeros((self.nreflections,), N.Float)
        for element, count in undefined_atom_count.items():
            a, b = atomic_scattering_factors[element.lower()]
            f_atom = N.sum(a[:, N.NewAxis]
                           * N.exp(-b[:, N.NewAxis]*self.ssq[N.NewAxis, :]))
            self.beta += count*f_atom*f_atom

    def _precomputeModelIndependentTerms(self):
        self.eps_beta_inv = 1./(self.epsilon*self.beta +
                                (2-self.centric)*self.exp_sigmas_sq)
        self.llk0 = (0.5*N.sum(N.log(2*N.take(self.eps_beta_inv,
                                              self.working_centric_indices)
                                     / N.pi))
                     + N.sum(N.log(2.*N.take(self.exp_amplitudes
                                             * self.eps_beta_inv,
                                             self.working_acentric_indices)))
                     ) / self.nwreflections

    def targetFunctionAndAmplitudeDerivatives(self):
        self.updateInternalState()
        if self.scale is None:
            self.optimizeScaleFactor()

        alpha_a = self.scale*self.alpha*self.model_amplitudes
        arg1 = -(self.exp_amplitudes**2+alpha_a**2)*self.eps_beta_inv
        darg1 = -self.scale*alpha_a*self.alpha*self.eps_beta_inv
        darg2 = self.scale*self.alpha*self.exp_amplitudes*self.eps_beta_inv
        arg2 = darg2*self.model_amplitudes

        llk, dllk = _llkwd(arg1, arg2, darg1, darg2, 0.*self.ssq,
                           self.working_centric_indices,
                           self.working_acentric_indices)
        return llk-self.llk0, dllk

    def targetFunctionScaleFactorDerivative(self, scale):
        # A more efficient reimplementation.
        self.updateInternalState()
        alpha_a = scale*self.alpha*self.model_amplitudes
        p = -alpha_a*alpha_a*self.eps_beta_inv
        q = alpha_a*self.exp_amplitudes*self.eps_beta_inv
        return _llkdf(p, q, self.working_centric_indices,
                     self.working_acentric_indices)

    def optimizeScaleFactor(self):
        self.updateInternalState()
        # As a first approximation, calculate the optimal scale factor
        # for a least-squares residual.
        me = self.working_model_amplitudes*self.working_exp_amplitudes
        mm = self.working_model_amplitudes**2
        self.scale = N.sum(me)/N.sum(mm)
        # Find the exact optimum using the likelihood function.
        self.scale = _findZero(self.targetFunctionScaleFactorDerivative,
                               self.scale)


class MLWithModelErrorsRefinementEngine(RefinementEngine):

    """
    A RefinementEngine whose target function is the negative logarithm of
    the likelihood for an error model assuming uncorrelated structure factor
    coefficients and uncorrelated errors for the atomic model parameters.
    """

    def __init__(self, exp_amplitudes, asu_iterator, working_subset=None,
                 validation_subset=None):
        self.res_shells = None
        RefinementEngine.__init__(self, exp_amplitudes, asu_iterator,
                                  working_subset, validation_subset)
        nrefl_per_shell = min(50, max(3, self.nreflections/10))
        self._calculateModelAmplitudes()
        self.defineResolutionShells(nrefl_per_shell)
        self.findAlphaBeta()

    def targetFunctionAndAmplitudeDerivatives(self):
        self.updateInternalState()
        beta = N.array([self.beta(sq) for sq in self.ssq])
        eps_beta_inv = 1./(self.epsilon*beta +
                           (2-self.centric)*self.exp_sigmas_sq)
        alpha = N.array([self.alpha(sq) for sq in self.ssq])
        alpha_a = alpha*self.model_amplitudes
        arg1 = -(self.exp_amplitudes**2+alpha_a**2)*eps_beta_inv
        arg2 = alpha_a*self.exp_amplitudes*eps_beta_inv
        darg1 = -2.*alpha_a*alpha*eps_beta_inv
        darg2 = alpha*self.exp_amplitudes*eps_beta_inv

        llk = 0.
        dllk = 0.*self.ssq
        for ri in range(self.nreflections):
            if self.working_set[ri]:
                if self.centric[ri]:
                    llk -= 0.5*arg1[ri]+logcosh(arg2[ri]) \
                           + 0.5*N.log(2*eps_beta_inv[ri]/N.pi)
                    # cosh(x)' = sinh(x)
                    # log(cosh(x))' = tanh(x)
                    dllk[ri] = -(0.5*darg1[ri]+N.tanh(arg2[ri])*darg2[ri])
                else:
                    llk -= arg1[ri]+logI0(2.*arg2[ri]) \
                           + N.log(2.*self.exp_amplitudes[ri]*eps_beta_inv[ri])
                    # I0(x)' = I1(x)
                    # log(I0(x))' = I1(x)/I0(x)
                    dllk[ri] = -(darg1[ri]+2.*I1divI0(2*arg2[ri])*darg2[ri])
        return llk/self.nwreflections, dllk/self.nwreflections

    def _updateInternalState(self):
        RefinementEngine._updateInternalState(self)
        self.findAlphaBeta()

    def findAlphaBeta(self):
        if self.res_shells is None:
            return
        p = self.model_amplitudes*self.exp_amplitudes/self.epsilon
        t = None
        alpha = []
        beta = []
        for rsc, rsa in self.res_shells:
            a = b = c = d = 0.
            tw = len(rsc) + 2.*len(rsa)
            for ri in rsc:
                a += self.model_amplitudes[ri]**2/self.epsilon[ri]
                b += self.exp_amplitudes[ri]**2/self.epsilon[ri]
                c += p[ri]
                d += p[ri]*p[ri]
            for ri in rsa:
                a += 2.*self.model_amplitudes[ri]**2/self.epsilon[ri]
                b += 2.*self.exp_amplitudes[ri]**2/self.epsilon[ri]
                c += 2.*p[ri]
                d += 2.*p[ri]*p[ri]
            a /= tw
            b /= tw
            c /= tw
            d /= tw
            if d < a*b:
                t = 0.
            else:
                def g(t):
                    return N.sqrt(1.+4.*a*b*t*t)-2.*t*_l(t, p, rsc, rsa)-1.
                if t is None:
                    t = 1.
                while g(t) > 0.:
                    t = t/2.
                t1 = t
                while g(t) < 0.:
                    t = 2.*t
                t2 = t
                g1 = g(t1)
                g2 = g(t2)
                while t2-t1 > 1.e-3*t1:
                    t = t1-g1*(t2-t1)/(g2-g1)
                    gt = g(t)
                    if gt == 0.:
                        break
                    elif gt < 0:
                        t1 = t
                        g1 = gt
                    else:
                        t2 = t
                        g2 = gt
            s = N.sqrt(1.+4.*a*b*t*t)
            v = N.sqrt((s-1)/(2*a))
            u = N.sqrt((s+1)/(2*b))
            alpha.append(v/u)
            beta.append(1./(u*u))
        self.alpha = InterpolatingFunction((self.ssq_av_shell,),
                                         N.array([alpha[0]]+alpha+[alpha[-1]]))
        self.beta = InterpolatingFunction((self.ssq_av_shell,),
                                           N.array([beta[0]]+beta+[beta[-1]]))

    def defineResolutionShells(self, nrefl_per_shell):
        assert nrefl_per_shell <= self.nreflections
        indices = N.argsort(self.ssq)
        self.res_shells = []
        for first in range(0, len(indices), nrefl_per_shell/2):
            self.res_shells.append(indices[first:first+nrefl_per_shell])
            if len(self.res_shells[-1]) < nrefl_per_shell:
                self.res_shells[-1] = indices[-nrefl_per_shell:]
                break
        self.ssq_av_shell = N.array([0.99*N.minimum.reduce(self.ssq)] +
                                    [N.sum(N.take(self.ssq, rs))/len(rs)
                                     for rs in self.res_shells] + \
                                     [1.001*N.maximum.reduce(self.ssq)])
        # Separate each list into centric and acentric reflections
        self.res_shells = [(N.array([ri for ri in rs
                                     if self.centric[ri]], N.Int32),
                            N.array([ri for ri in rs
                                     if not self.centric[ri]], N.Int32))
                           for rs in self.res_shells]

#
# Time-critical functions used by the maximum-likelihood target functions
#
try:
    from CDTK_refinement import _llkwd, _llkdf, _l
except ImportError:
    import sys
    sys.error.write("Module CDTK_refinement is missing\n")

    def _llkwd(arg1, arg2, darg1, darg2, dllk, rc, ra):
        llk = 0.
        for ri in rc:
            llk -= 0.5*arg1[ri]+logcosh(arg2[ri])
            dllk[ri] = -darg1[ri]-N.tanh(arg2[ri])*darg2[ri]
        for ri in ra:
            llk -= arg1[ri]+logI0(2.*arg2[ri])
            dllk[ri] = -2.*(darg1[ri]+I1divI0(2*arg2[ri])*darg2[ri])
        nwreflections = len(rc) + len(ra)
        return llk/nwreflections, dllk/nwreflections

    def _llkdf(p, q, rc, ra):
        d = 0.
        for ri in rc:
            d -= p[ri] + q[ri]*N.tanh(q[ri])
        for ri in ra:
            d -= 2.*p[ri] + 2.*q[ri]*I1divI0(2.*q[ri])
        return d/(len(rc)+len(ra))

    def _l(t, p, rsc, rsa):
        lv = 0.
        for ri in rsc:
            lv += p[ri]*N.tanh(t*p[ri])
        for ri in rsa:
            p2 = 2.*p[ri]
            lv += p2*I1divI0(t*p2)
        return lv / (len(rsc) + 2.*len(rsa))

#
# Find a zero of a function by bisection.
# Note: this is not a good algorithm for general functions, but it is
# OK for the functions used in this module.
#
def _findZero(g, t):
    while g(t) > 0.:
        t = t/1.2
    t1 = t
    while g(t) < 0.:
        t = 1.2*t
    t2 = t
    g1 = g(t1)
    g2 = g(t2)
    while t2-t1 > 1.e-4*t1:
        t = t1-g1*(t2-t1)/(g2-g1)
        gt = g(t)
        if abs(gt) < 1.e-10:
            break
        elif gt < 0:
            t1 = t
            g1 = gt
        else:
            t2 = t
            g2 = gt
    return t
