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

The class L{MaximumLikelihoodRefinementEngine} calculates a
maximum-likelihood target functions as well as its derivatives with
respect to the model parameters (positions and ADPs). There is
no minimization algorithm and no support for restraints of any kind.
"""

from CDTK.Reflections import ResolutionShell
from CDTK_math import I1divI0, logI0, logcosh
from Scientific.Functions.Interpolation import InterpolatingFunction
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

    def __init__(self, exp_amplitudes, asu_iterator):
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
        """
        mask = exp_amplitudes.data_available
        self.reflection_set = exp_amplitudes.reflection_set
        self.exp_amplitudes = N.repeat(exp_amplitudes.array[:, 0], mask)
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
            adpa = adp.array
            adps.append([adpa[0,0], adpa[1,1], adpa[2,2],
                         adpa[1,2], adpa[0,2], adpa[0,1]])
            occupancies.append(occupancy)
        self.ids = ids
        self.id_dict = id_dict
        self.elements = elements
        self.positions = N.array(positions)
        self.adps = N.array(adps)
        self.occupancies = N.array(occupancies)
        self.natoms = len(self.elements)
        sv, p = self.reflection_set.sVectorArrayAndPhasesForASU()
        self.sv = N.repeat(sv, mask, axis=1)
        self.p = N.repeat(p, mask, axis=1)
        self.ssq = N.add.reduce(self.sv[0]*self.sv[0], axis=1)
        sm = self.reflection_set.symmetryAndCentricityArrays()
        self.epsilon = N.repeat(sm[:, 0], mask)
        self.centric = N.repeat(sm[:, 1], mask)
        self.signalParameterUpdate()

    def signalParameterUpdate(self):
        """
        Recalculate all internally stored data that depends on the
        model parameters.
        """
        self.calculateModelAmplitudes()

    def _evaluateModel(self, sf, pd, adpd, deriv):
        from CDTK.AtomicScatteringFactors import atomic_scattering_factors
        twopii = 2.j*N.pi
        twopisq = -2.*N.pi**2
        tt = N.transpose([ [[1, 0, 0], [0, 0, 0], [0, 0, 0,]],
                           [[0, 0, 0], [0, 1, 0], [0, 0, 0,]],
                           [[0, 0, 0], [0, 0, 0], [0, 0, 1,]],
                           [[0, 0, 0], [0, 0, 1], [0, 1, 0,]],
                           [[0, 0, 1], [0, 0, 0], [1, 0, 0,]],
                           [[0, 1, 0], [1, 0, 0], [0, 0, 0,]] ])
        for i in range(self.natoms):
            a, b = atomic_scattering_factors[self.elements[i]]
            f_atom = N.sum(a[:, N.NewAxis]
                           * N.exp(-b[:, N.NewAxis]*self.ssq[N.NewAxis, :]))
            adp = N.innerproduct(tt, self.adps[i])
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

    def calculateModelAmplitudes(self):
        """
        Calculate the structure factor amplitudes for the model using the
        current paramters.
        """
        sf = N.zeros(self.ssq.shape, N.Complex)
        self._evaluateModel(sf, None, None, None)
        self.structure_factor = sf
        self.model_amplitudes = N.absolute(sf)

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
        return target, pd

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
        return target, -2.*N.pi**2*adpd

#
# RefinementEngine with a maximum-likelihood target function
#
class MaximumLikelihoodRefinementEngine(RefinementEngine):

    """
    A RefinementEngine whose target function is the negative logarithm of
    the likelihood for an error model assuming uncorrelated structure factor
    coefficients and uncorrelated errors for the atomic model parameters.
    """

    def __init__(self, exp_amplitudes, asu_iterator):
        self.res_shells = None
        RefinementEngine.__init__(self, exp_amplitudes, asu_iterator)
        nrefl_per_shell = min(50, max(3, len(self.ssq)/10))
        while True:
            self.defineResolutionShells(nrefl_per_shell)
            try:
                self.findAlphaBeta()
                break
            except ParameterError:
                nrefl_per_shell += nrefl_per_shell/2


    def targetFunctionAndAmplitudeDerivatives(self):
        eps_beta_inv = 1./(self.epsilon 
                           * N.array([self.beta(sq) for sq in self.ssq]))
        alpha = N.array([self.alpha(sq) for sq in self.ssq])
        alpha_a = alpha*self.model_amplitudes
        arg1 = -(self.exp_amplitudes**2+alpha_a**2)*eps_beta_inv
        arg2 = alpha_a*self.exp_amplitudes*eps_beta_inv
        darg1 = -2.*alpha_a*alpha*eps_beta_inv
        darg2 = alpha*self.exp_amplitudes*eps_beta_inv

        llk = 0.
        dllk = 0.*self.ssq
        for ri in range(len(self.ssq)):
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
        return llk, dllk

    def signalParameterUpdate(self):
        self.calculateModelAmplitudes()
        self.findAlphaBeta()

    def findAlphaBeta(self):
        if self.res_shells is None:
            return
        w = 2-self.centric
        p = self.model_amplitudes*self.exp_amplitudes/self.epsilon
        t = None
        alpha = []
        beta = []
        for rs in self.res_shells:
            a = b = c = d = tw = 0.
            for ri in rs:
                a += w[ri]*self.model_amplitudes[ri]**2/self.epsilon[ri]
                b += w[ri]*self.exp_amplitudes[ri]**2/self.epsilon[ri]
                c += w[ri]*p[ri]
                d += w[ri]*p[ri]*p[ri]
                tw += w[ri]
            a /= tw
            b /= tw
            c /= tw
            d /= tw
            if d < a*b:
                t = 0.
                # This solution corresponds to alpha=0 and doesn't make
                # physical sense.
                raise ParameterError()
            else:
                def g(t):
                    l = 0.
                    for ri in rs:
                        if self.centric[ri]:
                            h = N.tanh(t*p[ri])
                        else:
                            x = 2.*t*p[ri]
                            h = I1divI0(x)
                        l += w[ri]*p[ri]*h
                    l /= tw
                    return N.sqrt(1.+4.*a*b*t*t)-2.*t*l-1.
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
        assert nrefl_per_shell <= len(self.ssq)
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

#
# An exception used by the maximum-likelihood target function
#
class ParameterError(Exception):
    pass

