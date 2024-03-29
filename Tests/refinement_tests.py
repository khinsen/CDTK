# Test structure refinement code
#
# This file is part of the Crystallographic Data Toolkit and
# distributed under the CeCILL-C licence. See the file LICENCE
# for the full text of this licence.
#
# Written by Konrad Hinsen.
#

import unittest

from Scientific.IO.TextFile import TextFile
from Scientific.IO.PDB import Structure
from Scientific.Geometry import Vector, Tensor, delta
from Scientific import N
from CDTK.Utility import largestAbsoluteElement
from CDTK_symtensor import SymmetricTensor

from CDTK.MMCIF import MMCIFStructureFactorData
from CDTK.Reflections import ReflectionSubset
from CDTK.Refinement import RefinementEngine, LeastSquaresRefinementEngine, \
                            MLRefinementEngine, \
                            MLWithModelErrorsRefinementEngine
from CDTK.SubsetRefinement import AtomSubsetRefinementEngine
from CDTK import Units

# The tests for derivatives with respect to positions and ADPs work
# correctly only for this slightly modified maximum-likelihood refinement
# engine which doesn't recompute the optimal alpha/beta values after each
# parameter update. In fact, the derivatives computed by the ML refinement
# engine are approximates ones that do not take into account the contribution
# to the change of the target function due to the change of alpha and beta.
class ModifiedMLRefinementEngine(MLWithModelErrorsRefinementEngine):
    _updateInternalState = RefinementEngine._updateInternalState

class CommonRefinementTests2ONX(unittest.TestCase):

    def _setUp(self):
        cif_data = MMCIFStructureFactorData('2onx-sf.cif.gz', fill=True)
        self.reflections = cif_data.reflections
        self.exp_amplitudes = cif_data.data

        all = [r for r in self.reflections]
        self.work = ReflectionSubset(self.reflections, all[:200])
        self.free = ReflectionSubset(self.reflections, all[200:])

        s = Structure('2ONX.pdb.gz')
        self.asu_atoms = sum(([atom for atom in residue] for residue in s), [])

    def _test_internal_structures(self):
        for i in self.re.working_centric_indices:
            self.assert_(self.re.working_set[i])
            self.assert_(self.re.centric[i])
        for i in self.re.working_acentric_indices:
            self.assert_(self.re.working_set[i])
            self.assert_(not self.re.centric[i])
        self.assert_(len(self.re.working_centric_indices)
                     + len(self.re.working_acentric_indices)
                     == N.sum(self.re.working_set))

    def _test_amplitude_derivatives(self, target, precision):
        sum_sq_ref = self.re.targetFunction()
        sum_sq, deriv = self.re.targetFunctionAndAmplitudeDerivatives()
        self.assertEqual(sum_sq, sum_sq_ref)
        self.assertAlmostEqual(sum_sq, target)
        da = 0.005*N.minimum.reduce(self.re.model_amplitudes)
        max_error = 0.
        for ri in range(len(self.re.ssq)):
            a = self.re.model_amplitudes[ri]
            self.re.model_amplitudes[ri] = a + da
            self.re.working_model_amplitudes = \
                N.repeat(self.re.model_amplitudes, self.re.working_set)
            sum_sq_p, dummy = self.re.targetFunctionAndAmplitudeDerivatives()
            self.re.model_amplitudes[ri] = a - da
            self.re.working_model_amplitudes = \
                N.repeat(self.re.model_amplitudes, self.re.working_set)
            sum_sq_m, dummy = self.re.targetFunctionAndAmplitudeDerivatives()
            self.re.model_amplitudes[ri] = a
            deviation = deriv[ri] - (sum_sq_p-sum_sq_m)/(2.*da)
            if deriv[ri] == 0.:
                self.assert_(self.re.working_set[ri] == 0)
                self.assert_(deviation == 0.)
            else:
                max_error = max(max_error, abs(deviation/deriv[ri]))
        self.assert_(max_error < precision)

    def _test_position_derivatives(self, precision):
        llk, pd = self.re.targetFunctionAndPositionDerivatives()
        dp = 0.0001
        max_error = 0.
        for atom_id in self.atom_ids:
            p = self.re.getPosition(atom_id)
            gradient = Vector(0., 0., 0.)
            for v in [Vector(1.,0.,0.), Vector(0.,1.,0.), Vector(0.,0.,1.)]:
                self.re.setPosition(atom_id, p+dp*v)
                llk_p, dummy = self.re.targetFunctionAndPositionDerivatives()
                self.re.setPosition(atom_id, p-dp*v)
                llk_m, dummy = self.re.targetFunctionAndPositionDerivatives()
                gradient += (llk_p-llk_m)/(2.*dp) * v
            self.re.setPosition(atom_id, p)
            error = (gradient-pd[atom_id]).length()/pd[atom_id].length()
            max_error = max(max_error, error)
        self.assert_(max_error < precision)

    def _test_ADP_derivatives(self, precision):
        llk, adpd = self.re.targetFunctionAndADPDerivatives()
        dp = 0.000005
        num_adpd = []
        for atom_id in self.atom_ids:
            adp = self.re.getADP(atom_id)
            gradient = N.zeros((6,), N.Float)
            for t in [SymmetricTensor(1., 0., 0., 0., 0., 0.),
                      SymmetricTensor(0., 1., 0., 0., 0., 0.),
                      SymmetricTensor(0., 0., 1., 0., 0., 0.),
                      SymmetricTensor(0., 0., 0., 1., 0., 0.),
                      SymmetricTensor(0., 0., 0., 0., 1., 0.),
                      SymmetricTensor(0., 0., 0., 0., 0., 1.)]:
                self.re.setADP(atom_id, adp+dp*t)
                llk_p, dummy = self.re.targetFunctionAndADPDerivatives()
                self.re.setADP(atom_id, adp-dp*t)
                llk_m, dummy = self.re.targetFunctionAndADPDerivatives()
                gradient += (llk_p-llk_m)/(2.*dp) * t.array
            num_adpd.append(gradient)
            self.re.setADP(atom_id, adp)
        error = largestAbsoluteElement((N.array(num_adpd)-adpd.array)/adpd.array)
        self.assert_(error < precision)


class AllAtomLSQRefinementTests2ONX(CommonRefinementTests2ONX):

    def setUp(self):
        CommonRefinementTests2ONX._setUp(self)
        global rs
        rs = self.reflections
        self.re = LeastSquaresRefinementEngine(self.exp_amplitudes,
                  ((atom['serial_number'], atom['element'],
                    atom['position']*Units.Ang,
                    atom['temperature_factor']*Units.Ang**2*delta/(8.*N.pi**2),
                    atom['occupancy'])
                    for atom in self.asu_atoms),
                  self.work, self.free)
        self.atom_ids = [atom['serial_number'] for atom in self.asu_atoms]

    test_internal_structures = CommonRefinementTests2ONX._test_internal_structures
 
    def test_amplitude_derivatives(self):
        CommonRefinementTests2ONX._test_amplitude_derivatives(self,
                                  1.4620002571745807, 6.e-8)

    def test_position_derivatives(self):
        CommonRefinementTests2ONX._test_position_derivatives(self, 2.7e-5)

    def test_ADP_derivatives(self):
        CommonRefinementTests2ONX._test_ADP_derivatives(self, 1.e-3)


class AllAtomMLRefinementTests2ONX(CommonRefinementTests2ONX):

    def setUp(self):
        CommonRefinementTests2ONX._setUp(self)
        self.re = MLRefinementEngine(self.exp_amplitudes,
                  ((atom['serial_number'], atom['element'],
                    atom['position']*Units.Ang,
                    atom['temperature_factor']*Units.Ang**2*delta/(8.*N.pi**2),
                    atom['occupancy'])
                    for atom in self.asu_atoms),
                  self.work, self.free)
        self.re.optimizeScaleFactor()
        self.atom_ids = [atom['serial_number'] for atom in self.asu_atoms]

    test_internal_structures = CommonRefinementTests2ONX._test_internal_structures
    def test_scale_factor(self):
        llk = self.re.targetFunction()
        scale = self.re.scale
        self.assertAlmostEqual(scale, 0.95006033349406649)
        for f in [0.9, 0.95, 1.05, 1.1]:
            self.re.scale = f*scale
            self.assert_(self.re.targetFunction() >= llk)
        self.re.scale = scale

    def test_amplitude_derivatives(self):
        CommonRefinementTests2ONX._test_amplitude_derivatives(self,
                                  4.6385834857607229, 1.e-6)

    def test_position_derivatives(self):
        CommonRefinementTests2ONX._test_position_derivatives(self, 3.e-5)
 
    def test_ADP_derivatives(self):
        CommonRefinementTests2ONX._test_ADP_derivatives(self, 2.e-3)


class AllAtomMLWMERefinementTests2ONX(CommonRefinementTests2ONX):

    def setUp(self):
        CommonRefinementTests2ONX._setUp(self)
        self.re = ModifiedMLRefinementEngine(self.exp_amplitudes,
                  ((atom['serial_number'], atom['element'],
                    atom['position']*Units.Ang,
                    atom['temperature_factor']*Units.Ang**2*delta/(8.*N.pi**2),
                    atom['occupancy'])
                    for atom in self.asu_atoms),
                  self.work, self.free)
        self.atom_ids = [atom['serial_number'] for atom in self.asu_atoms]

    test_internal_structures = CommonRefinementTests2ONX._test_internal_structures
    def test_amplitude_derivatives(self):
        CommonRefinementTests2ONX._test_amplitude_derivatives(self,
                                  2.7832439338960988, 1.e-7)

    def test_position_derivatives(self):
        CommonRefinementTests2ONX._test_position_derivatives(self, 5.e-6)

    def test_ADP_derivatives(self):
        CommonRefinementTests2ONX._test_ADP_derivatives(self, 5.e-4)


class CalphaRefinementTests2ONX(CommonRefinementTests2ONX):

    def setUp(self):
        CommonRefinementTests2ONX._setUp(self)
        self.re = MLRefinementEngine(self.exp_amplitudes,
                  ((atom['serial_number'], atom['element'],
                    atom['position']*Units.Ang,
                    atom['temperature_factor']*Units.Ang**2*delta/(8.*N.pi**2),
                    atom['occupancy'])
                    for atom in self.asu_atoms),
                  self.work, self.free)
        self.aa_re = self.re
        self.atom_ids = [atom['serial_number']
                         for atom in self.asu_atoms
                         if atom.name == 'CA']
        self.re = AtomSubsetRefinementEngine(self.aa_re, self.atom_ids)

    def test_position_derivatives(self):
        CommonRefinementTests2ONX._test_position_derivatives(self, 2.e-4)

    def test_ADP_derivatives(self):
        CommonRefinementTests2ONX._test_ADP_derivatives(self, 2.e-4)


def suite():
    loader = unittest.TestLoader()
    s = unittest.TestSuite()
    s.addTest(loader.loadTestsFromTestCase(AllAtomLSQRefinementTests2ONX))
    s.addTest(loader.loadTestsFromTestCase(AllAtomMLRefinementTests2ONX))
    s.addTest(loader.loadTestsFromTestCase(AllAtomMLWMERefinementTests2ONX))
    s.addTest(loader.loadTestsFromTestCase(CalphaRefinementTests2ONX))
    return s

if __name__ == '__main__':
    unittest.main()
