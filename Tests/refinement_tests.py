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
from CDTK.Utility import largestAbsoluteElement, compactSymmetricTensor

from CDTK.mmCIF import mmCIFFile
from CDTK.SpaceGroups import space_groups
from CDTK.Crystal import UnitCell
from CDTK.Reflections import ReflectionSet
from CDTK.ReflectionData import ExperimentalAmplitudes, StructureFactor
from CDTK.Refinement import RefinementEngine, MaximumLikelihoodRefinementEngine
from CDTK.SubsetRefinement import AtomSubsetRefinementEngine
from CDTK import Units

# The tests for derivatives with respect to positions and ADPs work
# correctly only for this slightly modified maximum-likelihood refinement
# engine which doesn't recompute the optimal alpha/beta values after each
# parameter update. In fact, the derivatives computed by the ML refinement
# engine are approximates ones that do not take into account the contribution
# to the change of the target function due to the change of alpha and beta.
class ModifiedMLRefinementEngine(MaximumLikelihoodRefinementEngine):
    _updateInternalState = RefinementEngine._updateInternalState

class CommonRefinementTests2ONX(unittest.TestCase):

    def _setUp(self):
        cif_file = mmCIFFile()
        cif_file.load_file(TextFile('2onx-sf.cif.gz'))
        cif_data = cif_file[0]

        cell_data = cif_data['cell']
        cell = UnitCell(float(cell_data['length_a'])*Units.Ang,
                        float(cell_data['length_b'])*Units.Ang,
                        float(cell_data['length_c'])*Units.Ang,
                        float(cell_data['angle_alpha'])*Units.deg,
                        float(cell_data['angle_beta'])*Units.deg,
                        float(cell_data['angle_gamma'])*Units.deg)

        space_group = space_groups[cif_data['symmetry']['space_group_name_H-M']]

        self.reflections = ReflectionSet(cell, space_group)
        for r in cif_data['refln']:
            h = int(r['index_h'])
            k = int(r['index_k'])
            l = int(r['index_l'])
            ri = self.reflections.getReflection((h, k, l))

        max_resolution, min_resolution = self.reflections.resolutionRange()
        self.reflections.fillResolutionSphere(max_resolution, min_resolution)

        self.exp_amplitudes = ExperimentalAmplitudes(self.reflections)

        for r in cif_data['refln']:
            h = int(r['index_h'])
            k = int(r['index_k'])
            l = int(r['index_l'])
            ri = self.reflections[(h, k, l)]
            if r['f_meas_au'] != '?':
                self.exp_amplitudes[ri] = N.array([float(r['f_meas_au']),
                                                   float(r['f_meas_sigma_au'])])

        s = Structure('2ONX.pdb.gz')
        self.asu_atoms = sum(([atom for atom in residue] for residue in s), [])
        self.re = ModifiedMLRefinementEngine(self.exp_amplitudes,
                  ((atom['serial_number'], atom['element'],
                    atom['position']*Units.Ang,
                    atom['temperature_factor']*Units.Ang**2*delta/(8.*N.pi**2),
                    atom['occupancy'])
                    for atom in self.asu_atoms))

    def _test_amplitude_derivatives(self):
        llk, dllk = self.re.targetFunctionAndAmplitudeDerivatives()
        self.assertAlmostEqual(llk, 685.75243727)
        da = 0.01*N.minimum.reduce(self.re.model_amplitudes)
        for ri in range(len(self.re.ssq)):
            a = self.re.model_amplitudes[ri]
            self.re.model_amplitudes[ri] = a + da
            llk_p, dummy = self.re.targetFunctionAndAmplitudeDerivatives()
            self.re.model_amplitudes[ri] = a - da
            llk_m, dummy = self.re.targetFunctionAndAmplitudeDerivatives()
            self.re.model_amplitudes[ri] = a
            deviation = dllk[ri] - (llk_p-llk_m)/(2.*da)
            self.assert_(deviation/dllk[ri] < 2.e-6)

    def _test_position_derivatives(self):
        llk, pd = self.re.targetFunctionAndPositionDerivatives()
        dp = 0.0001
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
            self.assert_(error < 1.5e-5)

    def _test_ADP_derivatives(self):
        llk, adpd = self.re.targetFunctionAndADPDerivatives()
        dp = 0.00001
        num_adpd = []
        for atom_id in self.atom_ids:
            adp = self.re.getADP(atom_id)
            gradient = N.zeros((6,), N.Float)
            for t in [Tensor([[1., 0., 0.], [0., 0., 0.], [0., 0., 0.]]),
                      Tensor([[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]]),
                      Tensor([[0., 0., 0.], [0., 0., 0.], [0., 0., 1.]]),
                      Tensor([[0., 0., 0.], [0., 0., 1.], [0., 1., 0.]]),
                      Tensor([[0., 0., 1.], [0., 0., 0.], [1., 0., 0.]]),
                      Tensor([[0., 1., 0.], [1., 0., 0.], [0., 0., 0.]])]:
                self.re.setADP(atom_id, adp+dp*t)
                llk_p, dummy = self.re.targetFunctionAndADPDerivatives()
                self.re.setADP(atom_id, adp-dp*t)
                llk_m, dummy = self.re.targetFunctionAndADPDerivatives()
                gradient += (llk_p-llk_m)/(2.*dp) \
                             * compactSymmetricTensor(t.array)
            num_adpd.append(gradient)
            self.re.setADP(atom_id, adp)
        error = largestAbsoluteElement((N.array(num_adpd)-adpd.array)/adpd.array)
        self.assert_(error < 2.e-3)

class AllAtomRefinementTests2ONX(CommonRefinementTests2ONX):

    def setUp(self):
        CommonRefinementTests2ONX._setUp(self)
        self.atom_ids = [atom['serial_number'] for atom in self.asu_atoms]

    test_amplitude_derivatives = CommonRefinementTests2ONX._test_amplitude_derivatives
    test_position_derivatives = CommonRefinementTests2ONX._test_position_derivatives
    test_ADP_derivatives = CommonRefinementTests2ONX._test_ADP_derivatives

class CalphaRefinementTests2ONX(CommonRefinementTests2ONX):

    def setUp(self):
        CommonRefinementTests2ONX._setUp(self)
        self.aa_re = self.re
        self.atom_ids = [atom['serial_number']
                         for atom in self.asu_atoms
                         if atom.name == 'CA']
        self.re = AtomSubsetRefinementEngine(self.aa_re, self.atom_ids)

    test_position_derivatives = CommonRefinementTests2ONX._test_position_derivatives
    test_ADP_derivatives = CommonRefinementTests2ONX._test_ADP_derivatives

def suite():
    loader = unittest.TestLoader()
    s = unittest.TestSuite()
    s.addTest(loader.loadTestsFromTestCase(AllAtomRefinementTests2ONX))
    s.addTest(loader.loadTestsFromTestCase(CalphaRefinementTests2ONX))
    return s

if __name__ == '__main__':
    unittest.main()
