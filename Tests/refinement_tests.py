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
from Scientific.Geometry import delta
from Scientific import N

from CDTK.mmCIF import mmCIFFile
from CDTK.SpaceGroups import space_groups
from CDTK.Crystal import UnitCell
from CDTK.Reflections import ReflectionSet
from CDTK.ReflectionData import ExperimentalAmplitudes, StructureFactor
from CDTK.Refinement import MaximumLikelihoodRefinementEngine
from CDTK import Units


class RefinementTests2ONX(unittest.TestCase):

    def setUp(self):
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
        asu_atoms = sum(([atom for atom in residue] for residue in s), [])
        self.re = MaximumLikelihoodRefinementEngine(self.exp_amplitudes,
                  ((atom['serial_number'], atom['element'],
                    atom['position']*Units.Ang,
                    atom['temperature_factor']*Units.Ang**2*delta/(8.*N.pi**2),
                    atom['occupancy'])
                    for atom in asu_atoms))

    def test_amplitude_derivatives(self):
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

    def test_position_derivatives(self):
        llk, pd = self.re.targetFunctionAndPositionDerivatives()
        dp = 0.0001
        max_deviation = 0.
        for i in range(len(self.re.positions)):
            for xyz in range(3):
                p = self.re.positions[i, xyz]
                self.re.positions[i, xyz] = p+dp
                self.re.calculateModelAmplitudes()
                llk_p, dummy = self.re.targetFunctionAndPositionDerivatives()
                self.re.positions[i, xyz] = p-dp
                self.re.calculateModelAmplitudes()
                llk_m, dummy = self.re.targetFunctionAndPositionDerivatives()
                self.re.positions[i, xyz] = p
                deviation = pd[i, xyz] - (llk_p-llk_m)/(2.*dp)
                self.assert_(abs(deviation/pd[i, xyz]) < 7.e-5)

    def test_ADP_derivatives(self):
        llk, adpd = self.re.targetFunctionAndADPDerivatives()
        dp = 0.00001
        max_deviation = 0.
        for i in range(len(self.re.positions)):
            for el in range(6):
                p = self.re.adps[i, el]
                self.re.adps[i, el] = p+dp
                self.re.calculateModelAmplitudes()
                llk_p, dummy = self.re.targetFunctionAndADPDerivatives()
                self.re.adps[i, el] = p-dp
                self.re.calculateModelAmplitudes()
                llk_m, dummy = self.re.targetFunctionAndADPDerivatives()
                self.re.adps[i, el] = p
                deviation = adpd[i, el] - (llk_p-llk_m)/(2.*dp)
                self.assert_(abs(deviation/adpd[i, el]) < 2.e-3)

def suite():
    loader = unittest.TestLoader()
    s = unittest.TestSuite()
    s.addTest(loader.loadTestsFromTestCase(RefinementTests2ONX))
    return s

if __name__ == '__main__':
    unittest.main()
