# Test structure factor calculations
#
# This file is part of the Crystallographic Data Toolkit and
# distributed under the CeCILL-C licence. See the file LICENCE
# for the full text of this licence.
#
# Written by Konrad Hinsen.
#

import unittest
import operator

from mmLib.mmCIF import mmCIFFile

from Scientific.IO.TextFile import TextFile
from Scientific.IO.PDB import Structure
from Scientific.Geometry import Tensor
from Scientific import N

from CDTK.SpaceGroups import space_groups
from CDTK.Crystal import UnitCell
from CDTK.Reflections import ReflectionSet, \
     ExperimentalAmplitudes, StructureFactor
from CDTK import Units

class StructureFactorTests2ONX(unittest.TestCase):

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
        self.model_sf = StructureFactor(self.reflections)

        for r in cif_data['refln']:
            h = int(r['index_h'])
            k = int(r['index_k'])
            l = int(r['index_l'])
            ri = self.reflections[(h, k, l)]
            self.model_sf[ri] = \
                  float(r['f_calc'])*N.exp(1j*float(r['phase_calc'])*Units.deg)
            if r['f_meas_au'] != '?':
                self.exp_amplitudes[ri] = N.array([float(r['f_meas_au']),
                                                   float(r['f_meas_sigma_au'])])

        self.s = Structure('2ONX.pdb.gz')
        assert N.fabs(float(cell_data['length_a'])-self.s.a) < 1.e-7
        assert N.fabs(float(cell_data['length_b'])-self.s.b) < 1.e-7
        assert N.fabs(float(cell_data['length_c'])-self.s.c) < 1.e-7
        assert N.fabs(float(cell_data['angle_alpha'])-self.s.alpha) < 1.e-7
        # beta is a bit different in the PDB and reflection files
        assert N.fabs(float(cell_data['angle_beta'])-self.s.beta) < 0.0021
        assert N.fabs(float(cell_data['angle_gamma'])-self.s.gamma) < 1.e-7


    def checkSymmetry(self, sf):
        for r in self.reflections:
            other = self.reflections[(-r.h, -r.k, -r.l)]
            d = sf[r]-N.conjugate(sf[other])
            self.assert_((d*N.conjugate(d)).real < 1.e-10)

    def test_sf(self):

        # Tests on read-in data
        self.assert_(len(self.reflections) == 1416)
        self.assert_(self.exp_amplitudes.rFactor(self.exp_amplitudes) == 0.)
        self.assert_(self.model_sf.rFactor(self.model_sf) == 0.)
        self.assert_(N.fabs(self.exp_amplitudes.rFactor(self.model_sf)-0.1842)
                     < 5.e-5)
        self.checkSymmetry(self.model_sf)

        # Tests on addition/subtraction
        twice_model_sf = self.model_sf+self.model_sf
        d = twice_model_sf.array-2.*self.model_sf.array
        self.assert_(N.maximum.reduce(N.absolute(d)) < 1.e-14)
        zero_sf = self.model_sf-self.model_sf
        self.assert_(N.maximum.reduce(N.absolute(zero_sf.array)) < 1.e-14)
        self.assertRaises(AssertionError, operator.add,
                          self.model_sf, self.exp_amplitudes)
        self.assertRaises(AssertionError, operator.sub,
                          self.model_sf, self.exp_amplitudes)

        # Tests on multiplication/division
        squared_model_sf = self.model_sf*self.model_sf
        d = squared_model_sf/self.model_sf - self.model_sf
        self.assert_(N.maximum.reduce(N.absolute(d.array)) < 1.e-12)
        squared_amplitudes = self.exp_amplitudes*self.exp_amplitudes
        d = squared_amplitudes/self.exp_amplitudes - self.exp_amplitudes
        self.assert_(N.maximum.reduce(N.absolute(d.array[:, 0])) < 1.e-12)
        
        # Tests on structure factor calculations
        asu_atoms = sum(([atom for atom in residue] for residue in self.s), [])

        sf_from_asu = StructureFactor(self.reflections)
        sf_from_asu.calculateFromAsymmetricUnitAtoms(
            (atom['element'], atom['position']*Units.Ang,
             atom['temperature_factor']*Units.Ang**2/(8.*N.pi**2),
             atom['occupancy'])
            for atom in asu_atoms)

        unit_cell_atom_data = []
        for tr in self.s.cs_transformations:
            for atom in asu_atoms:
                unit_cell_atom_data.append((atom['element'],
                                            tr(atom['position'])*Units.Ang,
                                            atom['temperature_factor'] 
                                              * Units.Ang**2/(8.*N.pi**2),
                                            atom['occupancy']))

        sf_from_unit_cell = StructureFactor(self.reflections)
        sf_from_unit_cell.calculateFromUnitCellAtoms(unit_cell_atom_data)

        self.assert_(sf_from_unit_cell.rFactor(sf_from_asu)
                     < 2.e-5)
        self.assert_(N.fabs(self.exp_amplitudes.rFactor(sf_from_asu)-0.1964)
                     < 5.e-5)
        self.assert_(N.fabs(self.exp_amplitudes.rFactor(sf_from_unit_cell)
                            -0.1964)
                     < 5.e-5)
        self.assert_(N.fabs(self.model_sf.rFactor(sf_from_asu)-0.0749)
                     < 5.e-5)
        self.assert_(N.fabs(self.model_sf.rFactor(sf_from_unit_cell)-0.0749)
                     < 5.e-5)
        self.checkSymmetry(self.model_sf)

    def test_scaling(self):
        u = Tensor([[0.1, 0., 0.02],
                    [0., 0.08, 0.],
                    [0.02, 0., 0.12]])
        k = 2.5
        test_sf = k*self.model_sf.applyDebyeWallerFactor(u)
        scaled, k_fit, u_fit = self.model_sf.scaleTo(test_sf, 0)
        self.assertAlmostEqual(k_fit, k, 12)
        self.assert_(N.maximum.reduce(N.fabs(N.ravel((u-u_fit).array)))
                     < 1.e-14)
        scaled, k_fit, u_fit = test_sf.scaleTo(self.model_sf, 0)
        self.assertAlmostEqual(k_fit, 1./k, 12)
        self.assert_(N.maximum.reduce(N.fabs(N.ravel((u+u_fit).array)))
                     < 1.e-14)

        test_sf = k*self.model_sf.applyDebyeWallerFactor(-u)
        sf_scaled, k_fit, u_fit = test_sf.scaleTo(self.exp_amplitudes, 5)
        self.assert_(abs(k_fit-1./k) < 2.e-2)
        self.assert_(N.maximum.reduce(N.fabs(N.ravel((u-u_fit).array)))
                     < 1.5e-4)
        self.assert_(self.exp_amplitudes.rFactor(sf_scaled) < 0.185)


class StructureFactorAssignmentTests(unittest.TestCase):

    def test_P31(self):
        cell = UnitCell(3., 3., 4.,
                        90.*Units.deg, 90.*Units.deg, 120.*Units.deg)
        res_max = 0.5
        res_min = 10.
        reflections = ReflectionSet(cell, space_groups['P 31'],
                                    res_max, res_min)
        sf = StructureFactor(reflections)
        value = 1.1-0.8j
        for r in reflections:
            for re in r.symmetryEquivalents():
                sf[re] = value
                self.assert_(N.absolute(sf[re]-value) < 1.e-14)
                ri = reflections[(-re.h, -re.k, -re.l)]
                self.assert_(N.absolute(sf[ri]-N.conjugate(value)) < 1.e-13)

def suite():
    loader = unittest.TestLoader()
    s = unittest.TestSuite()
    s.addTest(loader.loadTestsFromTestCase(StructureFactorTests2ONX))
    s.addTest(loader.loadTestsFromTestCase(StructureFactorAssignmentTests))
    return s

if __name__ == '__main__':
    unittest.main()
