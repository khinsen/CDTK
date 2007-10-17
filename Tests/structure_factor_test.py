import unittest

from mmLib.mmCIF import mmCIFFile

from Scientific.IO.TextFile import TextFile
from Scientific.IO.PDB import Structure
from Scientific import N

from Crystallography.SpaceGroups import space_groups
from Crystallography.Crystal import UnitCell
from Crystallography.Reflections import ReflectionSet, \
     ExperimentalAmplitudes, StructureFactor
from Crystallography import Units

class StructureFactorTests(unittest.TestCase):

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


    def test_sf(self):

        # Tests on read-in data
        self.assert_(len(self.reflections) == 1416)
        self.assert_(self.exp_amplitudes.rFactor(self.exp_amplitudes) == 0.)
        self.assert_(self.model_sf.rFactor(self.model_sf) == 0.)
        self.assert_(N.fabs(self.exp_amplitudes.rFactor(self.model_sf)-0.1842)
                     < 5.e-5)

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


if __name__ == '__main__':
    unittest.main()
