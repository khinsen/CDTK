# Test electron density map functions
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
from Scientific import N

from CDTK.MMCIF import MMCIFStructureFactorData
from CDTK.Maps import ElectronDensityMap
from CDTK.ReflectionData import StructureFactor
from CDTK import Units

class ElectronDensityMapTests(unittest.TestCase):

    def setUp(self):
        cif_data = MMCIFStructureFactorData('2onx-sf.cif.gz', fill=True)
        self.reflections = cif_data.reflections

        self.s = Structure('2ONX.pdb.gz')
        assert N.fabs(float(cif_data.cell['length_a'])-self.s.a) < 1.e-7
        assert N.fabs(float(cif_data.cell['length_b'])-self.s.b) < 1.e-7
        assert N.fabs(float(cif_data.cell['length_c'])-self.s.c) < 1.e-7
        assert N.fabs(float(cif_data.cell['angle_alpha'])-self.s.alpha) < 1.e-7
        # beta is a bit different in the PDB and reflection files
        assert N.fabs(float(cif_data.cell['angle_beta'])-self.s.beta) < 0.0021
        assert N.fabs(float(cif_data.cell['angle_gamma'])-self.s.gamma) < 1.e-7

    def test_sf_from_map(self):
        asu_atoms = sum(([atom for atom in residue] for residue in self.s), [])
        unit_cell_atom_data = []
        for tr in self.s.cs_transformations:
            for atom in asu_atoms:
                unit_cell_atom_data.append((atom, atom['element'],
                                            tr(atom['position'])*Units.Ang,
                                            atom['temperature_factor'] 
                                              * Units.Ang**2/(8.*N.pi**2),
                                            atom['occupancy']))

        hmax, kmax, lmax = self.reflections.maxHKL()
        density_map = ElectronDensityMap(self.reflections.cell,
                                         4*hmax, 4*kmax, 4*lmax)
        density_map.calculateFromUnitCellAtoms(unit_cell_atom_data)

        sf_from_map = StructureFactor(self.reflections)
        sf_from_map.calculateFromElectronDensityMap(density_map)

        sf_from_unit_cell = StructureFactor(self.reflections)
        sf_from_unit_cell.calculateFromUnitCellAtoms(unit_cell_atom_data)

        self.assert_(sf_from_unit_cell.rFactor(sf_from_map) < 1.e-3)
        d = N.absolute(sf_from_unit_cell.array-sf_from_map.array)
        self.assert_(N.maximum.reduce(d) < 0.13)
        self.assert_(N.average(d) < 0.02)

        map_from_sf = ElectronDensityMap(self.reflections.cell,
                                         4*hmax, 4*kmax, 4*lmax)
        map_from_sf.calculateFromStructureFactor(sf_from_unit_cell)
        sf_from_test_map = StructureFactor(self.reflections)
        sf_from_test_map.calculateFromElectronDensityMap(map_from_sf)
        d = N.absolute(sf_from_unit_cell.array-sf_from_test_map.array)
        self.assert_(sf_from_unit_cell.rFactor(sf_from_test_map) < 1.e-15)
        self.assert_(N.maximum.reduce(d) < 1.e-13)


def suite():
    return unittest.TestLoader().loadTestsFromTestCase(ElectronDensityMapTests)

if __name__ == '__main__':
    unittest.main()
