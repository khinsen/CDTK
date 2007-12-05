import unittest

from mmLib.mmCIF import mmCIFFile

from Scientific.IO.TextFile import TextFile
from Scientific.IO.PDB import Structure
from Scientific import N

from CDTK.SpaceGroups import space_groups
from CDTK.Crystal import UnitCell, ElectronDensityMap
from CDTK.Reflections import ReflectionSet, StructureFactor
from CDTK import Units

class ElectronDensityMapTests(unittest.TestCase):

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

        self.s = Structure('2ONX.pdb.gz')
        assert N.fabs(float(cell_data['length_a'])-self.s.a) < 1.e-7
        assert N.fabs(float(cell_data['length_b'])-self.s.b) < 1.e-7
        assert N.fabs(float(cell_data['length_c'])-self.s.c) < 1.e-7
        assert N.fabs(float(cell_data['angle_alpha'])-self.s.alpha) < 1.e-7
        # beta is a bit different in the PDB and reflection files
        assert N.fabs(float(cell_data['angle_beta'])-self.s.beta) < 0.0021
        assert N.fabs(float(cell_data['angle_gamma'])-self.s.gamma) < 1.e-7

    def test_sf_from_map(self):
        asu_atoms = sum(([atom for atom in residue] for residue in self.s), [])
        unit_cell_atom_data = []
        for tr in self.s.cs_transformations:
            for atom in asu_atoms:
                unit_cell_atom_data.append((atom['element'],
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
