# Test cased for methods that require MMTK.
# This test script will fail if MMTK is not installed.

import unittest

from mmLib.mmCIF import mmCIFFile

from MMTK import *
from MMTK.PDB import PDBConfiguration
from MMTK.PDBMoleculeFactory import PDBMoleculeFactory

from CDTK.SpaceGroups import space_groups
from CDTK.Crystal import UnitCell, ElectronDensityMap
from CDTK.Reflections import ReflectionSet, StructureFactor

from Scientific.IO.TextFile import TextFile
from Scientific.Geometry import delta
from Scientific import N


class MMTKTests(unittest.TestCase):

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

        conf = PDBConfiguration('2ONX.pdb.gz')
        factory = PDBMoleculeFactory(conf)
        self.universe = factory.retrieveUnitCell()
        assert self.reflections.cell.isCompatibleWith(self.universe, 1.e-3)

        self.adps = ParticleTensor(self.universe)
        for atom in self.universe.atomList():
            b = atom.temperature_factor/(8.*N.pi**2)
            self.adps[atom] = b*delta

    def test_sf_from_map(self):
        hmax, kmax, lmax = self.reflections.maxHKL()
        density_map = ElectronDensityMap(self.reflections.cell,
                                         4*hmax, 4*kmax, 4*lmax)
        density_map.calculateFromUniverse(self.universe, self.adps)

        sf_from_map = StructureFactor(self.reflections)
        sf_from_map.calculateFromElectronDensityMap(density_map)

        sf_from_universe = StructureFactor(self.reflections)
        sf_from_universe.calculateFromUniverse(self.universe, self.adps)

        self.assert_(sf_from_universe.rFactor(sf_from_map) < 1.e-3)
        d = N.absolute(sf_from_universe.array-sf_from_map.array)
        self.assert_(N.maximum.reduce(d) < 0.13)
        self.assert_(N.average(d) < 0.02)

        map_from_sf = ElectronDensityMap(self.reflections.cell,
                                         4*hmax, 4*kmax, 4*lmax)
        map_from_sf.calculateFromStructureFactor(sf_from_universe)
        sf_from_test_map = StructureFactor(self.reflections)
        sf_from_test_map.calculateFromElectronDensityMap(map_from_sf)
        d = N.absolute(sf_from_universe.array-sf_from_test_map.array)
        self.assert_(sf_from_universe.rFactor(sf_from_test_map) < 1.e-15)
        self.assert_(N.maximum.reduce(d) < 1.e-13)

if __name__ == '__main__':
    unittest.main()
