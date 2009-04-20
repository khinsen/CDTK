# Test cases for methods that require MMTK.
# This test script will fail if MMTK is not installed.
#
# This file is part of the Crystallographic Data Toolkit and
# distributed under the CeCILL-C licence. See the file LICENCE
# for the full text of this licence.
#
# Written by Konrad Hinsen.
#

import unittest

from MMTK import *
from MMTK.PDB import PDBConfiguration
from MMTK.PDBMoleculeFactory import PDBMoleculeFactory

from CDTK.MMCIF import MMCIFStructureFactorData
from CDTK.Maps import ElectronDensityMap
from CDTK.ReflectionData import StructureFactor

from Scientific.IO.TextFile import TextFile
from Scientific.Geometry import delta
from Scientific import N


class MMTKTests(unittest.TestCase):

    def setUp(self):
        cif_data = MMCIFStructureFactorData('2onx-sf.cif.gz', fill=True)
        self.reflections = cif_data.reflections

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
        density_map = ElectronDensityMap.fromUniverse(
                          self.reflections.cell,
                          4*hmax, 4*kmax, 4*lmax,
                          self.universe, self.adps)

        sf_from_map = StructureFactor.fromElectronDensityMap(
                          self.reflections, density_map)

        sf_from_universe = StructureFactor.fromUniverse(
                               self.reflections, self.universe, self.adps)

        self.assert_(sf_from_universe.rFactor(sf_from_map) < 1.e-3)
        d = N.absolute(sf_from_universe.array-sf_from_map.array)
        self.assert_(N.maximum.reduce(d) < 0.13)
        self.assert_(N.average(d) < 0.02)

        map_from_sf = ElectronDensityMap.fromStructureFactor(
                          self.reflections.cell,
                          4*hmax, 4*kmax, 4*lmax, sf_from_universe)
        sf_from_test_map = StructureFactor.fromElectronDensityMap(
                               self.reflections, map_from_sf)
        d = N.absolute(sf_from_universe.array-sf_from_test_map.array)
        self.assert_(sf_from_universe.rFactor(sf_from_test_map) < 1.e-15)
        self.assert_(N.maximum.reduce(d) < 1.e-13)


def suite():
    return unittest.TestLoader().loadTestsFromTestCase(MMTKTests)

if __name__ == '__main__':
    unittest.main()
