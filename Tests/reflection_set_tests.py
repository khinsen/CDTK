# Tests for classes Reflection and ReflectionSet
#
# This file is part of the Crystallographic Data Toolkit and
# distributed under the CeCILL-C licence. See the file LICENCE
# for the full text of this licence.
#
# Written by Konrad Hinsen.
#

import unittest
from CDTK.Crystal import UnitCell
from CDTK.Reflections import ReflectionSet, FrozenReflectionSet
from CDTK.SpaceGroups import space_groups
from CDTK.HDF5 import HDF5Store
from CDTK import Units
from Scientific.Geometry import Vector
from Scientific import N
import cPickle
import h5py

class ReflectionSetTests(unittest.TestCase):

    def setUp(self):
        self.compact = False

    def _shellTest(self, reflections):
        subsets = reflections.resolutionShells(10)
        self.assert_(len(subsets) == 10)
        nr = 0
        for shell in subsets:
            nr += len(shell)
            for r in shell:
                s = r.sVector().length()
                self.assert_(s >= shell.s_min)
                self.assert_(s < shell.s_max)
        self.assertEqual(len(reflections), nr)

    def _subsetTest(self, reflections):
        subsets = reflections.randomlyAssignedSubsets([0.1, 0.5, 0.4])
        rs = set((r.h, r.k, r.l, r.index)
                 for r in reflections)
        for s in subsets:
            for r in s:
                rs.remove((r.h, r.k, r.l, r.index))
        self.assert_(len(rs) == 0)

    def _intersectionTest(self, reflections):
        subset = reflections.randomlyAssignedSubsets([0.3])[0]
        inter = reflections.intersection(subset)
        self.assert_(inter.cell, subset.reflection_set.cell)
        self.assert_(inter.space_group, subset.reflection_set.space_group)
        in_subset = set((r.h, r.k, r.l) for r in subset)
        in_inter = set((r.h, r.k, r.l) for r in inter)
        self.assertEqual(in_subset, in_inter)

    def _symmetryTest(self, reflections):
        max_h, max_k, max_l = reflections.maxHKL()
        min_s, max_s = reflections.sRange()
        r1, r2, r3 = reflections.crystal.cell.reciprocalBasisVectors()
        sg = reflections.crystal.space_group
        for h in range(-max_h, max_h+1):
            for k in range(-max_k, max_k+1):
                for l in range(-max_l, max_l+1):
                    sv = h*r1 + k*r2 + l*r3
                    s = sv.length()
                    if s < min_s or s > max_s:
                        continue
                    r0 = reflections[(h, k, l)]
                    if r0.isSystematicAbsence():
                        # Phase factors are not used for absent reflections
                        continue
                    p0 = r0.phase_factor
                    if r0.sf_conjugate:
                        p0 = N.conjugate(p0)
                    eq, phases = sg.symmetryEquivalentMillerIndices(
                                                         N.array([h, k, l]))
                    for (h, k, l), p in zip(eq, phases):
                        p_r = p0*p
                        r = reflections[(h, k, l)]
                        p_test = r.phase_factor
                        if r.sf_conjugate:
                            p_test = N.conjugate(p_test)
                        self.assert_(abs(p_test - p_r) < 1.e-13)

    def _equalityTest(self, reflections):
        self.assertEqual(reflections, reflections)
        subset = reflections.randomlyAssignedSubsets([0.3])[0]
        self.assertNotEqual(reflections, subset)

    def test_P1(self):
        cell = UnitCell(Vector(1., 0., 0.),
                        Vector(0., 1., 0.),
                        Vector(0., 0., 1.))
        res_max = 0.5
        res_min = 10.
        reflections = ReflectionSet(cell, space_groups['P 1'], res_max, res_min,
                                    compact=self.compact)
        self.assert_(reflections.isComplete())
        nr = sum([r.n_symmetry_equivalents for r in reflections]) + \
             sum([r.n_symmetry_equivalents
                  for r in reflections.systematic_absences])
        self.assertEqual(reflections.totalReflectionCount(), nr)
        self.assert_(reflections.totalReflectionCount() ==
                     2*len(reflections.minimal_reflection_list))
        for r in reflections:
            self.assert_(len(r.symmetryEquivalents()) == 2)
            self.assert_(res_max <= r.resolution() <= res_min)
            self.assert_(not r.isCentric())
            self.assert_(r.symmetryFactor() == 1)
        self._shellTest(reflections)
        self._subsetTest(reflections)
        self._symmetryTest(reflections)
        self._intersectionTest(reflections)
        self._equalityTest(reflections)

        frozen = reflections.freeze()
        self.assert_(frozen is reflections.freeze())
        self.assert_(frozen is frozen.freeze())
        self.assert_(frozen.isComplete())
        self.assert_(frozen.totalReflectionCount() ==
                     2*len(frozen._reflections))
        for r in frozen:
            self.assert_(reflections.hasReflection(r.h, r.k, r.l))
            self.assert_(len(r.symmetryEquivalents()) == 2)
            self.assert_(res_max <= r.resolution() <= res_min)
            self.assert_(not r.isCentric())
            self.assert_(r.symmetryFactor() == 1)
        for r in reflections:
            self.assert_(frozen.hasReflection(r.h, r.k, r.l))
            self.assert_(r.index == frozen[(r.h, r.k, r.l)].index)
        self._shellTest(frozen)
        self._subsetTest(frozen)
        self._symmetryTest(frozen)
        self._intersectionTest(frozen)
        self._equalityTest(frozen)

    def test_P31(self):
        cell = UnitCell(3., 3., 4.,
                        90.*Units.deg, 90.*Units.deg, 120.*Units.deg)
        res_max = 0.5
        res_min = 10.
        sg = space_groups['P 31']
        reflections = ReflectionSet(cell, sg, res_max, res_min,
                                    compact=self.compact)
        self.assert_(reflections.isComplete())
        nr = sum([r.n_symmetry_equivalents for r in reflections]) + \
             sum([r.n_symmetry_equivalents
                  for r in reflections.systematic_absences])
        self.assertEqual(reflections.totalReflectionCount(), nr)
        for r in reflections:
            self.assert_(res_max <= r.resolution() <= res_min)
        for r in reflections.systematic_absences:
            self.assertEqual(r.h, 0)
            self.assertEqual(r.k, 0)
            self.assertNotEqual(r.l % 3, 0)
        self._shellTest(reflections)
        self._subsetTest(reflections)
        self._symmetryTest(reflections)
        self._intersectionTest(reflections)
        self._equalityTest(reflections)

        frozen = reflections.freeze()
        self.assert_(frozen is reflections.freeze())
        self.assert_(frozen is frozen.freeze())
        self.assert_(frozen.isComplete())
        for r in frozen:
            self.assert_(reflections.hasReflection(r.h, r.k, r.l))
            self.assert_(res_max <= r.resolution() <= res_min)
        for r in reflections:
            self.assert_(frozen.hasReflection(r.h, r.k, r.l))
            self.assert_(r.index == frozen[(r.h, r.k, r.l)].index)
        self._shellTest(frozen)
        self._subsetTest(frozen)
        self._symmetryTest(frozen)
        self._intersectionTest(frozen)
        self._equalityTest(frozen)

    def test_P43212(self):
        cell = UnitCell(Vector(1., 0., 0.),
                        Vector(0., 1., 0.),
                        Vector(0., 0., 1.5))
        res_max = 0.5
        res_min = 10.
        sg = space_groups['P 43 21 2']
        reflections = ReflectionSet(cell, sg, res_max, res_min,
                                    compact=self.compact)
        self.assert_(reflections.isComplete())
        nr = sum([r.n_symmetry_equivalents for r in reflections]) + \
             sum([r.n_symmetry_equivalents
                  for r in reflections.systematic_absences])
        self.assertEqual(reflections.totalReflectionCount(), nr)
        for r in reflections:
            self.assert_(res_max <= r.resolution() <= res_min)
            is_centric = r.isCentric()
            if r.h == 0 or r.k == 0 or r.l == 0:
                self.assert_(is_centric)
            elif r.h == r.k:
                self.assert_(is_centric)
            else:
                self.assert_(not is_centric)
            eps = r.symmetryFactor()
            if r.l == 0 and (r.h == r.k or r.h == -r.k):
                self.assert_(eps == 2)
            elif int(r.h == 0) + int(r.k == 0) + int(r.l == 0) == 2:
                if r.l != 0:
                    self.assert_(eps == 4)
                else:
                    self.assert_(eps == 2)
            else:
                self.assert_(eps == 1)
        for r in reflections.systematic_absences:
            self.assertEqual(int(r.h == 0) + int(r.k == 0) + int(r.l == 0), 2)
            if r.h != 0:
                self.assertEqual(r.h % 2, 1)
            if r.k != 0:
                self.assertEqual(r.k % 2, 1)
            if r.l != 0:
                self.assertNotEqual(r.l % 4, 0)
        self._shellTest(reflections)
        self._subsetTest(reflections)
        self._symmetryTest(reflections)
        self._intersectionTest(reflections)
        self._equalityTest(reflections)

        frozen = reflections.freeze()
        self.assert_(frozen is reflections.freeze())
        self.assert_(frozen is frozen.freeze())
        self.assert_(frozen.isComplete())
        for r in frozen:
            self.assert_(reflections.hasReflection(r.h, r.k, r.l))
            self.assert_(res_max <= r.resolution() <= res_min)
        for r in reflections:
            self.assert_(frozen.hasReflection(r.h, r.k, r.l))
            self.assert_(r.index == frozen[(r.h, r.k, r.l)].index)
        self._shellTest(frozen)
        self._subsetTest(frozen)
        self._symmetryTest(frozen)
        self._intersectionTest(frozen)
        self._equalityTest(frozen)

    def test_pickle(self):
        cell = UnitCell(3., 3., 4.,
                        90.*Units.deg, 90.*Units.deg, 120.*Units.deg)
        res_max = 0.5
        res_min = 10.
        reflections = ReflectionSet(cell, space_groups['P 31'],
                                    res_max, res_min,
                                    compact=self.compact)
        self.assert_(reflections.isComplete())

        string = cPickle.dumps(reflections)
        unpickled = cPickle.loads(string)
        self.assert_(unpickled.isComplete())
        self.assertEqual(len(reflections.reflection_map),
                         len(unpickled.reflection_map))
        self._compare(reflections, unpickled)

        frozen = reflections.freeze()
        self.assert_(frozen.isComplete())
        self._compare(reflections, frozen)

        string = cPickle.dumps(frozen)
        unpickled = cPickle.loads(string)
        self.assert_(unpickled.isComplete())
        self.assert_((frozen._reflections == unpickled._reflections).any())
        self.assert_((frozen._absences == unpickled._absences).any())
        self._compare(frozen, unpickled)

    def test_hdf5(self):
        cell = UnitCell(3., 3., 4.,
                        90.*Units.deg, 90.*Units.deg, 120.*Units.deg)
        res_max = 0.5
        res_min = 10.
        reflections = ReflectionSet(cell, space_groups['P 31'],
                                    res_max, res_min,
                                    compact=self.compact)
        frozen = reflections.freeze()

        with HDF5Store('test.h5', 'w') as store:
            store.store('reflections', reflections)
            store.store('frozen_reflections', frozen)

        with HDF5Store('test.h5', 'r') as store:
            retrieved = store.retrieve('reflections')
            retrieved_frozen = store.retrieve('frozen_reflections')

        self.assert_(isinstance(retrieved, ReflectionSet))
        self.assert_(isinstance(retrieved_frozen, ReflectionSet))
        self.assert_(isinstance(retrieved, FrozenReflectionSet))
        self.assert_(isinstance(retrieved_frozen, FrozenReflectionSet))

        self._compare(reflections, retrieved, False)
        self._compare(reflections, retrieved_frozen, False)

    def _compare(self, rs1, rs2, check_indices=True):
        self.assertEqual(len(rs1), len(rs2))
        self.assertEqual(len(rs1.minimal_reflection_list),
                         len(rs2.minimal_reflection_list))
        self.assertEqual(len(rs1.systematic_absences),
                         len(rs2.systematic_absences))
        self.assertEqual(rs1.totalReflectionCount(),
                         rs2.totalReflectionCount())
        self.assertEqual(rs1.s_min, rs2.s_min)
        self.assertEqual(rs1.s_max, rs2.s_max)
        for r in rs1:
            for re in r.symmetryEquivalents():
                rp = rs2[(re.h, re.k, re.l)]
                self.assertEqual(re.h, rp.h)
                self.assertEqual(re.k, rp.k)
                self.assertEqual(re.l, rp.l)
                if check_indices:
                    self.assertEqual(re.index, rp.index)
                self.assertEqual(re.sf_conjugate, rp.sf_conjugate)
                self.assertEqual(re.phase_factor, rp.phase_factor)
                self.assertEqual(re.n_symmetry_equivalents,
                                 rp.n_symmetry_equivalents)

        for r in rs1.systematic_absences:
            self.assert_(rs2[(r.h, r.k, r.l)]
                         in rs2.systematic_absences)

class CompactReflectionSetTests(ReflectionSetTests):

    def setUp(self):
        self.compact = True

def suite():
    loader = unittest.TestLoader()
    s = unittest.TestSuite()
    s.addTest(loader.loadTestsFromTestCase(ReflectionSetTests))
    s.addTest(loader.loadTestsFromTestCase(CompactReflectionSetTests))
    return s

if __name__ == '__main__':
    unittest.main()
