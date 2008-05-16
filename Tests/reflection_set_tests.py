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
from CDTK.Reflections import ReflectionSet, ResolutionShell
from CDTK.SpaceGroups import space_groups
from CDTK import Units
from Scientific.Geometry import Vector
from Scientific import N
import cPickle

class ReflectionSetTests(unittest.TestCase):

    def _shellTest(self, reflections, shells):
        nr = 0
        for rmin, rmax in shells:
            subset = ResolutionShell(reflections, rmin, rmax)
            nr += len(subset)
        self.assertEqual(len(reflections.minimal_reflection_list), nr)

    def _subsetTest(self, reflections):
        subsets = reflections.randomlyAssignedSubsets([0.1, 0.5, 0.4])
        for r in reflections:
            r.in_subset = False
        for s in subsets:
            for r in s:
                r.in_subset = True
        for r in reflections:
            self.assert_(r.in_subset)

    def _symmetryTest(self, reflections):
        max_h, max_k, max_l = reflections.maxHKL()
        min_s, max_s = reflections.sRange()
        r1, r2, r3 = reflections.crystal.unit_cell.reciprocalBasisVectors()
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

    def test_P1(self):
        cell = UnitCell(Vector(1., 0., 0.),
                        Vector(0., 1., 0.),
                        Vector(0., 0., 1.))
        res_max = 0.5
        res_min = 10.
        reflections = ReflectionSet(cell, space_groups['P 1'], res_max, res_min)
        nr = sum([r.n_symmetry_equivalents for r in reflections]) + \
             len(reflections.systematic_absences)
        self.assertEqual(len(reflections.reflection_map), nr)
        self.assert_(len(reflections.reflection_map) ==
                     2*len(reflections.minimal_reflection_list))
        for r in reflections:
            self.assert_(r.n_symmetry_equivalents == 2)
            self.assert_(res_max <= r.resolution() <= res_min)
            self.assert_(not r.isCentric())
            self.assert_(r.symmetryFactor() == 1)
        self._shellTest(reflections, [(0.5, 1.), (1., 5.), (5., 11.)])
        self._subsetTest(reflections)
        self._symmetryTest(reflections)

    def test_P31(self):
        cell = UnitCell(3., 3., 4.,
                        90.*Units.deg, 90.*Units.deg, 120.*Units.deg)
        res_max = 0.5
        res_min = 10.
        sg = space_groups['P 31']
        reflections = ReflectionSet(cell, sg, res_max, res_min)
        nr = sum([r.n_symmetry_equivalents for r in reflections]) + \
             len(reflections.systematic_absences)
        self.assertEqual(len(reflections.reflection_map), nr)
        for r in reflections:
            self.assert_(res_max <= r.resolution() <= res_min)
        for r in reflections.systematic_absences:
            self.assertEqual(r.h, 0)
            self.assertEqual(r.k, 0)
            self.assertNotEqual(r.l % 3, 0)
        self._shellTest(reflections, [(0.5, 1.), (1., 5.), (5., 11.)])
        self._subsetTest(reflections)
        self._symmetryTest(reflections)

    def test_P43212(self):
        cell = UnitCell(Vector(1., 0., 0.),
                        Vector(0., 1., 0.),
                        Vector(0., 0., 1.5))
        res_max = 0.1
        res_min = 10.
        sg = space_groups['P 43 21 2']
        reflections = ReflectionSet(cell, sg, res_max, res_min)
        nr = sum([r.n_symmetry_equivalents for r in reflections]) + \
             len(reflections.systematic_absences)
        self.assertEqual(len(reflections.reflection_map), nr)
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
        self._shellTest(reflections, [(0.1, 1.), (1., 5.), (5., 11.)])
        self._subsetTest(reflections)
        self._symmetryTest(reflections)

    def test_pickle(self):
        cell = UnitCell(3., 3., 4.,
                        90.*Units.deg, 90.*Units.deg, 120.*Units.deg)
        res_max = 0.5
        res_min = 10.
        reflections = ReflectionSet(cell, space_groups['P 31'],
                                    res_max, res_min)
        string = cPickle.dumps(reflections)
        unpickled = cPickle.loads(string)
        self.assertEqual(len(reflections), len(unpickled))
        self.assertEqual(len(reflections.minimal_reflection_list),
                         len(unpickled.minimal_reflection_list))
        self.assertEqual(len(reflections.reflection_map),
                         len(unpickled.reflection_map))
        self.assertEqual(len(reflections.systematic_absences),
                         len(unpickled.systematic_absences))
        for r in reflections:
            for re in r.symmetryEquivalents():
                rp = unpickled[(re.h, re.k, re.l)]
                self.assertEqual(re.h, rp.h)
                self.assertEqual(re.k, rp.k)
                self.assertEqual(re.l, rp.l)
                self.assertEqual(re.index, rp.index)
                self.assertEqual(re.sf_conjugate, rp.sf_conjugate)
                self.assertEqual(re.phase_factor, rp.phase_factor)
                self.assertEqual(re.n_symmetry_equivalents,
                                 rp.n_symmetry_equivalents)

        for r in reflections.systematic_absences:
            self.assert_(unpickled[(r.h, r.k, r.l)]
                         in unpickled.systematic_absences)

def suite():
    return unittest.TestLoader().loadTestsFromTestCase(ReflectionSetTests)

if __name__ == '__main__':
    unittest.main()
