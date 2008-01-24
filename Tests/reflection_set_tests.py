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

    def test_P1(self):
        cell = UnitCell(Vector(1., 0., 0.),
                        Vector(0., 1., 0.),
                        Vector(0., 0., 1.))
        res_max = 0.5
        res_min = 10.
        reflections = ReflectionSet(cell, space_groups['P 1'], res_max, res_min)
        nr = sum([r.n_symmetry_equivalents for r in reflections]) + \
             len(reflections.systematic_absences)
        self.assertEqual(len(reflections), nr)
        self.assert_(len(reflections) ==
                     2*len(reflections.minimal_reflection_list))
        for r in reflections:
            self.assert_(r.n_symmetry_equivalents == 2)
            self.assert_(res_max <= r.resolution() <= res_min)
            self.assert_(not r.isCentric())
            self.assert_(r.symmetryFactor() == 1)
        self._shellTest(reflections, [(0.5, 1.), (1., 5.), (5., 11.)])
        self._subsetTest(reflections)

    def test_P31(self):
        cell = UnitCell(3., 3., 4.,
                        90.*Units.deg, 90.*Units.deg, 120.*Units.deg)
        res_max = 0.5
        res_min = 10.
        reflections = ReflectionSet(cell, space_groups['P 31'],
                                    res_max, res_min)
        nr = sum([r.n_symmetry_equivalents for r in reflections]) + \
             len(reflections.systematic_absences)
        self.assertEqual(len(reflections), nr)
        for r in reflections:
            self.assert_(res_max <= r.resolution() <= res_min)
            sg = reflections.space_group
            for rot, tn, td in sg.transposed_transformations:
                h, k, l = N.dot(rot, N.array([r.h, r.k, r.l]))
                p = N.exp(-2j*N.pi*N.dot(N.array([r.h, r.k, r.l]), (tn*1.)/td))
                self.assert_(N.absolute(p-reflections[(h, k, l)].phase_factor)
                             < 1.e-14)
        for r in reflections.systematic_absences:
            self.assertEqual(r.h, 0)
            self.assertEqual(r.k, 0)
            self.assertNotEqual(r.l % 3, 0)
        self._shellTest(reflections, [(0.5, 1.), (1., 5.), (5., 11.)])
        self._subsetTest(reflections)

    def test_P43212(self):
        cell = UnitCell(Vector(1., 0., 0.),
                        Vector(0., 1., 0.),
                        Vector(0., 0., 1.5))
        res_max = 0.1
        res_min = 10.
        reflections = ReflectionSet(cell, space_groups['P 43 21 2'],
                                    res_max, res_min)
        nr = sum([r.n_symmetry_equivalents for r in reflections]) + \
             len(reflections.systematic_absences)
        self.assertEqual(len(reflections), nr)
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
