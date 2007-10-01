import unittest
from Crystallography.Crystal import UnitCell
from Crystallography.Reflections import ReflectionSet
from Crystallography.SpaceGroups import space_groups
from Scientific.Geometry import Vector
from Scientific import N


class ReflectionSetTests(unittest.TestCase):

    def test_P1(self):
        cell = UnitCell(Vector(1., 0., 0.),
                        Vector(0., 1., 0.),
                        Vector(0., 0., 1.))
        res_max = 0.5
        res_min = 10.
        reflections = ReflectionSet(cell, space_groups['P 1'], res_max, res_min)
        nr = sum([r.symmetry_count for r in reflections]) + \
             len(reflections.systematic_absences)
        self.assertEqual(len(reflections), nr)
        self.assert_(len(reflections) ==
                     2*len(reflections.minimal_reflection_list))
        for r in reflections:
            self.assert_(r.symmetry_count == 2)
            self.assert_(res_max <= r.resolution() <= res_min)
     
    def test_P43212(self):
        cell = UnitCell(Vector(1., 0., 0.),
                        Vector(0., 1., 0.),
                        Vector(0., 0., 1.5))
        res_max = 0.5
        res_min = 10.
        reflections = ReflectionSet(cell, space_groups['P 43 21 2'],
                                    res_max, res_min)
        nr = sum([r.symmetry_count for r in reflections]) + \
             len(reflections.systematic_absences)
        self.assertEqual(len(reflections), nr)
        for r in reflections:
            self.assert_(res_max <= r.resolution() <= res_min)
        for r in reflections.systematic_absences:
            self.assertEqual((r.h == 0) + (r.k == 0) + (r.l == 0), 2)
            if r.h != 0:
                self.assertEqual(r.h % 2, 1)
            if r.k != 0:
                self.assertEqual(r.k % 2, 1)
            if r.l != 0:
                self.assertNotEqual(r.l % 4, 0)

     
if __name__ == '__main__':
    unittest.main()