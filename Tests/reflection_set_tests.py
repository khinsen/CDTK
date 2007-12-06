import unittest
from CDTK.Crystal import UnitCell
from CDTK.Reflections import ReflectionSet
from CDTK.SpaceGroups import space_groups
from CDTK import Units
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


def suite():
    return unittest.TestLoader().loadTestsFromTestCase(ReflectionSetTests)

if __name__ == '__main__':
    unittest.main()
