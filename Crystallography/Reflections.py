from Scientific import N

class Reflection(object):

    def __init__(self, h, k, l, reflection_set, reflection_index):
        self.h = h
        self.k = k
        self.l = l
        self.reflection_set = reflection_set
        self.reflection_index = reflection_index
        self.symmetry_count = None

    def _getarray(self):
        return N.array([self.h, self.k, self.l])
    array = property(_getarray)

    def __cmp__(self, other):
        pref1 = 4*(self.h >= 0) + 2*(self.k >= 0) + (self.l >= 0)
        pref2 = 4*(other.h >= 0) + 2*(other.k >= 0) + (other.l >= 0)
        return cmp(pref1, pref2) \
               or cmp(self.h, other.h) \
               or cmp(self.k, other.k) \
               or cmp(self.l, other.l)

    def __repr__(self):
        return "Reflection(%d, %d, %d)" % (self.h, self.k, self.l)

    def __hash__(self):
        return 400*self.h + 20*self.k + self.l

    def sVector(self):
        r1, r2, r3 = self.reflection_set.cell.reciprocal_basis
        return self.h*r1 + self.k*r2 + self.l*r3

    def qVector(self):
        return 2.*N.pi*self.sVector()

    def resolution(self):
        return 1./self.sVector().length()

    def isSystematicAbsence(self):
        hkl = N.array([self.h, self.k, self.l])
        pf = {}
        for rot, tn, td in self.reflection_set.space_group.transformations:
            hkl_rot = tuple(N.dot(N.transpose(rot), hkl))
            t = (tn*1.)/td
            pf[hkl_rot] = pf.get(hkl_rot, 0.) + N.exp(2j*N.pi*N.dot(hkl, t))
        for z in pf.values():
            if abs(z) > 1.e-12:
                return False
        return True

    def equivalentReflections(self):
        rs = self.reflection_set
        ri = self.reflection_index
        equivalents = rs.space_group.equivalentMillerIndices(self.array)
        equivalents.extend([-hkl for hkl in equivalents])
        unique_reflections = \
              set([Reflection(h, k, l, rs, ri) for h, k, l in equivalents])
        n = len(unique_reflections)
        for r in unique_reflections:
            r.symmetry_count = n
        return unique_reflections


class ReflectionSet(object):

    def __init__(self, cell, space_group, max_resolution):
        self.cell = cell
        self.space_group = space_group
        inv_sq_resolution = 1./max_resolution**2
        r1, r2, r3 = self.cell.reciprocal_basis
        h_max = int(N.sqrt(inv_sq_resolution/(r1*r1)))
        k_max = int(N.sqrt(inv_sq_resolution/(r2*r2)))
        l_max = int(N.sqrt(inv_sq_resolution/(r3*r3)))
        self.minimal_reflection_list = []
        self.reflection_map = {}
        self.systematic_absences = set()
        rotations = [N.transpose(t[0])
                     for t in self.space_group.transformations]
        for h in range(-h_max, h_max+1):
            s1 = h*r1
            for k in range(-k_max, k_max+1):
                s2 = k*r2
                for l in range(-l_max, l_max+1):
                    s3 = l*r3
                    s = s1+s2+s3
                    if s*s <= inv_sq_resolution:
                        hkl = Reflection(h, k, l, self,
                                         len(self.minimal_reflection_list))
                        if self.reflection_map.has_key((hkl.h, hkl.k, hkl.l)):
                            continue
                        equivalents = list(hkl.equivalentReflections())
                        equivalents.sort()
                        for r in equivalents:
                            self.reflection_map[(r.h, r.k, r.l)] = r
                        hkl = equivalents[-1]
                        if hkl.isSystematicAbsence():
                            for r in equivalents:
                                r.reflection_index = None
                                self.systematic_absences.add(r)
                        else:
                            self.minimal_reflection_list.append(hkl)
        self.minimal_reflection_list.sort()

    def __iter__(self):
        for r in self.minimal_reflection_list:
            yield r

    def __len__(self):
        return len(self.reflection_map)

    def __getitem__(self, item):
        return self.reflection_map[item]
