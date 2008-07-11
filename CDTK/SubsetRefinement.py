# Structure refinement using representative atoms
#
# This file is part of the Crystallographic Data Toolkit and
# distributed under the CeCILL-C licence. See the file LICENCE
# for the full text of this licence.
#
# Written by Konrad Hinsen.
#

"""
Structure refinement using a subset of representative atoms
"""

from CDTK.Refinement import RefinementEngine, \
                            AtomDataArray, AtomPositionDataArray
from CDTK.Utility import symmetricTensorRotationMatrix
from Scientific.Geometry import Vector, Tensor
from Scientific import N, LA
import itertools

class AtomSubsetRefinementEngine(RefinementEngine):

    """
    RefinementEngine for an atom subset
    
    An AtomSubsetRefinementEngine works as a driver for an all-atom
    RefinementEngine. It works on a subset of the atoms (e.g. the
    C-alpha atoms of a protein). Every change of the parameters of these
    atoms is extended to the remaining atoms by an interpolation scheme.
    The idea behind this approach is that both atom displacements and atomic
    ADPs are slowly-varying functions of position when considered at low
    resolutions.
    
    A typical refinement protocol based on this class would be the
    following:

     1. Create a RefinementEngine for the all-atom refinement task.
     2. Create an AtomSubsetRefinementEngine for a suitable atom subset.
     3. Refine at the subset level.
     4. Refine at the all-atom level using the all-atom RefinementEngine.

    The interpolation scheme for atom displacements and atomic ADPs assigns
    values to each atom that is not in the subset based on the values for
    the atoms that are in the subset as well as their images constructed by
    applying the symmetry operations of the space group. The weight of each
    atom in the interpolation is a function of its minimum-image distance
    vector from the target atom. The default weights are given by the
    inverse of the squared length of the distance vector.
    """

    def __init__(self, all_atom_refinement_engine, subset_atom_ids,
                 distance_power=2):
        """
        @param all_atom_refinement_engine: the underlying all-atom
                                           refinement engine
        @type all_atom_refinement_engine: L{CDTK.Refinement.RefinementEngine}
        @param subset_atom_ids: the ids of the atoms that are part of the
                                subset to be used in refinement
        @type subset_atom_ids: sequence
        @param distance_power: the power of the length of the distance
                               vector in the calculation of the weight
        @type distance_power: C{int}
        """
        assert isinstance(all_atom_refinement_engine, RefinementEngine)
        self.re = all_atom_refinement_engine
        self.ids = subset_atom_ids
        self.power = distance_power
        try:
            self.aa_indices = [self.re.id_dict[id] for id in self.ids]
        except KeyError, e:
            raise ValueError("Atom %s is not used in refinement engine %s"
                             % (str(e.args[0]),
                                str(all_atom_refinement_engine)))
        self.natoms = len(self.ids)
        self.id_dict = dict(zip(self.ids, range(self.natoms)))
        self.positions = N.take(self.re.positions, self.aa_indices)
        self.position_updates = 0.*self.positions
        self.adps = N.take(self.re.adps, self.aa_indices)
        self.occupancies = N.take(self.re.occupancies, self.aa_indices)

        sg = self.re.reflection_set.space_group
        cell = self.re.reflection_set.cell
        symops = cell.cartesianCoordinateSymmetryOperations(sg)
        unit_cell_subset = []
        for id in self.ids:
            for tr in symops:
                index = self.id_dict[id]
                rm = tr.tensor.array
                rmt = symmetricTensorRotationMatrix(rm)
                unit_cell_subset.append((index, rm, rmt,
                                         tr(Vector(self.positions[index]))))

        self.position_interpolation = N.zeros((self.re.natoms, 3,
                                               self.natoms, 3), N.Float)
        self.adp_interpolation = N.zeros((self.re.natoms, 6,
                                          self.natoms, 6), N.Float)
        for i in range(self.re.natoms):
            if i in self.aa_indices:
                si = self.aa_indices.index(i)
                for k in range(3):
                    self.position_interpolation[i, k, si, k] = 1.
                for k in range(6):
                    self.adp_interpolation[i, k, si, k] = 1.
            else:
                nb = []
                r = Vector(self.re.positions[i])
                for sindex, rot_v, rot_adp, position in unit_cell_subset:
                    d = cell.minimumImageDistanceVector(r, position)
                    nb.append((d, sindex, rot_v, rot_adp))
                self._interpolate(i, nb)
        self.position_interpolation.shape = (3*self.re.natoms, 3*self.natoms)
        self.adp_interpolation.shape = (6*self.re.natoms, 6*self.natoms)

        self.state_valid = False

    def _interpolate(self, aa_index, atom_data):
        atom_data = [(1./d[0].length()**self.power,) + d[1:]
                     for d in atom_data]
        total_weight = sum(d[0] for d in atom_data)
        for w, sindex, rot_v, rot_adp in atom_data:
            w /= total_weight
            self.position_interpolation[aa_index, :, sindex, :] += w*rot_v
            self.adp_interpolation[aa_index, :, sindex, :] += w*rot_adp

    def setPosition(self, atom_id, position):
        # Redefine setPosition to keep track of the change in position in
        # addition to the absolute value, because interpolation is done on the
        # position change vectors.
        index = self.id_dict[atom_id]
        self.position_updates[index] += position.array-self.positions[index]
        self.positions[index] = position.array
        self.state_valid = False

    def _updateInternalState(self):
        dr = N.dot(self.position_interpolation,
                   N.reshape(self.position_updates, (3*self.natoms,)))
        self.re.positions += N.reshape(dr, self.re.positions.shape)
        dadp = N.dot(self.adp_interpolation,
                     N.reshape(self.adps, (6*self.natoms,)))
        self.re.adps = N.reshape(dadp, self.re.adps.shape)
        self.position_updates[:] = 0.
        self.re.state_valid = False

    def rFactors(self):
        self.updateInternalState()
        return self.re.rFactors()

    def targetFunction(self):
        self.updateInternalState()
        return self.re.targetFunction()

    def targetFunctionAndAmplitudeDerivatives(self):
        self.updateInternalState()
        return self.re.targetFunctionAndAmplitudeDerivatives()

    def targetFunctionAndPositionDerivatives(self):
        self.updateInternalState()
        target, pd = self.re.targetFunctionAndPositionDerivatives()
        pd = N.reshape(N.dot(N.transpose(self.position_interpolation),
                             N.reshape(pd.array, (3*self.re.natoms,))),
                       (self.natoms, 3))
        return target, AtomPositionDataArray(self, pd)

    def targetFunctionAndADPDerivatives(self):
        self.updateInternalState()
        target, adpd = self.re.targetFunctionAndADPDerivatives()
        adpd = N.reshape(N.dot(N.transpose(self.adp_interpolation),
                               N.reshape(adpd.array, (6*self.re.natoms,))),
                         (self.natoms, 6))
        return target, AtomDataArray(self, adpd)
