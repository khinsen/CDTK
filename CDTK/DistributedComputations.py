# Routines for distributed computations
#
# This file is part of the Crystallographic Data Toolkit and
# distributed under the CeCILL-C licence. See the file LICENCE
# for the full text of this licence.
#
# Written by Konrad Hinsen.
#

#import Scientific.DistributedComputing.TaskManager
#Scientific.DistributedComputing.TaskManager.debug = True
#import Scientific.DistributedComputing.MasterSlave
#Scientific.DistributedComputing.MasterSlave.debug = True

from Scientific.DistributedComputing.MasterSlave \
     import initializeMasterProcess, TaskRaisedException, GlobalStateValue
import os

tasks = initializeMasterProcess("CDTK_%d" % os.getpid(),
                                slave_module="CDTK.SlaveProcesses")
nprocs = int(os.environ.get("CDTK_DISTRIBUTED_PROCESSES", 0))
if nprocs > 0:
    tasks.launchSlaveJobs(nprocs)

def _evaluateModel_distributed(self, sf, pd, adpd, deriv):
    if not self._distribution_initialized:
        self._distributed_state_id = \
               tasks.setGlobalState(sv = self.sv,
                                    p = self.p,
                                    f_atom = self.f_atom,
                                    e_indices = self.element_indices)
        self._distribution_initialized = True
    ntasks = max(int(os.environ.get("CDTK_DISTRIBUTED_REFINEMENT_TASKS", 0)), 1)
    n = self.natoms/ntasks
    if self.natoms % ntasks > 0: n += 1
    state_id = self._distributed_state_id
    task_ids = []
    index = 0
    while index < self.natoms:
        if sf is None:
            assert deriv is not None
            if pd is None:
                # Calculate ADP derivatives from given amplitude derivatives
                assert adpd is not None
                task_ids.append(tasks.requestTask("sf_adpd", index, index+n,
                                   self.positions[index:index+n],
                                   self.adps[index:index+n],
                                   self.occupancies[index:index+n],
                                   deriv,
                                   self.structure_factor,
                                   self.model_amplitudes,
                                   GlobalStateValue(state_id, "e_indices"),
                                   GlobalStateValue(state_id, "f_atom"),
                                   GlobalStateValue(state_id, "sv"),
                                   GlobalStateValue(state_id, "p")))
            else:
                # Calculate position derivatives from given
                # amplitude derivatives
                task_ids.append(tasks.requestTask("sf_pd", index, index+n,
                                   self.positions[index:index+n],
                                   self.adps[index:index+n],
                                   self.occupancies[index:index+n],
                                   deriv,
                                   self.structure_factor,
                                   self.model_amplitudes,
                                   GlobalStateValue(state_id, "e_indices"),
                                   GlobalStateValue(state_id, "f_atom"),
                                   GlobalStateValue(state_id, "sv"),
                                   GlobalStateValue(state_id, "p")))
        else:
            # Calculate structure factors
            task_ids.append(tasks.requestTask("sf", index, index+n,
                               self.positions[index:index+n],
                               self.adps[index:index+n],
                               self.occupancies[index:index+n],
                               GlobalStateValue(state_id, "e_indices"),
                               GlobalStateValue(state_id, "f_atom"),
                               GlobalStateValue(state_id, "sv"),
                               GlobalStateValue(state_id, "p")))
        index += n
    exception = False
    while task_ids:
        try:
            task_id, tag, result = tasks.retrieveResult()
            task_ids.remove(task_id)
            if tag == "sf":
                sf += result
            elif tag == "sf_pd":
                i1, i2, term = result
                pd[i1:i2] = term
            else:
                i1, i2, term = result
                adpd[i1:i2] = term
        except TaskRaisedException, e:
            print "Task %s raised %s" % (e.task_id, str(e.exception))
            print e.traceback
            exception = True
    if exception:
        raise ValueError("Failure in distributed execution")


def _distributed_refinement_cleanup(self):
    if self._distribution_initialized:
        tasks.deleteGlobalState(self._distributed_state_id)
