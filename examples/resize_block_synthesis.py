""" Resize a 4-qubit circuit using gate dependency rule """
from bqskit.compiler import MachineModel
from bqskit.compiler.compiler import Compiler
from bqskit.compiler.task import CompilationTask
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import CXGate
from bqskit.passes import *
from resize.gatedependencyresize import GateDependencyResize
from resize.qfactor_resizable_checking import get_resizable_pairs_qfactor
from resize.qfactor_resizable_checking import reduce_block_size
from resize.block_synthesis import BlockSynthesis
from resize.utils import update_coupling_graph

import numpy as np

qc = Circuit(4)
qc.append_gate(CXGate(), (0, 3))
qc.append_gate(CXGate(), (2, 3))
qc.append_gate(CXGate(), (1, 2))
qc.append_gate(CXGate(), (0, 1))
qc.append_gate(CXGate(), (2, 3))

# Get the resizable pairs based on gate dependency rule
gate_dependency_resize = GateDependencyResize(qc)
resizable_pairs_gate_dep = gate_dependency_resize.get_resizable_qubit_pairs(qc)
print('resizable pairs based on gate dependency:', resizable_pairs_gate_dep)

# Get the resizable pairs based on instantiation (via qfactor)
resizable_pairs_qfactor = get_resizable_pairs_qfactor(qc)
print('resizable pairs based on qfactor:', resizable_pairs_qfactor)

# reduce the block size for circuit re-synthesis
resizable_pair, block_reduced = reduce_block_size(qc, resizable_pairs_qfactor)
block_1, block_2 = block_reduced[0], block_reduced[1]

initial_coupling = [(0, 1), (1, 2), (2, 3), (3, 4)]
updated_map = update_coupling_graph([resizable_pair[0]], [resizable_pair[1]], initial_coupling, qc.num_qudits)
print(updated_map)
model = MachineModel(qc.num_qudits, coupling_graph=updated_map)

workflow = [
    SetModelPass(model),
    BlockSynthesis(block_1,
                   block_2),
    UnfoldPass(),
    QuickPartitioner(block_size=3),
    ForEachBlockPass(
        ScanningGateRemovalPass(),
    ),
    UnfoldPass(),
]

task = CompilationTask(qc, workflow)
with Compiler() as compiler:
    resizable_qc = compiler.compile(task)

gate_dependency_resize2 = GateDependencyResize(resizable_qc)
resizable_pairs_gate_dep2 = gate_dependency_resize2.get_resizable_qubit_pairs(resizable_qc)
print('-----after re-synthesis----')
print('resizable pairs based on gate dependency:', resizable_pairs_gate_dep2)
resized_qc = gate_dependency_resize2.run()
print('resized circuit qubit count:', resized_qc.num_qudits)