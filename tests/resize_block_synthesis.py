""" Resize a 4-qubit circuit using gate dependency rule """
from bqskit.compiler import MachineModel
from bqskit.compiler.compiler import Compiler
from bqskit.compiler.task import CompilationTask
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import CXGate
from bqskit.passes import *
from resize.qfactor_resizable_checking import get_resizable_pairs_qfactor
from resize.qfactor_resizable_checking import reduce_block_size
from resize.gatedependencyresize import GateDependencyResize
from resize.blocklayer import BlockLayerGenerator
from resize.utils import update_coupling_graph


qc = Circuit(4)
qc.append_gate(CXGate(), (0, 3))
qc.append_gate(CXGate(), (2, 3))
qc.append_gate(CXGate(), (1, 2))
qc.append_gate(CXGate(), (0, 1))
qc.append_gate(CXGate(), (2, 3))

print('original qudit number:', qc.num_qudits)

workflow = [
    GateDependencyResize()
]

task = CompilationTask(qc, workflow)
with Compiler() as compiler:
    resized_qc = compiler.compile(task)

print('resized circuit based on gate dependency:', resized_qc.num_qudits)

# Get the resizable pairs based on instantiation (via qfactor)
resizable_pairs_qfactor = get_resizable_pairs_qfactor(qc)
print('resizable pairs based on qfactor:', resizable_pairs_qfactor)

# reduce the block size for circuit re-synthesis
resizable_pair, block_reduced = reduce_block_size(qc, resizable_pairs_qfactor)
block_1, block_2 = block_reduced[0], block_reduced[1]

initial_coupling = [(0, 1), (1, 2), (2, 3)]

updated_map = update_coupling_graph([resizable_pair[0]], [resizable_pair[1]], initial_coupling, qc.num_qudits)
model = MachineModel(qc.num_qudits, coupling_graph=updated_map)

workflow2 = [
    SetModelPass(model),
    UpdateDataPass(key='block1', val=block_1),
    UpdateDataPass(key='block2', val=block_2),
    LEAPSynthesisPass(
        layer_generator=BlockLayerGenerator(),
    ),
    UnfoldPass(),
    QuickPartitioner(block_size=3),
    ForEachBlockPass(
        ScanningGateRemovalPass(),
    ),
    UnfoldPass(),
]

task = CompilationTask(qc, workflow2)
with Compiler() as compiler:
    resizable_qc = compiler.compile(task)


task = CompilationTask(resizable_qc, workflow)
with Compiler() as compiler:
    resized_qc = compiler.compile(task)


print('resized circuit qubit count:', resized_qc.num_qudits)