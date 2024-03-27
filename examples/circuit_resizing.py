""" Resize a 4-qubit circuit using gate dependency rule """
from bqskit.compiler import MachineModel
from bqskit.compiler.compiler import Compiler
from bqskit.compiler.task import CompilationTask
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import CXGate
from bqskit.ir.gates import HGate
from bqskit.ir.gates import ZGate
from bqskit.passes import *
from resize.gatedeppredicate import ResizingGateDependencyPredicate
from resize.qfactorpredicate import ResizingQFactorPredicate
from resize.gatedependencyresize import GateDependencyResize
from resize.blocklayer import BlockLayerGenerator
import logging

qc1 = Circuit(4)
for i in range(3):
    qc1.append_gate(HGate(), i)
qc1.append_gate(ZGate(), 3)
for i in range(3):
    qc1.append_gate(CXGate(), (i, 3))
for i in range(4):
    qc1.append_gate(HGate(), i)

qc2 = Circuit(4)
qc2.append_gate(CXGate(), (0, 3))
qc2.append_gate(CXGate(), (2, 3))
qc2.append_gate(CXGate(), (1, 2))
qc2.append_gate(CXGate(), (0, 1))
qc2.append_gate(CXGate(), (2, 3))


initial_coupling = [(0, 1), (1, 2), (2, 3)]
model = MachineModel(qc1.num_qudits, coupling_graph=initial_coupling)

workflow = [
    SetModelPass(model),
    IfThenElsePass(
        ResizingGateDependencyPredicate(),
        GateDependencyResize(),
        IfThenElsePass(
            ResizingQFactorPredicate(),
            [LEAPSynthesisPass(
                layer_generator=BlockLayerGenerator(),
            ),
            UnfoldPass(),
            QuickPartitioner(block_size=3),
            ForEachBlockPass(
                ScanningGateRemovalPass(),
            ),
            UnfoldPass(),
            GateDependencyResize(),
            ],
            LogPass(
                'Unable to resize the circuit;',
                logging.WARNING,
            ),
        )
    )
]

task1 = CompilationTask(qc1, workflow)
task2 = CompilationTask(qc2, workflow)
with Compiler() as compiler:
    resized_qc1 = compiler.compile(task1)
    resized_qc2 = compiler.compile(task2)

print('resized qc1 qubit count:', resized_qc1.num_qudits)
print('resized qc2 qubit count:', resized_qc2.num_qudits)