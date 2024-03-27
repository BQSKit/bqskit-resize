""" Resize a 10-qubit Bernstein Vazirani (BV) circuit using gate dependency rule """

from bqskit.compiler.compiler import Compiler
from bqskit.compiler.task import CompilationTask
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import HGate
from bqskit.ir.gates import ZGate
from bqskit.ir.gates import CXGate

from resize.gatedependencyresize import GateDependencyResize

# Create a 10-qubit BV circuit
qc = Circuit(4)
for i in range(3):
    qc.append_gate(HGate(), i)
qc.append_gate(ZGate(), 3)
for i in range(3):
    qc.append_gate(CXGate(), (i, 3))
for i in range(4):
    qc.append_gate(HGate(), i)

workflow = [
    GateDependencyResize()
]

task = CompilationTask(qc, workflow)
with Compiler() as compiler:
    resized_qc = compiler.compile(task)


print('original qudit number:', qc.num_qudits)
print('original circuit depth:', qc.multi_qudit_depth)

print('resized qudit number:', resized_qc.num_qudits)
print('resized circuit depth:', resized_qc.multi_qudit_depth)





