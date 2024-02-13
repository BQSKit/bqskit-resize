""" Resize a 10-qubit Bernstein Vazirani (BV) circuit using gate dependency rule """

from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import HGate
from bqskit.ir.gates import ZGate
from bqskit.ir.gates import CXGate

from resize.gatedependencyresize import GateDependencyResize

# Create a 10-qubit BV circuit
qc = Circuit(10)
for i in range(9):
    qc.append_gate(HGate(), i)
qc.append_gate(ZGate(), 9)
for i in range(9):
    qc.append_gate(CXGate(), (i, 9))
for i in range(10):
    qc.append_gate(HGate(), i)

gate_dep = GateDependencyResize(qc)
# The resized circuit
resized_qc = gate_dep.run()

print('original qudit number:', qc.num_qudits)
print('original circuit depth:', qc.multi_qudit_depth)

print('resized qudit number:', resized_qc.num_qudits)
print('resized circuit depth:', resized_qc.multi_qudit_depth)


