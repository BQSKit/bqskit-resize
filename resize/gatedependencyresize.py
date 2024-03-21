from __future__ import annotations

import numpy as np
from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import MeasurementPlaceholder
from bqskit.ir.gates import Reset
from .utils import ending_point
from .utils import starting_point
from .utils import update_mapping_list
import logging

_logger = logging.getLogger(__name__)

class GateDependencyResize(BasePass):
    """
    A quantum circuit resizing algorithm based on gate dependencies.

    References:
        Niu, Siyuan, et al. "Powerful Quantum Circuit Resizing with Resource Efficient Synthesis."
        arXiv preprint arXiv:2311.13107 (2023).
    """

    def __init__(
            self,
            # circ: Circuit,
            cost_func: str ='max_reuse',
            resizing_method: str = 'greedy',
            ) -> None:
        """
        Create a gate dependency resize object.

        Args:
            circ (Circuit): The circuit to resize.

            cost_func (str): The cost function to evaluate the resizable pair, which can
                be chosen between 'max_reuse' and 'min_depth'.
                'max_reuse' aims to resize the circuit with as many qubits as possible.
                'min_depth' aims to resize the circuit while considering the circuit depth.
                (Default: 'max_reuse')

            resizing_method: The resizing method to resize the circuit based on gate dependencies, which
                can be chosen between 'greedy' or 'bfs'.
                'greedy' picks the local optimal resized circuit.
                'bfs' stands for bread first search and picks the global optimal resized circuit but is
                computational expensive.
                (Default: 'greedy')
        """
        # self.circuit = circ
        self.cost_func = cost_func
        # Classical register to store the results from mid-circuit measurement
        # self.cregs = [('resize', self.circuit.num_qudits)]
        if resizing_method in ['greedy', 'bfs']:
            self.resizing_method = resizing_method
        else:
            raise ValueError('Invalid resizing method. Should choose between "greedy" and "bfs".')

    def get_independent_qubits(self, qubit: int, cycle_opts: dict, circuit: Circuit) -> list[int]:
        """
        Get a list of qubits that can be reused by the input qubit.
        If the list is empty, it means that we cannot reuse this qubit for any others.

        Args:
            qubit (int): check if this qubit is reusable for other qubits.
            cycle_opts (dict): The gates for each cycle in a reverse order.
            circuit (Circuit): the circuit to resize.
        """
        dependent_qubits = {qubit}
        end_point = len(cycle_opts) - 1
        for cycle in range(end_point, -1, -1):
            for opt in cycle_opts[cycle]:
                if any(q in opt.location for q in dependent_qubits):
                    dependent_qubits.update(q for q in opt.location)
        # check if the finish of this qubit depends on the finish of all the other qubits,
        # if not, we can reuse this qubit for other qubits.
        independent_qubits = [q for q in range(circuit.num_qudits) if q not in dependent_qubits]
        return independent_qubits

    def get_resizable_qubit_pairs(self, circuit: Circuit) -> dict[int, list]:
        """
        Get all the possible resizable qubit pairs for the input circuit.

        Args:
            circuit (Circuit): The input circuit to resize.
        """
        resizable_qubit_pairs = {}
        ending_points = ending_point(circuit)
        for qubit in range(circuit.num_qudits):
            # for each qubit, from its ending point to the start, reversely collecting a list of gates in the same cycle
            qubit_opt_reverse_order = {cycle: [] for cycle in range(ending_points[qubit] + 1)}
            for cycle, op in circuit.operations_with_cycles():
                if cycle <= ending_points[qubit]:
                    qubit_opt_reverse_order[cycle].append(op)
            resizable_qubit_pairs[qubit] = self.get_independent_qubits(qubit, qubit_opt_reverse_order, circuit)
        return resizable_qubit_pairs

    def cost_function(self, circuit: Circuit) -> int:
        """
        The cost of the resized circuit candidate. The lower is the better.

        Args:
            circuit (circuit): The resized circuit candidate to evaluate.
        """
        if self.cost_func == 'max_reuse':
            # indicates how many resizable pair can this resized circuit further have.
            resizale_pairs = self.get_resizable_qubit_pairs(circuit)
            num_resizale_pairs = len([item for sublist in resizale_pairs.values() for item in sublist])
            if self.resizing_method == 'greedy':
                return -num_resizale_pairs
            else:
                return circuit.num_qudits
        elif self.cost_func == 'min_depth':
            # indicates the circuit depth (only considering multi-qubit gate) for this resized circuit candidate.
            depth = circuit.multi_qudit_depth
            return depth
        else:
            raise ValueError('Invalid cost function type. Should choose between "max_reuse" and "min_depth".')

    def update_circuit(self, circuit: Circuit, q_reuse: int, q_to_use: int, target: Circuit) -> Circuit:
        """
        Resize the circuit and insert mid-circuit measurement and reset to reuse 'q_reuse' for 'q_to_use'.

        Args:
            circuit (circuit): the circuit to update.
            q_reuse (int): the qubit that we reuse.
            q_to_use (int): the qubit that is reused for.
        """
        new_circuit = Circuit(circuit.num_qudits - 1)
        partial_circuit = circuit.copy()
        mapping = {i: i for i in range(circuit.num_qudits)}
        mapping = update_mapping_list(mapping, q_reuse, q_to_use)
        resize_point = ending_point(partial_circuit)[q_reuse]
        starting_points = starting_point(partial_circuit)
        q_to_use_starting_point = starting_points[q_to_use]
        qubits_between = {q_to_use: [q_to_use_starting_point]}
        for cycle, op in partial_circuit.operations_with_cycles():
            if q_to_use_starting_point <= cycle < resize_point:
                if any(l in qubits_between for l in op.location):
                    for l in op.location:
                        if l in qubits_between.keys():
                            qubits_between[l].append(cycle)
                        else:
                            qubits_between[l] = [cycle]
        qubits_between = {key: value[0] for key, value in qubits_between.items()}
        partial_circuit_copy = partial_circuit.copy()
        for cycle, op in partial_circuit_copy.operations_with_cycles():
            if cycle <= resize_point and not any(q == q_to_use for q in op.location):
                q_between = [l for l in op.location if l in qubits_between.keys()]
                flag = False
                for q in q_between:
                    if cycle > qubits_between[q]:
                        flag = True
                if flag is True:
                    continue
                new_circuit.append_gate(op.gate, location=[mapping[i] for i in op.location], params=op.params)
                partial_circuit.remove(op)
        mph_idx = len([op for op in circuit if isinstance(op.gate, Reset)])
        cregs = [('resize', target.num_qudits)]
        mph = MeasurementPlaceholder(cregs, {mapping[q_reuse]: ('resize', mph_idx)})
        new_circuit.append_gate(mph, location=[mapping[q_reuse]])
        new_circuit.append_gate(Reset(), location=[mapping[q_reuse]])
        for op in partial_circuit:
            new_circuit.append_gate(op.gate, location=[mapping[i] for i in op.location], params=op.params)
        return new_circuit

    def greedy(self, resizable_qubit_pairs: dict[int, list], target: Circuit) -> Circuit:
        """
        A greedy algorithm to find the best resized circuit.
        For the input circuit, during each iteration, we only reuse one qubit (i.e., insert one MMR). The greedy algorithm
        picks the locally best resized circuit for the next round of resizing. The process is repeated until we cannot
        find other resizing possibilities.

        Args:
                resizable_qubit_pairs (dict): the possible resizable pairs for the input circuit to resize.
        """
        # The circuit with the smallest cost is preferable. So we start the initial cost to the infinitive.
        best_cost = np.Inf
        # Some circuits might have the same cost values. We store them in a list and randomly pick one for the next round.
        best_circuits = []
        circuit = target.copy()
        best_circ = target.copy()
        while any(value for value in resizable_qubit_pairs.values()):
            for q_reuse, qs_to_use in resizable_qubit_pairs.items():
                for q_to_use in qs_to_use:
                    # For each resizable pair, we update the circuit by reusing qubit and inserting one MMR
                    update_cir = self.update_circuit(circuit, q_reuse, q_to_use, target)
                    cost = self.cost_function(update_cir)
                    if cost < best_cost:
                        best_cost = cost
                        best_circuits = [update_cir]
                    elif cost == best_cost:
                        best_circuits.append(update_cir)
            # Randomly pick up a circuit from the list of best circuits with the same cost
            best_circ = best_circuits[np.random.randint(len(best_circuits))]
            resizable_qubit_pairs = self.get_resizable_qubit_pairs(best_circ)
            circuit = best_circ
            # If the best circuit is still resizable, we start a new round of resizing.
            if any(value for value in resizable_qubit_pairs.values()):
                best_cost = np.Inf
                best_circuits = []
        return best_circ

    def bfs(self, resizable_qubit_pairs: dict[int, list], target: Circuit) -> Circuit:
        """
             A breath first search algorithm to find the best resized circuit.
             For the input circuit, we explore all the possible resizing candidates and pick the best circuit.

             Args:
                     resizable_qubit_pairs (dict): the possible resizable pairs for the input circuit to resize.
        """
        # Queue of nodes to visit, each node is a tuple (circuit, reused_qubits, q_reuse, q_to_use)
        queue = [(target, resizable_qubit_pairs, None, None)]
        best_cir = target.copy()
        best_cost = np.inf
        current_cost = np.Inf
        while queue:
            current_cir, current_reused_qubits, current_q_reuse, current_q_to_use = queue.pop(0)  # Dequeue a node
            if not any(value for value in current_reused_qubits.values()):
                if current_q_reuse is not None:
                    current_cost = self.cost_function(current_cir)
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_cir = current_cir
            else:
                # If the circuit is already resizable, we add all the possible resizable candidates to the queue
                for q_reuse, qs_to_use in current_reused_qubits.items():
                    for q_to_use in qs_to_use:
                        # add mid-circuit measurement and reset
                        new_cir = self.update_circuit(current_cir, q_reuse, q_to_use, target)
                        # get the new resetable qubit from the updated circuit
                        new_reused_qubits = self.get_resizable_qubit_pairs(new_cir)
                        queue.append((new_cir, new_reused_qubits, q_reuse, q_to_use))  # Enqueue the new node
        return best_cir

    async def run(self, circuit: Circuit, data: PassData) -> None:
        input_circuit = circuit.copy()
        resizable_qubit_pairs = self.get_resizable_qubit_pairs(input_circuit)
        if self.resizing_method == 'greedy':
            resized_circuit = self.greedy(resizable_qubit_pairs, input_circuit)
        else:
            resized_circuit = self.bfs(resizable_qubit_pairs, input_circuit)
        circuit.become(resized_circuit)