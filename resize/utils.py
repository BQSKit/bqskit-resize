"""This module contains various utility functions for quantum circuit resizing algorithms."""
from __future__ import annotations
from bqskit.ir.circuit import Circuit

import logging
_logger = logging.getLogger(__name__)


def get_independent_qubits(qubit: int, cycle_opts: dict, circuit: Circuit) -> list[int]:
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

def get_resizable_qubit_pairs(circuit: Circuit) -> dict[int, list]:
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
        resizable_qubit_pairs[qubit] = get_independent_qubits(qubit, qubit_opt_reverse_order, circuit)
    return resizable_qubit_pairs

def ending_point(circuit: Circuit) -> dict[int, int]:
    """
    The ending circle of all the qubits from the input circuit.

    Args:
        circuit (Circuit): the evaluated circuit.
    """
    qubit_ending_points = {i: 0 for i in range(circuit.num_qudits)}
    for cycle, op in circuit.operations_with_cycles():
        for l in op.location:
            qubit_ending_points[l] = cycle
    return qubit_ending_points

def starting_point(circuit: Circuit) -> dict[int, int]:
    """
    The starting circle of all the qubits from the input circuit.

    Args:
        circuit (Circuit): the evaluated circuit.
    """
    qubit_starting_points = {}
    for cycle, op in circuit.operations_with_cycles():
        for l in op.location:
            if l not in qubit_starting_points.keys():
                qubit_starting_points[l] = cycle
    for i in range(circuit.num_qudits):
        if i not in qubit_starting_points.keys():
            qubit_starting_points[i] = 0
    return qubit_starting_points

def update_mapping_list(mapping: dict, q_reuse: int, q_to_use:int) -> dict[int, int]:
    """
    Update the mapping between the index of the current circuit and the resized circuit.
    """
    updated_mapping = mapping.copy()
    updated_mapping[q_to_use] = updated_mapping[q_reuse]
    values = sorted(list(set(updated_mapping.values())))
    scaled_values = {v: i for i, v in enumerate(values)}
    return {k: scaled_values[v] for k, v in updated_mapping.items()}

def update_coupling_graph(
        qs_reuse: list,
        qs_to_use: list,
        initial_coupling: list,
        num_qudits: int,
) -> list:
    """
    Update the coupling graph of the hardware based on resized qubits using reverse design.

    Args:
        qs_reuse (list): a list of qubits to reuse.
        qs_to_use (list): a list of qubits that are reused for.
        initial_coupling (list): the initial coupling of the target hardware.
        num_qudits (int): the number of qudits for the input circuit.
    """
    new_coupling = []
    initial_mapping = {i: i for i in range(num_qudits)}
    new_mapping = initial_mapping.copy()
    for q_reuse, q_to_use in zip(qs_reuse, qs_to_use):
        new_mapping = update_mapping_list(new_mapping, q_reuse, q_to_use)
    new_mapping_inverse = {i: [] for i in range(num_qudits - len(qs_to_use))}
    for key, value in new_mapping.items():
        new_mapping_inverse[value].append(key)

    for (i, j) in initial_coupling:
        if i not in new_mapping_inverse.keys() or j not in new_mapping_inverse.keys():
            continue
        for p in new_mapping_inverse[i]:
            for q in new_mapping_inverse[j]:
                new_coupling.append(sorted((p, q)))
    new_coupling = [tuple(pair) for pair in new_coupling]
    return new_coupling