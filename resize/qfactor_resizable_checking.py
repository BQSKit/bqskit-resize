from __future__ import annotations

from bqskit.ir import Circuit
from bqskit.ir.gates import VariableUnitaryGate
from itertools import combinations
import logging
import multiprocessing
import numpy as np

_logger = logging.getLogger(__name__)


def resizable_pair_checking(args) -> tuple[int, int] | None:
    """
    Check if the circuit can reuse `q_reuse` for `q_to_use` for resizing via instantiation (qFactor).

    Args:
        q_reuse (int): the qubit to reuse.
        q_to_use (int): the qubit that is reused for.
        qc (Circuit): the circuit to resize.
        threshold (float): if the circuit is resizable by this qubit pair, the Hilbert-Schmidt distance between the
        instantiated circuit and the input circuit should be below the threshold.
    """
    q_reuse, q_to_use, qc, threshold = args
    new_circuit = Circuit(qc.num_qudits)
    q_no_reuse_to_use = [q for q in range(qc.num_qudits) if q != q_reuse and q != q_to_use]
    opt1_qubits = q_no_reuse_to_use.copy()
    opt1_qubits.append(q_reuse)
    opt1_qubits.sort()
    opt2_qubits = q_no_reuse_to_use.copy()
    opt2_qubits.append(q_to_use)
    opt2_qubits.sort()
    new_circuit.append_gate(VariableUnitaryGate(qc.num_qudits - 1), opt1_qubits)
    new_circuit.append_gate(VariableUnitaryGate(qc.num_qudits - 1), opt2_qubits)

    new_circuit.instantiate(
        qc.get_unitary(),
        method='qfactor',
        diff_tol_a=1e-12,  # Stopping criteria for distance change
        diff_tol_r=1e-6,  # Relative criteria for distance change
        dist_tol=1e-12,  # Stopping criteria for distance
        max_iters=100000,  # Maximum number of iterations
        min_iters=1000,  # Minimum number of iterations
        slowdown_factor=0,  # Larger numbers slowdown optimization to avoid local minima
    )

    dist = new_circuit.get_unitary().get_distance_from(qc.get_unitary(), 1)
    if dist < threshold:
        return q_reuse, q_to_use
    return None

def get_resizable_pairs_qfactor(qc: Circuit, threshold: float = 1e-10, num_cpus: int = None) -> list:
    """
    For input n-qubit circuit, we evaluate all the qubit pairs using multiprocessing, which is n(n-1) in total,
    to check the resizability of the qubit pair via instantiation.

    Args:
        qc (Circuit): the input circuit to resize.
        threshold (float): if the circuit is resizable by this qubit pair, the Hilbert-Schmidt distance between the
        instantiated circuit and the input circuit should be below the threshold.
        num_cpus (int): the number of cpus allocated by the user to process the resizable pair checking in parallel.
    """
    if num_cpus is None:
        # Use half of the available CPUs, but at least two
        num_processors = max(2, multiprocessing.cpu_count() // 2)
    else:
        # Ensure the user-specified number of CPUs does not exceed the available CPUs
        available_cpus = multiprocessing.cpu_count()
        # At least 1 CPU, and at most the number of available CPUs
        num_processors = min(max(1, num_cpus), available_cpus)
    pairs_to_process = [(q_reuse, q_to_use, qc, threshold) for q_reuse in range(qc.num_qudits) for q_to_use in
                        range(qc.num_qudits) if q_reuse != q_to_use]

    # Initialize multiprocessing Pool
    pool = multiprocessing.Pool(processes=num_processors)
    results = pool.map(resizable_pair_checking, pairs_to_process)
    pool.close()
    pool.join()
    # Filter out None results, which means the qubit pair is not resizable
    resizable_pairs = [result for result in results if result is not None]
    return resizable_pairs

def get_blocks(qs_to_use: list, qs_reuse: list, num_qudits: int) -> (list, list):
    """
    Obtain the two blocks with the qubits included.
    """
    q_block1 = list(range(num_qudits))
    q_block2 = q_block1.copy()
    for q_to_use in qs_to_use:
        q_block1.remove(q_to_use)
    for q_reuse in qs_reuse:
        q_block2.remove(q_reuse)
    q_block1.sort()
    q_block2.sort()
    return q_block1, q_block2

def process_reduce_blocks(args) -> tuple | None:
    """
    Checking if the circuit with reduced sized of blocks can represent the target unitary.
    """
    q_subblock1, q_subblock2, qc, best_block_size, threshold = args
    if len(q_subblock1) + len(q_subblock2) <= best_block_size:
        new_circuit = Circuit(qc.num_qudits)
        new_circuit.append_gate(VariableUnitaryGate(len(q_subblock1)), q_subblock1)
        new_circuit.append_gate(VariableUnitaryGate(len(q_subblock2)), q_subblock2)

        # Instantiate the new circuit (assuming this is a blocking, CPU-bound operation)
        new_circuit.instantiate(
            qc.get_unitary(),
            method='qfactor',
            diff_tol_a=1e-12,  # Stopping criteria for distance change
            diff_tol_r=1e-6,  # Relative criteria for distance change
            dist_tol=1e-12,  # Stopping criteria for distance
            max_iters=100000,  # Maximum number of iterations
            min_iters=1000,  # Minimum number of iterations
            slowdown_factor=0,  # Larger numbers slowdown optimization
        )

        dist = new_circuit.get_unitary().get_distance_from(qc.get_unitary(), 1)
        if dist < threshold:
            return len(q_subblock1) + len(q_subblock2), [q_subblock1, q_subblock2]
    return None

def reduce_block_size(qc: Circuit, resize_pairs: list, threshold: float = 1e-10, num_cpus: int = None) -> (tuple, list):
    """
    Reduce the size of the blocks for the resizable-checking circuit to mitigate the overhead for block unitary
    synthesis process.

    Args:
        qc (Circuit): the circuit to resize.
        resize_pairs (list): the resizable pairs that can be reused for resizing.
        threshold (float): the threshold to guarantee the Hilbert-Schmidt distance between two circuits.
    """
    reduced_blocks = {}
    best_block_size = (qc.num_qudits - 1) * 2
    for pair in resize_pairs:
        reduced_blocks[pair] = []
        q_block1, q_block2 = get_blocks([pair[1]], [pair[0]], qc.num_qudits)
        q_subblock1s = []
        q_subblock2s = []
        # The smallest size of the block is set to two.
        for l in range(2, len(q_block1) + 1):
            for b1 in combinations(q_block1, l):
                if pair[0] in b1:
                    q_subblock1s.append(b1)
        for l in range(2, len(q_block2) + 1):
            for b2 in combinations(q_block2, l):
                if pair[1] in b2:
                    q_subblock2s.append(b2)

        tasks = [(q_subblock1, q_subblock2, qc, best_block_size, threshold) for q_subblock1 in q_subblock1s for
                 q_subblock2 in q_subblock2s]

        if num_cpus is None:
            # Use half of the available CPUs, but at least two
            num_processors = max(2, multiprocessing.cpu_count() // 2)
        else:
            # Ensure the user-specified number of CPUs does not exceed the available CPUs
            available_cpus = multiprocessing.cpu_count()
            # At least 1 CPU, and at most the number of available CPUs
            num_processors = min(max(1, num_cpus), available_cpus)

        with multiprocessing.Pool(processes=num_processors) as pool:
            results = pool.map(process_reduce_blocks, tasks)

        # Filter out None results and update best_block_size if necessary
        for result in filter(None, results):
            block_size, blocks = result
            if block_size <= best_block_size:
                reduced_blocks[pair].append(blocks)
                best_block_size = block_size
    # filter all the blocks that are equal to the best block size
    filtered_blocks = {
        key: [block for block in value if sum(len(t) for t in block) == best_block_size]
        for key, value in reduced_blocks.items()
    }
    filtered_blocks = {k: v for k, v in filtered_blocks.items() if v}
    keys_list = list(filtered_blocks.keys())
    random_resizable_pair = keys_list[np.random.randint(len(keys_list))]
    random_correspond_block = filtered_blocks[random_resizable_pair][np.random.randint(len(filtered_blocks[random_resizable_pair]))]

    return random_resizable_pair, random_correspond_block