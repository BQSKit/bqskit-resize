"""This module implements the ResizingQFactorPredicate class."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from bqskit.passes.control.predicate import PassPredicate
from bqskit.qis.graph import CouplingGraph
from .qfactor_resizable_checking import get_resizable_pairs_qfactor
from .qfactor_resizable_checking import reduce_block_size
from .utils import update_coupling_graph

if TYPE_CHECKING:
    from bqskit.compiler.passdata import PassData
    from bqskit.ir.circuit import Circuit

_logger = logging.getLogger(__name__)


class ResizingQFactorPredicate(PassPredicate):
    """Check if the circuit is resizable based on qfactor instantiation."""
    def get_truth_value(self, circuit: Circuit, data: PassData) -> bool:
        """Call this predicate, see :class:`PassPredicate` for more info."""
        resizable_qubit_pairs = get_resizable_pairs_qfactor(circuit)
        if len(resizable_qubit_pairs) == 0:
            return False
        else:
            resizable_pair, block_reduced = reduce_block_size(circuit, resizable_qubit_pairs)
            block_1, block_2 = block_reduced[0], block_reduced[1]
            initial_coupling = data.connectivity
            updated_map = update_coupling_graph([resizable_pair[0]], [resizable_pair[1]], initial_coupling,
                                                circuit.num_qudits)
            data['block1'] = block_1
            data['block2'] = block_2
            data.model.coupling_graph = CouplingGraph(updated_map)
            return True

