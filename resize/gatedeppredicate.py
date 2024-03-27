"""This module implements the ResizingGateDependencyPredicate class."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from bqskit.passes.control.predicate import PassPredicate
from .utils import get_resizable_qubit_pairs

if TYPE_CHECKING:
    from bqskit.compiler.passdata import PassData
    from bqskit.ir.circuit import Circuit

_logger = logging.getLogger(__name__)


class ResizingGateDependencyPredicate(PassPredicate):
    """Check if the circuit is resizable based on gate dependency."""
    def get_truth_value(self, circuit: Circuit, data: PassData) -> bool:
        """Call this predicate, see :class:`PassPredicate` for more info."""
        resizable_qubit_pairs = get_resizable_qubit_pairs(circuit)
        num_resizable_pairs = len([item for sublist in resizable_qubit_pairs.values() for item in sublist])
        if num_resizable_pairs == 0:
            return False
        else:
            return True

