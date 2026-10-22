"""Run tscompare's unchanged ARF calculation with exact integer clade bitsets.

The stock CladeMap represents each descendant sample set as a frozenset. On
large cohorts its repeated union, difference, hashing and equality operations
dominate runtime. Python integers encode the same sets exactly, with sample i
represented by bit i. Only the clade iterator's representation changes; span
matching, tie breaking and metric denominators remain tscompare's own code.
"""

import collections
import logging

import tscompare
import tscompare.methods
import tskit

logger = logging.getLogger(__name__)


class BitsetCladeMap(tscompare.methods.CladeMap):
    """Use exact bitsets with the existing edge-difference iterator contract."""

    def __init__(self, ts):
        self._nil = 0
        self._nodes = collections.defaultdict(set)
        self._clades = collections.defaultdict(int)
        self.tree_sequence = ts
        self.tree = ts.first(sample_lists=True)
        for node in self.tree.nodes():
            clade = 0
            for sample in self.tree.samples(node):
                clade |= 1 << int(sample)
            self._nodes[clade].add(node)
            self._clades[node] = clade
        self._prev = self._clades.copy()
        self._diff = ts.edge_diffs()
        next(self._diff)

    def _propagate(self, edge, downdate=False):
        nodes = set()
        node = edge.parent
        clade = self._clades[edge.child]
        while node != tskit.NULL:
            last = self._clades[node]
            updated = last & ~clade if downdate else last | clade
            self._clades[node] = updated
            if last != 0:
                self._nodes[last].remove(node)
                if len(self._nodes[last]) == 0:
                    del self._nodes[last]
            self._nodes[updated].add(node)
            nodes.add(node)
            node = self.tree.parent(node)
        return nodes

    def clades(self):
        return self._nodes.keys() - {self._nil}

    def __getitem__(self, clade):
        if clade in self:
            return frozenset(self._nodes[clade])
        return frozenset()


def haplotype_arf(comparison, reference):
    """Temporarily use the exact bitset iterator in this single-threaded worker.

    Each pt7 comparison runs in an independent process. Restore the library
    class even on failure, so other callers retain tscompare's normal behavior.
    """
    original = tscompare.methods.CladeMap
    tscompare.methods.CladeMap = BitsetCladeMap
    try:
        return tscompare.haplotype_arf(comparison, reference)
    finally:
        tscompare.methods.CladeMap = original
