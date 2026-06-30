"""Tests for src/qldpc/circuits/surgery/hmatrix/PPM_joint_cellulation.py."""

from __future__ import annotations

import numpy as np
import pytest


def test_skip_tree_fullrank_on_K4_matches_H_R() -> None:
    """SkipTree full-rank: T_ind · G · P_ind = H_R for the complete graph K_4."""
    import networkx as nx

    from qldpc.circuits.surgery.hmatrix.PPM_joint_cellulation import (
        _canonical_H_R,
        _skip_tree_fullrank,
    )

    G_nx = nx.complete_graph(4)
    n = 4
    edges = sorted(tuple(sorted(e)) for e in G_nx.edges())
    edge_index_verts = {e: i for i, e in enumerate(edges)}
    G_mat = np.zeros((len(edges), n), dtype=np.int_)
    for (u, v), i in edge_index_verts.items():
        G_mat[i, u] = 1
        G_mat[i, v] = 1

    T_ind, P_ind = _skip_tree_fullrank(G_nx, root=0, edge_index_verts=edge_index_verts)
    H_R = _canonical_H_R(n)

    assert T_ind.shape == (n - 1, len(edges))
    assert P_ind.shape == (n, n)
    # SkipTree key identity: T_ind · G · P_ind == H_R over GF(2)
    product = (T_ind @ G_mat @ P_ind) % 2
    assert np.array_equal(product, H_R), f"got\n{product}\nwant\n{H_R}"
    # Paper Theorem 7: (3,2)-sparsity is a general invariant of SkipTree.
    assert T_ind.sum(axis=1).max() <= 3
    assert T_ind.sum(axis=0).max() <= 2


def test_build_aux_graph_weight2_rows_become_edges() -> None:
    """F rows of weight 2 → graph edges; vertex set = {0, ..., |V_0|-1}."""
    from qldpc.circuits.surgery.hmatrix.PPM_joint_cellulation import _build_aux_graph_strict

    incidence = np.array([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]], dtype=np.uint8)
    G_nx, edge_idx = _build_aux_graph_strict(incidence)
    assert set(G_nx.nodes) == {0, 1, 2, 3}
    assert set(tuple(sorted(e)) for e in G_nx.edges) == {(0, 1), (1, 2), (2, 3)}
    assert edge_idx[(0, 1)] == 0
    assert edge_idx[(1, 2)] == 1
    assert edge_idx[(2, 3)] == 2


def test_build_aux_graph_filters_hyperedges() -> None:
    """F rows of weight >= 3 (hyperedges) are silently skipped; weight-2 rows survive."""
    from qldpc.circuits.surgery.hmatrix.PPM_joint_cellulation import _build_aux_graph_strict

    incidence = np.array(
        [
            [1, 1, 0, 0, 0],  # weight-2 → edge (0,1)
            [1, 1, 1, 1, 0],  # weight-4 hyperedge → skipped
            [0, 0, 1, 1, 0],  # weight-2 → edge (2,3)
            [0, 0, 0, 1, 1],  # weight-2 → edge (3,4)
        ],
        dtype=np.uint8,
    )
    G_nx, edge_idx = _build_aux_graph_strict(incidence)
    assert set(G_nx.nodes) == {0, 1, 2, 3, 4}
    # Three weight-2 rows → three edges; hyperedge row contributes nothing
    assert G_nx.number_of_edges() == 3
    assert (0, 1) in edge_idx
    assert (2, 3) in edge_idx
    assert (3, 4) in edge_idx
    # Hyperedge would have produced edges (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
    # but only edges from weight-2 rows are present
    assert (0, 2) not in edge_idx
    assert (0, 3) not in edge_idx
    assert (1, 3) not in edge_idx


def test_build_aux_graph_rejects_weight1_row() -> None:
    """F rows of weight 1 raise ValueError (dangling edge / no-op stabilizer)."""
    from qldpc.circuits.surgery.hmatrix.PPM_joint_cellulation import _build_aux_graph_strict

    incidence = np.array([[1, 1, 0, 0], [0, 0, 1, 0]], dtype=np.uint8)
    with pytest.raises(ValueError, match=r"weight 1"):
        _build_aux_graph_strict(incidence)


def test_connect_induced_subgraph_no_op_when_connected() -> None:
    """If induced subgraph is already connected, no edges are added."""
    import networkx as nx

    from qldpc.circuits.surgery.hmatrix.PPM_joint_cellulation import _connect_induced_subgraph

    G_aux = nx.path_graph(4)  # 0-1-2-3
    extra = _connect_induced_subgraph(G_aux, ports=(0, 1, 2, 3))
    assert extra == []
    assert set(tuple(sorted(e)) for e in G_aux.edges) == {(0, 1), (1, 2), (2, 3)}


def test_connect_induced_subgraph_adds_edges_to_disconnected_components() -> None:
    """Disconnected induced subgraph gets one bridging edge per missing connection."""
    import networkx as nx

    from qldpc.circuits.surgery.hmatrix.PPM_joint_cellulation import _connect_induced_subgraph

    # G_aux: 0-1   2-3 (two separate components)
    G_aux = nx.Graph()
    G_aux.add_edges_from([(0, 1), (2, 3)])
    extra = _connect_induced_subgraph(G_aux, ports=(0, 1, 2, 3))
    assert len(extra) == 1  # exactly one bridge needed
    (u, v) = extra[0]
    # Endpoints must come from different original components
    assert {u, v} & {0, 1} and {u, v} & {2, 3}
    # G_aux mutated: induced subgraph now connected
    assert nx.is_connected(G_aux.subgraph((0, 1, 2, 3)))


def test_cellulate_caps_cycle_length() -> None:
    """After cellulation, every basis cycle has length <= cap."""
    import networkx as nx

    from qldpc.circuits.surgery.hmatrix.PPM_joint_cellulation import _cellulate_port_subgraph

    # 10-cycle: 0-1-2-...-9-0 has one length-10 basis cycle
    G_aux = nx.cycle_graph(10)
    added = _cellulate_port_subgraph(G_aux, ports=tuple(range(10)), max_len=6)
    assert len(added) >= 1
    # All basis cycles now bounded
    sub = G_aux.subgraph(tuple(range(10)))
    assert max((len(c) for c in nx.cycle_basis(sub)), default=0) <= 6


def test_cellulate_no_op_when_already_short() -> None:
    """If all basis cycles are short, no edges are added."""
    import networkx as nx

    from qldpc.circuits.surgery.hmatrix.PPM_joint_cellulation import _cellulate_port_subgraph

    G_aux = nx.cycle_graph(4)  # one 4-cycle
    added = _cellulate_port_subgraph(G_aux, ports=(0, 1, 2, 3), max_len=6)
    assert added == []


def test_cellulate_raises_when_port_cycle_has_no_available_chord() -> None:
    """RuntimeError when a port-subgraph cycle exists but every (i, j) pair
    is already an edge — i.e. the port subgraph is complete on those vertices."""
    import networkx as nx

    from qldpc.circuits.surgery.hmatrix.PPM_joint_cellulation import _cellulate_port_subgraph

    # 7-cycle 0-1-2-3-4-5-6-0 plus ALL chords among {0..6} → complete graph K_7.
    # cycle_basis still surfaces cycles of length > max_len in K_7 (basis cycles
    # are length-3 triangles), so no long cycle exists in this case.
    # Instead: make a 7-cycle without any extra edges, then call with max_len=2.
    G = nx.cycle_graph(7)
    ports = tuple(range(7))
    # Already a complete graph K_7? No — cycle_graph(7) has only 7 edges.
    # Pre-saturate with all possible chords so no chord can be added:
    for i in range(7):
        for j in range(i + 2, 7):
            if not G.has_edge(i, j) and (i, j) != (0, 6):
                G.add_edge(i, j)
    # Now every (i, j) with j >= i+2 in the 7-cycle is already an edge.
    # A length-7 basis cycle no longer exists (it's broken into triangles),
    # so max_len=6 finds no long cycle and returns []. Use max_len=2 to force
    # the failure path:
    with pytest.raises(RuntimeError, match=r"No chord found"):
        _cellulate_port_subgraph(G, ports, max_len=2)


def test_cellulate_port_subgraph_breaks_long_port_cycle() -> None:
    """Ports are a strict subset of vertices, with a long cycle on the port
    subgraph. Cellulation breaks the port cycle without inspecting non-port
    edges elsewhere in G_aux."""
    import networkx as nx

    from qldpc.circuits.surgery.hmatrix.PPM_joint_cellulation import _cellulate_port_subgraph

    G = nx.Graph()
    # 8-cycle on port vertices 0..7
    G.add_edges_from([(i, (i + 1) % 8) for i in range(8)])
    # Non-port "decoration": dangling vertex 100 attached to port 0
    G.add_edge(0, 100)
    ports = tuple(range(8))
    added = _cellulate_port_subgraph(G, ports, max_len=6)
    assert len(added) >= 1
    # All chord endpoints must be ports (cycle vertices are port vertices)
    for u, v in added:
        assert u in ports and v in ports
    # The non-port vertex 100 was not touched
    assert G.has_edge(0, 100)
    # All port-subgraph basis cycles now bounded
    sub = G.subgraph(ports)
    for c in nx.cycle_basis(sub):
        assert len(c) <= 6


def test_cellulate_port_subgraph_skips_non_port_cycle() -> None:
    """Long cycle entirely on non-port vertices is ignored; no edges added."""
    import networkx as nx

    from qldpc.circuits.surgery.hmatrix.PPM_joint_cellulation import _cellulate_port_subgraph

    G = nx.Graph()
    # Long non-port cycle: 10-11-12-...-17-10 (length 8)
    G.add_edges_from(
        [(10, 11), (11, 12), (12, 13), (13, 14), (14, 15), (15, 16), (16, 17), (17, 10)]
    )
    # Short port cycle: triangle on 0,1,2
    G.add_edges_from([(0, 1), (1, 2), (2, 0)])
    ports = (0, 1, 2)
    n_edges_before = G.number_of_edges()
    added = _cellulate_port_subgraph(G, ports, max_len=6)
    assert added == []
    assert G.number_of_edges() == n_edges_before


def test_canonical_H_R_rejects_w_below_2() -> None:
    """_canonical_H_R(w=1) raises (rep-code needs w >= 2)."""
    from qldpc.circuits.surgery.hmatrix.PPM_joint_cellulation import _canonical_H_R

    with pytest.raises(ValueError, match="w >= 2"):
        _canonical_H_R(1)


def test_skip_tree_fullrank_defaults_edge_index_when_omitted() -> None:
    """_skip_tree_fullrank with edge_index_verts=None builds the default index dict
    from S.edges() order — matches the explicit-dict path."""
    import networkx as nx

    from qldpc.circuits.surgery.hmatrix.PPM_joint_cellulation import _skip_tree_fullrank

    G_nx = nx.complete_graph(4)
    T_explicit, P_explicit = _skip_tree_fullrank(
        G_nx,
        root=0,
        edge_index_verts={tuple(sorted(e)): i for i, e in enumerate(G_nx.edges())},
    )
    T_default, P_default = _skip_tree_fullrank(G_nx, root=0)
    assert np.array_equal(T_default, T_explicit)
    assert np.array_equal(P_default, P_explicit)
