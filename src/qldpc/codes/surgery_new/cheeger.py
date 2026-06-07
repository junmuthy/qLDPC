"""Cheeger and distance boost transformations for surgery gadgets."""

from __future__ import annotations

import dataclasses

import galois
import numpy as np

from qldpc.codes.common import CSSCode
from qldpc.codes.surgery import (  # NOTE: temporary — fixed in Task 5
    SurgeryLayout,
    _assemble_merged_HX,
    _assemble_merged_HZ,
    _build_layered_blocks,
    _build_layout,
    _compute_gauge_fix,
)


@dataclasses.dataclass(frozen=True, eq=False)
class BoostResult:
    """Statistics about a Cheeger boost run."""

    extra_qubits_added: int
    final_h_lower_bound: float
    iterations: int
    terminated_by: str  # "target_reached" | "max_qubits_exhausted" | "no_progress"


def _exact_boundary_cheeger(F: galois.FieldArray) -> tuple[float, np.ndarray]:
    """Exact boundary Cheeger constant of F per Webster §II.A Definition 1.

    Helper / sanity-check tool. NOT used by ``boost_gadget_cheeger`` (which
    follows Williamson-Yoder / Webster: random edge addition + distance
    verification). Kept for diagnostic use to debug the cut structure
    when a boost run is unexpectedly long.

    For bipartite incidence F: V -> C (V = X-check χ_i indices, C = κ_j
    indices), the boundary ∂v of v ⊆ V is the subset of C with an odd
    number of neighbours in v. The boundary Cheeger constant is

        h(F) = min_{v ⊆ V, 1 ≤ |v| ≤ |V|/2} |∂v| / |v|.

    Computes h(F) exactly by Gray-code enumeration over all subsets v.
    Tractable for |V| ≤ 26 (≈ 67M subsets; ~5-30 s in numpy). Raises if
    |V| > 26.

    Args:
        F: GF(2) restriction matrix of shape (|C|, |V|).

    Returns:
        (h, v_star_indicator) where v_star_indicator is a length-|V|
        binary numpy array marking the worst cut. If |V| < 2, returns
        (inf, zero vector) — boost is not applicable.

    Raises:
        ValueError: if |V| > 26 (exhaustive enumeration is infeasible).
    """
    F_arr = np.asarray(F).astype(np.int8)
    n_C, n_V = F_arr.shape
    if n_V < 2:
        return float("inf"), np.zeros(n_V, dtype=np.int8)
    if n_V > 26:
        raise ValueError(
            f"_exact_boundary_cheeger requires |V| ≤ 26; got |V|={n_V}. "
            f"Use _spectral_cheeger_lower_bound for larger problems."
        )

    # Bit-pack F columns: F_col_ints[i] is a Python int with bit r set iff
    # F[r, i] = 1. Boundary as Python int allows O(1) XOR + popcount.
    F_col_ints = [
        int.from_bytes(np.packbits(F_arr[:, i][::-1]).tobytes()[::-1], "little")
        for i in range(n_V)
    ]
    boundary_int = 0
    subset_mask = 0
    half = n_V // 2
    best_h = float("inf")
    best_mask = 0
    total = 1 << n_V

    for k in range(1, total):
        bit = (k & -k).bit_length() - 1
        subset_mask ^= 1 << bit
        boundary_int ^= F_col_ints[bit]
        size = subset_mask.bit_count()
        if 1 <= size <= half:
            cut = boundary_int.bit_count()
            if cut < best_h * size:
                best_h = cut / size
                best_mask = subset_mask

    v_star = np.zeros(n_V, dtype=np.int8)
    for i in range(n_V):
        if best_mask & (1 << i):
            v_star[i] = 1
    return best_h, v_star


def _spectral_cheeger_lower_bound(F: galois.FieldArray) -> float:
    """Spectral lower bound on the boundary Cheeger constant of F.

    Returns ``lambda_2(F_float @ F_float.T) / 2.0``, where F_float =
    F.astype(np.float64). This is a tractable lower bound based on the
    discrete Cheeger inequality and is what boost_gadget_cheeger uses to
    decide when to stop adding augmentation qubits.

    Args:
        F: GF(2) restriction matrix of shape (|C_0|, |V_0|).

    Returns:
        Non-negative float lower bound on h(F).
    """
    F_float = np.asarray(F).astype(np.float64)
    if F_float.shape[0] < 2:
        return 0.0
    M = F_float @ F_float.T
    eigenvalues = np.linalg.eigvalsh(M)
    lambda_2 = float(eigenvalues[1])
    return max(0.0, lambda_2 / 2.0)


def boost_gadget_cheeger(
    merged: CSSCode,
    layout: SurgeryLayout,
    *,
    target_h: float = 1.0,
    max_extra_qubits: int | None = None,
    seed: int | None = None,
) -> tuple[CSSCode, SurgeryLayout, BoostResult]:
    """Heuristic Cheeger augmentation by random degree-2 edge addition.

    Implements Webster (arXiv:2511.15989) §II.A end's "+n" trick:
    iteratively add new κ' ancilla qubits to the gadget, each connecting
    a random pair of X-checks (χ_i, χ_j) not already directly connected
    via another κ, until the spectral lower bound on the boundary Cheeger
    constant of F reaches target_h.

    Args:
        merged: merged CSSCode returned by build_layered_surgery_code.
        layout: the associated SurgeryLayout (used to read F).
        target_h: target Cheeger lower bound. Default 1.0.
        max_extra_qubits: cap on additions. None = unbounded.
        seed: RNG seed for reproducibility.

    Returns:
        (boosted_merged, boosted_layout, result).

    Raises:
        ValueError: target_h <= 0, max_extra_qubits < 0, or F too small.
    """
    if target_h <= 0:
        raise ValueError(f"target_h must be positive, got {target_h}.")
    if max_extra_qubits is not None and max_extra_qubits < 0:
        raise ValueError(f"max_extra_qubits must be >= 0, got {max_extra_qubits}.")
    if layout.F.shape[1] < 2:
        raise ValueError(
            f"F has {layout.F.shape[1]} columns; need >= 2 X-checks to add an edge."
        )

    rng = np.random.default_rng(seed)
    field = layout.F.__class__
    F = np.asarray(layout.F).astype(np.int_).copy()
    n_X = F.shape[1]

    def _existing_pairs(F_arr: np.ndarray) -> set[tuple[int, int]]:
        pairs: set[tuple[int, int]] = set()
        for row in F_arr:
            ones = np.flatnonzero(row)
            for i in range(len(ones)):
                for j in range(i + 1, len(ones)):
                    pairs.add((int(ones[i]), int(ones[j])))
        return pairs

    extra = 0
    iterations = 0
    terminated_by = "no_progress"
    h_lb = _spectral_cheeger_lower_bound(field(F))
    max_iter_inner = 10 * n_X * n_X

    while True:
        iterations += 1
        h_lb = _spectral_cheeger_lower_bound(field(F))
        if h_lb >= target_h:
            terminated_by = "target_reached"
            break
        if max_extra_qubits is not None and extra >= max_extra_qubits:
            terminated_by = "max_qubits_exhausted"
            break
        if iterations > max_iter_inner:
            terminated_by = "no_progress"
            break

        pairs = _existing_pairs(F)
        candidate = None
        for _attempt in range(n_X * 2):
            i, j = sorted(int(x) for x in rng.choice(n_X, 2, replace=False))
            if (i, j) not in pairs:
                candidate = (i, j)
                break
        if candidate is None:
            terminated_by = "no_progress"
            break

        new_row = np.zeros(n_X, dtype=np.int_)
        new_row[candidate[0]] = 1
        new_row[candidate[1]] = 1
        F = np.vstack([F, new_row])
        extra += 1

    augmented_F = field(F)
    G = _compute_gauge_fix(augmented_F)
    blocks = _build_layered_blocks(augmented_F, layout.num_layers)
    n_data = layout.num_data_qubits
    n_extra = extra  # number of new κ' qubits added

    data_x = np.asarray(merged.matrix_x[layout.hx_row_kind == "data"]).astype(np.int_)
    data_z = np.asarray(merged.matrix_z[layout.hz_row_kind == "data"]).astype(np.int_)
    data_x = field(data_x[:, :n_data])
    data_z_original = field(data_z[:, :n_data])

    # Extend data_z with `n_extra` synthetic zero rows. Each synthetic row
    # represents a "data Z-check" that doesn't touch any data qubit; its only
    # role is to anchor the new κ' qubit in the merged code's c0 region so
    # _assemble_merged_HZ's identity-injection slicing works.
    if n_extra > 0:
        synthetic_z = field.Zeros((n_extra, n_data))
        data_z_extended = field(np.vstack([np.asarray(data_z_original), np.asarray(synthetic_z)]))
    else:
        data_z_extended = data_z_original

    data_code_proxy = CSSCode(data_x, data_z_extended, is_subsystem_code=False)

    # Extend c0_indices to include the new synthetic Z-check rows. Their
    # indices come right after the original data_z's rows.
    n_z_data_original = data_z_original.shape[0]
    new_c0_indices = np.concatenate([
        layout.c0_indices,
        np.arange(n_z_data_original, n_z_data_original + n_extra, dtype=np.int_),
    ])

    HX_new = _assemble_merged_HX(data_code_proxy, blocks, layout.v0_indices)
    HZ_new = _assemble_merged_HZ(data_code_proxy, blocks, G, new_c0_indices)

    boosted_merged = CSSCode(HX_new, HZ_new, is_subsystem_code=False)
    boosted_layout = _build_layout(
        data_code_proxy, blocks, G, layout.v0_indices, new_c0_indices, augmented_F
    )
    return boosted_merged, boosted_layout, BoostResult(
        extra_qubits_added=extra,
        final_h_lower_bound=float(h_lb),
        iterations=iterations,
        terminated_by=terminated_by,
    )


@dataclasses.dataclass(frozen=True, eq=False)
class DistanceBoostResult:
    """Statistics about a Williamson-Yoder / Webster distance-verify boost run."""

    extra_qubits_added: int
    target_distance: int
    final_d_x_bound: int | float
    final_d_z_bound: int | float
    trials_attempted: int
    terminated_by: str  # "target_reached" | "max_qubits_exhausted" | "no_progress"


def _reassemble_gadget_with_new_F(
    merged: CSSCode,
    layout: SurgeryLayout,
    augmented_F: galois.FieldArray,
    n_extra: int,
) -> tuple[CSSCode, SurgeryLayout]:
    """Rebuild merged code + layout from an augmented restriction matrix.

    Boost-added κ' qubits (rows beyond the original C_0) are GAUGE qubits:
    they have no data-Z extension (no S_j in C_0 to extend). Their Z-stab
    contribution comes purely through the augmented gauge-fix matrix
    G_aug = basis of left null space of F_aug.

    CSS commutation derivation:
      - chi_i restricted to κ_aug = F_aug[:, i].
      - data Z S_j (j in original C_0) restricted to κ_aug = e_j on κ_orig,
        zero on κ'.
      - overlap = [q_i in S_j] + F_aug[j, i] = 0 (mod 2, by F[j, i] def).
      - chi_i ⋅ gauge_fix γ ∈ G_aug = γ ⋅ F_aug[:, i] = 0 (mod 2,
        by γ in ker(F_aug^T)).
    """
    field = augmented_F.__class__
    G_aug = _compute_gauge_fix(augmented_F)
    blocks = _build_layered_blocks(augmented_F, layout.num_layers)
    n_data = layout.num_data_qubits

    data_x_arr = np.asarray(merged.matrix_x[layout.hx_row_kind == "data"]).astype(np.int_)
    data_z_arr = np.asarray(merged.matrix_z[layout.hz_row_kind == "data"]).astype(np.int_)
    data_x_gf = field(data_x_arr[:, :n_data])
    data_z_gf = field(data_z_arr[:, :n_data])

    data_code_proxy = CSSCode(data_x_gf, data_z_gf, is_subsystem_code=False)

    HX_new = _assemble_merged_HX(data_code_proxy, blocks, layout.v0_indices)

    # Manually build HZ_new (instead of _assemble_merged_HZ) so that the new
    # κ' qubits get NO data-Z extension — only G_aug rows mention them.
    n_merged = n_data + blocks.total_ancilla
    n_kappa_orig = int(layout.F.shape[0])

    old_z = field.Zeros((data_z_gf.shape[0], n_merged))
    old_z[:, :n_data] = data_z_gf
    c1_slice = blocks.ancilla_col_slice(1)
    I_partial = field.Identity(n_kappa_orig)
    # Place identity at (original c0_indices) x (original κ columns) only.
    old_z[layout.c0_indices, n_data + c1_slice.start : n_data + c1_slice.start + n_kappa_orig] = I_partial

    even_rows = []
    for i in range(2, blocks.num_layers, 2):
        row_block = field.Zeros((blocks.n_c0, n_merged))
        prev_slice = blocks.ancilla_col_slice(i - 1)
        cur_slice = blocks.ancilla_col_slice(i)
        next_slice = blocks.ancilla_col_slice(i + 1)
        I_c0_full = field.Identity(blocks.n_c0)
        row_block[:, n_data + prev_slice.start : n_data + prev_slice.stop] = I_c0_full
        row_block[:, n_data + cur_slice.start : n_data + cur_slice.stop] = blocks.F
        row_block[:, n_data + next_slice.start : n_data + next_slice.stop] = I_c0_full
        even_rows.append(row_block)

    gauge_rows: list[galois.FieldArray] = []
    if G_aug.shape[0] > 0:
        gf = field.Zeros((G_aug.shape[0], n_merged))
        cL_slice = blocks.ancilla_col_slice(blocks.num_layers)
        gf[:, n_data + cL_slice.start : n_data + cL_slice.stop] = G_aug
        gauge_rows.append(gf)

    HZ_new = field(np.vstack([old_z, *even_rows, *gauge_rows]))

    boosted_merged = CSSCode(HX_new, HZ_new, is_subsystem_code=False)
    boosted_layout = _build_layout(
        data_code_proxy, blocks, G_aug, layout.v0_indices, layout.c0_indices, augmented_F
    )
    return boosted_merged, boosted_layout


def _augment_F_with_random_edges(
    F_base: np.ndarray,
    n_new_edges: int,
    rng: np.random.Generator,
) -> np.ndarray | None:
    """Add n_new_edges random degree-2 rows to F (each connects two distinct
    columns not already directly connected via another existing row).

    Returns None if a collision-free sample could not be drawn within the
    attempt budget.
    """
    F = F_base.copy()
    n_X = F.shape[1]
    if n_X < 2:
        return None

    def _existing_pairs(arr: np.ndarray) -> set[tuple[int, int]]:
        pairs: set[tuple[int, int]] = set()
        for row in arr:
            ones = np.flatnonzero(row)
            for i in range(len(ones)):
                for j in range(i + 1, len(ones)):
                    pairs.add((int(ones[i]), int(ones[j])))
        return pairs

    pairs = _existing_pairs(F)
    new_rows: list[np.ndarray] = []
    for _ in range(n_new_edges):
        candidate = None
        for _attempt in range(n_X * 4):
            i, j = sorted(int(x) for x in rng.choice(n_X, 2, replace=False))
            if (i, j) not in pairs:
                candidate = (i, j)
                break
        if candidate is None:
            return None
        pairs.add(candidate)
        row = np.zeros(n_X, dtype=np.int_)
        row[candidate[0]] = 1
        row[candidate[1]] = 1
        new_rows.append(row)
    if not new_rows:
        return F
    return np.vstack([F, np.stack(new_rows)])


def boost_gadget_cheeger_combinatorial(
    merged: CSSCode,
    layout: SurgeryLayout,
    *,
    target_h: float = 1.0,
    max_extra_qubits: int = 50,
    seed: int | None = None,
) -> tuple[CSSCode, SurgeryLayout, BoostResult]:
    """Greedy combinatorial Cheeger boost — deterministic distance guarantee.

    Computes the exact boundary Cheeger constant h(F) via subset enumeration
    (Webster Def 1 / Cross Def 3). When h < target_h, identifies the worst
    cut v* and adds a κ qubit (degree-2 row of F) with one endpoint in v*
    and one outside, which monotonically increases |∂v*| by 1 without
    decreasing any other |∂v|.

    By Cross §III Thm 6, h(F) >= 1 implies d_merged >= d_data, so reaching
    target_h = 1.0 GUARANTEES distance preservation (no decoder verification
    needed). Tractable for |V_0| <= 26 (Webster's family up to l=255).

    Compared with boost_gadget_distance (BP+OSD-verified): this method
    provides a deterministic mathematical guarantee at the cost of possibly
    over-adding edges (since h >= 1 is sufficient but not necessary). For
    Webster's BB-code seeds, |V_0| = wt(X̄) is small (6, 10, 16, 26).

    Args:
        merged: merged CSSCode from build_layered_surgery_code.
        layout: associated SurgeryLayout (used to read F).
        target_h: Cheeger target. Default 1.0 (Cross Thm 6 threshold).
        max_extra_qubits: cap on additions. Default 50.
        seed: RNG seed for tie-breaking in edge selection.

    Returns:
        (boosted_merged, boosted_layout, BoostResult). final_h_lower_bound
        field holds the EXACT achieved h, not a lower bound.

    Raises:
        ValueError: |V_0| > 26 (enumeration infeasible) or target_h <= 0.
    """
    if target_h <= 0:
        raise ValueError(f"target_h must be positive, got {target_h}.")
    if max_extra_qubits < 0:
        raise ValueError(f"max_extra_qubits must be >= 0, got {max_extra_qubits}.")

    rng = np.random.default_rng(seed)
    field = layout.F.__class__
    F = np.asarray(layout.F).astype(np.int_).copy()
    n_V = F.shape[1]
    if n_V > 26:
        raise ValueError(
            f"|V_0| = {n_V} > 26; exact Cheeger enumeration infeasible. "
            f"Use boost_gadget_distance (BP+OSD) instead."
        )
    if n_V < 2:
        return merged, layout, BoostResult(
            extra_qubits_added=0, final_h_lower_bound=float("inf"),
            iterations=0, terminated_by="target_reached",
        )

    half = n_V // 2
    # Phase 1: Gray-code enumerate all 1 <= |v| <= half subsets, recording
    # (mask, |v|, |∂v|) into flat arrays for vectorized incremental updates.
    F_col_ints = [
        int.from_bytes(
            np.packbits(F[:, i][::-1]).tobytes()[::-1], "little"
        ) for i in range(n_V)
    ]
    total = 1 << n_V
    masks_buf: list[int] = []
    sizes_buf: list[int] = []
    cuts_buf: list[int] = []
    boundary_int = 0
    subset_mask = 0
    for k in range(1, total):
        bit = (k & -k).bit_length() - 1
        subset_mask ^= 1 << bit
        boundary_int ^= F_col_ints[bit]
        size = subset_mask.bit_count()
        if 1 <= size <= half:
            masks_buf.append(subset_mask)
            sizes_buf.append(size)
            cuts_buf.append(boundary_int.bit_count())

    masks = np.array(masks_buf, dtype=np.uint64)
    sizes = np.array(sizes_buf, dtype=np.int32)
    cuts = np.array(cuts_buf, dtype=np.int32)

    def _existing_pairs(arr: np.ndarray) -> set[tuple[int, int]]:
        pairs: set[tuple[int, int]] = set()
        for row in arr:
            ones = np.flatnonzero(row)
            for a in range(len(ones)):
                for b in range(a + 1, len(ones)):
                    pairs.add((int(ones[a]), int(ones[b])))
        return pairs

    extra = 0
    iterations = 0
    terminated_by = "no_progress"

    while True:
        iterations += 1
        # Find min h = cuts/sizes via vectorized search.
        # Use cuts * sizes_max comparison to avoid float division.
        h_num = cuts.astype(np.int64)
        h_den = sizes.astype(np.int64)
        # ratio = h_num / h_den; pick argmin
        idx = int(np.argmin(h_num / h_den))
        h = float(h_num[idx] / h_den[idx])
        worst_mask = int(masks[idx])

        if h >= target_h:
            terminated_by = "target_reached"
            break
        if extra >= max_extra_qubits:
            terminated_by = "max_qubits_exhausted"
            break

        v_star_arr = np.array(
            [(worst_mask >> i) & 1 for i in range(n_V)], dtype=np.int8
        )
        inside = np.flatnonzero(v_star_arr).tolist()
        outside = np.flatnonzero(1 - v_star_arr).tolist()
        if not inside or not outside:
            terminated_by = "no_progress"
            break

        rng.shuffle(inside)
        rng.shuffle(outside)
        pairs = _existing_pairs(F)
        chosen = None
        for i in inside:
            for j in outside:
                a, b = (i, j) if i < j else (j, i)
                if (a, b) not in pairs:
                    chosen = (a, b)
                    break
            if chosen is not None:
                break
        if chosen is None:
            terminated_by = "no_progress"
            break

        new_row = np.zeros(n_V, dtype=np.int_)
        new_row[chosen[0]] = 1
        new_row[chosen[1]] = 1
        F = np.vstack([F, new_row])
        extra += 1

        # Vectorized cut update: for each subset, cut += parity of
        # |v ∩ {i, j}| = bit_i XOR bit_j of the subset mask.
        bit_i = ((masks >> chosen[0]) & np.uint64(1)).astype(np.int32)
        bit_j = ((masks >> chosen[1]) & np.uint64(1)).astype(np.int32)
        cuts += (bit_i ^ bit_j)

    augmented_F = field(F)
    boosted_merged, boosted_layout = _reassemble_gadget_with_new_F(
        merged, layout, augmented_F, extra
    )
    return boosted_merged, boosted_layout, BoostResult(
        extra_qubits_added=extra,
        final_h_lower_bound=float(h),
        iterations=iterations,
        terminated_by=terminated_by,
    )


def boost_gadget_distance(
    merged: CSSCode,
    layout: SurgeryLayout,
    *,
    target_distance: int,
    max_extra_qubits: int = 30,
    num_trials_per_step: int = 20,
    decoder_trials: int = 10,
    seed: int | None = None,
) -> tuple[CSSCode, SurgeryLayout, DistanceBoostResult]:
    """Williamson-Yoder / Webster distance-verifying gadget boost.

    Per the procedure described in Cain et al. arXiv:2503.10390 and adopted
    by Webster: iteratively add small random batches of degree-2 edges to F,
    use BP+OSD upper-bound on merged code distance to fast-reject any
    augmentation whose deformed code falls below the target distance.

    Starts from n_extra = 0 (verify bare gadget already meets target). For
    each n_extra, samples ``num_trials_per_step`` random degree-2 augmentations
    and returns the first one that passes the BP+OSD test for BOTH X and Z
    distances (each tested with ``decoder_trials`` random trials).

    Args:
        merged: merged CSSCode returned by build_layered_surgery_code.
        layout: associated SurgeryLayout.
        target_distance: minimum X- and Z-distance required for acceptance
            (usually the data code's distance d_data).
        max_extra_qubits: cap on number of new κ' qubits to consider.
        num_trials_per_step: how many random augmentations to try per
            n_extra value before incrementing.
        decoder_trials: trials for each get_distance_bound_with_decoder call.
        seed: RNG seed for reproducibility.

    Returns:
        (boosted_merged, boosted_layout, DistanceBoostResult).

    Raises:
        ValueError: target_distance <= 0 or max_extra_qubits < 0.

    Notes:
        BP+OSD gives an UPPER bound on distance. ``d_bound >= target_distance``
        means the decoder could not find a logical operator of weight below
        target; with enough trials this is a strong heuristic that
        d_merged >= target_distance, but not a proof. For exact verification,
        post-process accepted candidates with ``merged.get_distance_exact()``.
    """
    from qldpc.objects import Pauli as _Pauli

    if target_distance <= 0:
        raise ValueError(f"target_distance must be positive, got {target_distance}.")
    if max_extra_qubits < 0:
        raise ValueError(f"max_extra_qubits must be >= 0, got {max_extra_qubits}.")

    rng = np.random.default_rng(seed)
    field = layout.F.__class__
    F_base = np.asarray(layout.F).astype(np.int_)

    trials_attempted = 0

    def _passes_decoder(code: CSSCode) -> tuple[bool, int | float, int | float]:
        bx = code.get_distance_bound_with_decoder(_Pauli.X, num_trials=decoder_trials)
        if bx < target_distance:
            return False, bx, -1
        bz = code.get_distance_bound_with_decoder(_Pauli.Z, num_trials=decoder_trials)
        return (bz >= target_distance), bx, bz

    # n_extra = 0: test bare gadget first.
    ok, bx0, bz0 = _passes_decoder(merged)
    trials_attempted += 1
    if ok:
        return merged, layout, DistanceBoostResult(
            extra_qubits_added=0,
            target_distance=target_distance,
            final_d_x_bound=bx0,
            final_d_z_bound=bz0,
            trials_attempted=trials_attempted,
            terminated_by="target_reached",
        )

    for n_extra in range(1, max_extra_qubits + 1):
        best_bx: int | float = -1
        best_bz: int | float = -1
        for _trial in range(num_trials_per_step):
            trials_attempted += 1
            F_aug = _augment_F_with_random_edges(F_base, n_extra, rng)
            if F_aug is None:
                continue
            try:
                boosted_merged, boosted_layout = _reassemble_gadget_with_new_F(
                    merged, layout, field(F_aug), n_extra
                )
            except Exception:
                continue
            ok, bx, bz = _passes_decoder(boosted_merged)
            best_bx = max(best_bx, bx)
            best_bz = max(best_bz, bz) if bz != -1 else best_bz
            if ok:
                return boosted_merged, boosted_layout, DistanceBoostResult(
                    extra_qubits_added=n_extra,
                    target_distance=target_distance,
                    final_d_x_bound=bx,
                    final_d_z_bound=bz,
                    trials_attempted=trials_attempted,
                    terminated_by="target_reached",
                )

    return merged, layout, DistanceBoostResult(
        extra_qubits_added=0,
        target_distance=target_distance,
        final_d_x_bound=bx0,
        final_d_z_bound=bz0,
        trials_attempted=trials_attempted,
        terminated_by="max_qubits_exhausted",
    )
