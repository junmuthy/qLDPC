"""Cheeger and distance boost transformations for surgery gadgets."""

from __future__ import annotations

import dataclasses

import galois
import numpy as np
import numpy.typing as npt

from qldpc.codes.common import CSSCode
from .gadget import GadgetLayout, _assemble_HX_L1


@dataclasses.dataclass(frozen=True, eq=False)
class SurgeryLayout:
    """Provenance of qubits and checks in a merged L=1 surgery code.

    Internal type used by the boost dispatcher and the legacy-layout bridge
    (_gadget_to_legacy_layout / _legacy_to_gadget).

    Attributes:
        num_data_qubits: Number of qubits in the original data code.
        num_ancilla_qubits: Number of κ ancilla qubits (= |C_0| + boost extras).
        num_data_x_checks: Number of rows of HX_merged that are "data" rows
            (i.e. the original X-checks). Boost code uses this to slice off
            data rows from the merged check matrix when extracting the data
            block.
        num_data_z_checks: Number of rows of HZ_merged that are "data" rows.
        v0_indices: Indices (within data qubits) of supp(X̄_M) = V_0.
        c0_indices: Row indices (within H_Z of data code) of Z-checks adjacent
            to V_0 = C_0.
        F: Step-1 restriction matrix; shape (|C_0|, |V_0|).
        G: Step-2 gauge-fix basis; rows span the left null space of F.
    """

    num_data_qubits: int
    num_ancilla_qubits: int
    num_data_x_checks: int
    num_data_z_checks: int
    v0_indices: npt.NDArray[np.int_]
    c0_indices: npt.NDArray[np.int_]
    F: galois.FieldArray
    G: galois.FieldArray


def _compute_gauge_fix(F: galois.FieldArray) -> galois.FieldArray:
    """Compute G whose rows form a basis of the left null space of F."""
    return F.left_null_space()


def _build_layout(
    data_code: CSSCode,
    F: galois.FieldArray,
    G: galois.FieldArray,
    v0_indices: np.ndarray,
    c0_indices: np.ndarray,
) -> SurgeryLayout:
    """Assemble SurgeryLayout from the building blocks (L=1)."""
    n_data = data_code.num_qubits
    n_ancilla = int(F.shape[0])
    num_data_x_checks = int(data_code.matrix_x.shape[0])
    num_data_z_checks = int(data_code.matrix_z.shape[0])
    return SurgeryLayout(
        num_data_qubits=n_data,
        num_ancilla_qubits=n_ancilla,
        num_data_x_checks=num_data_x_checks,
        num_data_z_checks=num_data_z_checks,
        v0_indices=v0_indices,
        c0_indices=c0_indices,
        F=F,
        G=G,
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


def cheeger_constant(g: GadgetLayout) -> float:
    """Boundary Cheeger constant of a gadget's F matrix (Webster §II.A Def 1).

    Returns the exact h(F) when |V_0| ≤ 26 (Gray-code subset enumeration),
    otherwise the spectral lower bound. Either way:

        h(g) ≥ 1   ⇒   surgery on this gadget preserves code distance
                       (Webster Lemma 9; structural argument, no decoder).
        h(g) <  1   ⇒   distance may degrade; consider boost_gadget(g, target=1.0).

    Use as a pre-flight check before deciding whether to call boost_gadget.
    """
    F = galois.GF(2)(np.asarray(g.F).astype(int))
    if F.shape[1] <= 26:
        h, _ = _exact_boundary_cheeger(F)
        return h
    return _spectral_cheeger_lower_bound(F)


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
    # Use the same gauge-qubit reassembly as the combinatorial variant.
    # The earlier synthetic-data-Z extension path was wrong: it injected
    # identity Z rows on the new κ' qubits, breaking CSS commutation with
    # the chi rows of HX. Routing through _reassemble_gadget_with_new_F
    # (which treats boost-added qubits as pure gauge qubits with Z support
    # only through G_aug = ker(F_aug^T)) preserves HX @ HZ^T = 0.
    boosted_merged, boosted_layout = _reassemble_gadget_with_new_F(
        merged, layout, augmented_F, extra,
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
    """Rebuild merged code + layout from an augmented restriction matrix (L=1).

    Boost-added κ' qubits (rows of F_aug beyond original C_0) are GAUGE qubits:
    they have no data-Z extension (no S_j in C_0 to extend). Their Z-stab
    contribution comes purely through the augmented gauge-fix matrix
    G_aug = basis of left null space of F_aug.
    """
    field = augmented_F.__class__
    G_aug = _compute_gauge_fix(augmented_F)
    n_data = layout.num_data_qubits

    data_x_arr = np.asarray(merged.matrix_x[:layout.num_data_x_checks]).astype(np.int_)
    data_z_arr = np.asarray(merged.matrix_z[:layout.num_data_z_checks]).astype(np.int_)
    data_x_gf = field(data_x_arr[:, :n_data])
    data_z_gf = field(data_z_arr[:, :n_data])

    data_code_proxy = CSSCode(data_x_gf, data_z_gf, is_subsystem_code=False)

    HX_data_uint8 = np.asarray(data_x_gf).astype(np.uint8)
    F_uint8 = np.asarray(augmented_F).astype(np.uint8)
    HX_new_uint8 = _assemble_HX_L1(HX_data_uint8, layout.v0_indices, F_uint8)
    HX_new = field(HX_new_uint8.astype(np.int_).tolist())

    # Manually build HZ_new: new κ' qubits get NO data-Z extension — only G_aug
    # rows mention them.
    n_c0 = int(augmented_F.shape[0])
    n_merged = n_data + n_c0
    n_kappa_orig = int(layout.F.shape[0])

    old_z = field.Zeros((data_z_gf.shape[0], n_merged))
    old_z[:, :n_data] = data_z_gf
    I_partial = field.Identity(n_kappa_orig)
    old_z[layout.c0_indices, n_data:n_data + n_kappa_orig] = I_partial

    gauge_rows: list[galois.FieldArray] = []
    if G_aug.shape[0] > 0:
        gf = field.Zeros((G_aug.shape[0], n_merged))
        gf[:, n_data:] = G_aug
        gauge_rows.append(gf)

    HZ_new = field(np.vstack([old_z, *gauge_rows]))

    boosted_merged = CSSCode(HX_new, HZ_new, is_subsystem_code=False)
    boosted_layout = _build_layout(
        data_code_proxy, augmented_F, G_aug, layout.v0_indices, layout.c0_indices,
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


def _gadget_to_legacy_layout(g):
    """Convert a GadgetLayout into the legacy (CSSCode, SurgeryLayout) pair
    consumed by boost_gadget_cheeger* / boost_gadget_distance.

    For basis=Pauli.Z, we SWAP HX/HZ so the legacy boost code (designed for
    X-basis chi rows in HX_merged) sees the chi rows where it expects them.
    The boost result is dual-swapped back in _legacy_to_gadget.
    """
    from qldpc.objects import Pauli
    F2 = galois.GF(2)
    n = g.code.num_qudits
    n_anc = len(g.C0)
    mX_data = int(g.code.matrix_x.shape[0])
    mZ_data = int(g.code.matrix_z.shape[0])

    if g.basis is Pauli.X:
        HX_for_legacy = g.HX_merged
        HZ_for_legacy = g.HZ_merged
        # After (no) swap: HX_for_legacy data rows = mX_data; HZ data rows = mZ_data.
        num_data_x_checks = mX_data
        num_data_z_checks = mZ_data
    else:  # Pauli.Z: swap so chi rows are in HX_for_legacy
        HX_for_legacy = g.HZ_merged
        HZ_for_legacy = g.HX_merged
        # After swap: HX_for_legacy data rows = mZ_data (the original HZ data part);
        #             HZ_for_legacy data rows = mX_data.
        num_data_x_checks = mZ_data
        num_data_z_checks = mX_data

    merged = CSSCode(
        F2(np.asarray(HX_for_legacy).astype(np.int_).tolist()),
        F2(np.asarray(HZ_for_legacy).astype(np.int_).tolist()),
        is_subsystem_code=False,
    )
    layout = SurgeryLayout(
        num_data_qubits=n,
        num_ancilla_qubits=n_anc,
        num_data_x_checks=num_data_x_checks,
        num_data_z_checks=num_data_z_checks,
        v0_indices=np.array(g.V0, dtype=np.int_),
        c0_indices=np.array(g.C0, dtype=np.int_),
        F=F2(np.asarray(g.F).astype(np.int_).tolist()),
        G=F2(np.asarray(g.G).astype(np.int_).tolist()),
    )
    return merged, layout


def _legacy_to_gadget(merged: CSSCode, layout: SurgeryLayout, original_g):
    """Reconstruct a GadgetLayout from a boost result (merged + legacy SurgeryLayout).

    For basis=Pauli.Z, we undo the HX/HZ swap applied in _gadget_to_legacy_layout.
    """
    from qldpc.objects import Pauli
    HX_m_legacy = np.asarray(merged.matrix_x).astype(np.uint8)
    HZ_m_legacy = np.asarray(merged.matrix_z).astype(np.uint8)
    if original_g.basis is Pauli.X:
        HX_m = HX_m_legacy
        HZ_m = HZ_m_legacy
    else:  # undo the basis-Z swap
        HX_m = HZ_m_legacy
        HZ_m = HX_m_legacy
    F_new = np.asarray(layout.F).astype(np.uint8)
    G_new = np.asarray(layout.G).astype(np.uint8)
    n = original_g.code.num_qudits
    n_anc_new = HX_m.shape[1] - n
    kappa_qubits = tuple(range(n, n + n_anc_new))
    return dataclasses.replace(
        original_g,
        F=F_new, G=G_new,
        HX_merged=HX_m, HZ_merged=HZ_m,
        kappa_qubits=kappa_qubits,
    )


def boost_gadget(
    gadget,
    *,
    method: str,
    target: float,
    seed: int | None = None,
    **kwargs,
):
    """Single entry point for Cheeger / distance boost.

    Args:
        gadget: a GadgetLayout from build_gadget.
        method: 'spectral' | 'combinatorial' | 'distance'.
        target: target Cheeger constant (for spectral / combinatorial) or
            target distance (for distance method; cast via int(target)).
        seed: RNG seed.
        **kwargs: forwarded to the underlying boost function.

    Returns:
        A NEW GadgetLayout with boosted F, G, HX_merged, HZ_merged,
        kappa_qubits.
    """
    merged0, layout0 = _gadget_to_legacy_layout(gadget)
    if method == "spectral":
        boosted_merged, boosted_layout, _ = boost_gadget_cheeger(
            merged0, layout0, target_h=target, seed=seed, **kwargs,
        )
    elif method == "combinatorial":
        boosted_merged, boosted_layout, _ = boost_gadget_cheeger_combinatorial(
            merged0, layout0, target_h=target, seed=seed, **kwargs,
        )
    elif method == "distance":
        boosted_merged, boosted_layout, _ = boost_gadget_distance(
            merged0, layout0, target_distance=int(target), seed=seed, **kwargs,
        )
    else:
        raise ValueError(f"unknown method: {method!r}")
    return _legacy_to_gadget(boosted_merged, boosted_layout, gadget)
