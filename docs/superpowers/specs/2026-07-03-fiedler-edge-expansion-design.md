# Fiedler-sweep edge expansion for large |V| (design)

**Date:** 2026-07-03
**Status:** approved, implementing
**Module:** `src/qldpc/circuits/surgery/hmatrix/edge_expanded.py`

## Problem

`algorithm_1` (arXiv:2410.02753 Alg 1) reaches Cheeger `h ≥ 1` by adding edges,
scoring cuts with the exact parity boundary over **all** `2^|V|` vertex subsets
(`cheeger_constant` / `sparsest_cut`, bit-packed Gray-code sweep). It hard-errors
at `|V| > 26` (`2^26 ≈ 6.7e7` is the practical ceiling). This blocks any
high-weight logical: e.g. the product X̄₁···X̄₉ of the bb_18 [[248,10,18]] code
has support **100**, so `build_gadget` on it raises instead of producing a gadget.

## Approach: restrict the cut family, reuse the scoring

The exponential part is *which* subsets are enumerated, not how a cut is scored.
Keep the exact parity-boundary scoring; change only the candidate cut family.

- **|V| ≤ 26** — unchanged: full `2^|V|` enumeration. Every existing gadget stays
  byte-identical (golden tests green); single-operator gadgets never hit the new
  path.
- **|V| > 26** — **Fiedler sweep cuts**: order vertices by the Fiedler vector
  (2nd eigenvector of the graph Laplacian `L = D − A`, clique-expanded for
  hyperedges), take the `|V|−1` threshold cuts `S_k = {first k in order}` with
  `|S_k| ≤ |V|/2`, each scored with the same exact parity boundary. `O(|V|)` cuts.

## Components

1. `_fiedler_sweep_cuts(incidence)` → candidate cuts (masks + sizes) via
   `numpy.linalg.eigh` on the dense `|V|×|V|` Laplacian (no scipy; |V|≈100 is ms).
2. `cheeger_constant` / `sparsest_cut` → dispatch on `|V|`: exact enumeration
   ≤ 26, else min-ratio / argmin over the Fiedler sweep cuts. (Also stops
   `cheeger_constant` hanging if ever called on a big graph.)
3. `algorithm_1` → dispatch at the top: **≤ 26** runs the existing exact
   incremental loop verbatim (untouched); **> 26** runs a new loop that each
   iteration recomputes the Fiedler sweep cuts on the current graph, stops when
   the best sweep-cut ratio ≥ 1, else adds one edge across the sparsest sweep cut
   between min-degree endpoints (candidate maximizing the swept ratio — the
   paper's rule over the sweep family). Keeps the `max_extra` guard.
4. Remove the `|V| > 26` hard error.

## Guarantee

For the large path `h ≥ 1` is **best-effort** (Fiedler-sweep proxy): the sweep may
miss a sparser cut, so the reported Cheeger is a spectral estimate, not certified.
A `log`/comment states this. Everything ≤ 26 stays certified-exact.

## Testing

- Grid/expander with |V| > 26 and a known good expansion → Fiedler path returns
  `h ≥ 1` with a reasonable added-edge count.
- Small graph (|V| ≤ 20) run through **both** paths → the sweep's cheeger estimate
  brackets the exact value (sanity, sweep not wildly off).
- Existing golden/e2e tests unchanged → proves the ≤ 26 path is byte-identical.
- End-to-end: `build_gadget` on the weight-100 bb_18 product runs, yields a
  CSS-valid gadget; report its actual `(|Q'|, S_X', S_Z')`. (Our representative is
  weight 100, not Cain's weight-104, so the triple need not equal `(189,104,86)`.)
