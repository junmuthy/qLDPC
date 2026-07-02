# Spec: replace Cohen Y-merge with CHRY $|Y_+\rangle$-free gauging measurement

Goal: measure mixed joint Paulis ($\bar Z_l\otimes\bar X_r$, $\bar X_l\otimes\bar Z_r$, and
$\bar Y\otimes\cdots$) **without** preparing any qubit in $|Y_+\rangle$ and **without** the
final-round null-space detector synthesis. Keep our existing **1 ancilla layer + $d$ rounds**
structure (no spatial layers).

Primary source: Cross, He, Rall, Yoder, *Improved QLDPC Surgery: Logical Measurements and
Bridging Codes*, arXiv:2407.18393 — §3.6 (joint), **§3.7 "Gauge-fixed Y Ancilla System" +
Thm 20** (mixed), §3.8 (bridging different codes). Improves on Cohen–Kim–Bartlett–Brown
arXiv:2110.10794 (which needs $2d-1$ layers and the $|Y_+\rangle$ adapter).

---

## 0. The one idea

Cohen puts the X/Z mixing **into a qubit's state** (the adapter qubit → forced $|Y_+\rangle$,
leaving a $\prod_q Y_{a_q}$ residual in obs0 → needs Y-detach + null-space detectors).

CHRY puts the X/Z mixing **into a single non-CSS *check*** (the mixed-type check $q_1$, plus
mixed bridge gauge checks $U^B$). Every **qubit** stays in a CSS eigenstate ($|0\rangle$ or
$|+\rangle$), measured in $X$ or $Z$. obs0 is the gauging product $\bar m=\prod_v m_v$ of the
in-circuit vertex-check outcomes — no destructive-readout residual, hence **no null-space**.

---

## 1. Verified building blocks (from arXiv:2407.18393)

### 1a. Single-side gauging measurement (§3.1–3.2) — we already have this (our §2 gadget)
- Ancilla graph: vertex checks $V$ act as **$X$ on edge qubits** (for an X-system; $Z$ for a
  Z-system), $Z$-checks are cycles $U$.
- Protocol (= the A.4 protocol in arXiv:2506.03094): init interface qubits $C_1$ in
  $|0\rangle$; measure all deformed-code checks for $R\ge d$ rounds; $\bar m=\prod_{v}m_v$;
  split by measuring $C_1$ in $Z$; path correction with $L_q$.

### 1b. Same-basis joint $\bar X_1\bar X_2$ (§3.6) — also already have (our §3 bridge)
- Add $|B|=\min(|\bar X_1|,|\bar X_2|)$ **bridge qubits** $B$, each checked by one $V_{j_1}^{(1)}$
  and one $V_{j_2}^{(2)}$ check (matrices $S^{(1)},S^{(2)}$: one 1 per column, $\le$1 per row).
- Bridge gauge checks $U^B$: $|B|-1$ of them, each on exactly two bridge qubits (Lemma 15).
- **Init $B$ in $|0\rangle$, measure in $Z$** (both systems are X). $R\ge d$ rounds.
- Encodes $k-1$ logicals (Lemma 16); fault distance $\ge d$ (Thm 17). §3.8 Thm 24: same works
  across two **different** codeblocks $G_1,G_2$.

### 1c. Mixed $\bar Y=\bar X\cdot\bar Z$, single code, overlap on one qubit $q_0$ (§3.7, Fig. 5)
This is the key new piece. Structure (Figure 5):
- Top half = X-system (layers $V_i^X$, qubits $C_i^Z$), bottom half = Z-system ($V_i^Z$,
  $C_i^X$), joined on the **first layer** by bridge qubits $B$ and **one mixed-type check
  $q_1$** connected to the overlap qubit $q_0$.
- Connectivity matrix (Eq. 54), rows = checks $\{q_1, V_1^Z, V_1^X\}$, cols = qubits
  $\{C_1^X, C_1^Z, B\}$:
  ```
  G = [ fZ^T   fX^T   0  ]   <- q1   (mixed: Z-part on C1^X via fZ^T, X-part on C1^Z via fX^T)
      [ FZ^T    0     SZ ]   <- V1^Z (Z-checks; touch C1^X and bridge B via SZ)
      [  0     FX^T   SX ]   <- V1^X (X-checks; touch C1^Z and bridge B via SX)
  ```
  Bridge gauge checks $U^B$ = solutions $(v_X,v_Z,v_B)$ of $G(v_X\,v_Z\,v_B)^T=0$:
  $X(v_X\in C_1^X)\,Z(v_Z\in C_1^Z)\,X(v_B\in B)$ — **mixed gauge checks**. With $|Q_1|=1$
  (single $q_1$) there are exactly $|B|$ gauge checks and **no extra gauge qubits** (Lemma 18,
  Thm 20: encodes $k-1$ qubits).
- **Initialization (Thm 21 proof) — the whole point:**
  > "the additional qubits starting in $|+\rangle$ for the bottom half of Figure 5 and the
  > bridge and in $|0\rangle$ for the top half"
  i.e. **X-system qubits $|0\rangle$ (measure $Z$); Z-system qubits + bridge $|+\rangle$
  (measure $X$). NO $|Y_+\rangle$, no Y-basis.**
  Init group (Eq. 56): $\langle Z(v):v\in C^Z_{j\,\text{odd}},\ X(v):v\in C^X_{j\,\text{odd}},
  \ X(b):b\in B\rangle$.
- $q_1$ is measured each round like any other check, via a syndrome ancilla with **mixed
  CX/CZ couplings** (it's a non-CSS check). The data/bridge qubits are never in a Y state.
- obs0 $=\bar m=\prod_v m_v$ over the X- and Z-system vertex checks (in-circuit), $R\ge d$.

---

## 2. RESOLVED — mixed via per-qubit local basis change (SJOY24, arXiv:2410.03628)

SJOY24 §II ("Auxiliary graph LDPC surgery") settles it. The universal adapter is **only ever
defined for a Z-type operator**, and arbitrary Paulis are handled by a **local basis change**:

> "we assume $Z$ is a Z-type Pauli operator, which is **without loss of generality if we choose
> the appropriate local basis for each qubit of $L$ and allow the code to be non-CSS**."

Protocol (SJOY24 §II, four steps): (1) init all edge qubits in **$|+\rangle$**, (2) measure
deformed-code checks $\ge d$ rounds, (3) measure edge qubits in the **$X$ basis**, (4) path
Pauli correction. Vertex checks $A_v=Z(q)\prod_{e\ni v}Z(e)$ (Eq. 7), cycle checks
$B_c=\prod_{e\in c}X(e)$ (Eq. 8). **No $|Y_+\rangle$, no Y-basis, no $\prod Y$ residual.**

### How this gives $\bar Z_l\otimes\bar X_r$ with no $q_1$ and no Cohen merge
The two operators are on **disjoint** codes, so **no physical qubit carries both X and Z** (the
$q_0$/$q_1$ machinery of CHRY §3.7 is only needed for the single-code overlap = genuine
single-qubit $\bar Y$). Therefore:

1. Put $\bar X_r$ into Z-type form by a **local Hadamard frame on the X-side**. Two options:
   - **(preferred, fully CSS)** transversal physical $H$ on **all** of code $r$ → code $r$ becomes
     its dual $r^\perp$ ($H_X^r\!\leftrightarrow\!H_Z^r$, still CSS), and $\bar X_r\to\bar Z_{r^\perp}$.
     Then measure $\bar Z_l\otimes\bar Z_{r^\perp}$ as an **ordinary same-basis ZZ merge**
     (our existing §3 path), edges $|+\rangle$/$X$. Undo $H$ (or track) at the end; the outcome
     equals the eigenvalue of $\bar Z_l\otimes\bar X_r$ since $H\bar Z_{r^\perp}H=\bar X_r$.
   - **(alt, non-CSS)** local $H$ only on $\mathrm{supp}(x_r)$ → code $r$ non-CSS on those
     qubits (mixed checks, measured via CY/CX+CZ), but adapter still CSS. Use if transversal-$H$
     dual layout is inconvenient.
2. Run the **existing same-basis bridge + SkipTree verbatim** on the now-uniform-basis pair.
   SkipTree acts on the port/repetition-code graph ($T\,G\,P=H_C$) and is **basis-agnostic** —
   used identically for both sides, no change needed.

Net: mixed = "rotate X-side to Z-type, then same-basis surgery." This is *option 1* from the
earlier discussion but with a **local/transversal** $H$ (free, depth-1), **not** an expensive
logical Hadamard. $|Y_+\rangle$, $\prod Y_{a_q}$, Y-detach, and the null-space detector all
vanish; SkipTree is untouched.

### Hard constraint
**Must reuse the current SkipTree** (`bridge.py` `_run_skiptree_on_port_subgraph`, Swaroop
Thm 7) unchanged. The rotation is upstream of the auxiliary-graph/adapter build, so SkipTree
sees an ordinary Z-type port graph on both sides.

---

## 3. Mapping to our code

| CHRY object | our object |
|---|---|
| X/Z-system ancilla graph, vertex/cycle checks | `gadget.py`: `incidence` ($H'$), `gauge` ($\ker$), `support` ($V_0$) |
| bridge qubits $B$, $S^{(1)}/S^{(2)}$ | `bridge.py`: adapter $\mathcal A$, port/label maps, $H_R$ |
| same-basis bridge (§3.6) | our same-basis `_stitch_intercode` (CSS, already $|Y_+\rangle$-free) |
| mixed $q_1$ + $U^B$ + CSS init (§3.7) | **replaces** `_stitch_to_joint_code_mixed` + the `|Y_+\rangle`/`MY` path in `circuit.py` |

Central change: the mixed path becomes **"rotate X-side to Z-type (local/transversal $H$) →
reuse the same-basis bridge path."** Concretely in `circuit.py` / `joint_layout.py`:
- Replace `_stitch_to_joint_code_mixed` + the Cohen merge with: apply the $H$ frame on the
  X-side (prefer transversal $H$ on all of code $r$ → dual code, fully CSS), then route to the
  **existing same-basis stitch/bridge** (`_stitch_intercode`).
- bridge init: `_bridge_init_pauli = Pauli.Y` → CSS (`|+\rangle`, measure $X$, per SJOY24 four
  steps); drop `RY`/`MY`; emit the $H$ frame gates on the X-side data at init and detach.
- obs0: `∏ m_v` (in-circuit vertex checks only); **drop the `∏ Y_{a_q}` adapter term** — no
  residual.
- detectors: adapter-Y-vs-cycle conflict gone → **delete the `null_space()` combo block**
  (`circuit.py` ~1749); plain per-row + inter-round detectors suffice.
- SkipTree (`bridge.py`): **unchanged** (hard constraint).
- only if using the non-CSS alt (local $H$ on `supp(x_r)` only): code $r$'s rotated checks are
  mixed → measure via CY/CX+CZ. Avoid unless transversal-$H$ dual layout is impractical.

---

## 4. Validation-first plan (don't claim done without these)

1. Build the disjoint-mixed merged symplectic check matrices (per §2 resolution above).
2. GF(2) checks: center commutes; $\bar Z_l\otimes\bar X_r$ is a deterministic stabilizer of
   the center; individual $\bar X_l,\bar Z_r$ remain free logicals (the soundness battery we
   already ran for the Cohen version).
3. Build circuit, confirm: (a) DEM compiles deterministically, (b)
   `test_mixed_basis_dem_has_no_undetectable_observable_error` passes **without** any
   null-space detectors, (c) operational distance $\ge$ Cohen version.
4. Only then delete the Cohen `|Y_+\rangle` + null-space code.

## 5. Honesty ledger (mistakes corrected during analysis)
- main.tex §4.5 over-attributed $|Y_+\rangle$ prep to arXiv:2110.10794 — that paper only gives
  the $y=i\,v_X v_Z$ generator merge, not the prep/measure basis. (Already flagged; fix
  pending.)
- "X⊗Z native mixed merge / Y via Clifford / d layers" framing was imprecise — corrected: it's
  the WY24/CHRY gauging measurement; Y is native via $L_q$; layers are an unrelated (spatial
  FT) axis we don't use.
