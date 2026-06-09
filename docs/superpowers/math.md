qLDPC Surgery 实现的数学总结

  0. 输入与符号

  CSS code on $n$ data qubits with parity-check matrices $H_X \in \mathbb{F}_2^{m_X \times n}$, $H_Z \in
  \mathbb{F}_2^{m_Z \times n}$ satisfying $H_X H_Z^\top = 0$.

  Logical operator $\bar{X}_M$ specified by binary support vector $\mathbf{x} \in \mathbb{F}_2^n$ with
  $H_Z \mathbf{x} = 0$, $\mathbf{x} \notin \mathrm{rowspan}(H_X)$.

  ---
  1. Single-gadget (intra-code) — build_layered_surgery_code

  1.1 Step 1：restriction matrix $F$

  $$
  V_0 := \mathrm{supp}(\mathbf{x}) \subseteq [n], \quad C_0 := {j : (H_Z){j,\cdot}|{V_0} \neq 0} =
  \mathrm{supp}(H_Z \mathbf{1}{V_0})|{\text{nonzero rows}}.
  $$

  Wait, more carefully: $C_0 = {j : \exists i \in V_0, (H_Z)_{ji} = 1}$.

  $$
  F := (H_Z)_{C_0, V_0} \in \mathbb{F}_2^{|C_0| \times |V_0|}.
  $$

  Key invariant (from $H_Z \mathbf{x} = 0$ restricted to $C_0$):
  $$
  F \cdot \mathbf{1}_{V_0} = \mathbf{0} \quad \Longleftrightarrow \quad |S_j \cap V_0| \equiv 0 \pmod{2}
  ;; \forall j \in C_0.
  $$

  1.2 Step 2：gauge-fix basis $G$

  $$
  G \in \mathbb{F}_2^{r \times |C_0|}, \quad \mathrm{rowspan}(G) = \ker(F^\top), \quad r = |C_0| -
  \mathrm{rank}(F).
  $$

  i.e. $G F = 0$ over $\mathbb{F}_2$.

  1.3 Step 3：layer structure ($L$ odd)

  Per-layer qubit counts:

  ┌─────────────────────────────────┬──────┬─────────────────────────────────────────────────────────┐
  │            Layer $i$            │ Size │                          Type                           │
  ├─────────────────────────────────┼──────┼─────────────────────────────────────────────────────────┤
  │ 0 (data)                        │ $    │ V_0                                                     │
  ├─────────────────────────────────┼──────┼─────────────────────────────────────────────────────────┤
  │ 1, 3, …, $L$ (odd)              │ $    │ C_0                                                     │
  ├─────────────────────────────────┼──────┼─────────────────────────────────────────────────────────┤
  │ 2, 4, …, $L-1$ (even, $\geq 2$) │ $    │ V_0                                                     │
  ├─────────────────────────────────┼──────┼─────────────────────────────────────────────────────────┤
  │ gauge-fix                       │ —    │ $r$ syndrome ancillas (not separate qubits in our impl) │
  └─────────────────────────────────┴──────┴─────────────────────────────────────────────────────────┘

  Total ancilla qubits: $\kappa = \lceil L/2 \rceil \cdot |C_0| + \lfloor L/2 \rfloor \cdot |V_0|$.

  1.4 Step 4：merged check matrices

  For $L=1$ (Webster-style 3-step gadget), with new qubits $\kappa_j$ for $j \in C_0$ (size $|C_0|$):

  $H_X^{\text{merged}}$ — rows × columns $= (m_X + |V_0|) \times (n + |C_0|)$:
  $$
  H_X^{\text{merged}} = \begin{pmatrix} H_X & \mathbf{0}{m_X \times |C_0|} \ E{V_0}^\top & F^\top
  \end{pmatrix}
  $$

  where $E_{V_0} \in \mathbb{F}2^{n \times |V_0|}$ is the inclusion matrix of $V_0$, so $E{V_0}^\top$ has
  a single 1 per row at the position of $q_i \in V_0$. Each row $i$ of $E_{V_0}^\top$ corresponds to one
  $\chi$-check at data qubit $q_i$, with $F^\top$ giving its connection to the $\kappa_j$ ancillas.

  $H_Z^{\text{merged}}$ — rows × columns $= (m_Z + r) \times (n + |C_0|)$:
  $$
  H_Z^{\text{merged}} = \begin{pmatrix} H_Z & \tilde{F} \ \mathbf{0}_{r \times n} & G \end{pmatrix}
  $$

  where $\tilde{F} \in \mathbb{F}2^{m_Z \times |C_0|}$ is $F$ embedded back into the full $H_Z$ row space:
   $\tilde{F}{j, k} = 1$ iff row $j$ of $H_Z$ is the $k$-th row of $C_0$.

  1.5 Correctness invariants

  (a) CSS commutation — $H_X^{\text{merged}} (H_Z^{\text{merged}})^\top = 0$:
  $$
  \begin{aligned}
  H_X \tilde{F}^\top + 0 \cdot G^\top &= 0 \quad &\text{(top-right block of product)} \
  E_{V_0}^\top H_Z^\top + F^\top \tilde{F}^\top &= 0 \quad &\text{(by } F = H_Z|_{C_0,V_0}\text{)} \
  0 \cdot H_Z^\top + F^\top G^\top &= (G F)^\top = 0 \quad &\text{(by definition of } G\text{)}
  \end{aligned}
  $$

  (b) $k$ preserved: gadget alone has same logical dimension as data code (the $\chi$-rows together with
  gauge-fix $G$-rows add and remove exactly matched degrees of freedom).

  ---
  2. Joint measurement — universal adapter (arXiv:2410.03628 §IV / §VII)

  2.1 Adapter width and port subsets

  Given two GadgetLayouts (g_l, g_r) measuring X̄_l, X̄_r (or Z̄_l, Z̄_r) we pick port subsets
  𝒫_l* ⊆ V_0^(l), 𝒫_r* ⊆ V_0^(r) of equal size w = min(|V_0^(l)|, |V_0^(r)|), and a bijection
  𝒜: 𝒫_l* → 𝒫_r*. Each adapter edge ∈ 𝒜 is one new "adapter" data qubit.

  2.2 Auxiliary graph augmentation

  For each side build 𝒢_s = (V_0^(s), {edges = weight-2 F_s rows}). When 𝒢_s[𝒫_s*] is
  disconnected, add weight-2 edges (= new κ qubits) until connected. Cellulate basis cycles
  to length ≤ max_len.

  When F has rows of weight ≥ 4 (hyperedges, even-weight forced by F·1_{V_0}=0),
  they are kept in F_aug so the gadget structure is unchanged but skipped in the
  auxiliary graph 𝒢_s. SkipTree assigns T_s zero columns to hyperedge rows
  (existing skip at _run_skiptree_on_port_subgraph), so the SkipTree key identity
  T_s · F_aug · P_s = H_R reduces to its restriction onto the weight-2 sub-
  incidence and holds automatically. CSS commutation, κ-cancellation, joint
  observable, and dim−1 all hold by direct calculation. Paper Eq. 9's perfect-
  matching decomposition (§II.C) is not applied; the structural distance
  argument of Theorem 12 is replaced by empirical LER smoke tests, per the
  paper's own remark at the end of §IV.

  2.3 SkipTree key identity

  Run SkipTree (paper §III Algorithm 1) on 𝒢_s_aug to obtain T_s ∈ F_2^{(w-1) × |E_aug|},
  P_s ∈ F_2^{|V_0^(s)| × w} satisfying

    T_s · G_s_aug · P_s = H_R    (canonical full-rank rep-code parity, (w-1) × w)

  where G_s_aug = F_aug^(s) is the auxiliary-graph incidence matrix.

  2.4 Merged check matrices (basis=X, inter-code)

  H_X^merged blocks (rows × support):
     data H_X^(l)     : m_X^(l) × data_l
     data H_X^(r)     : m_X^(r) × data_r
     χ^(l)            : |V_0^(l)| × (data_l + κ_l_aug + adapter)  via E_V0^T, F_aug^(l)^T, Π_l
     χ^(r)            : |V_0^(r)| × (data_r + κ_r_aug + adapter)  via E_V0^T, F_aug^(r)^T, Π_r

  H_Z^merged blocks:
     data H_Z^(l) ext : m_Z^(l) × (data_l + κ_l_aug)
     data H_Z^(r) ext : m_Z^(r) × (data_r + κ_r_aug)
     G^(l)_aug        : r_l × κ_l_aug
     G^(r)_aug        : r_r × κ_r_aug
     new cycle-Z      : (w-1) × (κ_l_aug + κ_r_aug + adapter)  via [T_l | T_r | H_R]

  Π_s ∈ F_2^{|V_0^(s)| × w} satisfies Π_s[v, k] = 1 iff v ∈ 𝒫_s* and label_s(v) = k.

  basis=Z is the symmetric X↔Z dual; intra-code merges the two data column blocks.

  2.5 Commutation (CSS)

  The only non-trivial pairing is χ^(s) vs new cycle-Z^(s'). For s = s':
    (χ_v on κ + adapter) · (cycle_c on κ + adapter)
      = (T_s · F_aug^(s))[c, v] + H_R[c, label_s(v)] · [v ∈ 𝒫_s*]
      = (T_s · G_s_aug · P_s)[c, v]                       (by SkipTree identity)
      = H_R[c, k] − H_R[c, k]                              (when label_s(v) = k)
      = 0.
  For v ∉ 𝒫_s*, both halves are zero (T_s has zero columns on edges outside 𝒢_s[𝒫_s*]).

  2.6 Joint observable (α* derivation)

  α* picks Σ χ^(l) + Σ χ^(r). On the merged register:
     data side: 1_{V_0^(l)} + 1_{V_0^(r)} = x_l + x_r (XOR support, joint X̄_l X̄_r).
     κ side:    F_aug^(s)^T · 1_{V_0^(s)} = 0          (κ-cancellation).
     adapter:   Σ_{v ∈ 𝒫_l*} e_{label_l(v)} + Σ_{v ∈ 𝒫_r*} e_{label_r(v)} = 1_𝒜 + 1_𝒜 = 0.

  New cycle-Z rows are not in α* (they're Z-type; orthogonal to the X-type joint observable).

  ---
  3. Cheeger boost — boost_gadget_cheeger

  3.1 Spectral lower bound

  For $F \in \mathbb{F}2^{|C_0| \times |V_0|}$ viewed as bipartite incidence matrix, lift to integer
  matrix $F\mathbb{R}$. The boundary Cheeger constant satisfies
  $$
  h(F) \geq \frac{\lambda_2(L_F)}{2}, \quad L_F = F_\mathbb{R}^\top F_\mathbb{R} - D_{V_0}
  $$
  where $\lambda_2$ is the second-smallest eigenvalue and $D_{V_0}$ is the diagonal of qubit degrees.

  3.2 Random augmentation

  Add $\Delta$ new $\kappa$ qubits, each with a random sparse extension column appended to $F$:
  $$
  F \mapsto \begin{pmatrix} F & R \end{pmatrix}, \quad R \in \mathbb{F}_2^{|C_0| \times \Delta}.
  $$

  Iteratively sample $R$, recompute $\lambda_2(F^\top F)/2$, accept if it increases. Tracked via
  BoostResult.terminated_by ∈ {target_reached, max_qubits_exhausted, no_progress}.

  ---
  4. 验证矩阵（完整证据链）

  ┌──────────────────────┬────────────────────────────────────────────────────────────┬──────────────────────────────────────────────────────────────────┐
  │      Invariant       │                         Statement                          │                           验证手段                               │
  ├──────────────────────┼────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────┤
  │ CSS commutation      │ H_X^merged · H_Z^merged^T = 0                              │ 直接 GF(2) 矩阵乘                                                │
  ├──────────────────────┼────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────┤
  │ SkipTree identity    │ T_s · F_aug · P_s = H_R                                    │ _test.py test_skip_tree_fullrank_on_K4 +                          │
  │                      │                                                            │ test_build_bridge_skiptree_invariant_holds                        │
  ├──────────────────────┼────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────┤
  │ Adapter cycle weight │ new cycle-Z rows weight ≤ 8                                │ test_adapter_cycle_check_weight_bounded                           │
  ├──────────────────────┼────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────┤
  │ Cellulation cap      │ basis cycles ≤ cellulate_max_len                           │ test_cellulation_caps_aug_aux_cycle_length_on_webster             │
  ├──────────────────────┼────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────┤
  │ Joint membership     │ (x_l + x_r) ⊗ 0 ∈ rowspan(H_X^merged)                    │ test_stitch_*_joint_logical_in_stabilizer                         │
  ├──────────────────────┼────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────┤
  │ Singletons excluded  │ x_s ⊗ 0 ∉ rowspan(H_X^merged)                             │ test_stitch_*_singletons_excluded                                 │
  ├──────────────────────┼────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────┤
  │ Dim count            │ k_joint = k_l + k_r - 1 (inter) or k - 1 (intra)          │ test_stitch_*_k_reduces_by_one                                    │
  ├──────────────────────┼────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────┤
  │ LER monotone         │ LER(p) non-increasing as p decreases                       │ test_joint_ppm_ler_monotone_steane_intercode                      │
  └──────────────────────┴────────────────────────────────────────────────────────────┴──────────────────────────────────────────────────────────────────┘