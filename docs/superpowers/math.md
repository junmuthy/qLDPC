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
  2. Joint measurement — build_joint_measurement_code

  2.1 Bridge data qubits

  Given two logical-X support vectors $\mathbf{x}_1, \mathbf{x}_2 \in \mathbb{F}_2^n$ with $V_0^{(1)} =
  \mathrm{supp}(\mathbf{x}_1)$, $V_0^{(2)} = \mathrm{supp}(\mathbf{x}_2)$ (Webster: disjoint):
  $$
  w := \min(|V_0^{(1)}|, |V_0^{(2)}|).
  $$

  Introduce $w$ new bridge data qubits $b_0, b_1, \ldots, b_{w-1}$.

  2.2 Bridge $X$-stabilizers $U_B$

  Path graph on bridge qubits, $w-1$ rows:
  $$
  U_B = \begin{pmatrix} 1 & 1 & 0 & \cdots & 0 \ 0 & 1 & 1 & \cdots & 0 \ \vdots & & \ddots & \ddots &
  \vdots \ 0 & \cdots & 0 & 1 & 1 \end{pmatrix} \in \mathbb{F}_2^{(w-1) \times w}.
  $$

  Telescoping property:
  $$
  \sum_{i=0}^{w-2} (U_B)_{i,\cdot} = \mathbf{e}0 + \mathbf{e}{w-1} \in \mathbb{F}_2^w.
  $$

  2.3 Endpoint $\chi$-extension

  In each gadget, pick row $\chi_0^{(s)}$ (first $\chi$ row in $V_0^{(s)}$) and extend with $X$ on the
  corresponding bridge endpoint:
  $$
  \chi_0^{(1)} \mapsto \chi_0^{(1)} \otimes X_{b_0}, \qquad \chi_0^{(2)} \mapsto \chi_0^{(2)} \otimes
  X_{b_{w-1}}.
  $$

  2.4 Merged register layout

  $$
  \underbrace{n}{\text{data}} + \underbrace{|C_0^{(1)}|}{\kappa^{(1)}} +
  \underbrace{|C_0^{(2)}|}{\kappa^{(2)}} + \underbrace{w}{\text{bridge}} = n_{\text{merged}}.
  $$

  2.5 $H_X^{\text{joint}}$ — block form

  Stack of three blocks:
  $$
  H_X^{\text{joint}} = \begin{pmatrix}
  H_X & 0 & 0 & 0 \
  E_{V_0^{(1)}}^\top & F^{(1)\top} & 0 & \mathbf{e}0 \cdot \delta{i=0} \
  E_{V_0^{(2)}}^\top & 0 & F^{(2)\top} & \mathbf{e}{w-1} \cdot \delta{i=0} \
  0 & 0 & 0 & U_B
  \end{pmatrix}
  $$

  where $\delta_{i=0}$ means only the row $\chi_0^{(s)}$ carries the bridge column entry.

  2.6 $H_Z^{\text{joint}}$ — block form (κ-splicing)

  For each data Z-check $j \in [m_Z]$:
  $$
  H_Z^{\text{joint}}[j, :n] = (H_Z){j,\cdot}, \quad H_Z^{\text{joint}}[j, n:n+|C_0^{(1)}|] =
  \tilde{F}^{(1)}{j,\cdot}, \quad H_Z^{\text{joint}}[j, n+|C_0^{(1)}|:n+|C_0^{(1)}|+|C_0^{(2)}|] =
  \tilde{F}^{(2)}_{j,\cdot}.
  $$

  Gauge-fix rows for both gadgets stacked on $\kappa^{(1)}, \kappa^{(2)}$ blocks separately:
  $$
  \begin{pmatrix} 0 & G^{(1)} & 0 & 0 \ 0 & 0 & G^{(2)} & 0 \end{pmatrix}.
  $$

  No Z-extension onto bridge qubits.

  2.7 Cross §3.6 protocol formula (verified)

  Define $\alpha^* \in \mathbb{F}_2^{m_X^{\text{joint}}}$ by
  $$
  \alpha^*_r = \begin{cases} 0 & r \in \text{data } H_X \text{ rows} \ 1 & r \in \chi^{(1)} \cup
  \chi^{(2)} \cup U_B \end{cases}
  $$

  Then
  $$
  \alpha^{*\top} H_X^{\text{joint}} = (\mathbf{x}1 + \mathbf{x}2, \mathbf{0}{\kappa^{(1)}},
  \mathbf{0}{\kappa^{(2)}}, \mathbf{0}_w).
  $$

  Derivation:
  $$
  \begin{aligned}
  \sum_{i \in V_0^{(1)}} \chi_i^{(1)} &= \mathbf{x}1 \otimes \underbrace{F^{(1)\top}
  \mathbf{1}{V_0^{(1)}}}{= 0 \text{ since } F^{(1)} \mathbf{1} = 0} \otimes X{b_0} \
  &= \mathbf{x}1 \otimes \mathbf{0}{\kappa^{(1)}} \otimes X_{b_0}. \
  \sum_i \chi_i^{(2)} &= \mathbf{x}2 \otimes \mathbf{0}{\kappa^{(2)}} \otimes X_{b_{w-1}}. \
  \sum U_B \text{ rows} &= \mathbf{0} \otimes \mathbf{0} \otimes \mathbf{0} \otimes (X_{b_0} +
  X_{b_{w-1}}). \
  \text{Sum} &= (\mathbf{x}_1 + \mathbf{x}_2) \otimes \mathbf{0} \otimes \mathbf{0} \otimes \mathbf{0}.
  \quad \blacksquare
  \end{aligned}
  $$

  This is exactly $\bar{X}_1 \otimes \bar{X}_2$ as an operator on the merged register.

  2.8 Logical-qubit reduction

  $$
  k_{\text{joint}} = k_{\text{data}} - 1
  $$

  (Cross §3.6: one logical dof consumed by $\bar{X}_1 \bar{X}_2$ measurement). Verified via
  CSSCode.dimension.

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

  ┌──────────────────┬───────────────────────────────────────────────────────┬───────────────────────┐
  │    Invariant     │                       Statement                       │       验证手段        │
  ├──────────────────┼───────────────────────────────────────────────────────┼───────────────────────┤
  │ CSS commutation  │ $H_X^{\text{merged}} H_Z^{\text{merged}\top} = 0$     │ 直接 GF(2) 矩阵乘     │
  ├──────────────────┼───────────────────────────────────────────────────────┼───────────────────────┤
  │ κ-cancellation   │ $F \mathbf{1}_{V_0} = 0$                              │ 推出 $\sum \chi_i$ 在 │
  │                  │                                                       │  κ 上为 0             │
  ├──────────────────┼───────────────────────────────────────────────────────┼───────────────────────┤
  │ Bridge           │ $\sum U_B = \mathbf{e}0 + \mathbf{e}{w-1}$            │ 直接构造              │
  │ telescoping      │                                                       │                       │
  ├──────────────────┼───────────────────────────────────────────────────────┼───────────────────────┤
  │ Joint membership │ $\bar{X}_1 \bar{X}_2 \otimes \mathbf{0} \in           │ $\mathrm{rank}$ check │
  │                  │ \mathrm{rowspan}(H_X^{\text{joint}})$                 │                       │
  ├──────────────────┼───────────────────────────────────────────────────────┼───────────────────────┤
  │ Singletons       │ $\bar{X}_s \otimes \mathbf{0} \notin                  │ $\mathrm{rank}$ check │
  │ excluded         │ \mathrm{rowspan}$                                     │                       │
  ├──────────────────┼───────────────────────────────────────────────────────┼───────────────────────┤
  │ Protocol formula │ $\alpha^{*\top} H_X^{\text{joint}} = (\mathbf{x}_1 +  │ 直接矩阵乘            │
  │                  │ \mathbf{x}_2, \mathbf{0})$                            │                       │
  ├──────────────────┼───────────────────────────────────────────────────────┼───────────────────────┤
  │ Dim count        │ $k_{\text{joint}} = k_{\text{data}} - 1$              │ CSSCode.dimension     │
  ├──────────────────┼───────────────────────────────────────────────────────┼───────────────────────┤
  │ Webster Table I  │ $\kappa + \chi + r = {19, 31, 49, 79}$, $2w - 1 =     │ numerical exact match │
  │ sizes            │ {11, 19, 31, 51}$                                     │                       │
  └──────────────────┴───────────────────────────────────────────────────────┴───────────────────────┘