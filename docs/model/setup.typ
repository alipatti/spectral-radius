#set table(stroke: (x, y) => (
  left: if x > 0 { 0.5pt } else { 0pt },
  bottom: if y == 0 { 0.5pt } else { 0pt },
))

= A minimal coupled graph–opinion model

Agents hold positions in a $d$-dimensional opinion space
and are linked by a symmetric weighted social structure.
Each state variable adjusts partially toward two targets:
an exogenous anchor (intrinsic position, intrinsic ties)
and an endogenously determined social target
(the neighbor average, opinion similarity).
The two social targets are cross-coupled:
the opinion target is a function of the graph,
and the tie target is a function of the opinions.

== Notation

- $X_t in RR^(N times d)$:
  rows are agents' position vectors $x_(i t)$; columns are mean-zero
  ($bold(1)^top X_t = 0$).
- $A_t in SS^N$: symmetric weighted social structure,
  with exogenous anchor $A_0$ (geography, work, family).
- $L_t = D_t - A_t in SS^N$ with degree matrix
  $D_t = op("diag")(A_t bold(1))$: network Laplacian
  (we'll often work with this -- the math is nicer);
  $L_0 = L(A_0)$.
- $S_t = X_t X_t^top slash norm(X_t)_F^2$:
  opinion similarity (requires $X_t != 0$).

== Dynamics

$
  X_(t+1) & = X_t
            + kappa_0 (X_0 - X_t)
            - kappa thin L_t X_t \
          & = X_t
            + kappa_0 (X_0 - X_t)
            + kappa thin D_t (macron(X)_t - X_t) \
  A_(t+1) & = A_t
            + rho_0 (A_0 - A_t)
            + rho (S_t - A_t)
$ <eqn:dynamics>

The second line rewrites the neighbor pull in the same
target-minus-state form as the tie update:
$macron(X)_t = D_t^(-1) A_t X_t$ stacks the (tie-weighted) average
positions of each agent's neighbors, and the pull toward it is scaled
by degree, so better-connected agents adjust faster.
Both equations are partial adjustment toward a weighted average of an
intrinsic type and a social best response (@tbl:structure).

#figure(
  table(
    columns: 4,
    table.header([], [*State*], [*Anchor (Exogenous)*], [*Social Target*]),

    [opinions], [$X_t$], [$X_0$: prior / intrinsic position], [neighbor mean via $-kappa L_t X_t$],

    [ties], [$A_t$], [$A_0$: intrinsic structure], [$S_t$: opinion similarity],
  ),
  caption: [The shared partial-adjustment structure of the two updates.],
  placement: auto,
) <tbl:structure>

This is the form Friedkin--Johnsen takes as the Nash equilibrium of a
quadratic disagreement game
(#link("https://www2.cs.siu.edu/~hexmoor/classes/CS539-F10/Friedkin.pdf")[and the reversion to $X_0$ appears to be true empirically]);
the tie equation can be microfounded the same way
(cost of deviating from intrinsic ties plus cost of holding ties to
dissimilar people).

#figure(
  table(
    columns: 2,
    table.header([*Parameter*], [*Description*]),
    [$X_0$], [intrinsic opinions],
    [$A_0$], [intrinsic network (geography, work, family)],
    [$kappa_0$], [rate of reversion toward intrinsic positions $X_0$],
    [$kappa$], [rate of attraction toward neighbors' positions],
    [$rho_0$], [rate of reversion toward intrinsic network $A_0$],
    [$rho$], [rate of adjustment toward opinion similarity $S_t$],
    [$kappa_0 \/ kappa$],
    [_opinion stubbornness_: how much intrinsic position resists social
      influence],

    [$rho_0 \/ rho$],
    [_social stubbornness_: how much intrinsic structure resists
      homophilous rewiring],
  ),
  caption: [Model parameters.],
  placement: auto,
) <tbl:params>

A few remarks:

+ We use the unnormalized Laplacian rather than the neighbor average
  $D_t^(-1) A_t X_t$: the normalized version breaks the symmetry of the
  update matrix, does not preserve mean-zero opinions, and $D_t^(-1) L_t$
  is not a function of $L_t$ alone.
  (Degree normalization returns as an extension.)

+ Mean-zero opinion columns are preserved:
  since $bold(1)^top L_t = 0$ and $bold(1)^top X_0 = 0$,
  $
    bold(1)^top X_(t+1) = (1 - kappa_0) thin bold(1)^top X_t,
  $
  which vanishes by induction.
  Consequently $S_t$ has zero row sums
  ($S_t bold(1) = norm(X_t)_F^(-2) X_t (X_t^top bold(1)) = 0$),
  and the Laplacian recursion is affine:
  $
    L_(t+1)
    = (1 - rho_0 - rho) thin L_t + rho_0 thin L_0 - rho thin S_t.
  $

+ Positivity of $A_t$ is conditional, not guaranteed.
  $S_t$ is PSD with unit trace, so $|S_(t, i j)| <= 1 \/ 2$,
  and nonnegativity of $A_(t+1)$ is assured where
  $rho_0 (A_0)_(i j) >= rho \/ 2$.
  Ties can go negative exactly where $A_0$ is sparse
  (anticorrelated strangers).
  We treat this as a feature:
  polarization plausibly requires effectively negative ties.

+ The anchor keeps the network nondegenerate:
  total tie weight obeys
  $bold(1)^top A_(t+1) bold(1)
  = (1 - rho_0) thin bold(1)^top A_t bold(1)
  + rho_0 thin bold(1)^top A_0 bold(1)$
  (the similarity term drops out because $S_t bold(1) = 0$),
  so it converges to $bold(1)^top A_0 bold(1)$
  rather than decaying to zero.

== Quantity of interest

We care about the covariance structure of agents' opinions
$Sigma_t = X_t^top X_t \/ N$, and in particular, its largest eigenvalue
$lambda_1(Sigma_t) = norm(Sigma_t)_2$.

== Model solution

Setting both adjustments in the dynamics to zero
gives the fixed point as the same object twice:
$
  A^* & = (rho_0 A_0 + rho S^*) / (rho_0 + rho), \
  X^* & = kappa_0 (kappa_0 I + kappa L^*)^(-1) X_0.
$
The tie equilibrium is a convex combination of the intrinsic graph and
the opinion-induced graph;
the opinion equilibrium is the resolvent of the equilibrium Laplacian
applied to intrinsic positions.
Note this is a coupled fixed point:
$S^*$ depends on $X^*$ and $L^*$ on $A^*$.
The equilibrium covariance follows directly:
$
  Sigma^*
  = kappa_0^2 / N thin X_0^top (kappa_0 I + kappa L^*)^(-2) X_0,
$
which inherits the full rank of $X_0$
(the resolvent is invertible), so the spectrum is nondegenerate
without any noise.

=== Spectrum of the equilibrium covariance (sketch)

Writing $R = (kappa_0 I + kappa L^*)^(-1)$ and expanding in the
eigenbasis $L^* = sum_k ell_k phi_k phi_k^top$,
$
  Sigma^*
  = kappa_0^2 / N sum_k (c_k c_k^top) / (kappa_0 + kappa ell_k)^2,
  quad c_k = X_0^top phi_k:
$
each network mode contributes the loading of $X_0$ on it, amplified by
the squared resolvent, so the dominant opinion axis is $X_0$'s
projection onto the *softest* modes of the equilibrium network
(small or negative $ell_k$).

Since $s_j (R X_0) in [sigma_min (R), sigma_max (R)] dot s_j (X_0)$,
the whole spectrum is sandwiched mode-wise:
$
  (kappa_0 / (kappa_0 + kappa ell_max))^2 lambda_j (Sigma_0)
  <= lambda_j (Sigma^*)
  <= (kappa_0 / (kappa_0 + kappa ell_min))^2 lambda_j (Sigma_0),
$
with $ell_min, ell_max$ the extreme eigenvalues of $L^*$ on
$bold(1)^perp$ and $Sigma_0 = X_0^top X_0 \/ N$.
Amplification ($ell_min < 0$) requires signed ties, and the upper
bound diverges as $ell_min arrow.b -kappa_0 \/ kappa$.

To eliminate $L^*$: the tie fixed point gives
$L^* = (rho_0 L_0 - rho S^*) \/ (rho_0 + rho)$,
and $0 prec.eq S^* prec.eq I$ (PSD, unit trace) sandwiches $L^*$
between $(rho_0 L_0 - rho I) \/ (rho_0 + rho)$ and
$rho_0 L_0 \/ (rho_0 + rho)$.
On $bold(1)^perp$ the floor of $L_0$ is its Fiedler value
$lambda_2 (L_0)$ (for $A_0 >= 0$), so
$
  lambda_1 (Sigma^*)
  <= ((kappa_0 (rho_0 + rho))
      / (kappa_0 (rho_0 + rho) + kappa (rho_0 lambda_2 (L_0) - rho)))^2
    lambda_1 (Sigma_0),
$
valid when the denominator is positive, i.e. under the sufficient
stability condition
$
  kappa rho < kappa_0 (rho_0 + rho) + kappa rho_0 lambda_2 (L_0).
$ <eqn:stability>
The product of social rates $kappa rho$ (the feedback loop) must be
beaten by the anchors, with the intrinsic network's cohesion
$lambda_2 (L_0)$ entering as a stabilizer.
The slack is the alignment of $S^*$ with the soft mode -- but the
dynamics create that alignment ($S^*$ concentrates on the modes $X^*$
occupies and softens them further), so we expect this condition to be
close to necessary for generic $X_0$.

In plain English: disagreement survives along the weakest seams of the
social fabric -- the ways of splitting society that the network is
least equipped to bridge.
Homophily makes ties follow opinions, which erodes exactly those seams
further, so the model has one central feedback loop:
opinions cluster along a divide, ties dissolve across it, and the
divide deepens.
Whether the loop runs away is decided by three forces:
how stubborn people are about their intrinsic positions,
how anchored relationships are in intrinsic structure
(family, work, geography),
and how cohesive that intrinsic structure is to begin with.
If the anchors win, society ends up looking like a modestly sharpened
version of its intrinsic differences.
If the social forces win, an initially minor difference gets amplified
into a single dominant ideological axis that absorbs opinion variance
-- polarization as a spectral instability.

*TODO:* existence and uniqueness of the coupled fixed point;
convergence rate; conditions for stability.

*TODO:* upgrade the sketch to a proposition:
make the self-consistent alignment argument rigorous and show the
sufficient condition above is (close to) necessary --
the dominant mode runs away when the social forces on ties and
opinions jointly outweigh the two anchors
(a joint statement about the opinion stubbornness $kappa_0 \/ kappa$
and the social stubbornness $rho_0 \/ rho$).

*TODO:* exact short-term dynamics?

== Extensions

- *Microfoundations.*
  The partial-adjustment form is already the shape of a best response:
  opinions as Nash equilibrium of a quadratic disagreement game
  (Friedkin--Johnsen; Bindel, Kleinberg & Oren 2015)
  or Gaussian Bayesian learning;
  ties as optimal attention allocation under a time budget with
  homophilous preferences (Currarini, Jackson & Pin 2009;
  Sethi & Yildiz 2016).
  Remaining work is the formal derivation.
  Addressing this is essential to publication in econ.

- *Noise.*
  Add an opinion shock $sigma thin xi_t$ that is mean zero across
  agents ($bold(1)^top xi_t = 0$, realization by realization, so that
  opinion centering is preserved).
  Not needed for a nondegenerate spectrum -- the anchor already
  prevents rank collapse -- but it buys two things:
  a stationary distribution rather than a fixed point
  (needed to take the model to data), and noise amplification near the
  instability threshold:
  conditional on freezing the graph at $L^*$,
  the per-mode stationary variance is
  $sigma^2 \/ (1 - (1 - kappa_0 - kappa ell_k)^2)$
  for $ell_k$ the $k$-th eigenvalue of $L^*$,
  which diverges as a mode approaches instability --
  an early-warning signal of polarization, and arguably the
  empirically observable version of the threshold.

- *Continuous time.*
  Nothing in the model is essentially discrete; the discrete updates
  are the Euler scheme of the corresponding SDE/ODE system.
  Likely a pure win:
  stability becomes a sign condition on eigenvalues rather than a
  step-size condition,
  the stationary covariance solves a Lyapunov equation,
  and slow-fast machinery (Tikhonov / averaging) applies when the tie
  and opinion timescales separate.
  The tie equation is also linear in the ties given the opinion path,
  so the network can be integrated out exactly --
  an exponentially weighted moving average of past similarity plus the
  anchor -- leaving a single opinion equation with a fading memory
  kernel.

- *Exogenous traits.*
  Each agent has fixed traits
  (e.g. race, sex, location)
  that influence their social network update rule and/or noise shocks.

- *Alternative similarity measures.*
  Cosine similarity, RBF kernel.
  Switching will likely kill analytical tractability:
  any similarity without zero (or constant) row sums reintroduces $D_t$
  into the Laplacian recursion.

- *Network-dependent network updates.*
  Common-neighbor / Katz terms $Phi(A_t) = A_t^2 \/ c$
  (Liben-Nowell & Kleinberg 2007; triadic closure).
  Tractable (polynomial in $A_t$),
  but amplifies existing cuts rather than bridging them,
  consistent with Asikainen et al. (2020).

- *Heterogeneous adjustment rates.*
  Per-agent rates $kappa_(0 i), kappa_i$
  (heterogeneous susceptibility is the original Friedkin--Johnsen
  formulation, and stubborn-agent / zealot models are well studied on
  static networks).
  The coupling adds something new:
  a stubborn agent's position is stable, so similarity to them is
  stable, so ties to them accumulate rather than churn --
  stubborn agents should endogenously accrete degree and become
  community anchors, a testable prediction
  (centrality increasing in stubbornness).
  Two opposed stubborn agents plausibly sustain two-community
  polarization even with nonnegative ties.
  Costs: with rate matrix $K$ not proportional to $I$,
  the update loses symmetry and mean-zero opinions are not preserved
  (same tractability axis as degree normalization, remark (1)).
  The minimal tractable version is two types (stubborn / flexible),
  which keeps block structure and merges naturally with the traits and
  influencer extensions.

- *Influencers / broadcasters.*
  Either via explicit two-pool model (influencers/influenced)
  or by allowing endogenous creation of influencers by giving agents with large
  degree more weight in the opinion-update equation.
  Not sure if this will have the effects I want.

- *Degree-dependent updating.* $X_(t+1) = (I - kappa D_t^(-beta) L_t) X_t$.
  $beta = 1$ is attention normalization
  (each agent spreads unit attention);
  $beta > 1$ makes well-connected agents update more slowly
  (stubborn influencers).
  Preserves real spectra; center of mass becomes $pi prop d^beta$.
  Justified by precision-weighted (Bayesian) updating.
  Note the caveats in remark (1) above.

- *Multi-issue structure.*
  Similarity computed on a subset of issue dimensions or on fixed traits,
  so that cross-cutting cleavages persist.
  Predicts which issues join the dominant axis and which remain off-axis.
