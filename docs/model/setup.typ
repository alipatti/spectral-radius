= A minimal coupled graph–opinion model

Agents hold positions in a $d$-dimensional opinion space
and are linked by a symmetric weighted social structure.
Each period, agents move toward the positions of their neighbors
and ties strengthen (weaken) between agents with similar (dissimilar) positions.

== Notation

- $X_t in cal(X) = RR^(N times d)$:
  rows are agents' position vectors $x_(i t)$; columns are mean-zero
  ($bold(1)^top X_t = 0$).
- $A_t in SS^N$: symmetric weighted social structure
  (adjacency or similarity matrix).
- $L_t = D_t - A_t in SS^n$ with degree matrix
  $D_t = op("diag")(A_t bold(1))$: network Laplacian (we'll often work
  with this -- the math is nicer).

== Detailed Dynamics

At each time step, agents

+ move towards the positions of their neighbors at some rate $mu > 0$,
+ revert towards their initial dispositions $X_0$ at some rate $kappa > 0$
  (this is needed for stationary and also appears to be true
  #link("https://www2.cs.siu.edu/~hexmoor/classes/CS539-F10/Friedkin.pdf")[empirically]),
+ receive a stationary, mean zero opinion shock $xi_t$
  (this is needed to prevent $X_t$ from collapsing to rank-one).

In the network,

+ ties strengthen between agents with similar positions and weaken between those with dissimilar ones at a rate $rho in (0, 1)$. Similarity is given by the matrix
$
  S_t = X_t X_t^top slash norm(X_t)_F^2.
$ <eqn:similarity-def>

In sum, the update equations are

$
  X_(t+1) & = (1-kappa) (I - mu L_t) X_t + kappa X_0 + xi_t \
  A_(t+1) & = (1-rho) A_t + rho thin S_t
$

// TODO: can we write both these equations in FJ form?


== Quantity of interest

We care about the covariance structure of agents' opinions $Sigma_t = X_t^top X_t \/ N$, and in particular, its largest eigenvalue $lambda_1(Sigma_t) = norm(Sigma_t)_2$.

== Reparameterization

*NOTE:* for now we consider only deterministic processes ($xi_t tilde.op 0$).
I think we can still re-parameterize with noise, but it will no longer be exact. Claude can probably work on this once we understand the noiseless dynamics.

The natural state $(A_t, X_t)$ is difficult to analyze:
the Laplacian $L_t$ depends awkwardly on $A_t$
and $X_t$ enters the network update quadratically via its Gram matrix
$G_t = X_t X_t^top$.
We first establish a few facts:

+ The Laplacian map $L : W |-> op("diag")(W bb(1)) - W$ is linear in $W$.

+ Because $X_t$ has mean-zero columns, $S_t$ has mean-zero rows:
  $
    S_t bold(1)
    = norm(X_t)_F^(-2) X_t (X_t^top bold(1))
    = norm(X_t)_F^(-2) X_t 0
    = 0.
  $
  With the linearity of $L$, this gives $L(S_t) = -S_t$.

// NOTE: what does this mean? claude produced it
+ The mean-zero columns are preserved by the opinion update,
  since $bold(1)^top L_t = 0$ gives
  $bold(1)^top X_(t+1) = (1-kappa) bold(1)^top X_t = 0$.

For the network update, combining (1)
and (2) with the network update equation yields
$
  L_(t+1) & = (1-rho) L_t + rho L(S_t). \
          & = (1-rho) L_t - rho S_t. \
          & = (1-rho) thin L_t - rho thin G_t / (tr G_t),
$
which is linear in $L_t$ and depends only on $L_t$ and $G_t$.
// NOTE: the following is no longer true with FJ opinion shrinkage back to X_0
// CLAUDE: please fix this
For the opinions, defining $P_t = (1-kappa)(I - mu L_t)$
and observing $X_(t+1) = P_t X_t$ gives
$
  G_(t+1) = P_t thin G_t thin P_t,
$
which is linear in $G_t$ and depends only on $L_t$ and $G_t$.

Therefore these two equations together define an evolution map
$f : (L_t, G_t) |-> (L_(t+1), G_(t+1))$.
Crucially, the quantity of interest is recoverable from $G_t$ alone:
the first $d$ eigenvalues of $G_t \/ N$ are exactly the spectrum of $Sigma_t$.

== Model Solution

*TODO:* long term behavior: fixed points, convergence rate, conditions for stability, etc. may want or need to add shrinkage towards $A_0$.

*TODO:* exact short-term dynamics?

== Extensions

- *Microfoundations.*
  Opinion update as best response in a quadratic disagreement game
  (Friedkin–Johnsen; Bindel, Kleinberg & Oren 2015)
  or as Gaussian Bayesian learning;
  network update as optimal attention allocation under a time budget with
  homophilous preferences (Currarini, Jackson & Pin 2009; Sethi & Yildiz 2016).
  Addressing this is essential to publication in econ.

- *Exogenous traits.*
  Each agent has fixed traits
  (e.g. race, sex, location)
  that influence their social network update rule and/or noise shocks.

- *Alternative similarity measures.*
  Cosine similarity, RBF kernel.
  Switching will likely kill analytical tractability.
  (Any similarity without zero row sums reintroduces $D_t$ and I think will
  break the reparameterization).
  Getting polarization from the model will
  (I think) require $S_t$ to sometimes be negative.

- *Network-dependent network updates.*
  Common-neighbor / Katz terms $Phi(A_t) = A_t^2 \/ c$
  (Liben-Nowell & Kleinberg 2007; triadic closure).
  Tractable (polynomial in $A_t$),
  but amplifies existing cuts rather than bridging them,
  consistent with Asikainen et al. (2020).

- *Influencers / broadcasters.*
  Either via explicit two-pool model (influencers/influenced)
  or by allowing endogenous creation of influencers by giving agents with large
  degree more weight in the opinion-update equation.
  Not sure if this will have the effects I want.

- *Degree-dependent updating.* $X_(t+1) = (I - mu D_t^(-beta) L_t) X_t$.
  $beta = 1$ is attention normalization
  (each agent spreads unit attention);
  $beta > 1$ makes well-connected agents update more slowly
  (stubborn influencers).
  Preserves real spectra; center of mass becomes $pi prop d^beta$.
  Justified by precision-weighted (Bayesian) updating.

- *Multi-issue structure.*
  Similarity computed on a subset of issue dimensions or on fixed traits,
  so that cross-cutting cleavages persist.
  Predicts which issues join the dominant axis and which remain off-axis.
