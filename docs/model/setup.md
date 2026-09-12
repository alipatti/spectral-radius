# A minimal coupled graph–opinion model

Agents hold positions in a $d$-dimensional opinion space
and are linked by a symmetric weighted social structure.
Each period, agents move toward the positions of their neighbors
and ties strengthen (weaken) between agents with similar (dissimilar) positions.

## Notation

- $X_t \in \mathcal X = \mathbb{R}^{N \times d}$:
  rows are agents' position vectors $x_{it}$; columns are mean-zero
  ($\mathbf{1}^\top X_t = 0$).
- $A_t \in \mathbb{S}^{N}$: symmetric weighted social structure
  (adjacency or similarity matrix).
- $L_t = D_t - A_t \in \mathbb S^n$ with degree matrix
  $D_t = \mathrm{diag}(A_t \mathbf{1})$: network Laplacian (we'll often work
  with this -- the math is nicer).

## Detailed Dynamics

At each time step, agents

1. move towards the positions of their neighbors at some rate $\mu >0$
2. revert towards their initial dispositions $X_0$ at some rate $\kappa > 0$
   (this is needed for stationary and also appears to be true
   [empirically](https://www2.cs.siu.edu/~hexmoor/classes/CS539-F10/Friedkin.pdf)):
3. receive a stationary, mean zero opinion shock $\xi_t$
   (this is needed to prevent $X_t$ from rank-one collapse,
   which doesn't appear in the data).

In the network, ties strengthen between agents with similar positions
and weaken between those with dissimilar ones at a rate $\rho \in (0, 1)$ with
similarity determined by a similarity matrix $S_t$.
In sum, the update equations are

$$
  X_{t+1} = (1-\kappa) (I - \mu L_t) X_t + \kappa X_0 + \xi_t
$$

$$
  A_{t+1} = (1-\rho) A_t + \rho\, S_t
$$
with
$$
  S_t = X_t X_t^\top / ||X_t||^2_F
$$

<!-- TODO: can we write both these equations in FJ form. computing fixed points will be simpler. (see claude transcript) -->

Stability requires $\mu < 2/\lambda_{\max}(L_t)$.
<!-- TODO: check this. why true? -->

## Quantity of interest

$\Sigma_t = X_t^\top X_t / N$, the $d \times d$ issue covariance,
and in particular $\lambda_1(\Sigma_t) = \|\Sigma_t\|_2$
and the share $\lambda_1 / \mathrm{tr}\,\Sigma_t$.
$\Sigma_t$ shares its nonzero spectrum with the Gram matrix $X_t X_t^\top$,
which is what links the observable issue covariance to the network dynamics.

## Reparameterization

**NOTE:** for now we consider only deterministic processes ($\xi_t \sim 0$).
I think we can still re-parameterize with noise, but it will no longer be exact.
(Claude alluded to a "mean-field" argument.)

The natural state $(A_t, X_t)$ is difficult to analyze:
the Laplacian $L_t$ depends awkwardly on $A_t$
and $X_t$ enters the network update only via its Gram matrix
$G_t = X_t X_t^\top$.
We first establish a few facts:

1. The Laplacian map $L : W \mapsto \diag(W \mathbb 1) − W$ is linear in $W$

2. Because $X_t$ has mean-zero columns, $S_t$ has mean-zero rows:
   $$
     S_t \mathbf 1
       = ||X_t||^{-2}_F X_t (X_t^\top \mathbf 1)
       = ||X_t||^{-2}_F X_t 0
       = 0.
   $$
   Therefore $L(S_t) = -S_t$.

3. The mean-zero columns are preserved by the opinion update,
   since $\mathbf 1^\top L_t = 0$ gives
   $\mathbf 1^\top X_{t+1} = (1-\kappa)\mathbf 1^\top X_t = 0$.

For the network update, combining (1)
and (2) with the network update equation yields
$$
  L_{t+1} = (1-\rho)L_t + \rho L(S_t).
          = (1-\rho)L_t - \rho S_t.
          = (1-\rho)\,L_t - \rho\,\frac{G_t}{\mathrm{tr}\,G_t},
$$
which is linear in $L_t$ and depends only on $L_t$ and $G_t$.

For the opinions, defining $P_t = (1-\kappa)I - \mu L_t$
and observing $X_{t+1} = P_t X_t$ gives
$$
  G_{t+1} = P_t\,G_t\,P_t,
$$
which is linear in $G_t$ and depends only on $L_t$ and $G_t$.

Therefore these two equations together define a bilinear evolution map
$f : (L_t, G_t) \mapsto (L_{t+1}, G_{t+1})$.
Crucially, the quantity of interest is recoverable from $G_t$ alone:
the first $d$ eigenvalues of $G_t/N$ are exactly the spectrum of $\Sigma_t$.

This reparameterization is useful for three reasons.
It removes the degree matrix,
which is the only nonlinear-in-$A_t$ object in the model,
leaving a map on two symmetric matrices whose entries enter linearly
(in $L$) and quadratically (in $G$).
It makes bounds immediate:
Weyl's inequalities on $L_{t+1}$
and $\|P G P\| \le \|P\|^2 \|G\|$ yield a scalar recursion in the extreme
eigenvalues that bounds $\lambda_1(\Sigma_t)$ for any initial condition.
And it identifies the fixed point:
the stationary Gram matrix is diagonal in the eigenbasis of $L$,
so $L_\infty$ and $G_\infty$ commute
and the long-run system reduces to $N-1$ scalar equations,
one per Laplacian mode, regardless of whether $L_0$ and $G_0$ commute.
Commutation at $t=0$ is not required for these results;
it only makes the transient exactly solvable as well.

**TODO:** solve for fixed points. may need to add shrinkage $A_0$.

## Extensions

In rough order of importance

- **Randomness in the reparameterization.**

- **Microfoundations.**
  Opinion update as best response in a quadratic disagreement game
  (Friedkin–Johnsen; Bindel, Kleinberg & Oren 2015)
  or as Gaussian Bayesian learning;
  network update as optimal attention allocation under a time budget with
  homophilous preferences (Currarini, Jackson & Pin 2009; Sethi & Yildiz 2016).
  Addressing this is essential to publication in econ.

- **Exogenous traits.**
  Each agent has fixed traits
  (e.g. race, sex, location)
  that influence their social network update rule and/or noise shocks.

- **Alternative similarity measures.**
  Cosine similarity, RBF kernel.
  Switching will likely kill analytical tractability.
  (Any similarity without zero row sums reintroduces $D_t$ and I think will
  break the reparameterization).
  Getting polarization from the model will
  (I think) require $S_t$ to sometimes be negative.

- **Network-dependent network updates.**
  Common-neighbor / Katz terms $\Phi(A_t) = A_t^2/c$
  (Liben-Nowell & Kleinberg 2007; triadic closure).
  Tractable (polynomial in $A_t$),
  but amplifies existing cuts rather than bridging them,
  consistent with Asikainen et al. (2020).

- **Influencers / broadcasters.**
  Either via explicit two-pool model (influencers/influenced)
  or by allowing endogenous creation of influencers by giving agents with large
  degree more weight in the opinion-update equation.
  Not sure if this will have the effects I want.

- **Degree-dependent updating.** $X_{t+1} = (I - \mu D_t^{-\beta} L_t) X_t$.
  $\beta=1$ is attention normalization
  (each agent spreads unit attention);
  $\beta>1$ makes well-connected agents update more slowly
  (stubborn influencers).
  Preserves real spectra; center of mass becomes $\pi \propto d^\beta$.
  Justified by precision-weighted (Bayesian) updating.

- **Multi-issue structure.**
  Similarity computed on a subset of issue dimensions or on fixed traits,
  so that cross-cutting cleavages persist.
  Predicts which issues join the dominant axis and which remain off-axis.
