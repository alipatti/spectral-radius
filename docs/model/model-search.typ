#import "@ali/template:0.1.0" as ali
#show: ali.template

= Model search
A living document.
Started 2026-10-16; last updated 2026-10-30.
Each round records what subagents tried and found.
All numbers come from small simulations and are illustrative rather
than robust.

== Process

Each round dispatches a few subagents in parallel on distinct
directions or open problems, then records their findings here as a new
dated round section and updates _Current status_.
The user steers between and during rounds.

- Subagents read this document and `setup.typ` (the original model,
  with its microfoundations section) and, for data, the paper
  (`paper/gss.tex`, `paper/decompositions.tex`).
- Experiments stay small: under 1 GB of RAM and a few minutes, run with
  `uv run --with numpy python ...` in a scratch directory.
  Subagents do not edit repo files or download data.
- Reports lead with the verdict and the model in equations, give
  numbers, and flag what is derived, checked by simulation, or
  conjectured.
- Commit this document at least once per round, after its write-up.
- Write-ups here are concise and readable top to bottom.
  Superseded findings stay in their round section;
  _Current status_ always reflects the latest view.
- Scratch scripts from rounds 0--2 were not kept;
  every result they produced is recorded below.
  Round 3 scripts are in `/tmp/claude-1000/r3-*/` and may not persist.

== Criteria

Hard criteria.
The model should

+ couple opinions and networks (the form of coupling is open);
+ be economically compelling at the micro level and not too exotic for
  an economics journal;
+ produce interesting equilibria and, less importantly, dynamics;
+ build up through a series of simpler models;
+ match the data (stylized facts below);
+ be tractable enough that agents' behavior is a differential
  equation, even if we cannot solve it;
+ converge as $N -> infinity$ to a well-defined limit in which effects
  neither vanish nor explode
  (the `setup.typ` model's homophily is $O(1 \/ N)$);
+ treat party as descriptive:
  a partition of model output used for decompositions, not an input to
  the dynamics
  (revised 2026-10-30; previously, party structure had to arise from
  the model).

Soft criteria:

+ is parsimonious: emergent properties should come from a few simple,
  understandable micro characteristics, not from added bells and
  whistles;
+ keeps meeting opportunities $M$ flexible:
  $M$ is an exogenous primitive, possibly time-indexed ($M_t$), and
  results should hold for broad classes of $M$
  (geography, work, family, platforms)
  rather than depend on one story such as online mixing,
  since many $M$ can produce the same phenomenon;
+ reduces to Friedkin--Johnsen in a parameter limit, and ideally to a
  standard homophilous network-formation model when opinions are fixed;
+ connects to existing economics literature
  (e.g. Gentzkow--Shapiro on media and selective exposure);
+ explains topic heterogeneity with one parameter:
  the same mechanism gives rising disagreement on most topics and
  concentration on race and abortion;
+ is robust to ordinal measurement
  (predictions survive monotone recoding of survey items);
+ has a counterfactual lever with a policy reading,
  e.g. platform design (selectivity);
+ yields testable comparative statics beyond fitting the facts.

== Stylized facts

From the GSS analysis in the paper (eight topics, 1990s--2024;
covariance computed within topic across its items):

+ The spectral radius rises on most topics since the 1990s
  (not free speech or race, which are flat with spikes).
+ The rise is driven mostly by total variance, not concentration.
  Concentration is flat or falling for most topics
  (falling for speech and police).
+ Exceptions: race, where concentration drives the entire early-2020s
  spike, and abortion, whose total variance rises to about 2010 and
  then falls.
+ Within-party polarization is comparable to or higher than overall
  polarization.
+ Between-party divergence drives race and welfare;
  within-party change drives affirmative action, speech and police.
+ Demographic groups (race, sex, geography, education, religion)
  explain little, except education and religion for abortion.

The model meets facts 4 and 5 through a party readout of its output
(hard criterion 8);
the readout should match how party ID sorts on opinions in the GSS.

The model in `setup.typ` gets the first-order fact backwards:
it produces consensus (total variance falls to about 5%) with
sharpened alignment.
The main target is rising _disagreement_, with concentration as a
topic-specific outcome.

== Current status

Round 0 found that the `setup.typ` model gets the main fact backwards
and needs negative ties to do anything.

Round 1 converged: all four directions independently arrived at
nonnegative ties or attention weighted by a Gaussian (logit) kernel in
opinion distance, $e^(-norm(x_i - x_j)^2 \/ h)$.
Disagreement rises through fragmentation, as $h$ falls or contact
thins, and concentration depends on a topic-level factor.
The directions contribute complementary pieces:

#figure(
  table(
    columns: 2,
    table.header([*Piece*], [*Source*]),
    [one potential, exact gradient play, nonnegative ties], [B],
    [exact mean-field closure and tipping condition $kappa \/ kappa_0 > 8$],
    [D (row-normalized)],

    [meeting opportunities, falling contact, falling concentration], [A],
    [within- vs between-party split], [A, C],
    [topic heterogeneity], [anisotropy or factor structure of $X_0$ (B, D);
      identity loading (A); salience (C)],
  ),
  caption: [What each round-1 direction contributes.],
)

Shared open problems:

+ The kernel width $h$ is in opinion units,
  so results may not survive ordinal recoding,
  and a scale-free $h$ kills the trend mechanism.
+ Attraction only, so the trace never exceeds that of intrinsic
  positions: observed disagreement is intrinsic dispersion
  re-emerging.
+ Why $h$ falls (or contact thins) needs an external story.
+ The coupling to opinions is often quantitatively secondary to
  exogenous drift (A).
  Endogenous identity is the candidate feedback loop.
+ Nothing yet explains the abortion U-shape.

Round 2 produced a synthesis model
(FJ anchor, saturating disagreement cost, ties in the same utility;
one potential; exact FJ limit; mean-field limit)
that matches rising trace, topic-specific concentration and
hysteresis, but has no exact closure near the transition.
Falling contact is the most defensible primary driver;
$h$ can be pinned to a fixed ordered-probit scale;
and groups do not emerge without a categorization step,
while two-sidedness needs an outside force.

Round 3 points to the attention budget (row normalization) as the
base model.
It is the only route to exact analytics:
no saturating potential model closes on Gaussians (derived),
and the potential model's large-$N$ limit is a core of consensus plus
a periphery of detached extremes, with a band of equilibria rather than
a formula.
The budget also removes majority absorption, which makes emergent
groups balanced.
The budget model is still gradient play of a standard utility
(a logit inclusive value), but not a potential game.
The full candidate model is written out at the end of round 3.
Other findings:

- The isotropic concentration spike is a finite-$N$ artifact
  (excess shrinks like $N^(-1\/2)$, no pitchfork).
  A race-style spike needs a factor loading or a shock to the top mode,
  and the share peak comes with a rising trace, not a flat one.
- Two balanced, stable, emergent groups (0.53/0.47 in every seed) come
  from learned categories, a pivotal (Duverger) joining rule, and the
  budget with a same-category tie bonus.
  Groups differ on loaded topics; with moderate loading within-group
  variance on unloaded topics matches overall, as in the GSS.
- The GSS sociability test is feasible and sharp:
  1.3$times$ trace growth needs base-period high- and middle-sociability
  traces summing to at most 1.31 times the low tercile's.
  The data is not cached locally, so nothing is checked yet.

After round 3 the user simplified the candidate
(see _Revisions after round 3_):
the attention budget is adopted;
online mixing is dropped and meeting opportunities $M$ are an
exogenous, flexible primitive;
group machinery is dropped and party is a readout of the model's
output.
The base model is now direction D's, written as a logit inclusive
value, with five primitives
($Sigma_0$, $kappa_0$, $kappa$, $h$, $M$).

Round 4 worked out the budget model's structure.

- *Gaussian branch, then clusters.*
  For any dense $M$ independent of $x_0$, the linear map
  $x = A x_0$ with the closure's $alpha_k$ is an exact equilibrium,
  so $M$ drops out (derived).
  It is what the system reaches at large $h$.
  Below a threshold $h_"cl" (c)$ it loses stability to clustering, for
  every $c$ tested ($1.5$--$10$), and the population settles into one of
  many multi-cluster equilibria with higher trace.
  Multiplicity and hysteresis come from clustering and appear for every
  $c$ tested;
  the closure's $c > 8$ is only the condition for affine multiplicity.
  Rising trace and the loaded-topic concentration hump survive, and are
  robust across three kinds of $M$.
- *Party readout.*
  A noisy ordered logit on the fixed intrinsic ideology direction,
  applied to current opinions, reproduces the fact-5 ordering with one
  loading per topic, but only if sorting (readout precision) rises over
  time.
  Within-party above overall (fact 4) is impossible on the latent scale
  and arises only through ordinal coding.
- *Race.*
  No latent shock gives a spike with flat trace (derived).
  The flat observed trace comes from measurement:
  7 of the 9 race items are binary, so their coded variance is pinned
  while covariances track latent correlations.
  A salience shock to the top mode, amplified near threshold, then
  shows up almost entirely as concentration.
- *Sociability* enters as time spent socializing, $kappa_i = kappa s_i$,
  as one rung above the base model.
  Dispersion in $s$ turns tipping into gradual detachment of the least
  sociable extremes;
  the GSS test needs a log-sd of $s$ of roughly 0.5--1.

Next steps:
+ the GSS checks the data now allows:
  whether race reverted by 2022--24, its latent (probit) trace,
  and the sociability test;
+ whether the paper's regime is above or below $h_"cl"$, and what the
  cluster regime means (many clusters, not two);
+ whether rising sorting is an acceptable exogenous driver of the party
  readout, or needs a source;
+ reconcile closure and particles on race reversion (random populations
  lock in; quadrature nodes revert).

== Round 4 (2026-10-30): structure, party, race, sociability

Four tasks from the post-round-3 next steps.
Base model throughout:
$
  dot(x)_i = kappa_0 (x_(i 0) - x_i)
    + kappa (sum_j hat(w)_(i j) x_j - x_i),
  quad hat(w)_(i j) prop m_(i j) e^(-norm(x_i - x_j)^2 \/ h),
$
with $kappa_0 = 1$ and $c = kappa \/ kappa_0$.
In this kernel's units the closure is
$
  alpha_k (1 + c / (1 + 2 alpha_k^2 sigma_k^2 \/ h)) = 1,
$
with multiple roots iff $c > 8$.
Scripts are in `/tmp/claude-1000/r4-*/`.

=== Large-$N$ structure

Verdict: no core--periphery at $N = infinity$ with identical agents,
but a band of clustered equilibria with wide hysteresis.

*Derived.*

+ *$M$ drops out.*
  For any dense row-stochastic $M$ built independently of $x_0$,
  $x = A x_0$ is an exact equilibrium with the same $alpha_k$ as complete
  $M$:
  every agent faces the same Gaussian population.
+ *No individual bistability on this branch.*
  Given a Gaussian field each agent's best response is linear and
  unique.
  Under the budget even isolated extremes give all their attention to
  their nearest others.
+ *Polynomial perturbations are exact eigenfunctions.*
  With $rho_k = alpha_k^2 sigma_k^2 \/ (alpha_k^2 sigma_k^2 + h \/ 2)$,
  a degree-$n$ perturbation of mode $k$ has eigenvalue
  $
    lambda_n = -1 - c (1 - rho) (1 - (n + 1) rho^n).
  $
  The case $n = 1$ is D's affine tipping.
  Since $sup_n (1 - rho)(n + 1) rho^n -> 1 \/ e$ as $rho -> 1$,
  the Gaussian branch is unstable at small $h$ whenever $c > e$.
+ *Local bulk analysis* (heuristic):
  clustering at small $h$ iff $c > e \/ 2 approx 1.36$
  (conjectured; the sociability task derived the same threshold
  independently).
  The update is an anchored blurring mean shift, which is why it
  clusters.

*Checked* (exact 1-d solver on a weighted 601-node grid;
2-d on 1440 polar nodes; particles at $N = 400$, seeds 1--3, and
$N = 1600$).

#figure(
  table(
    columns: 4,
    table.header([$c$], [$h_"cl"$], [*Closure trace at $h_"cl"$*],
      [*Polynomial criterion*]),
    [1.5], [.014], [.98], [never],
    [2], [.079], [.84], [never],
    [3], [.141], [.56], [sets in at .070],
    [5], [.127], [.24], [exact],
    [8], [.074], [.15], [exact],
    [10], [.053], [.33], [exact],
  ),
  caption: [Clustering threshold of the Gaussian branch (1-d).],
)

- Above $h_"cl"$ the reached state is the Gaussian branch;
  below it, one of many stable multi-cluster states, with cluster
  spacing about $2.6 sqrt(h)$ (conjectured).
  At $c = 3$, $h = 0.149$ the exact trace is 0.83 against the closure's
  0.54.
- Hysteresis is wider than the closure predicts and appears at
  $c = 3$ and 8, not only above 8.
  At $c = 8$, $h = 0.104$ the trace is 0.024 sweeping down and 0.85
  sweeping up;
  at $c = 12$ clusters persist up to $h approx 0.4$--$0.9$.
- Finite $N$ departs from the Gaussian branch at a larger $h$:
  extreme agents seed clusters.
- 2-d loaded topic ($c = 3$, intrinsic share 0.70):
  the share hump survives (peak 0.93 at $h = 0.24$ against the closure's
  0.90); the trace rises monotonically.
- Non-complete $M$ (ring, four blocks, and a location correlated 0.5
  with the loaded axis):
  with $M$ independent of $x_0$ the closure holds above $h_"cl"$ and
  rising trace, hump and hysteresis are all robust.
  Only $M$ correlated with $x_0$ changes things qualitatively:
  the $h -> infinity$ baseline rises (trace 0.20, share 0.90) and the
  hump becomes a plateau.

Proposition-ready claim:
for $x_0 tilde N(0, Sigma)$ and dense $M$ independent of $x_0$,
$x = A x_0$ is an equilibrium for every $M$, unstable whenever
$max_n lambda_n > 0$ (derived);
below $h_"cl" (c)$ the reached state is one of a band of clustered
equilibria with trace at least the closure's (checked).

=== Party readout

Verdict: a one-line readout works, but the between/within split needs
sorting to rise over time.

*Readout.*
Party $in {D, I, R}$ is an ordered logit of $theta' x_i + s epsilon_i$
with cutpoints $plus.minus tau$ (about 35% independents, since the
paper's coding puts leaners with independents).
The direction $theta$ is the top eigenvector of $Sigma_0$, fixed in
time, applied to _current_ opinions $x$ because party ID is endogenous
to opinions.
The decomposition follows `spectral.py`:
$rho_w = sum_g p_g lambda_1 (Sigma_g)$, $rho_b = rho - rho_w$.

*Derived.*

+ *Rank-one downdate.*
  With $x$ Gaussian,
  $Sigma_g = Sigma - r_g (Sigma theta)(Sigma theta)' \/ theta' Sigma theta$,
  so for each topic only its loading on the readout score and the
  sorting $r_g$ matter.
+ *Within never exceeds overall on the latent scale*
  (log-concavity and Brascamp--Lieb give $r_g >= 0$).
+ *The direction is fixed in time*:
  $A$ shares the eigenbasis of $Sigma_0$.
+ *Uniform scaling carries no information*:
  if contact falls and the readout is fixed, the between share of the
  change equals the between share of the level.
+ *Sorting alone* moves within into between one for one.

*Checked* (closure plus Monte Carlo; six topics, 21 items;
loadings on a common ideology factor:
race 0.75, welfare 0.70, spending 0.45, affirmative action 0.30,
police 0.25, speech 0.15).

#figure(
  table(
    columns: 4,
    table.header([*Topic*], [*Contact only*], [*Contact + sorting*],
      [*Selectivity*]),
    [race], [.54], [1.82], [.67],
    [welfare], [.47], [1.57], [.66],
    [spending], [.35], [1.16], [.56],
    [aff. action], [.14], [.45], [.33],
    [police], [.10], [.33], [.26],
    [speech], [.05], [.18], [.13],
  ),
  caption: [
    Between-party share of the change in spectral radius.
    Contact falls from $c = 0.6$ to 0.35 (about 1.3$times$ growth);
    sorting is readout noise $s$ falling from 1.5 to 0.3.
    Shares above 1 mean within-party polarization fell.
  ],
)

- One loading per topic reproduces the fact-5 ordering
  (race and welfare between, the rest within) once sorting rises.
  Endogenous sorting from falling contact is negligible
  ($eta^2$ 0.62 to 0.64).
- Readout noise matters a lot ($eta^2$ is 0.77 at $s = 0.1$ and 0.07 at
  $s = 2$), so $s$ must be pinned from data, e.g. the GSS
  ideology--party correlation by year.
- *Fact 4 under ordinal coding.*
  With binary items cut asymmetrically, the party whose mean sits near
  a cutpoint gets _higher_ variance than the population
  (affirmative action: one party at 1.23 times overall, weighted
  $rho_w \/ rho = 1.01$).
  Which party depends on the direction of skew.
- *Spending switch.*
  Rising sorting alone moves a topic shock from 0.17--0.24 between to
  0.37--0.51 between: not a switch.
  It needs a shock on the partisan component in 2020 (the topic's
  loading, or the common factor).
  A shock to the common ideology factor raises every topic and race's
  radius by 1.6$times$:
  a candidate single 2020 shock behind both the race spike and the
  spending bump (conjectured).
- The direction is well estimated at finite $N$ while the eigengap is
  large (cosine 0.994 at $N = 200$) but rotates under large topic
  shocks.
- Caveat: with 21 items at $h = 2$ the closure fails
  (simulated trace 0.06--0.12 against 0.023):
  the mean-field limit needs $h$ growing with the number of items.

=== Race spike as an amplified shock

Verdict: in latent opinions, no natural shock gives a concentration
spike with flat trace;
with the paper's item coding it does.

*Model.*
In the closure the modes decouple exactly:
$
  dot(alpha)_k = 1 - alpha_k - (c alpha_k) / (1 + z_k alpha_k^2),
  quad z_k = 2 sigma_k^2 \/ h,
$
so a shock confined to one mode moves only that mode.
The bistable window in $z$ is $(31.25, 32)$ at $c = 9$ and
$(59.7, 82.3)$ at $c = 16$.
Shock: the top mode's intrinsic variance rises 15% for three time units.
Readout: latent plus unit probit noise, fixed cutpoints, coded on
$[-1, 1]$ as in the paper.

*No-go* (derived).
For a top-mode-only shock with latent share $p$, the trace's part of
the log spectral-radius change is at least $p$ regardless of
amplification.
A trace-preserving rotation keeps the latent trace flat only if all
modes respond equally, which means no amplification.

*Measurement* (checked).
Seven of the nine race items are binary;
their coded variance is $1 - "mean"^2$ whatever the latent dispersion,
while their covariances track latent correlations.
With the race item formats the share responds 12--40 times more than
the trace (4--7 times if all items were Likert).

#figure(
  table(
    columns: 4,
    table.header([$c$], [$z \/ z^*$], [*Latent trace, share*],
      [*Observed trace, share*]),
    [4], [0.5], [$times 1.16$, .60 $->$ .65], [$times 1.006$, .23 $->$ .26],
    [7], [0.97], [$times 2.39$, .81 $->$ .92], [$times 1.033$, .24 $->$ .35],
    [14], [0.97], [$times 11.2$, .74 $->$ .98],
    [$times 1.084$, .24 $->$ .51, no reversion],
  ),
  caption: [
    Response at the end of the shock (closure).
    $z^*$ is the up-fold for $c > 8$, or the point of maximum elasticity
    otherwise.
  ],
)

*Reversion.*
In the closure, amplification and slow decay are the same quantity:
both are $1 \/ F_alpha$ with
$F_alpha = 1 + c (1 - z alpha^2) \/ (1 + z alpha^2)^2$,
so spikes rise fast and decay slowly
(at $c = 7$, elasticity 11 and relaxation time 3.1 against 0.125 for
plain FJ).
Outside the bistable window reversion is full;
inside it a long enough shock tips the mode permanently.
But random finite populations lock in after the shock in every run
(9-d at $N = 800$ and 1600; 1-d random samples at $N = 3200$),
while 1-d quadrature nodes revert as the closure says.
Likely tail agents seeding clusters (see _Large-$N$ structure_);
whether lock-in survives at $N = infinity$ for random populations is
open.

*Background driver.*
With $c$ falling slowly, the same shock is 4--5 times larger late in the
sample ("why 2020").
The cost is a visible accelerating pre-trend in the observed share
before the shock (0.24 to 0.35 on one path), which conflicts with a
flat series unless it is within noise.

*Rejection.*
The model is rejected by a large spike that also decays fast
(amplification implies slow decay),
or by a latent (probit) race trace that stays flat through 2020--22.
A cheap check: other binary-heavy topics should also show
share-dominated changes.
Whether race reverted by 2022--24 was not checked.

=== Sociability

Verdict: sociability should enter as individual contact volume,
as one rung above the base model ($s equiv 1$).

*Model.*
$
  u_i = -kappa_0 / 2 norm(x_i - x_(i 0))^2
        + tau_i (kappa h) / 2 log sum_j m_(i j) e^(-norm(x_i - x_j)^2 \/ h),
$
where $tau_i$ is time spent socializing (the number of conversations,
each worth the inclusive value).
The budget still holds per conversation, so degree stays irrelevant;
the GSS SOC items measure frequency, which is $tau$.
Write $s_i = tau_i \/ macron(tau)$, lognormal with mean 1 and
independent of $x_0$.
At steady state only $kappa_i \/ kappa_(0 i)$ matters, so heterogeneous
stubbornness folds into $s_i$ (derived).

*Closure.*
The population becomes a scale mixture of Gaussians, so the exact
closure is per type: $x (1 + c s) - c s thin m(x) = x_0$, with the mean
field $m$ computed by quadrature (derived).
A moment closure,
$
  alpha(s) = (1 + u) / (1 + u + c s),
  quad u = 2 V \/ h,
  quad V = sigma^2 EE_s (alpha(s)^2),
$
is within about 3% of particles for log-sd 0.5 and up to 10% off for
log-sd 1 ($N = 1600$, seeds 1--3).

*Calibration* (Gaussian closure; $kappa$ falls so that aggregate trace
grows 1.3$times$).

- Base-period $R_M + R_H <= 1.31$ needs a log-sd of $s$ of at least
  about 0.5 under perfect measurement, or about 1 if the measured index
  correlates 0.6 with true $log s$.
  The model then predicts gaps of 30--80%;
  raw gaps of 5--10% would reject proportional $kappa_i$.
- The low bin is not flat ($g_L$ is 1.06--1.29), so the round-3 bound's
  $g_L approx 1$ is too strict; calibrate with $g \/ g_L$ directly.
- The ordering $R_H < R_M < 1$ and rising $R_H$ hold in every case.
- *A new restriction* (derived, exact in the linear limit):
  the cross-sectional elasticity of trace with respect to contact equals
  the time-series elasticity with respect to mean contact,
  $partial log alpha^2 \/ partial log s = -2 c s \/ (1 + c s)$.
  With finite $h$ the time series is amplified,
  so the ratio of the two measures how much the kernel matters.

*Tipping with heterogeneity* (checked, $N = 800$ and 1600).
There is no macroscopic fold.
The least sociable extreme agents detach one by one,
so the core--periphery picture reappears in the budget model through
heterogeneity.
$R_H$ is non-monotone in this regime, reversing the GSS test's
prediction there,
and hysteresis is total: the fragmented state is absorbing.

*Alternatives.*
Heterogeneous stubbornness is equivalent at steady state.
Putting the anchor inside the budget saturates the anchor pull and has
no closure (rejected).
Leaving sociability out predicts $R equiv 1$, the null of the same
test.
Deferred: $m_(i j) prop s_j$, so that sociable people are also heard
more.

== Revisions after round 3 (2026-10-30)

User decisions, made between rounds:

- *Adopt the attention budget* and give up the potential.
- *No online story.*
  $M$ is exogenous and may later be parameterized and time-indexed,
  but no particular story for it is built in;
  many $M$ can give rise to the same phenomenon.
- *No group machinery.*
  Categories, memberships, the group bonus and the pivotal rule are
  dropped from the base model
  (kept in round 3 as a recorded extension).
  Party is a partition of the model's output, used for the
  within/between decompositions and not fed back into the dynamics.
  Hard criterion 8 and the soft criteria were revised to match.

*Revised candidate.*
Agents choose opinions $x_i$ to maximize
$
  u_i = underbrace(-kappa_0 / 2 norm(x_i - x_(i 0))^2, "anchor")
        + underbrace((kappa h) / 2 log sum_j, "attention budget")
          underbrace(m_(i j), "opportunity")
          underbrace(e^(-norm(x_i - x_j)^2 \/ h), "homophily"),
$
so that
$
  dot(x)_i = kappa_0 (x_(i 0) - x_i)
    + kappa (sum_j hat(w)_(i j) x_j - x_i),
  quad hat(w)_(i j)
    = (m_(i j) e^(-norm(x_i - x_j)^2 \/ h))
      / (sum_k m_(i k) e^(-norm(x_i - x_k)^2 \/ h)).
$
The attention shares $hat(w)_(i j)$ are the network.

#figure(
  table(
    columns: 3,
    table.header([*Parameter*], [*Meaning*], [*Role*]),
    [$Sigma_0$ (per topic)], [covariance of intrinsic positions],
    [topic heterogeneity via factor structure],

    [$kappa_0$], [stubbornness], [time scale; normalize to 1],
    [$kappa$], [susceptibility $times$ contact volume],
    [falling contact raises disagreement],

    [$h$], [selectivity (kernel width)],
    [in ordered-probit noise units; secondary driver],

    [$M$ (or $M_t$)], [meeting opportunities], [exogenous; only its
      pattern matters],
  ),
  caption: [
    Primitives of the revised candidate.
    With complete $M$ and Gaussian $x_0$ the exact closure depends only
    on $c = kappa \/ kappa_0$ and $sigma_k \/ h$, and tips iff $c > 8$.
  ],
)

*Normalizing $M$* (derived).
Each agent's weights are divided by their own row sum,
so scaling any row of $M$ leaves the dynamics unchanged.
The natural convention is row-stochastic $M$
($sum_j m_(i j) = 1$):
each row is a distribution over whom $i$ meets.
This removes $N$ redundant row scales,
builds the $1 \/ N$ scaling into the large-$N$ limit
(a row-normalized graphon),
and makes the $h -> infinity$ limit FJ on the stochastic matrix $M$:
$
  x^* = kappa_0 (kappa_0 I + kappa (I - M))^(-1) x_0.
$

*Consequence for sociability.*
Under the budget the number of contacts has no effect, only whom one
meets.
The GSS sociability test relies on high-contact agents being more
compressed, so sociability must enter as individual contact volume,
e.g. $kappa_i = kappa s_i$.
This would separate how much one socializes ($kappa_i$),
whom one meets ($m_(i dot)$),
and whom one listens to (the kernel).
Open; not yet adopted.

*Party readout.*
Party is assigned by a rule on positions, e.g. a noisy logit on a fixed
axis (imperfect sorting, as in the GSS).
Round 3 found the within/between pattern by topic maps onto factor
loading, so the readout probably reproduces it (conjectured).

== Round 3 (2026-10-16): closure, spike, groups, GSS test

Four tasks from round 2's next steps:
a large-$N$ closure of the synthesis model in the transition regime,
replicating the isotropic concentration spike,
emergent two-sided groups,
and a design for the GSS sociability test.
Units: $kappa_0 = 1$, unit intrinsic variance per axis,
$c = kappa lambda \/ kappa_0$.

=== Large-$N$ closure

Verdict:
the potential model's $N -> infinity$ limit is solvable numerically
but not in closed form, and no model of its kind can be.

*Exact fixed point* (derived).
With $mu$ the push-forward of $N(0, Sigma_0)$ under the stationary map
$F$, each agent's position $F(x_0)$ is a stationary point of
$
  psi(x; x_0) = 1 / 2 norm(x - x_0)^2 + c (U * mu)(x),
  quad U(z) = -h / 4 e^(-2 norm(z)^2 \/ h),
$
(the $a_0$ term adds a second kernel and was not explored).
Placing agents on deterministic quadrature nodes of $N(0, Sigma_0)$
gives a deterministic particle system;
in 1-d, 400 and 800 nodes agree to within 0.01 in trace.
Tensor grids truncate the tails and miss the transition;
polar or radial nodes are needed.

*Core and periphery* (checked).
The stationary state is a consensus core
($norm(x_0) < a$, compressed) plus a periphery of extreme agents who
stay near their intrinsic positions, so $F$ jumps at the core edge.
Each agent is individually bistable, so there is a continuum of
equilibria indexed by $a$:
the down sweep in $h$ keeps the largest core, the up sweep the
smallest.
Hysteresis is real at $N = infinity$ and wide
(at $c = 8$, from $h approx 0.18$ to beyond 3),
but it is built from per-agent folds, not a macroscopic saddle-node;
no macroscopic fold appears for $c in [4, 12]$.
The finite-$N$ jumps of round 2 are single tail agents detaching.

*Gaussian-core closure* (derived; checked).
Treat the core as $N(0, s^2 I)$ with mass $m$ and ignore the
periphery's field.
With $g = h \/ 2$ and $G = g + 2 s^2$, each agent solves
$
  x_0 = x (1 + c m (g \/ G)^(d \/ 2 + 1) e^(-norm(x)^2 \/ G)),
$
with $m$ and $s^2$ set self-consistently.
Agents are bistable iff
$
  c m (g \/ G)^(d \/ 2 + 1) > e^(3 \/ 2) \/ 2 approx 2.24
$
in any dimension, replacing the round-2 condition $c > 5.09$.
The closure matches the down branch to 0.01--0.02 and underestimates the
up branch.

#figure(
  table(
    columns: 5,
    table.header([$h$], [*Exact*], [*Gaussian core*], [*Round-2
      closure*], [*$N = 400$, 3 seeds*]),
    [3.0], [.014 / .057], [.014 / .057], [.013], [up .03--.11],
    [1.17], [.042 / .422], [.042 / .220], [.015], [.02--.10 / .34--.37],
    [0.73], [.160 / .669], [.160 / .559], [.017], [.14--.20 / .38--.73],
    [0.46], [.441 / .833], [.441 / .781], [.023], [],
    [0.23], [.889 / .933], [.882 / .923], [.839], [],
  ),
  caption: [
    Trace relative to intrinsic, down / up sweep, 1-d, $c = 8$.
    Exact is 800 quantile nodes.
  ],
)

In 2-d ($c = 8$, 960 polar nodes) the loaded topic
($Sigma_0 = op("diag")(1.4, 0.6)$, share 0.70) has a genuine
concentration hump at $N = infinity$ (share 0.90 at $h approx 1.08$),
while the isotropic topic's share stays at 0.500.

*No-go for exact closure with a potential* (derived).
With a translation-invariant pair potential the force on an agent is
$-(f * N)(x)$ with $f(z) = w(z) z$;
linearity in $x$ forces $w$ constant, which is linear FJ.
Normalizing by a population mean of $K$ keeps the potential but only
rescales $c$;
symmetrized row normalization keeps a potential but adds a non-Gaussian
factor.
Direction D's attention budget, which does close exactly, is gradient
play of
$
  u_i = -kappa_0 / 2 norm(x_i - theta_i)^2
        + kappa h' log sum_j m_(i j) e^(-norm(x_i - x_j)^2 \/ h'),
$
a logit inclusive value:
a standard utility, but not a potential game.
Unchecked: whether D's model also has core--periphery equilibria
off the Gaussian branch.

=== Symmetry-breaking spike

Verdict: a finite-$N$ artifact, not symmetry breaking (checked).
Same synthesis model on a complete meeting graph, $d = 4$,
quasi-static sweeps of $h$ from 8 to 0.5 and back,
20 seeds at $N = 200, 400$ and 3 seeds at $N = 800, 1600$.
Configurations: no endogenous ties ($lambda = 0$),
below threshold ($c = 3$) and above ($c = 8$).

- *It replicates at small $N$.*
  At $N = 400$ every seed spikes;
  the excess of peak share over the sample share of $x_0$ (0.285) is
  0.10 below threshold and 0.29 [0.15, 0.40] above it.
- *It shrinks like $N^(-1\/2)$.*
  Above threshold the excess is 0.40, 0.29, 0.21, 0.16 at
  $N = 200, 400, 800, 1600$.
- *It is driven by escapees.*
  At the peak one to five extreme agents have left the compressed
  core and carry 55--78% of the top-axis variance
  (the core's trace is 0.03--0.09 of intrinsic).
  This is the closure's core--periphery picture at finite $N$.
- *Exactly isotropic $x_0$* (orbits of the signed cyclic group,
  covariance exactly $c I$, with $10^(-6)$ kicks) gives share 0.250 in
  all 36 runs.
- *No pitchfork.*
  On the symmetric branch the isotropic Hessian mode always softens
  first and the anisotropic modes stay stiffer
  (checked at $N = 192, 384$; the $N -> infinity$ claim is
  conjectured from scaling and $O(d)$ equivariance).
- *Loaded topic* (share 0.45):
  excess 0.35, 0.31, 0.32, 0.27 at $N = 200$ to 1600, so the hump is a
  population effect.
  At $N = 200$ the unloaded spike is as large as the loaded one,
  which is why one seed looked convincing in round 2.

For race:
spontaneous symmetry breaking cannot replace a factor loading or a
shock to the top mode.
Nor does the spike come with a flat trace:
at the share peak the trace is already 2--6 times its pre-transition
level.
A concentration spike with flat trace needs a shock confined to the top
mode or a measurement effect;
falling $h$ cannot produce it.

=== Emergent two-sided groups

Verdict: learned categories, a pivotal joining rule and the attention
budget with a same-category tie bonus give exactly two balanced groups
in every seed, with no party label (checked;
$N = 400$, 9 items in three topics, 8 available categories, random
initial memberships, 10 seeds).

*Model.*
Opinions under the budget,
$
  dot(x)_i = kappa_0 (x_(i 0) - x_i)
    + kappa ((sum_j W_(i j) x_j) / (sum_j W_(i j)) - x_i),
  quad W_(i j) = m_(i j) K_(i j) (1 + gamma r_i dot r_j),
$
where $r_i$ is $i$'s soft membership over categories
(the synthesis model's two-kernel weight slots into $W$).
A category's platform is its members' mean $mu_k$ and its share $s_k$;
it wins with probability $p = op("softmax")(tau s)$.
With $v_(i k) = -1 \/ 2 norm(x_i - mu_k)^2$ and
$macron(v)_i = sum_l p_l v_(i l)$, memberships adjust as
$
  dot(r)_(i k) = epsilon (op("softmax")_k (beta K p_k (v_(i k) - macron(v)_i))
                 - r_(i k)).
$
The payoff is Myerson--Weber vote value with pivot probabilities
$P_(k l) prop p_k p_l$ (derived; the reduced form for Poisson-game
pivot probabilities is conjectured).
With two live parties it equals sincere voting, so there is no
bandwagon;
balance comes from 2-means geometry (the boundary sits at the midpoint
of the two means), not from median-voter logic (derived).
Neither the budget nor the membership game is a potential game,
so convergence is checked only by simulation.

*Attention budget.*
Round 2's majority absorption came mostly from the mixture prior
$log pi_k$ in EM categorization (bigger categories judged more
likely), amplified by the tie bonus when total contact is not
budgeted:
with $K = 2$ and the prior, the top group's share is 0.82 without the
budget and 0.57 with it;
without the prior it is 0.51--0.54 either way.
Without the budget a minority member's total pull is smaller,
$prop 1 + gamma pi$ (argued).
The budget does not stop the cascade to 8 groups as precision rises:
that cascade is deterministic annealing on the categorization side.

*Duverger force.*

#figure(
  table(
    columns: 3,
    table.header([*Case*], [*Groups*], [*Top share*]),
    [pivotal + budget + bonus, 5 of 6 cells], [2 (all seeds)],
    [0.53 (0.50--0.57)],

    [same, 4000 steps or $N = 800$], [2], [0.52--0.53],
    [same, near-deterministic elections ($tau = 40$)], [sometimes 3],
    [third up to 0.28],

    [no tie bonus], [2 plus centrist remnant 0.01--0.09], [],
    [no budget, fixed total contact], [2.3 on average], [third up to 0.21],
    [expressive + pivotal, skewed factor], [5--8], [ENP 3.2--3.5],
    [pure pivotal, skewed factor], [2], [0.60 (0.53--0.67)],
  ),
  caption: [Emergent groups across 10 seeds ($tau = 10$--20 unless noted).],
)

*Topics.*
With a strong loading on topic 1 the between-group share is 0.57 on
topic 1 and 0.11 on the others,
but the within-group $lambda_1$ on topic 1 is only 0.28 of overall,
too extreme for the GSS
(mostly mechanical: splitting a normal axis removes $2 \/ pi$ of its
variance).
A weaker loading gives a between-share of 0.38 and within/overall 0.42
on the loaded topic, and 0.01 and 1.03 on the unloaded one,
matching the GSS fact that within-party disagreement is comparable to
overall.
So the "between for race and welfare, within for the rest" split maps
onto factor loading.

*Micro story.*
People sort into camps whose platform is their members' average.
A camp is worth joining only insofar as it can win, so support drains
from camps that cannot, and two survive (Duverger; Cox 1997).
Attention is a fixed budget, tilted toward fellow members,
so camps reinforce themselves without a larger camp swallowing a
smaller one.
An entry cost (Palfrey; citizen-candidate) would set a number of
groups that scales with the cost rather than exactly two (untested).
Literature to verify:
Myerson--Weber (1993), Palfrey (1989), Osborne--Slivinski (1996),
Besley--Coate (1997), Fryer--Jackson (2008), Shayo (2009).

=== GSS sociability test

Verdict: feasible and likely decisive, but unchecked:
the GSS data is not built locally.
Item years and coverage below are from memory and need verifying.

*Measures and bins.*
Primary: the within-year weighted rank of the mean of reversed
`SOCOMMUN` (neighbors, closest to cross-cutting geographic contact
$m_(i j)$) and `SOCFREND`, cut into within-year terciles
(ties broken at random).
`SOCREL` separately (non-elective, so less reverse causality, but
relatives share $x_0$);
`SOCBAR` as robustness only.
`NUMGIVEN` (1985, 2004) is unusable as a series.

*Statistic.*
For topic $k$, bin $b$ and five-year period $p$, let
$op("tr")_(b p k)$ be the within-cell trace (paper coding, `WTSSPS`,
pairwise-complete) and
$R_(b p k) = op("tr")_(b p k) \/ op("tr")_(L p k)$.
Mean-field FJ gives $x_i approx s_i x_(i 0)$ with
$s_i = 1 \/ (1 + c d_i)$, which predicts (derived, given $x_0$
independent of sociability):

+ ordering $R_H <= R_M <= 1$ in every period, never above 1;
+ $R_H$ rises over time;
+ the low bin rises less than the high bin (not exactly flat:
  the low tercile is itself compressed; synthetic data gives +14%);
+ in the linear limit $R_H$ follows a common path across topics.

*Variance regression.*
At the respondent-item level, with
$y_(i j) = (x_(i j) - hat(mu)_(j p))^2 \/ hat(sigma)_(j p)^2$
and $r_i$ the sociability rank,
$
  y_(i j) = alpha_(j p) + beta_p (r_i - 1 \/ 2) + Z_i' gamma_p
            + epsilon_(i j),
$
weighted, clustered by respondent (or design bootstrap).
Predictions: $beta_p < 0$ and rising toward 0 over time;
PPML gives $beta_p approx log(s_H^2 \/ s_L^2)$.

*Calibration* (derived; an accounting identity).
With equal bin means and terciles, the pooled trace is the average of
bin traces, so growth $g$ relative to low-bin growth $g_L$ satisfies
$
  g / g_L = macron(R)_"end" / macron(R)_0 <= 1 / macron(R)_0,
  quad macron(R) = (1 + R_M + R_H) / 3.
$
With $g_L approx 1$, growth of 1.2, 1.3, 1.5 needs base-period
$R_M + R_H <= 1.5, 1.31, 1.0$:
a large gap, easy to detect or reject.
The agent's prior is that raw gaps are 5--10%, which would rule out
falling contact as the sole driver (conjectured).

*Threats.*
Sociability correlated with intrinsic positions
(DFL-reweight bins to base-period covariates, or residualize items on
demographics);
composition drift (within-year ranks);
reverse causality, since extremists withdrawing gives the same sign
(identify from the time interaction, `SOCREL`, a placebo on the
demographically predicted component, and the 2006--14 panels if they
carry the SOC items);
ceilings (use the trace, not the share);
COVID-era mode change (show with and without 2020--21).

*Falsification.*
Falling contact is rejected if the base gap is absent or too small for
the observed growth, if $beta_p$ is flat while the trace rises,
if the low bin rises as fast as the high bin, if $R > 1$,
or if mean contact does not fall.
Contact and rising selectivity both close the gap;
they differ in that contact closes it only on measures whose levels
fell, along a common path across topics, and within parties,
whereas selectivity closes it first on high-variance topics and
through between-cluster variance (conjectured).

*To compute once data is built*
(script at `/tmp/claude-1000/r3-gss/sociability_test.py`):
item availability by year (critical: whether SOC items were asked in
2021--24);
$g_k$ from 1990--94 to 2020--24 and to 2015--19;
base-period $R_M$, $R_H$ and $g_(L k)$ per topic.

=== Candidate model after round 3

A snapshot assembled from the subsections above
(recorded 2026-10-30; not yet simulated as one model).
Agents choose opinions $x_i$ to maximize
$
  u_i = underbrace(-kappa_0 / 2 norm(x_i - x_(i 0))^2, "anchor")
        + underbrace((kappa h) / 2 log sum_j, "attention budget")
          (underbrace(m_(i j), "opportunity")
           underbrace(e^(-norm(x_i - x_j)^2 \/ h), "homophily")
           underbrace((1 + gamma r_i dot r_j), "group bonus")).
$

- *Anchor.*
  Pull toward the intrinsic position $x_(i 0)$ with stubbornness
  $kappa_0$.
  Topic heterogeneity comes from the factor structure of $x_(i 0)$.
- *Attention budget.*
  The log of a sum is a logit inclusive value:
  each agent spreads unit attention over contacts in proportion to
  their weights, with susceptibility $kappa$.
  It keeps effects $O(1)$ as $N -> infinity$, stops larger groups from
  pulling harder, and gives the exact closure.
- *Opportunity.*
  Who can be met at all: geography plus an online share $omega$.
- *Homophily.*
  Attention falls with opinion distance at selectivity $1 \/ h$,
  read as the probability that a contact's view is relevant.
  This couples opinions to the network.
- *Group bonus.*
  Extra attention to members of one's own category, with soft
  memberships $r_i$ and strength $gamma$.

Gradient play gives
$
  dot(x)_i = kappa_0 (x_(i 0) - x_i)
    + kappa ((sum_j W_(i j) x_j) / (sum_j W_(i j)) - x_i),
$
where $W_(i j)$ is the product of the three weights;
the attention shares $W_(i j) \/ sum_k W_(i k)$ are the network.

Groups are learned, not given.
A category's platform $mu_k$ is its members' mean and its share is
$s_k$.
Memberships follow a logit best response (precision $beta$) to the
payoff
$
  underbrace(p_k, "win prob.") thick
  underbrace((v_(i k) - macron(v)_i), "gain").
$

- *Win probability.*
  A noisy plurality election, $p = op("softmax")(tau s)$:
  categories that cannot win are worth little, so support drains from
  them and two survive (Duverger).
- *Gain.*
  How much better platform $k$ is for $i$ than the expected winner,
  with $v_(i k) = -1 \/ 2 norm(x_i - mu_k)^2$ and
  $macron(v)_i = sum_l p_l v_(i l)$.
  With two live groups this reduces to sincere voting, and balance
  comes from the boundary sitting midway between the two platforms.

*Ladder.*
$h -> infinity$ with $gamma = 0$ is FJ (DeMarzo--Vayanos--Zwiebel) on
row-normalized $m$;
fixed opinions give logit latent-space homophily;
$gamma = 0$ with Gaussian $x_0$ has direction D's exact mode-by-mode
closure, with tipping and hysteresis.
Drivers are contact ($kappa$, $m$), selectivity ($h$), online mixing
($omega$), and shocks to $x_0$.
Costs: no potential function (convergence checked only by
simulation), and three pieces for groups.

*Swaps and alternatives* considered at this point:

- keep the potential model (Welsch cost, ties in the same utility):
  gains a Lyapunov function, loses closed forms and group balance;
- an attention floor $m_(i j) (a_0 + K_(i j))$:
  cross-cutting contact that ignores opinions;
- slow ties that relax toward the logit target, for network dynamics;
- an entry cost instead of the pivotal rule
  (number of groups scales with the cost);
  no tie bonus (leaves a centrist remnant);
  or two imposed parties as a benchmark;
- topic heterogeneity from salience or identity loading instead of the
  factor structure of $x_0$;
- noise, for a stationary distribution and early-warning variance;
- drifting $x_0$ or cohort replacement to escape the trace bound
  (exogenous).

== Round 2 (2026-10-16): synthesis and open problems

Four tasks:
a synthesis model,
emergent identity (no baked-in parties),
measurement and scale,
and drivers of the time trend.

=== Synthesis model

Three essential pieces:
a Friedkin--Johnsen anchor,
a saturating disagreement cost,
and tie choice in the same utility.
Agent $i$'s utility is
$
  -kappa_0 / 2 norm(x_i - x_(i 0))^2
  - 1 / N sum_j (kappa A_(i j) phi(norm(x_i - x_j)^2)
                 - beta A_(i j) + c / 2 A_(i j)^2 \/ m_(i j)),
$
with the Welsch cost $phi(r) = h \/ 2 (1 - e^(-r \/ h))$.
Setting the tie benefit $beta = kappa h \/ 2 + c a_0$ gives
nonnegative optimal ties in closed form,
$
  A_(i j)^* = m_(i j) (a_0 + lambda K_(i j)),
  quad K_(i j) = e^(-norm(x_i - x_j)^2 \/ h),
  quad lambda = kappa h \/ (2 c),
$
and the opinion ODE
$
  dot(x)_i = kappa_0 (x_(i 0) - x_i)
    + kappa / N sum_j A_(i j) K_(i j) (x_j - x_i).
$
One potential $Phi(X, A)$, bounded below, covers both, so gradient
play converges.
With ties at their optimum the influence weight is
$m_(i j) (a_0 K_h + lambda K_(h \/ 2))$, a sum of two Gaussian kernels.

Optional one-parameter add-ons:
endogenous ties ($lambda$; not needed for tipping or concentration,
but they make the network respond to opinions),
geography and online share in $m_(i j)$,
and row normalization (exact closure, but loses the potential).
Party is dropped.
The synthesis agent expected clusters to arise endogenously,
but the emergent-identity runs below found none without a
categorization step.

*Ladder.*
$h -> infinity$ is exactly FJ on the network $m (a_0 + lambda)$
(checked: trace 0.135 vs 0.132 predicted).
Fixing opinions at $x_0$ gives the Hoff--Raftery--Handcock latent-space
model with Currarini--Jackson--Pin meeting opportunities.

*Large $N$.*
Ties scaled by $1 \/ N$ give $O(1)$ degrees and a McKean--Vlasov
(graphon mean-field) limit, by standard propagation-of-chaos results
(conditions not checked).
$N = 400$ and $1600$ agree to within 0.05 except at the transition.

*No exact closure.*
In the symmetric version extreme agents have fewer effective
neighbors, so the fixed point is nonlinear and non-Gaussian.
Replacing the neighbor factor by its mean gives a tipping condition
$kappa lambda \/ kappa_0 > 5.09$ in 1-d and is accurate away from the
transition, but fails badly inside it.
The exact closure of D requires the attention budget.

*Facts* ($N = 400$, checked; hysteresis runs one seed):

- Unloaded topic: trace $0.13 -> 0.30 -> 3.38 -> 3.95$ as $h$ falls
  from 50 to 0.5, share flat at 0.27.
- Loaded topic: share peaks at 0.64 (from 0.455) at $h = 5$ and
  returns.
- Above the tipping threshold there is hysteresis:
  at $h = 3$ the loaded trace is 0.32 sweeping down and 2.44 sweeping
  up.
- The _unloaded_ topic shows a transient share spike to 0.56 from
  spontaneous symmetry breaking: one axis tips first even with
  isotropic priors.
  A race-style spike may need no factor loading (one seed).

Open: no closure in the transition regime,
the symmetry breaking needs more seeds,
$h$ is in opinion units,
and the trace stays below that of $x_0$.

=== Emergent identity

Stopped early (the session moved); one seed, 9 dimensions
(three topics of three items), Gaussian intrinsic positions with a
cross-topic ideology factor.
Verdict: a two-way split along the dominant axis is the generic first
symmetry break, but no model produced stable, balanced two-sidedness,
and the plainest model produced no groups at all.

- *Imposed two parties* (superseded by the criterion, but informative).
  Logit party choice with precision $beta$ on salience-weighted
  distance to party means, plus a co-partisan tie bonus $e^gamma$.
  Parties separate when $2 beta a^2 lambda_max (Sigma_0 W) + b > 1$,
  where $a$ is the shrinkage of intrinsic positions and $b$ the
  feedback from party ties (derived; onset 1.05--1.2 times the
  predicted threshold).
  The transition is mostly continuous: real feedback, little tipping.
- *Learned categories* (categorical thinking, Fryer--Jackson,
  Mullainathan).
  Agents fit up to 8 soft prototypes to the population by EM with
  precision $beta$; same-category ties get a bonus.
  This is deterministic annealing:
  a unimodal population's first split is always into two along the
  top eigenvector of the salience-weighted covariance (derived).
  - With no tie bonus, two large groups (0.47, 0.39) appear, but
    further increases in $beta$ give 6--8 groups.
    Groups differ on the high-salience topic (between-group share
    0.62) and not on the others (0.02):
    between-group divergence only on identity-loaded topics.
  - With a tie bonus the majority absorbs the minority (0.94/0.06):
    with fixed contact intensity a larger group pulls harder.
    A balanced split appears only at a large bonus and $beta$.
- *Bounded cost on a complete network, no categories* ($N = 800$):
  no groups at any $h$; both principal axes stay unimodal.
  The Hegselmann--Krause clustering of direction B does not appear with
  Gaussian anchors in 9 dimensions.

Conclusions:
groups need a categorization step;
two-sidedness needs an outside force, and the minimal candidate is one
that rewards exactly two coalitions, such as winner-take-all elections
(Duverger; conjectured, untested).
Within-group variance matches overall on unloaded topics but is far
below it on loaded topics, more extreme than in the GSS.
Open: an attention budget (may remove majority absorption),
a Duverger force, finite-$N$ checks, multiple seeds.

=== Measurement and scale

*Units of $h$.*
A monotone recoding changes the measurement map, not the model,
so the problem is only pinning the latent scale.
Normalizing $h$ by current dispersion kills the trend;
normalizing by any time-invariant reference does not.
The natural reference is a pooled ordered-probit measurement model with
time-invariant cutpoints and a year-specific latent covariance
$Sigma_t$:
$h$ is then in units of the fixed probit noise
(argued, not fitted).
Unlike per-year polychorics this keeps the trace trend,
and it gives the paper a measurement-invariant latent $Sigma_t$.

*Ordinal robustness* (model B, $N = 300$, $h$ falling, latent
opinions through fixed cutpoints and several item formats; checked,
one seed):

- The direction of the trace rise survives every scheme except binary
  items cut at the median (already at maximal variance).
  Magnitudes are compressed: latent $times 58$ became $times 2$--8
  observed.
- The share is _not_ robust to mixed item formats within a topic.
  Items hit their ceilings at different rates, so isotropic latent
  opinions show observed share rising $0.27 -> 0.53$.
  `unit_variance_at_t0` does not fix this.
- Measurement noise creates spurious concentration on loaded topics:
  as latent dispersion grows, attenuation weakens and observed
  correlations rise ($0.43 -> 0.49$ observed with a flat latent share).
  So a flat observed share with rising trace implies a slightly
  _falling_ latent share.

Both points bear on the paper's concentration decomposition directly:
topics mixing binary and Likert items (possibly abortion) need a
latent-scale check.

*The trace bound.*
With attraction only, observed growth $g$ requires base-year
compression of at least $1 - 1 \/ g$.
A growth of 1.3--1.5 needs roughly $kappa macron(d) \/ kappa_0 approx 0.3$
(conjectured; GSS growth not checked against this).
The story is defensible as declining social cohesion, and it predicts
that growth decelerates as the covariance approaches that of intrinsic
positions.
Noise raises the ceiling but does not remove it;
drifting $X_0$ or cohort replacement escapes it but is exogenous.

*A testable signature* (checked, one seed):
as influence weakens, isolated agents stay near intrinsic dispersion
while connected agents converge to it.
Low-degree tercile trace goes $2.75 -> 3.79$,
high-degree tercile $0.47 -> 3.92$ (intrinsic 4.06).
With GSS sociability items (e.g. `SOCFREND`, `SOCOMMUN`) the model
predicts flat variance among low-sociability respondents, rising
variance among high-sociability respondents, and a closing gap.
This needs intrinsic positions independent of sociability.

=== Drivers of the time trend

Computed with D's exact mean-field closure ($c = kappa \/ kappa_0$,
kernel width $h$) plus finite-$N$ runs for online mixing.
Two derived constraints:

+ On the stable branch each mode's surviving fraction rises with
  $sigma_k \/ h$ and falls with $c$,
  so the trace is monotone in both.
  A U-shape on one topic cannot come from a common monotone driver.
+ Contact volume and susceptibility enter only through $c$
  (or $kappa lambda$), so they are never separable in cross-sections.

#figure(
  table(
    columns: 4,
    table.header([*Driver*], [*Trace*], [*Share*], [*Evidence*]),
    [contact falls ($c arrow.b$)],
    [broad, simultaneous rise on all topics],
    [flat or mild hump; falls on identity-loaded topics (A)],
    [time use; discussion networks; GSS social-contact items],

    [selectivity rises ($h arrow.b$)],
    [rises first and most on high-variance topics],
    [hump on loaded topics ($0.50 -> 0.83 -> 0.51$); flat otherwise],
    [partisan cable timing (Martin--Yurukoglu);
      Boxell--Gentzkow--Shapiro age pattern],

    [online mixing rises ($omega arrow.t$)],
    [_falls_],
    [falls],
    [between-place variance should fall (needs geography)],

    [topic shock to $X_0$], [spike on one topic], [spike if on top mode],
    [event timing],

    [composition (cohort, education, religion)], [any shape], [any],
    [DFL reweighting],
  ),
  caption: [Signatures of candidate drivers (checked in one
    parameterization).],
)

- *Spikes as amplified shocks.*
  A 10% shock to the factor variance of a loaded topic has elasticity
  1.5--2 normally but 35 near the tipping point,
  where the topic jumps from consensus to disagreement:
  a candidate for the race spike.
  Hysteresis predicts only partial reversion afterward;
  full reversion in 2022--24 would reject this.
- *Abortion U.*
  Must be a topic-specific change in intrinsic positions,
  e.g. cohort replacement or education and religion composition,
  consistent with the paper's decomposition.
  Test: DFL reweighting should remove most of the U.
- The switch of spending and speech spikes between within- and
  between-party needs endogenous party structure.

Recommendation:
falling cross-cutting contact as the primary driver
(broad trace rise with roughly flat share, simple micro story,
testable within the GSS),
rising selectivity from partisan cable rather than the internet as a
secondary driver for loaded topics,
online mixing as a counter-force,
and spikes and the abortion U as topic shocks or composition amplified
near threshold.
Literature magnitudes are from memory and need verifying.

== Round 1 (2026-10-16): four candidate models

Four directions, each built up from Friedkin--Johnsen:
(A) network formation first,
(B) bounded disagreement cost,
(C) endogenous salience,
(D) social learning with selective exposure.

=== A: network formation first

Friedkin--Johnsen opinions on a logit tie network that relaxes toward
a target:
$
  dot(x)_i & = kappa_0 (x_(i 0) - x_i)
             + kappa sum_j A_(i j) (x_j - x_i), \
  dot(A)_(i j) & = rho (hat(A)_(i j) (X) - A_(i j)), \
  hat(A)_(i j) & = lambda / macron(d) thin m_(i j)
                   exp(-gamma_z bold(1)(z_i != z_j)
                       - gamma_x norm(x_i - x_j)^2 \/ h^2).
$
Here $m_(i j) = (1 - omega) m^"geo"_(i j) + omega$ is the opportunity
to meet (geography plus a share $omega$ of online mixing),
$z_i$ is an exogenous identity such as party,
and $lambda$ is total contact.
Opinions are the Bindel--Kleinberg--Oren game;
ties are logit dyadic formation with Currarini--Jackson--Pin meeting
opportunities.
An equilibrium exists by Brouwer (derived).

The ladder:

+ *FJ on a fixed network.*
  $Sigma^* = kappa_0^2 \/ N thin X_0^top R^2 X_0$.
+ *Ties unrelated to opinions.*
  If rows of $X_0$ are iid and independent of $A$,
  $
    EE(Sigma^*) = q(L) Sigma_0,
    quad q = 1 / N sum_k (kappa_0 / (kappa_0 + kappa ell_k))^2
  $
  (derived).
  The share is unchanged and the trace scales by $q$, which falls with
  cohesion.
  Clustered networks preserve more disagreement than random mixing.
+ *Homophily on identity.*
  With $X_0 = z b^top + E$,
  $Sigma^* approx q_z sigma_z^2 b b^top + q Sigma_E$ with $q_z > q$.
  The share rises with identity loading $beta = norm(b)$ times
  $gamma_z$:
  one parameter for topic heterogeneity.
+ *Homophily on opinions.*
  Bounded; never explodes.
+ *Endogenous identity* (not done):
  the natural source of real feedback and multiple equilibria.

Results ($N = 400$, one seed, checked):

- Topic not loaded on identity: across every scenario the share stays
  at 0.28 while the trace ranges from 0.48 to 2.09.
- Falling contact ($lambda: 8 -> 1.5$):
  trace $0.48 -> 2.09$ with flat share;
  for the loaded topic, trace $0.89 -> 2.94$ and share
  _falls_ $0.57 -> 0.43$ (like speech and police).
- Party homophily ($gamma_z: 0 -> 3$) raises the share only for loaded
  topics: $0.29 -> 0.28$ at $beta = 0$, $0.50 -> 0.74$ at
  $beta = 1.5$.
- Opinion homophily ($gamma_x: 0 -> 4$): trace $1.02 -> 1.69$,
  share $0.39 -> 0.42$.
- Combined drift (contact falls, $gamma_z$ rises): unloaded topic trace
  $times 4$ with flat share; loaded topic trace $times 4.6$ with share
  $0.39 -> 0.49$.
- Online mixing ($omega: 0 -> 0.9$) _lowers_ the trace,
  against the popular story.

Assessment: the strongest micro foundation so far, a clean ladder, and
a good fit to the main facts via one parameter (identity loading).
But the fit comes mostly from exogenous drift in contact and party
homophily, and the coupling to opinions is quantitatively secondary;
a referee will ask why coupling is needed.
Equilibria are unique and smooth, so there is no tipping.
Endogenous identity is where coupling would earn its keep.
Ordinal robustness is not addressed.

=== B: bounded disagreement cost

Replace the quadratic disagreement cost with the Welsch cost
$phi(r) = h \/ 2 (1 - e^(-r \/ h))$, which is $approx r \/ 2$ for
$r << h$ and saturates at $h \/ 2$.

+ *FJ* is the limit $h -> infinity$.
+ *Bounded cost on a fixed network $A_0 >= 0$.*
  $
    u_i = -kappa_0 / 2 norm(x_i - x_(i 0))^2
          - kappa sum_j A_(i j) phi(norm(x_i - x_j)^2)
  $
  gives the gradient-play ODE
  $
    dot(x)_i = kappa_0 (x_(i 0) - x_i)
      + kappa sum_j A_(i j) K_(i j) (x_j - x_i),
    quad K_(i j) = e^(-norm(x_i - x_j)^2 \/ h).
  $
  This is FJ on the effective network $W = A compose K(X)$, so it already
  couples opinions and network, and the fixed-point formulas of
  `setup.typ` hold given $W^*$.
  It is an exact potential game with a potential bounded below,
  so it converges, with many equilibria.
  Micro reading: robust Bayesian updating (a neighbor's signal may be
  uninformative), a soft Hegselmann--Krause.
+ *Endogenous ties in the same utility.*
  Adding a tie benefit $b$ and quadratic tie costs toward $A_0$ gives
  the tie target $((kappa h \/ 2) K_(i j) + rho_0 A_(0, i j)) \/ (rho_0 + rho)$
  with $b = kappa h \/ 2$, which is never negative.
  One potential $Phi(X, A)$ covers both, so unconstrained gradient play
  is exact:
  no budget, no $1 \/ T$ normalization, no integrability problem.

Results ($N = 150$, $d = 4$, one seed, checked;
the tie benefit was simplified to a fixed scale):

- Trace rises monotonically as $h$ falls:
  trace relative to $X_0$ is 0.09, 0.40, 0.74, 1.00 at
  $h = 100, 3, 2, 0.5$.
  Attraction only, so the trace never exceeds that of $X_0$:
  the rise is dispersed intrinsic positions re-emerging.
- Concentration rises part-way through the drift, and only when $X_0$
  has a dominant axis: the share peaks at 0.59 (from 0.46) around
  $h approx 3$--5 and returns to 0.46; flat for isotropic $X_0$.
  Pairs separated along the long axis stop interacting first.
  Endogenous ties amplify the peak to 0.75.
- Clusters scale like $("spread" \/ sqrt(h))^d$, as in
  Hegselmann--Krause.
- Hysteresis with endogenous ties: drifting $h$ down and back up gives
  different paths over $h in [3, 9]$, a possible abortion U-shape
  (weak evidence).
- No negative ties in any run.

Assessment: the cleanest micro structure (one potential, a Lyapunov
function, exact gradient play, nonnegative ties) and interesting
equilibria.
Topic heterogeneity comes from the anisotropy of $X_0$.
Weak on ordinal robustness ($h$ in item units) and silent on party.
Why $h$ falls needs an external story.

=== C: endogenous salience

A ladder of four levels, each nesting the previous one:
$
  "FJ:" quad & dot(x)_i = kappa_0 (x_(i 0) - x_i)
                 + kappa sum_j A_(i j) (x_j - x_i), \
  "logit ties:" quad & A_(i j) = A_(0, i j)
                 exp(-lambda norm(x_i - x_j)_W^2 \/ s),
                 quad s = 2 op("tr") Sigma, \
  "salience:" quad & W = d thin Sigma^p \/ op("tr") Sigma^p, \
  "identity:" quad & "party" z_i = plus.minus 1 "adds" gamma bold(1)(z_i != z_j)
                 "to the tie exponent and" beta z_i u "to" x_(i 0).
$
Ties are nonnegative and adjust fast.
Logit ties are standard (random-utility dyadic formation, rational
inattention).
The salience exponent $p$ says how strongly attention flows to the most
contested combinations of items.

Results ($N = 200$, $d = 4$, checked by simulation):

- Homophily raises total variance through fragmentation:
  trace $0.04, 1.08, 1.88, 2.92$ at $lambda = 0, 2, 4, 8$.
- Salience separates topic types.
  At $lambda = 4$ the share is 0.32 for $p = 0$ (flat),
  0.69 for $p = 2$, and 0.75 for $p = 4$ (collapse),
  from 0.31 initially.
  Across seeds the sign is robust but the size is noisy
  ($p = 2$ gives 0.44--0.59).
- Party: without party homophily ($gamma = 0$) within-party
  $lambda_1$ is 0.7--1.2 times the overall value, as in the GSS;
  with $gamma = beta = 1$ it falls to 0.34--0.6
  (between-party dominates, as for race and welfare).
- A Gaussian mean-field closure reduces the fixed point to $d$ scalar
  equations, one per eigen-axis of $Sigma$.
  Salience makes the cross-pull on axis $k$ fall with $sigma_k$,
  so disagreement feeds on itself and one axis can win.
  The closure is conjectured; it is within 0.05 of simulation in some
  regimes but overstates concentration in others.
- Drifting $lambda$ up slowly gives a lower trace than starting from
  $X_0$ (1.10 vs 1.86), suggesting bistability between cohesive and
  fragmented states (conjectured).

Assessment: matches rising trace, topic-specific concentration and the
within/between split, with no negative ties.
The weak point is the salience rule $W prop Sigma^p$, which needs a
proper rational-inattention derivation or replacement by a topic-level
salience parameter.
It does not explain the abortion U-shape or the race spike, and
salience depends on item coding, which is a risk for ordinal
robustness.

=== D: social learning with selective exposure

Agent $i$ has a persistent prior $theta_i$ (the role of $X_0$) and a
belief $x_i$.
Each period she reads a unit of content from her exposure set $m_(i j)$
and weights each source by the posterior probability that it is
relevant (shares her type rather than being noise):
$
  dot(x)_i = kappa_0 (theta_i - x_i)
    + kappa (sum_j hat(w)_(i j) (x) x_j - x_i),
  quad
  hat(w)_(i j) = (m_(i j) e^(-norm(x_i - x_j)^2 \/ 2 h^2))
                 / (sum_k m_(i k) e^(-norm(x_i - x_k)^2 \/ 2 h^2)).
$
Row normalization is an attention budget, so effects are $O(1)$ as
$N -> infinity$.
The ladder is FJ with fixed row-stochastic weights (DeGroot,
DeMarzo--Vayanos--Zwiebel persuasion bias), then precision weights,
then relevance weights, then drift in selectivity $1 \/ h$ (feed
personalization) and exposure $m$.
As $h -> infinity$ the model is linear FJ on $m$.

*Exact mean-field closure* ($N -> infinity$, complete $m$,
$theta tilde N(0, Sigma)$; derived).
Gaussian conjugacy makes the kernel average linear, so the fixed point
is $x = alpha compose theta$ mode by mode in the eigenbasis of
$Sigma$:
$
  alpha_k (1 + c / (1 + alpha_k^2 sigma_k^2 \/ h^2)) = 1,
  quad c = kappa / kappa_0.
$
The left side is non-monotone iff $c > 8$,
so there are multiple equilibria with hysteresis,
and the mode with the largest $sigma_k$ tips first.
At $N = 600$ simulation matches the closure to within 0.001.

#figure(
  table(
    columns: 6,
    table.header([$c$], [$h$], [A trace], [A share], [B trace], [B share]),
    [3], [5], [0.40], [0.534], [0.25], [0.250],
    [3], [0.5], [2.15], [0.858], [0.41], [0.250],
    [3], [0.25], [4.86], [0.622], [2.45], [0.250],
    [3], [0.05], [6.34], [0.534], [3.94], [0.250],
    [12], [0.25], [1.66], [0.987], [0.03], [0.250],
  ),
  caption: [
    Mean-field fixed points.
    Topic A loads on a common factor (prior trace 6.40, share 0.531);
    topic B does not (prior trace 4.00, share 0.25).
    At $c = 12$ topic A has tipped and B has not.
  ],
)

- Rising selectivity raises the trace on every topic.
- The share of the unloaded topic is exactly flat;
  the share of the loaded topic rises and then falls.
- With $c > 8$ the loaded topic shows a concentration spike before a
  general rise in disagreement:
  a candidate story for the race spike (conjectured).
- Finite $N$ (400, sparse exposure) confirms the pattern and the two
  branches, but mean field breaks down when $N h^D$ is small.
- Inference about source _precision_ (Student-t weights, as in
  Gentzkow--Shapiro) fails: heavy tails keep distant sources
  influential and the trace barely rises.
  The story must be relevance, not precision.

Assessment: the best analytics so far (exact closure, sharp tipping
condition) and a standard micro story, weakened by naive DeGroot
aggregation.
Topic heterogeneity comes from the factor structure of priors alone,
with no salience rule.
There is no potential function.
Not explained: falling concentration (speech, police) and the abortion
U-shape.
Not ordinally robust: $h$ is in opinion units, and making it
scale-free kills the trend mechanism.

== Round 0 (2026-10-16): review of the `setup.typ` model

Seven subagents.

=== Bottom line

The core mechanism holds up:
homophilous rewiring erodes the network's weakest seams,
which concentrates opinion variance on one axis.
As written the model has three problems.

+ *Homophily is too weak.*
  $S$ has unit trace, so $S_(i j) approx 1 \/ N$ against anchor ties of
  order one.
  Homophily matters only if $rho_0 \/ rho$ is of order $1 \/ N$.
+ *No saturation.*
  Below threshold the share barely moves;
  above it the top mode explodes.
  There is no stable polarized state.
+ *Amplification needs repulsion everywhere.*
  Every route to a sharp single axis runs through negative ties,
  and 42--51% of pairs end up negative.
  The empirical case for backfire effects is thin.

The pivotal decision is whether ties can be negative.

#figure(
  table(
    columns: 3,
    table.header([], [*Signed ties*], [*Nonnegative ties*]),
    [story], [spectral instability, amplification], [fragmentation: a minor
      difference stops being dissolved],
    [interpretation of $A$], [influence weights; negative = backfire], [contact
      or time],
    [linearity], [kept], [lost (piecewise affine)],
    [main risk], [weak evidence for backfire], [no amplification,
      $lambda_j (Sigma^*) <= lambda_j (Sigma_0)$],
  ),
  caption: [The two coherent versions of the model.],
) <tab:fork>

=== Problems with the dynamics

Ranked by severity, each with a suggested fix.

+ *Scale mismatch.*
  At $N = 200$ with $kappa_0 = kappa = rho = 0.05$
  the share goes $0.278, 0.314, 0.402$ at $rho_0 = 0.2, 0.005, 0.001$.
  Fix: scale $S$ in degree units, e.g. $c G \/ (op("tr") G \/ N)$.
+ *No saturation.*
  Fix: a bounded disagreement cost (see tie targets below),
  or make near-threshold noise amplification the headline mechanism.
+ *Repulsion.*
  Fix: justify negative ties as affective polarization restricted to
  salient out-groups, or drop them (@tab:fork).
+ *Gram similarity is alignment times extremity, not closeness.*
  Two identical moderates have $S_(i j) approx 0$.
  Fix: frame ties as driven by shared direction (co-partisanship),
  and check robustness to a closeness-based similarity.
+ *No time trend.*
  The model converges to a fixed point.
  Fix: slow parameter drift (falling $rho_0$ as geography and work
  matter less, rising $rho$ with social media).
  The share is convex in $rho_0$ near threshold,
  so drift gives an accelerating rise.
+ *Weak identification.*
  $Sigma^*$ confounds unobserved $X_0$ with network amplification.
  Fix: panels, network data, or the noise early-warning signature.
+ *Total variance collapses* to about 5% of its initial level:
  consensus plus sharper alignment.
  The model speaks only to the share.
  Say so explicitly.

=== Signed ties and the budget

Three agents worked on open problem 1 of the microfoundation
(a time budget is inconsistent with signed ties).

*The fixed-row-sum constraint is forced.*
Among affine constraints on symmetric $A$,
fixing net row sums is the only one under which projection yields
exactly $G$ for every $X$ (derived).
Splitting $A = A^+ - A^-$ with a budget on the total fails:
nothing gets projected out.

*Influence reading.*
Read $A$ as influence weights and $d_i$ as total susceptibility.
Stubbornness is then a trait,
ties decide whom you listen to rather than how much,
and a negative tie is backfire.
This also answers open problem 2:
susceptibility must go somewhere, so equality is natural.
Each agent's opinion problem stays concave with negative ties.
Cost: $A$ is no longer a contact network.

*A soft degree penalty is fragile.*
Replacing the budget with $gamma sum_i (d_i - b_i)^2$
diverges for small $gamma$ and, at $rho = 0.2$, for every stable
$gamma$.
Extreme agents lose degree, are repelled from everyone, and this
compounds.
The budget does real stabilizing work.

*Nonnegative ties with a time budget.*
The linear model is the exact interior limit:
with a complete $A_0$ and small $rho$ the two agree to machine
precision.
With $A >= 0$ every Laplacian is PSD, so no mode is amplified
and runs that diverge in the linear model converge.
The share still rises, from 0.42 to between 0.46 and 0.84, by
fragmentation.
In the 0.84 run every pair with $S_(i j) < 0$ is clipped to zero and
the graph splits along the dominant cut.
A budget _ceiling_ never binds, since ties are pure cost.

*Nonnegative regime of the linear model.*
Staying in the regime where the linear model keeps $A >= 0$
costs the central mechanism.

- With $A^* >= 0$, $L^* succ.eq 0$, so $lambda_j (Sigma^*) <= lambda_j (Sigma_0)$
  for every $j$.
  The share can rise only by shrinking the other modes faster.
- On a sparse $A_0$ nonnegativity needs a full-support floor
  $A_0 = B + epsilon (bold(1) bold(1)^top - I)$
  with $rho_0 epsilon >= rho max_(i j) (-S_(i j))$.
- Homophily can at most cancel the floor:
  $lambda_max (S) <= N max_(i != j) (-S_(i j))$ for PSD $S$ with zero row
  sums,
  so the stability condition holds automatically.
  In the runs homophily cancelled about 15% of the floor's stiffening.
- Remark 3 is far from necessary:
  the binding floor is about $2 rho \/ (rho_0 N)$,
  roughly 20 times below remark 3's $rho \/ (2 rho_0)$ at $N = 80$.
- Across 32 runs ($N = 80$, initial share 0.29)
  homophily raised the share by at most 1.1x.
  The intrinsic structure $B$ alone reaches 0.55,
  but the floor that nonnegativity forces erases this.

Keeping ties nonnegative therefore requires a different source of
softening than negative cross-ties.

=== Tie targets beyond Gram

Keep a Gram-type tie target and fix saturation on the opinion side.

- Under the budget projection any kernel $K$ becomes $J K J$
  (kernel-PCA similarity).
  Squared Euclidean distance reduces exactly to Gram.
  RBF, cosine and $p$-norms are new but break the $(L, G)$
  reparameterization;
  $p$-norms also depend on issue coding.
- Endogenous salience $W prop Sigma^p$ in a Mahalanobis distance gives
  a target $prop G^(p + 1)$.
  This keeps every property the spectral analysis uses,
  but gained little at $p = 1$.
- A bounded disagreement cost such as Welsch,
  $phi(r) = h (1 - e^(-r \/ h))$,
  gives the opinion pull $A_(i j) e^(-d_(i j)^2 \/ h) (x_j - x_i)$.
  Repulsion between distant agents fades, which yields saturation.
  The same cost gives the kernel tie target,
  so the microfoundation survives.
  Saturation requires a fixed bandwidth $h$:
  a scale-free $h$ explodes again.

#figure(
  table(
    columns: 4,
    table.header([*Target, opinion pull*], [$s = 2$], [$s = 4$], [$s = 30$]),
    [Gram, linear (current)], [0.955], [explodes], [explodes],
    [Gram, Welsch], [0.712], [0.922], [0.997],
    [RBF $>= 0$, Welsch], [0.511], [0.502], [0.478],
  ),
  caption: [
    Final $lambda_1$ share by tie target and opinion pull
    ($N = 100$, initial share 0.47).
    The scale $s$ multiplies the tie target; the current model is
    $s = 1$, with share 0.62.
    The Welsch run at $s = 30$ has bounded variance.
  ],
) <tab:targets>

=== Opportunity network

Split $A_0$ into an opportunity network $M$
(geography, work, family) and baseline ties $A_0 = c M$,
and let a share $omega$ of homophily ignore opportunity:
$
  Delta A
  = rho_0 (A_0 - A) + rho ((1 - omega) M compose S + omega S - A).
$
The current model is $omega = 1$.

- Homophily gated by block $M$ cannot amplify a between-region divide:
  it stiffens within-region contrasts only (derived).
- Rising $omega$ (online contact) pushes the system across the
  threshold along the geographic axis.
  This is a candidate time trend.
- Near threshold ($N = 200$, four regions, initial share 0.36,
  $rho = 0.1$), the share is 0.87 at $omega = 0$,
  0.97 at $omega = 0.5$, and explodes at $omega = 1$.
  Negative cross-region weight rises with $omega$.
  Away from threshold $omega$ barely matters.
- Cost: $(M compose S) bold(1) != 0$, so degrees depend on opinions
  and the $(L, G)$ reparameterization breaks.
  Low-rank or block $M$, or perturbation around
  $M = m bold(1) bold(1)^top$, are the tractable routes.
- The budget microfoundation carries over with tie cost
  $A_(i j)^2 \/ (2 M_(i j))$ (derived, not checked).
- Keep $X_0$.
  Noise-only reversion is better as an addition;
  bounded confidence is redundant given signed $S$.

=== Identification and data

Use three signatures rather than full structural estimation.

+ *Do trace and share move together?*
  Repeated cross-sections, feasible now.
  In simulation, rising cohesion ($kappa$ or $rho_0 \/ rho$ up)
  moves them in opposite directions:
  trace $5.08 -> 2.13$ and share $0.272 -> 0.332$ as $kappa$ goes
  from 0.002 to 0.02.
  Only with $S$ scaled up near threshold does homophily raise both.
  Under the current $1 \/ T$ scaling, more homophily dilutes ties and
  _lowers_ the share.
  The GSS decomposition in `paper/gss.tex` finds rising trace with
  flat or falling share for most topics
  (share rises only for race in the 2020s and abortion).
  Read through the model, most topics show declining cohesion,
  not homophilous feedback.
+ *Persistence along the top axis.*
  Panels.
  Fit a permanent component plus per-direction AR(1),
  $Gamma(h) = P + phi^h V$,
  using cross-issue autocovariances to avoid item measurement error.
  Near threshold $phi$ should be highest on the dominant axis.
  Data: GSS panels 2006--14, ANES 2016--20, CES 2010--14.
+ *Movers separate $kappa$ from $kappa_0$.*
  Change in party registration after moving to a place of different
  partisanship (Cantoni--Pons design), using voter files.

#figure(
  table(
    columns: 3,
    table.header([*Parameter*], [*Data*], [*Assumption*]),
    [$rho_0 \/ rho$, $kappa \/ kappa_0$ (one index)],
    [GSS 1972--2024, ANES cumulative],
    [stable $X_0$, $L_0$; slow drift; fixed item scaling],

    [$kappa_0 + kappa macron(d)$, per-axis $phi$],
    [GSS, ANES, CES panels],
    [stationary within window],

    [$kappa$ vs $kappa_0$], [L2 movers], [move timing exogenous],
    [level and drift of $rho$],
    [spousal concordance by cohort (L2); GSS discussant modules],
    [these ties represent $A$],

    [$X_0$ vs amplification],
    [traits model $X_0 = Z B$, DFL reweighting],
    [demographics act only through $X_0$],

    [$omega$],
    [between- vs within-county top-axis variance (CES, geocodes)],
    [county approximates $M$],
  ),
  caption: [
    Identification.
    The fixed point depends on $(rho_0, rho)$ only through
    $rho_0 \/ rho$ (checked in simulation).
  ],
) <tab:ident>

Practical issues:
the share is not invariant to item rescaling,
so fix one per-item scaling across years
(`unit_variance_at_t0` in `surveys/encodings.py`, `ranks_at_t0` as
robustness);
per-year polychorics would erase the trace signal.
Cohort replacement breaks stable $X_0$.
CES is the best public source for place-level work.

State as assumptions (untestable):
symmetric ties, the $1 \/ T$ normalization,
the anchor functional forms,
observed eras being near the fixed point,
and respondents standing in for the unobserved network.

*Restricted data*, ranked:

+ *GSS sensitive geography* (state 1973--, county 1993--, tract
  1998--) via NORC.
  Gives $M$ over a long time series on the paper's items.
  Worth it.
+ *ANES restricted geocodes* via the ICPSR Virtual Data Enclave
  (IRB and data use agreement).
  Complements GSS.
+ *L2 voter file.*
  Princeton DDSS hosts it but only for Princeton affiliates.
  Ask Harvard Library or IQSS about Redivis access.
  Gives movers ($kappa$), spousal concordance ($rho$) and residential
  segregation ($M$), but only party.
+ *Meta Social Connectedness Index.*
  Public; county-to-county friendship as a proxy for $L_0$ and $M$.
  Free, so use it.
+ *Add Health restricted.*
  Real school friendship networks, but thin political attitudes, and
  the PI must be faculty.
+ *Meta research access.*
  Slow and uncertain; only if the rising-$omega$ story becomes central.

Not verified: GSS fees and timelines, Harvard L2 access, and which
network modules record alters' politics.
The DDSS page returned 403, so its contents come from search snippets.

=== Decisions

+ Signed or nonnegative ties (@tab:fork).
+ Whether to give up opinion-side linearity for saturation (Welsch).
+ How to scale $S$ relative to $A_0$.
  Under the current scaling the model cannot produce co-moving trace
  and share, and the GSS data mostly show the opposite pattern anyway.
+ Whether to adopt $M$ and $omega$ as the source of the time trend.
