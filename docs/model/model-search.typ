#import "@ali/template:0.1.0" as ali
#show: ali.template

= Model search
A living document.
Started 2026-10-16; last updated 2026-10-16.
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
- Write-ups here are concise and readable top to bottom.
  Superseded findings stay in their round section;
  _Current status_ always reflects the latest view.
- Scratch scripts from rounds 0--2 were not kept;
  every result they produced is recorded below.

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
+ not bake in party: group or party structure, including two-sidedness,
  should arise from the model.

Soft criteria:

+ is parsimonious: emergent properties should come from a few simple,
  understandable micro characteristics, not from added bells and
  whistles;
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
  e.g. online contact or platform design;
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

Next steps:
+ an exact or better closure in the transition regime;
+ replicate the symmetry-breaking spike across seeds;
+ attention budget and a Duverger-type force for emergent two-sided
  groups;
+ the GSS sociability test of the trace bound.

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
(`unit_variance_at_t0` in `variable_coding.py`, `ranks_at_t0` as
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
