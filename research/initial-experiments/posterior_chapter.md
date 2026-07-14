# Posterior Inference and the Joint Score: from the Kalman Smoother to a Non-Gaussian Message-Passing Hypothesis

*Draft chapter for the MSc thesis of Giovanni Mantovani, Bocconi University, April 2026.*
*Supervisor: Prof. Marc Mézard. Tutor: Jérôme Garnier-Brun.*

---

## 1. Scope of this chapter

The preceding chapters have established, in two settings, what the joint score of a diffused sequence looks like when the clean signal is Markov. In the Gaussian AR(1) case the joint score is affine in $x$ and the object that carries all the structure is the precision matrix $\Sigma_t^{-1}$ [`thesis_part1_expanded.pdf`, `long_derivation_note.pdf`]. In the Laplace AR(1) case the joint score becomes nonlinear and the Gaussian notion of a fixed precision matrix is replaced by a position-dependent curvature field $H_t(x) = -\nabla^2 \log p_t(x)$, whose two-dimensional structure at $K = 1$ is entirely controlled by a single scalar function of a single residual coordinate [`laplace_K1_benchmark_memo.pdf`, `laplace_K1_compendium.pdf`].

These two results look formally different: one is matrix-algebraic, the other is geometric. This chapter reconciles them by moving the discussion to a common ground which is more general than either — **the posterior measure $p(a \mid x)$ over the clean sequence given the noisy sequence**. The vector Tweedie identity [`DraftAR1.pdf`, `companion_explanations.pdf`] states that the joint score is, up to simple linear rescalings, the posterior mean $\mathbb{E}[a \mid x]$. The question *"what is the structure of the joint score?"* is therefore the same question as *"what is the structure of the posterior inference problem?"*.

Reframed this way, the story becomes unified. Markovianity of the prior implies that the posterior is a chain-structured Gibbs measure, independently of whether the innovation law is Gaussian or not. Forward–backward message passing on that chain computes the posterior mean — and therefore the joint score — in linear time *in $K$*. The Gaussian case is the one where the messages are themselves finite-dimensional (Gaussians, two scalars per frame) and forward–backward reduces to the Kalman smoother [`DraftAR1.pdf`]. The non-Gaussian case is the one where the messages are general densities and exact message passing is no longer finite-dimensional — but the chain factorisation of the posterior remains exact, and this is what a learned architecture can exploit.

This reframing motivates the concrete working hypothesis that closes the chapter: if the clean data is known (or modelled) to be Markov, a score network whose parameterisation mirrors the forward–backward message-passing structure of the posterior should be more sample-efficient than a generic architecture that has to discover this locality from data. We state the hypothesis carefully, identify its scope, and flag two places where it can fail — one of which, the K=2 obstruction derived in the previous chapter, is *not* a failure of the hypothesis but an instructive subtlety about what form of Markov structure the architecture should encode.

---

## 2. The Tweedie identity and why the score is a posterior-inference problem

We work throughout in the setting fixed by the rest of the thesis. Frames $a_0, \dots, a_{K-1} \in \mathbb{R}$ are generated from a prior $P_0(a)$ and each frame is corrupted independently by an Ornstein–Uhlenbeck channel:

$$
X_k \;=\; e^{-t}\, a_k \;+\; \sqrt{\Delta_t}\, Z_k, \qquad Z_k \overset{\text{i.i.d.}}{\sim} \mathcal{N}(0,1), \qquad \Delta_t \;=\; 1 - e^{-2t}.
\tag{2.1}
$$

The joint noisy density is

$$
P_t(x) \;=\; \int P_0(a) \,\prod_{k=0}^{K-1} \mathcal{N}\!\bigl(x_k;\, e^{-t} a_k,\, \Delta_t\bigr)\, \mathrm{d}a.
\tag{2.2}
$$

The **joint score** is $S(x, t) := \nabla_x \log P_t(x) \in \mathbb{R}^K$; it is not the stack of per-frame marginal scores $\partial_{x_k} \log p_t(x_k)$, which would discard every inter-frame correlation. The two coincide only when the clean frames are independent; for any Markov chain with nonzero coupling they differ, and the relevant object for all subsequent structural claims is the joint score.

The Tweedie identity expresses the joint score in terms of the posterior mean of the clean sequence. Differentiating (2.2) under the integral and rearranging gives, componentwise,

$$
\boxed{\;
S_k(x, t) \;=\; \frac{e^{-t}\, \mathbb{E}[a_k \mid X = x] \;-\; x_k}{\Delta_t}
\;}
\tag{2.3}
$$

This holds for any prior $P_0$ with integrable first moment and is the starting point of every structural result that follows. Two remarks are in order.

**Remark 2.1 (what Tweedie does and does not say).** Equation (2.3) is a *rewriting*, not a reduction of difficulty: the hardness of computing $S(x, t)$ is exactly the hardness of computing the posterior mean. What (2.3) buys us is *a change of perspective*. The joint score inherits whatever tractability the posterior inference problem has, and in particular whatever *graphical* tractability the posterior has. This is the observation that makes Markovianity useful.

**Remark 2.2 (the role of $e^{-t}$ and $\Delta_t$).** The prefactor $e^{-t}$ rescales the latent frame into the signal scale at time $t$; the denominator $\Delta_t$ is the OU noise variance. Together they ensure the small-$t$ limit is well-defined ($S \to -\infty \cdot$ singular when $x$ is off-support of the clean law) and the large-$t$ limit is $S(x,t) \to -x/\Delta_t$, the score of the Gaussian prior to which the diffusion collapses at infinite time.

---

## 3. The central observation: Markovianity is preserved by independent coordinatewise noising

The whole subsequent development rests on a single elementary observation.

**Proposition 3.1 (Markov-preserving OU channel).** *Suppose the clean prior is Markov,*
$$
P_0(a_0, \dots, a_{K-1}) \;=\; p_0(a_0) \prod_{k=0}^{K-2} M(a_{k+1} \mid a_k),
\tag{3.1}
$$
*and the noising channel is coordinatewise independent,*
$$
P(x \mid a) \;=\; \prod_{k=0}^{K-1} \mathcal{N}\!\bigl(x_k;\, e^{-t} a_k,\, \Delta_t\bigr).
\tag{3.2}
$$
*Then the joint law of $(a, x)$ has factor-graph structure*
$$
P(a, x) \;=\; p_0(a_0)\, \mathcal{N}(x_0; e^{-t}a_0, \Delta_t) \prod_{k=0}^{K-2} M(a_{k+1}\mid a_k)\, \mathcal{N}(x_{k+1}; e^{-t} a_{k+1}, \Delta_t),
\tag{3.3}
$$
*and in particular the posterior factorises as*
$$
p(a \mid x) \;\propto\; p_0(a_0) \,\mathcal{N}(x_0; e^{-t}a_0, \Delta_t) \prod_{k=0}^{K-2} M(a_{k+1}\mid a_k)\, \mathcal{N}(x_{k+1}; e^{-t} a_{k+1}, \Delta_t).
\tag{3.4}
$$

*The posterior $p(a\mid x)$ is therefore a Gibbs measure on a one-dimensional chain factor graph with unary potentials at every vertex (the observations $x_k$) and binary potentials on every edge (the transition kernels $M$).*

*Proof.* Both (3.1) and (3.2) are Markovian over the same one-dimensional chain and are independent of one another. Multiplying them gives (3.3). The observations $x$ are known, so (3.4) is obtained by treating the Gaussians $\mathcal{N}(x_k; e^{-t}a_k, \Delta_t)$ as unary $x_k$-conditional potentials in $a_k$. The resulting factor graph is a tree (in fact a chain), so the posterior decomposes into unary and binary clique terms with no loops. $\square$

Proposition 3.1 is not deep — it is essentially a restatement of conditional independence — but it is the *load-bearing* fact of this chapter, because it is what guarantees that forward–backward message passing is applicable. Two points deserve comment.

**Remark 3.2 (what the OU channel contributes).** The OU channel is coordinatewise independent by construction (equation (3.2)). Any noising channel of the form $P(x\mid a) = \prod_k P_k(x_k\mid a_k)$ would give the same result; the specific Gaussian form of the OU channel enters only through the unary potentials at the bottom of the factor graph. The channel does not add any temporal coupling of its own; all coupling between frames in the noisy density $P_t(x)$ is inherited from the Markov prior.

**Remark 3.3 (why this is a theorem about the posterior, not about $P_t(x)$).** It is tempting to infer from (3.3) that $P_t(x) = \int P(a, x)\, \mathrm{d}a$ also has a clean factor-graph structure, with local potentials on $x$. This is false in general. Marginalising $a$ out of (3.3) destroys the nearest-neighbour factorisation of the Gibbs measure. The noisy $P_t(x)$ is Markov in $x$ (a direct consequence: $X_2 \perp X_0 \mid X_1$ whenever $a_2 \perp a_0 \mid a_1$ and the channels are independent), but it is not expressible as a product of independent bonds when the prior is non-Gaussian. This is precisely the content of the K=2 obstruction [`laplace_K1_benchmark_memo.pdf`, Thm. 13]. The posterior factor graph and the joint-density factorisation are distinct objects; they agree in the Gaussian case and part ways in the non-Gaussian case. We return to this in Section 6.

---

## 4. Forward–backward on the posterior chain: the general algorithm

With Proposition 3.1 in hand, the posterior mean $\mathbb{E}[a_k \mid x]$ is computed by the standard forward–backward algorithm on a chain. We give it first in full generality, without assuming Gaussianity.

Define forward and backward messages
$$
\alpha_k(a_k) \;:=\; p(a_k,\, x_0,\dots, x_k),
\qquad
\beta_k(a_k) \;:=\; p(x_{k+1},\dots, x_{K-1}\mid a_k).
\tag{4.1}
$$

They satisfy the recursions
$$
\alpha_0(a_0) \;=\; p_0(a_0)\, \mathcal{N}(x_0; e^{-t}a_0, \Delta_t),
\tag{4.2}
$$
$$
\alpha_{k+1}(a_{k+1}) \;=\; \mathcal{N}(x_{k+1}; e^{-t}a_{k+1}, \Delta_t) \int M(a_{k+1}\mid a_k)\, \alpha_k(a_k)\, \mathrm{d}a_k,
\tag{4.3}
$$

with the backward pass obtained symmetrically:
$$
\beta_{K-1}(a_{K-1}) \;=\; 1,
\qquad
\beta_k(a_k) \;=\; \int M(a_{k+1}\mid a_k)\, \mathcal{N}(x_{k+1}; e^{-t}a_{k+1}, \Delta_t)\, \beta_{k+1}(a_{k+1})\, \mathrm{d}a_{k+1}.
\tag{4.4}
$$

The marginal posteriors are then
$$
p(a_k \mid x) \;\propto\; \alpha_k(a_k)\, \beta_k(a_k),
\tag{4.5}
$$
and the posterior means required for the Tweedie identity (2.3) are
$$
\mathbb{E}[a_k \mid x] \;=\; \int a_k\, p(a_k \mid x)\, \mathrm{d}a_k.
\tag{4.6}
$$

**Theorem 4.1 (general posterior-factor-graph representation of the joint score).** *Let $P_0$ be a Markov prior as in (3.1) and let $X$ be the OU-noised sequence as in (2.1). Then the joint score (2.3) has the representation*
$$
S_k(x, t) \;=\; \frac{e^{-t}}{\Delta_t}\, \frac{\int a_k\, \alpha_k(a_k)\, \beta_k(a_k)\, \mathrm{d}a_k}{\int \alpha_k(a_k)\, \beta_k(a_k)\, \mathrm{d}a_k} \;-\; \frac{x_k}{\Delta_t},
\tag{4.7}
$$
*where $\alpha_k, \beta_k$ are the forward and backward messages defined by (4.2)–(4.4).*

*Proof.* The Tweedie identity (2.3) expresses $S_k$ through $\mathbb{E}[a_k \mid x]$. Proposition 3.1 establishes that the posterior $p(a\mid x)$ is a chain-structured Gibbs measure. Forward–backward on a chain factor graph is an exact algorithm; combining (4.5) and (4.6) with (2.3) yields (4.7). $\square$

**Remark 4.2 (scope of Theorem 4.1).** The theorem is a structural statement, not an algorithmic one. It says that *for every Markov prior and OU channel, the joint score is expressed through two local recursions on a chain*. It does not, by itself, claim linear-time computation: equations (4.3)–(4.4) involve integrals over arbitrary densities, which are not generically finite-dimensional. The algorithmic claim is a *specialisation* to priors for which the messages live in a finite-dimensional family.

**Remark 4.3 (the role of the one-dimensional chain structure).** The chain assumption is doing real work. If the prior graph were a tree, the analogous factorisation would still hold and forward–backward would generalise straightforwardly. For a general graph with cycles, the chain algorithm above is replaced by loopy belief propagation, which is no longer exact in general. Videos, trajectories, and sequential time-series data are one-dimensional chains by construction, so the chain case is not a restrictive hypothesis for the physical domain this thesis targets.

---

## 5. Specialisation 1: Gaussian AR(1) and the Kalman smoother

The Gaussian case is the one where the messages $\alpha_k, \beta_k$ are themselves Gaussian densities, each described by a mean and a variance. Forward–backward therefore collapses to the propagation of four scalars per frame — the forward mean and variance $(\hat a_{k\mid k}, P_{k\mid k})$ and the smoothed mean and variance $(\hat a^s_{k\mid K-1}, P^s_{k\mid K-1})$ — and the integrals (4.3), (4.4) close in closed form. This is the Kalman filter–smoother.

Concretely, for the scalar Gaussian AR(1) prior
$$
a_0 \sim \mathcal{N}(\mu_0, \sigma_0^2), \qquad a_{k+1} = \alpha a_k + \eta_k, \qquad \eta_k \overset{\text{i.i.d.}}{\sim} \mathcal{N}(0, \sigma_\eta^2),
\tag{5.1}
$$
coupled to the OU channel (2.1), the forward recursion reads [`DraftAR1.pdf`, Sect. 10.2]

$$
\hat a_{k+1\mid k} \;=\; \alpha \hat a_{k\mid k}, \qquad P_{k+1\mid k} \;=\; \alpha^2 P_{k\mid k} + \sigma_\eta^2,
\tag{5.2}
$$
$$
K_{k+1} \;=\; \frac{e^{-t} P_{k+1\mid k}}{e^{-2t} P_{k+1\mid k} + \Delta_t}, \qquad \hat a_{k+1\mid k+1} \;=\; \hat a_{k+1\mid k} + K_{k+1} \bigl(x_{k+1} - e^{-t} \hat a_{k+1\mid k}\bigr),
\tag{5.3}
$$

and the backward Rauch–Tung–Striebel smoother is
$$
L_k \;=\; \alpha P_{k\mid k} P_{k+1\mid k}^{-1}, \qquad \hat a^s_{k\mid K-1} \;=\; \hat a_{k\mid k} + L_k \bigl(\hat a^s_{k+1\mid K-1} - \hat a_{k+1\mid k}\bigr).
\tag{5.4}
$$

Substituting $\mathbb{E}[a_k \mid x] = \hat a^s_{k\mid K-1}$ into (2.3) gives the joint score in $O(K)$ operations. The equivalent matrix expression, $S(x,t) = -\Sigma_t^{-1}(x - \mu_t)$, is a special case of (4.7) — it is what (4.7) evaluates to when forward and backward messages are explicitly Gaussian and the integrals can be done analytically.

Two structural consequences are worth naming.

**Observation 5.1 (why the Gaussian case is special).** What makes the Gaussian case analytically rigid is *conjugacy*: the Gaussian-times-Gaussian integrand in (4.3) is itself Gaussian, and the integral over $a_k$ produces another Gaussian. The forward message belongs to the same family at every $k$, so it is fully described by two scalars. The non-Gaussian case breaks conjugacy: starting from a Gaussian prior on $a_0$ (which is how the Laplace AR(1) model of this thesis is set up), the forward message after one step of the recursion is *Normal–Laplace-convolved-with-Gaussian-observation*, which is already outside the Gaussian family.

**Observation 5.2 (the $O(K)$ claim is a Gaussian artefact, not a Markov artefact).** It is a common informal claim that "Markov priors admit $O(K)$ exact score computation." This is correct *for Gaussian Markov priors* and *conjugate-family Markov priors more generally*; it is incorrect in full generality. What Markovianity buys is the chain factor-graph structure of the posterior (Theorem 4.1). What Gaussianity additionally buys is that the messages on that chain are themselves finite-dimensional. The architecture hypothesis we formulate below is consistent with this distinction: it claims structural alignment between architecture and posterior factor graph, *not* exact $O(K)$ score computation at test time.

---

## 6. Specialisation 2: Laplace K=1, and what happens at K=2

The Laplace AR(1) model of the previous chapter is the simplest setting in which conjugacy fails and Theorem 4.1 still gives exact, closed-form insight. We take the prior $a_0 \sim \mathcal{N}(\mu_0, \sigma_0^2)$, $a_{k+1} = \alpha a_k + \varepsilon_k$ with $\varepsilon_k \sim \text{Lap}(0, b)$ i.i.d.

### 6.1 K = 1 as exact message passing at one bond

At $K = 1$ the chain has two vertices $a_0, a_1$ and one bond. Forward–backward collapses: there is only the single edge $a_0 \to a_1$, and the posterior is

$$
p(a_0, a_1 \mid x_0, x_1) \;\propto\; \mathcal{N}(a_0; \mu_0, \sigma_0^2)\, \mathcal{N}(x_0; e^{-t}a_0, \Delta_t)\, \text{Lap}(a_1 - \alpha a_0;\, 0, b)\, \mathcal{N}(x_1; e^{-t}a_1, \Delta_t).
\tag{6.1}
$$

Integrating $a_0$ out first — exploiting the Gaussian–Gaussian conjugacy on the $a_0$ end, which the Laplace factor does not disturb because it only couples $a_0$ and $a_1$ linearly through $\alpha a_0$ — yields the factorisation

$$
p_t(x_0, x_1) \;=\; \underbrace{\mathcal{N}(x_0; e^{-t}\mu_0, v)}_{\text{Gaussian site } g_t(x_0)} \;\cdot\; \underbrace{h_t(r)}_{\text{1D non-Gaussian bond}},
\qquad
r \;:=\; x_1 - \nu - \rho x_0,
\tag{6.2}
$$

where the structural constants $v, \rho, \nu, \tilde b, \tau$ are defined in Table 3.1 of `laplace_K1_compendium.pdf` and the residual density $h_t$ is the Normal–Laplace density

$$
h_t(r) \;=\; \frac{e^{a^2/2}}{2\tilde b}\left[\,e^{-r/\tilde b}\, \Phi\!\Big(\tfrac{r}{\tau} - a\Big) \;+\; e^{+r/\tilde b}\, \Phi\!\Big(-\tfrac{r}{\tau} - a\Big)\,\right], \qquad a = \tau/\tilde b.
\tag{6.3}
$$

[`laplace_K1_benchmark_memo.pdf`, Thm. 1; `laplace_ar1_ou_note.pdf`, eq. 55]. The joint score follows by differentiation and has the compact form [`laplace_K1_benchmark_memo.pdf`, Thm. 2]

$$
S(x,t) \;=\; \begin{pmatrix} -\dfrac{x_0 - e^{-t}\mu_0}{v} \,-\, \rho\, \psi_t(r) \\[4pt] \psi_t(r) \end{pmatrix}, \qquad \psi_t(r) := \frac{h_t'(r)}{h_t(r)}.
\tag{6.4}
$$

**Interpretation via Theorem 4.1.** The representation (6.2) *is* the output of the forward step of message passing: the site term $g_t(x_0)$ is the marginal of the forward message $\alpha_0$ (expressed in $x_0$-coordinates), and the bond $h_t(r)$ is what remains after integrating $a_0$ out, conditioned on the observation $x_1$. The residual variable $r$ is precisely the posterior-mean-compensated innovation: $r = x_1 - \rho x_0 - \nu$ is the component of $x_1$ orthogonal, in posterior-covariance sense, to the information already carried by $x_0$. The Gaussian benchmark $r^{(G)}_t \sim \mathcal{N}(0, \tau_G^2)$ with $\tau_G^2 = \tau^2 + 2\tilde b^2$ recovers (6.2) with $h_t$ replaced by a Gaussian density, so the deviation of $h_t$ from $\mathcal{N}(0, \tau_G^2)$ is a clean, scalar measure of how far the posterior is from conjugacy.

### 6.2 K = 2: the posterior is still Markov, but the density factorisation breaks

At $K = 2$ Theorem 4.1 still applies: the posterior $p(a_0, a_1, a_2 \mid x_0, x_1, x_2)$ is a chain-structured Gibbs measure with two Laplace bonds, and forward–backward is exact. The messages are no longer in a closed family (the forward message $\alpha_1(a_1)$ is a Gaussian-convolved Laplace-convolved-with-Gaussian-observation hybrid, related to $h_t$), so *exact* computation via message passing is no longer finite-dimensional, but the chain structure of the inference problem is intact.

What fails at $K = 2$ — and this is the content of the previous chapter [`laplace_K1_benchmark_memo.pdf`, Thm. 13; `laplace_K1_compendium.pdf`, Ch. 13] — is the *direct factorisation of the joint noisy density $p_t(x_0, x_1, x_2)$ as a product of independent one-dimensional bonds*. In the rotated innovation-frequency coordinates $(\ell_0, \ell_1, \ell_2) = (k_0, k_1 + \alpha k_2, k_2)$, the Fourier representation reads

$$
\hat q_t(\ell_0, \ell_1, \ell_2) \;=\; \exp\!\Bigl[\, i e^{-t}\mu_0(\ell_0 + \alpha \ell_1) - \tfrac12 \ell^\top M \ell \,\Bigr] \cdot \frac{1}{1+\tilde b^2 \ell_1^2} \cdot \frac{1}{1+\tilde b^2 \ell_2^2},
\tag{6.5}
$$

with the Gaussian-sector covariance $M$ having cross-covariance $M_{12} = -\alpha \Delta_t$ between the two innovation frequencies. The non-Gaussianity is diagonal — one rational factor per innovation, each depending on one innovation frequency only — but the Gaussian sector couples $\ell_1$ and $\ell_2$. Consequently the joint density $p_t(x_0, x_1, x_2)$ is *not* a product of a Gaussian site and two independent Laplace-like bonds.

**The two Markov structures, and why they part ways.** The K=2 obstruction looks like bad news if one expects "Markovianity $\Rightarrow$ product factorisation." But Theorem 4.1 is not about product factorisation of the noisy density; it is about chain factorisation of the *posterior*. These are two different Markov properties, and they disagree in the non-Gaussian case for a specific and understandable reason. The Gaussian sector of (6.5) has Gaussian cross-coupling $M_{12} = -\alpha \Delta_t$, which arises from the shared OU noise at the vertex $a_1$ that sits between both innovations; this shared noise is invisible to the posterior (the posterior is *conditional* on $x$, so noise has been absorbed into the observation), but it is visible to the joint density (which has *marginalised* the latents out, sharing the OU noise structure with both bonds).

A clean summary:

| Object | What Markovianity gives at K=2 (Gaussian) | What Markovianity gives at K=2 (Laplace) |
|---|---|---|
| Clean prior $P_0$ | chain factorisation by definition | chain factorisation by definition |
| Posterior $p(a\mid x)$ | chain Gibbs, Gaussian messages | chain Gibbs, non-Gaussian messages |
| Joint noisy $P_t(x)$ | tridiagonal precision | tridiagonal Fourier Gaussian sector, **but** joint density not a product of independent 1D bonds |

The architectural takeaway is that a score network which mirrors the *posterior* factor graph is structurally sound in both cases, whereas one that mirrors a *joint-density* product factorisation is sound in the Gaussian case only. This is the design decision we argue for in the next section.

---

## 7. Architectural hypothesis: structured posterior parameterisation as an inductive bias

### 7.1 Motivation

We now address the question that motivates this chapter: can the Markov structure of the clean data be used to design a more sample-efficient score (or flow velocity) network?

The information-theoretic argument in favour is unambiguous. If the data is Markov, the Bayes-optimal joint score depends on each frame $x_k$ only through the conditional expectation $\mathbb{E}[a_k \mid x]$, which in turn is exactly computable from local messages on a chain factor graph (Theorem 4.1). A network with no built-in structure must *learn* this locality from data; a network whose parameterisation enforces it by construction does not. At any finite training budget, the structured network has a strictly smaller hypothesis class and — provided the class still contains the truth — is expected to reach lower Bayes risk with fewer samples. This is the standard inductive-bias argument, and it rests on nothing beyond the factor-graph representation of the posterior.

### 7.2 The hypothesis

Concretely, we propose to parameterise the score network by mimicking the forward–backward structure of (4.7):

$$
s_\theta(x, t)_k \;=\; G_\theta\!\bigl(\,\mu^{\theta}_{k-1\to k}(x, t),\; \bar\mu^{\theta}_{k+1\to k}(x, t),\; x_k,\; t\bigr),
\tag{7.1}
$$

where $\mu^{\theta}_{k-1\to k}$ is a learned forward message summarising $x_{0:k}$, $\bar\mu^{\theta}_{k+1\to k}$ is a learned backward message summarising $x_{k+1:K-1}$, and $G_\theta$ is a learned local combiner. Crucially, the forward message is computed by the same learned recursion at every $k$ (and similarly for the backward message), so the parameter count is *independent of sequence length $K$*. At training time, the denoising score-matching loss is evaluated against exact ground truth from the benchmark (next section), making the comparison sharp.

**Hypothesis 7.1 (structured posterior parameterisation as inductive bias).** *Let $s^{\text{MP}}_\theta$ denote the message-passing score network of (7.1), $s^{\text{LW}}_\theta$ a local-window network with $(2w+1)$-frame receptive field, and $s^{\text{DENSE}}_\theta$ a dense all-to-all network with comparable parameter count. For Markov priors with sufficient coupling and sequence length $K \gtrsim 2w+1$, and diffusion times $t \lesssim t_\star$ where Markov structure is non-trivially present in the posterior:*
$$
\mathcal{E}(s^{\text{MP}}_\theta, t; n) \;<\; \mathcal{E}(s^{\text{LW}}_\theta, t; n) \;<\; \mathcal{E}(s^{\text{DENSE}}_\theta, t; n),
\qquad \mathcal{E}(s, t; n) := \mathbb{E}_{x \sim p_t} \|s(x, t) - S(x, t)\|^2,
\tag{7.2}
$$
*at fixed training-sample budget $n$, with the inequality becoming sharp at small $n$ and small $t$.*

### 7.3 Scope, failure modes, and what the Gaussian limit tells us

**In favour.** By construction, (7.1) contains the Kalman smoother as a special case: taking $\mu^{\theta}_{k-1\to k}$ and $\bar\mu^{\theta}_{k+1\to k}$ to be the affine forward/backward Gaussian messages and $G_\theta$ to be the affine combiner recovers the exact score in the Gaussian AR(1) case. The structured parameterisation therefore does not lose expressivity in the setting where we know the answer.

**Failure mode 1: the scope of validity in $t$.** The posterior factor graph of Section 4 is local in $k$ at every $t$, but the *effective range* of the posterior couplings is not. At small $t$ the coupling is dominated by the prior's nearest-neighbour structure and the locality of (7.1) is well-matched to the truth. At large $t$ the posterior becomes nearly factorised across $k$ (the observations become almost pure noise, so the posterior tends to the prior), but at intermediate $t$ the precision $\Sigma_t^{-1}$ of the Gaussian benchmark fills in band-by-band at order $t^{d-1}$ [`thesis_part1_expanded.pdf`, eq. 4.11]: long-range couplings in the posterior grow. A fixed-width architecture with $(2w+1)$-receptive-field will miss this filling-in; the message-passing architecture, being recursive in $k$, does not have a hard range limit, but the quality of its messages as summaries depends on their parameterisation. The hypothesis as stated is therefore conditional on using a sufficiently expressive message family, and is strongest at small-to-moderate $t$.

**Failure mode 2: when the clean data is not actually Markov.** For real videos, perfect Markovianity is an approximation. Higher-order dependencies are present (e.g. momentum in trajectories is captured by a 2nd-order Markov model, not a 1st-order one). The architecture (7.1) generalises naturally to higher-order chains by widening the message context, so this is a parameterisation choice rather than a fundamental obstruction. The thesis programme is to establish the principle in the 1st-order Markov toy model and then characterise its range.

**Failure mode 3: conjugacy-free messages.** In the Gaussian case the optimal messages are two scalars per frame. In the Laplace case the optimal forward message at $K = 1$ is essentially the Normal–Laplace density $h_t$, parameterised by the bond (one scalar coordinate $r$). At $K \geq 2$ the exact messages live in a progressively richer family. A learned network with a fixed-dimensional message representation is not an exact message passer; it is an amortised approximation to one. The validity of Hypothesis 7.1 is therefore strongest in the regime where a low-dimensional summary of the message captures most of its information — which is exactly the regime where the K=1 curvature $\kappa_t$ is concentrated in a narrow band and the Gaussian benchmark is a good tangent [`laplace_K1_benchmark_memo.pdf`, Figs. 1, 4].

### 7.4 What this chapter does not claim

We do not claim that a message-passing architecture computes the exact score at test time; it does so only in the Gaussian case. We do not claim that the architecture is universally better; its advantage is expected to concentrate at small training-sample sizes and at the diffusion times where Markov structure is informative. And we do not claim that the joint noisy density $P_t(x)$ factorises into independent bonds; the K=2 obstruction says it does not, and the architecture (7.1) is specifically designed to respect posterior locality rather than joint-density factorisation, precisely to side-step that obstruction.

---

## 8. Connection to the existing benchmarks and next steps

The benchmarks built in the Gaussian AR(1) and Laplace AR(1) chapters provide exactly what is needed to test Hypothesis 7.1:

- Gaussian AR(1) at all $K$: the exact joint score is computable in closed form as $-\Sigma_t^{-1}(x - \mu_t)$, and — by Theorem 4.1 specialised — identically computable in $O(K)$ by the Kalman smoother [`DraftAR1.pdf`, Sect. 10; `companion_explanations.pdf`, Sect. 30]. The two computations agree to floating-point precision, giving a cross-checked ground truth.
- Laplace AR(1) at $K = 1$: the exact joint score is given in closed form by (6.4), and has been validated to machine precision against quadrature, Fourier inversion, and Monte Carlo [`laplace_K1_benchmark_memo.pdf`, Sect. 4].
- Laplace AR(1) at $K = 2$: the joint score is accessible by 3D Fourier inversion of the explicit characteristic function (6.5), with Monte-Carlo KDE agreement at the percent level [`laplace_K1_benchmark_memo.pdf`, Sect. 5.3].

These three rungs — conjugate–linear, non-conjugate–nonlinear at a single bond, non-conjugate–nonlinear at two bonds — constitute a benchmark ladder on which any learned score can be scored against exact ground truth. The accompanying Python module `scores_exact.py` (described in the companion note) implements all three.

The concrete next steps, in order of priority:

1. **Verify the Kalman smoother agrees with $\Sigma_t^{-1}$ at machine precision** on a grid of $(\alpha, \sigma_\eta^2, t, K)$ values. This is the cross-check that the $O(K)$ Gaussian path is genuinely exact, not an $O(K)$ approximation with a controlled error. The Python module already does this.
2. **Implement and validate exact Laplace K=1 and K=2 scores** from the closed form and the Fourier inversion respectively, with Monte-Carlo checks.
3. **Formulate the simplest architectural experiment** that can falsify Hypothesis 7.1: fix $K$ (say $K = 16$), fix $t$ at a small and an intermediate value, vary $n \in \{64, 256, 1024, 4096\}$, and compare dense vs local-window vs message-passing networks of matched parameter count. Report score MSE against the exact ground truth. The experiment is cheap; its outcome is informative independently of direction.

A negative result would itself be informative — it would suggest that the architectural inductive bias is not the bottleneck for dynamic-object diffusion, pointing instead toward training dynamics or loss design. A positive result would be the first concrete demonstration that the posterior-factor-graph structure has a learning-efficiency payoff, and would justify a larger-scale investigation of message-passing score architectures for sequential diffusion.

---

## Appendix A. Notation summary

| Symbol | Meaning |
|---|---|
| $a_k$ | clean frame, $k = 0, \dots, K-1$ |
| $x_k$ | OU-noised frame at diffusion time $t$ |
| $\beta := e^{-t}$, $\Delta_t := 1 - e^{-2t}$ | OU shrinkage and variance |
| $P_0, P_t$ | clean and noised joint densities |
| $S(x, t) = \nabla_x \log P_t(x)$ | joint score |
| $H_t(x) = -\nabla^2_x \log P_t(x)$ | curvature field |
| $\alpha_k, \beta_k$ | forward and backward messages (chain factor graph) |
| Gaussian AR(1): $\alpha, \sigma_\eta^2$ | AR coefficient and innovation variance |
| Gaussian AR(1): $\Sigma_t = e^{-2t}\Sigma_0 + \Delta_t I$ | noisy covariance |
| Laplace K=1: $v, s^2, \tilde b, \tau, \rho, \nu, c, a, \tau_G^2$ | structural constants, Table 3.1 of `laplace_K1_compendium.pdf` |
| Laplace K=1: $h_t(r), \psi_t(r), \kappa_t(r)$ | residual density, score, curvature |

---

## Appendix B. On language: messages, propagators, and the two Markov properties

Three distinct notions have been called "the propagator" in various parts of the research record, and it is worth disambiguating them one last time.

- **The Gaussian precision propagator** is the matrix operator $x \mapsto -\Sigma_t^{-1}(x - \mu_t)$ that maps observation to score in one affine shot. It exists exactly in the Gaussian case and does not admit a direct non-Gaussian analogue.
- **The Kalman smoother propagator** is the pair of $O(K)$ recursions (5.2)–(5.4) that compute the posterior mean, and hence the score via Tweedie, in linear time. It is the Gaussian-conjugate specialisation of Theorem 4.1.
- **The posterior message propagator** (general Markov case) is the abstract pair of recursions (4.3)–(4.4) on the chain factor graph. It is exact in principle and of finite parametric complexity only when the messages live in a closed family.

The thesis is organised around the third, general notion. The first two are special cases that matter because they are exact, cheap, and provide the ground truth for benchmarking.

Similarly, two distinct Markov properties must be kept separate:
- **Posterior Markovianity**: $p(a \mid x)$ factorises as a chain Gibbs measure. This holds for every Markov prior, at every $t$ (Proposition 3.1).
- **Joint-density Markovianity**: $P_t(x)$ factorises as a product of independent site and bond densities. This holds in the Gaussian case, and fails at $K \geq 2$ in the Laplace case.

Every architectural decision in this thesis will rest on the first; the second is, by the K=2 obstruction, a promise that Gaussianity makes and non-Gaussianity breaks.
