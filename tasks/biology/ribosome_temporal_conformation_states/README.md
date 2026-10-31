# Ribosome Temporal Conformation States

> Infer molecular fluorescence-state trajectories and transition probabilities from a complete ensemble of noisy, finite single-molecule donor–acceptor traces.
> Domain: biology | Keywords: fluorescence imaging, molecular dynamics, latent-state inference | Difficulty: Hard

## Background

Single-molecule fluorescence follows molecular motion without averaging all molecules together. Energy transfer between nearby fluorophores changes their relative emission intensities as molecular geometry changes. Repeated observations can reveal switching between long-lived conformational or photophysical states that overlap strongly in their noisy signals.

These experiments support studies of ribosomes, protein folding and other biomolecular machines. Their central difficulty is distinguishing genuine temporal transitions from measurement variation and molecule-to-molecule heterogeneity. Finite observation times and photobleaching truncate trajectories, so state populations and transition probabilities must be inferred jointly rather than estimated by a single intensity threshold.

## Problem Description

Use the complete author-distributed `L1-L9` fluorescence ensemble:590traces,500acquired frames and two detector channels per trace. The supplied author intervals specify the previously selected valid observation region of every trace. They are known observation-support annotations, not conformation labels. Retain every trace and every frame in those intervals; outside them, the output uses an explicit inactive sentinel.

Define the apparent fluorescence ratio at an active frame as

```math
e_{st}=\frac{I_{st,1}}{I_{st,0}+I_{st,1}},
```

where s indexes molecules, t indexes acquired frames, and the two I values are the supplied detector intensities. This apparent ratio is dimensionless. No absolute transfer-efficiency, fluorophore correction factor, structural distance or observation time in seconds is supplied. No clipping of noisy ratios to[0,1] is part of the model.

Infer a **two-state working model**. Each trajectory has an independent initial state and follows the same discrete-time Markov transition matrix A. Its observation distribution is Gaussian:

```math
z_{s,0}\sim\mathrm{Categorical}(\pi),\qquad
P(z_{s,t+1}=j\mid z_{st}=i)=A_{ij},\qquad
e_{st}\mid z_{st}=k\sim\mathcal N(\mu_k,v_k).
```

Here z is a latent state0or1, pi is the shared initial-state probability vector, A contains per-observed-frame transition probabilities, mu gives the two emission means, and v gives their positive emission variances. Each row of A and each probability vector sums toone. A sequence boundary resets the initial distribution; never create a transition between the end of one molecule and the start of another.

The Gaussian states describe this selected model's temporal fluorescence behavior. They are not asserted to be uniquely identified structural conformations. In this acquisition, a three-state Gaussian model can improve predictive likelihood; static heterogeneity or misspecification may account for that improvement. The declared inverse remains conditional on the specified two-state model, and there is no measured true state path.

Save `output/reconstruction.npz` relative to the task workspace with all six arrays:

| Key | Shape | Dtype | Meaning |
|---|---|---|---|
| reconstruction | (1,590,500) | int64 | Latent model-state paths; batch,molecule,acquired frame. Use0or1 inside the supplied half-open intervals and exactly−1 outside. |
| state_mean | (1,2) | float64 | Gaussian emission means in apparent-ratio units; batch,state. |
| state_variance | (1,2) | float64 | Positive Gaussian variance estimates in squared apparent-ratio units; batch,state. |
| transition_probability | (1,2,2) | float64 | Dimensionless transition probabilities per observed frame; batch,origin state,destination state; each row sums toone. |
| initial_probability | (1,2) | float64 | Shared initial-state probability; batch,state; sums toone. |
| population_fraction | (1,2) | float64 | Mean posterior state occupancy over all active frames; batch,state; sums toone. This differs from the initial-state distribution. |

Emission parameters and probability arrays must correspond jointly to the submitted paths. A global interchange of the two state labels is physically equivalent: evaluation sorts emission means and applies the **same** permutation to paths, both transition axes and all state-indexed vectors. No separate per-trace relabeling, fitted signal rescaling or frame shift is allowed.

Evaluation uses the conditional posterior-quality, population-consistency and joint decoding criteria below. A single global state permutation is handled consistently by ordering emission means. No reference parameter/path identity or across-trace transition is required.

The method-independent conditional quality target is the joint HMM point-parameter log posterior. With observed apparent ratios e, ordered means mu, precision lambda=1/v, and observation-only prior centers m0 at the linear-interpolated one-third/two-thirds quantiles, maximize log p(e|mu,v,A,pi)+sum log Gamma(lambda;shape2.5,rate0.01)+sum log Normal(mu;m0,variance1/(0.25lambda)). The Dirichlet priors on pi and each A row have concentration one and contribute a constant. Priors occur once for the ensemble. This point-posterior quality is a v2 evaluation criterion, not a relabeling of the historical approximate variational ELBO.

Evaluation accepts a mean log-posterior deficit no greater than1e-8 per active frame relative to the independently attained standard, together with the original observation log-likelihood floor1.0067846856737876 per frame and own-model Viterbi-path agreement at least0.995. Population_fraction must approximate the ordinary forward-backward posterior occupancy of the submitted HMM within max(1e-6,1.1 times the independent standard approximation discrepancy); it is not the Viterbi hard-state fraction. Better posterior objectives or more accurate occupancy approximations pass. All six quantities remain coupled to the same model, independent sequence boundaries and fixed output support, without identity to a local variational parameter/path array.

## Data Description

`data/raw_data.npz` contains every original two-channel acquisition frame plus the author's preexisting observation intervals. No fitted state labels, learned emission levels or transition matrices are supplied. Frames outside the intervals remain in the public acquisition; they are not silently discarded or fitted as a molecular state.

| Key | Shape | Dtype | Description |
|---|---|---|---|
| intensity | (1,590,500,2) | float64 | Original author detector intensities in arbitrary detector units; batch,molecule,frame,channel. Channel1 is the numerator of the declared apparent ratio. |
| interval_start | (1,590) | int64 | Inclusive first valid acquired-frame index; dimensionless frame index; batch,molecule. |
| interval_stop | (1,590) | int64 | Exclusive end of each native valid interval; dimensionless frame index; batch,molecule. |

**Acquisition metadata**

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Author acquisition label | L1-L9 | dimensionless | Original single-molecule fluorescence ensemble label |
| Molecules / acquired frames / channels |590/500/2| dimensionless | Complete original ensemble, no trace subset |
| Active observations |64281| frames | Sum of native interval lengths across all590molecules |
| Native interval lengths |6to499| frames | All supplied intervals are used, including short traces |
| Temporal spacing | One observed-frame step | frame | A is a discrete-time transition matrix; a reliable seconds calibration is not supplied |
| Intensity calibration | Original detector units | arbitrary units | No conversion to photon counts or absolute transfer efficiency is assumed |
| State count | Two | dimensionless | Declared conditional working model, not a unique structural-state claim |
| Gaussian emission law | Shared mean and positive variance per state | apparent-ratio units and their square | Approximate observation model; no known instrumental noise variance is supplied |
| Initial-state boundary | Independent reset for every molecule | dimensionless | Shared pi; no inter-molecule transitions |
| Valid-frame convention | Start inclusive,stop exclusive | frame index | Public author observation support; outside-state sentinel−1 |
| Alignment | One joint global state permutation | dimensionless | Canonicalization by emission means; all state-indexed outputs transform together |

## Method Hints

A shared variational Gaussian hidden-Markov model can pool transition evidence across short trajectories. Forward–backward recursions estimate state and adjacent-state responsibilities while respecting independent sequence boundaries. Normal–Gamma emission priors and Dirichlet probability priors give explicit conjugate updates from aggregated soft counts. Apply each shared prior **once**, rather than once per molecule. Use normalized posterior probability estimates for likelihood and joint Viterbi decoding.

For the selected estimator, use state-mean prior centers equal to the one-third and two-thirds quantiles of all active apparent ratios. Conditional on Gaussian precision lambda, the state mean has a normal prior with that center and precision0.25lambda; lambda has a Gamma prior with shape2.5 and rate0.01. Initial-state and transition Dirichlet concentrations areone. Variance is reported as the inverse geometric posterior precision, exp(−E[log lambda]). Posterior mean pi and row-normalized posterior A define decoding; population_fraction is the distinct posterior occupancy. These conventions define the inference model without interpreting an approximate historical objective as exact Bayesian evidence. Compare additional-state and heterogeneity models as diagnostics rather than claiming that two states are uniquely established by the data.

## References

Verma,A.R., Ray,K., Bodick,M., Kinz-Thompson,C.D. and Gonzalez,R.L.Jr.(2024). *Increasing the accuracy of single-molecule data analysis using tMAVEN*. Biophysical Journal. [Published paper](https://doi.org/10.1016/j.bpj.2024.01.022).

The selected numerical method is the published shared Gaussian-HMM family through its original public numerical core, with explicitly qualified posterior updates and coherent original Viterbi decoding. The application's alternative population-initialized decoder is a separately identified comparison, not the same path posterior.
