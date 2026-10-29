# Nascent Transcription Promoter Dynamics

> Recover latent promoter activity and switching parameters from real fluorescence traces whose elongation memory obscures the underlying transcription events.
> Domain: biology | Keywords: fluorescence inference, stochastic inverse problems, transcription dynamics | Difficulty: Hard

## Background

Nascent RNA fluorescent labeling reports transcription while polymerases move along a gene. The fluorescence persists after initiation, so a bright focus combines contributions from many preceding promoter states. Time-resolved microscopy therefore measures a delayed, temporally mixed version of promoter activity.

Recovering promoter activity helps investigate how cells regulate gene expression. Distinguishing frequent weak activation from rare strong activation is difficult when transcription events overlap and measurements fluctuate. A latent stochastic process and an explicit elongation model are needed to interpret these signals.

## Problem Description

Use all63 supplied author-selected real Drosophila ush fluorescence traces. A two-state promoter has OFF state0 and ON state1. Its column-stochastic transition matrix is A, with A(i,j) the probability of next state i given current state j. State-dependent fluorescence loading amplitudes are v0 and v1, with v1>v0. The observation model is

```math
y_t=\sum_{j=0}^{W-1}w_j v_{z_{t-j}}+\epsilon_t,\qquad \epsilon_t\sim N(0,\sigma^2).
```

y is measured fluorescence, z the unknown promoter state, w the known elongation weights, sigma the unknown measurement standard deviation, t the frame index and W the memory length. Before the first observed frame all promoter history states equal OFF; their fluorescence loading contributions remain in the observation model. The known response weights and memory length are specified in Acquisition metadata, with weights ordered most recent to oldest. This is a conditional finite-memory model; it does not provide independent knowledge of true promoter states.

Save `output/reconstruction.npz` with the following real float32/float64 finite arrays. No label permutation or intensity rescaling is applied in evaluation.

| Output key | Shape | Meaning |
|---|---|---|
| posterior | (1,63,86) | Dimensionless posterior ON probability, axes batch,trace,time; entries beyond each trace length are exactly zero |
| transition | (1,2,2) | Dimensionless A, axes batch,next state,current state; columns sum to one |
| emission | (1,2) | OFF/ON loading amplitudes in fluorescence units per weight; nonnegative and ON greater than OFF |
| noise | (1,1) | Positive measurement standard deviation in fluorescence units |
| initial | (1,2) | Dimensionless initial OFF/ON probabilities summing to one |

Probabilities allow1e-6 numerical roundoff beyond the unit interval. Evaluation uses valid-frame posterior RMS error, absolute transition/initial RMS errors, relative emission-vector L2 error, and relative noise error. All quantities must pass. These compare a validated reference interpretation, not experimentally observed hidden states. Memory truncation and model assumptions limit identifiability.

## Data Description

This is the author example's regional selection, rather than an entire embryo. Its preprocessing sorts original traces by observed mean fluorescence, removes the twenty weakest, selects region0, and then retains all valid frames of the remaining63 traces. Selection uses observation metadata, not inferred promoter states.

| Key | Shape | Dtype | Description |
|---|---|---|---|
| fluorescence | (1,63,86) | float64 | Author-processed measured fluorescence units, batch,trace,time; zero padding beyond each length |
| lengths | (1,63) | int64 | Valid recorded frame counts, batch,trace; padding is not an observation |

### Acquisition metadata

| Parameter | Value | Unit | Meaning |
|---|---|---|---|
| Sample | Drosophila melanogaster, ush gene | identity | Experimental nascent RNA fluorescence example |
| Modality | MS2-MCP time-lapse microscopy | type | Author-processed single-nucleus traces |
| Frame interval | 20 | s | Observation spacing |
| Elongation memory | 19 | frame | Delayed contributions in the acquisition model |
| MS2 labeling traversal | 30 | s | Author-example modeling value, not independently calibrated in this export |
| Response weights | 1/3,11/12,then seventeen ones | dimensionless | Most recent to oldest contribution |
| Prehistory | OFF state at every unavailable earlier frame | convention | Full OFF loading remains included |
| Intensity calibration | Relative author fluorescence export | fluorescence unit | No molecule-count calibration supplied |
| Noise | Independent Gaussian residual | fluorescence unit | Unknown standard deviation inferred jointly |
| Temporal boundary | Separate finite traces | convention | No wraparound or concatenation |

**Evaluation certification:** The current numerical evaluator is not yet certified as a method-independent public benchmark; standard-method agreement must not be interpreted as independent physical accuracy.

## Method Hints

Use a compound-state hidden Markov model to retain the preceding promoter history. Forward/backward message passing yields posterior activity and transition counts. Alternate these probabilities with Gaussian emission and noise updates. A retained beam can make the long-memory state space tractable, but its approximation must remain explicit. The likelihood and emission normal equations must use the same prehistory convention; subtracting early OFF contributions in only one stage creates an inconsistent inverse model.

## References

Bowles et al., *Scalable inference of transcriptional kinetic parameters from MS2 time series data*, Bioinformatics, DOI [10.1093/bioinformatics/btab765](https://doi.org/10.1093/bioinformatics/btab765). Experimental data example: Hoppe et al., Developmental Cell2020, as identified in the original author data distribution.
