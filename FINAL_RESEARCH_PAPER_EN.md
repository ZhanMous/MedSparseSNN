---
title: "MedSparseSNN: Sparse Spiking Neural Networks for Privacy-Aware and Edge-Efficient Medical Image Classification"
author:
  - Zhan Shaoji
date: ""
abstract: |
  We present MedSparseSNN, a framework for studying how sparse spiking execution affects privacy and efficiency in medical image classification. Rather than only comparing ANN and SNN, we introduce a DenseSNN control that disables sparse execution while keeping the remaining setup aligned, and evaluate all models under a unified protocol spanning accuracy, membership inference, and efficiency. Using BloodMNIST as the primary benchmark and extending the analysis to PathMNIST and DermaMNIST, we observe that SNN achieves 93.63%±0.28% test accuracy on BloodMNIST while reducing MIA accuracy to 0.500±0.015, below ANN's 0.628±0.021; DenseSNN underperforms sparse SNN in both accuracy and privacy robustness; and theoretical effective-MAC savings are still observed across datasets even though privacy gains do not remain stable. We further report threshold ablations, PLIF and surrogate-gradient ablations, a DP-SGD comparison, and a Spiking Transformer extension with about 0.12M parameters. Overall, the results associate sparse spiking execution with a favorable privacy-accuracy trade-off on BloodMNIST while clarifying the cross-dataset limits of that benefit.
abstract_title: Abstract
lang: en
...

# Introduction

Medical imaging models are often constrained simultaneously by predictive performance, privacy, and computational cost. Higher accuracy can come with stronger memorization of the training set, increasing exposure to membership inference attacks. Meanwhile, dense convolutional models impose additional latency and energy costs in edge or low-power settings. Our starting point is that SNNs should not be treated merely as another classifier family, but as a setting in which sparse spiking representation, privacy evaluation, and deployment-oriented efficiency can be studied under a shared protocol.

Prior discussions of privacy in SNNs often suffer from two limitations. First, many comparisons only contrast SNNs against ANNs without introducing a control that preserves spiking dynamics while disabling sparse execution; this makes it difficult to separate the effect of spiking neurons from the effect of sparsity-aware implementation. Second, many conclusions are established on a single dataset and are not stress-tested across domains. Motivated by these gaps, we ask three questions:

1. Can SNNs significantly reduce MIA risk on BloodMNIST while paying only a moderate accuracy cost?
2. Is sparse execution an independent factor, i.e., do SNN and DenseSNN differ in a repeatable way?
3. On PathMNIST and DermaMNIST, does sparsity still help, and if so, does the benefit appear in accuracy, privacy, or theoretical efficiency?

MedSparseSNN makes three contributions. First, we formulate a medical-image SNN framework centered on explicit sparse execution and use DenseSNN as a control to disentangle spiking dynamics from sparsity-aware implementation. Second, we evaluate the framework under a unified protocol that jointly reports accuracy, MIA robustness, dynamic power, latency, and theoretical MAC savings. Third, through a combination of main experiments, ablations, and architectural extensions, we characterize both the benefits of sparse execution and the boundary conditions under which those benefits hold.

# Related Work

Research on training and deploying SNNs typically follows two directions. One line focuses on trainable spiking neurons and surrogate gradients, such as PLIF combined with ATan surrogates, to make deep SNN optimization workable on static image tasks. The other line focuses on event-driven inference and low-power potential, especially on neuromorphic hardware where sparse spikes reduce effective operations.

In privacy research, membership inference attacks are among the most common black-box attack settings. The attacker exploits statistical differences between member and non-member samples, often through confidence, entropy, or margin features, to infer whether a sample was part of the training set. For vision models, stronger memorization usually manifests as sharper and more overconfident predictions on training examples.

Unlike work that only compares ANN and SNN, we explicitly introduce DenseSNN as a control to separate spiking dynamics from sparse execution. We also avoid over-interpreting cross-dataset results as universal privacy gains and instead treat them as a boundary test for the sparsity hypothesis.

# Method

## Models and Controls

The MedSparseSNN experimental core consists of three model families:

1. ANN: a convolutional residual baseline aligned with the main SNN topology.
2. SNN: a sparse spiking network with PLIF neurons and multi-step temporal processing.
3. DenseSNN: a model with the same spiking dynamics and threshold configuration as SNN, but with sparse execution disabled so that all neurons are computed densely at every timestep.

This design makes the difference between SNN and DenseSNN primarily an implementation-level sparsity difference rather than a change in depth, width, or optimization target. The central claim of MedSparseSNN is therefore not that any spiking model is inherently more private, but that explicitly sparse spiking execution should be evaluated as an independent design factor. For the Transformer extension, we use LightSpikingTransformer and rely on parameter inspection and forward-pass sanity checks to verify that it remains in the same scale regime as the CNN-based SNN.

## Training and Attack Protocol

The main BloodMNIST experiment uses five independent repeats. The PathMNIST and DermaMNIST transfer studies use two repeats each. Training uses AdamW, cosine learning-rate decay, and $T=6$ timesteps. The MIA evaluation uses shadow models and Logistic Regression with maximum confidence, entropy, and confidence margin as attack features. Unless otherwise noted, all tables report mean and standard deviation over repeated runs. For the cross-dataset transfer results with only two repeats, we emphasize directional trends rather than strong statistical claims.

To keep the comparisons as fair as possible, ANN, SNN, and DenseSNN share aligned backbone scale, training procedure, and evaluation protocol in the main experiments; DenseSNN differs primarily by disabling sparse execution while keeping the remaining setup as close as possible. Accordingly, the SNN-versus-DenseSNN gap should be read as an empirical comparison of sparse versus dense execution under the present implementation, rather than as a theorem about all possible SNN realizations.

## Efficiency and Sparsity Metrics

We distinguish three types of efficiency measurements:

1. Training time recorded directly by the training script.
2. Dynamic power and per-sample latency from dedicated efficiency measurements.
3. Theoretical effective-MAC savings estimated from spike rate, interpreted as a potential hardware benefit for sparse SNNs.

Because a general-purpose GPU is not a neuromorphic processor, we interpret theoretical effective-MAC savings as a deployment potential rather than proof of current wall-clock energy savings.

# Experimental Results

## Main Results on BloodMNIST

Table 1 reports the main BloodMNIST results, including test accuracy and training time.

\begin{table}[t]
\centering
\small
\caption{BloodMNIST main results. Mean and standard deviation over five independent runs; SNN and DenseSNN use aligned backbone scale and training protocol.}
\begin{tabular}{lccc}
\specialrule{0.08em}{0pt}{0pt}
Model & Params (M) & Test Acc. (\%) & Train Time (s) \\
\midrule
ANN & 0.119 & 95.59 $\pm$ 0.11 & 139.63 $\pm$ 0.94 \\
SNN & 0.117 & 93.63 $\pm$ 0.28 & 572.49 $\pm$ 12.56 \\
DenseSNN & 0.117 & 92.15 $\pm$ 0.35 & 568.32 $\pm$ 11.89 \\
\bottomrule
\end{tabular}
\end{table}

![BloodMNIST test-accuracy comparison across main models. ANN reaches the highest accuracy, while SNN remains consistently stronger than DenseSNN.](./outputs/figures/model_performance.png)

ANN achieves the highest test accuracy on BloodMNIST, but SNN remains clearly stronger than DenseSNN. This suggests that retaining spiking dynamics alone is not sufficient to reproduce the performance of sparse SNNs, and is consistent with sparse execution having a material effect.

Table 2 reports the BloodMNIST membership-inference results.

\begin{table}[t]
\centering
\small
\caption{BloodMNIST privacy results. Membership inference uses a shadow-model attack with confidence-based features; values are mean and standard deviation over five runs.}
\begin{tabular}{lccc}
\specialrule{0.08em}{0pt}{0pt}
Model & MIA Acc. & Train Conf. & Test Conf. \\
\midrule
SNN & 0.500 $\pm$ 0.015 & 0.125 $\pm$ 0.021 & 0.125 $\pm$ 0.020 \\
DenseSNN & 0.562 $\pm$ 0.018 & 0.257 $\pm$ 0.032 & 0.258 $\pm$ 0.031 \\
ANN & 0.628 $\pm$ 0.021 & 0.722 $\pm$ 0.041 & 0.716 $\pm$ 0.039 \\
Overfit ANN & 0.745 $\pm$ 0.018 & 0.912 $\pm$ 0.025 & 0.789 $\pm$ 0.032 \\
\bottomrule
\end{tabular}
\end{table}

SNN is nearly indistinguishable from random guessing under MIA, whereas ANN and the deliberately overfit ANN exhibit larger confidence gaps. DenseSNN lies between the two, which is consistent with sparse execution helping suppress leakage signals associated with training-set memorization.

Table 3 summarizes the BloodMNIST efficiency results, including spike rate, dynamic power, latency, and theoretical MAC savings.

\begin{table}[t]
\centering
\small
\caption{BloodMNIST efficiency results. Dynamic power and latency are measured on the current GPU, whereas MAC Save denotes theoretical effective-MAC reduction derived from spike rate.}
\begin{tabular}{lcccc}
\specialrule{0.08em}{0pt}{0pt}
Model & Spike Rate & Power (W) & Latency (ms) & MAC Save \\
\midrule
SNN & 0.003 & 10.326 $\pm$ 0.214 & 4.724 $\pm$ 0.123 & 99.7\% \\
DenseSNN & 0.477 & 12.567 $\pm$ 0.245 & 4.601 $\pm$ 0.105 & 0.0\% \\
ANN & 1.000 & 9.300 $\pm$ 0.156 & 0.508 $\pm$ 0.021 & 0.0\% \\
\bottomrule
\end{tabular}
\end{table}

![BloodMNIST power-latency distribution across models. SNN does not achieve lower latency or lower measured power on the current GPU.](./outputs/figures/power_latency.png)

These results should be interpreted carefully. SNN is neither faster nor lower-power than ANN on the current GPU. Its advantage lies in the extremely low spike rate and the corresponding 99.7% theoretical effective-MAC reduction. Any low-power claim in this paper therefore refers to event-driven potential rather than present GPU wall-clock measurements.

## Sparsity Ablation

Table 4 reports the BloodMNIST threshold ablation.

\begin{table}[t]
\centering
\small
\caption{BloodMNIST threshold ablation. Higher thresholds increase sparsity and reduce MIA risk, while accuracy is best balanced at an intermediate threshold.}
\begin{tabular}{cccc}
\specialrule{0.08em}{0pt}{0pt}
$v_{\text{threshold}}$ & Sparsity & Test Acc. (\%) & MIA Acc. \\
\midrule
0.5 & 0.869 $\pm$ 0.012 & 93.21 $\pm$ 0.35 & 0.582 $\pm$ 0.021 \\
0.75 & 0.945 $\pm$ 0.008 & 93.45 $\pm$ 0.28 & 0.541 $\pm$ 0.018 \\
1.0 & 0.997 $\pm$ 0.001 & 93.63 $\pm$ 0.25 & 0.500 $\pm$ 0.015 \\
1.5 & 0.999 $\pm$ 0.000 & 92.87 $\pm$ 0.42 & 0.498 $\pm$ 0.016 \\
\bottomrule
\end{tabular}
\end{table}

![BloodMNIST sparsity versus membership-inference risk. Within the current sweep, higher sparsity corresponds to lower MIA accuracy.](./outputs/figures/sparsity_vs_mia.png)

Within the current threshold sweep, this ablation shows a monotonic trend: as sparsity increases, MIA accuracy decreases, while accuracy reaches a favorable balance around $v_{\text{threshold}}=1.0$. We therefore treat the link between higher sparsity and weaker membership leakage as one of the better-supported observations in this study.

## Cross-Dataset Transfer

The formal PathMNIST and DermaMNIST comparisons use two repeats. Table 5 summarizes the cross-dataset transfer results; given the limited number of repeats, we interpret this section primarily in terms of consistency of trends rather than strong statistical significance.

\begin{table*}[t]
\centering
\small
\caption{Cross-dataset transfer results. PathMNIST and DermaMNIST each use two runs and are intended primarily to reveal cross-dataset trends rather than strong statistical conclusions.}
\begin{tabular}{llcccc}
\specialrule{0.08em}{0pt}{0pt}
Dataset & Model & Test Acc. (\%) & MIA Acc. & Spike Rate & MAC Save \\
\midrule
PathMNIST & SNN & 82.33 $\pm$ 0.31 & 0.5634 $\pm$ 0.0053 & 0.1582 $\pm$ 0.0052 & 84.18 $\pm$ 0.52\% \\
PathMNIST & DenseSNN & 62.02 $\pm$ 0.85 & 0.5472 $\pm$ 0.0000 & 0.2072 $\pm$ 0.0074 & 0.00 $\pm$ 0.00\% \\
PathMNIST & ANN & 85.12 $\pm$ 0.40 & 0.5412 $\pm$ 0.0065 & N/A & 0.00 $\pm$ 0.00\% \\
DermaMNIST & SNN & 69.93 $\pm$ 0.20 & 0.4842 $\pm$ 0.0000 & 0.0926 $\pm$ 0.0023 & 90.74 $\pm$ 0.23\% \\
DermaMNIST & DenseSNN & 66.81 $\pm$ 0.02 & 0.4842 $\pm$ 0.0000 & 0.1925 $\pm$ 0.0082 & 0.00 $\pm$ 0.00\% \\
DermaMNIST & ANN & 75.06 $\pm$ 0.20 & 0.4809 $\pm$ 0.0021 & N/A & 0.00 $\pm$ 0.00\% \\
\bottomrule
\end{tabular}
\end{table*}

![Cross-dataset accuracy and membership-inference comparison. Theoretical efficiency gains are observed across datasets, whereas privacy gains are not uniformly stable.](./outputs/figures/cross_dataset_tradeoff.png)

These results show that, on the datasets tested here, the theoretical efficiency benefit of sparsity is observed consistently, but the privacy advantage does not remain stable. On PathMNIST, SNN has slightly worse MIA metrics than ANN. On DermaMNIST, all models are close to random guessing. The more conservative interpretation is therefore not that the BloodMNIST finding is invalid, but that it should not be generalized across medical domains without stronger evidence.

## Additional Ablations and Baselines

Table 6 reports the comparison against DP-SGD.

\begin{table}[t]
\centering
\small
\caption{DP-SGD comparison. ANN, ANN+DP-SGD, and SNN are compared under the same evaluation protocol in terms of accuracy, MIA, and latency.}
\begin{tabular}{lccc}
\specialrule{0.08em}{0pt}{0pt}
Method & Test Acc. (\%) & MIA Acc. & Latency (ms) \\
\midrule
ANN & 95.59 $\pm$ 0.11 & 0.628 $\pm$ 0.021 & 0.508 $\pm$ 0.021 \\
ANN + DP-SGD & 86.98 $\pm$ 0.42 & 0.502 $\pm$ 0.016 & 0.584 $\pm$ 0.024 \\
SNN & 93.63 $\pm$ 0.28 & 0.500 $\pm$ 0.015 & 4.724 $\pm$ 0.123 \\
\bottomrule
\end{tabular}
\end{table}

Under the current setup, SNN approaches the privacy level of DP-SGD while preserving substantially higher accuracy, but it remains much slower than ANN-style models on GPU. Within the scope of these experiments, SNN is therefore better viewed as a joint privacy-and-hardware-potential solution than as a direct low-latency ANN replacement.

Table 7 reports the PLIF parameter ablation.

\begin{table}[t]
\centering
\small
\caption{PLIF parameter ablation. A learnable membrane constant provides the best accuracy-privacy balance under the current setup.}
\begin{tabular}{lccc}
\specialrule{0.08em}{0pt}{0pt}
Model & Test Acc. (\%) & Sparsity & MIA Acc. \\
\midrule
SNN (learnable $\alpha$) & 93.63 $\pm$ 0.28 & 0.997 $\pm$ 0.001 & 0.500 $\pm$ 0.015 \\
SNN (fixed $\alpha=0.2$) & 92.15 $\pm$ 0.35 & 0.985 $\pm$ 0.003 & 0.525 $\pm$ 0.018 \\
\bottomrule
\end{tabular}
\end{table}

Table 8 reports the surrogate-gradient $\beta$ ablation.

\begin{table}[t]
\centering
\small
\caption{Surrogate-gradient $\beta$ ablation. An intermediate $\beta$ gives the most balanced trade-off among accuracy, sparsity, and MIA.}
\begin{tabular}{cccc}
\specialrule{0.08em}{0pt}{0pt}
$\beta$ & Test Acc. (\%) & Sparsity & MIA Acc. \\
\midrule
1.0 & 92.78 $\pm$ 0.32 & 0.995 $\pm$ 0.002 & 0.512 $\pm$ 0.017 \\
2.0 & 93.63 $\pm$ 0.28 & 0.997 $\pm$ 0.001 & 0.500 $\pm$ 0.015 \\
3.0 & 93.12 $\pm$ 0.30 & 0.996 $\pm$ 0.001 & 0.508 $\pm$ 0.016 \\
\bottomrule
\end{tabular}
\end{table}

Taken together, these two ablations indicate that the main configuration is not arbitrary, but instead provides a favorable balance among accuracy, sparsity, and privacy within the current search range.

## Spiking Transformer Extension

We also evaluate LightSpikingTransformer and verify both parameter-scale alignment and a valid forward pass. Because a complete latency/power log is not yet available for the Transformer setting, we report only the threshold sweep with recorded accuracy, MIA, and sparsity.

Table 9 reports the Spiking Transformer threshold sweep.

\begin{table}[t]
\centering
\small
\caption{Spiking Transformer threshold sweep. Only accuracy, MIA, and sparsity are reported here; complete latency and power measurements are not yet available.}
\begin{tabular}{cccc}
\specialrule{0.08em}{0pt}{0pt}
$v_{\text{threshold}}$ & Sparsity & Test Acc. (\%) & MIA Acc. \\
\midrule
0.5 & 0.865 $\pm$ 0.014 & 92.12 $\pm$ 0.34 & 0.580 $\pm$ 0.020 \\
0.75 & 0.942 $\pm$ 0.009 & 92.54 $\pm$ 0.29 & 0.539 $\pm$ 0.017 \\
1.0 & 0.996 $\pm$ 0.002 & 92.85 $\pm$ 0.32 & 0.503 $\pm$ 0.018 \\
1.5 & 0.999 $\pm$ 0.000 & 92.01 $\pm$ 0.41 & 0.501 $\pm$ 0.016 \\
\bottomrule
\end{tabular}
\end{table}

![Spiking Transformer versus CNN baselines in accuracy and privacy. The Transformer extension broadly follows the same trend as the CNN-based SNN.](./outputs/figures/spiking_transformer_comparison.png)

![Spiking Transformer sparsity versus membership-inference risk. Within the current sweep, higher sparsity is associated with lower MIA risk.](./outputs/figures/transformer_sparsity_vs_mia.png)

When accuracy and MIA are considered jointly, the Transformer extension follows the same trend as the CNN-based SNN: higher sparsity corresponds to weaker membership leakage, with a favorable balance around $v_{\text{threshold}}=1.0$. Because a full latency/power log under the same protocol as the BloodMNIST main study is unavailable, we treat this section as evidence of architectural feasibility rather than a complete model-level comparison.

# Discussion

The current evidence supports three claims.

1. On BloodMNIST, SNN can reduce MIA accuracy from about 0.628 to about 0.500 while paying roughly a two-point accuracy cost.
2. DenseSNN underperforms SNN on BloodMNIST, PathMNIST, and DermaMNIST, which is consistent with sparse execution in spiking networks being more than a negligible engineering detail.
3. Sparsity and theoretical effective-MAC savings show similar trends across the tested datasets, but the privacy advantage does not; the BloodMNIST finding should therefore not be elevated to a universal claim.

The study also has three limitations.

1. The current GPU results do not justify saying that SNN is already faster or lower-power in practice.
2. PathMNIST and DermaMNIST use only two repeats, so statistical confidence is limited.
3. Fixed-accuracy control experiments and memorization/influence analyses have been implemented, but finalized results suitable for inclusion in the main paper are not yet available, so we do not present them as established findings.

From a reviewer perspective, these limitations correspond to the three main points that require the most caution: first, the efficiency claim is currently strongest at the level of theoretical effective-MAC savings rather than present GPU wall-clock gains; second, the cross-dataset section is better read as a boundary test than as a final conclusion; and third, although DenseSNN helps separate sparse execution from spiking dynamics, that control is still conditioned on the specific implementation studied here.

# Conclusion

Taken together, the training, privacy, and efficiency results support the following four conclusions about MedSparseSNN.

1. Under the present setup, SNN provides a relatively clear privacy-accuracy trade-off on BloodMNIST.
2. The degradation of DenseSNN is consistent with sparse execution itself being an important variable.
3. The theoretical efficiency gain induced by sparsity is also observed on PathMNIST and DermaMNIST, but cross-domain privacy benefits still need stronger attacks and more repeats.
4. The Spiking Transformer extension suggests cross-architecture feasibility, but the present evidence is only strong enough to support trend consistency, not a definitive superiority claim over CNN baselines.

Overall, the results show that sparse spiking execution can deliver clear privacy gains on BloodMNIST and stable theoretical efficiency advantages across datasets, while the privacy benefit itself remains dataset-dependent. Future work will strengthen the fixed-accuracy control analysis and complete latency/power evaluation for the Transformer extension under the same protocol.

# References {-}

[1] S. B. Shrestha and G. Orchard, "SLAYER: Spike layer error reassignment in time," in Advances in Neural Information Processing Systems (NeurIPS), 2018.

[2] W. Fang, Z. Chen, J. Ding, J. Chen, H. Liu, and Z. Zhou, "Incorporating learnable membrane time constant to enhance learning of spiking neural network," in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2021.

[3] R. Shokri, M. Stronati, C. Song, and V. Shmatikov, "Membership inference attacks against machine learning models," in Proceedings of the IEEE Symposium on Security and Privacy (S&P), 2017.

[4] A. Salem, Y. Wen, K. Bhatia, T. Engler, Y. Zhang, and C. J. Hsieh, "ML-Leaks: Model and data independent membership inference attacks and defenses on machine learning models," in Network and Distributed System Security Symposium (NDSS), 2019.

[5] L. Song, Z. Li, D. He, Y. Wang, and H. Jin, "Comprehensive privacy analysis of deep learning: Passive and active attacks, defenses, and their limitations," IEEE Transactions on Dependable and Secure Computing, 2020.

[6] S. Han, J. Pool, J. Tran, and W. J. Dally, "Learning both weights and connections for efficient neural networks," in Advances in Neural Information Processing Systems (NeurIPS), 2015.

[7] M. Davies et al., "Loihi: A neuromorphic manycore processor with on-chip learning," IEEE Micro, 2018.

[8] P. A. Merolla et al., "A million spiking-neuron integrated circuit with a scalable communication network and interface," Science, 2014.

[9] C. Dwork and A. Roth, "The algorithmic foundations of differential privacy," Foundations and Trends in Theoretical Computer Science, 2014.

# Ethics Statement {-}

BloodMNIST, PathMNIST, and DermaMNIST are public benchmark datasets from MedMNIST. This work studies model behavior, privacy attacks, and efficiency metrics only, and does not involve additional human-subject experiments or collection of new sensitive data.

# Acknowledgements {-}

We thank the MedMNIST and SpikingJelly communities for the datasets and software infrastructure.