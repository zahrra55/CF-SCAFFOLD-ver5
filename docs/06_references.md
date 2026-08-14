# CF-SCAFFOLD Thesis — Complete Reference List (with direct links)

**Researcher:** Zahraa Ibrahim Khaleel  **Supervisor:** Prof. Dr. Mohammed Sahib Mehdi
**Two contributions:** (1) Fractal descriptor C = Ft/Fb for brain-tumor MRI classification;
(2) Class-presence-guided restricted-softmax for stabilizing federated learning under class-skew.

> **IMPORTANT — verify before final citation.** Links marked ✅ were confirmed in-session.
> Links marked ⚠️ are the official DOI/arXiv/venue pages for works cited from research
> reports; open each and confirm authors, year, page numbers, and the exact reported number
> before putting it in your thesis. This is standard scholarly practice, not optional.

---

## A. Dataset (your primary data)

1. **Cheng, J., Huang, W., Cao, S., Yang, R., Yang, W., Yun, Z., Wang, Z., Feng, Q. (2015).** "Enhanced Performance of Brain Tumor Classification via Tumor Region Augmentation and Partition." *PLOS ONE* 10(10): e0140381. — **This is your benchmark (91.28%) AND the dataset's source paper.** ✅
   https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0140381

2. **Cheng, J., et al. (2016).** "Retrieval of Brain Tumors by Adaptive Spatial Pooling and Fisher Vector Representation." *PLOS ONE* 11(6): e0157112. — companion retrieval paper. ✅
   https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0157112

3. **Figshare Brain Tumor Dataset** (3064 T1-CE slices, 233 patients), DOI/CC BY 4.0. ✅
   https://doi.org/10.6084/m9.figshare.1512427
   Kaggle mirror you used: https://www.kaggle.com/datasets/ashkhagan/figshare-brain-tumor-dataset
   Original MATLAB code (Cheng): https://github.com/chengjun583/brainTumorRetrieval

---

## B. Contribution 1 — classification comparison papers

### B.1 Hand-crafted / classical-feature baselines (closest to your method)

4. **Gumaei, A., et al. (2019).** "A Hybrid Feature Extraction Method With Regularized Extreme Learning Machine for Brain Tumor Classification." *IEEE Access* 7:36266–36273. — **94.23%, your strongest classical competitor.** ⚠️
   https://doi.org/10.1109/ACCESS.2019.2904145

5. **Ismael, M. R., & Abdel-Qader, I. (2018).** "Brain Tumor Classification via Statistical Features and Back-Propagation Neural Network." *IEEE EIT 2018*, pp. 252–257. — 91.9% (DWT + Gabor). ⚠️
   https://doi.org/10.1109/EIT.2018.8500308

### B.2 Deep-learning baselines (report protocol carefully)

6. **Deepak, S., & Ameer, P. M. (2019).** "Brain tumor classification using deep CNN features via transfer learning." *Computers in Biology and Medicine* 111:103345. — **98%, patient-level 5-fold (a fair-protocol deep baseline).** ⚠️
   https://doi.org/10.1016/j.compbiomed.2019.103345

7. **Swati, Z. N. K., et al. (2019).** "Brain tumor classification for MR images using transfer learning and fine-tuning." *Computerized Medical Imaging and Graphics* 75:34–46. — 94.82%, patient-respecting folds. ⚠️
   https://doi.org/10.1016/j.compmedimag.2019.05.001

8. **Afshar, P., Plataniotis, K. N., & Mohammadi, A. (2019).** "Capsule Networks for Brain Tumor Classification Based on MRI Images and Coarse Tumor Boundaries." *ICASSP 2019*, pp. 1368–1372. — CapsNet 86.56%; with tumor-boundary input **90.98%** (an earlier draft recorded 90.89%, which is wrong). ✅
   https://doi.org/10.1109/ICASSP.2019.8683759

9. **(Optional, attention-CNN comparator)** "Brain Tumor Classification Based on Attention Guided Deep Learning Model." *Int. J. Computational Intelligence Systems* (2022). ✅
   https://link.springer.com/article/10.1007/s44196-022-00090-9

10. **(Optional, transfer-learning framework)** "A Deep Learning-Based Framework for Automatic Brain Tumors Classification Using Transfer Learning." *Circuits, Systems, and Signal Processing* (2019). ✅
    https://link.springer.com/article/10.1007/s00034-019-01246-3

### B.3 Fractal-dimension prior art (to scope your Ft/Fb novelty)

11. **Iftekharuddin, K. M., et al. (2003).** "Fractal analysis of tumor in brain MR images." *Machine Vision and Applications* 13:352–362. — foundational box-counting FD for brain tumors. ⚠️
    https://doi.org/10.1007/s00138-002-0087-9

12. **Fractal-dimension biomarker paper (2023).** "Fractal dimension analysis of … glioma grading" (tumor + non-tumor FD), *Frontiers in Physiology* (2023). — 93% acc / 98% AUC; closest precedent to using tumor AND brain FD together. ⚠️
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10390093/

13. **Sánchez, J., & Martín-Landrove, M. (2022).** "Morphological and Fractal Properties of Brain Tumors." *Frontiers in Physiology* 13, art. 878391. ✅ *(author order corrected: Sánchez is first author, not Martín-Landrove)*
    https://doi.org/10.3389/fphys.2022.878391

### B.4 The patient-level vs slice-level (data-leakage) argument

14. **Yagis, E., et al. (2021).** "Effect of data leakage in brain MRI classification using 2D convolutional neural networks." *Scientific Reports* 11:22544. — **slice-level leakage inflates accuracy by 30–55 pp; your central rigor citation.** ✅
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8604922/  (DOI: https://doi.org/10.1038/s41598-021-01681-w)

---

## C. Contribution 2 — federated learning comparison papers

### C.1 Direct mechanistic ancestors (restricted / calibrated softmax)

15. **Li, X.-C., & Zhan, D.-C. (2021).** "FedRS: Federated Learning with Restricted Softmax for Label Distribution Non-IID Data." *KDD '21*, pp. 995–1005. — **the paper your restricted-softmax is closest to; also defines a Scaffold+RS variant.** ✅
    Official PDF: https://www.lamda.nju.edu.cn/lixc/papers/FedRS-KDD2021-Lixc.pdf
    ACM/DOI: https://doi.org/10.1145/3447548.3467254
    Code (FedRepo): https://github.com/lxcnju/FedRepo

16. **Zhang, J., et al. (2022).** "Federated Learning with Label Distribution Skew via Logits Calibration (FedLC)." *ICML 2022*, PMLR 162. — generalizes FedRS; +9.6 to +17.8 pp gains. ⚠️
    https://proceedings.mlr.press/v162/zhang22p/zhang22p.pdf  (arXiv: https://arxiv.org/abs/2209.00189)

17. **Guo, K., Ding, Y., Liang, J., Wang, Z., He, R., & Tan, T. (2025).** "Exploring Vacant Classes in Label-Skewed Federated Learning." *Proc. AAAI Conf. on Artificial Intelligence* 39:16960–16968. ✅ *(confirmed published at AAAI-25)*
    https://ojs.aaai.org/index.php/AAAI/article/view/33864

### C.2 Foundational FL optimizers (your baselines)

18. **McMahan, B., et al. (2017).** "Communication-Efficient Learning of Deep Networks from Decentralized Data (FedAvg)." *AISTATS 2017*. ⚠️
    https://arxiv.org/abs/1602.05629

19. **Karimireddy, S. P., et al. (2020).** "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning." *ICML 2020*. — **your core FL algorithm.** ⚠️
    https://arxiv.org/abs/1910.06378

20. **Li, T., Sahu, A. K., Zaheer, M., et al. (2020).** "Federated Optimization in Heterogeneous Networks (FedProx)." *MLSys 2020*. — you tested it; adds nothing on top (your honest null). ⚠️
    https://arxiv.org/abs/1812.06127

### C.3 GroupNorm vs BatchNorm under non-IID (justifies your norm choice)

21. **Hsieh, K., et al. (2020).** "The Non-IID Data Quagmire of Decentralized Machine Learning." *ICML 2020*. — GroupNorm beats BatchNorm under non-IID. ⚠️
    https://arxiv.org/abs/1910.00189

22. **Casella, B., Esposito, R., Sciarappa, A., Cavazzoni, C., & Aldinucci, M. (2024).** "Experimenting with Normalization Layers in Federated Learning on non-IID Scenarios." *IEEE Access* 12:47961–47971. ✅ *(published; do not cite the arXiv preprint)*
    https://doi.org/10.1109/ACCESS.2024.3383783

### C.4 Federated brain-tumor / medical MRI (domain anchors)

23. **Janbi, N., et al. (2024).** "FedBrain-Distill: Ensemble knowledge distillation for federated brain tumour classification." arXiv:2409.05359. ⚠️ *(preprint; verify before citing)*
    https://arxiv.org/abs/2409.05359

23b. **Fateh, A., Rezvani, Y., Moayedi, S., Rezvani, S., Fateh, F., Fateh, M., & Abolghasemi, V. (2025).** "BRISC: Annotated Dataset for Brain Tumor Segmentation and Classification." *Scientific Data*. ✅ *(title corrected; earlier draft said "brain recognition and image segmentation classification", which is wrong. The volume/article pair "12, 845" could not be confirmed — cite by DOI.)*
    https://doi.org/10.1038/s41597-026-06753-y — preprint arXiv:2506.14318
    **Independence caveat:** collated from existing public datasets including SARTAJ and Br35H, the same lineage as the Cheng/Figshare aggregate. Not independent of the primary dataset.

### C.5 Personalized FL (context for your leakage-guard discussion)

24. **Li, T., et al. (2021).** "Ditto: Fair and Robust Federated Learning Through Personalization." *ICML 2021*. ⚠️
    https://arxiv.org/abs/2012.04221

25. **Collins, L., et al. (2021).** "Exploiting Shared Representations for Personalized Federated Learning (FedRep)." *ICML 2021*. ⚠️
    https://arxiv.org/abs/2102.07078

26. **Li, X., et al. (2021).** "FedBN: Federated Learning on Non-IID Features via Local Batch Normalization." *ICLR 2021*. ⚠️
    https://arxiv.org/abs/2102.07623

27. **Wu, S., et al. (2022).** "Motley: Benchmarking Heterogeneity and Personalization in Federated Learning." arXiv:2206.09262. — **the support/query protocol whose leakage you exposed.** ⚠️
    https://arxiv.org/abs/2206.09262

28. **Chen, D., et al. (2022).** "pFL-Bench: A Comprehensive Benchmark for Personalized Federated Learning." *NeurIPS 2022 Datasets & Benchmarks*. ⚠️
    https://openreview.net/forum?id=2ptbv_JjYKA

---

## D. Methodological / statistical references

29. **Colas, C., Sigaud, O., & Oudeyer, P.-Y. (2018).** "How Many Random Seeds? Statistical Power Analysis in Deep RL Experiments." arXiv:1806.08295. — justifies your ≥20-seed protocol. ⚠️
    https://arxiv.org/abs/1806.08295

30. **scikit-learn** — Pedregosa et al. (2011), *JMLR* 12:2825–2830 (RandomForest, GroupKFold, metrics). ⚠️
    https://jmlr.org/papers/v12/pedregosa11a.html

31. **PyTorch** — Paszke et al. (2019), *NeurIPS 2019* (your FL implementation framework). ⚠️
    https://arxiv.org/abs/1912.01703

32. **scikit-image** — van der Walt et al. (2014), *PeerJ* 2:e453 (GLCM, regionprops, lacunarity). ⚠️
    https://doi.org/10.7717/peerj.453

---

## How to use this list
- The **must-cite core** for your two contributions: #1 (benchmark), #4 & #6 (classification comparators), #11–13 (fractal novelty), #14 (leakage argument), #15–17 (FL restricted-softmax family), #19 (SCAFFOLD), #21 (GroupNorm), #23 (medical-FL domain).
- Open every ⚠️ link and confirm the number you quote matches the paper's own table — research-report figures occasionally come from secondary summaries.
- Total: 32 entries spanning dataset, classification, federated learning, and methodology.
