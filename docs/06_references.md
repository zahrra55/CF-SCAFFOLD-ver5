# CF-SCAFFOLD: Complete Reference List

> **The organised reference catalogue is now [`references/README.md`](../references/README.md)**: all 189 references with links, purpose, where each is cited in Chapter 2, where it is used in the experiments, and a reading level. This file is kept as the verification record of the first 166 references (September and October 2026).

**Researcher:** Zahraa Ibrahim Khaleel  |  **Supervisor:** Prof. Dr. Mohammed Sahib Mehdi  |  Al-Nahrain University, College of Science, Department of Computer Science

This list replaces the earlier 32-item draft. It contains **166 references**, ordered by the stage of the pipeline where each one is used, with a direct link, the access route, the exact place and purpose of use, and how the record was verified.

## How every entry was verified (September 2026)

- **DOIs (122):** each DOI was looked up in Crossref (or DataCite for the two datasets); the registry title had to contain the expected title and the year had to match within one year (online-first vs issue year). The DOI was then resolved through `doi.org` to confirm it redirects to a live landing page. Open-access status and a free full-text link come from OpenAlex.
- **arXiv items (23):** checked through the arXiv API. Where a peer-reviewed version exists, the venue was confirmed on OpenReview (ICLR) or in the official NeurIPS/MLSys proceedings listing, and the citation points to that version. Eight items have no peer-reviewed version and are cited as preprints.
- **Proceedings/journal pages without DOI (16):** the official PMLR, NeurIPS, MLSys and JMLR pages were fetched; authors, volume and page range were read from the page's own citation meta-tags.
- **Books (4 ISBN + 2 DOI):** ISBN matched in Open Library; publisher and place taken from that record.
- **Holm (1979):** no DOI exists; confirmed through JSTOR stable id 4615733 and MR538597.

Result: **166 / 166 verified.** (150 in September 2026; 16 added in October 2026 for the shortcut and segmentation-robustness analyses.) Corrections made during verification are listed at the end of this file and in `07_verification_log.md`.

> Links resolve today. `doi.org` links are permanent by design; for subscription items use your university library proxy. The machine-readable results of the check are in `references_verification.csv`.

## Contents

- [0. Surveys, clinical context and motivation](#0-surveys-clinical-context-and-motivation)
- [A. Datasets and evaluation protocol](#a-datasets-and-evaluation-protocol)
- [B. Pre-processing, morphology and software](#b-pre-processing-morphology-and-software)
- [C. Fractal and texture descriptors (Pillar I)](#c-fractal-and-texture-descriptors-pillar-i)
- [D. Segmentation: automatic delineation attempt (Pillar I, Stages 11-12)](#d-segmentation-automatic-delineation-attempt-pillar-i-stages-11-12)
- [E. Classification, comparators and metrics (Pillar I)](#e-classification-comparators-and-metrics-pillar-i)
- [F. Federated learning algorithms (Pillar II)](#f-federated-learning-algorithms-pillar-ii)
- [G. Normalisation, optimisation and class imbalance (Pillar II)](#g-normalisation-optimisation-and-class-imbalance-pillar-ii)
- [H. Transfer learning and the frozen backbone (Pillar II, Stages 7, 9, 9B)](#h-transfer-learning-and-the-frozen-backbone-pillar-ii-stages-7-9-9b)
- [I. Statistical methodology (both pillars)](#i-statistical-methodology-both-pillars)
- [J. Dataset shortcuts and robustness to segmentation (Pillar I, Stages 14-16)](#j-dataset-shortcuts-and-robustness-to-segmentation-pillar-i-stages-14-16)
- [Using this list for the literature review](#using-this-list-for-the-literature-review)
- [Corrections found during verification](#corrections-found-during-verification)


## 0. Surveys, clinical context and motivation

*Literature-review backbone for both pillars.*

**[1]** P. Kairouz et al., "Advances and Open Problems in Federated Learning," *Foundations and Trends in Machine Learning*, vol. 14, no. 1–2, pp. 1–210, 2021, doi: 10.1561/2200000083.  
- **Key:** `kairouz2021`  ·  **Link:** https://doi.org/10.1561/2200000083
- **Access:** Open access (green); free copy: https://arxiv.org/pdf/1912.04977
- **Where and how it is used:** Ch.2 FL background; defines statistical heterogeneity (non-IID) as an open problem that motivates Pillar II.
- **Verified:** Crossref record matched (title, year); doi.org resolver HTTP 302

**[2]** T. Li, A. K. Sahu, A. Talwalkar, and V. Smith, "Federated Learning: Challenges, Methods, and Future Directions," *IEEE Signal Processing Magazine*, vol. 37, no. 3, pp. 50–60, 2020, doi: 10.1109/MSP.2020.2975749.  
- **Key:** `li2020challenges`  ·  **Link:** https://doi.org/10.1109/MSP.2020.2975749
- **Access:** Open access (green); free copy: https://arxiv.org/pdf/1908.07873
- **Where and how it is used:** Ch.2 FL overview; taxonomy of systems vs statistical heterogeneity.
- **Verified:** Crossref record matched (title, year); doi.org resolver HTTP 302

**[3]** H. Zhu, J. Xu, S. Liu, and Y. Jin, "Federated learning on non-IID data: A survey," *Neurocomputing*, vol. 465, pp. 371–390, 2021, doi: 10.1016/j.neucom.2021.07.098.  
- **Key:** `zhu2021noniid`  ·  **Link:** https://doi.org/10.1016/j.neucom.2021.07.098
- **Access:** Subscription (institutional access)
- **Where and how it is used:** Ch.2 taxonomy of label-, feature- and quantity-skew; positions class-absence as label-distribution skew.
- **Verified:** Crossref record matched (title, year); doi.org resolver HTTP 302

**[4]** M. Ye, X. Fang, B. Du, P. C. Yuen, and D. Tao, "Heterogeneous Federated Learning: State-of-the-art and Research Challenges," *ACM Computing Surveys*, vol. 56, no. 3, pp. 1–44, 2024, doi: 10.1145/3625558.  
- **Key:** `ye2023hfl`  ·  **Link:** https://doi.org/10.1145/3625558
- **Access:** Open access (bronze); free copy: https://dl.acm.org/doi/pdf/10.1145/3625558
- **Where and how it is used:** Ch.2 survey of data/model/server-level heterogeneity corrections; places restricted-softmax among client-side output-layer fixes.
- **Verified:** Crossref record matched (title, year); doi.org resolver HTTP 302

**[5]** H. Guan, P.-T. Yap, A. Bozoki, and M. Liu, "Federated learning for medical image analysis: A survey," *Pattern Recognition*, vol. 151, art. no. 110424, 2024, doi: 10.1016/j.patcog.2024.110424.  
- **Key:** `guan2024fedmed`  ·  **Link:** https://doi.org/10.1016/j.patcog.2024.110424
- **Access:** Open access (green); free copy: https://pmc.ncbi.nlm.nih.gov/articles/PMC10976951/pdf/nihms-1977935.pdf
- **Where and how it is used:** Ch.2 medical-imaging FL survey; supports the claim that most imaging FL work uses generic aggregation.
- **Verified:** Crossref record matched (title, year); doi.org resolver HTTP 302

**[6]** N. Rieke et al., "The future of digital health with federated learning," *npj Digital Medicine*, vol. 3, art. no. 119, 2020, doi: 10.1038/s41746-020-00323-1.  
- **Key:** `rieke2020future`  ·  **Link:** https://doi.org/10.1038/s41746-020-00323-1
- **Access:** Open access (gold); free copy: https://www.nature.com/articles/s41746-020-00323-1.pdf
- **Where and how it is used:** Ch.1 motivation: privacy constraints on multi-hospital MRI data.
- **Verified:** Crossref record matched (title, year); doi.org resolver HTTP 302

**[7]** G. A. Kaissis, M. R. Makowski, D. Rückert, and R. F. Braren, "Secure, privacy-preserving and federated machine learning in medical imaging," *Nature Machine Intelligence*, vol. 2, no. 6, pp. 305–311, 2020, doi: 10.1038/s42256-020-0186-1.  
- **Key:** `kaissis2020secure`  ·  **Link:** https://doi.org/10.1038/s42256-020-0186-1
- **Access:** Open access (hybrid); free copy: https://www.nature.com/articles/s42256-020-0186-1.pdf
- **Where and how it is used:** Ch.1 motivation: regulatory/privacy case for FL in imaging.
- **Verified:** Crossref record matched (title, year); doi.org resolver HTTP 302

**[8]** M. J. Sheller et al., "Federated learning in medicine: facilitating multi-institutional collaborations without sharing patient data," *Scientific Reports*, vol. 10, art. no. 12598, 2020, doi: 10.1038/s41598-020-69250-1.  
- **Key:** `sheller2020fl`  ·  **Link:** https://doi.org/10.1038/s41598-020-69250-1
- **Access:** Open access (gold); free copy: https://www.nature.com/articles/s41598-020-69250-1.pdf
- **Where and how it is used:** Ch.1/2 evidence that FL approaches centralized accuracy on brain-tumor data across institutions.
- **Verified:** Crossref record matched (title, year); doi.org resolver HTTP 302

**[9]** S. Pati et al., "Federated learning enables big data for rare cancer boundary detection," *Nature Communications*, vol. 13, art. no. 7346, 2022, doi: 10.1038/s41467-022-33407-5.  
- **Key:** `pati2022rare`  ·  **Link:** https://doi.org/10.1038/s41467-022-33407-5
- **Access:** Open access (gold); free copy: https://www.nature.com/articles/s41467-022-33407-5.pdf
- **Where and how it is used:** Ch.2 largest real-world FL study on glioblastoma MRI (71 sites); domain anchor for Pillar II.
- **Verified:** Crossref record matched (title, year); doi.org resolver HTTP 302

**[10]** G. Litjens et al., "A survey on deep learning in medical image analysis," *Medical Image Analysis*, vol. 42, pp. 60–88, 2017, doi: 10.1016/j.media.2017.07.005.  
- **Key:** `litjens2017survey`  ·  **Link:** https://doi.org/10.1016/j.media.2017.07.005
- **Access:** Open access (bronze); free copy: https://arxiv.org/abs/1702.05747
- **Where and how it is used:** Ch.2 deep-learning-in-imaging background for the CNN baselines.
- **Verified:** Crossref record matched (title, year); doi.org resolver HTTP 302

**[11]** R. Lopes and N. Betrouni, "Fractal and multifractal analysis: A review," *Medical Image Analysis*, vol. 13, no. 4, pp. 634–649, 2009, doi: 10.1016/j.media.2009.05.003.  
- **Key:** `lopes2009fractal`  ·  **Link:** https://doi.org/10.1016/j.media.2009.05.003
- **Access:** Subscription (institutional access)
- **Where and how it is used:** Ch.2 review of fractal analysis in medical imaging; background for Pillar I.
- **Verified:** Crossref record matched (title, year); doi.org resolver HTTP 302

**[12]** A. Di Ieva, F. Grizzi, H. Jelinek, A. J. Pellionisz, and G. A. Losa, "Fractals in the Neurosciences, Part I: General Principles and Basic Neurosciences," *The Neuroscientist*, vol. 20, no. 4, pp. 403–417, 2014, doi: 10.1177/1073858413513927.  
- **Key:** `dieva2014fractalsI`  ·  **Link:** https://doi.org/10.1177/1073858413513927
- **Access:** Subscription (institutional access)
- **Where and how it is used:** Ch.2 fractal geometry of the brain; justifies FD as a neuro-anatomical descriptor.
- **Verified:** Crossref record matched (title, year); doi.org resolver HTTP 302

**[13]** A. Di Ieva, F. J. Esteban, F. Grizzi, W. Klonowski, and M. Martín-Landrove, "Fractals in the neurosciences, Part II: Clinical applications and future perspectives," *The Neuroscientist*, vol. 21, no. 1, pp. 30–43, 2015, doi: 10.1177/1073858413513928.  
- **Key:** `dieva2015fractalsII`  ·  **Link:** https://doi.org/10.1177/1073858413513928
- **Access:** Subscription (institutional access)
- **Where and how it is used:** Ch.2 clinical fractal applications incl. neuro-oncology.
- **Verified:** Crossref record matched (title, year); doi.org resolver HTTP 302

**[14]** D. N. Louis et al., "The 2021 WHO Classification of Tumors of the Central Nervous System: a summary," *Neuro-Oncology*, vol. 23, no. 8, pp. 1231–1251, 2021, doi: 10.1093/neuonc/noab106.  
- **Key:** `louis2021who`  ·  **Link:** https://doi.org/10.1093/neuonc/noab106
- **Access:** Open access (bronze); free copy: https://academic.oup.com/neuro-oncology/article-pdf/23/8/1231/39535372/noab106.pdf
- **Where and how it is used:** Ch.1 clinical definitions of glioma, meningioma and pituitary tumour classes.
- **Verified:** Crossref record matched (title, year); doi.org resolver HTTP 302

**[15]** J. E. Villanueva-Meyer, M. C. Mabray, and S. Cha, "Current Clinical Brain Tumor Imaging," *Neurosurgery*, vol. 81, no. 3, pp. 397–415, 2017, doi: 10.1093/neuros/nyx103.  
- **Key:** `villanueva2017imaging`  ·  **Link:** https://doi.org/10.1093/neuros/nyx103
- **Access:** Open access (green); free copy: https://www.ncbi.nlm.nih.gov/pmc/articles/5581219
- **Where and how it is used:** Ch.1 why contrast-enhanced T1 MRI is the standard modality for these tumours.
- **Verified:** Crossref record matched (title, year); doi.org resolver HTTP 302

**[16]** R. J. Gillies, P. E. Kinahan, and H. Hricak, "Radiomics: Images Are More than Pictures, They Are Data," *Radiology*, vol. 278, no. 2, pp. 563–577, 2016, doi: 10.1148/radiol.2015151169.  
- **Key:** `gillies2016radiomics`  ·  **Link:** https://doi.org/10.1148/radiol.2015151169
- **Access:** Open access (green); free copy: https://www.ncbi.nlm.nih.gov/pmc/articles/4734157
- **Where and how it is used:** Ch.2 positions the hand-crafted 16-feature model within radiomics.
- **Verified:** Crossref record matched (title, year); doi.org resolver HTTP 302

**[17]** P. Lambin et al., "Radiomics: Extracting more information from medical images using advanced feature analysis," *European Journal of Cancer*, vol. 48, no. 4, pp. 441–446, 2012, doi: 10.1016/j.ejca.2011.11.036.  
- **Key:** `lambin2012radiomics`  ·  **Link:** https://doi.org/10.1016/j.ejca.2011.11.036
- **Access:** Open access (bronze); free copy: http://www.ejcancer.com/article/S0959804911009993/pdf
- **Where and how it is used:** Ch.2 origin of the radiomics feature-extraction paradigm.
- **Verified:** Crossref record matched (title, year); doi.org resolver HTTP 302


## A. Datasets and evaluation protocol

*Data sources, leakage-free evaluation, and the non-IID partition used in every federated stage.*

**[18]** J. Cheng et al., "Enhanced Performance of Brain Tumor Classification via Tumor Region Augmentation and Partition," *PLOS ONE*, vol. 10, no. 10, art. no. e0140381, 2015, doi: 10.1371/journal.pone.0140381.  
- **Key:** `cheng2015enhanced`  ·  **Link:** https://doi.org/10.1371/journal.pone.0140381
- **Access:** Open access (gold); free copy: https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0140381&type=printable
- **Where and how it is used:** Ch.3 source paper of the Figshare dataset (3064 slices, 233 patients); also the 0.913 benchmark Pillar I is compared against.
- **Verified:** Crossref record matched (title, year); doi.org resolver HTTP 302

**[19]** J. Cheng et al., "Retrieval of Brain Tumors by Adaptive Spatial Pooling and Fisher Vector Representation," *PLOS ONE*, vol. 11, no. 6, art. no. e0157112, 2016, doi: 10.1371/journal.pone.0157112.  
- **Key:** `cheng2016retrieval`  ·  **Link:** https://doi.org/10.1371/journal.pone.0157112
- **Access:** Open access (gold)
- **Where and how it is used:** Ch.3 companion paper describing the same dataset and its PID field used for patient grouping.
- **Verified:** Crossref record matched (title, year); doi.org resolver HTTP 302

**[20]** J. Cheng, "Brain tumor dataset," figshare, Dataset, version 5, Apr. 2017, doi: 10.6084/m9.figshare.1512427.v5.  
- **Key:** `cheng2017figshare`  ·  **Link:** https://doi.org/10.6084/m9.figshare.1512427.v5
- **Access:** Open data (CC BY 4.0)
- **Where and how it is used:** Ch.3 the primary dataset itself (version 5, CC BY 4.0); every stage loads these .mat files.
- **Verified:** DataCite record matched (title, year); doi.org resolver HTTP 302

**[21]** A. Fateh et al., "BRISC: Annotated Dataset for Brain Tumor Segmentation and Classification," *Scientific Data*, vol. 13, art. no. 361, 2026, doi: 10.1038/s41597-026-06753-y.  
- **Key:** `fateh2026brisc`  ·  **Link:** https://doi.org/10.1038/s41597-026-06753-y
- **Access:** Open access (gold); free copy: https://www.nature.com/articles/s41597-026-06753-y_reference.pdf
- **Where and how it is used:** Ch.5 external validation set for Pillar II (Stage 5 / 9B); independence caveat stated because it shares SARTAJ/Br35H lineage.
- **Verified:** Crossref record matched (title, year); doi.org resolver HTTP 302

**[22]** M. Nickparvar, "Brain Tumor MRI Dataset," Kaggle, Dataset, 2021, doi: 10.34740/KAGGLE/DSV/2645886.  
- **Key:** `nickparvar2021kaggle`  ·  **Link:** https://doi.org/10.34740/KAGGLE/DSV/2645886
- **Access:** Open data (Kaggle account)
- **Where and how it is used:** Ch.3 aggregate public dataset from which BRISC partly derives; cited to document the shared lineage.
- **Verified:** DataCite record matched (title, year); doi.org resolver HTTP 302

**[23]** B. H. Menze et al., "The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS)," *IEEE Transactions on Medical Imaging*, vol. 34, no. 10, pp. 1993–2024, 2015, doi: 10.1109/TMI.2014.2377694.  
- **Key:** `menze2015brats`  ·  **Link:** https://doi.org/10.1109/TMI.2014.2377694
- **Access:** Open access (hybrid); free copy: https://ieeexplore.ieee.org/ielx7/42/7283692/06975210.pdf
- **Where and how it is used:** Ch.2 BraTS benchmark; context for segmentation and the proposal's original dataset choice.
- **Verified:** Crossref record matched (title, year); doi.org resolver HTTP 302

**[24]** S. Bakas et al., "Advancing The Cancer Genome Atlas glioma MRI collections with expert segmentation labels and radiomic features," *Scientific Data*, vol. 4, art. no. 170117, 2017, doi: 10.1038/sdata.2017.117.  
- **Key:** `bakas2017tcga`  ·  **Link:** https://doi.org/10.1038/sdata.2017.117
- **Access:** Open access (gold); free copy: https://www.nature.com/articles/sdata2017117.pdf
- **Where and how it is used:** Ch.2 expert-labelled glioma MRI collections underlying BraTS.
- **Verified:** Crossref record matched (title, year); doi.org resolver HTTP 302

**[25]** E. Yagis et al., "Effect of data leakage in brain MRI classification using 2D convolutional neural networks," *Scientific Reports*, vol. 11, art. no. 22544, 2021, doi: 10.1038/s41598-021-01681-w.  
- **Key:** `yagis2021leakage`  ·  **Link:** https://doi.org/10.1038/s41598-021-01681-w
- **Access:** Open access (gold); free copy: https://www.nature.com/articles/s41598-021-01681-w.pdf
- **Where and how it is used:** Ch.3/4 central rigor citation: slice-level splits inflate accuracy 29-55 pp; our measured inflation (+3.4 pp) is compared against it.
- **Verified:** Crossref record matched (title, year); doi.org resolver HTTP 302

**[26]** S. Kapoor and A. Narayanan, "Leakage and the reproducibility crisis in machine-learning-based science," *Patterns*, vol. 4, no. 9, art. no. 100804, 2023, doi: 10.1016/j.patter.2023.100804.  
- **Key:** `kapoor2023leakage`  ·  **Link:** https://doi.org/10.1016/j.patter.2023.100804
- **Access:** Open access (gold); free copy: http://www.cell.com/article/S2666389923001599/pdf
- **Where and how it is used:** Ch.3 general taxonomy of data leakage in ML-based science; companion to Yagis.
- **Verified:** Crossref record matched (title, year); doi.org resolver HTTP 302

**[27]** G. Varoquaux and V. Cheplygina, "Machine learning for medical imaging: methodological failures and recommendations for the future," *npj Digital Medicine*, vol. 5, art. no. 48, 2022, doi: 10.1038/s41746-022-00592-y.  
- **Key:** `varoquaux2022failures`  ·  **Link:** https://doi.org/10.1038/s41746-022-00592-y
- **Access:** Open access (gold); free copy: https://www.nature.com/articles/s41746-022-00592-y.pdf
- **Where and how it is used:** Ch.3 methodological pitfalls (small test sets, leakage) motivating 20-seed patient-grouped protocol.
- **Verified:** Crossref record matched (title, year); doi.org resolver HTTP 302

**[28]** M. Roberts et al., "Common pitfalls and recommendations for using machine learning to detect and prognosticate for COVID-19 using chest radiographs and CT scans," *Nature Machine Intelligence*, vol. 3, no. 3, pp. 199–217, 2021, doi: 10.1038/s42256-021-00307-0.  
- **Key:** `roberts2021pitfalls`  ·  **Link:** https://doi.org/10.1038/s42256-021-00307-0
- **Access:** Open access (hybrid); free copy: https://www.nature.com/articles/s42256-021-00307-0.pdf
- **Where and how it is used:** Ch.3 evidence that flawed evaluation invalidates published imaging models; motivates the leakage guard.
- **Verified:** Crossref record matched (title, year); doi.org resolver HTTP 302

**[29]** T.-M. H. Hsu, H. Qi, and M. Brown, "Measuring the Effects of Non-Identical Data Distribution for Federated Visual Classification," arXiv preprint arXiv:1909.06335, 2019.  
- **Key:** `hsu2019dirichlet`  ·  **Link:** https://arxiv.org/abs/1909.06335
- **Access:** Open access (arXiv / proceedings)
- **Where and how it is used:** Ch.5 Dirichlet(alpha) client partition used to simulate class skew in every Pillar II stage.
- **Verified:** arXiv API record matched (title); abs page HTTP 200; no peer-reviewed version found, cited as preprint

**[30]** Y. Zhao, M. Li, L. Lai, N. Suda, D. Civin, and V. Chandra, "Federated Learning with Non-IID Data," arXiv preprint arXiv:1806.00582, 2018.  
- **Key:** `zhao2018noniid`  ·  **Link:** https://arxiv.org/abs/1806.00582
- **Access:** Open access (arXiv / proceedings)
- **Where and how it is used:** Ch.2/5 first empirical demonstration of FedAvg accuracy loss under label skew (weight divergence).
- **Verified:** arXiv API record matched (title); abs page HTTP 200; no peer-reviewed version found, cited as preprint

**[31]** Q. Li, Y. Diao, Q. Chen, and B. He, "Federated Learning on Non-IID Data Silos: An Experimental Study," in Proc. IEEE 38th Int. Conf. Data Eng. (ICDE), 2022, pp. 965–978, doi: 10.1109/ICDE53745.2022.00077.  
- **Key:** `li2022silos`  ·  **Link:** https://doi.org/10.1109/ICDE53745.2022.00077
- **Access:** Subscription (institutional access)
- **Where and how it is used:** Ch.5 benchmark of partition strategies (#C=k and Dir(beta)); justifies the partition design.
- **Verified:** Crossref record matched (title, year); doi.org resolver HTTP 302


## B. Pre-processing, morphology and software

*Everything that touches the image before a feature is computed, plus the libraries the code imports.*

**[32]** N. Otsu, "A Threshold Selection Method from Gray-Level Histograms," *IEEE Transactions on Systems, Man, and Cybernetics*, vol. 9, no. 1, pp. 62–66, 1979, doi: 10.1109/TSMC.1979.4310076.  
- **Key:** `otsu1979`  ·  **Link:** https://doi.org/10.1109/TSMC.1979.4310076
- **Access:** Subscription (institutional access)
- **Where and how it is used:** Ch.3/6 Otsu threshold on the asymmetry map in the symmetry-seeded contour (Stage 12).
- **Verified:** Crossref record matched (title, year); doi.org resolver HTTP 302

**[33]** J. Serra, *Image Analysis and Mathematical Morphology*. London, U.K.: Academic Press, 1982, ISBN 0-12-637240-3.  
- **Key:** `serra1982`  ·  **Link:** https://openlibrary.org/isbn/9780126372403
- **Access:** Book (library copy; no free full text)
- **Where and how it is used:** Ch.3 binary opening, hole filling, erosion/dilation used in brain_mask_of() and the tumour-border ring.
- **Verified:** ISBN record matched in Open Library

**[34]** P. Soille, *Morphological Image Analysis*, 2nd ed. Berlin, Germany: Springer, 2004, doi: 10.1007/978-3-662-05088-0.  
- **Key:** `soille2003`  ·  **Link:** https://doi.org/10.1007/978-3-662-05088-0
- **Access:** Book (library / publisher)
- **Where and how it is used:** Ch.3 modern reference for the same morphological operators and connected-component labelling.
- **Verified:** Crossref record matched (title, year); doi.org resolver HTTP 302

**[35]** R. C. Gonzalez and R. E. Woods, *Digital Image Processing*, 4th ed. New York, NY, USA: Pearson, 2018, ISBN 978-0-13-335672-4.  
- **Key:** `gonzalez2018dip`  ·  **Link:** https://openlibrary.org/isbn/9780133356724
- **Access:** Book (library copy; no free full text)
- **Where and how it is used:** Ch.3 textbook reference for bilinear/nearest-neighbour resizing to 128x128 and min-max intensity scaling.
- **Verified:** ISBN record matched in Open Library

**[36]** C. R. Harris et al., "Array programming with NumPy," *Nature*, vol. 585, no. 7825, pp. 357–362, 2020, doi: 10.1038/s41586-020-2649-2.  
- **Key:** `harris2020numpy`  ·  **Link:** https://doi.org/10.1038/s41586-020-2649-2
- **Access:** Open access (hybrid); free copy: https://www.nature.com/articles/s41586-020-2649-2.pdf
- **Where and how it is used:** Ch.3 numerical backbone of all feature extraction code.
- **Verified:** Crossref record matched (title, year); doi.org resolver HTTP 302

**[37]** P. Virtanen et al., "SciPy 1.0: fundamental algorithms for scientific computing in Python," *Nature Methods*, vol. 17, no. 3, pp. 261–272, 2020, doi: 10.1038/s41592-019-0686-2.  
- **Key:** `virtanen2020scipy`  ·  **Link:** https://doi.org/10.1038/s41592-019-0686-2
- **Access:** Open access (hybrid)
- **Where and how it is used:** Ch.3 scipy.ndimage morphology/labelling and scipy.stats (Wilcoxon, Kruskal-Wallis).
- **Verified:** Crossref record matched (title, year); doi.org resolver HTTP 302

**[38]** S. van der Walt et al., "scikit-image: image processing in Python," *PeerJ*, vol. 2, art. no. e453, 2014, doi: 10.7717/peerj.453.  
- **Key:** `vanderwalt2014skimage`  ·  **Link:** https://doi.org/10.7717/peerj.453
- **Access:** Open access (gold)
- **Where and how it is used:** Ch.3 GLCM (graycomatrix), regionprops shape features, morphological active contours.
- **Verified:** Crossref record matched (title, year); doi.org resolver HTTP 302

**[39]** F. Pedregosa et al., "Scikit-learn: Machine Learning in Python," *J. Mach. Learn. Res.*, vol. 12, pp. 2825–2830, 2011. [Online]. Available: https://jmlr.org/papers/v12/pedregosa11a.html  
- **Key:** `pedregosa2011sklearn`  ·  **Link:** https://jmlr.org/papers/v12/pedregosa11a.html
- **Access:** Open access (official proceedings / journal site)
- **Where and how it is used:** Ch.3/4 RandomForestClassifier, GroupKFold, StratifiedGroupKFold, GridSearchCV, balanced accuracy, ROC AUC.
- **Verified:** official page HTTP 200; title and citation meta-tags (authors, pages) read from the page

**[40]** A. Paszke et al., "PyTorch: An Imperative Style, High-Performance Deep Learning Library," in Adv. Neural Inf. Process. Syst. (NeurIPS), vol. 32, 2019. [Online]. Available: https://proceedings.neurips.cc/paper/2019/hash/bdbca288fee7f92f2bfa9f7012727740-Abstract.html  
- **Key:** `paszke2019pytorch`  ·  **Link:** https://proceedings.neurips.cc/paper/2019/hash/bdbca288fee7f92f2bfa9f7012727740-Abstract.html
- **Access:** Open access (official proceedings / journal site)
- **Where and how it is used:** Ch.5 framework for all federated training code (SmallNet, ResNet-18, SCAFFOLD).
- **Verified:** official page HTTP 200; title and citation meta-tags (authors, pages) read from the page

**[41]** J. D. Hunter, "Matplotlib: A 2D Graphics Environment," *Computing in Science & Engineering*, vol. 9, no. 3, pp. 90–95, 2007, doi: 10.1109/MCSE.2007.55.  
- **Key:** `hunter2007matplotlib`  ·  **Link:** https://doi.org/10.1109/MCSE.2007.55
- **Access:** Subscription (institutional access)
- **Where and how it is used:** Ch.4/5 all figures.
- **Verified:** Crossref record matched (title, year); doi.org resolver HTTP 302


## C. Fractal and texture descriptors (Pillar I)

*The descriptor C = Ft/Fb, its components, the 16-feature set, and known estimator biases.*

**[42]** B. B. Mandelbrot, *The Fractal Geometry of Nature*. San Francisco, CA, USA: W. H. Freeman, 1982, ISBN 978-0-7167-1186-5.  
- **Key:** `mandelbrot1982`  ·  **Link:** https://openlibrary.org/isbn/9780716711865
- **Access:** Book (library copy; no free full text)
- **Where and how it is used:** Ch.2/3 definition of fractal dimension underlying Ft and Fb.
- **Verified:** ISBN record matched in Open Library

**[43]** A. P. Pentland, "Fractal-Based Description of Natural Scenes," *IEEE Transactions on Pattern Analysis and Machine Intelligence*, vol. PAMI-6, no. 6, pp. 661–674, 1984, doi: 10.1109/TPAMI.1984.4767591.  
- **Key:** `pentland1984fractal`  ·  **Link:** https://doi.org/10.1109/TPAMI.1984.4767591
- **Access:** Subscription (institutional access)
- **Where and how it is used:** Ch.2 first use of fractal dimension as an image texture descriptor.
- **Verified:** Crossref record matched (title, year); doi.org resolver HTTP 302

**[44]** J. M. Keller, S. Chen, and R. M. Crownover, "Texture description and segmentation through fractal geometry," *Computer Vision, Graphics, and Image Processing*, vol. 45, no. 2, pp. 150–166, 1989, doi: 10.1016/0734-189X(89)90130-8.  
- **Key:** `keller1989texture`  ·  **Link:** https://doi.org/10.1016/0734-189X(89)90130-8
- **Access:** Subscription (institutional access)
- **Where and how it is used:** Ch.2 box-counting FD for grey-level texture; introduces lacunarity to disambiguate equal-FD textures.
- **Verified:** Crossref record matched (title, year); doi.org resolver HTTP 302

**[45]** N. Sarkar and B. B. Chaudhuri, "An efficient differential box-counting approach to compute fractal dimension of image," *IEEE Transactions on Systems, Man, and Cybernetics*, vol. 24, no. 1, pp. 115–120, 1994, doi: 10.1109/21.259692.  
- **Key:** `sarkar1994dbc`  ·  **Link:** https://doi.org/10.1109/21.259692
- **Access:** Subscription (institutional access)
- **Where and how it is used:** Ch.3 the exact algorithm in boxcount_fd(): differential box counting on the grey-level surface.
- **Verified:** Crossref record matched (title, year); doi.org resolver HTTP 302

**[46]** B. B. Chaudhuri and N. Sarkar, "Texture segmentation using fractal dimension," *IEEE Transactions on Pattern Analysis and Machine Intelligence*, vol. 17, no. 1, pp. 72–77, 1995, doi: 10.1109/34.368149.  
- **Key:** `chaudhuri1995texture`  ·  **Link:** https://doi.org/10.1109/34.368149
- **Access:** Subscription (institutional access)
- **Where and how it is used:** Ch.2 FD-based texture segmentation by the DBC authors; supports FD as a region descriptor.
- **Verified:** Crossref record matched (title, year); doi.org resolver HTTP 302

**[47]** K. M. Iftekharuddin, W. Jia, and R. Marsh, "Fractal analysis of tumor in brain MR images," *Machine Vision and Applications*, vol. 13, no. 5–6, pp. 352–362, 2003, doi: 10.1007/s00138-002-0087-9.  
- **Key:** `iftekharuddin2003fractal`  ·  **Link:** https://doi.org/10.1007/s00138-002-0087-9
- **Access:** Subscription (institutional access)
- **Where and how it is used:** Ch.2 first box-counting FD analysis of brain tumours in MRI; prior art for Ft.
- **Verified:** Crossref record matched (title, year); doi.org resolver HTTP 302

**[48]** K. M. Iftekharuddin, J. Zheng, M. A. Islam, and R. J. Ogg, "Fractal-based brain tumor detection in multimodal MRI," *Applied Mathematics and Computation*, vol. 207, no. 1, pp. 23–41, 2009, doi: 10.1016/j.amc.2007.10.063.  
- **Key:** `iftekharuddin2009multimodal`  ·  **Link:** https://doi.org/10.1016/j.amc.2007.10.063
- **Access:** Subscription (institutional access)
- **Where and how it is used:** Ch.2 FD + fractional Brownian motion features for tumour detection; prior art for FD-based classification.
- **Verified:** Crossref record matched (title, year); doi.org resolver HTTP 302

**[49]** D. Battalapalli, S. Vidyadharan, B. V. V. S. N. Prabhakar Rao, P. Yogeeswari, C. Kesavadas, and V. Rajagopalan, "Fractal dimension: analyzing its potential as a neuroimaging biomarker for brain tumor diagnosis using machine learning," *Frontiers in Physiology*, vol. 14, art. no. 1201617, 2023, doi: 10.3389/fphys.2023.1201617.  
- **Key:** `battalapalli2023fd`  ·  **Link:** https://doi.org/10.3389/fphys.2023.1201617
- **Access:** Open access (gold); free copy: https://www.frontiersin.org/articles/10.3389/fphys.2023.1201617/pdf
- **Where and how it is used:** Ch.2/4 closest precedent: uses tumour AND non-tumour FD as separate features; our Stage 10B shows the separate pair (Ft,Fb) indeed beats the ratio C.
- **Verified:** Crossref record matched (title, year); doi.org resolver HTTP 302

**[50]** J. Sánchez and M. Martín-Landrove, "Morphological and Fractal Properties of Brain Tumors," *Frontiers in Physiology*, vol. 13, art. no. 878391, 2022, doi: 10.3389/fphys.2022.878391.  
- **Key:** `sanchez2022fractal`  ·  **Link:** https://doi.org/10.3389/fphys.2022.878391
- **Access:** Open access (gold); free copy: https://www.frontiersin.org/articles/10.3389/fphys.2022.878391/pdf
- **Where and how it is used:** Ch.2 fractal/morphological characterisation of brain tumour interfaces; prior art for relative (ratio-type) fractal quantities.
- **Verified:** Crossref record matched (title, year); doi.org resolver HTTP 302

**[51]** T. G. Smith, G. D. Lange, and W. B. Marks, "Fractal methods and results in cellular morphology: dimensions, lacunarity and multifractals," *Journal of Neuroscience Methods*, vol. 69, no. 2, pp. 123–136, 1996, doi: 10.1016/S0165-0270(96)00080-5.  
- **Key:** `smith1996fractal`  ·  **Link:** https://doi.org/10.1016/S0165-0270(96)00080-5
- **Access:** Subscription (institutional access)
- **Where and how it is used:** Ch.3 practical box-counting guidance and known estimator biases.
- **Verified:** Crossref record matched (title, year); doi.org resolver HTTP 302

**[52]** G. Gonzato, F. Mulargia, and W. Marzocchi, "Practical application of fractal analysis: problems and solutions," *Geophysical Journal International*, vol. 132, no. 2, pp. 275–282, 1998, doi: 10.1046/j.1365-246x.1998.00461.x.  
- **Key:** `gonzato1998bias`  ·  **Link:** https://doi.org/10.1046/j.1365-246x.1998.00461.x
- **Access:** Open access (bronze); free copy: https://academic.oup.com/gji/article-pdf/132/2/275/1954393/132-2-275.pdf
- **Where and how it is used:** Ch.4 finite-size/sampling bias of box counting on small regions; explains the size confound (45% of C survives size removal).
- **Verified:** Crossref record matched (title, year); doi.org resolver HTTP 302

**[53]** G. Gonzato, F. Mulargia, and M. Ciccotti, "Measuring the fractal dimensions of ideal and actual objects: implications for application in geology and geophysics," *Geophysical Journal International*, vol. 142, no. 1, pp. 108–116, 2000, doi: 10.1046/j.1365-246x.2000.00133.x.  
- **Key:** `gonzato2000ideal`  ·  **Link:** https://doi.org/10.1046/j.1365-246x.2000.00133.x
- **Access:** Open access (bronze); free copy: https://academic.oup.com/gji/article-pdf/142/1/108/1602887/142-1-108.pdf
- **Where and how it is used:** Ch.4 placement/orientation bias of box counting; supports the box-size robustness test (corr 0.963-0.993).
- **Verified:** Crossref record matched (title, year); doi.org resolver HTTP 302

**[54]** C. Allain and M. Cloitre, "Characterizing the lacunarity of random and deterministic fractal sets," *Physical Review A*, vol. 44, no. 6, pp. 3552–3558, 1991, doi: 10.1103/PhysRevA.44.3552.  
- **Key:** `allain1991lacunarity`  ·  **Link:** https://doi.org/10.1103/PhysRevA.44.3552
- **Access:** Subscription (institutional access)
- **Where and how it is used:** Ch.3 gliding-box lacunarity definition behind lacun_t and lacun_b.
- **Verified:** Crossref record matched (title, year); doi.org resolver HTTP 302

**[55]** R. E. Plotnick, R. H. Gardner, W. W. Hargrove, K. Prestegaard, and M. Perlmutter, "Lacunarity analysis: A general technique for the analysis of spatial patterns," *Physical Review E*, vol. 53, no. 5, pp. 5461–5468, 1996, doi: 10.1103/PhysRevE.53.5461.  
- **Key:** `plotnick1996lacunarity`  ·  **Link:** https://doi.org/10.1103/PhysRevE.53.5461
- **Access:** Subscription (institutional access)
- **Where and how it is used:** Ch.3 lacunarity as a general spatial-pattern measure; justifies using it on binary masks.
- **Verified:** Crossref record matched (title, year); doi.org resolver HTTP 302

**[56]** R. M. Haralick, K. Shanmugam, and I. Dinstein, "Textural Features for Image Classification," *IEEE Transactions on Systems, Man, and Cybernetics*, vol. SMC-3, no. 6, pp. 610–621, 1973, doi: 10.1109/TSMC.1973.4309314.  
- **Key:** `haralick1973glcm`  ·  **Link:** https://doi.org/10.1109/TSMC.1973.4309314
- **Access:** Subscription (institutional access)
- **Where and how it is used:** Ch.3 GLCM contrast/correlation features (GLCM_t, GLCM_b, correlation_t).
- **Verified:** Crossref record matched (title, year); doi.org resolver HTTP 302

**[57]** C. E. Shannon, "A Mathematical Theory of Communication," *Bell Syst. Tech. J.*, vol. 27, no. 3, pp. 379–423, Jul. 1948, doi: 10.1002/j.1538-7305.1948.tb01338.x; and vol. 27, no. 4, pp. 623–656, Oct. 1948, doi: 10.1002/j.1538-7305.1948.tb00917.x.  
- **Key:** `shannon1948`  ·  **Link:** https://doi.org/10.1002/j.1538-7305.1948.tb01338.x
- **Access:** Subscription (institutional access)
- **Where and how it is used:** Ch.3 Shannon entropy of the tumour intensity histogram (Dispersion_t).
- **Verified:** Crossref record matched (title, year); doi.org resolver HTTP 302

**[58]** A. Zwanenburg et al., "The Image Biomarker Standardization Initiative: Standardized Quantitative Radiomics for High-Throughput Image-based Phenotyping," *Radiology*, vol. 295, no. 2, pp. 328–338, 2020, doi: 10.1148/radiol.2020191145.  
- **Key:** `zwanenburg2020ibsi`  ·  **Link:** https://doi.org/10.1148/radiol.2020191145
- **Access:** Open access (green); free copy: https://cris.maastrichtuniversity.nl/en/publications/a842e562-d051-4db2-9b40-ef1d54c71af4
- **Where and how it is used:** Ch.3 standard definitions of texture/shape features; used to align feature naming and discretisation.
- **Verified:** Crossref record matched (title, year); doi.org resolver HTTP 302

**[59]** J. J. M. van Griethuysen et al., "Computational Radiomics System to Decode the Radiographic Phenotype," *Cancer Research*, vol. 77, no. 21, pp. e104–e107, 2017, doi: 10.1158/0008-5472.CAN-17-0339.  
- **Key:** `vangriethuysen2017pyradiomics`  ·  **Link:** https://doi.org/10.1158/0008-5472.CAN-17-0339
- **Access:** Open access (bronze); free copy: https://cancerres.aacrjournals.org/content/canres/77/21/e104.full.pdf
- **Where and how it is used:** Ch.3 reference implementation of radiomic features; comparison point for our feature code.
- **Verified:** Crossref record matched (title, year); doi.org resolver HTTP 302


## D. Segmentation: automatic delineation attempt (Pillar I, Stages 11-12)

*Active contours, seeding by bilateral symmetry, and the overlap metrics used to score them.*

**[60]** M. Kass, A. Witkin, and D. Terzopoulos, "Snakes: Active contour models," *International Journal of Computer Vision*, vol. 1, no. 4, pp. 321–331, 1988, doi: 10.1007/BF00133570.  
- **Key:** `kass1988snakes`  ·  **Link:** https://doi.org/10.1007/BF00133570
- **Access:** Subscription (institutional access)
- **Where and how it is used:** Ch.4 origin of active contours (supervisor-suggested method); explains curvature penalty that smooths fractal borders.
- **Verified:** Crossref record matched (title, year); doi.org resolver HTTP 302

**[61]** S. Osher and J. A. Sethian, "Fronts propagating with curvature-dependent speed: Algorithms based on Hamilton-Jacobi formulations," *Journal of Computational Physics*, vol. 79, no. 1, pp. 12–49, 1988, doi: 10.1016/0021-9991(88)90002-2.  
- **Key:** `osher1988levelset`  ·  **Link:** https://doi.org/10.1016/0021-9991(88)90002-2
- **Access:** Open access (green); free copy: https://ntrs.nasa.gov/api/citations/19880001113/downloads/19880001113.pdf
- **Where and how it is used:** Ch.4 level-set formulation underlying region-based contours.
- **Verified:** Crossref record matched (title, year); doi.org resolver HTTP 302

**[62]** D. Mumford and J. Shah, "Optimal approximations by piecewise smooth functions and associated variational problems," *Communications on Pure and Applied Mathematics*, vol. 42, no. 5, pp. 577–685, 1989, doi: 10.1002/cpa.3160420503.  
- **Key:** `mumford1989`  ·  **Link:** https://doi.org/10.1002/cpa.3160420503
- **Access:** Open access (green); free copy: http://nrs.harvard.edu/urn-3:HUL.InstRepos:3637121
- **Where and how it is used:** Ch.4 Mumford-Shah functional, the energy Chan-Vese simplifies.
- **Verified:** Crossref record matched (title, year); doi.org resolver HTTP 302

**[63]** T. F. Chan and L. A. Vese, "Active contours without edges," *IEEE Transactions on Image Processing*, vol. 10, no. 2, pp. 266–277, 2001, doi: 10.1109/83.902291.  
- **Key:** `chan2001acwe`  ·  **Link:** https://doi.org/10.1109/83.902291
- **Access:** Subscription (institutional access)
- **Where and how it is used:** Ch.4 Chan-Vese energy minimised by morphological_chan_vese in Stage 11.
- **Verified:** Crossref record matched (title, year); doi.org resolver HTTP 302

**[64]** V. Caselles, R. Kimmel, and G. Sapiro, "Geodesic Active Contours," *International Journal of Computer Vision*, vol. 22, no. 1, pp. 61–79, 1997, doi: 10.1023/A:1007979827043.  
- **Key:** `caselles1997gac`  ·  **Link:** https://doi.org/10.1023/A:1007979827043
- **Access:** Subscription (institutional access)
- **Where and how it is used:** Ch.4 geodesic variant (morphological_geodesic_active_contour) tested in Stage 11.
- **Verified:** Crossref record matched (title, year); doi.org resolver HTTP 302

**[65]** P. Marquez-Neila, L. Baumela, and L. Alvarez, "A Morphological Approach to Curvature-Based Evolution of Curves and Surfaces," *IEEE Transactions on Pattern Analysis and Machine Intelligence*, vol. 36, no. 1, pp. 2–17, 2014, doi: 10.1109/TPAMI.2013.106.  
- **Key:** `marquezneila2014morph`  ·  **Link:** https://doi.org/10.1109/TPAMI.2013.106
- **Access:** Subscription (institutional access)
- **Where and how it is used:** Ch.4 the exact morphological-snakes algorithm implemented in scikit-image and used in Stages 11-12.
- **Verified:** Crossref record matched (title, year); doi.org resolver HTTP 302

**[66]** C. Li, C. Xu, C. Gui, and M. D. Fox, "Distance Regularized Level Set Evolution and Its Application to Image Segmentation," *IEEE Transactions on Image Processing*, vol. 19, no. 12, pp. 3243–3254, 2010, doi: 10.1109/TIP.2010.2069690.  
- **Key:** `li2010drlse`  ·  **Link:** https://doi.org/10.1109/TIP.2010.2069690
- **Access:** Subscription (institutional access)
- **Where and how it is used:** Ch.4 discussed alternative (DRLSE) for boundary refinement with low curvature penalty; not implemented, cited as future work.
- **Verified:** Crossref record matched (title, year); doi.org resolver HTTP 302

**[67]** Y. Liu, R. T. Collins, and W. E. Rothfus, "Robust midsagittal plane extraction from normal and pathological 3-D neuroradiology images," *IEEE Transactions on Medical Imaging*, vol. 20, no. 3, pp. 175–192, 2001, doi: 10.1109/42.918469.  
- **Key:** `liu2001midsagittal`  ·  **Link:** https://doi.org/10.1109/42.918469
- **Access:** Subscription (institutional access)
- **Where and how it is used:** Ch.4 bilateral-symmetry principle behind the symmetry-based seed (Stage 12).
- **Verified:** Crossref record matched (title, year); doi.org resolver HTTP 302

**[68]** S. X. Liu, "Symmetry and asymmetry analysis and its implications to computer-aided diagnosis: A review of the literature," *Journal of Biomedical Informatics*, vol. 42, no. 6, pp. 1056–1064, 2009, doi: 10.1016/j.jbi.2009.07.003.  
- **Key:** `liu2009symmetry`  ·  **Link:** https://doi.org/10.1016/j.jbi.2009.07.003
- **Access:** Open access (bronze); free copy: https://www.sciencedirect.com/science/article/pii/S1532046409000975/pdf
- **Where and how it is used:** Ch.4 review of asymmetry-based computer-aided diagnosis; supports symmetry seeding.
- **Verified:** Crossref record matched (title, year); doi.org resolver HTTP 302

**[69]** H. Khotanlou, O. Colliot, J. Atif, and I. Bloch, "3D brain tumor segmentation in MRI using fuzzy classification, symmetry analysis and spatially constrained deformable models," *Fuzzy Sets and Systems*, vol. 160, no. 10, pp. 1457–1473, 2009, doi: 10.1016/j.fss.2008.11.016.  
- **Key:** `khotanlou2009symmetry`  ·  **Link:** https://doi.org/10.1016/j.fss.2008.11.016
- **Access:** Open access (green); free copy: https://inria.hal.science/hal-01251278
- **Where and how it is used:** Ch.4 prior use of hemispheric symmetry to localise brain tumours before deformable-model refinement.
- **Verified:** Crossref record matched (title, year); doi.org resolver HTTP 302

**[70]** N. Gordillo, E. Montseny, and P. Sobrevilla, "State of the art survey on MRI brain tumor segmentation," *Magnetic Resonance Imaging*, vol. 31, no. 8, pp. 1426–1438, 2013, doi: 10.1016/j.mri.2013.05.002.  
- **Key:** `gordillo2013survey`  ·  **Link:** https://doi.org/10.1016/j.mri.2013.05.002
- **Access:** Open access (green); free copy: http://hdl.handle.net/2117/19748
- **Where and how it is used:** Ch.2 survey of classical tumour segmentation incl. deformable models.
- **Verified:** Crossref record matched (title, year); doi.org resolver HTTP 302

**[71]** O. Ronneberger, P. Fischer, and T. Brox, "U-Net: Convolutional Networks for Biomedical Image Segmentation," in Medical Image Computing and Computer-Assisted Intervention – MICCAI 2015 (Lecture Notes in Computer Science, vol. 9351), 2015, pp. 234–241, doi: 10.1007/978-3-319-24574-4_28.  
- **Key:** `ronneberger2015unet`  ·  **Link:** https://doi.org/10.1007/978-3-319-24574-4_28
- **Access:** Open access (bronze); free copy: https://link.springer.com/content/pdf/10.1007/978-3-319-24574-4_28.pdf
- **Where and how it is used:** Ch.6 future work: learned localiser as replacement for heuristic seeding.
- **Verified:** Crossref record matched (title, year); doi.org resolver HTTP 302

**[72]** F. Isensee, P. F. Jaeger, S. A. A. Kohl, J. Petersen, and K. H. Maier-Hein, "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation," *Nature Methods*, vol. 18, no. 2, pp. 203–211, 2021, doi: 10.1038/s41592-020-01008-z.  
- **Key:** `isensee2021nnunet`  ·  **Link:** https://doi.org/10.1038/s41592-020-01008-z
- **Access:** Open access (green); free copy: https://arxiv.org/pdf/1904.08128
- **Where and how it is used:** Ch.6 future work: state-of-the-art self-configuring segmentation baseline.
- **Verified:** Crossref record matched (title, year); doi.org resolver HTTP 302

**[73]** L. R. Dice, "Measures of the Amount of Ecologic Association Between Species," *Ecology*, vol. 26, no. 3, pp. 297–302, 1945, doi: 10.2307/1932409.  
- **Key:** `dice1945`  ·  **Link:** https://doi.org/10.2307/1932409
- **Access:** Subscription (institutional access)
- **Where and how it is used:** Ch.4 Dice coefficient used to score automatic vs expert masks (Stage 11: mean 0.034).
- **Verified:** Crossref record matched (title, year); doi.org resolver HTTP 302

**[74]** P. Jaccard, "The distribution of the flora in the alpine zone," *New Phytologist*, vol. 11, no. 2, pp. 37–50, 1912, doi: 10.1111/j.1469-8137.1912.tb05611.x.  
- **Key:** `jaccard1912`  ·  **Link:** https://doi.org/10.1111/j.1469-8137.1912.tb05611.x
- **Access:** Open access (bronze); free copy: https://onlinelibrary.wiley.com/doi/pdfdirect/10.1111/j.1469-8137.1912.tb05611.x
- **Where and how it is used:** Ch.4 Jaccard index (IoU) reported alongside Dice.
- **Verified:** Crossref record matched (title, year); doi.org resolver HTTP 302


## E. Classification, comparators and metrics (Pillar I)

*Classifier, evaluation metrics, nested cross-validation, and the published results we compare against.*

**[75]** L. Breiman, "Random Forests," *Machine Learning*, vol. 45, no. 1, pp. 5–32, 2001, doi: 10.1023/A:1010933404324.  
- **Key:** `breiman2001rf`  ·  **Link:** https://doi.org/10.1023/A:1010933404324
- **Access:** Open access (green); free copy: https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf
- **Where and how it is used:** Ch.3/4 Random Forest (300 trees) classifier of the 16-feature model.
- **Verified:** Crossref record matched (title, year); doi.org resolver HTTP 302

**[76]** K. H. Brodersen, C. S. Ong, K. E. Stephan, and J. M. Buhmann, "The Balanced Accuracy and Its Posterior Distribution," in Proc. 20th Int. Conf. Pattern Recognit. (ICPR), 2010, pp. 3121–3124, doi: 10.1109/ICPR.2010.764.  
- **Key:** `brodersen2010balanced`  ·  **Link:** https://doi.org/10.1109/ICPR.2010.764
- **Access:** Subscription (institutional access)
- **Where and how it is used:** Ch.3 justification of balanced accuracy under class imbalance (708/1426/930).
- **Verified:** Crossref record matched (title, year); doi.org resolver HTTP 302

**[77]** T. Fawcett, "An introduction to ROC analysis," *Pattern Recognition Letters*, vol. 27, no. 8, pp. 861–874, 2006, doi: 10.1016/j.patrec.2005.10.010.  
- **Key:** `fawcett2006roc`  ·  **Link:** https://doi.org/10.1016/j.patrec.2005.10.010
- **Access:** Subscription (institutional access)
- **Where and how it is used:** Ch.4 one-vs-rest ROC AUC of C per tumour class.
- **Verified:** Crossref record matched (title, year); doi.org resolver HTTP 302

**[78]** J. A. Hanley and B. J. McNeil, "The meaning and use of the area under a receiver operating characteristic (ROC) curve," *Radiology*, vol. 143, no. 1, pp. 29–36, 1982, doi: 10.1148/radiology.143.1.7063747.  
- **Key:** `hanley1982auc`  ·  **Link:** https://doi.org/10.1148/radiology.143.1.7063747
- **Access:** Subscription (institutional access)
- **Where and how it is used:** Ch.4 interpretation of AUC values (pituitary 0.907, glioma 0.735, meningioma 0.613).
- **Verified:** Crossref record matched (title, year); doi.org resolver HTTP 302

**[79]** S. Varma and R. Simon, "Bias in error estimation when using cross-validation for model selection," *BMC Bioinformatics*, vol. 7, art. no. 91, 2006, doi: 10.1186/1471-2105-7-91.  
- **Key:** `varma2006cvbias`  ·  **Link:** https://doi.org/10.1186/1471-2105-7-91
- **Access:** Open access (gold); free copy: https://bmcbioinformatics.biomedcentral.com/counter/pdf/10.1186/1471-2105-7-91
- **Where and how it is used:** Ch.3 why hyper-parameters are tuned by an inner GroupKFold (nested CV).
- **Verified:** Crossref record matched (title, year); doi.org resolver HTTP 302

**[80]** G. C. Cawley and N. L. C. Talbot, "On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation," *J. Mach. Learn. Res.*, vol. 11, pp. 2079–2107, 2010. [Online]. Available: https://jmlr.org/papers/v11/cawley10a.html  
- **Key:** `cawley2010overfit`  ·  **Link:** https://jmlr.org/papers/v11/cawley10a.html
- **Access:** Open access (official proceedings / journal site)
- **Where and how it is used:** Ch.3 selection bias from tuning on the test fold; supports nested CV.
- **Verified:** official page HTTP 200; title and citation meta-tags (authors, pages) read from the page

**[81]** M. R. Ismael and I. Abdel-Qader, "Brain Tumor Classification via Statistical Features and Back-Propagation Neural Network," in Proc. IEEE Int. Conf. Electro/Information Technology (EIT), 2018, pp. 252–257, doi: 10.1109/EIT.2018.8500308.  
- **Key:** `ismael2018statistical`  ·  **Link:** https://doi.org/10.1109/EIT.2018.8500308
- **Access:** Subscription (institutional access)
- **Where and how it is used:** Ch.4 hand-crafted-feature comparator on the same dataset (91.9%).
- **Verified:** Crossref record matched (title, year); doi.org resolver HTTP 302

**[82]** A. Gumaei, M. M. Hassan, M. R. Hassan, A. Alelaiwi, and G. Fortino, "A Hybrid Feature Extraction Method With Regularized Extreme Learning Machine for Brain Tumor Classification," *IEEE Access*, vol. 7, pp. 36266–36273, 2019, doi: 10.1109/ACCESS.2019.2904145.  
- **Key:** `gumaei2019hybrid`  ·  **Link:** https://doi.org/10.1109/ACCESS.2019.2904145
- **Access:** Open access (gold); free copy: https://ieeexplore.ieee.org/ielx7/6287639/8600701/08664160.pdf
- **Where and how it is used:** Ch.4 strongest classical comparator (94.23%).
- **Verified:** Crossref record matched (title, year); doi.org resolver HTTP 302

**[83]** J. S. Paul, A. J. Plassard, B. A. Landman, and D. Fabbri, "Deep learning for brain tumor classification," in Proc. SPIE Medical Imaging 2017: Biomedical Applications in Molecular, Structural, and Functional Imaging, vol. 10137, 2017, art. no. 1013710, doi: 10.1117/12.2254195.  
- **Key:** `paul2017deep`  ·  **Link:** https://doi.org/10.1117/12.2254195
- **Access:** Open access (green); free copy: https://pmc.ncbi.nlm.nih.gov/articles/PMC12965214/pdf/nihms-2062562.pdf
- **Where and how it is used:** Ch.4 early CNN comparator on the Figshare data.
- **Verified:** Crossref record matched (title, year); doi.org resolver HTTP 302

**[84]** P. Afshar, K. N. Plataniotis, and A. Mohammadi, "Capsule Networks for Brain Tumor Classification Based on MRI Images and Coarse Tumor Boundaries," in Proc. IEEE Int. Conf. Acoust., Speech Signal Process. (ICASSP), 2019, pp. 1368–1372, doi: 10.1109/ICASSP.2019.8683759.  
- **Key:** `afshar2019capsule`  ·  **Link:** https://doi.org/10.1109/ICASSP.2019.8683759
- **Access:** Subscription (institutional access)
- **Where and how it is used:** Ch.4 CapsNet comparator (90.98% with coarse boundaries).
- **Verified:** Crossref record matched (title, year); doi.org resolver HTTP 302

**[85]** Z. N. K. Swati et al., "Brain tumor classification for MR images using transfer learning and fine-tuning," *Computerized Medical Imaging and Graphics*, vol. 75, pp. 34–46, 2019, doi: 10.1016/j.compmedimag.2019.05.001.  
- **Key:** `swati2019transfer`  ·  **Link:** https://doi.org/10.1016/j.compmedimag.2019.05.001
- **Access:** Subscription (institutional access)
- **Where and how it is used:** Ch.4 transfer-learning comparator with patient-respecting folds (94.82%).
- **Verified:** Crossref record matched (title, year); doi.org resolver HTTP 302

**[86]** S. Deepak and P. M. Ameer, "Brain tumor classification using deep CNN features via transfer learning," *Computers in Biology and Medicine*, vol. 111, art. no. 103345, 2019, doi: 10.1016/j.compbiomed.2019.103345.  
- **Key:** `deepak2019transfer`  ·  **Link:** https://doi.org/10.1016/j.compbiomed.2019.103345
- **Access:** Subscription (institutional access)
- **Where and how it is used:** Ch.4 strongest same-protocol (patient-level) comparator (~98%); reported honestly as higher than ours.
- **Verified:** Crossref record matched (title, year); doi.org resolver HTTP 302

**[87]** H. H. Sultan, N. M. Salem, and W. Al-Atabany, "Multi-Classification of Brain Tumor Images Using Deep Neural Network," *IEEE Access*, vol. 7, pp. 69215–69225, 2019, doi: 10.1109/ACCESS.2019.2919122.  
- **Key:** `sultan2019multi`  ·  **Link:** https://doi.org/10.1109/ACCESS.2019.2919122
- **Access:** Open access (gold); free copy: https://ieeexplore.ieee.org/ielx7/6287639/8600701/08723045.pdf
- **Where and how it is used:** Ch.4 centralized CNN comparator; accuracy ceiling without FL constraints.
- **Verified:** Crossref record matched (title, year); doi.org resolver HTTP 302

**[88]** A. Rehman, S. Naz, M. I. Razzak, F. Akram, and M. Imran, "A Deep Learning-Based Framework for Automatic Brain Tumors Classification Using Transfer Learning," *Circuits, Systems, and Signal Processing*, vol. 39, no. 2, pp. 757–775, 2020, doi: 10.1007/s00034-019-01246-3.  
- **Key:** `rehman2020framework`  ·  **Link:** https://doi.org/10.1007/s00034-019-01246-3
- **Access:** Open access (green); free copy: https://figshare.com/articles/journal_contribution/A_deep_learning-based_framework_for_automatic_brain_tumors_classification_using_transfer_learning/20728501
- **Where and how it is used:** Ch.4 transfer-learning comparator (AlexNet/GoogLeNet/VGG).
- **Verified:** Crossref record matched (title, year); doi.org resolver HTTP 302

**[89]** M. M. Badža and M. Č. Barjaktarović, "Classification of Brain Tumors from MRI Images Using a Convolutional Neural Network," *Applied Sciences*, vol. 10, no. 6, art. no. 1999, 2020, doi: 10.3390/app10061999.  
- **Key:** `badza2020cnn`  ·  **Link:** https://doi.org/10.3390/app10061999
- **Access:** Open access (gold); free copy: https://www.mdpi.com/2076-3417/10/6/1999/pdf?version=1584256135
- **Where and how it is used:** Ch.4 CNN comparator reporting both record-wise and subject-wise splits (evidence for protocol sensitivity).
- **Verified:** Crossref record matched (title, year); doi.org resolver HTTP 302

**[90]** F. J. Díaz-Pernas, M. Martínez-Zarzuela, M. Antón-Rodríguez, and D. González-Ortega, "A Deep Learning Approach for Brain Tumor Classification and Segmentation Using a Multiscale Convolutional Neural Network," *Healthcare*, vol. 9, no. 2, art. no. 153, 2021, doi: 10.3390/healthcare9020153.  
- **Key:** `diazpernas2021multiscale`  ·  **Link:** https://doi.org/10.3390/healthcare9020153
- **Access:** Open access (gold); free copy: https://www.mdpi.com/2227-9032/9/2/153/pdf?version=1613629848
- **Where and how it is used:** Ch.4 multiscale CNN comparator (centralized ceiling).
- **Verified:** Crossref record matched (title, year); doi.org resolver HTTP 302

**[91]** M. Islam, M. T. Reza, M. Kaosar, and M. Z. Parvez, "Effectiveness of Federated Learning and CNN Ensemble Architectures for Identifying Brain Tumors Using MRI Images," *Neural Processing Letters*, vol. 55, no. 4, pp. 3779–3809, 2023, doi: 10.1007/s11063-022-11014-1.  
- **Key:** `islam2022ensemble`  ·  **Link:** https://doi.org/10.1007/s11063-022-11014-1
- **Access:** Open access (bronze); free copy: https://link.springer.com/content/pdf/10.1007/s11063-022-11014-1.pdf
- **Where and how it is used:** Ch.2 prior FL brain-tumour classification assuming balanced clients.
- **Verified:** Crossref record matched (title, year); doi.org resolver HTTP 302

**[92]** E. Albalawi et al., "Integrated approach of federated learning with transfer learning for classification and diagnosis of brain tumor," *BMC Medical Imaging*, vol. 24, art. no. 110, 2024, doi: 10.1186/s12880-024-01261-0.  
- **Key:** `albalawi2024fltl`  ·  **Link:** https://doi.org/10.1186/s12880-024-01261-0
- **Access:** Open access (gold)
- **Where and how it is used:** Ch.2 prior FL+VGG16 brain-tumour classification without label-skew analysis.
- **Verified:** Crossref record matched (title, year); doi.org resolver HTTP 302


## F. Federated learning algorithms (Pillar II)

*FedAvg, SCAFFOLD, FedProx, the restricted-softmax family, and the personalization/clustering work we tested and rejected.*

**[93]** H. B. McMahan, E. Moore, D. Ramage, S. Hampson, and B. Agüera y Arcas, "Communication-Efficient Learning of Deep Networks from Decentralized Data," in Proc. 20th Int. Conf. Artif. Intell. Statist. (AISTATS), PMLR, vol. 54, 2017, pp. 1273–1282. [Online]. Available: https://proceedings.mlr.press/v54/mcmahan17a.html  
- **Key:** `mcmahan2017fedavg`  ·  **Link:** https://proceedings.mlr.press/v54/mcmahan17a.html
- **Access:** Open access (official proceedings / journal site)
- **Where and how it is used:** Ch.5 FedAvg: the aggregation rule of the 7 original ablation rows and the baseline of every comparison.
- **Verified:** official page HTTP 200; title and citation meta-tags (authors, pages) read from the page

**[94]** S. P. Karimireddy, S. Kale, M. Mohri, S. Reddi, S. Stich, and A. T. Suresh, "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning," in Proc. 37th Int. Conf. Mach. Learn. (ICML), PMLR, vol. 119, 2020, pp. 5132–5143. [Online]. Available: https://proceedings.mlr.press/v119/karimireddy20a.html  
- **Key:** `karimireddy2020scaffold`  ·  **Link:** https://proceedings.mlr.press/v119/karimireddy20a.html
- **Access:** Open access (official proceedings / journal site)
- **Where and how it is used:** Ch.5 SCAFFOLD control variates (Option II) implemented in Stage 13 and the 8th ablation row; indistinguishable from FedAvg under restricted-softmax (d=0.21, p=0.546).
- **Verified:** official page HTTP 200; title and citation meta-tags (authors, pages) read from the page

**[95]** T. Li, A. K. Sahu, M. Zaheer, M. Sanjabi, A. Talwalkar, and V. Smith, "Federated Optimization in Heterogeneous Networks," in Proc. Mach. Learn. Syst. (MLSys), vol. 2, 2020, pp. 429–450. [Online]. Available: https://proceedings.mlsys.org/paper_files/paper/2020/hash/1f5fe83998a09396ebe6477d9475ba0c-Abstract.html  
- **Key:** `li2020fedprox`  ·  **Link:** https://proceedings.mlsys.org/paper_files/paper/2020/hash/1f5fe83998a09396ebe6477d9475ba0c-Abstract.html
- **Access:** Open access (official proceedings / journal site)
- **Where and how it is used:** Ch.5 FedProx proximal term (mu=0.01) tested; adds +0.1-0.2 pp (null).
- **Verified:** official page HTTP 200; title and citation meta-tags (authors, pages) read from the page

**[96]** J. Wang, Q. Liu, H. Liang, G. Joshi, and H. V. Poor, "Tackling the Objective Inconsistency Problem in Heterogeneous Federated Optimization," in Adv. Neural Inf. Process. Syst. (NeurIPS), vol. 33, 2020, pp. 7611–7623. [Online]. Available: https://proceedings.neurips.cc/paper/2020/hash/564127c03caab942e503ee6f810f54fd-Abstract.html  
- **Key:** `wang2020fednova`  ·  **Link:** https://proceedings.neurips.cc/paper/2020/hash/564127c03caab942e503ee6f810f54fd-Abstract.html
- **Access:** Open access (official proceedings / journal site)
- **Where and how it is used:** Ch.2 FedNova normalized averaging; related drift correction.
- **Verified:** official page HTTP 200; title and citation meta-tags (authors, pages) read from the page

**[97]** D. A. E. Acar, Y. Zhao, R. M. Navarro, M. Mattina, P. N. Whatmough, and V. Saligrama, "Federated Learning Based on Dynamic Regularization," in Proc. Int. Conf. Learn. Represent. (ICLR), 2021. [Online]. Available: https://openreview.net/forum?id=B7v4QMR6Z9w  
- **Key:** `acar2021feddyn`  ·  **Link:** https://openreview.net/forum?id=B7v4QMR6Z9w
- **Access:** Open access (arXiv / proceedings); free copy: https://arxiv.org/abs/2111.04263
- **Where and how it is used:** Ch.2 FedDyn; related client-drift correction.
- **Verified:** arXiv API record matched (title); abs page HTTP 200; peer-reviewed venue confirmed on OpenReview (ICLR 2021 Oral)

**[98]** Q. Li, B. He, and D. Song, "Model-Contrastive Federated Learning," in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2021, pp. 10708–10717, doi: 10.1109/CVPR46437.2021.01057.  
- **Key:** `li2021moon`  ·  **Link:** https://doi.org/10.1109/CVPR46437.2021.01057
- **Access:** Subscription (institutional access)
- **Where and how it is used:** Ch.2 MOON; representation-level correction; related to the Stage 7 contrastive null.
- **Verified:** Crossref record matched (title, year); doi.org resolver HTTP 302

**[99]** S. Reddi et al., "Adaptive Federated Optimization," in Proc. Int. Conf. Learn. Represent. (ICLR), 2021. [Online]. Available: https://openreview.net/forum?id=rOWz4RF7nJq  
- **Key:** `reddi2021adaptive`  ·  **Link:** https://openreview.net/forum?id=rOWz4RF7nJq
- **Access:** Open access (arXiv / proceedings); free copy: https://arxiv.org/abs/2003.00295
- **Where and how it is used:** Ch.2 server-side adaptive optimisers (FedAdam/FedYogi); related work.
- **Verified:** arXiv API record matched (title); abs page HTTP 200; peer-reviewed venue confirmed on OpenReview (ICLR 2021)

**[100]** X.-C. Li and D.-C. Zhan, "FedRS: Federated learning with restricted softmax for label distribution non-IID data," in Proc. 27th ACM SIGKDD Conf. Knowl. Discovery Data Mining (KDD), 2021, pp. 995–1005, doi: 10.1145/3447548.3467254.  
- **Key:** `li2021fedrs`  ·  **Link:** https://doi.org/10.1145/3447548.3467254
- **Access:** Subscription (institutional access)
- **Where and how it is used:** Ch.5 direct ancestor of the class-presence mask; alpha in FedRS is the axis swept in Stage 6; acknowledged that FedRS already reported alpha=0.
- **Verified:** Crossref record matched (title, year); doi.org resolver HTTP 302. Note: ACM Crossref record stores only the short title 'FedRS'; authors (Li, Zhan), venue (KDD '21) and pages 995-1005 match exactly.

**[101]** J. Zhang et al., "Federated Learning with Label Distribution Skew via Logits Calibration," in Proc. 39th Int. Conf. Mach. Learn. (ICML), PMLR, vol. 162, 2022, pp. 26311–26329. [Online]. Available: https://proceedings.mlr.press/v162/zhang22p.html  
- **Key:** `zhang2022fedlc`  ·  **Link:** https://proceedings.mlr.press/v162/zhang22p.html
- **Access:** Open access (official proceedings / journal site)
- **Where and how it is used:** Ch.5 FedLC comparator (0.653 vs HardMask 0.649, Holm p=0.123, not significant).
- **Verified:** official page HTTP 200; title and citation meta-tags (authors, pages) read from the page

**[102]** K. Guo, Y. Ding, J. Liang, Z. Wang, R. He, and T. Tan, "Exploring Vacant Classes in Label-Skewed Federated Learning," in Proc. AAAI Conf. Artif. Intell., vol. 39, no. 16, 2025, pp. 16960–16968, doi: 10.1609/aaai.v39i16.33864.  
- **Key:** `guo2025fedvls`  ·  **Link:** https://doi.org/10.1609/aaai.v39i16.33864
- **Access:** Open access (diamond); free copy: https://ojs.aaai.org/index.php/AAAI/article/download/33864/36019
- **Where and how it is used:** Ch.2/5 vacant-class distillation + logit suppression; closest recent competitor.
- **Verified:** Crossref record matched (title, year); doi.org resolver HTTP 302

**[103]** J. Lu, S. Li, K. Bao, P. Wang, Z. Qian, and S. Ge, "Federated Learning with Label-Masking Distillation," in Proc. 31st ACM Int. Conf. Multimedia (MM), 2023, pp. 222–232, doi: 10.1145/3581783.3611984.  
- **Key:** `lu2023labelmask`  ·  **Link:** https://doi.org/10.1145/3581783.3611984
- **Access:** Open access (gold); free copy: https://dl.acm.org/doi/pdf/10.1145/3581783.3611984
- **Where and how it is used:** Ch.2 label-masking distillation; related missing-class correction.
- **Verified:** Crossref record matched (title, year); doi.org resolver HTTP 302

**[104]** H. Wang, P. Zheng, X. Han, W. Xu, R. Li, and T. Zhang, "FedNLR: Federated Learning with Neuron-wise Learning Rates," in Proc. 30th ACM SIGKDD Conf. Knowl. Discovery Data Mining (KDD), 2024, pp. 3069–3080, doi: 10.1145/3637528.3672042.  
- **Key:** `wang2024fednlr`  ·  **Link:** https://doi.org/10.1145/3637528.3672042
- **Access:** Open access (gold); free copy: https://dl.acm.org/doi/pdf/10.1145/3637528.3672042
- **Where and how it is used:** Ch.2 neuron-wise learning rates motivated by neuron drift of absent classes.
- **Verified:** Crossref record matched (title, year); doi.org resolver HTTP 302

**[105]** M. Luo, F. Chen, D. Hu, Y. Zhang, J. Liang, and J. Feng, "No Fear of Heterogeneity: Classifier Calibration for Federated Learning with Non-IID Data," in Adv. Neural Inf. Process. Syst. (NeurIPS), vol. 34, 2021, pp. 5972–5984. [Online]. Available: https://proceedings.neurips.cc/paper/2021/hash/2f2b265625d76a6704b08093c652fd79-Abstract.html  
- **Key:** `luo2021ccvr`  ·  **Link:** https://proceedings.neurips.cc/paper/2021/hash/2f2b265625d76a6704b08093c652fd79-Abstract.html
- **Access:** Open access (official proceedings / journal site)
- **Where and how it is used:** Ch.2/5 CCVR: shows the classifier layer is most biased under non-IID; supports the output-layer diagnosis.
- **Verified:** official page HTTP 200; title and citation meta-tags (authors, pages) read from the page

**[106]** A. K. Menon, S. Jayasumana, A. S. Rawat, H. Jain, A. Veit, and S. Kumar, "Long-tail learning via logit adjustment," in Proc. Int. Conf. Learn. Represent. (ICLR), 2021. [Online]. Available: https://openreview.net/forum?id=37nvvqkCo5  
- **Key:** `menon2021logit`  ·  **Link:** https://openreview.net/forum?id=37nvvqkCo5
- **Access:** Open access (arXiv / proceedings); free copy: https://arxiv.org/abs/2007.07314
- **Where and how it is used:** Ch.5 logit adjustment theory underlying FedLC's calibration term.
- **Verified:** arXiv API record matched (title); abs page HTTP 200; peer-reviewed venue confirmed on OpenReview (ICLR 2021 Spotlight)

**[107]** A. Ghosh, J. Chung, D. Yin, and K. Ramchandran, "An Efficient Framework for Clustered Federated Learning," *IEEE Transactions on Information Theory*, vol. 68, no. 12, pp. 8076–8091, 2022, doi: 10.1109/TIT.2022.3192506.  
- **Key:** `ghosh2022ifca`  ·  **Link:** https://doi.org/10.1109/TIT.2022.3192506
- **Access:** Open access (green); free copy: https://arxiv.org/pdf/2006.04088
- **Where and how it is used:** Ch.5 IFCA clustered FL tested and withdrawn (+12.5 pp collapsed to +1.2 pp, p=0.88 over 20 seeds).
- **Verified:** Crossref record matched (title, year); doi.org resolver HTTP 302

**[108]** F. Sattler, K.-R. Müller, and W. Samek, "Clustered Federated Learning: Model-Agnostic Distributed Multitask Optimization Under Privacy Constraints," *IEEE Transactions on Neural Networks and Learning Systems*, vol. 32, no. 8, pp. 3710–3722, 2021, doi: 10.1109/TNNLS.2020.3015958.  
- **Key:** `sattler2021cfl`  ·  **Link:** https://doi.org/10.1109/TNNLS.2020.3015958
- **Access:** Open access (hybrid); free copy: https://ieeexplore.ieee.org/ielx7/5962385/9505270/09174890.pdf
- **Where and how it is used:** Ch.5 alternative clustered-FL formulation; context for the withdrawn clustering result.
- **Verified:** Crossref record matched (title, year); doi.org resolver HTTP 302

**[109]** T. Li, S. Hu, A. Beirami, and V. Smith, "Ditto: Fair and Robust Federated Learning Through Personalization," in Proc. 38th Int. Conf. Mach. Learn. (ICML), PMLR, vol. 139, 2021, pp. 6357–6368. [Online]. Available: https://proceedings.mlr.press/v139/li21h.html  
- **Key:** `li2021ditto`  ·  **Link:** https://proceedings.mlr.press/v139/li21h.html
- **Access:** Open access (official proceedings / journal site)
- **Where and how it is used:** Ch.5 personalized FL context for the leakage guard.
- **Verified:** official page HTTP 200; title and citation meta-tags (authors, pages) read from the page

**[110]** L. Collins, H. Hassani, A. Mokhtari, and S. Shakkottai, "Exploiting Shared Representations for Personalized Federated Learning," in Proc. 38th Int. Conf. Mach. Learn. (ICML), PMLR, vol. 139, 2021, pp. 2089–2099. [Online]. Available: https://proceedings.mlr.press/v139/collins21a.html  
- **Key:** `collins2021fedrep`  ·  **Link:** https://proceedings.mlr.press/v139/collins21a.html
- **Access:** Open access (official proceedings / journal site)
- **Where and how it is used:** Ch.5 FedRep personalization context.
- **Verified:** official page HTTP 200; title and citation meta-tags (authors, pages) read from the page

**[111]** S. Wu et al., "Motley: Benchmarking Heterogeneity and Personalization in Federated Learning," arXiv preprint arXiv:2206.09262, 2022.  
- **Key:** `wu2022motley`  ·  **Link:** https://arxiv.org/abs/2206.09262
- **Access:** Open access (arXiv / proceedings)
- **Where and how it is used:** Ch.5 support/query personalization protocol whose label leakage on per-patient data the leakage guard exposes (image-blind predictor = 1.000).
- **Verified:** arXiv API record matched (title); abs page HTTP 200; no peer-reviewed version found, cited as preprint

**[112]** D. Chen, D. Gao, W. Kuang, Y. Li, and B. Ding, "pFL-Bench: A Comprehensive Benchmark for Personalized Federated Learning," in Adv. Neural Inf. Process. Syst. (NeurIPS), Datasets and Benchmarks Track, vol. 35, 2022, pp. 9344–9360. [Online]. Available: https://proceedings.neurips.cc/paper_files/paper/2022/hash/3cc03e19fed71a2b9347d83921ca2e7d-Abstract-Datasets_and_Benchmarks.html  
- **Key:** `chen2022pflbench`  ·  **Link:** https://proceedings.neurips.cc/paper_files/paper/2022/hash/3cc03e19fed71a2b9347d83921ca2e7d-Abstract-Datasets_and_Benchmarks.html
- **Access:** Open access (arXiv / proceedings); free copy: https://arxiv.org/abs/2206.03655
- **Where and how it is used:** Ch.5 personalized-FL benchmark; context for rejecting per-patient personalization.
- **Verified:** arXiv API record matched (title); abs page HTTP 200; venue confirmed in the official proceedings listing

**[113]** M. J. Sheller, G. A. Reina, B. Edwards, J. Martin, and S. Bakas, "Multi-institutional Deep Learning Modeling Without Sharing Patient Data: A Feasibility Study on Brain Tumor Segmentation," in Brainlesion: Glioma, Multiple Sclerosis, Stroke and Traumatic Brain Injuries (BrainLes 2018) (Lecture Notes in Computer Science, vol. 11383), 2019, pp. 92–104, doi: 10.1007/978-3-030-11723-8_9.  
- **Key:** `sheller2019brats`  ·  **Link:** https://doi.org/10.1007/978-3-030-11723-8_9
- **Access:** Open access (green); free copy: https://www.ncbi.nlm.nih.gov/pmc/articles/6589345
- **Where and how it is used:** Ch.2 first FL brain-tumour segmentation feasibility study.
- **Verified:** Crossref record matched (title, year); doi.org resolver HTTP 302

**[114]** I. Dayan et al., "Federated learning for predicting clinical outcomes in patients with COVID-19," *Nature Medicine*, vol. 27, no. 10, pp. 1735–1743, 2021, doi: 10.1038/s41591-021-01506-3.  
- **Key:** `dayan2021exam`  ·  **Link:** https://doi.org/10.1038/s41591-021-01506-3
- **Access:** Open access (hybrid); free copy: https://www.nature.com/articles/s41591-021-01506-3.pdf
- **Where and how it is used:** Ch.1 large clinical FL deployment; evidence FL is practical across hospitals.
- **Verified:** Crossref record matched (title, year); doi.org resolver HTTP 302

**[115]** J. Lyu, K. Zhou, and Y. Zhong, "A statistical theory of overfitting for imbalanced classification," arXiv preprint arXiv:2502.11323, 2025.  
- **Key:** `lyu2025imbalanced`  ·  **Link:** https://arxiv.org/abs/2502.11323
- **Access:** Open access (arXiv / proceedings)
- **Where and how it is used:** Ch.5 theory of logit/margin behaviour for minority classes in imbalanced classification; partially anticipates the gradient argument for absent classes, cited to bound that claim.
- **Verified:** arXiv API record matched (title); abs page HTTP 200; no peer-reviewed version found, cited as preprint


## G. Normalisation, optimisation and class imbalance (Pillar II)

*GroupNorm vs BatchNorm, and every optimisation intervention in the eight-row ablation.*

**[116]** Y. Wu and K. He, "Group Normalization," in Computer Vision – ECCV 2018 (Lecture Notes in Computer Science, vol. 11217), 2018, pp. 3–19, doi: 10.1007/978-3-030-01261-8_1.  
- **Key:** `wu2018groupnorm`  ·  **Link:** https://doi.org/10.1007/978-3-030-01261-8_1
- **Access:** Open access (green); free copy: https://arxiv.org/pdf/1803.08494
- **Where and how it is used:** Ch.5 GroupNorm used in SmallNet in place of BatchNorm.
- **Verified:** Crossref record matched (title, year); doi.org resolver HTTP 302

**[117]** S. Ioffe and C. Szegedy, "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift," in Proc. 32nd Int. Conf. Mach. Learn. (ICML), PMLR, vol. 37, 2015, pp. 448–456. [Online]. Available: https://proceedings.mlr.press/v37/ioffe15.html  
- **Key:** `ioffe2015bn`  ·  **Link:** https://proceedings.mlr.press/v37/ioffe15.html
- **Access:** Open access (official proceedings / journal site)
- **Where and how it is used:** Ch.5 BatchNorm, whose batch statistics break under non-IID clients (reason it is avoided).
- **Verified:** official page HTTP 200; title and citation meta-tags (authors, pages) read from the page

**[118]** K. Hsieh, A. Phanishayee, O. Mutlu, and P. Gibbons, "The Non-IID Data Quagmire of Decentralized Machine Learning," in Proc. 37th Int. Conf. Mach. Learn. (ICML), PMLR, vol. 119, 2020, pp. 4387–4398. [Online]. Available: https://proceedings.mlr.press/v119/hsieh20a.html  
- **Key:** `hsieh2020quagmire`  ·  **Link:** https://proceedings.mlr.press/v119/hsieh20a.html
- **Access:** Open access (official proceedings / journal site)
- **Where and how it is used:** Ch.5 evidence that BatchNorm fails and GroupNorm helps under non-IID data.
- **Verified:** official page HTTP 200; title and citation meta-tags (authors, pages) read from the page

**[119]** X. Li, M. Jiang, X. Zhang, M. Kamp, and Q. Dou, "FedBN: Federated Learning on Non-IID Features via Local Batch Normalization," in Proc. Int. Conf. Learn. Represent. (ICLR), 2021. [Online]. Available: https://openreview.net/forum?id=6YEQUn0QICG  
- **Key:** `li2021fedbn`  ·  **Link:** https://openreview.net/forum?id=6YEQUn0QICG
- **Access:** Open access (arXiv / proceedings); free copy: https://arxiv.org/abs/2102.07623
- **Where and how it is used:** Ch.5 local BatchNorm for feature skew; contrasted with our label-skew setting.
- **Verified:** arXiv API record matched (title); abs page HTTP 200; peer-reviewed venue confirmed on OpenReview (ICLR 2021 Poster)

**[120]** B. Casella, R. Esposito, A. Sciarappa, C. Cavazzoni, and M. Aldinucci, "Experimenting With Normalization Layers in Federated Learning on Non-IID Scenarios," *IEEE Access*, vol. 12, pp. 47961–47971, 2024, doi: 10.1109/ACCESS.2024.3383783.  
- **Key:** `casella2024norm`  ·  **Link:** https://doi.org/10.1109/ACCESS.2024.3383783
- **Access:** Open access (gold); free copy: https://ieeexplore.ieee.org/ielx7/6287639/6514899/10486903.pdf
- **Where and how it is used:** Ch.5 recent empirical comparison of normalisation layers in non-IID FL.
- **Verified:** Crossref record matched (title, year); doi.org resolver HTTP 302

**[121]** R. Pascanu, T. Mikolov, and Y. Bengio, "On the difficulty of training recurrent neural networks," in Proc. 30th Int. Conf. Mach. Learn. (ICML), PMLR, vol. 28, 2013, pp. 1310–1318. [Online]. Available: https://proceedings.mlr.press/v28/pascanu13.html  
- **Key:** `pascanu2013clip`  ·  **Link:** https://proceedings.mlr.press/v28/pascanu13.html
- **Access:** Open access (official proceedings / journal site)
- **Where and how it is used:** Ch.5 gradient-norm clipping (max_norm=1.0) tested in the ablation; hurts in isolation.
- **Verified:** official page HTTP 200; title and citation meta-tags (authors, pages) read from the page

**[122]** I. Loshchilov and F. Hutter, "SGDR: Stochastic Gradient Descent with Warm Restarts," in Proc. Int. Conf. Learn. Represent. (ICLR), 2017. [Online]. Available: https://openreview.net/forum?id=fdWX2wmN5D  
- **Key:** `loshchilov2017sgdr`  ·  **Link:** https://openreview.net/forum?id=fdWX2wmN5D
- **Access:** Open access (arXiv / proceedings); free copy: https://arxiv.org/abs/1608.03983
- **Where and how it is used:** Ch.5 cosine learning-rate schedule tested (+warmup_cosine row, 16/20 collapses).
- **Verified:** arXiv API record matched (title); abs page HTTP 200; peer-reviewed venue confirmed on OpenReview (ICLR (Poster) 2017)

**[123]** P. Goyal et al., "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour," arXiv preprint arXiv:1706.02677, 2017.  
- **Key:** `goyal2017warmup`  ·  **Link:** https://arxiv.org/abs/1706.02677
- **Access:** Open access (arXiv / proceedings)
- **Where and how it is used:** Ch.5 linear warm-up schedule combined with cosine decay in the ablation.
- **Verified:** arXiv API record matched (title); abs page HTTP 200; no peer-reviewed version found, cited as preprint

**[124]** M. Buda, A. Maki, and M. A. Mazurowski, "A systematic study of the class imbalance problem in convolutional neural networks," *Neural Networks*, vol. 106, pp. 249–259, 2018, doi: 10.1016/j.neunet.2018.07.011.  
- **Key:** `buda2018imbalance`  ·  **Link:** https://doi.org/10.1016/j.neunet.2018.07.011
- **Access:** Open access (green); free copy: https://arxiv.org/pdf/1710.05381
- **Where and how it is used:** Ch.5 oversampling as imbalance remedy; basis of the balanced-sampling row (0.517, does not fix collapse).
- **Verified:** Crossref record matched (title, year); doi.org resolver HTTP 302

**[125]** Y. Cui, M. Jia, T.-Y. Lin, Y. Song, and S. Belongie, "Class-Balanced Loss Based on Effective Number of Samples," in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2019, pp. 9260–9269, doi: 10.1109/CVPR.2019.00949.  
- **Key:** `cui2019classbalanced`  ·  **Link:** https://doi.org/10.1109/CVPR.2019.00949
- **Access:** Subscription (institutional access)
- **Where and how it is used:** Ch.5 re-weighting alternative to balanced sampling; related work.
- **Verified:** Crossref record matched (title, year); doi.org resolver HTTP 302


## H. Transfer learning and the frozen backbone (Pillar II, Stages 7, 9, 9B)

*Pre-trained ResNet-18, linear probing, contrastive learning, and prior work that bounds the frozen-backbone claim.*

**[126]** K. He, X. Zhang, S. Ren, and J. Sun, "Deep Residual Learning for Image Recognition," in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2016, pp. 770–778, doi: 10.1109/CVPR.2016.90.  
- **Key:** `he2016resnet`  ·  **Link:** https://doi.org/10.1109/CVPR.2016.90
- **Access:** Open access (green); free copy: https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html
- **Where and how it is used:** Ch.5 ResNet-18 backbone of the frozen-feature experiment.
- **Verified:** Crossref record matched (title, year); doi.org resolver HTTP 302

**[127]** J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei, "ImageNet: A large-scale hierarchical image database," in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2009, pp. 248–255, doi: 10.1109/CVPR.2009.5206848.  
- **Key:** `deng2009imagenet`  ·  **Link:** https://doi.org/10.1109/CVPR.2009.5206848
- **Access:** Subscription (institutional access)
- **Where and how it is used:** Ch.5 pre-training dataset of the frozen ResNet-18.
- **Verified:** Crossref record matched (title, year); doi.org resolver HTTP 302

**[128]** J. Yosinski, J. Clune, Y. Bengio, and H. Lipson, "How transferable are features in deep neural networks?," in Adv. Neural Inf. Process. Syst. (NeurIPS), vol. 27, 2014. [Online]. Available: https://proceedings.neurips.cc/paper_files/paper/2014/hash/532a2f85b6977104bc93f8580abbb330-Abstract.html  
- **Key:** `yosinski2014transferable`  ·  **Link:** https://proceedings.neurips.cc/paper_files/paper/2014/hash/532a2f85b6977104bc93f8580abbb330-Abstract.html
- **Access:** Open access (arXiv / proceedings); free copy: https://arxiv.org/abs/1411.1792
- **Where and how it is used:** Ch.5 why early ImageNet features transfer to MRI.
- **Verified:** arXiv API record matched (title); abs page HTTP 200; venue confirmed in the official proceedings listing

**[129]** S. Kornblith, J. Shlens, and Q. V. Le, "Do Better ImageNet Models Transfer Better?," in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2019, pp. 2656–2666, doi: 10.1109/CVPR.2019.00277.  
- **Key:** `kornblith2019transfer`  ·  **Link:** https://doi.org/10.1109/CVPR.2019.00277
- **Access:** Subscription (institutional access)
- **Where and how it is used:** Ch.5 fixed-feature (linear-probe) transfer protocol used in Stage 9.
- **Verified:** Crossref record matched (title, year); doi.org resolver HTTP 302

**[130]** M. Raghu, C. Zhang, J. Kleinberg, and S. Bengio, "Transfusion: Understanding Transfer Learning for Medical Imaging," in Adv. Neural Inf. Process. Syst. (NeurIPS), vol. 32, 2019. [Online]. Available: https://proceedings.neurips.cc/paper_files/paper/2019/hash/eb1e78328c46506b46a4ac4a1e378b91-Abstract.html  
- **Key:** `raghu2019transfusion`  ·  **Link:** https://proceedings.neurips.cc/paper_files/paper/2019/hash/eb1e78328c46506b46a4ac4a1e378b91-Abstract.html
- **Access:** Open access (arXiv / proceedings); free copy: https://arxiv.org/abs/1902.07208
- **Where and how it is used:** Ch.5 limits of ImageNet transfer in medical imaging; explains modest frozen-backbone gains.
- **Verified:** arXiv API record matched (title); abs page HTTP 200; venue confirmed in the official proceedings listing

**[131]** A. Kumar, A. Raghunathan, R. Jones, T. Ma, and P. Liang, "Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution," in Proc. Int. Conf. Learn. Represent. (ICLR), 2022. [Online]. Available: https://openreview.net/forum?id=UYneFzXSJWh  
- **Key:** `kumar2022distort`  ·  **Link:** https://openreview.net/forum?id=UYneFzXSJWh
- **Access:** Open access (arXiv / proceedings); free copy: https://arxiv.org/abs/2202.10054
- **Where and how it is used:** Ch.5 linear probing vs fine-tuning trade-off; justifies freezing the backbone.
- **Verified:** arXiv API record matched (title); abs page HTTP 200; peer-reviewed venue confirmed on OpenReview (ICLR 2022 Oral)

**[132]** T. Chen, S. Kornblith, M. Norouzi, and G. Hinton, "A Simple Framework for Contrastive Learning of Visual Representations," in Proc. 37th Int. Conf. Mach. Learn. (ICML), PMLR, vol. 119, 2020, pp. 1597–1607. [Online]. Available: https://proceedings.mlr.press/v119/chen20j.html  
- **Key:** `chen2020simclr`  ·  **Link:** https://proceedings.mlr.press/v119/chen20j.html
- **Access:** Open access (official proceedings / journal site)
- **Where and how it is used:** Ch.5 linear-evaluation protocol and contrastive loss used in the Stage 7 bio-contrastive test.
- **Verified:** official page HTTP 200; title and citation meta-tags (authors, pages) read from the page

**[133]** P. Khosla et al., "Supervised Contrastive Learning," in Adv. Neural Inf. Process. Syst. (NeurIPS), vol. 33, 2020, pp. 18661–18673. [Online]. Available: https://proceedings.neurips.cc/paper_files/paper/2020/hash/d89a66c7c80a29b1bdbab0f2a1a94af8-Abstract.html  
- **Key:** `khosla2020supcon`  ·  **Link:** https://proceedings.neurips.cc/paper_files/paper/2020/hash/d89a66c7c80a29b1bdbab0f2a1a94af8-Abstract.html
- **Access:** Open access (arXiv / proceedings); free copy: https://arxiv.org/abs/2004.11362
- **Where and how it is used:** Ch.5 supervised contrastive loss adapted in Stage 7 (preregistered criteria not met).
- **Verified:** arXiv API record matched (title); abs page HTTP 200; venue confirmed in the official proceedings listing

**[134]** J. Nguyen, J. Wang, K. Malik, M. Sanjabi, and M. Rabbat, "Where to Begin? On the Impact of Pre-Training and Initialization in Federated Learning," in Proc. Int. Conf. Learn. Represent. (ICLR), 2023. [Online]. Available: https://openreview.net/forum?id=Mpa3tRJFBb  
- **Key:** `nguyen2023wheretobegin`  ·  **Link:** https://openreview.net/forum?id=Mpa3tRJFBb
- **Access:** Open access (arXiv / proceedings); free copy: https://arxiv.org/abs/2206.15387
- **Where and how it is used:** Ch.5 pre-trained initialisation reduces the damage of heterogeneity in FL; explains the shrinking gain (+2.2 pp) under a frozen backbone.
- **Verified:** arXiv API record matched (title); abs page HTTP 200; peer-reviewed venue confirmed on OpenReview (ICLR 2023 notable top 25%)

**[135]** H.-Y. Chen, C.-H. Tu, Z. Li, H.-W. Shen, and W.-L. Chao, "On the Importance and Applicability of Pre-Training for Federated Learning," in Proc. Int. Conf. Learn. Represent. (ICLR), 2023. [Online]. Available: https://openreview.net/forum?id=5f04jup4aHa  
- **Key:** `chen2023pretrainfl`  ·  **Link:** https://openreview.net/forum?id=5f04jup4aHa
- **Access:** Open access (arXiv / proceedings); free copy: https://arxiv.org/abs/2206.11488
- **Where and how it is used:** Ch.5 same finding from a second group; supports the frozen-backbone interpretation.
- **Verified:** arXiv API record matched (title); abs page HTTP 200; peer-reviewed venue confirmed on OpenReview (ICLR 2023)

**[136]** D. Park, H. Yeo, and J.-W. Lee, "OvA-LP: A Simple and Efficient Framework for Federated Learning on Non-IID Data," arXiv preprint arXiv:2511.05028, 2025.  
- **Key:** `park2025ovalp`  ·  **Link:** https://arxiv.org/abs/2511.05028
- **Access:** Open access (arXiv / proceedings)
- **Where and how it is used:** Ch.5 one-vs-all linear probing on a frozen backbone for non-IID FL; anticipates the frozen-backbone observation (Stage 9), so it is cited to bound that claim's novelty.
- **Verified:** arXiv API record matched (title); abs page HTTP 200; no peer-reviewed version found, cited as preprint

**[137]** M. Haseeb, S. Masood, M. A. Sohail, M. F. Shoaib, and M. Tahir, "Mechanistic Evidence for Preserved-but-Misaligned Representations in Non-IID FedAvg," arXiv preprint arXiv:2512.23043, 2025.  
- **Key:** `haseeb2025mechanistic`  ·  **Link:** https://arxiv.org/abs/2512.23043
- **Access:** Open access (arXiv / proceedings)
- **Where and how it is used:** Ch.5 shows non-IID FedAvg keeps representations largely intact while the classifier misaligns; independent support for the output-layer diagnosis and for the shrinking gain under a frozen backbone.
- **Verified:** arXiv API record matched (title); abs page HTTP 200; no peer-reviewed version found, cited as preprint


## I. Statistical methodology (both pillars)

*Every test, interval, effect size and protocol decision used in the results chapters.*

**[138]** F. Wilcoxon, "Individual Comparisons by Ranking Methods," *Biometrics Bulletin*, vol. 1, no. 6, pp. 80–83, 1945, doi: 10.2307/3001968.  
- **Key:** `wilcoxon1945`  ·  **Link:** https://doi.org/10.2307/3001968
- **Access:** Subscription (institutional access)
- **Where and how it is used:** Ch.3-5 paired Wilcoxon signed-rank test for every seed-paired comparison.
- **Verified:** Crossref record matched (title, year); doi.org resolver HTTP 302

**[139]** S. Holm, "A simple sequentially rejective multiple test procedure," *Scand. J. Statist.*, vol. 6, no. 2, pp. 65–70, 1979. [Online]. Available: https://www.jstor.org/stable/4615733  
- **Key:** `holm1979`  ·  **Link:** https://www.jstor.org/stable/4615733
- **Access:** JSTOR (institutional access or free JSTOR reading account)
- **Where and how it is used:** Ch.5 Holm step-down correction over the main-comparison family.
- **Verified:** no DOI exists; identity confirmed via JSTOR id 4615733 and Mathematical Reviews MR538597

**[140]** B. Efron, "Bootstrap Methods: Another Look at the Jackknife," *The Annals of Statistics*, vol. 7, no. 1, pp. 1–26, 1979, doi: 10.1214/aos/1176344552.  
- **Key:** `efron1979bootstrap`  ·  **Link:** https://doi.org/10.1214/aos/1176344552
- **Access:** Open access (bronze); free copy: https://projecteuclid.org/journals/annals-of-statistics/volume-7/issue-1/Bootstrap-Methods-Another-Look-at-the-Jackknife/10.1214/aos/1176344552.pdf
- **Where and how it is used:** Ch.5 bootstrap 95% CI of mean paired differences (10,000 resamples).
- **Verified:** Crossref record matched (title, year); doi.org resolver HTTP 302

**[141]** B. Efron and R. J. Tibshirani, *An Introduction to the Bootstrap*. Boston, MA, USA: Springer US, 1993, doi: 10.1007/978-1-4899-4541-9.  
- **Key:** `efron1993intro`  ·  **Link:** https://doi.org/10.1007/978-1-4899-4541-9
- **Access:** Book (library / publisher)
- **Where and how it is used:** Ch.3 percentile bootstrap interval as implemented.
- **Verified:** Crossref record matched (title, year); doi.org resolver HTTP 302

**[142]** J. Cohen, *Statistical Power Analysis for the Behavioral Sciences*, 2nd ed. Hillsdale, NJ, USA: Lawrence Erlbaum Associates, 1988, ISBN 978-0-8058-0283-2.  
- **Key:** `cohen1988power`  ·  **Link:** https://openlibrary.org/isbn/9780805802832
- **Access:** Book (library copy; no free full text)
- **Where and how it is used:** Ch.5 Cohen's d effect size and its small/medium/large conventions.
- **Verified:** ISBN record matched in Open Library

**[143]** D. Lakens, "Calculating and reporting effect sizes to facilitate cumulative science: a practical primer for t-tests and ANOVAs," *Frontiers in Psychology*, vol. 4, art. no. 863, 2013, doi: 10.3389/fpsyg.2013.00863.  
- **Key:** `lakens2013effect`  ·  **Link:** https://doi.org/10.3389/fpsyg.2013.00863
- **Access:** Open access (gold); free copy: https://www.frontiersin.org/articles/10.3389/fpsyg.2013.00863/pdf
- **Where and how it is used:** Ch.5 paired-design effect size (d_z) exactly as computed by _cohend().
- **Verified:** Crossref record matched (title, year); doi.org resolver HTTP 302

**[144]** W. H. Kruskal and W. A. Wallis, "Use of Ranks in One-Criterion Variance Analysis," *Journal of the American Statistical Association*, vol. 47, no. 260, pp. 583–621, 1952, doi: 10.1080/01621459.1952.10483441.  
- **Key:** `kruskal1952`  ·  **Link:** https://doi.org/10.1080/01621459.1952.10483441
- **Access:** Subscription (institutional access)
- **Where and how it is used:** Ch.4 Kruskal-Wallis test of C across the three tumour classes.
- **Verified:** Crossref record matched (title, year); doi.org resolver HTTP 302

**[145]** J. Demšar, "Statistical Comparisons of Classifiers over Multiple Data Sets," *J. Mach. Learn. Res.*, vol. 7, pp. 1–30, 2006. [Online]. Available: https://jmlr.org/papers/v7/demsar06a.html  
- **Key:** `demsar2006`  ·  **Link:** https://jmlr.org/papers/v7/demsar06a.html
- **Access:** Open access (official proceedings / journal site)
- **Where and how it is used:** Ch.3 recommends Wilcoxon over t-test for classifier comparison; justifies the test choice.
- **Verified:** official page HTTP 200; title and citation meta-tags (authors, pages) read from the page

**[146]** T. G. Dietterich, "Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms," *Neural Computation*, vol. 10, no. 7, pp. 1895–1923, 1998, doi: 10.1162/089976698300017197.  
- **Key:** `dietterich1998`  ·  **Link:** https://doi.org/10.1162/089976698300017197
- **Access:** Subscription (institutional access)
- **Where and how it is used:** Ch.3 pitfalls of naive significance tests over resampled splits.
- **Verified:** Crossref record matched (title, year); doi.org resolver HTTP 302

**[147]** X. Bouthillier et al., "Accounting for Variance in Machine Learning Benchmarks," in Proc. Mach. Learn. Syst. (MLSys), vol. 3, 2021, pp. 747–769. [Online]. Available: https://proceedings.mlsys.org/paper_files/paper/2021/hash/0184b0cd3cfb185989f858a1d9f5c1eb-Abstract.html  
- **Key:** `bouthillier2021variance`  ·  **Link:** https://proceedings.mlsys.org/paper_files/paper/2021/hash/0184b0cd3cfb185989f858a1d9f5c1eb-Abstract.html
- **Access:** Open access (arXiv / proceedings); free copy: https://arxiv.org/abs/2103.03098
- **Where and how it is used:** Ch.3 seed variance as a major source of benchmark error; supports the 20-seed protocol.
- **Verified:** arXiv API record matched (title); abs page HTTP 200; venue confirmed in the official proceedings listing

**[148]** C. Colas, O. Sigaud, and P.-Y. Oudeyer, "How Many Random Seeds? Statistical Power Analysis in Deep Reinforcement Learning Experiments," arXiv preprint arXiv:1806.08295, 2018.  
- **Key:** `colas2018seeds`  ·  **Link:** https://arxiv.org/abs/1806.08295
- **Access:** Open access (arXiv / proceedings)
- **Where and how it is used:** Ch.3 statistical power vs number of seeds.
- **Verified:** arXiv API record matched (title); abs page HTTP 200; no peer-reviewed version found, cited as preprint

**[149]** B. A. Nosek, C. R. Ebersole, A. C. DeHaven, and D. T. Mellor, "The preregistration revolution," *Proceedings of the National Academy of Sciences*, vol. 115, no. 11, pp. 2600–2606, 2018, doi: 10.1073/pnas.1708274114.  
- **Key:** `nosek2018prereg`  ·  **Link:** https://doi.org/10.1073/pnas.1708274114
- **Access:** Open access (green); free copy: https://www.ncbi.nlm.nih.gov/pmc/articles/5856500
- **Where and how it is used:** Ch.3 rationale for fixing success criteria before running (Stage 7 and Stage 13 preregistered thresholds).
- **Verified:** Crossref record matched (title, year); doi.org resolver HTTP 302

**[150]** L. Hubert and P. Arabie, "Comparing partitions," *Journal of Classification*, vol. 2, no. 1, pp. 193–218, 1985, doi: 10.1007/BF01908075.  
- **Key:** `hubert1985ari`  ·  **Link:** https://doi.org/10.1007/BF01908075
- **Access:** Subscription (institutional access)
- **Where and how it is used:** Ch.5 adjusted Rand index used to show clustered-FL assignments are unstable (ARI=0.017).
- **Verified:** Crossref record matched (title, year); doi.org resolver HTTP 302


## J. Dataset shortcuts and robustness to segmentation (Pillar I, Stages 14-16)

*The documented slice-selection shortcut in the Figshare data, the general shortcut-learning literature, and studies of how radiomic features change with the segmentation method.*

**[151]** D. Wallis and I. Buvat, "Clever Hans effect found in a widely used brain tumour MRI dataset," *Medical Image Analysis*, vol. 77, art. no. 102368, 2022, doi: 10.1016/j.media.2022.102368.  
- **Key:** `wallis2022clever`  ·  **Link:** https://doi.org/10.1016/j.media.2022.102368
- **Access:** Open access (hybrid); free copy: https://www.sciencedirect.com/science/article/pii/S1361841522000214/pdf
- **Where and how it is used:** Ch.4 primary prior work: showed tumour type in the Figshare dataset is predictable without the tumour (skull outline 0.74, three non-tissue features 0.77). Stage 14 replicates it (0.750 from brain-mask geometry); Stage 15 reproduces two of their three features (0.742) and tests whether tumour-region features rely on the shortcut.
- **Verified:** Crossref record matched (title, year); doi.org resolver HTTP 302

**[152]** B. P. Vangala, S. Guda, L. Peddi, and N. Vangala, "Scores That Hold, Benchmarks That Leak: Measuring Dataset Contamination in Public Brain-Tumor MRI Classification," arXiv preprint arXiv:2610.00421, 2026.  
- **Key:** `vangala2026scores`  ·  **Link:** https://arxiv.org/abs/2610.00421
- **Access:** Open access (arXiv / proceedings)
- **Where and how it is used:** Ch.4 preprint: on the merged Kaggle corpus, file-header features alone separate tumour from no tumour at 0.959 balanced accuracy and most traceable test images share a patient with training; companion evidence of non-anatomical shortcuts in public brain-tumour benchmarks.
- **Verified:** arXiv API record matched (title); abs page HTTP 200

**[153]** R. Geirhos et al., "Shortcut learning in deep neural networks," *Nature Machine Intelligence*, vol. 2, no. 11, pp. 665–673, 2020, doi: 10.1038/s42256-020-00257-z.  
- **Key:** `geirhos2020shortcut`  ·  **Link:** https://doi.org/10.1038/s42256-020-00257-z
- **Access:** Open access (green); free copy: https://arxiv.org/pdf/2004.07780
- **Where and how it is used:** Ch.2/4 general definition of shortcut learning used to frame Stages 14-15.
- **Verified:** Crossref record matched (title, year); doi.org resolver HTTP 302

**[154]** A. J. DeGrave, J. D. Janizek, and S.-I. Lee, "AI for radiographic COVID-19 detection selects shortcuts over signal," *Nature Machine Intelligence*, vol. 3, no. 7, pp. 610–619, 2021, doi: 10.1038/s42256-021-00338-7.  
- **Key:** `degrave2021covid`  ·  **Link:** https://doi.org/10.1038/s42256-021-00338-7
- **Access:** Open access (hybrid); free copy: https://www.nature.com/articles/s42256-021-00338-7.pdf
- **Where and how it is used:** Ch.2/4 canonical medical-imaging example of models exploiting non-pathological image cues.
- **Verified:** Crossref record matched (title, year); doi.org resolver HTTP 302

**[155]** J. R. Zech, M. A. Badgeley, M. Liu, A. B. Costa, J. J. Titano, and E. K. Oermann, "Variable generalization performance of a deep learning model to detect pneumonia in chest radiographs: A cross-sectional study," *PLOS Medicine*, vol. 15, no. 11, p. e1002683, 2018, doi: 10.1371/journal.pmed.1002683.  
- **Key:** `zech2018pneumonia`  ·  **Link:** https://doi.org/10.1371/journal.pmed.1002683
- **Access:** Open access (gold); free copy: https://journals.plos.org/plosmedicine/article/file?id=10.1371/journal.pmed.1002683&type=printable
- **Where and how it is used:** Ch.2/4 hospital-level confounding in chest radiographs; context for possible site effects in Figshare.
- **Verified:** Crossref record matched (title, year); doi.org resolver HTTP 302

**[156]** S. Lapuschkin, S. Wäldchen, A. Binder, G. Montavon, W. Samek, and K.-R. Müller, "Unmasking Clever Hans predictors and assessing what machines really learn," *Nature Communications*, vol. 10, art. no. 1096, 2019, doi: 10.1038/s41467-019-08987-4.  
- **Key:** `lapuschkin2019cleverhans`  ·  **Link:** https://doi.org/10.1038/s41467-019-08987-4
- **Access:** Open access (gold); free copy: https://www.nature.com/articles/s41467-019-08987-4.pdf
- **Where and how it is used:** Ch.2 origin of the Clever Hans framing for machine-learning predictors.
- **Verified:** Crossref record matched (title, year); doi.org resolver HTTP 302

**[157]** L. Oakden-Rayner, J. Dunnmon, G. Carneiro, and C. Re, "Hidden stratification causes clinically meaningful failures in machine learning for medical imaging," *Proceedings of the ACM Conference on Health, Inference, and Learning*, pp. 151–159, 2020, doi: 10.1145/3368555.3384468.  
- **Key:** `oakden2020hidden`  ·  **Link:** https://doi.org/10.1145/3368555.3384468
- **Access:** Open access (gold)
- **Where and how it is used:** Ch.4 hidden stratification; motivates evaluating inside strata (Stages 14B and 15).
- **Verified:** Crossref record matched (title, year); doi.org resolver HTTP 302

**[158]** M. A. Badgeley et al., "Deep learning predicts hip fracture using confounding patient and healthcare variables," *npj Digital Medicine*, vol. 2, art. no. 31, 2019, doi: 10.1038/s41746-019-0105-1.  
- **Key:** `badgeley2019hip`  ·  **Link:** https://doi.org/10.1038/s41746-019-0105-1
- **Access:** Open access (gold); free copy: https://www.nature.com/articles/s41746-019-0105-1.pdf
- **Where and how it is used:** Ch.4 confounding by acquisition and patient variables in medical imaging models.
- **Verified:** Crossref record matched (title, year); doi.org resolver HTTP 302

**[159]** J. W. Gichoya et al., "AI recognition of patient race in medical imaging: a modelling study," *The Lancet Digital Health*, vol. 4, no. 6, pp. e406–e414, 2022, doi: 10.1016/S2589-7500(22)00063-2.  
- **Key:** `gichoya2022race`  ·  **Link:** https://doi.org/10.1016/S2589-7500(22)00063-2
- **Access:** Open access (gold); free copy: http://www.thelancet.com/article/S2589750022000632/pdf
- **Where and how it is used:** Ch.4 evidence that imaging models extract information not visible to readers; context for shortcut analysis.
- **Verified:** Crossref record matched (title, year); doi.org resolver HTTP 302

**[160]** C. Tinauer, M. Sackl, R. Stollberger, R. Schmidt, S. Ropele, and C. Langkammer, "Skull-stripping induces shortcut learning in MRI-based Alzheimer’s disease classification," *Insights into Imaging*, vol. 16, art. no. 283, 2025, doi: 10.1186/s13244-025-02158-4.  
- **Key:** `tinauer2025skull`  ·  **Link:** https://doi.org/10.1186/s13244-025-02158-4
- **Access:** Open access (gold); free copy: https://link.springer.com/content/pdf/10.1186/s13244-025-02158-4.pdf
- **Where and how it is used:** Ch.4 brain-outline geometry as a shortcut in MRI classification (Alzheimer's disease); closest analogue to the brain-mask geometry shortcut measured in Stage 14.
- **Verified:** Crossref record matched (title, year); doi.org resolver HTTP 302

**[161]** C. Vásquez-Venegas et al., "Detecting and Mitigating the Clever Hans Effect in Medical Imaging: A Scoping Review," *Journal of Imaging Informatics in Medicine*, vol. 38, no. 4, pp. 2563–2579, 2024, doi: 10.1007/s10278-024-01335-z.  
- **Key:** `vasquez2024cleverhans`  ·  **Link:** https://doi.org/10.1007/s10278-024-01335-z
- **Access:** Open access (green); free copy: https://pmc.ncbi.nlm.nih.gov/articles/PMC12344020/pdf/10278_2024_Article_1335.pdf
- **Where and how it is used:** Ch.2/4 scoping review: most imaging studies do not test for shortcuts; supports the motivation for Stages 14-15.
- **Verified:** Crossref record matched (title, year); doi.org resolver HTTP 302

**[162]** F. Tixier, H. Um, R. J. Young, and H. Veeraraghavan, "Reliability of tumor segmentation in glioblastoma: Impact on the robustness of MRI‐radiomic features," *Medical Physics*, vol. 46, no. 8, pp. 3582–3591, 2019, doi: 10.1002/mp.13624.  
- **Key:** `tixier2019reliability`  ·  **Link:** https://doi.org/10.1002/mp.13624
- **Access:** Open access (green); free copy: https://www.ncbi.nlm.nih.gov/pmc/articles/6692188
- **Where and how it is used:** Ch.4 edge, shape and size-zone features are the most affected by the segmentation method; context for Stage 16 where shape features degraded under U-Net masks.
- **Verified:** Crossref record matched (title, year); doi.org resolver HTTP 302

**[163]** C. Haarburger, G. Müller-Franzes, L. Weninger, C. Kuhl, D. Truhn, and D. Merhof, "Radiomics feature reproducibility under inter-rater variability in segmentations of CT images," *Scientific Reports*, vol. 10, art. no. 12688, 2020, doi: 10.1038/s41598-020-69534-6.  
- **Key:** `haarburger2020reproducibility`  ·  **Link:** https://doi.org/10.1038/s41598-020-69534-6
- **Access:** Open access (gold); free copy: https://www.nature.com/articles/s41598-020-69534-6.pdf
- **Where and how it is used:** Ch.4 reproducibility of radiomic features under manual and probabilistic automatic segmentations; context for Stage 16.
- **Verified:** Crossref record matched (title, year); doi.org resolver HTTP 302

**[164]** M. G. Poirot et al., "Robustness of radiomics to variations in segmentation methods in multimodal brain MRI," *Scientific Reports*, vol. 12, art. no. 16712, 2022, doi: 10.1038/s41598-022-20703-9.  
- **Key:** `poirot2022robustness`  ·  **Link:** https://doi.org/10.1038/s41598-022-20703-9
- **Access:** Open access (gold); free copy: https://www.nature.com/articles/s41598-022-20703-9.pdf
- **Where and how it is used:** Ch.4 small differences between segmentation methods change radiomic features and their predictive value in brain MRI; context for Stage 16.
- **Verified:** Crossref record matched (title, year); doi.org resolver HTTP 302

**[165]** Y. Suter et al., "Radiomics for glioblastoma survival analysis in pre-operative MRI: exploring feature robustness, class boundaries, and machine learning techniques," *Cancer Imaging*, vol. 20, art. no. 55, 2020, doi: 10.1186/s40644-020-00329-8.  
- **Key:** `suter2020glioblastoma`  ·  **Link:** https://doi.org/10.1186/s40644-020-00329-8
- **Access:** Open access (gold); free copy: https://cancerimagingjournal.biomedcentral.com/track/pdf/10.1186/s40644-020-00329-8
- **Where and how it is used:** Ch.4 shape features robust under simulated perturbations, in contrast with automatic-mask studies; cited to show the finding depends on the setting.
- **Verified:** Crossref record matched (title, year); doi.org resolver HTTP 302

**[166]** M. Nadeem, A. Shaheen, M. F. A. Chaudhary, and H. Mohy-ud-Din, "Towards robust radiomics and radiogenomics predictive models for brain tumor characterization," arXiv preprint arXiv:2406.03583, 2024.  
- **Key:** `nadeem2024robust`  ·  **Link:** https://arxiv.org/abs/2406.03583
- **Access:** Open access (arXiv / proceedings)
- **Where and how it is used:** Ch.4 across seven CNN segmenters on BraTS, shape features were least stable and most stable features were texture features; closest precedent for the Stage 16 feature-level pattern.
- **Verified:** arXiv API record matched (title); abs page HTTP 200


## Using this list for the literature review

Numbers refer to the list above. Each group is a paragraph-sized theme; together they cover a review of either pillar alone or of the whole thesis.

### Pillar I: fractal descriptor and classification

- **Brain-tumour types and MRI context:** [14], [15]
- **Radiomics and hand-crafted features:** [17], [16], [58], [59], [56]
- **Fractal geometry and its estimation:** [42], [43], [44], [45], [46], [11], [51], [52], [53]
- **Fractals in the brain and in brain tumours:** [12], [13], [47], [48], [50], [49]
- **Lacunarity and entropy:** [54], [55], [57]
- **Classification on the Figshare dataset:** [18], [81], [82], [83], [84], [85], [86], [87], [88], [89], [90]
- **Evaluation rigour and leakage:** [25], [26], [27], [28], [79], [80]
- **Segmentation and its limits:** [60], [63], [64], [65], [66], [70], [69], [68], [71], [72]
- **Shortcuts in the Figshare data and in medical imaging:** [151], [152], [153], [156], [154], [155], [158], [157], [159], [160], [161]
- **Feature stability under automatic segmentation:** [162], [163], [164], [165], [166]

### Pillar II: federated learning under class skew

- **Federated learning foundations:** [93], [1], [2]
- **Non-IID data and client drift:** [30], [29], [3], [31], [4], [94], [95], [96], [97], [99], [98]
- **Label skew and missing classes:** [100], [101], [102], [103], [104], [105], [106], [115]
- **Normalisation under non-IID:** [117], [116], [118], [119], [120]
- **Pre-training and frozen backbones in FL:** [134], [135], [129], [131], [130], [136], [137]
- **Personalized and clustered FL:** [109], [110], [111], [112], [107], [108]
- **FL in medicine and brain-tumour imaging:** [6], [7], [113], [8], [9], [114], [5], [91], [92]

### Combined review (both pillars)

- **Open framing:** [6], [7], [5], [10]
- **Bridge: interpretable features vs deep models under privacy constraints:** [16], [17], [49], [9], [8]
- **Methodology shared by both pillars:** [25], [145], [138], [139], [140], [143], [147], [149]

## Corrections found during verification

- Kornblith et al. (CVPR 2019): pages are **2656–2666** in IEEE Xplore (the DOI version). 2661–2671 is the CVF open-access pagination and does not match the DOI.
- Islam et al., *Neural Processing Letters*: issue year **2023**, vol. 55, no. 4, pages **3779–3809** (not 3779–3798).
- MOON (Li, He, Song, CVPR 2021): pages **10708–10717**. Class-balanced loss (Cui et al., CVPR 2019): pages **9260–9269**.
- Gonzato et al.: the finite-size/sampling-bias paper is the **1998** one with **Marzocchi** (GJI 132(2):275–282). The **2000** paper has **Ciccotti** as third author and concerns placement/orientation bias. Both are now listed separately.
- Liu, *J. Biomed. Inform.* 2009 symmetry review: DOI is **10.1016/j.jbi.2009.07.003** (a DOI ending .008 points to a different paper).
- FL medical-imaging survey in *Pattern Recognition* 2024: first author is **H. Guan** (Guan, Yap, Bozoki, Liu), not "Rahman".
- arXiv:2502.11323 is Lyu, Zhou and Zhong, *A statistical theory of overfitting for imbalanced classification*; its described use was corrected to match what the paper actually studies.
- BRISC: *Scientific Data* **vol. 13, art. no. 361, 2026** (the "vol. 12, art. 845, 2025" form is wrong). Sánchez is the first author of the 2022 *Frontiers in Physiology* fractal paper.
- Eight former arXiv citations now point to their peer-reviewed versions: FedDyn, Adaptive FL, logit adjustment, FedBN (ICLR 2021); SGDR (ICLR 2017); fine-tuning distortion (ICLR 2022); Where-to-Begin and Pre-training-for-FL (ICLR 2023). pFL-Bench now cites NeurIPS 2022 Datasets and Benchmarks; Bouthillier et al. cites MLSys 2021.
- Mandelbrot (1982) was published in **San Francisco** by W. H. Freeman (Open Library record).
- The ACM record for FedRS stores only the short title "FedRS"; the full title is restored in the citation.
- A "recent brain-tumour-specific data-leakage paper" suggested earlier does not exist as a peer-reviewed publication and is not cited. Kapoor and Narayanan (2023) is used as the general companion to Yagis et al. (2021).
