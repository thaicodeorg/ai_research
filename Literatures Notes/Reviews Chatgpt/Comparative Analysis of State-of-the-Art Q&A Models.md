---
create: 2026-10-01
type: fleet
tags:
source: "[[CHF_3.pdf]]"
---

# Comparative Analysis of State-of-the-Art Q&A Models

The following is a **section-by-section critical literature review** of the uploaded paper. It follows the paper’s main sections and subsections, distinguishing the author’s reported findings from methodological limitations.

**Paper reviewed:** Özkurt, C. (2024). _Comparative Analysis of State-of-the-Art Q&A Models: BERT, RoBERTa, DistilBERT, and ALBERT on SQuAD v2 Dataset_. _Chaos and Fractals, 1_(1), 19–30. [https://doi.org/10.69882/adba.chf.2024073](https://doi.org/10.69882/adba.chf.2024073).

### 1. Abstract

Özkurt (2024) investigates the performance of four transformer-based question-answering models—BERT, RoBERTa, DistilBERT, and ALBERT—using the SQuAD v2 dataset. The study considers both questions with answers available in the supplied context and questions that require the model to return no answer. ALBERT achieves the highest reported performance, with an Exact Match score of 86.85% and an F1 score of 89.91%. The abstract positions the comparison as a practical contribution to model selection for question-answering applications.

From a critical perspective, the abstract clearly identifies the models, benchmark, and principal findings, but provides insufficient information about model checkpoints and experimental controls. Its implications for real-world applications therefore extend beyond the evidence presented. The findings establish performance differences under the reported evaluation conditions, rather than demonstrating that one model is universally superior.

### 2. Introduction and Previous Research

The introduction situates the study within the development of transformer-based natural language processing and transfer learning. It argues that pretrained language models have improved question-answering systems by enabling richer contextual representations. SQuAD v2 provides the central evaluation setting because it tests both answer extraction and the ability to recognize when a passage does not contain an answer. This distinction is important: a reliable question-answering system must avoid supplying unsupported answers as well as identify correct ones.

The paper incorporates its previous-research discussion into the introduction rather than presenting a separate related-work section. It discusses several research directions, including dynamic question answering through RealTime QA, multi-document question answering through Visconde, transformer comparisons on SQuAD, and methods for filtering unanswerable questions. As presented by Özkurt, these studies show that question-answering performance depends on multiple components, including retrieval quality, contextual understanding, model architecture, and answerability decisions.

The introduction also cites applications such as spam detection, fake-news classification, named entity recognition, and Bengali question answering. These examples demonstrate the broad applicability of pretrained transformers, but their connection to the specific SQuAD v2 comparison is sometimes indirect. A stronger synthesis would group prior studies by research problem and explain precisely what remains unresolved in existing comparisons.

The study’s contribution is primarily **comparative evaluation of established models**, rather than development of a new architecture. Its research gap would be clearer if it identified a specific deficiency in previous experiments, such as inconsistent evaluation subsets, insufficient analysis of unanswerable questions, or missing measurements of computational cost.

### 3. Materials and Methods

The methodology compares four pretrained transformer models adapted to question answering. The author emphasizes the value of evaluating multiple architectures and reporting results separately for answerable and unanswerable questions. This design can reveal whether a model’s overall score reflects strong answer extraction, reliable abstention, or a combination of both.

However, the experimental description does not provide enough detail to establish a fully controlled comparison. Exact checkpoint identifiers, consistent model sizes, preprocessing settings, and a complete set of training hyperparameters are not clearly documented. Consequently, observed differences may reflect checkpoint selection and implementation choices alongside architectural differences.

#### 3.1 BERT and Its Impact

This subsection presents BERT as a major development in contextual language representation. Its relevance to question answering lies in jointly processing the question and passage so that token representations reflect their surrounding context. Pretraining supplies general language representations, while supervised fine-tuning adapts those representations to answer extraction.

One architectural correction is necessary when using this paper in a literature review. BERT is based on a stack of bidirectional transformer **encoders**. The paper’s later description of BERT as having encoder and decoder components should not be repeated as a description of standard BERT. The original BERT study describes task adaptation through additional output layers. [ACL Anthology](https://aclanthology.org/N19-1423/?utm_source=chatgpt.com)

#### 3.2 Application in Question Answering

The paper explains that SQuAD v2 extends answer extraction by introducing questions whose answers are absent from the passage. This makes the benchmark relevant to systems that must decide whether sufficient evidence exists before responding. The subsection therefore establishes answerability detection as an important aspect of question-answering reliability.

Nevertheless, performance on this benchmark does not directly establish effectiveness in open-domain search, multi-document reasoning, or conversational applications. The experiment supplies a context passage to the model; it does not evaluate an entire retrieval-and-answering system. Its conclusions are most appropriately applied to passage-based reading comprehension.

#### 3.3 Datasets

The study uses SQuAD v2 and reports 130,319 training examples and 11,873 validation examples. It describes the dataset’s question identifiers, context passages, answer text, answer positions, and answerability labels. The inclusion of unanswerable questions supports an evaluation that considers both extracting evidence and declining to answer when evidence is unavailable.

A significant issue emerges in the results table: BERT, RoBERTa, and ALBERT are evaluated on 11,873 examples, whereas DistilBERT is evaluated on only 6,078. DistilBERT’s evaluation also has a different proportion of answerable and unanswerable questions. Without explaining how this subset was selected, the study cannot establish that all four models were assessed under equivalent conditions.

A more reproducible dataset description would also specify question IDs, filtering rules, maximum sequence length, treatment of long passages, and the mapping between character-level answer positions and token-level labels.

#### 3.4 Model Architecture

**BERT.** The paper describes the combination of question and context tokens, contextual representation through attention, and prediction of answer positions. This provides a useful introduction to extractive question answering. However, its discussion sometimes mixes sequence classification with answer-span prediction. An extractive QA model predicts the beginning and end of an answer within the passage; it should not be described simply as producing an answer class from the `[CLS]` representation.

**DistilBERT.** DistilBERT is presented as a compressed alternative to BERT that offers potential advantages when computational resources are limited. The comparison introduces the useful question of how much answer quality is retained in a smaller model. However, the study does not report latency, memory consumption, throughput, or training time. Its recommendation for resource-constrained applications is therefore not directly demonstrated by the experiment.

**RoBERTa.** The paper associates RoBERTa with changes to BERT’s pretraining procedure, including dynamic masking, removal of next-sentence prediction, and training on larger text collections. These characteristics provide a rationale for evaluating whether improved pretraining transfers to question answering. The subsection also discusses an antonym or negation test, but does not provide a sufficiently detailed protocol or quantified results to support a reproducible robustness conclusion.

**ALBERT.** ALBERT is introduced as a parameter-efficient BERT variant. For an accurate architectural discussion, its factorized embedding parameterization and cross-layer parameter sharing should be distinguished from its sentence-order prediction objective. These mechanisms are documented in the original ALBERT paper. [arxiv.org](https://arxiv.org/html/1909.11942v6?utm_source=chatgpt.com)

The reviewed paper’s claims that ALBERT-base and ALBERT-xxlarge contain 117 billion and 137 billion parameters are erroneous. Its discussion of self-distillation also does not clearly establish whether that procedure was implemented in the reported experiment. These descriptions should not be used as evidence about the evaluated checkpoint or training pipeline.

#### 3.5 Training Process

The study reports that the models were fine-tuned for three epochs on SQuAD v2. Fine-tuning is an appropriate way to adapt pretrained language representations to supervised question answering. Nevertheless, equal epoch counts alone do not establish equal computational budgets or optimal training conditions for different models.

The paper does not present learning curves, experiments across multiple epoch counts, or repeated runs that would substantiate its claim that ALBERT learns faster. The supported conclusion is narrower: ALBERT achieved the highest reported scores after the specified training procedure. Demonstrating faster convergence would require performance measurements throughout training under clearly defined budgets.

#### 3.6 Performance Metrics

The results include overall Exact Match and F1 scores, separate scores for answerable and unanswerable questions, and scores obtained at selected no-answer thresholds. This breakdown is useful because overall performance can conceal weaknesses in answer extraction or abstention.

Several metric explanations in the paper require correction. In standard SQuAD v2 evaluation:

|Metric|Correct interpretation|
|---|---|
|Exact Match|Percentage of predictions matching a reference answer after normalization|
|F1|Average token-overlap F1 between predictions and reference answers|
|HasAns_exact / HasAns_f1|Scores on questions with reference answers|
|NoAns_exact / NoAns_f1|Scores on questions without reference answers|
|Best_exact / Best_f1|Highest scores obtained by varying the no-answer threshold|

`NoAns_exact` therefore does not mean the number of incorrect answers, and these standard evaluation outputs should not be presented as newly introduced metrics. SQuAD answer F1 also differs from an answerability classification F1 calculated from a confusion matrix. [GitHub](https://github.com/huggingface/evaluate/blob/main/metrics/squad_v2/README.md?plain=1&utm_source=chatgpt.com)

Threshold optimization deserves careful treatment. Selecting the best threshold on an evaluation set describes the best achievable score on that set; evidence of generalization requires selecting the threshold on separate development data and then testing it independently.

### 4. Results

The following values reproduce the paper’s Table 1, rounded to two decimal places.

|Model|Overall EM (%)|Overall F1 (%)|Answerable F1 (%)|Unanswerable EM (%)|Evaluation examples|
|---|---|---|---|---|---|
|BERT-medium|65.96|70.12|76.13|64.12|11,873|
|DistilBERT|64.89|68.18|76.63|60.42|6,078|
|RoBERTa|79.87|82.91|84.03|81.80|11,873|
|ALBERT|86.85|89.91|90.58|89.25|11,873|

ALBERT achieves the highest reported overall scores and performs strongly on both answerable and unanswerable questions. RoBERTa ranks second and exceeds BERT-medium in both categories. BERT-medium and DistilBERT have closer overall scores, although DistilBERT’s different evaluation subset prevents a clean direct comparison.

The numerical results also clarify an inconsistency in the narrative: the paper sometimes describes BERT as closely following ALBERT, but its table places RoBERTa ahead of BERT. The ranking supported by the reported overall scores is **ALBERT → RoBERTa → BERT-medium → DistilBERT**.

These findings are useful as reported benchmark observations, but their explanatory strength is limited. The study does not provide repeated-run variability, confidence intervals, or sufficiently controlled model configurations. It therefore cannot establish how much of the difference arises from architecture, model size, prior checkpoint training, hyperparameters, or evaluation selection.

### 5. Conclusion

Özkurt concludes that ALBERT provides the strongest performance under the study’s conditions, while RoBERTa offers strong results and DistilBERT remains potentially useful where computational resources are limited. The conclusion also identifies further training, parameter adjustment, additional datasets, and ensemble methods as possible directions for improvement.

The strongest contribution is the joint examination of answer extraction and unanswerable-question handling. However, the claims about faster learning and resource efficiency exceed the measurements reported. Stronger conclusions would require identical evaluation examples, clearly identified checkpoints, repeated experiments, learning curves, and direct computational measurements.

### 6. Research Gaps Identified from the Review

The paper suggests several concrete opportunities for further research:

- **Controlled comparison:** Evaluate all models on identical question IDs with transparent preprocessing and checkpoint selection.
- **Accuracy and efficiency:** Measure Exact Match and F1 alongside inference time, memory use, and training cost.
- **Answerability calibration:** Examine whether models’ confidence scores reliably indicate when they should abstain.
- **Training sensitivity:** Compare learning curves, training budgets, hyperparameters, and multiple random seeds.
- **Generalization:** Test performance on additional domains, languages, and datasets.
- **Robustness:** Conduct reproducible evaluations involving negation, paraphrasing, misleading contexts, and incomplete evidence.

### 7. Literature Review Paragraph for Academic Writing

Özkurt (2024) compared BERT-medium, RoBERTa, DistilBERT, and ALBERT on the SQuAD v2 question-answering benchmark, examining both answer extraction and unanswerable-question handling. Following three reported fine-tuning epochs, ALBERT achieved the highest overall Exact Match and F1 scores of 86.85% and 89.91%, respectively, followed by RoBERTa. However, the comparison is limited by incomplete reporting of model configurations and training settings, as well as the use of fewer evaluation examples for DistilBERT. Furthermore, the absence of repeated experiments and computational measurements restricts conclusions about statistical reliability, convergence speed, and deployment efficiency. The study consequently provides useful comparative observations while highlighting the need for controlled and reproducible evaluations of question-answering accuracy, abstention, and computational cost.