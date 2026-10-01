---
create: 2026-10-01
type: fleet
tags:
source: "[[1-s2.0-S1877050924030527-main-BERT-and-Roberta-model.pdf]]"
---

# BERT and RoBERTa Models for Enhanced Detection of Depression in Social Media Text

The following is a **section-by-section critical literature review** of Kurniadi et al. (2024), _“BERT and RoBERTa Models for Enhanced Detection of Depression in Social Media Text,”_ published in _Procedia Computer Science_, volume 245, pages 202–209. It follows the paper’s sections and distinguishes reported findings from critical interpretation. 1-s2.0-S1877050924030527-main-B…

## 1. Abstract and Introduction

### 1.1. Review of the Abstract

Kurniadi et al. (2024) investigate the application of BERT and RoBERTa to depression-related text classification using Reddit posts obtained from Kaggle. The abstract identifies depression detection as the research problem and presents transformer-based natural language processing as the proposed approach. Both models reportedly achieve approximately 98% accuracy, accompanied by high precision, recall, and F1 scores. These findings suggest that pretrained language models can effectively distinguish between the two labels in the selected dataset. 1-s2.0-S1877050924030527-main-B…

However, the abstract does not describe the dataset size, annotation procedure, evaluation split, or uncertainty surrounding the results. Consequently, the reported accuracy should be interpreted as performance within the study’s experimental setting. It does not establish that the models can reliably identify clinically diagnosed depression or generalize to other populations and platforms.

### 1.2. Review of the Introduction

The introduction establishes the importance of early depression identification and explains why social media text may provide useful information about emotional experiences. The authors position their work within several existing approaches, including electronic health record analysis, speech analysis, and classification of personal disclosures on social media. Their central objective is to compare BERT and RoBERTa for identifying depression-related content through contextual language representations. 1-s2.0-S1877050924030527-main-B…

The motivation is relevant because social media language can contain indirect expressions, informal wording, and contextual meanings that are difficult to capture using simple keyword features. Nevertheless, the introduction provides a limited explanation of the specific research gap. Previous studies had already applied BERT and RoBERTa to similar tasks. Therefore, the contribution is best understood as an **application and comparative evaluation on a selected dataset**, rather than the development of a new architecture.

The introduction also discusses mental health challenges in Indonesia, whereas the experiment does not evaluate an Indonesian-language dataset. This creates a gap between the broader motivation and the evidence produced by the study.

## 2. Previous Research

The previous research section describes a progression from conventional machine learning to recurrent neural networks and transformer-based models. The table below summarizes these studies **as described by Kurniadi et al.**; their original publications have not been independently reviewed here. 1-s2.0-S1877050924030527-main-B…

|Study discussed|Approach|Finding reported in the reviewed paper|
|---|---|---|
|Zaman et al.|Fine-tuned RoBERTa|Approximately 90% accuracy for multilevel depression classification|
|Suhas et al.|Conventional machine learning|Random Forest performed best among the evaluated algorithms|
|Kanaan et al.|Neural networks with word embeddings|CNN–LSTM achieved 97.56% accuracy; LSTM achieved 97.48%|
|Fatimah et al.|Naïve Bayes, SVM, Logistic Regression, and Random Forest|Reported accuracies included 74.36% for Naïve Bayes and 76.42% for SVM|
|Chowanda et al.|BERT–BiLSTM|Highlighted the potential of hybrid models and the need for Indonesian-language research|
|Karna et al.|BERT and RoBERTa with dense layers|Reported approximately 98% accuracy|

Collectively, these studies motivate the use of contextual language representations for mental health-related text analysis. Conventional classifiers offer useful baselines, while neural and transformer-based approaches can represent more complex linguistic relationships. The discussion of Indonesian-language research also highlights the importance of cultural and linguistic variation.

However, the review mainly summarizes individual studies rather than critically comparing their datasets, annotation quality, evaluation procedures, and limitations. The reported accuracies cannot establish a reliable ranking across studies because the classification tasks and experimental conditions differ. For example, multilevel depression classification is a different task from binary classification.

An additional weakness is inconsistent citation numbering: several in-text references do not correspond correctly to the numbered bibliography. A subsequent literature review should verify the original publications before reusing these secondary claims.

## 3. Methodology

### 3.1. Dataset

The study uses a Kaggle dataset containing **1,203 entries** with two fields: `clean text` and `label`. Labels indicate depression (`1`) or non-depression (`0`). The class distribution is approximately balanced, with 49.6% depression-labelled entries and 50.4% non-depression-labelled entries. 1-s2.0-S1877050924030527-main-B…

The balanced distribution reduces the possibility that high accuracy results simply from predicting the majority class. Nevertheless, the dataset is relatively small, and the paper provides insufficient information about how the labels were established. It does not explain whether labels represent clinical diagnoses, self-reported conditions, subreddit membership, or another annotation strategy.

This distinction matters because the model learns to predict the dataset’s labels. Without evidence about label validity, strong classification performance cannot be treated as proof of clinical depression detection. The paper also does not document checks for duplicate posts or overlapping users across evaluation partitions, leaving possible leakage insufficiently addressed.

### 3.2. Pre-processing

Pre-processing involves model-specific tokenization, conversion of text into numerical token identifiers, and construction of attention masks to distinguish input tokens from padding. The experimental sequence length is set to 128 tokens. 1-s2.0-S1877050924030527-main-B…

These steps are appropriate for transformer-based text classification. However, the paper does not specify the tokenizer checkpoints, truncation strategy, padding configuration, or cleaning operations already applied to the `clean text` field. This limits reproducibility. The 128-token limit may also exclude relevant context from longer posts, but the study does not report post-length statistics or the proportion of truncated examples.

### 3.3. BERT

The paper introduces BERT as a transformer encoder that learns contextual representations through masked language modelling and next sentence prediction. This provides a theoretical basis for examining language whose meaning depends on surrounding words. 1-s2.0-S1877050924030527-main-B…

Several technical statements require correction. BERT uses **bidirectional self-attention**; the paper’s suggestion that “unidirectional” is a more precise description is incorrect. Its architecture table also incorrectly lists the layer counts. The original BERT publication specifies **12 layers for BERT Base and 24 layers for BERT Large**. Furthermore, BERT selects 15% of tokens as prediction targets, but only 80% of those selected tokens are replaced with `[MASK]`; the remainder are randomly replaced or retained. [aclanthology.org](https://aclanthology.org/N19-1423.pdf?utm_source=chatgpt.com)

The methodological description devotes considerable attention to BERT pretraining while providing limited information about the actual classification implementation. The exact checkpoint, classification head, and which parameters were updated are not clearly documented. Consequently, the theoretical explanation is more detailed than the reproducible experimental procedure.

### 3.4. RoBERTa

The authors describe RoBERTa as a refinement of BERT that removes next sentence prediction and modifies the pretraining procedure through dynamic masking, larger batches, and other training changes. These features provide a reasonable motivation for comparing its representations with those of BERT. 1-s2.0-S1877050924030527-main-B…

The original RoBERTa study supports the importance of changes to the pretraining recipe, including longer training, more data, removal of next sentence prediction, and dynamic masking. These are properties of its upstream pretraining procedure. [arxiv.org](https://arxiv.org/pdf/1907.11692?utm_source=chatgpt.com)

However, the depression-classification study does not isolate these changes experimentally. It therefore cannot determine whether dynamic masking, batch size, or another pretraining factor caused any observed difference. Its description also blurs static masking with dynamic masking; the latter generates masking patterns as sequences are presented during pretraining.

## 4. Experiment and Results

### 4.1. Experimental Settings

The experiments were conducted using Google Colab. The following hyperparameters are reported. 1-s2.0-S1877050924030527-main-B…

|Hyperparameter|Value|
|---|---|
|Learning rate|\(2 \times 10^{-5}\)|
|Training epochs|5|
|Dropout rate|0.1|
|Sequence length|128 tokens|
|Optimizer|AdamW|

Reporting these settings helps readers understand the training configuration. Nevertheless, important details remain unspecified, including batch size, random seed, data partition sizes, checkpoint selection, and whether the final metrics were calculated on an independent test set.

The paper also does not report repeated runs or confidence intervals. These omissions make it difficult to assess how stable the reported performance is.

### 4.2. Classification Results

The paper reports the following results. 1-s2.0-S1877050924030527-main-B…

|Model|Accuracy|Precision|Recall|F1 score|
|---|---|---|---|---|
|BERT|98%|0.98|0.98|0.98|
|RoBERTa|98%|0.99|0.97|0.98|

Both models achieve the same reported accuracy and F1 score. RoBERTa has slightly higher precision and slightly lower recall. However, the paper does not specify whether precision, recall, and F1 are calculated for the depression class or use macro, micro, or weighted averaging.

**If these are depression-class metrics**, RoBERTa’s results suggest fewer false positive predictions but more missed depression-labelled examples relative to BERT. Without the averaging definition and a confusion matrix, this interpretation remains conditional.

The results support comparable performance on the selected dataset. They do not establish a statistically significant advantage for either model or demonstrate improvement over conventional baselines, since those baselines were not evaluated under the same conditions.

### 4.3. Training and Validation Loss

The authors report decreasing training loss alongside an increase in validation loss during part of training. They identify this divergence as a possible indication of overfitting. They also observe a smaller training–validation loss gap for RoBERTa and interpret it as greater resilience. 1-s2.0-S1877050924030527-main-B…

The divergence warrants investigation, although a loss curve alone does not fully characterize generalization. Five epochs neither confirm nor rule out overfitting. Similarly, a smaller loss gap does not establish robustness to new platforms, languages, or populations. Such claims would require repeated experiments and evaluation on independent datasets.

## 5. Conclusion

Kurniadi et al. conclude that BERT and RoBERTa are promising approaches for classifying depression-related Reddit text, with both achieving 98% accuracy. They propose evaluating broader datasets and incorporating Indonesian-language data in future research. 1-s2.0-S1877050924030527-main-B…

The study contributes an empirical comparison of established transformer models on a small, nearly balanced dataset. Its findings support their potential for this classification setting, while the limited reporting of annotation, data partitioning, model implementation, and statistical uncertainty restricts stronger conclusions.

For subsequent research, the most defensible gaps are:

- **Label validity:** establish how text labels relate to depression and how annotations are verified.
- **Generalization:** evaluate independent datasets, other platforms, and additional languages.
- **Evaluation reliability:** use documented splits, checks for user overlap, repeated runs, and uncertainty estimates.
- **Comparative evidence:** evaluate conventional and neural baselines under identical conditions.
- **Error analysis:** examine misclassified posts to identify linguistic and contextual weaknesses.

A literature-review synthesis suitable for academic writing is:

> Kurniadi et al. (2024) compared BERT and RoBERTa for binary classification of depression-related Reddit text using 1,203 labelled entries. Both models achieved a reported accuracy of 98% and an F1 score of 0.98, suggesting comparable performance within the selected experimental setting. However, insufficient documentation of label construction, data partitioning, model configuration, and statistical uncertainty limits reproducibility and conclusions about generalization. The study therefore motivates further evaluation using independently validated datasets, transparent experimental procedures, and multilingual settings. 1-s2.0-S1877050924030527-main-B…