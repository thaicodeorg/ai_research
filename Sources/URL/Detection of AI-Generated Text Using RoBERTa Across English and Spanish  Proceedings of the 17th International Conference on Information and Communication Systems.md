---
title: "Detection of AI-Generated Text Using RoBERTa Across English and Spanish | Proceedings of the 17th International Conference on Information and Communication Systems"
source: "https://dl.acm.org/doi/full/10.1145/3812734.3813713"
author:
published:
created: 2026-09-30
description:
tags:
  - "clippings"
---
AI Summary

## Abstract

### Abstract

Due to the rapid growth of large language models (LLMs), it has become increasingly hard to differentiate human-created content from artificial intelligence (AI) created content, particularly in cross-lingual detection. In this paper, the transformer models were employed to classify AI-generated text in multi-lingual using the AuTexTification 2023 dataset, which contains both English and Spanish data. The RoBERTa model was trained using stratified five‑fold cross‑validation, dynamic thresholding, and automatic mixed precision, ensuring an accurate, stable, and reproducible evaluation of the data. The AuTexTification dataset comprises 55,677 English samples and 52,191 Spanish samples, split across domains. In general, RoBERTa has demonstrated excellent performance at detecting AI-Generated Text in both languages. The model achieved an accuracy of 82%, with 0.84 F1-score on 21,832 samples of the English domain, correctly identifying approximately 83% of the AI-generated texts and about 81% of the human-generated texts. The model had an accuracy of about 79% on the Spanish domain test and achieved an F1-score of 0.75. This indicates that the RoBERTa model has strong multi-lingual robustness to be used for AI-generated text detection and the expected decline of performance on morphologically complex languages such as Spanish. The results from both domains highlight the need to have diverse datasets and multilingual models in order to ensure that the AI text detection systems will work reliably in the global digital environment.

### AI Summary

To view this AI-generated plain language summary, you must have Premium access.

## 1 Introduction

Natural Language Processing (NLP) has evolved, much like human intelligence, as shown in Figure [1](#fig1). One of the earliest forms of NLP was dedicated to rule-based systems. It was an early type of NLP that relied heavily upon human-defined rules using "if-then" statements. While these systems could achieve a high level of accuracy, they struggled with the vast number of ambiguous words and phrases in human speech \[[^2]\]. This gave way to the Statistical Era (1990s–2010s), NLP relied upon the use of large collections of text, creating statistical probabilities of the sequences of words that would appear together. Instead of focusing on "grammatically correct" phrases, researchers began to examine the likelihood that certain phrases would appear together. In 2017, there was a radical shift with the development of the Transformer architecture for deep learning.

![](https://dl.acm.org/cms/10.1145/3812734.3813713/asset/f29b77fd-91a7-4324-9f77-ef06ae5eba09/assets/images/large/icics2026-86-fig1.jpg)

The Evolution of Natural Language Processing (NLP).

While the previous technologies (RNNs and LSTMs) processed a text input in parts, meaning that they could easily forget information included early within the sentence, transformers apply an attention mechanism to enable them to process all text inputs simultaneously \[[^7]\]. The development of the transformer has allowed models to capture and express long-distance dependencies and context in ways that were not possible with traditional NLP approaches, shifting from a simple pattern-matching approach to an advanced generative-based understanding of language, as in Table [1](#tab1).

Large Language Models (LLMs) are a technology with the potential to simultaneously further advance society and harm traditional systems of personal trust. A LLM is a productivity-enhancing tool that makes it easier for humans to conduct complex research, thanks to its ability to provide the necessary context for complex problems; automates the creation of computer code; provides democratized access to expert knowledge through the development of customized tutoring, etc. However, while LLMs serve as a valuable productivity-enhancing tool for students and businesses, LLMs pose a systemic threat to Academic Integrity through the ability to easily produce extraordinarily high-quality content that compromises student academic authenticity. On the other hand, the emergence of agent-like Artificial Intelligence (AI), and other new developments in the development of synthetic personas will further enhance the crisis of digital trust in the workplace and on the Internet by creating far more believable misinformation and increasingly harder to detect \[[^16]\].

Detecting AI-generated content has become a substantial competition between the AI models’ content and the Technology that has developed to find them. As of 2026, detecting AI-generated text is still a probability at some level rather than a field of study, but the likelihood of AI-produced work can be ascertained with two different approaches \[[^4]\]. Both are typically categorized as "Forensic" analysis and "watermarking." The Forensic tools used, such as GPTZero and Copyleaks, use linguistic patterns, word choice, and word structural variation as a basis to estimate if the text was likely to be AI-produced or not. One of the major challenges of the Forensic testing approach is that there are numerous false positives among technical or non-native English-speaking authors, who produce works in ways that are naturally similar to the types of patterns that AI uses. Watermarking techniques consist of using a form of statistical signal created during the text production process that identifies and verifies whether the text is AI-generated. Because of the more comprehensive nature of Watermarking, the inability to rely on a Forensic approach to determine if the content is AI-generated has led to difficulty finding AI-generated content by attempting different kinds of external (or visible) watermarking to determine if they are AI-generated \[[^11]\].

Currently, the vast majority of our knowledge regarding AI detection technology is based in English, resulting in a serious lack of coverage for other languages and ultimately yielding less than reliable detection tools that may be used with other languages \[[^28]\]. The language that is most affected by not being represented within the AI generated text detection research is Spanish - one of the most prevalent spoken languages globally - and due to its complex morphology, verb conjugation systems, and flexibility in terms of how words are ordered within a statement or question, proves to be difficult to detect by an AI trained only on English-speaking textual corpuses. For AI detection technology to serve international communities, it is necessary to adopt multilingual approaches to accurately represent the linguistic differences in how people communicate.

<table><tbody><tr><td><b>Era</b></td><td><b>Core Technology</b></td><td>Primary Method</td><td>Major Limitation</td></tr><tr><td>Rule- Based</td><td>Symbolic Logic</td><td>Handwritten grammar rules</td><td>Brittle; cannot handle nuance or slang.</td></tr><tr><td>Statistical</td><td>Probabilistic Models</td><td>Frequency and N-grams</td><td>Lacks deep contextual understanding.</td></tr><tr><td rowspan="2">Deep Learning (DL)</td><td>Neural Networks</td><td>Word Embeddings (Word2Vec)</td><td>Sequential processing is slow and forgetful.</td></tr><tr><td>Transformer, Attention</td><td>Parallel processing of sequences</td><td>Computationally expensive to train.</td></tr></tbody></table>

Key Milestones in the NLP Paradigm Shift.

While there has been some progress in AI-generated text detection, there continue to be substantial research gaps. In particular, current detection/systems fail to generalize well across languages, especially for English and Spanish, limiting their ability to handle multiple languages at once. Domain variability is also an issue because models trained on a given type of text (like tweets) tend to do poorly when tested on other types of data (like legal documents or news articles). There are also very few labeled datasets in Spanish, so the performance of detection systems will be impacted as well. Because language models are changing rapidly, keeping detection relevant will require adaptive software systems. Finally, existing methods and standards for evaluating detection accuracy and reliability are not very clear or established, and this makes progressing in this area of research more difficult.

AI text detection is evolving rapidly as the BERT (Bidirectional Encoder Representations from Transformers) \[[^24]\] and RoBERTa (a variant of BERT) \[[^15]\] models are becoming the basis for representing and analyzing many different languages by being able to understand more complex, nuanced meanings of words in a context. Unlike generative models like GPT (Generative Pre-Training Transformer), which process text in one direction by considering only the word that comes after each other when estimating the probability of a new word, BERT and RoBERTa utilize bidirectional processing to create an input structure for further comparison (semantically and structurally) for both human-created text vs machine-created text.

The study’s primary goal is to assess the ability of BERT and RoBERTa to correctly interpret and attribute machine-generated text in both English and Spanish, via their evaluation against the AuTexTification 2023 dataset \[[^19]\] which was created specifically for detecting machine-generated text. Accuracy will be a measure of success; however, the primary concern for this research is the robustness of the model being evaluated. One of the study’s main goals is to ascertain whether the additional robustness of the RoBERTa pre-training process will enable it to better overcome the language gap between these two languages than the BERT pre-training process. As such, outcomes from this study will demonstrate the ability of the RoBERTa model to withstand the potential misuse of LLMs in multilingual contexts. The main contribution of the paper can be summarized as follows: • The proposed method provides a controlled empirical evaluation of transformer-based detection under consistent cross-lingual settings, focusing on reproducibility, threshold optimization, and comparative robustness across languages. • The proposed approach achieved 82% accuracy and 0.84 F1-score in English, and 79% accuracy and 0.75 F1-score in Spanish, demonstrating strong cross-lingual evaluation capability. • The method performed significantly better than multiple baseline systems, such as BOW+LR, which achieved a 65.78 F1 score, and the transformer baseline, which achieved a 57.10 F1 score, suggesting that contextual transformer models are the best option to use for detecting AI-generated text in multi-lingual situations.

The paper is organized as follows: Section [2](#sec-2) provides an overview of recent research on detecting AI-generated text and previous studies of multi-lingual transformer models. Section [3](#sec-3) describes the experimental methodology, including dataset characteristics, data preprocessing, model architecture, training configuration, and evaluation metrics. Following that, Section [4](#sec-4) contains both the experimental results as well as a discussion on how well the proposed RoBERTa-based detection model performed in both the English and Spanish domains of AI-generated text. Finally, Section [5](#sec-5) concludes with a description of future research directions in the field of multilingual AI-generated text detection.

## 2 Related work

The rapid growth of LLMs has disrupted the digital environment by producing coherent text in various languages. One of the challenges of detecting AI-generated text is that LLMs have become more sophisticated in generating natural, ossified text, as well as rapidly evolving across domains and having limited training data with diversity and evasion methods such as paraphrasing, wherein multi-lingual detection becomes even more complex \[[^13]\]. However, the advancements in detecting AI-generated text, there are still research related to milti-lingual generalization, domain adaptation, and adversarial robustness, which this study intends to take on by reviewing state-of-the-art methods, evaluating them against English text, providing a non-exhaustive list of challenges for multi-lingual generalization, and suggesting evaluation frameworks that can make it more resilient from the threats related to AI and the use of LLMs and to enhance trust in multilingual spaces. Detecting AI-generated text across languages is difficult due to transfer learning limitations, where detectors trained in one language (English) perform poorly on others (Spanish) due to structural differences. Variability in multilingual LLMs’ performance, such as inconsistent quality across language models, makes detection consistency even more difficult \[[^14]\].

Preda et al. combined multiple transformers, including SciBERT, DeBERTa and XLNet, multi-task learning, and virtual adversarial training to detect AI-generated text. The result was an F1 score of about 66-67% across both languages, thanks to the ensemble nature of the models bringing additional stability. A key advantage of this approach is that it enables generalization across different domains using the differing representations of information provided by the specialized models. However, one of the limitations with respect to generalizing across languages in this example stems from reliance on predominantly mono-lingual models versus using a multi-lingual transformer \[[^18]\].

Espin-Riofrio et al. 2023 provided a way of aggregating embeddings from all layers of the BERT model, capturing the stylistic signals that are distributed across model layers. Despite being able to capture substantial amounts of stylistic information when trained with the proposed method, the performance on official test data was lower than expected and revealed signs of overfitting. Their approach of aggregating the embeddings captures a greater number of stylistic cues, and there is the potential for positive impacts on multilingual detection if applied to multilingual versions of BERT. However, their study does not include an evaluation of the robustness of their method for multi-lingual applications, nor does it provide any indication that this approach would apply to low-resource languages, since it was trained with two variants of the BERT model for English and Spanish only \[[^5]\].

Gagiano et al. created a binary classification/machine learning model using "Transformer" methods like RoBERTa type, using data augmentation. They used Pre-Trained Multilingual models in the Augmenting step, which increases their Domain-robustness. However, the results indicate that they experienced problems transferring Binary Classification and Machine Learning Models "across domains and "multi-lingual". Specifically, there was insufficient increase in F1 Scores (approx. Detection: 63%) across Data Domains and Languages, thus indicating the need for additional Fine-tuning of the Models for both English and Spanish \[[^9]\]. Elamine et al. combine a deep Sequential neural model for detection and an SVM for attribution, evaluated on English and Spanish. The system benefits from simplicity and multilingual applicability due to feature-based SVM classification. Nonetheless, performance is modest (English F1 54%, Spanish 51%), highlighting sensitivity to dataset imbalance and difficulty generalizing across languages, especially due to limited exploitation of multilingual embeddings \[[^3]\].

Espin-Riofrio et al. used Perplexity-based Features, Pattern-based Features, and Connector-based Features with Supervised Models and a "Bagging" (Ensemble) Approach. Although Performance within the Language Model exhibited language-agnostic possibilities when employing Perplexity-based Features alone, the results of the analysis performed poorly within the test dataset, indicating the challenges faced regarding multi-Linguistic Transfer due to Differences in the Word Distribution Across each Language. The main advantages of this approach stem from its Language Model Agnostic Feature-Design, yet the challenges appear to arise from inconsistent Language Transfer Performance (Perplexity) due to Differences in the Word Distribution for each Language \[[^6]\].

Villegas Trejo et al. investigated the text representations, n-grams, and classical ML models (logistic regression, random Forest, and SVM) examined for detecting binary text in Spanish and English and attributing models to them. This study shows that advantages in transparency and efficiency are provided by the use of these classical models, and they can be utilized without relying on deep model architectures that are specific to a language. However, these surface representations of the language do not have good multilingual generalization due to the substantial differences in the use of stylistic markers within languages. Therefore, the multi-lingual robustness of these representations was not very high with respect to the six languages tested in this study \[[^25]\].

Fernández García et al. utilized three systems to perform their evaluations of three systems: SVMs, speculative versions of partially multilingual transformer models, and an ensemble of multilingual transformer models with logistic regression. The best macro-F1 result (0.805) is achieved by the ensemble of multilingual models, which has an excellent capacity for multi-lingual generalization across the languages of Spanish, English, Portuguese, Catalan, Basque, and Galician. The multilingual training of the model is its dominant advantage; however, potential performance decreases in cases of underrepresented languages should be noted. High computational requirements for this ensemble of models should also be recognized \[[^10]\].

Scheibe and Mandl fine-tuned DeBERTaV2 to classify text written by humans (i.e., authors) and by computer-generated sources. These researchers also analyzed the readability and lexical diversity of text generated by computers, concluding that these texts are typically less diverse than text written by humans; therefore, on balance, fine-tuning introduced high levels of lexical diversity. In addition to providing excellent performance in monolingual conditions (macro-F1@67), DeBERTaV2 does not provide good multi-language detection because it lacks the capability of multilingual processing via pre-training. The advantages provided by the availability of linguistic metrics for fine-tuning DeBERTaV2 do not overcome the liability of only using one model within a multitude of languages \[[^20]\].

Alonso‑Simón et al. used a linguistic‑feature‑driven system that uses LinearSVC with tf‑idf, character/token n‑grams, POS n‑grams, and punctuation features. It performs strongly in Spanish (F1 = 70.6%) but less so in English (68.33%). The advantage is language independent feature engineering, but true multi‑lingual generalization is limited because features rely on language‑specific morphology and syntax \[[^22]\]. Sheykhlan et al. conducted a fine-tuning process of three different multilingual BERT-like models (ErnieM, BLOOM-560m and mDeBERTaV3) and subsequently utilized soft voting ensembling methods to create a new ensemble model for binary and multi-class detection across several Iberian languages. The results of this study indicate that the ensemble model achieved a better overall performance than the individual models, demonstrating the value of using diverse models for improving multilingual robustness. However, the key remaining challenge is that all of the models used were high-resource multilingual models, and there is still uncertainty about their influence on performance in lower-resource languages \[[^21]\].

Gritsay et al. assess the performance of the XLM‑RoBERTa, mDeBERTa, and MiniLM‑V2 models using fine‑tuning, extensive pre‑processing, and data expansion methods for training data; therefore, the method provides a multilingual approach that can inherently support the generalization of multi‑lingual models, achieving a representation of approximately 66% for the F1 score within the English language. The results demonstrated the strengths of using multilingual encoders that have been developed with cross‑language transfer. The authors identify weaknesses resulting from domain drift and the lower performance level on the languages with the least training representation in their sample \[[^12]\]. Fernández‑Hernández et al. conducted experiments utilizing both classical machine learning and neural network models in order to distinguish between AI and human text in various domains and languages. There was variability in methodology between authors; however, their studies illustrate the common deficiency regarding the generalization of cross‑lingual models, specifically the tendency for models to over‑fit the domain/language‑specific elements. Therefore, the strength of their study is the comprehensive assessment of various architectures, while the limitation lies in the absence of a unified framework for multilingual modeling \[[^8]\].

To demonstrate the effectiveness of deep contextual representations for identifying and classifying AI-generated content, Wang et al. introduced a BERT-based framework for detecting and classifying AI-generated content, demonstrating the effectiveness of deep contextual representations \[[^26]\]. Alghamdi et al. created an adapted BERT framework (called ABERT) that has been optimized for efficiently detecting AI-generated fake news \[[^1]\]. To detect AI-generated news, Wang et al. utilized both BERT and fine-tuned RoBERTa models to showcase the benefits of transformer-based architectures in terms of capturing semantic and stylistic patterns \[[^27]\].

Transformer-Based Models (e.g., those used in Preda et al., Gagiano et al, Hildesheim, KaramiTeam, Gritsay et al., and Human-After-All) - All employed "fine-tuner" multilingual or monolingual Transformers, which took advantage of the contextual encoding capabilities of this type of model. All of the Transformers employed in these papers demonstrated an ability to outperform simpler models in predicting text produced by AI and to provide heightened multi-linguistic generalization capabilities. Fernández-Garcia & Segura-Bedmar and KaramiTeam also provided evidence to support the use of ensemble learning with multilingual encoders or groups of multilingual Transformers.

## 3 Experimental work

### 3.1 Detection Model Selection

The selection of a detection model will also need to be feasible in terms of computational requirements versus achieving the highest possible level of accuracy and language support based on the opportunities provided by the development of more advanced forms of AI-generated text. For example, as the sophistication of the current forms of algorithms to reduce text can be replicated by AI systems, the model used for detecting must also reflect the same level of improvement in language capability and ability to model the development of language, as shown by BERT. In this regard, the BERT model for detecting was selected in this research because it can handle the challenges related to the "Short Text Dominance" and "Domain Differences" found in the AuTexTification dataset. Traditional algorithms utilized for identifying subtleties in synthetic writing utilize simple statistical analysis for the written texts; by using a bidirectional approach, the BERT model detects syntactical and semantic artifacts associated with both the English language (in this case) and the Spanish language (in this case) far more accurately than traditional methods.

### 3.2 Dataset Overview

The AuTexTification dataset was generated from the IberLEF 2023 shared task \[[^19]\], for two purposes: (1) for distinguishing whether a text is written by a human versus generated by a machine, and (2) to determine which machine model generated the text using 6 AI models. The English domain of the AuTexTification 2023 dataset consists of a large, balanced collection of English-language texts labeled as either human-written or AI-generated, as presented in Table [2](#tab2).

<table><tbody><tr><td><b>Split</b></td><td><b>Class</b></td><td><b>AI GEN</b></td><td><b>HUM</b></td><td><b><i>Σ</i></b> <b>(Subtask 1)</b></td></tr><tr><td colspan="5"><b>English Domain</b></td></tr><tr><td>Training subset</td><td>Legal</td><td>5,124</td><td>5,244</td><td>10,368</td></tr><tr><td></td><td>Tweets</td><td>5,813</td><td>5,884</td><td>11,697</td></tr><tr><td></td><td>How-to</td><td>5,862</td><td>5,918</td><td>11,780</td></tr><tr><td>Testing subset</td><td>News</td><td>5,464</td><td>5,464</td><td>10,928</td></tr><tr><td></td><td>Reviews</td><td>5,726</td><td>5,178</td><td>10,904</td></tr><tr><td>Total</td><td></td><td>27,989</td><td>27,688</td><td>55,677</td></tr><tr><td colspan="5"></td></tr><tr><td colspan="5"><b>Spanish Domain</b></td></tr><tr><td>Training subset</td><td>Legal</td><td>4,846</td><td>4,358</td><td>9,204</td></tr><tr><td></td><td>Tweets</td><td>5,739</td><td>5,634</td><td>11,373</td></tr><tr><td></td><td>How-to</td><td>5,690</td><td>5,795</td><td>11,485</td></tr><tr><td>Testing subset</td><td>News</td><td>5,514</td><td>5,223</td><td>10,737</td></tr><tr><td></td><td>Reviews</td><td>5,695</td><td>3,697</td><td>9,392</td></tr><tr><td>Total</td><td></td><td>27,484</td><td>24,707</td><td>52,191</td></tr></tbody></table>

The AuTexTification 2023 dataset analysis

AI-generated texts in the English subset are produced by multiple generative models, introducing stylistic and structural diversity that challenges detection systems beyond single-model artifacts. With a substantial number of labeled instances and a near-balanced ratio between human and AI text, the English dataset provides a strong foundation for training and evaluating detection models, as well as for analyzing baseline performance in a high-resource language. The Spanish domain includes texts labeled as human-authored or AI-generated, spanning multiple domains and generated by various AI models. This subset captures linguistic features specific to Spanish, such as richer morphology and syntactic variation, which pose additional challenges for detecting AI-generated text.

The AuTexTification 2023 dataset provides BERT and RoBERTa models with some major challenges where the train and test environments differ vastly. The first challenge will be the multi-domain generalization gap; since the models are trained on an unbalanced mix of informal tweets, encyclopedic factual Wiki entries, and overly technical legal documents but evaluated against news and reviews, there is a heightened possibility of feature leakage. The second challenge is the variability of text lengths; while BERT-based models are designed to work with short, dense contexts like tweets, they require sufficient tokenization strategies to allow them to deal with longer, more complex sequences typically found in news articles and reviews, while still retaining the "global" context of the passage.

### 3.3 The Proposed Method

The proposed framework for detecting AI-generated text employs transformer-based Deep Learning models. The framework can be modular, easily modified and extended, and can be reproduced to generate consistent results. By combining the RoBERTa architecture for solid text representations with careful data preparation, k-fold cross-validation, and thorough evaluation methods, the entire process is outlined in Figure [2](#fig2). The experimental workflow for this method is systematic and begins with data preparation, leading to final evaluation and analysis. The main objectives of this framework are to provide a robust method for identifying AI-generated text from multiple datasets, to alleviate the effects of overfitting with the use of k-fold cross-validation, to utilize Automatic Mixed Precision (AMP) to effectively optimize training time, and to support transparent evaluation through the operation of standard classification metrics.

![](https://dl.acm.org/cms/10.1145/3812734.3813713/asset/45c15f15-bfc3-44a9-a5d8-e13730637c38/assets/images/large/icics2026-86-fig2.jpg)

The proposed RoBERTa-based AI-generated text detection framework.

The first stage of the proposed method focuses on preparing the textual data for model training and evaluation. The data preparation process includes data loading, text preprocessing, dataset analysis, and constructing a data loader. To tokenize text input for use with transformers, the framework uses a pre-trained RoBERTa tokenizer from Hugging Face’s model repository, then initializes a RoBERTa-based classifier for fine-tuning. The optimizer and learning rate scheduler are also set up as part of the training environment, to ensure that training progresses smoothly and converges stably. K-fold cross-validation is also used as part of the training process to help improve generalization to other datasets and to reduce bias from the specific dataset used in training.

In k-fold cross-validation, the dataset is divided into k partitions or "folds." Automatic Mixed Precision (AMP) is used during training to reduce memory usage and improve training speed while maintaining numerical stability. During training, accuracy and loss are monitored for each fold, which is used to produce one overall estimate of the model’s performance. After completing k-fold cross-validation, the framework selects the best-performing hyperparameter combination based on the averaged validation results across all folds. The model is then retrained using the optimal hyperparameters on the complete training dataset. The final trained model is then evaluated on an unseen test set (held-out or held out) to determine its ability to generalize to new or real-world data. The evaluation results are summarized in a confusion matrix and standard classification measures (i.e., accuracy, precision, recall, and F1-score).

### 3.4 Evaluation Metrics

We evaluate our model using accuracy, precision, recall, and F1-score, complemented by confusion matrices and cross-validation statistics. Given the evolving nature of AI-generated text and potential class imbalance, the F1-score is emphasized for robust assessment. Additionally, we tune and report optimal classification thresholds to adaptively balance false positives and negatives, which is critical for reliable AI-generated text detection in real-world scenarios \[[^17]\] \[[^23]\]. The following evaluation metrics are used: Accuracy: Proportion of total correct predictions, and good when classes are balanced, Precision (Positive Predictive Value): Proportion of predicted positives that are positive, used to minimize false positives, Recall (Sensitivity): Proportion of actual positives that are correctly identified and used to minimize false negatives, F1-Score: Harmonic mean of precision and recall, and used to balance both false positives and false negatives. Confusion Matrix: Gives a complete picture of True Positives (TP), False Positives (FP), True Negatives (TN), and False Negatives (FN)\[[^17]\].

### 3.5 Data Preparation

The data preparation phase for the AuTexTification dataset is designed to address the linguistic variability of English and Spanish while mitigating the challenges posed by short text sequences and domain shifts. Given that the Spanish subset features complex morphological structures and the English subset is characterized by "Short Text Dominance," a unified yet language-sensitive preprocessing pipeline is essential. Initial steps involve cleaning the text to remove non-linguistic noise, such as specialized tokens, metadata, or artifacts from the generative process, ensuring that the RoBERTa model focuses purely on semantic and syntactic features rather than formatting biases.

For handling the specific requirements of the RoBERTa architecture, tokenization is conducted using language-specific subword tokenizers (WordPiece for English and the BETO tokenizer for Spanish). To manage the variance in document length, we implement a strategic padding and truncation approach. Since the training sets are dominated by texts under 20 tokens (particularly in the English Tweet domain), but evaluation sets include longer News and Legal articles, we set a maximum sequence length that preserves the narrative flow of longer documents while preventing excessive computational overhead. Furthermore, class balancing techniques are applied to ensure that the near-perfect label distribution observed in the training sets (approximately 50% Human and 50% AI-generated) is maintained across all domain-specific folds, providing a stable foundation for the model to learn the subtle "statistical signatures" of machine-generated text in both languages.

### 3.6 Training Configuration

The training process was conducted using PyTorch on an NVIDIA RTX 3080 GPU, equipped with 12 GB of GDDR6X RAM and 8960 cores. A classification head was integrated into the final layer of BERT, specifically tailored for binary classification (Human-written versus AI-generated) for each domain (English and Spanish). A total of 10 training trials were conducted to identify suitable hyperparameters configurations for the proposed models.

During these trials, a broad search space was initially explored and subsequently refined around optimal regions. The learning rate varied over a wide range (from 1 × 10 <sup>− 5</sup> to 5 × 10 <sup>− 4</sup>), followed by a focused search near the best-performing values. Weight decay parameters were examined within the range of 0 to 0.1 to mitigate overfitting. Higher weight decay values (0.175 for English and 0.2 for Spanish) were selected based on empirical tuning to reduce overfitting observed during preliminary experiments. Batch sizes of 16, 32, and 64 were evaluated to balance computational efficiency and gradient stability.

The number of training epochs ranged from 3 to 10, with early stopping applied based on validation loss improvements to prevent overfitting. In addition, the classification threshold for the positive class probability was tuned beyond the default value of 0.5 to optimize the F1-score and achieve a better balance between precision and recall. To further address potential class imbalance, focal loss was incorporated in selected experiments, with the focal-alpha parameter varied in the range of 0.25 to 0.75 and focal-gamma selected from 1.0,2.0, and 5.0.

Finally, model robustness was assessed using a Stratified K-Fold cross-validation strategy, with the number of splits typically set between 2 and 5. The optimum training parameters that achieve the best accuracy for the RoBERTa model for English and Spanish are shown in Table [3](#tab3).

<table><tbody><tr><td><b>Parameter</b></td><td><b>English</b></td><td><b>Spanish</b></td></tr><tr><td>Model Architecture</td><td colspan="2">RoBERTa</td></tr><tr><td>Number of Labels</td><td colspan="2">2</td></tr><tr><td>Learning Rate</td><td>2 × 10 <sup>− 5</sup></td><td>3 × 10 <sup>− 5</sup></td></tr><tr><td>Batch Size</td><td>64</td><td>64</td></tr><tr><td>Epochs</td><td>10</td><td>100</td></tr><tr><td>Max Sequence Length</td><td>100</td><td>10</td></tr><tr><td>Weight Decay</td><td>0.175</td><td>0.2</td></tr><tr><td>Dropout Probability</td><td>0.3</td><td>0.4</td></tr><tr><td>Label Smoothing</td><td>0.1</td><td>0.1</td></tr><tr><td>Cross-Validation</td><td>5-fold</td><td>5-fold</td></tr><tr><td>Threshold Tuning</td><td>Dynamic</td><td>Dynamic</td></tr></tbody></table>

Overview of the hyperparameters and settings used during model training.

## 4 Results and Discussion

### 4.1 English Domain Results

The English evaluation was conducted on a test set of 21,832 samples, maintaining a balanced distribution between 10,642 human-authored and 11,190 AI-generated texts. This practice ensures a robust and unbiased assessment of the model. The fine-tuned BERT model achieved an overall accuracy of 0.6087, characterized by a precision of 0.7206 for human text and a notably high recall of 0.8812 for AI-generated content. The model results are as in Table [4](#tab4). The RoBERTa model on the English subset achieves the highest overall performance, with a macro-averaged precision of 0.84 and recall of 0.83, indicating balanced detection of both human-written and AI-generated text. Its superior F1-score (0.84) confirms strong class-wise robustness, while the average accuracy of 0.82 demonstrates consistent generalization through cross-validation folds.

| **Metric** | **Score** |
| --- | --- |
| Average Overall Accuracy | 0.82 |
| Precision | 0.84 |
| Recall | 0.83 |
| F1-score | 0.84 |

Performance results of the model on the English subset of the dataset

In the confusion matrix as shown in Figure [3](#fig3), the model has accurately recognized 83% of AI-generated text and 81% of human-generated text, with a low percentage of misclassified AI-generated text (false negative) at 17% and human-generated text at 19% (false positive). The overall distribution of AI and human-generated text evaluation results indicates a balance between measuring the recall versus the precision of a project; AI-generated text has a recall rate of 84% and a precision of 83%, and the overall model accuracy is 82%, as shown in Figure [4](#fig4). This represents consistent results across classes with relatively few false negatives, indicating a strength of the overall model to detect and identify AI-generated written text relative to other non-AI source written materials. Therefore, the model is effective for reliably detecting AI-texts accurately.

![](https://dl.acm.org/cms/10.1145/3812734.3813713/asset/f38f93c3-c6f6-4914-a5c2-805127c00f1d/assets/images/large/icics2026-86-fig3.jpg)

The Model’s confusion matrix over the English subset.

![](https://dl.acm.org/cms/10.1145/3812734.3813713/asset/dbc2ec84-445b-4464-a0ca-10ad327891db/assets/images/large/icics2026-86-fig4.jpg)

The model Performance over the English subset.

### 4.2 Spanish Domain Results

The Spanish subset evaluation was conducted on a test set of 20,129 samples, consisting of a balanced distribution of 9,865 human-authored and 10,246 machine-generated texts to ensure a rigorous assessment of the model’s performance. The model results are shown in Table [5](#tab5). According to the confusion matrix of Spanish subset evaluation, as shown in Figure [5](#fig5), the RoBERTa model classifies 7,992 AI-generated texts (true positive) and 7,057 human-written texts (true negative) correctly, while incorrectly classifying 2,808 human-written texts as AI-generated (false positive) and 2,254 AI-generated texts as human-written (false negative). This gives a total accurate rate of 0.79, a precision of 0.74, and a recall rate of 0.78, which displays an acceptable tradeoff between false positives and false negatives, as shown in Figure [6](#fig6).

| **Metric** | **Score** |
| --- | --- |
| Average Overall Accuracy | 0.79 |
| Precision | 0.74 |
| Recall | 0.78 |
| F1-score | 0.75 |

Performance results of the model on the Spanish subset of the dataset

![](https://dl.acm.org/cms/10.1145/3812734.3813713/asset/d3396d56-a2c8-4564-815b-03f076db842b/assets/images/large/icics2026-86-fig5.jpg)

The Model’s confusion matrix over the Spanish subset.

![](https://dl.acm.org/cms/10.1145/3812734.3813713/asset/40feb44c-93fa-4a7c-a26d-9f03f0622ac2/assets/images/large/icics2026-86-fig6.jpg)

The model Performance over the Spanish subset.

Low false positive numbers indicate the extent of difficulty that the RoBERTa model had distinguishing between human-written and AI-generated Spanish text; therefore its relatively higher recall shows better ability to detect AI-generated Spanish text than a moderate amount of false positives indicates that there is much greater syntax variation and linguistic complexity in Spanish compared to English, and as such supports prior research indicating that RoBERTa will perform reasonably well in producing robust output on Spanish datasets, although its performance was lower than what would be expected with the same degree of certainty when using an English dataset, and therefore is consistent with the performance expectations found in both multi-linguistic and lower-resource language studies.

### 4.3 Discussion

Experimental results show that transformer-based architectures like RoBERTa deliver very robust performance when detecting AI-generated text in both English and Spanish subsets of the AuTexTification dataset. The evaluation on the English Subset used an unbiased evaluation; this evaluation was conducted on a Balanced Test Set (balanced number of sample sizes for each language) with 21,832 samples providing an equal sample size for both languages; therefore, the evaluation results provide a very good representation. Furthermore, RoBERTa provided substantially higher Precision, Recall, F1 Score, and Accuracy, at 0.84, 0.83, 0.84, and 0.82 macro averaging, respectively. Additionally, the Confusion Matrix demonstrates that the RoBERTa model provided an 83% True Positive Rate for AI-generated text and 81% True Positive Rate for Human-Written text; this means that the false negative rate for RoBERTa was 17% and the false positive rate was 19%, indicating a good balance between Precision and Recall. These results confirm that RoBERTa is highly robust and reliable for detecting AI-Generated Text in English.

When compared with recent work on the AuTexTification English domain \[[^21]\], The proposed RoBERTa-based approach is competitive with several established baselines and mid-ranking systems, as shown in Table [6](#tab6). Specifically, its macro-F1 score of 0.84 exceeds the performance of commonly reported baseline methods such as BOW+LR (65.78), Transformer baseline (57.10), and Random baseline (50.00), and is also substantially higher than several participating systems ranked below the top tier. The top-ranked system in shared task conditions (TALN-UPF, HB-plus) has a higher macro-F1 score of 80.91; however, many top systems use extensive feature engineering, ensembling methods, or specific optimizations for tasks. By contrast, the approach proposed here uses one fine-tuned transformer and applies uniform thresholds across all experiments. The intent is to emphasize generalizability and reproducibility compared to optimizing the leader board position.

| **Method** | **Macro-F1 (%)** |
| --- | --- |
| TALN-UPF, HB\_plus | 80.91 |
| TALN-UPF, HP | 74.16 |
| CIC-IPN-CsCog | 74.13 |
| turquoise\_titans | 65.79 |
| BOW+LR (Baseline) | 65.78 |
| turing\_testers | 60.64 |
| LDSE (Baseline) | 60.35 |
| Transformer (Baseline) | 57.10 |
| Random (Baseline) | 50.00 |
| UAEMex | 33.87 |
| **The Proposed Method** | **84.00** |

Comparison AuTexTification 2023 shared task English domain

Out of the 20,129 samples evaluated within the Spanish subset, half were created by humans, whereas half were generated by AI. Using the RoBERTa model to classify these samples yielded an accuracy score of 0.79; a precision score of 0.74; a recall score of 0.78; and an F1 score of 0.75. The confusion matrix resulting from the model indicated that there were 7992 correctly identified AI-generated documents and 7057 documents incorrectly identified as being written by humans.

Furthermore, since the model still has a high Recall score, this indicates that there is still sufficient sensitivity within the RoBERTa model to detect! AI-generated Spanish-language text. Lastly, the higher level of 20% compared to the English text type is due to the greater complexity of Spanish’s language structure, including more variation in morphology and syntax than in English.

In comparison with recent studies on the Spanish domain of the AuTexTification dataset, as shown in Table [7](#tab7), the achieved macro-F1 score of 0.75 places the proposed approach above several baseline systems, including BOW+LR (62.40), LDSE baseline (63.58), and Random baseline (50.00), while remaining competitive with transformer-based baselines reported in the shared task. Although the top-ranked systems in Spanish (e.g., TALN-UPF with a macro-F1 of 70.77) exhibit strong performance under shared-task conditions, the observed performance gap between English and Spanish in this study is consistent with prior multi-lingual and lower-resource language research.

RoBERTa is fairly effective at detecting AI-generated text in both English and Spanish, but its ability to detect AI-generated text is limited by the different types of languages and the fact that it does not have as much training data for Spanish as it does for English, as evidenced by the results of this study. The performance gap between English and Spanish suggests sensitivity of transformer-based detectors to linguistic variability, which aligns with prior multilingual NLP findings. The comparison to systems from the AuTexTification 2023 shared task should not be viewed as strictly equivalent since each participating system has unique preprocessing pipelines, training data, hyperparameter tuning, or use of external resources/ensembles. Therefore, results from Tables [6](#tab6) and [7](#tab7) are to be understood as benchmark performance comparisons at best rather than absolute, soundly controlled benchmarks. The primary goal of this comparison is to contextualize it relative to the broader space of existing approaches.

| **Team** | **Macro-F1 (%)** |
| --- | --- |
| TALN-UPF | 70.77 |
| Ling\_UCM | 70.60 |
| Transformer (Baseline) | 68.52 |
| GLPSI | 63.90 |
| LDSE (Baseline) | 63.58 |
| BOW+LR (Baseline) | 62.40 |
| bucharest | 56.49 |
| Random (Baseline) | 50.00 |
| LKE\_BUAP | 31.60 |
| **The Proposed Method** | **75.00** |

Comparison AuTexTification 2023 shared task Spanish domain.

## 5 Conclusions

In this research paper, we evaluated the ability of transformer-based architectures to identify AI-generated text against human-generated text across a variety of languages via the AuTexTification 2023 dataset. This work proposes a RoBERTa-based system for detecting AI-generated text, which shows very strong performance and generalizes well between both domains and languages. The RoBERTa-based system reached an accuracy of 82% and an F1 score of 0.84 for the English dataset, meaning that this model can accurately differentiate between texts written by humans and texts generated by AI in high-resource languages. Based on the confusion matrix analysis, RoBERTa correctly identified 83% of the AI-generated texts and 81% of the human-written texts from the English dataset, showing that the precision and recall for human-written texts have a balanced trade-off (17% for false negatives and 19% for false positives). For the Spanish dataset, the RoBERTa system reached an accuracy of 79% and an F1 score of 0.75 by correctly identifying 7,992 AI-generated and 7,057 human-written texts out of a possible 20,129. Although the overall performance is lower in Spanish than in English, our performance results are still competitive to past project benchmark values and exceed several baseline systems described in the AuTexTification shared task. The comparison further shows that our proposed approach works better than other baseline systems like BOW+LR, which achieved 65.78 F1; LDSE baseline (63.58 F1); or Random (50 F1). Thus, this shows that contextual transformer representations can be more effective tools to detect very subtle or complex stylistic and/or semantic signals related to machine-generated content. However, there remain significant disparities in performance between the two languages (9% better performance for English than Spanish), which demonstrates that there are substantial challenges regarding multi-lingual generalization due to morphological differences. Future work should be directed toward incorporating larger multilingual datasets and domain-diverse corpora, as this study does not perform cross-lingual transfer, which remains an important direction for future work.

[^1]: J. Alghamdi, Y. Lin, and S. Luo. 2025. ABERT: Adapting BERT model for efficient detection of human and AI-generated fake news. *International Journal of Information Management Data Insights* 5, 2 (2025), 100353.

[Go to Citation](#core-Bib0001-1)

[Crossref](https://doi.org/10.1016/j.jjimei.2025.100353)

[Google Scholar](https://scholar.google.com/scholar_lookup?title=ABERT%3A+Adapting+BERT+model+for+efficient+detection+of+human+and+AI-generated+fake+news&author=J.+Alghamdi&author=Y.+Lin&author=S.+Luo&publication_year=2025&pages=100353&doi=10.1016%2Fj.jjimei.2025.100353)

[^2]: W. Chen, Zoran Milosevic, Fethi A. Rabhi, and Andrew Berry. 2023. Real-time analytics: Concepts, architectures, and ML/AI considerations. *IEEE Access* 11 (2023), 71634–71657.

[Go to Citation](#core-Bib0002-1)

[Crossref](https://doi.org/10.1109/ACCESS.2023.3296845)

[Google Scholar](https://scholar.google.com/scholar_lookup?title=Real-time+analytics%3A+Concepts%2C+architectures%2C+and+ML%2FAI+considerations&author=W.+Chen&author=Zoran+Milosevic&author=Fethi%C2%A0A.+Rabhi&author=Andrew+Berry&publication_year=2023&pages=71634-71657&doi=10.1109%2FACCESS.2023.3296845)

[^3]: M. Elamine, A. Mekki, and L. H. Belguith. 2023. A Method for Identifying Bot Generated Text: Notebook for IberLEF 2023. In *IberLEF@SEPLN*. [http://ceur-ws.org/Vol-3496](http://ceur-ws.org/Vol-3496)

[Go to Citation](#core-Bib0003-1)

[Google Scholar](https://scholar.google.com/scholar_lookup?title=A+Method+for+Identifying+Bot+Generated+Text%3A+Notebook+for+IberLEF+2023&author=M.+Elamine&author=A.+Mekki&author=L.%C2%A0H.+Belguith&publication_year=2023)

[^4]: A. M. Elkhatat, K. Elsaid, and S. Almeer. 2023. Evaluating the efficacy of AI content detection tools in differentiating between human and AI-generated text. *International Journal for Educational Integrity* 19, 1 (2023), 1–16.

[Go to Citation](#core-Bib0004-1)

[Crossref](https://doi.org/10.1007/s40979-023-00112-2)

[Google Scholar](https://scholar.google.com/scholar_lookup?title=Evaluating+the+efficacy+of+AI+content+detection+tools+in+differentiating+between+human+and+AI-generated+text&author=A.%C2%A0M.+Elkhatat&author=K.+Elsaid&author=S.+Almeer&publication_year=2023&pages=1-16&doi=10.1007%2Fs40979-023-00112-2)

[^5]: C. Espin-Riofrio, J. Ortiz-Zambrano, and A. Montejo-Ráez. 2023. SINAI at AuTexTification in IberLEF 2023: Combining All Layer Embeddings for Automatically Generated Texts. In *IberLEF@SEPLN*. [http://ceur-ws.org/Vol-3496](http://ceur-ws.org/Vol-3496)

[Go to Citation](#core-Bib0005-1)

[Google Scholar](https://scholar.google.com/scholar_lookup?title=SINAI+at+AuTexTification+in+IberLEF+2023%3A+Combining+All+Layer+Embeddings+for+Automatically+Generated+Texts&author=C.+Espin-Riofrio&author=J.+Ortiz-Zambrano&author=A.+Montejo-R%C3%A1ez&publication_year=2023)

[^6]: C. Espin-Riofrio, J. Ortiz-Zambrano, and A. Montejo-Ráez. 2024. SINAI at IberAuTexTification in IberLEF 2024: Perplexity Metrics and Text Features for Classifying Automatically Generated Text. In *IberLEF@SEPLN*. [http://ceur-ws.org/Vol-3496](http://ceur-ws.org/Vol-3496)

[Go to Citation](#core-Bib0006-1)

[Google Scholar](https://scholar.google.com/scholar_lookup?title=SINAI+at+IberAuTexTification+in+IberLEF+2024%3A+Perplexity+Metrics+and+Text+Features+for+Classifying+Automatically+Generated+Text&author=C.+Espin-Riofrio&author=J.+Ortiz-Zambrano&author=A.+Montejo-R%C3%A1ez&publication_year=2024)

[^7]: Z. Feng. 2023. Past and present of natural language processing. In *Formal Analysis for Natural Language Processing: A Handbook*. Springer, 3–48.

[Go to Citation](#core-Bib0007-1)

[Crossref](https://doi.org/10.1007/978-981-16-5172-4_1)

[Google Scholar](https://scholar.google.com/scholar_lookup?title=Past+and+present+of+natural+language+processing&author=Z.+Feng&publication_year=2023&pages=3-48&doi=10.1007%2F978-981-16-5172-4_1)

[^8]: A. Fernández-Hernández, J. L. Arboledas-Márquez, J. Ariza-Merino, and S. M. J. Zafra. 2023. Taming the Turing Test: Exploring Machine Learning Approaches to Discriminate Human vs. AI-Generated Texts. In *IberLEF@SEPLN*. [http://ceur-ws.org/Vol-3496](http://ceur-ws.org/Vol-3496)

[Go to Citation](#core-Bib0008-1)

[Google Scholar](https://scholar.google.com/scholar_lookup?title=Taming+the+Turing+Test%3A+Exploring+Machine+Learning+Approaches+to+Discriminate+Human+vs.+AI-Generated+Texts&author=A.+Fern%C3%A1ndez-Hern%C3%A1ndez&author=J.%C2%A0L.+Arboledas-M%C3%A1rquez&author=J.+Ariza-Merino&author=S.%C2%A0M.%C2%A0J.+Zafra&publication_year=2023)

[^9]: R. Gagiano, H. Fayek, M. M.-H. Kim, J. Biggs, and X. Zhang. 2023. IberLEF 2023 AuTexTification: Automated Text Identification Shared Task-Team OD-21. In *IberLEF@SEPLN*. [http://ceur-ws.org/Vol-3496](http://ceur-ws.org/Vol-3496)

[Go to Citation](#core-Bib0009-1)

[Google Scholar](https://scholar.google.com/scholar_lookup?title=IberLEF+2023+AuTexTification%3A+Automated+Text+Identification+Shared+Task-Team+OD-21&author=R.+Gagiano&author=H.+Fayek&author=M.%C2%A0M.-H.+Kim&author=J.+Biggs&author=X.+Zhang&publication_year=2023)

[^10]: J. F. García and I. Segura-Bedmar. 2024. Human After All: Using Transformer Based Models to Identify Automatically Generated Text. In *Proceedings of the IberLEF*. [http://ceur-ws.org/Vol-3496](http://ceur-ws.org/Vol-3496)

[Go to Citation](#core-Bib0010-1)

[Google Scholar](https://scholar.google.com/scholar_lookup?title=Human+After+All%3A+Using+Transformer+Based+Models+to+Identify+Automatically+Generated+Text&author=J.%C2%A0F.+Garc%C3%ADa&author=I.+Segura-Bedmar&publication_year=2024)

[^11]: D. Ghiurău and D. E. Popescu. 2024. Distinguishing reality from AI: approaches for detecting synthetic content. *Computers* 14, 1 (2024), 1.

[Go to Citation](#core-Bib0011-1)

[Crossref](https://doi.org/10.3390/computers14010001)

[Google Scholar](https://scholar.google.com/scholar_lookup?title=Distinguishing+reality+from+AI%3A+approaches+for+detecting+synthetic+content&author=D.+Ghiur%C4%83u&author=D.%C2%A0E.+Popescu&publication_year=2024&pages=1&doi=10.3390%2Fcomputers14010001)

[^12]: G. Gritsay, A. Grabovoy, A. Kildyakov, and Y. Chekhovich. 2023. Automated Text Identification: Multilingual Transformer-based Models Approach. In *IberLEF@SEPLN*. [http://ceur-ws.org/Vol-3496](http://ceur-ws.org/Vol-3496)

[Go to Citation](#core-Bib0012-1)

[Google Scholar](https://scholar.google.com/scholar_lookup?title=Automated+Text+Identification%3A+Multilingual+Transformer-based+Models+Approach&author=G.+Gritsay&author=A.+Grabovoy&author=A.+Kildyakov&author=Y.+Chekhovich&publication_year=2023)

[^13]: M. U. Hadi, R. Qureshi, A. Shah, M. Irfan, A. Zafar, M. B. Shaikh, N. Akhtar, J. Wu, and S. Mirjalili. 2023. Large language models: a comprehensive survey of its applications, challenges, limitations, and future prospects. *Authorea Preprints* 1, 3 (2023), 1–26. [https://arxiv.org/abs/2303.18223](https://arxiv.org/abs/2303.18223)

[Go to Citation](#core-Bib0013-1)

[Google Scholar](https://scholar.google.com/scholar_lookup?title=Large+language+models%3A+a+comprehensive+survey+of+its+applications%2C+challenges%2C+limitations%2C+and+future+prospects&author=M.%C2%A0U.+Hadi&author=R.+Qureshi&author=A.+Shah&author=M.+Irfan&author=A.+Zafar&author=M.%C2%A0B.+Shaikh&author=N.+Akhtar&author=J.+Wu&author=S.+Mirjalili&publication_year=2023&pages=1-26)

[^14]: H. Huang, T. Tang, D. Zhang, W. X. Zhao, T. Song, Y. Xia, and F. Wei. 2023. Not all languages are created equal in LLMs: Improving multilingual capability by cross-lingual-thought prompting. *arXiv preprint* (2023). [https://arxiv.org/abs/2305.12070](https://arxiv.org/abs/2305.12070)

[Go to Citation](#core-Bib0014-1)

[Google Scholar](https://scholar.google.com/scholar_lookup?title=Not+all+languages+are+created+equal+in+LLMs%3A+Improving+multilingual+capability+by+cross-lingual-thought+prompting&author=H.+Huang&author=T.+Tang&author=D.+Zhang&author=W.%C2%A0X.+Zhao&author=T.+Song&author=Y.+Xia&author=F.+Wei&publication_year=2023)

[^15]: Y. Liu, M. Ott, N. Goyal, J. Du, M. Joshi, D. Chen, O. Levy, M. Lewis, L. Zettlemoyer, and V. Stoyanov. 2019. RoBERTa: A robustly optimized BERT pretraining approach. *arXiv preprint* (2019). [https://arxiv.org/abs/1907.11692](https://arxiv.org/abs/1907.11692)

[Go to Citation](#core-Bib0015-1)

[Google Scholar](https://scholar.google.com/scholar_lookup?title=RoBERTa%3A+A+robustly+optimized+BERT+pretraining+approach&author=Y.+Liu&author=M.+Ott&author=N.+Goyal&author=J.+Du&author=M.+Joshi&author=D.+Chen&author=O.+Levy&author=M.+Lewis&author=L.+Zettlemoyer&author=V.+Stoyanov&publication_year=2019)

[^16]: D. Myers, R. Mohawesh, V. I. Chellaboina, A. L. Sathvik, P. Venkatesh, Y.-H. Ho, H. Henshaw, M. Alhawawreh, D. Berdik, and Y. Jararweh. 2024. Foundation and large language models: fundamentals, challenges, opportunities, and social impacts. *Cluster Computing* 27, 1 (2024), 1–26. [https://arxiv.org/abs/2402.06196](https://arxiv.org/abs/2402.06196)

[Go to Citation](#core-Bib0016-1)

[Google Scholar](https://scholar.google.com/scholar_lookup?title=Foundation+and+large+language+models%3A+fundamentals%2C+challenges%2C+opportunities%2C+and+social+impacts&author=D.+Myers&author=R.+Mohawesh&author=V.%C2%A0I.+Chellaboina&author=A.%C2%A0L.+Sathvik&author=P.+Venkatesh&author=Y.-H.+Ho&author=H.+Henshaw&author=M.+Alhawawreh&author=D.+Berdik&author=Y.+Jararweh&publication_year=2024&pages=1-26)

[^17]: D. M. Powers. 2020. Evaluation: from precision, recall and F-measure to ROC, informedness, markedness and correlation. *arXiv preprint* (2020). [https://arxiv.org/abs/2010.16061](https://arxiv.org/abs/2010.16061)

[Google Scholar](https://scholar.google.com/scholar_lookup?title=Evaluation%3A+from+precision%2C+recall+and+F-measure+to+ROC%2C+informedness%2C+markedness+and+correlation&author=D.%C2%A0M.+Powers&publication_year=2020)

[^18]: A.-A. Preda, D.-C. Cercel, T. Rebedea, and C.-G. Chiru. 2023. UPB at IberLEF-2023 AuTexTification: Detection of Machine-Generated Text using Transformer Ensembles. In *arXiv preprint*. [https://arxiv.org/abs/2306.05755](https://arxiv.org/abs/2306.05755)

[Go to Citation](#core-Bib0018-1)

[Google Scholar](https://scholar.google.com/scholar_lookup?title=UPB+at+IberLEF-2023+AuTexTification%3A+Detection+of+Machine-Generated+Text+using+Transformer+Ensembles&author=A.-A.+Preda&author=D.-C.+Cercel&author=T.+Rebedea&author=C.-G.+Chiru&publication_year=2023)

[^19]: A. M. Sarvazyan, J. Á. González, M. Franco-Salvador, F. Rangel, B. Chulvi, and P. Rosso. 2023. Overview of AuTexTification at IberLEF 2023: Detection and attribution of machine-generated text in multiple domains. *arXiv preprint* (2023). [https://arxiv.org/abs/2309.11298](https://arxiv.org/abs/2309.11298)

[Google Scholar](https://scholar.google.com/scholar_lookup?title=Overview+of+AuTexTification+at+IberLEF+2023%3A+Detection+and+attribution+of+machine-generated+text+in+multiple+domains&author=A.%C2%A0M.+Sarvazyan&author=J.%C2%A0%C3%81.+Gonz%C3%A1lez&author=M.+Franco-Salvador&author=F.+Rangel&author=B.+Chulvi&author=P.+Rosso&publication_year=2023)

[^20]: T. Scheibe and T. Mandl. 2023. Univ. of Hildesheim at AuTexTification 2023: Detection of Automatically Generated Texts. In *IberLEF@SEPLN*. [http://ceur-ws.org/Vol-3496](http://ceur-ws.org/Vol-3496)

[Go to Citation](#core-Bib0020-1)

[Google Scholar](https://scholar.google.com/scholar_lookup?title=Univ.+of+Hildesheim+at+AuTexTification+2023%3A+Detection+of+Automatically+Generated+Texts&author=T.+Scheibe&author=T.+Mandl&publication_year=2023)

[^21]: M. K. Sheykhlan, S. K. Abdoljabbar, and M. N. Mahmoudabad. 2024. KaramiTeam at IberAuTexTification: Soft Voting Ensemble for Distinguishing AI-Generated Texts. In *CEUR Workshop Proceedings, Proceedings of the Iberian Languages Evaluation Forum (IberLEF 2024)*. Valladolid, Spain. [http://ceur-ws.org/Vol-3740](http://ceur-ws.org/Vol-3740)

[Google Scholar](https://scholar.google.com/scholar_lookup?title=KaramiTeam+at+IberAuTexTification%3A+Soft+Voting+Ensemble+for+Distinguishing+AI-Generated+Texts&author=M.%C2%A0K.+Sheykhlan&author=S.%C2%A0K.+Abdoljabbar&author=M.%C2%A0N.+Mahmoudabad&publication_year=2024)

[^22]: L. A. Simón, J. A. G. Gimeno, A. M. F.-P. Cesteros, M. F. Trinidad, and M. V. E. Vidal. 2023. Using Linguistic Knowledge for Automated Text Identification. In *IberLEF@SEPLN*. [http://ceur-ws.org/Vol-3496](http://ceur-ws.org/Vol-3496)

[Go to Citation](#core-Bib0022-1)

[Google Scholar](https://scholar.google.com/scholar_lookup?title=Using+Linguistic+Knowledge+for+Automated+Text+Identification&author=L.%C2%A0A.+Sim%C3%B3n&author=J.%C2%A0A.%C2%A0G.+Gimeno&author=A.%C2%A0M.+F.-P.+Cesteros&author=M.%C2%A0F.+Trinidad&author=M.%C2%A0V.%C2%A0E.+Vidal&publication_year=2023)

[^23]: M. Sokolova and G. Lapalme. 2009. A systematic analysis of performance measures for classification tasks. *Information Processing & Management* 45, 4 (2009), 427–437.

[Go to Citation](#core-Bib0023-1)

[Digital Library](https://dl.acm.org/doi/10.1016/j.ipm.2009.03.002)

[Google Scholar](https://scholar.google.com/scholar_lookup?title=A+systematic+analysis+of+performance+measures+for+classification+tasks&author=M.+Sokolova&author=G.+Lapalme&publication_year=2009&pages=427-437&doi=10.1016%2Fj.ipm.2009.03.002)

[^24]: I. Tenney, D. Das, and E. Pavlick. 2019. BERT rediscovers the classical NLP pipeline. *arXiv preprint* (2019). [https://arxiv.org/abs/1905.05950](https://arxiv.org/abs/1905.05950)

[Go to Citation](#core-Bib0024-1)

[Google Scholar](https://scholar.google.com/scholar_lookup?title=BERT+rediscovers+the+classical+NLP+pipeline&author=I.+Tenney&author=D.+Das&author=E.+Pavlick&publication_year=2019)

[^25]: Z. Villegas-Trejo, H. Gómez-Adorno, S.-L. Ojeda-Trueba, M. Montes-y Gomez, F. Rangel, S. Jimenez-Zafra, M. Casavantes, B. Altuna, M. Alvarez-Carmona, and G. Bel-Enguix. 2023. Exploring Text Representations for Detecting Automatically Generated Text. In *IberLEF@SEPLN*. [http://ceur-ws.org/Vol-3496](http://ceur-ws.org/Vol-3496)

[Go to Citation](#core-Bib0025-1)

[Google Scholar](https://scholar.google.com/scholar_lookup?title=Exploring+Text+Representations+for+Detecting+Automatically+Generated+Text&author=Z.+Villegas-Trejo&author=H.+G%C3%B3mez-Adorno&author=S.-L.+Ojeda-Trueba&author=M.+Montes-y+Gomez&author=F.+Rangel&author=S.+Jimenez-Zafra&author=M.+Casavantes&author=B.+Altuna&author=M.+Alvarez-Carmona&author=G.+Bel-Enguix&publication_year=2023)

[^26]: H. Wang, J. Li, and Z. Li. 2024. AI-generated text detection and classification based on BERT deep learning algorithm. *arXiv preprint* (2024). [https://arxiv.org/abs/2401.01234](https://arxiv.org/abs/2401.01234)

[Go to Citation](#core-Bib0026-1)

[Google Scholar](https://scholar.google.com/scholar_lookup?title=AI-generated+text+detection+and+classification+based+on+BERT+deep+learning+algorithm&author=H.+Wang&author=J.+Li&author=Z.+Li&publication_year=2024)

[^27]: Z. Wang, J. Cheng, C. Cui, and C. Yu. 2023. Implementing BERT and fine-tuned RoBERTa to detect AI generated news by ChatGPT. *arXiv preprint* (2023). [https://arxiv.org/abs/2305.05698](https://arxiv.org/abs/2305.05698)

[Go to Citation](#core-Bib0027-1)

[Google Scholar](https://scholar.google.com/scholar_lookup?title=Implementing+BERT+and+fine-tuned+RoBERTa+to+detect+AI+generated+news+by+ChatGPT&author=Z.+Wang&author=J.+Cheng&author=C.+Cui&author=C.+Yu&publication_year=2023)

[^28]: Y. Zhou and J. Wang. 2024. Detecting AI-generated texts in cross-domains. In *Proceedings of the ACM Symposium on Document Engineering 2024*. 1–4.

[Go to Citation](#core-Bib0028-1)

[Digital Library](https://dl.acm.org/doi/10.1145/3685650.3685673)

[Google Scholar](https://scholar.google.com/scholar_lookup?title=Detecting+AI-generated+texts+in+cross-domains&author=Y.+Zhou&author=J.+Wang&publication_year=2024&pages=1-4&doi=10.1145%2F3685650.3685673)