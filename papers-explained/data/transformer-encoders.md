---
markmap:
  color: "#2980b9"
  maxWidth: 400
  initialExpandLevel: 3
---

# Transformer Encoders

## &nbsp;
### [BERT](https://arxiv.org/abs/1810.04805) [📑](https://ritvik19.medium.com/papers-explained-02-bert-31e59abc0615)
#### Purpose:
* Pre-train deep bidirectional representations from unlabeled text for various NLP tasks.
* Improve fine-tuning based approaches by alleviating the unidirectionality constraint.
#### Method:
* Masked Language Model (MLM): Randomly mask tokens and predict them based on context.
* Next Sentence Prediction (NSP): Jointly pre-train text-pair representations.
#### Advantages:
* State-of-the-art results on various NLP tasks.
* Simple architecture and effective for both sentence-level and token-level tasks.
* Fine-tuning allows adaptation to a wide range of tasks without significant modifications.

## Improving BERT

### [RoBERTa](https://arxiv.org/abs/1907.11692) [📑](https://ritvik19.medium.com/papers-explained-03-roberta-81db014e35b9)
#### Purpose:
* Improve BERT pre-training by optimizing training parameters and data.
* Address the inconsistencies and missing details in the original BERT implementation.
#### Method:
* Larger batch sizes, longer training time, and more data.
* Dynamic masking instead of static masking.
* Removal of NSP task.
#### Advantages:
* Improved performance and robustness compared to BERT.
* More efficient pre-training process.
* Better generalization to downstream tasks.

### [DeBERTa](https://arxiv.org/abs/2006.03654) [📑](https://ritvik19.medium.com/papers-explained-08-deberta-a808d9b2c52d)
#### Purpose:
* Improve BERT and RoBERTa by enhancing attention mechanism and decoding.
* Achieve better performance on both NLU and NLG tasks.
#### Method:
* Disentangled attention: Separate vectors for content and position, disentangled attention matrices.
* Enhanced mask decoder: Incorporate absolute position embeddings in the decoding layer.
* Virtual adversarial training for fine-tuning.
#### Advantages:
* Significant performance gains on a wide range of NLP tasks.
* Improved training efficiency.
* Better generalization and robustness.

## Compressing BERT

### [TinyBERT](https://arxiv.org/abs/1909.10351) [📑](https://ritvik19.medium.com/papers-explained-05-tiny-bert-5e36fe0ee173)
#### Purpose:
* Compress BERT to a smaller size and accelerate inference for resource-constrained devices.
* Maintain accuracy while reducing model complexity.
#### Method:
* Knowledge distillation from a larger BERT model.
* Two-stage learning: general distillation on a large corpus and task-specific distillation.
* Transformer distillation with multiple loss functions to fit different representations.
#### Advantages:
* Significant reduction in size and inference time compared to BERT.
* Retains a high percentage of the teacher BERT's performance.
* Flexible model configuration due to the two-stage distillation framework.

### [ALBERT](https://arxiv.org/abs/1909.11942) [📑](https://ritvik19.medium.com/papers-explained-07-albert-46a2a0563693)
#### Purpose:
* Reduce BERT's memory consumption and training time for larger models.
* Improve parameter efficiency and scalability.
#### Method:
* Factorized embedding parameterization to decouple embedding size from hidden size.
* Cross-layer parameter sharing to reduce the number of parameters.
* Sentence order prediction (SOP) as a self-supervised loss to improve inter-sentence coherence.
#### Advantages:
* Significant reduction in parameters and memory footprint.
* Faster training speed.
* Better performance on downstream tasks, especially those with multi-sentence inputs.

### [FastBERT](https://arxiv.org/abs/2004.02178) [📑](https://ritvik19.medium.com/papers-explained-37-fastbert-5bd246c1b432)
#### Purpose:
* Accelerate BERT inference by adapting the inference time based on input complexity.
* Maintain accuracy while reducing computational cost.
#### Method:
* Self-distillation with a student-teacher framework.
* Adaptive inference with early exit based on uncertainty estimation.
* Training with both original and distilled data.
#### Advantages:
* Significant speedup in inference time compared to BERT.
* Minimal accuracy loss.
* Suitable for industrial scenarios with varying input complexity.

### [MobileBERT](https://arxiv.org/abs/2004.02984) [📑](https://ritvik19.medium.com/papers-explained-36-mobilebert-933abbd5aaf1)
#### Purpose:
* Compress and accelerate BERT for deployment on resource-limited mobile devices.
* Maintain task-agnostic capabilities for fine-tuning on various NLP tasks.
#### Method:
* Bottleneck structure with reduced width but similar depth to BERTLARGE.
* Knowledge transfer from a specially designed teacher model (IB-BERT).
* Feature map transfer and attention transfer for layer-wise knowledge distillation.
#### Advantages:
* 4.3x smaller and 5.5x faster than BERTBASE.
* Competitive performance on GLUE and SQuAD benchmarks.
* Low latency on mobile devices.

## Distilled Versions of BERT/RoBERTa

### [DistilBERT](https://arxiv.org/abs/1910.01108) [📑](https://ritvik19.medium.com/papers-explained-06-distil-bert-6f138849f871)
#### Purpose:
* Create a smaller and faster version of BERT while retaining most of its capabilities.
* Enable efficient operation on edge devices and under resource constraints.
#### Method:
* Knowledge distillation from a larger BERT model during pre-training.
* Triple loss combining language modeling, distillation, and cosine-distance losses.
* Removal of token-type embeddings and pooler.
#### Advantages:
* 40% smaller and 60% faster than BERT while maintaining 97% of its performance.
* Cheaper to pre-train and suitable for on-device computations.
* Good performance on various tasks, including classification and question answering.

### [DistilRoBERTa](https://huggingface.co/distilbert/distilroberta-base) [📑](https://medium.com/dair-ai/papers-explained-06-distil-bert-6f138849f871#a260)
#### Purpose:
* Create a distilled version of RoBERTa with reduced size and faster inference.
* Maintain the benefits of RoBERTa while improving efficiency.
#### Method:
* Knowledge distillation from RoBERTa-base using a similar procedure to DistilBERT.
* Reduced number of layers compared to the teacher model.
#### Advantages:
* 82M parameters compared to 125M parameters for RoBERTa-base.
* Twice as fast as RoBERTa-base on average.
* Good performance on GLUE tasks after fine-tuning.

## Improving DeBERTa

### [DeBERTa v2](https://huggingface.co/docs/transformers/en/model_doc/deberta-v2) [📑](https://ritvik19.medium.com/papers-explained-08-deberta-a808d9b2c52d#f5e1)
#### Purpose:
* Build upon DeBERTa to further improve performance and efficiency.
#### Method:
* Introduces several optimization techniques, including improved parameter initialization and training procedures.
* The sources do not provide specific details on the new methods employed by DeBERTa V2.
#### Advantages:
* Outperforms DeBERTa on various NLP tasks.
* More efficient training with less data.
* The sources do not explicitly list the advantages of DeBERTa V2.

### [DeBERTa v3](https://arxiv.org/abs/2111.09543) [📑](https://ritvik19.medium.com/papers-explained-182-deberta-v3-65347208ce03)
#### Purpose:
* Enhance DeBERTa with ELECTRA-style pre-training and gradient-disentangled embedding sharing.
* Further improve efficiency and performance, particularly for smaller models.
#### Method:
* Replaced masked language modeling (MLM) with replaced token detection (RTD).
* Gradient-disentangled embedding sharing (GDES) for more efficient knowledge transfer.
* Combines RTD, GDES, and the disentangled attention mechanism from DeBERTa.
#### Advantages:
* Improved training efficiency and performance compared to DeBERTa.
* Especially effective for smaller models, achieving significant gains with reduced parameters.
* Demonstrates strong performance even with less data compared to RoBERTa and XLNet. 

## Sentence Embeddings

### [Sentence BERT](https://arxiv.org/abs/1908.10084) [📑](https://ritvik19.medium.com/papers-explained-04-sentence-bert-5159b8e07f21)
#### Purpose:
* Generate semantically meaningful sentence embeddings for tasks like similarity comparison.
* Improve efficiency compared to using BERT for pairwise sentence scoring.
#### Method:
* Siamese BERT network architecture to encode two sentences simultaneously.
* Fine-tuning on various objective functions, including classification and regression.
* Different pooling strategies to derive sentence embeddings from BERT outputs.
#### Advantages:
* State-of-the-art performance on sentence similarity and other tasks.
* Significantly faster inference compared to pairwise sentence scoring.
* Adaptable to specific tasks through fine-tuning.

### [ColBERT](https://arxiv.org/abs/2004.12832) [📑](https://medium.com/@ritvik19/papers-explained-88-colbert-fe2fd0509649)
#### Purpose:
* Achieve efficient and effective passage retrieval using contextualized late interaction over BERT.
* Improve the efficiency of BERT-based ranking models while maintaining high accuracy.
#### Method:
* Generates token-level embeddings for both queries and documents.
* Late interaction between query and document embeddings using a MaxSim operator.
* Efficient scoring and ranking based on the interaction matrix.
#### Advantages:
* Significantly faster than traditional BERT-based re-ranking models.
* Comparable or better retrieval effectiveness compared to more complex models.
* Suitable for large-scale retrieval tasks with high efficiency requirements.

### [ColBERT v2](https://arxiv.org/abs/2112.01488) [📑](https://ritvik19.medium.com/papers-explained-89-colbertv2-7d921ee6e0d9)
#### Purpose
* Improve Efficiency and Effectiveness of Late Interaction Retrieval
#### Method
* Denoised Supervision: distillation from a cross-encoder (MiniLM) and hard negative mining, to improve the model's robustness and accuracy by providing more informative training signals.
* Residual Compression: leverages the observation that ColBERT token embeddings tend to cluster around specific semantic centroids. 
* It compresses embeddings by storing the nearest centroid ID and a quantized residual vector, drastically reducing storage requirements.
#### Advantages
* Achieves state-of-the-art results on various in-domain and out-of-domain benchmarks.
* Residual compression achieves a 6-10x reduction in storage size compared to the original ColBERT.