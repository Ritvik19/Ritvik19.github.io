---
markmap:
  color: "#2980b9"
  maxWidth: 400
  initialExpandLevel: 3
---

# Vision Transformers

## &nbsp;
### [Vision Transformer (ViT)](https://arxiv.org/abs/2010.11929) [📑](https://ritvik19.medium.com/papers-explained-25-vision-transformers-e286ee8bc06b)
#### Purpose
- Apply the Transformer architecture directly to images for image classification.
#### Method
- Splits an image into fixed-size patches.
- Linearly embeds each patch and adds position embeddings.
- Feeds the sequence of embedded patches into a standard Transformer encoder.
- Uses an extra learnable "classification token" for classification.
#### Advantages
- Achieves excellent results on image classification tasks, comparable to state-of-the-art convolutional networks, while requiring fewer computational resources.
- Benefits from the scalability and efficiency of NLP Transformer architectures and their implementations.
- Self-attention allows for global information integration from the lowest layers, increasing attention distance with network depth.

## Studies Related to Vision Transformer

### [What do Vision Transformers Learn? A Visual Exploration](https://arxiv.org/abs/2212.06727) [📑]()

#### Purpose
* **Understanding ViT Mechanisms:** Investigate the internal workings of Vision Transformers (ViTs) and the features they learn, as limited understanding exists compared to Convolutional Neural Networks (CNNs).
* **Visualize ViT Features:** Overcome the challenges of visualizing ViT representations and establish effective methods for visual exploration.
* **Compare ViTs and CNNs:** Explore the similarities and differences in behavior between ViTs and CNNs, particularly regarding feature extraction and reliance on image components.
* **Analyze Impact of Language Supervision:** Study the effect of training ViTs with language model supervision, such as CLIP, on the types of features they extract.

#### Method
* **Visualization Framework:**
    * **Activation Maximization:** Apply gradient descent to synthesize images that maximally activate specific neurons, similar to techniques used for CNNs.
    * **Improved Visualization Techniques:** Incorporate methods like total variation regularization, jitter augmentation, color shift augmentation, augmentation ensembling, and Gaussian smoothing to enhance image quality.
    * **Pairing with ImageNet Examples:** Supplement visualizations with images from the ImageNet dataset that strongly activate the corresponding feature, providing context and interpretability.
    * **Patch-wise Activation Maps:** Generate activation maps by passing images through the network, revealing how different patches contribute to feature activation and highlighting spatial information preservation.
* **Model Selection:**
    * **ViT-B16 as Primary Model:** Focus on ViT-B16, a widely used variant, for detailed demonstrations and analysis.
    * **Extensive Visualization of Other Variants:** Validate the visualization method on a wide range of ViT architectures, including DeiT, CoaT, ConViT, PiT, Swin, and Twin, to ensure generalizability.
* **Targeted Analysis:**
    * **Spatial Information Preservation:** Investigate how patch tokens maintain spatial information throughout the network layers, examining activation patterns and comparing with CNN behavior.
    * **Last-Layer Token Mixing:** Analyze the role of the last attention block in globalizing information and its similarity to average pooling in CNNs. 
    * **Background and Foreground Feature Reliance:**  Compare ViT and CNN performance when selectively masking foreground or background information in images, using ImageNet bounding boxes to assess dependence on different image components.
    * **Impact of High-Frequency Information:** Evaluate the sensitivity of ViTs and CNNs to the removal of high-frequency image content through low-pass filtering, revealing differences in reliance on texture information.
    * **CLIP-Trained ViT Feature Analysis:** Examine the emergence of abstract concept detectors in ViTs trained with language supervision, contrasting them with object-specific features in image-classifier ViTs.

#### Findings
* **Spatial Information Preservation (Except Last Layer):** ViTs preserve spatial information in patch tokens throughout all layers except the final attention block. This behavior contrasts with CNNs where spatial information is restricted by the receptive field size.
* **Last Layer as Learned Global Pooling:** The final attention block in ViTs functions similarly to average pooling in CNNs, mixing information from all patches and discarding spatial localization.
* **Effective Background Information Usage:** ViTs leverage background information more effectively than CNNs for classification and are more resilient to foreground removal.
* **Reduced Reliance on High-Frequency Information:** ViTs are less sensitive to the loss of high-frequency image content compared to CNNs, indicating less dependence on texture for prediction.
* **Abstract Concept Detection in CLIP-Trained Models:** Language model supervision leads ViTs to develop features that detect abstract concepts, parts of speech, and conceptual categories, going beyond object-specific features found in image classifiers.

### [ConvNets Match Vision Transformers at Scale](https://arxiv.org/abs/2310.16764) [📑]()

#### Purpose 
* **Challenge the Belief that Vision Transformers Outperform ConvNets at Scale:** Many believe that while ConvNets excel on small to moderate datasets, they fall short compared to Vision Transformers when trained on web-scale datasets. This research aims to challenge this assumption.
* **Evaluate the Scaling Properties of ConvNets on Large Datasets:** The study seeks to determine if ConvNets can achieve comparable performance to Vision Transformers when provided with similar computational resources and training data.
* **Re-examine the Effectiveness of ConvNets in the Era of Large-Scale Pre-Training:** The research investigates whether ConvNets can benefit significantly from pre-training on massive datasets like Vision Transformers do.

#### Method
* **Model Selection: NFNets:**  The study utilizes the NFNet family of models, a purely convolutional architecture known for its high performance. The choice stems from the fact that NFNets were among the last ConvNets to achieve state-of-the-art results on ImageNet, demonstrating their potential.
* **Dataset: JFT-4B:**  NFNets are trained on the JFT-4B dataset, a large-scale dataset containing approximately 4 billion labeled images across 30,000 classes. This dataset is commonly used for training foundation models, allowing for a fair comparison with Vision Transformers trained on similar data.
* **Compute Budgets:**  The research explores a range of compute budgets for pre-training, from 0.4k to 110k TPU-v4 core compute hours. This approach enables a comparison of model performance across various levels of computational resources.
* **Training Procedure:**
    * **Varying Depth and Width:** NFNets of different depths and widths are trained to investigate the impact of model size on performance.
    * **Epoch Budgets and Learning Rate Tuning:** Models are trained with various epoch budgets, and the learning rate is meticulously tuned for each budget to optimize performance.
    * **Training Optimizations:** The training leverages techniques like Stochastic Gradient Descent with Momentum, Adaptive Gradient Clipping (AGC), and removal of near-duplicate images from ImageNet in JFT-4B.
* **Fine-tuning on ImageNet:** To assess the generalizability of the pre-trained models, they are fine-tuned on the ImageNet dataset using techniques like sharpness aware minimization (SAM), stochastic depth, and dropout.

#### Findings
* **Scaling Laws:**  A clear log-log scaling law is observed between validation loss on JFT-4B and the compute budget used for pre-training. This finding mirrors the scaling laws previously observed in language modeling with transformers, suggesting that ConvNets exhibit similar scaling behavior.
* **Model Size and Epoch Budget:**  The optimal model size and epoch budget, leading to the lowest validation loss, increase with the available compute budget. This observation suggests that larger models and longer training durations are beneficial when more computational resources are available.
* **Learning Rate Trends:**  The optimal learning rate, minimizing validation loss, exhibits predictable behavior. It tends to be consistent for small epoch budgets but decreases gradually as the epoch budget and model size increase.
* **Competitive Performance on ImageNet:**  Fine-tuned NFNets achieve ImageNet Top-1 accuracy levels comparable to Vision Transformers trained with similar compute budgets. This key finding challenges the notion of Vision Transformers being superior at scale.
* **Impact of Pre-training:**  Large-scale pre-training significantly boosts the performance of NFNets on ImageNet, highlighting the importance of large datasets for training ConvNets.
* **Fine-tuning Considerations:**  The study observes that pre-trained models with the lowest validation loss on JFT-4B don't always translate to the highest Top-1 accuracy on ImageNet after fine-tuning.  Fine-tuning often favors slightly larger models and shorter epoch budgets, potentially due to the larger models' capacity to adapt to the new task.

## Enhancing Efficiency and Performance

### [Data-efficient Image Transformer (DeiT)](https://arxiv.org/abs/2012.12877) [📑](https://ritvik19.medium.com/papers-explained-39-deit-3d78dd98c8ec)
#### Purpose:
- Improve data efficiency of ViT and achieve competitive performance with less training data.
#### Method:
- Employs improved training strategies and distillation. 
#### Advantages:
- Reduces training time and data requirements compared to ViT.
- Achieves performance comparable to CNNs with similar complexity and efficiency.

### [Efficient ViT](https://arxiv.org/abs/2305.07027) [📑](https://ritvik19.medium.com/papers-explained-229-efficient-vit-cc87fbefbe49)
#### Purpose:
- Improve the efficiency of vision transformers by addressing memory access and computation redundancy.
#### Method:
- Designs a new building block with a sandwich layout, using a single memory-bound MHSA layer between efficient FFN layers.
- Introduces a cascaded group attention (CGA) module that feeds attention heads with different splits of the full feature to enhance diversity and reduce computation.
- Redistributes parameters to prioritize critical network components like value projections while shrinking less important ones like hidden dimensions in FFNs.
#### Advantages:
- Achieves a good balance between speed and accuracy, outperforming existing efficient models.
- Demonstrates good transferability to various downstream tasks.

### [Shape-Optimized Vision Transformer (SoViT)](https://arxiv.org/abs/2305.13035) [📑](https://ritvik19.medium.com/papers-explained-234-sovit-a0ce3c7ef480)
#### Purpose:
- Optimize the shape (width and depth) of vision transformer models to achieve compute-optimal designs.
#### Method:
- Extends and refines existing scaling law methods to optimize multiple shape dimensions jointly, requiring fewer experiments.
- Applies this method to ViT, optimizing the model for a compute budget equivalent to ViT-g/14. 
#### Advantages:
- Results in a smaller, faster model with the same quality as a much larger ViT model trained with the same compute.
- Achieves competitive results across multiple tasks, including image classification, captioning, VQA, and zero-shot transfer.

## Adapting ViT for Diverse Vision Tasks

### [Swin Transformer](https://arxiv.org/abs/2103.14030) [📑](https://ritvik19.medium.com/papers-explained-26-swin-transformer-39cf88b00e3e)
#### Purpose:
- Adapt Transformer for a wider range of vision tasks, including dense prediction tasks.
#### Method:
- Introduces a hierarchical structure with shifted windows for multi-scale processing.
- Employs shifted window attention to limit self-attention computation to local windows while allowing for cross-window connections.
#### Advantages
- Achieves linear computational complexity with respect to image size.
- Suitable as a general-purpose backbone for various vision tasks, including image classification and dense prediction tasks.
- Achieves state-of-the-art performance on various vision benchmarks.

### [Swin Transformer V2](https://arxiv.org/abs/2111.09883) [📑](https://ritvik19.medium.com/papers-explained-215-swin-transformer-v2-53bee16ab668)
#### Purpose:
- Scale up Swin Transformer to handle larger model capacity and higher resolution images.
#### Method:
- Introduces a residual-post-norm method and cosine attention to improve training stability.
- Employs a log-spaced continuous position bias (Log-CPB) to transfer models pre-trained with low-resolution images to downstream tasks with high-resolution inputs.
- Uses a self-supervised pre-training method, SimMIM, to reduce the need for large amounts of labeled data.
- Incorporates techniques like zero-optimizer, activation checkpointing, and sequential self-attention computation to address memory issues.
#### Advantages:
- Enables training of larger Swin Transformer models (up to 3 billion parameters) with higher resolution images (up to 1536x1536).
- Achieves state-of-the-art performance on various vision benchmarks with less training data and time compared to other large vision models.


## Hybridizing with Convolutional Techniques

### [Convolutional Vision Transformer (CvT)](https://arxiv.org/abs/2103.15808) [📑](https://ritvik19.medium.com/papers-explained-199-cvt-fb4a5c05882e)
#### Purpose:
- Improve ViT by combining the strengths of CNNs and Transformers.
#### Method:
- Introduces convolutional operations for token embedding, incorporating shift, scale, and distortion invariance.
- Replaces linear projections with depth-wise separable convolutions in the transformer block.
#### Advantages:
- Achieves state-of-the-art performance on ImageNet-1k with fewer parameters and lower FLOPs compared to other ViTs and ResNets.
- Maintains performance gains when pre-trained on larger datasets and fine-tuned to downstream tasks.
- Eliminates the need for positional encoding, simplifying the design and allowing for variable input resolutions.

### [LeViT (Vision Transformer in ConvNet's Clothing)](https://arxiv.org/abs/2104.01136) [📑](https://ritvik19.medium.com/papers-explained-205-levit-89a2defc2d18)
#### Purpose:
- Optimize the trade-off between accuracy and efficiency for high-speed inference in ViTs.
#### Method:
- Adopts a pyramid structure with pooling, inspired by LeNet, for efficient multi-resolution processing.
- Introduces the attention bias, a new way to integrate positional information.
#### Advantages:
- Offers better speed-accuracy trade-offs than ViT/DeiT models, particularly for small and medium-sized architectures.
- Achieves better inference speed on GPUs, CPUs, and ARM hardware.

### [FastViT](https://arxiv.org/abs/2303.14189) [📑](https://ritvik19.medium.com/papers-explained-225-fastvit-f1568536ed34)
#### Purpose:
- Design a hybrid vision transformer model with a strong latency-accuracy trade-off.
#### Method:
- Combines transformers and convolutional layers to efficiently capture both local and global information.
- Uses RepMixer, a token mixer with structural reparameterization, to reduce memory access costs.
- Employs factorized convolutions with train-time overparameterization in the stem and patch embedding layers for improved efficiency and performance.
#### Advantages:
- Aims for state-of-the-art latency-accuracy trade-off in vision transformer models.

### [Multi-Axis Vision Transformer (MaxViT)](https://arxiv.org/abs/2204.01697) [📑](https://ritvik19.medium.com/papers-explained-210-maxvit-6c68cc515413)
#### Purpose:
- Design an efficient and scalable vision transformer that can handle high-resolution images.
#### Method:
- Introduces multi-axis attention, which combines blocked local and dilated global attention for capturing both local and global spatial interactions.
- Blends multi-axis attention with convolutions in a simple hierarchical backbone architecture.
#### Advantages:
- Achieves linear complexity with respect to image size while allowing for global-local spatial interactions at arbitrary input resolutions.
- Excels on a wide range of vision tasks, including image classification, object detection, and visual aesthetic assessment.
- Demonstrates strong generative modeling capabilities.

## Exploring Self-Supervised Learning

### [BERT Pre-Training of Image Transformers (BEiT)](https://arxiv.org/abs/2106.08254) [📑](https://ritvik19.medium.com/papers-explained-27-beit-b8c225496c01)
#### Purpose:
- Pre-train vision transformers effectively using a masked image modeling task inspired by BERT in NLP.
#### Method:
- Proposes a masked image modeling (MIM) pre-training task.
- Uses two views of each image: image patches and visual tokens (discrete tokens) obtained from a discrete variational autoencoder (VAE).
- Masks some image patches and tasks the model with predicting the original visual tokens based on the corrupted image patches.
#### Advantages:
- Achieves competitive results on image classification and semantic segmentation tasks compared to previous pre-training methods.
- Enables self-supervised pre-training of vision transformers, leveraging large-scale image data without labels.
- Demonstrates the ability to distinguish semantic regions and object boundaries without human annotation.

### [Masked AutoEncoder (MAE)](https://arxiv.org/abs/2111.06377) [📑](https://ritvik19.medium.com/papers-explained-28-masked-autoencoder-38cb0dbed4af)
#### Purpose:
- Explore self-supervised learning for vision using a masked autoencoding approach.
#### Method:
- Masks a high portion of random patches in an image.
- Tasks the model with reconstructing the missing patches based on the visible ones.
#### Advantages:
- Encourages holistic understanding of images by requiring the model to infer missing information.
- Reduces redundancy in the input data and creates a challenging self-supervisory task.
- The design of the decoder plays a crucial role in determining the semantic level of the learned representations.

## Optimizing for Mobile and Edge Devices

### [MobileViT](https://arxiv.org/abs/2110.02178) [📑](https://ritvik19.medium.com/papers-explained-40-mobilevit-4793f149c434)
#### Purpose: 
- Design an efficient and scalable vision transformer that can handle high-resolution images.
#### Method:
- Introduces multi-axis attention, which combines blocked local and dilated global attention for capturing both local and global spatial interactions.
- Blends multi-axis attention with convolutions in a simple hierarchical backbone architecture.
#### Advantages:
- Achieves linear complexity with respect to image size while allowing for global-local spatial interactions at arbitrary input resolutions.
- Excels on a wide range of vision tasks, including image classification, object detection, and visual aesthetic assessment.
- Demonstrates strong generative modeling capabilities.

### [EfficientFormer](https://arxiv.org/abs/2206.01191) [📑](https://ritvik19.medium.com/papers-explained-220-efficientformer-97c91540af19)
#### Purpose:
- Optimize vision transformer models for fast inference speed on resource-constrained edge devices.
#### Method:
- Performs a comprehensive latency analysis of ViT and its variants on mobile devices.
- Introduces a dimension-consistent design paradigm, using 4D MetaBlocks and 3D MHSA blocks.
- Employs a latency-driven slimming method to search for optimal model configurations based on mobile inference speed.
#### Advantages:
- Achieves ultra-fast inference speed while maintaining competitive performance on image classification, object detection, and segmentation tasks.
- Outperforms existing transformer models in terms of speed and accuracy.