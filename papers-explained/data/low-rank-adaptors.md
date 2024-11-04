---
markmap:
  color: "#2980b9"
  maxWidth: 400
  initialExpandLevel: 3
---

# Parameter-Efficient Fine-Tuning (PEFT)

## &nbsp;
### [LoRA](https://arxiv.org/abs/2106.09685): Low-Rank Adaptation [📑](https://ritvik19.medium.com/papers-explained-lora-a48359cecbfa) [📑](https://ritvik19.medium.com/papers-explained-review-06-parameter-efficient-finetuning-6934fafa74e5#05fa)

* **Purpose:** 
    * Efficiently adapt pre-trained LLMs to downstream tasks.
* **Method:** 
    * Freezes the pre-trained model weights (W).
    * Injects trainable low-rank matrices A and B into each Transformer layer, resulting in a low-rank update ∆W = BA.
    * Primarily focuses on adapting attention weights (Wq, Wk, Wv, Wo) while freezing MLP modules.
* **Advantages:**
    * **Drastically Reduced Parameters:** Up to 10,000 times fewer trainable parameters than full fine-tuning.
    * **Lower Memory Footprint:** Significant reduction in GPU memory requirements.
    * **No Inference Latency:** Merging trainable matrices with frozen weights during deployment eliminates additional inference time.
* **Limitations:**
    * **Batching Complexity:** Batching inputs with different A and B matrices for various tasks can be challenging.
    * **Potential Inadequacy:** Low-rank updates may not be sufficient for complex tasks requiring significant representation learning.

## Rank and Budget Allocation

### [DyLoRA](https://arxiv.org/abs/2210.07558): Dynamic Search-Free Low-Rank Adaptation [📑](https://ritvik19.medium.com/papers-explained-review-06-parameter-efficient-finetuning-6934fafa74e5#7fb6)
* **Purpose:** 
    * Solve the problem of optimal rank selection in LoRA.
* **Method:** 
    * Randomly samples a rank from a pre-defined range during training.
* **Advantage:** 
    * Eliminates the need for separate training for different rank values, providing flexibility and potentially better performance across a wider range of ranks.

### [AdaLoRA](https://arxiv.org/abs/2303.10512): Adaptive Budget Allocation [📑](https://ritvik19.medium.com/papers-explained-review-06-parameter-efficient-finetuning-6934fafa74e5#620f)
* **Purpose:** 
    * Adaptively allocate the parameter budget among weight matrices based on their importance. 
* **Method:** 
    * Parameterizes incremental updates using singular value decomposition (SVD).
    * Prunes singular values of less important updates to reduce their parameter budget.
* **Advantages:**
    * Improved performance, especially in low-budget settings.
    * Avoids intensive exact SVD computations.

### [DoRA](https://arxiv.org/abs/2402.09353): Decomposing LoRA into Single-Rank Components for Dynamic Pruning [📑](https://ritvik19.medium.com/papers-explained-review-06-parameter-efficient-finetuning-6934fafa74e5#028e)
* **Purpose:**  
    * Enhance fine-tuning efficiency by dynamically distributing the parameter budget.
* **Method:**
    * Decomposes high-rank LoRA layers into single-rank components.
    * Dynamically prunes components with smaller contributions during training, guided by importance scores.
    * Uses a regularization penalty to promote stable pruning.
* **Advantage:** 
    * Achieves competitive performance compared to LoRA and full fine-tuning with a more efficient parameter budget allocation.

## Memory Efficiency and Context Length

### [QLoRA](https://arxiv.org/abs/2305.14314): Quantized Low-Rank Adaptation [📑](https://ritvik19.medium.com/papers-explained-146-qlora-a6e7273bc630) [📑](https://ritvik19.medium.com/papers-explained-review-06-parameter-efficient-finetuning-6934fafa74e5#3992)
* **Purpose:** 
    * Reduce memory usage for fine-tuning large models.
* **Method:** 
    * Combines LoRA with 4-bit quantization of the pre-trained model and introduces memory management techniques:
    * **4-bit NormalFloat (NF4):** A new data type for quantizing normally distributed weights.
    * **Double Quantization:** Quantizes the quantization constants to further reduce memory.
    * **Paged Optimizers:** Manages memory spikes.
* **Advantage:** 
    * Enables fine-tuning of large models (e.g., 65B parameters) on a single GPU while preserving performance.

### [LoRA-FA](https://arxiv.org/abs/2308.03303): Freezing the Projection-Down Weight [📑](https://ritvik19.medium.com/papers-explained-review-06-parameter-efficient-finetuning-6934fafa74e5#c229)
* **Purpose:** 
    * Reduce activation memory costs during fine-tuning. 
* **Method:** 
    * Freezes the projection-down weight (A) and only updates the projection-up weight (B) in each LoRA layer.
* **Advantage:** 
    * Reduces overall memory cost (up to 1.4x compared to LoRA) without impacting performance or requiring expensive recomputation.

### [LongLoRA](https://arxiv.org/abs/2309.12307): Combining Shifted Sparse Attention and Improved LoRA [📑](https://ritvik19.medium.com/papers-explained-147-longlora-24f095b93611) [📑](https://ritvik19.medium.com/papers-explained-review-06-parameter-efficient-finetuning-6934fafa74e5#8108)
* **Purpose:** 
    * Extend the context size of pre-trained LLMs efficiently. 
* **Method:** 
    * Combines improved LoRA with Shifted Sparse Attention (S2-Attn).
    * **Improved LoRA:** Includes trainable embedding and normalization layers.
    * **S2-Attn:** Approximates long context training with sparse local attention, switching to full attention during inference.
* **Advantages:**
    * Extends context length significantly (e.g., Llama2 7B from 4k to 100k context). 
    * Retains original model architecture and compatibility with optimization techniques like Flash-Attention2.
    * Reduces memory and training time compared to full fine-tuning.

### [VeRA](https://arxiv.org/abs/2310.11454): Vector-based Random Matrix Adaptation [📑](https://ritvik19.medium.com/papers-explained-review-06-parameter-efficient-finetuning-6934fafa74e5#5bb3)

* **Purpose:** 
    * Further enhance parameter efficiency and reduce memory footprint compared to LoRA. 
* **Method:** 
    * Employs a single pair of frozen, randomly initialized matrices (A and B) shared across all layers. 
    * Utilizes trainable scaling vectors (b and d) to adapt each layer.
* **Advantages:**
    * **Extreme Parameter Reduction:** Requires significantly fewer trainable parameters than LoRA, especially at low ranks.
    * **Minimal Memory Requirements:**  Only needs to store the scaling vectors and a seed to regenerate the random matrices. 
    * **No Inference Latency:** Similar to LoRA, trained components can be merged with the original weights.

## Performance Enhancement

### [Delta-LoRA](https://arxiv.org/abs/2309.02411): Updating Pre-Trained Weights [📑](https://ritvik19.medium.com/papers-explained-review-06-parameter-efficient-finetuning-6934fafa74e5#a4ec)
* **Purpose:** 
    * Address the limitations of solely relying on low-rank updates in LoRA. 
* **Method:** 
    * Updates both the low-rank matrices (A, B) and the original pre-trained weights (W) using the delta of their product (A(t+1)B(t+1) - A(t)B(t)).
    * Removes the Dropout layer from the LoRA module to ensure accurate delta calculation.
* **Advantage:** 
    * Achieves better performance than LoRA and other low-rank adaptation methods by incorporating the original weights in the adaptation process.

### [LoRA+](https://arxiv.org/abs/2402.12354): Different Learning Rates for A and B [📑](https://ritvik19.medium.com/papers-explained-review-06-parameter-efficient-finetuning-6934fafa74e5#fd31)
* **Purpose:**  
    * Improve feature learning in LoRA.
* **Method:** 
    * Employs different learning rates for the A and B matrices based on theoretical analysis of the infinite-width limit. 
* **Advantage:** 
    * Enhances performance compared to standard LoRA.

### [MoRA](https://arxiv.org/abs/2405.12130): High-Rank Updating with Square Matrices [📑](https://ritvik19.medium.com/papers-explained-review-06-parameter-efficient-finetuning-6934fafa74e5#21a4)
* **Purpose:** 
    * Improve performance, especially on memory-intensive tasks, by enabling high-rank updates. 
* **Method:** 
    * Uses a square matrix for updating instead of the low-rank matrices in LoRA, while maintaining a similar number of trainable parameters.
    * Introduces non-parameter operators to reduce input and increase output dimensions for the square matrix, allowing for merging with the original model.
* **Advantages:**
    * Outperforms LoRA on memory-intensive tasks.
    * Achieves comparable performance on other tasks like instruction tuning and mathematical reasoning.