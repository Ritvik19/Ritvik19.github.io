document.getElementById("title").innerHTML =
  "A Survey of Small Language Models";

((getMarkmap, getOptions, root2, jsonOptions) => {
  const markmap = getMarkmap();
  window.mm = markmap.Markmap.create(
    "svg#mindmap",
    (getOptions || markmap.deriveOptions)(jsonOptions),
    root2
  );
})(
  () => window.markmap,
  null,
  {
    content:
      '<a href="https://arxiv.org/abs/2410.20011">A Survey of Small Language Models</a>',
    children: [
      {
        content: "&nbsp;",
        children: [
          {
            content:
              "General techniques used for optimizing small language models, categorized by type of model optimization",
            children: [
              {
                content:
                  '<table data-lines="6,19">\n<thead data-lines="6,7">\n<tr data-lines="6,7">\n<th style="text-align:left">Technique</th>\n<th style="text-align:left">General Mechanism</th>\n<th style="text-align:center">Training Compute</th>\n<th style="text-align:center">Dataset Size</th>\n<th style="text-align:center">Inference Runtime</th>\n<th style="text-align:center">Memory</th>\n<th style="text-align:center">Storage Space</th>\n<th style="text-align:center">Latency</th>\n</tr>\n</thead>\n<tbody data-lines="8,19">\n<tr data-lines="8,9">\n<td style="text-align:left"><strong>Model Architectures</strong></td>\n<td style="text-align:left"></td>\n<td style="text-align:center"></td>\n<td style="text-align:center"></td>\n<td style="text-align:center"></td>\n<td style="text-align:center"></td>\n<td style="text-align:center"></td>\n<td style="text-align:center"></td>\n</tr>\n<tr data-lines="9,10">\n<td style="text-align:left"></td>\n<td style="text-align:left">Lightweight Models</td>\n<td style="text-align:center">✓</td>\n<td style="text-align:center"></td>\n<td style="text-align:center">✓</td>\n<td style="text-align:center">✓</td>\n<td style="text-align:center"></td>\n<td style="text-align:center">✓</td>\n</tr>\n<tr data-lines="10,11">\n<td style="text-align:left"></td>\n<td style="text-align:left">Efficient Self-Attention</td>\n<td style="text-align:center">✓</td>\n<td style="text-align:center"></td>\n<td style="text-align:center">✓</td>\n<td style="text-align:center">✓</td>\n<td style="text-align:center"></td>\n<td style="text-align:center">✓</td>\n</tr>\n<tr data-lines="11,12">\n<td style="text-align:left"></td>\n<td style="text-align:left">Neural Arch. Search</td>\n<td style="text-align:center"></td>\n<td style="text-align:center"></td>\n<td style="text-align:center">✓</td>\n<td style="text-align:center">✓</td>\n<td style="text-align:center">✓</td>\n<td style="text-align:center"></td>\n</tr>\n<tr data-lines="12,13">\n<td style="text-align:left"><strong>Training Techniques</strong></td>\n<td style="text-align:left"></td>\n<td style="text-align:center"></td>\n<td style="text-align:center"></td>\n<td style="text-align:center"></td>\n<td style="text-align:center"></td>\n<td style="text-align:center"></td>\n<td style="text-align:center"></td>\n</tr>\n<tr data-lines="13,14">\n<td style="text-align:left"></td>\n<td style="text-align:left">Pre-training</td>\n<td style="text-align:center">✓</td>\n<td style="text-align:center">✓</td>\n<td style="text-align:center">✓</td>\n<td style="text-align:center">✓</td>\n<td style="text-align:center">✓</td>\n<td style="text-align:center"></td>\n</tr>\n<tr data-lines="14,15">\n<td style="text-align:left"></td>\n<td style="text-align:left">Finetuning</td>\n<td style="text-align:center">✓</td>\n<td style="text-align:center">✓</td>\n<td style="text-align:center"></td>\n<td style="text-align:center"></td>\n<td style="text-align:center"></td>\n<td style="text-align:center"></td>\n</tr>\n<tr data-lines="15,16">\n<td style="text-align:left"><strong>Model Compression</strong></td>\n<td style="text-align:left"></td>\n<td style="text-align:center"></td>\n<td style="text-align:center"></td>\n<td style="text-align:center"></td>\n<td style="text-align:center"></td>\n<td style="text-align:center"></td>\n<td style="text-align:center"></td>\n</tr>\n<tr data-lines="16,17">\n<td style="text-align:left"></td>\n<td style="text-align:left">Pruning</td>\n<td style="text-align:center"></td>\n<td style="text-align:center"></td>\n<td style="text-align:center">✓</td>\n<td style="text-align:center">✓</td>\n<td style="text-align:center">✓</td>\n<td style="text-align:center">✓</td>\n</tr>\n<tr data-lines="17,18">\n<td style="text-align:left"></td>\n<td style="text-align:left">Quantization</td>\n<td style="text-align:center"></td>\n<td style="text-align:center"></td>\n<td style="text-align:center">✓</td>\n<td style="text-align:center">✓</td>\n<td style="text-align:center">✓</td>\n<td style="text-align:center">✓</td>\n</tr>\n<tr data-lines="18,19">\n<td style="text-align:left"></td>\n<td style="text-align:left">Knowledge Distillation</td>\n<td style="text-align:center"></td>\n<td style="text-align:center">✓</td>\n<td style="text-align:center"></td>\n<td style="text-align:center"></td>\n<td style="text-align:center"></td>\n<td style="text-align:center"></td>\n</tr>\n</tbody>\n</table>',
                children: [],
                payload: { lines: "6,19" },
              },
            ],
            payload: { lines: "4,5" },
          },
        ],
        payload: { lines: "3,4" },
      },
      {
        content: "Model Architectures",
        children: [
          {
            content: "Lightweight Architectures",
            children: [
              {
                content:
                  '<a href="https://arxiv.org/abs/2308.02019">BabyLLaMA</a> and <a href="https://arxiv.org/abs/2409.17312">BabyLLaMA-2</a> distill knowledge from multiple teachers into a 58M and a 345M model respectively, demonstrating that distillation can exceed teacher models’ performance particularly under data-constrained conditions.',
                children: [],
                payload: { lines: "23,24" },
              },
              {
                content:
                  '<a href="https://arxiv.org/abs/2401.02385">TinyLLaMA</a> <a href="https://ritvik19.medium.com/papers-explained-93-tinyllama-6ef140170da9">📑</a> uses FlashAttention (Dao et al., 2022) to optimize memory overhead while maintaining competitive performance for various downstream tasks.',
                children: [],
                payload: { lines: "24,25" },
              },
              {
                content:
                  '<a href="https://arxiv.org/abs/2402.16840">MobilLLaMA</a> applies a parameter-sharing scheme that reduces both pretraining and deployment costs.',
                children: [],
                payload: { lines: "25,26" },
              },
              {
                content:
                  '<a href="https://arxiv.org/abs/2402.14905">Mobile LLM</a> <a href="https://ritvik19.medium.com/papers-explained-216-mobilellm-2d7fdd5acd86">📑</a>introduces embedding-sharing and grouped-query attention mechanisms with block-wise weight sharing to reduce latency.',
                children: [],
                payload: { lines: "26,28" },
              },
            ],
            payload: { lines: "22,23" },
          },
          {
            content: "Efficient Self-Attention Approximations",
            children: [
              {
                content:
                  '<a href="https://arxiv.org/abs/2001.04451">Reformer</a> <a href="https://ritvik19.medium.com/papers-explained-165-reformer-4445ad305191">📑</a> Improves self-attention complexity from O(N^2) to O(N log N) by replacing dot product attention with locality-sensitivity hashing.',
                children: [],
                payload: { lines: "29,30" },
              },
              {
                content:
                  '<a href="https://arxiv.org/abs/2003.05997">Routing Transformer</a> Uses a Sparse Routing Module based on online k-means clustering to reduce the complexity of the attention computation.',
                children: [],
                payload: { lines: "30,31" },
              },
              {
                content:
                  '<a href="https://arxiv.org/abs/2006.04768">Linformer</a> Expresses self-attention as a linear dot-product of kernel feature maps, reducing quadratic complexity.',
                children: [],
                payload: { lines: "31,32" },
              },
              {
                content:
                  '<a href="https://arxiv.org/abs/2006.16236">Fast Autoregressive Transformers with Linear Attention</a> Expresses self-attention as a linear dot-product of kernel feature maps, reducing quadratic complexity and viewing transformers as recurrent neural networks for faster inference.',
                children: [],
                payload: { lines: "32,33" },
              },
              {
                content:
                  '<a href="https://arxiv.org/abs/2102.03902">Nyströmformer</a> Uses the Nystrom method to approximate the self-attention operation with strong empirical performance.',
                children: [],
                payload: { lines: "33,34" },
              },
              {
                content:
                  '<a href="https://arxiv.org/abs/2312.00752">Mamba</a> introduces a selective state space model with input-dependent transitions for linear time and space complexity.',
                children: [],
                payload: { lines: "34,35" },
              },
              {
                content:
                  '<a href="https://arxiv.org/abs/2305.13048">RWKV</a> combines elements of transformers and RNNs with a linear attention mechanism for linear time and space complexity.',
                children: [],
                payload: { lines: "35,36" },
              },
              {
                content:
                  '<a href="https://arxiv.org/abs/2004.05150">Longformer</a> <a href="https://ritvik19.medium.com/papers-explained-38-longformer-9a08416c532e">📑</a> Uses a combination of local windowed attention and task-specific global attention, scaling linearly with input length for memory efficiency.',
                children: [],
                payload: { lines: "36,38" },
              },
            ],
            payload: { lines: "28,29" },
          },
          {
            content: "Neural Architecture Search Techniques",
            children: [
              {
                content:
                  '<a href="https://arxiv.org/abs/2402.14905">Mobile LLM</a> <a href="https://ritvik19.medium.com/papers-explained-216-mobilellm-2d7fdd5acd86">📑</a> investigates the impact of model depth (i.e., number of layers) and width (i.e., number of heads) on performance, effectively conducting a targeted architecture search within a smaller parameter range.',
                children: [],
                payload: { lines: "39,40" },
              },
              {
                content:
                  '<a href="https://arxiv.org/abs/2409.17372">Search for Efficient Large Language Models</a> reduce the search space by exploring an appropriate initialization for the search.',
                children: [],
                payload: { lines: "40,42" },
              },
            ],
            payload: { lines: "38,39" },
          },
          {
            content: "Small Multi-modal Models",
            children: [
              {
                content:
                  '<a href="https://internvl.github.io/blog/2024-07-02-InternVL-2.0/">InternVL2</a> Utilizes outputs from intermediate layers of large visual encoders instead of relying solely on the final outputs. This reduces the size and complexity of the visual encoder while maintaining performance.',
                children: [],
                payload: { lines: "43,44" },
              },
              {
                content:
                  '<a href="https://arxiv.org/abs/2407.07726">PaliGemma</a> <a href="https://ritvik19.medium.com/papers-explained-197-pali-gemma-6899e871998e">📑</a> Employs smaller and more efficient vision encoder architectures compared to traditional large-scale encoders. This reduces the computational cost and memory footprint of the model.',
                children: [],
                payload: { lines: "44,45" },
              },
              {
                content:
                  "Monolithic Multi-Modal Models completely eliminates the separate visual encoder and instead uses lightweight architectures to generate visual tokens directly.",
                children: [],
                payload: { lines: "45,46" },
              },
              {
                content:
                  '<a href="https://arxiv.org/abs/2405.09818">Chameleon</a> <a href="https://ritvik19.medium.com/papers-explained-143-chameleon-6cddfdbceaa8">📑</a> Employs a VQ-VAE model to encode and decode images into discrete tokens.',
                children: [],
                payload: { lines: "46,47" },
              },
              {
                content:
                  '<a href="https://arxiv.org/abs/2410.08202">Mono-InternVL</a>  Uses an MLP to generate visual tokens for image patches and incorporates a modality-specific feedforward network (multi-modal Mixture-of-Experts) to differentiate between modalities.',
                children: [],
                payload: { lines: "47,49" },
              },
            ],
            payload: { lines: "42,43" },
          },
        ],
        payload: { lines: "20,21" },
      },
      {
        content: "Training Techniques",
        children: [
          {
            content: "Pre-training Techniques",
            children: [
              {
                content: "Mixed Precision Training Techniques",
                children: [
                  {
                    content: "Automatic Mixed Precision (AMP):",
                    children: [
                      {
                        content:
                          "Keeps a master copy of weights in 32-bit floating-point (FP32) precision.",
                        children: [],
                        payload: { lines: "54,55" },
                      },
                      {
                        content:
                          "Performs arithmetic operations in 16-bit floating-point (FP16) precision.",
                        children: [],
                        payload: { lines: "55,56" },
                      },
                    ],
                    payload: { lines: "53,56" },
                  },
                  {
                    content: "Brain Floating Point (BFLOAT16):",
                    children: [
                      {
                        content:
                          "Offers a greater dynamic range with more exponent bits than FP16.",
                        children: [],
                        payload: { lines: "57,58" },
                      },
                      {
                        content:
                          "Demonstrates superior training performance and representation accuracy compared to FP16.",
                        children: [],
                        payload: { lines: "58,59" },
                      },
                    ],
                    payload: { lines: "56,59" },
                  },
                  {
                    content: "8-bit Floating-Point (FP8):",
                    children: [
                      {
                        content:
                          "Supported by NVIDIA's latest Hopper architecture.",
                        children: [],
                        payload: { lines: "60,61" },
                      },
                      {
                        content:
                          "Enables even greater computational efficiency for large-scale language models.",
                        children: [],
                        payload: { lines: "61,63" },
                      },
                    ],
                    payload: { lines: "59,63" },
                  },
                ],
                payload: { lines: "52,53" },
              },
              {
                content: "Optimization and Stability Techniques",
                children: [
                  {
                    content:
                      "Adam and AdamW Optimizers: Commonly used optimizers for training language models.",
                    children: [],
                    payload: { lines: "64,65" },
                  },
                  {
                    content:
                      "Memory-Efficient Optimizers: Adafactor and Sophia improve training speed and efficiency.",
                    children: [],
                    payload: { lines: "65,66" },
                  },
                  {
                    content:
                      "Gradient Clipping: Prevents exploding gradients and stabilizes training.",
                    children: [],
                    payload: { lines: "66,68" },
                  },
                ],
                payload: { lines: "63,64" },
              },
              {
                content: "Distributed Training Techniques",
                children: [
                  {
                    content: "Zero Redundancy Data Parallelism (ZeRO):",
                    children: [
                      {
                        content: "Three stages of optimization:",
                        children: [
                          {
                            content: "ZeRO-1 partitions optimizer states.",
                            children: [],
                            payload: { lines: "71,72" },
                          },
                          {
                            content: "ZeRO-2 adds gradient partitioning.",
                            children: [],
                            payload: { lines: "72,73" },
                          },
                          {
                            content:
                              "ZeRO-3 further partitions model parameters.",
                            children: [],
                            payload: { lines: "73,74" },
                          },
                        ],
                        payload: { lines: "70,74" },
                      },
                    ],
                    payload: { lines: "69,74" },
                  },
                  {
                    content: "Fully Sharded Data Parallel (FSDP):",
                    children: [
                      {
                        content: "Implements similar concepts to ZeRO.",
                        children: [],
                        payload: { lines: "75,76" },
                      },
                      {
                        content:
                          "Enables training with larger batch sizes, improving efficiency and scalability.",
                        children: [],
                        payload: { lines: "76,78" },
                      },
                    ],
                    payload: { lines: "74,78" },
                  },
                ],
                payload: { lines: "68,69" },
              },
            ],
            payload: { lines: "50,51" },
          },
          {
            content: "Fine-tuning Techniques",
            children: [
              {
                content: "Parameter-Efficient Fine-Tuning (PEFT)",
                children: [
                  {
                    content:
                      '<a href="https://arxiv.org/abs/2106.09685">LoRA (Low-Rank Adaptation)</a> <a href="https://ritvik19.medium.com/papers-explained-lora-a48359cecbfa">📑</a>: Uses low-rank decomposition to update a small subset of model parameters, keeping most pre-trained weights fixed.',
                    children: [],
                    payload: { lines: "81,82" },
                  },
                  {
                    content:
                      '<a href="https://arxiv.org/abs/2104.08691">Prompt Tuning</a>: Inserts learnable prompts into the input data to guide the model\'s attention and improve performance on specific tasks.',
                    children: [],
                    payload: { lines: "82,83" },
                  },
                  {
                    content:
                      '<a href="https://arxiv.org/abs/2303.16199">Llama-Adapter</a>: Adds learnable adapter modules to LLaMA\'s attention blocks, allowing for task-specific fine-tuning without modifying the core model architecture.',
                    children: [],
                    payload: { lines: "83,84" },
                  },
                  {
                    content:
                      '<a href="https://arxiv.org/abs/2405.17741">Dynamic Adapters</a>: Combines multiple adapters as a mixture-of-experts model, enabling multi-tasking and preventing forgetting of previously learned information.',
                    children: [],
                    payload: { lines: "84,86" },
                  },
                ],
                payload: { lines: "80,81" },
              },
              {
                content: "Data Augmentation",
                children: [
                  {
                    content:
                      '<a href="https://arxiv.org/abs/2302.13007">AugGPT</a>: Rephrases training samples using ChatGPT to generate diverse and varied input data.',
                    children: [],
                    payload: { lines: "87,88" },
                  },
                  {
                    content:
                      '<a href="https://arxiv.org/abs/2304.12244">Evol-Instruct</a> <a href="https://ritvik19.medium.com/papers-explained-127-wizardlm-65099705dfa3">📑</a>: Uses multistep revisions to generate increasingly complex and diverse open-domain instructions.',
                    children: [],
                    payload: { lines: "88,89" },
                  },
                  {
                    content:
                      '<a href="https://arxiv.org/abs/2310.11716">Reflection-tuning</a>: Refines both instructions and responses using GPT-4 based on predefined criteria, enhancing data quality and instruction-response consistency.',
                    children: [],
                    payload: { lines: "89,90" },
                  },
                  {
                    content:
                      '<a href="https://arxiv.org/abs/2408.01323">FANNO</a>: Augments instructions and generates responses by incorporating external knowledge sources through retrieval-augmented generation.',
                    children: [],
                    payload: { lines: "90,91" },
                  },
                  {
                    content:
                      '<a href="https://arxiv.org/abs/2403.15042">LLM2LLM</a>: Generates more challenging training samples based on the model\'s predictions on existing data, pushing the model to learn more robust representations.',
                    children: [],
                    payload: { lines: "91,93" },
                  },
                ],
                payload: { lines: "86,87" },
              },
            ],
            payload: { lines: "78,79" },
          },
        ],
        payload: { lines: "49,50" },
      },
      {
        content: "Model Compression Techniques",
        children: [
          {
            content: "Pruning Techniques",
            children: [
              {
                content:
                  '<a href="https://arxiv.org/abs/2301.00774">SparseGPT</a>: Reformulates pruning as a sparse regression problem. Optimizes both remaining and pruned weights using a layer-wise approximate regression solver.',
                children: [],
                payload: { lines: "96,97" },
              },
              {
                content:
                  '<a href="https://arxiv.org/abs/2306.11695">Wanda</a>: Considers weights and activations during pruning without requiring weight updates.',
                children: [],
                payload: { lines: "97,98" },
              },
              {
                content:
                  '<a href="https://arxiv.org/abs/2102.04010">n:m Pruning Strategy</a>: Prunes exactly n weights out of every m. Balances pruning flexibility and computational efficiency for significant speedups',
                children: [],
                payload: { lines: "98,99" },
              },
              {
                content:
                  '<a href="https://arxiv.org/abs/2210.06313">Neuron Sparsity</a>: Observes prevalent neuron sparsity, particularly in feed-forward networks, to guide pruning.',
                children: [],
                payload: { lines: "99,100" },
              },
              {
                content:
                  '<a href="https://arxiv.org/abs/2310.17157">Contextual Sparsity</a>: Proposes using smaller neural networks to dynamically prune based on input.',
                children: [],
                payload: { lines: "100,101" },
              },
              {
                content:
                  '<a href="https://arxiv.org/abs/2310.04564">Activation Sparsity</a>: Changes activation functions to ReLU and fine-tunes to enhance activation sparsity.',
                children: [],
                payload: { lines: "101,103" },
              },
            ],
            payload: { lines: "95,96" },
          },
          {
            content: "Quantization",
            children: [
              {
                content:
                  '<a href="https://arxiv.org/abs/2210.17323">GPTQ</a>: Performs layer-wise, weight-only quantization. Uses inverse Hessian matrices to reduce reconstruction error, improving compressed model accuracy.',
                children: [],
                payload: { lines: "104,105" },
              },
              {
                content:
                  '<a href="https://arxiv.org/abs/2306.00978">AWQ</a> and <a href="https://arxiv.org/abs/2206.01861">ZeroQuant</a>: Incorporate activations into quantization to better assess the importance of weights. Allow for more effective optimization during weight quantization, enhancing model efficiency.',
                children: [],
                payload: { lines: "105,106" },
              },
              {
                content:
                  '<a href="https://arxiv.org/abs/2401.18079">K/V Cache Quantization</a>: Specifically quantizes Key-Value cache for efficient long-sequence length inference.',
                children: [],
                payload: { lines: "106,107" },
              },
              {
                content:
                  '<a href="https://arxiv.org/abs/2211.10438">SmoothQuant</a>: Addresses activation quantization outliers by migrating quantization difficulty from activations to weights. Reduces the impact of outliers in activation distributions, stabilizing the quantization process.',
                children: [],
                payload: { lines: "107,108" },
              },
              {
                content:
                  '<a href="https://arxiv.org/abs/2405.16406">SpinQuant</a>: Applies rotation matrices to move outliers to a new space, handling activation quantization challenges. Improves quantization accuracy by transforming outlier values effectively.',
                children: [],
                payload: { lines: "108,109" },
              },
              {
                content:
                  '<a href="https://arxiv.org/abs/2305.17888">LLM-QAT</a> and <a href="https://arxiv.org/abs/2402.10787">EdgeQAT</a>: Use distillation with float16 models to correct quantization error, leading to stronger model performance. Enhances model resilience to quantization-induced degradation by incorporating quantization awareness during training.',
                children: [],
                payload: { lines: "109,111" },
              },
            ],
            payload: { lines: "103,104" },
          },
          {
            content: "Knowledge Distillation Techniques",
            children: [
              {
                content:
                  '<a href="https://arxiv.org/abs/1503.02531">Classical Knowledge Distillation</a>: Involves training a smaller, efficient "student" model to replicate the behavior of a larger, more complex "teacher" model.',
                children: [],
                payload: { lines: "112,113" },
              },
              {
                content:
                  '<a href="https://arxiv.org/abs/2308.02019">BabyLLaMA</a> and <a href="https://arxiv.org/abs/2409.17312">BabyLLaMA-2</a>: Developed models using a Llama teacher model, showing that distillation from a robust teacher can outperform traditional pre-training on the same dataset.',
                children: [],
                payload: { lines: "113,114" },
              },
              {
                content:
                  '<a href="https://arxiv.org/abs/2306.08543">Distillation Loss Modification</a>: Introduces modifications to the distillation loss, enhancing student models\' response quality, calibration, and lowering exposure bias.',
                children: [],
                payload: { lines: "114,115" },
              },
              {
                content:
                  '<a href="https://arxiv.org/abs/2307.15190">Sequence-Level Distillation with f-Divergences</a>: Uses a generalized f-divergence in the distillation loss to improve sequence-level distillation outcomes.',
                children: [],
                payload: { lines: "115,116" },
              },
              {
                content:
                  '<a href="https://arxiv.org/abs/2210.01351">Layer-Wise Distillation with Task-Aware Filters</a>: Employs filters that distill only task-specific knowledge from the teacher, enhancing layer-wise distillation.',
                children: [],
                payload: { lines: "116,117" },
              },
              {
                content:
                  'Fusion of Multiple Teacher Models (<a href="https://arxiv.org/abs/2401.10491">Knowledge Fusion of Large Language Models</a>, <a href="https://arxiv.org/abs/2408.07990">FuseChat</a>): Merges output probability distributions from multiple language models as teachers to effectively distill knowledge into smaller models.',
                children: [],
                payload: { lines: "117,118" },
              },
              {
                content:
                  '<a href="https://arxiv.org/abs/2402.12030">Universal Logit Distillation Loss</a>: Addresses the limitation of needing the same tokenizer and data availability between teacher and student by introducing a universal logit distillation inspired by optimal transport.',
                children: [],
                payload: { lines: "118,119" },
              },
              {
                content:
                  'Distillation Combined with Pruning (<a href="https://arxiv.org/abs/2408.11796">LLM Pruning and Distillation in Practice: The Minitron Approach</a>, <a href="https://arxiv.org/abs/2407.14679">Compact Language Models via Pruning and Knowledge Distillation</a>): Involves an iterative process of pruning a large model and retraining it with distillation losses to achieve smaller, efficient models.',
                children: [],
                payload: { lines: "119,120" },
              },
              {
                content:
                  '<a href="https://arxiv.org/abs/2305.02301">Distillation with Rationales</a>: Adds rationales as extra supervision during distillation, making the process more sample-efficient and improving performance on benchmarks like NLI, Commonsense QA, and arithmetic reasoning.',
                children: [],
                payload: { lines: "120,121" },
              },
              {
                content:
                  'Distillation of Reasoning Chains (<a href="https://arxiv.org/abs/2405.19737">Beyond Imitation</a>, <a href="https://arxiv.org/abs/2212.08410">Teaching Small Language Models to Reason</a>, <a href="https://arxiv.org/abs/2212.10071">Large Language Models Are Reasoning Teachers</a>, <a href="https://arxiv.org/abs/2301.12726">Specializing Smaller Language Models towards Multi-Step Reasoning</a>): Transfers reasoning chains along with labels from a larger model to a smaller one, enhancing arithmetic, multi-step math, symbolic, and commonsense reasoning abilities in the distilled model.',
                children: [],
                payload: { lines: "121,122" },
              },
            ],
            payload: { lines: "111,112" },
          },
        ],
        payload: { lines: "93,94" },
      },
    ],
    payload: { lines: "1,2" },
  },
  { color: ["#2980b9"], maxWidth: 800, initialExpandLevel: 3 }
);
