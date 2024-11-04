document.getElementById("title").innerHTML = "Low-Rank Adaptors";

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
    content: "Low-Rank Adaptors",
    children: [
      {
        content: "&nbsp;",
        children: [
          {
            content:
              '<a href="https://arxiv.org/abs/2106.09685">LoRA</a>: Low-Rank Adaptation <a href="https://ritvik19.medium.com/papers-explained-lora-a48359cecbfa">📑</a> <a href="https://ritvik19.medium.com/papers-explained-review-06-parameter-efficient-finetuning-6934fafa74e5#05fa">📑</a>',
            children: [
              {
                content: "<strong>Purpose:</strong>",
                children: [
                  {
                    content:
                      "Efficiently adapt pre-trained LLMs to downstream tasks.",
                    children: [],
                    payload: { lines: "7,8" },
                  },
                ],
                payload: { lines: "6,8" },
              },
              {
                content: "<strong>Method:</strong>",
                children: [
                  {
                    content: "Freezes the pre-trained model weights (W).",
                    children: [],
                    payload: { lines: "9,10" },
                  },
                  {
                    content:
                      "Injects trainable low-rank matrices A and B into each Transformer layer, resulting in a low-rank update ∆W = BA.",
                    children: [],
                    payload: { lines: "10,11" },
                  },
                  {
                    content:
                      "Primarily focuses on adapting attention weights (Wq, Wk, Wv, Wo) while freezing MLP modules.",
                    children: [],
                    payload: { lines: "11,12" },
                  },
                ],
                payload: { lines: "8,12" },
              },
              {
                content: "<strong>Advantages:</strong>",
                children: [
                  {
                    content:
                      "<strong>Drastically Reduced Parameters:</strong> Up to 10,000 times fewer trainable parameters than full fine-tuning.",
                    children: [],
                    payload: { lines: "13,14" },
                  },
                  {
                    content:
                      "<strong>Lower Memory Footprint:</strong> Significant reduction in GPU memory requirements.",
                    children: [],
                    payload: { lines: "14,15" },
                  },
                  {
                    content:
                      "<strong>No Inference Latency:</strong> Merging trainable matrices with frozen weights during deployment eliminates additional inference time.",
                    children: [],
                    payload: { lines: "15,16" },
                  },
                ],
                payload: { lines: "12,16" },
              },
              {
                content: "<strong>Limitations:</strong>",
                children: [
                  {
                    content:
                      "<strong>Batching Complexity:</strong> Batching inputs with different A and B matrices for various tasks can be challenging.",
                    children: [],
                    payload: { lines: "17,18" },
                  },
                  {
                    content:
                      "<strong>Potential Inadequacy:</strong> Low-rank updates may not be sufficient for complex tasks requiring significant representation learning.",
                    children: [],
                    payload: { lines: "18,20" },
                  },
                ],
                payload: { lines: "16,20" },
              },
            ],
            payload: { lines: "4,5" },
          },
        ],
        payload: { lines: "3,4" },
      },
      {
        content: "Rank and Budget Allocation",
        children: [
          {
            content:
              '<a href="https://arxiv.org/abs/2210.07558">DyLoRA</a>: Dynamic Search-Free Low-Rank Adaptation <a href="https://ritvik19.medium.com/papers-explained-review-06-parameter-efficient-finetuning-6934fafa74e5#7fb6">📑</a>',
            children: [
              {
                content: "<strong>Purpose:</strong>",
                children: [
                  {
                    content:
                      "Solve the problem of optimal rank selection in LoRA.",
                    children: [],
                    payload: { lines: "24,25" },
                  },
                ],
                payload: { lines: "23,25" },
              },
              {
                content: "<strong>Method:</strong>",
                children: [
                  {
                    content:
                      "Randomly samples a rank from a pre-defined range during training.",
                    children: [],
                    payload: { lines: "26,27" },
                  },
                ],
                payload: { lines: "25,27" },
              },
              {
                content: "<strong>Advantage:</strong>",
                children: [
                  {
                    content:
                      "Eliminates the need for separate training for different rank values, providing flexibility and potentially better performance across a wider range of ranks.",
                    children: [],
                    payload: { lines: "28,30" },
                  },
                ],
                payload: { lines: "27,30" },
              },
            ],
            payload: { lines: "22,23" },
          },
          {
            content:
              '<a href="https://arxiv.org/abs/2303.10512">AdaLoRA</a>: Adaptive Budget Allocation <a href="https://ritvik19.medium.com/papers-explained-review-06-parameter-efficient-finetuning-6934fafa74e5#620f">📑</a>',
            children: [
              {
                content: "<strong>Purpose:</strong>",
                children: [
                  {
                    content:
                      "Adaptively allocate the parameter budget among weight matrices based on their importance.",
                    children: [],
                    payload: { lines: "32,33" },
                  },
                ],
                payload: { lines: "31,33" },
              },
              {
                content: "<strong>Method:</strong>",
                children: [
                  {
                    content:
                      "Parameterizes incremental updates using singular value decomposition (SVD).",
                    children: [],
                    payload: { lines: "34,35" },
                  },
                  {
                    content:
                      "Prunes singular values of less important updates to reduce their parameter budget.",
                    children: [],
                    payload: { lines: "35,36" },
                  },
                ],
                payload: { lines: "33,36" },
              },
              {
                content: "<strong>Advantages:</strong>",
                children: [
                  {
                    content:
                      "Improved performance, especially in low-budget settings.",
                    children: [],
                    payload: { lines: "37,38" },
                  },
                  {
                    content: "Avoids intensive exact SVD computations.",
                    children: [],
                    payload: { lines: "38,40" },
                  },
                ],
                payload: { lines: "36,40" },
              },
            ],
            payload: { lines: "30,31" },
          },
          {
            content:
              '<a href="https://arxiv.org/abs/2402.09353">DoRA</a>: Decomposing LoRA into Single-Rank Components for Dynamic&nbsp;Pruning <a href="https://ritvik19.medium.com/papers-explained-review-06-parameter-efficient-finetuning-6934fafa74e5#028e">📑</a>',
            children: [
              {
                content: "<strong>Purpose:</strong>",
                children: [
                  {
                    content:
                      "Enhance fine-tuning efficiency by dynamically distributing the parameter budget.",
                    children: [],
                    payload: { lines: "42,43" },
                  },
                ],
                payload: { lines: "41,43" },
              },
              {
                content: "<strong>Method:</strong>",
                children: [
                  {
                    content:
                      "Decomposes high-rank LoRA layers into single-rank components.",
                    children: [],
                    payload: { lines: "44,45" },
                  },
                  {
                    content:
                      "Dynamically prunes components with smaller contributions during training, guided by importance scores.",
                    children: [],
                    payload: { lines: "45,46" },
                  },
                  {
                    content:
                      "Uses a regularization penalty to promote stable pruning.",
                    children: [],
                    payload: { lines: "46,47" },
                  },
                ],
                payload: { lines: "43,47" },
              },
              {
                content: "<strong>Advantage:</strong>",
                children: [
                  {
                    content:
                      "Achieves competitive performance compared to LoRA and full fine-tuning with a more efficient parameter budget allocation.",
                    children: [],
                    payload: { lines: "48,50" },
                  },
                ],
                payload: { lines: "47,50" },
              },
            ],
            payload: { lines: "40,41" },
          },
        ],
        payload: { lines: "20,21" },
      },
      {
        content: "Memory Efficiency and Context Length",
        children: [
          {
            content:
              '<a href="https://arxiv.org/abs/2305.14314">QLoRA</a>: Quantized Low-Rank Adaptation <a href="https://ritvik19.medium.com/papers-explained-146-qlora-a6e7273bc630">📑</a> <a href="https://ritvik19.medium.com/papers-explained-review-06-parameter-efficient-finetuning-6934fafa74e5#3992">📑</a>',
            children: [
              {
                content: "<strong>Purpose:</strong>",
                children: [
                  {
                    content:
                      "Reduce memory usage for fine-tuning large models.",
                    children: [],
                    payload: { lines: "54,55" },
                  },
                ],
                payload: { lines: "53,55" },
              },
              {
                content: "<strong>Method:</strong>",
                children: [
                  {
                    content:
                      "Combines LoRA with 4-bit quantization of the pre-trained model and introduces memory management techniques:",
                    children: [],
                    payload: { lines: "56,57" },
                  },
                  {
                    content:
                      "<strong>4-bit NormalFloat (NF4):</strong> A new data type for quantizing normally distributed weights.",
                    children: [],
                    payload: { lines: "57,58" },
                  },
                  {
                    content:
                      "<strong>Double Quantization:</strong> Quantizes the quantization constants to further reduce memory.",
                    children: [],
                    payload: { lines: "58,59" },
                  },
                  {
                    content:
                      "<strong>Paged Optimizers:</strong> Manages memory spikes.",
                    children: [],
                    payload: { lines: "59,60" },
                  },
                ],
                payload: { lines: "55,60" },
              },
              {
                content: "<strong>Advantage:</strong>",
                children: [
                  {
                    content:
                      "Enables fine-tuning of large models (e.g., 65B parameters) on a single GPU while preserving performance.",
                    children: [],
                    payload: { lines: "61,63" },
                  },
                ],
                payload: { lines: "60,63" },
              },
            ],
            payload: { lines: "52,53" },
          },
          {
            content:
              '<a href="https://arxiv.org/abs/2308.03303">LoRA-FA</a>: Freezing the Projection-Down Weight <a href="https://ritvik19.medium.com/papers-explained-review-06-parameter-efficient-finetuning-6934fafa74e5#c229">📑</a>',
            children: [
              {
                content: "<strong>Purpose:</strong>",
                children: [
                  {
                    content:
                      "Reduce activation memory costs during fine-tuning.",
                    children: [],
                    payload: { lines: "65,66" },
                  },
                ],
                payload: { lines: "64,66" },
              },
              {
                content: "<strong>Method:</strong>",
                children: [
                  {
                    content:
                      "Freezes the projection-down weight (A) and only updates the projection-up weight (B) in each LoRA layer.",
                    children: [],
                    payload: { lines: "67,68" },
                  },
                ],
                payload: { lines: "66,68" },
              },
              {
                content: "<strong>Advantage:</strong>",
                children: [
                  {
                    content:
                      "Reduces overall memory cost (up to 1.4x compared to LoRA) without impacting performance or requiring expensive recomputation.",
                    children: [],
                    payload: { lines: "69,71" },
                  },
                ],
                payload: { lines: "68,71" },
              },
            ],
            payload: { lines: "63,64" },
          },
          {
            content:
              '<a href="https://arxiv.org/abs/2309.12307">LongLoRA</a>: Combining Shifted Sparse Attention and Improved&nbsp;LoRA <a href="https://ritvik19.medium.com/papers-explained-147-longlora-24f095b93611">📑</a> <a href="https://ritvik19.medium.com/papers-explained-review-06-parameter-efficient-finetuning-6934fafa74e5#8108">📑</a>',
            children: [
              {
                content: "<strong>Purpose:</strong>",
                children: [
                  {
                    content:
                      "Extend the context size of pre-trained LLMs efficiently.",
                    children: [],
                    payload: { lines: "73,74" },
                  },
                ],
                payload: { lines: "72,74" },
              },
              {
                content: "<strong>Method:</strong>",
                children: [
                  {
                    content:
                      "Combines improved LoRA with Shifted Sparse Attention (S2-Attn).",
                    children: [],
                    payload: { lines: "75,76" },
                  },
                  {
                    content:
                      "<strong>Improved LoRA:</strong> Includes trainable embedding and normalization layers.",
                    children: [],
                    payload: { lines: "76,77" },
                  },
                  {
                    content:
                      "<strong>S2-Attn:</strong> Approximates long context training with sparse local attention, switching to full attention during inference.",
                    children: [],
                    payload: { lines: "77,78" },
                  },
                ],
                payload: { lines: "74,78" },
              },
              {
                content: "<strong>Advantages:</strong>",
                children: [
                  {
                    content:
                      "Extends context length significantly (e.g., Llama2 7B from 4k to 100k context).",
                    children: [],
                    payload: { lines: "79,80" },
                  },
                  {
                    content:
                      "Retains original model architecture and compatibility with optimization techniques like Flash-Attention2.",
                    children: [],
                    payload: { lines: "80,81" },
                  },
                  {
                    content:
                      "Reduces memory and training time compared to full fine-tuning.",
                    children: [],
                    payload: { lines: "81,83" },
                  },
                ],
                payload: { lines: "78,83" },
              },
            ],
            payload: { lines: "71,72" },
          },
          {
            content:
              '<a href="https://arxiv.org/abs/2310.11454">VeRA</a>: Vector-based Random Matrix Adaptation <a href="https://ritvik19.medium.com/papers-explained-review-06-parameter-efficient-finetuning-6934fafa74e5#5bb3">📑</a>',
            children: [
              {
                content: "<strong>Purpose:</strong>",
                children: [
                  {
                    content:
                      "Further enhance parameter efficiency and reduce memory footprint compared to LoRA.",
                    children: [],
                    payload: { lines: "86,87" },
                  },
                ],
                payload: { lines: "85,87" },
              },
              {
                content: "<strong>Method:</strong>",
                children: [
                  {
                    content:
                      "Employs a single pair of frozen, randomly initialized matrices (A and B) shared across all layers.",
                    children: [],
                    payload: { lines: "88,89" },
                  },
                  {
                    content:
                      "Utilizes trainable scaling vectors (b and d) to adapt each layer.",
                    children: [],
                    payload: { lines: "89,90" },
                  },
                ],
                payload: { lines: "87,90" },
              },
              {
                content: "<strong>Advantages:</strong>",
                children: [
                  {
                    content:
                      "<strong>Extreme Parameter Reduction:</strong> Requires significantly fewer trainable parameters than LoRA, especially at low ranks.",
                    children: [],
                    payload: { lines: "91,92" },
                  },
                  {
                    content:
                      "<strong>Minimal Memory Requirements:</strong>  Only needs to store the scaling vectors and a seed to regenerate the random matrices.",
                    children: [],
                    payload: { lines: "92,93" },
                  },
                  {
                    content:
                      "<strong>No Inference Latency:</strong> Similar to LoRA, trained components can be merged with the original weights.",
                    children: [],
                    payload: { lines: "93,95" },
                  },
                ],
                payload: { lines: "90,95" },
              },
            ],
            payload: { lines: "83,84" },
          },
        ],
        payload: { lines: "50,51" },
      },
      {
        content: "Performance Enhancement",
        children: [
          {
            content:
              '<a href="https://arxiv.org/abs/2309.02411">Delta-LoRA</a>: Updating Pre-Trained Weights <a href="https://ritvik19.medium.com/papers-explained-review-06-parameter-efficient-finetuning-6934fafa74e5#a4ec">📑</a>',
            children: [
              {
                content: "<strong>Purpose:</strong>",
                children: [
                  {
                    content:
                      "Address the limitations of solely relying on low-rank updates in LoRA.",
                    children: [],
                    payload: { lines: "99,100" },
                  },
                ],
                payload: { lines: "98,100" },
              },
              {
                content: "<strong>Method:</strong>",
                children: [
                  {
                    content:
                      "Updates both the low-rank matrices (A, B) and the original pre-trained weights (W) using the delta of their product (A(t+1)B(t+1) - A(t)B(t)).",
                    children: [],
                    payload: { lines: "101,102" },
                  },
                  {
                    content:
                      "Removes the Dropout layer from the LoRA module to ensure accurate delta calculation.",
                    children: [],
                    payload: { lines: "102,103" },
                  },
                ],
                payload: { lines: "100,103" },
              },
              {
                content: "<strong>Advantage:</strong>",
                children: [
                  {
                    content:
                      "Achieves better performance than LoRA and other low-rank adaptation methods by incorporating the original weights in the adaptation process.",
                    children: [],
                    payload: { lines: "104,106" },
                  },
                ],
                payload: { lines: "103,106" },
              },
            ],
            payload: { lines: "97,98" },
          },
          {
            content:
              '<a href="https://arxiv.org/abs/2402.12354">LoRA+</a>: Different Learning Rates for A and&nbsp;B <a href="https://ritvik19.medium.com/papers-explained-review-06-parameter-efficient-finetuning-6934fafa74e5#fd31">📑</a>',
            children: [
              {
                content: "<strong>Purpose:</strong>",
                children: [
                  {
                    content: "Improve feature learning in LoRA.",
                    children: [],
                    payload: { lines: "108,109" },
                  },
                ],
                payload: { lines: "107,109" },
              },
              {
                content: "<strong>Method:</strong>",
                children: [
                  {
                    content:
                      "Employs different learning rates for the A and B matrices based on theoretical analysis of the infinite-width limit.",
                    children: [],
                    payload: { lines: "110,111" },
                  },
                ],
                payload: { lines: "109,111" },
              },
              {
                content: "<strong>Advantage:</strong>",
                children: [
                  {
                    content: "Enhances performance compared to standard LoRA.",
                    children: [],
                    payload: { lines: "112,114" },
                  },
                ],
                payload: { lines: "111,114" },
              },
            ],
            payload: { lines: "106,107" },
          },
          {
            content:
              '<a href="https://arxiv.org/abs/2405.12130">MoRA</a>: High-Rank Updating with Square&nbsp;Matrices <a href="https://ritvik19.medium.com/papers-explained-review-06-parameter-efficient-finetuning-6934fafa74e5#21a4">📑</a>',
            children: [
              {
                content: "<strong>Purpose:</strong>",
                children: [
                  {
                    content:
                      "Improve performance, especially on memory-intensive tasks, by enabling high-rank updates.",
                    children: [],
                    payload: { lines: "116,117" },
                  },
                ],
                payload: { lines: "115,117" },
              },
              {
                content: "<strong>Method:</strong>",
                children: [
                  {
                    content:
                      "Uses a square matrix for updating instead of the low-rank matrices in LoRA, while maintaining a similar number of trainable parameters.",
                    children: [],
                    payload: { lines: "118,119" },
                  },
                  {
                    content:
                      "Introduces non-parameter operators to reduce input and increase output dimensions for the square matrix, allowing for merging with the original model.",
                    children: [],
                    payload: { lines: "119,120" },
                  },
                ],
                payload: { lines: "117,120" },
              },
              {
                content: "<strong>Advantages:</strong>",
                children: [
                  {
                    content: "Outperforms LoRA on memory-intensive tasks.",
                    children: [],
                    payload: { lines: "121,122" },
                  },
                  {
                    content:
                      "Achieves comparable performance on other tasks like instruction tuning and mathematical reasoning.",
                    children: [],
                    payload: { lines: "122,123" },
                  },
                ],
                payload: { lines: "120,123" },
              },
            ],
            payload: { lines: "114,115" },
          },
        ],
        payload: { lines: "95,96" },
      },
    ],
    payload: { lines: "1,2" },
  },
  { color: ["#2980b9"], maxWidth: 400, initialExpandLevel: 3 }
);
