const model_merging = [
  {
    title: "Model Soup",
    link: "d0db49797b90#29bf",
    date: "March 2022",
    description:
      "Model soups, created by averaging the weights of multiple fine-tuned models with different hyperparameters, improve accuracy and robustness without increasing inference time, often outperforming the best individual model in a hyperparameter sweep, while also generalizing to other tasks and improving out-of-distribution performance.",
    tags: ["Model Merging"],
  },
  {
    title: "ColD Fusion",
    link: "452f33101a91",
    date: "December 2022",
    description:
      "A method enabling the benefits of multitask learning through distributed computation without data sharing and improving model performance.",
    tags: ["Model Merging"],
  },
  {
    title: "Spherical Linear Interpolation",
    link: "d0db49797b90#745f",
    date: "",
    description:
      "Blends two models by interpolating their normalized weight vectors along the shortest great-circle arc on the high-dimensional unit hypersphere, thereby preserving angular relationships between their parameter spaces for a smooth combination of learned behaviors.",
    tags: ["Model Merging"],
  },
  {
    title: "Nearswap",
    link: "d0db49797b90#745f",
    date: "",
    description:
      "Softly blends two models by swapping in secondary-model parameters for any weight whose difference from the base-model parameter is below a threshold t—otherwise retaining the base weight—to incorporate similar features while preserving the base model’s structure.",
    tags: ["Model Merging"],
  },
  {
    title: "Task Arithmetic",
    link: "d0db49797b90#38ff",
    date: "December 2022",
    description:
      "A paradigm for editing pre-trained neural networks by performing arithmetic operations on task vectors, which are derived by subtracting pre-trained model weights from fine-tuned weights. This method allows for multi-task learning, targeted forgetting of undesirable behaviors or entire tasks, and leveraging analogies between tasks to improve performance on data-scarce tasks.",
    tags: ["Model Merging"],
  },
  {
    title: "Trim, Elect Sign & Merge (TIES)",
    link: "d0db49797b90#5901",
    date: "June 2023",
    description:
      "Addresses interference between parameters by first trimming redundant, low-magnitude parameter changes, then resolving sign conflicts between remaining parameters by selecting the sign with the largest total magnitude across models, and finally averaging only the parameters with agreeing signs.",
    tags: ["Model Merging"],
  },
  {
    title: "Drop And REscale (DARE)",
    link: "d0db49797b90#ae3d",
    date: "November 2023",
    description:
      "Sparsifies the delta parameters (difference between fine-tuned and pre-trained weights) of multiple models by randomly setting a proportion (p) to zero and rescaling the remaining parameters by 1/(1-p) to approximate original embeddings. Then, it applies existing model merging techniques (like averaging or Task Arithmetic) to the sparsified models.",
    tags: ["Model Merging"],
  },
  {
    title: "Model Breadcrumbs",
    link: "d0db49797b90#4aca",
    date: "December 2023",
    description:
      "Constructs sparse weight sets ('breadcrumbs') by subtracting pre-trained weights from fine-tuned weights and then applying a sparsification process to remove outliers and negligible changes.",
    tags: ["Model Merging"],
  },
  {
    title: "Model Stock",
    link: "d0db49797b90#2570",
    date: "March 2024",
    description:
      "Leverages the observation that fine-tuned weights with varying random seeds lie on a thin shell in weight space, and proximity to the center of this shell correlates with improved performance. By strategically averaging a small number of fine-tuned models (often just two) with the pre-trained model using a geometrically derived interpolation ratio, Model Stock approximates a center-close weight, achieving comparable or superior performance.",
    tags: ["Model Merging"],
  },
  {
    title: "NuSLERP (Normalized SLERP)",
    link: "d0db49797b90#e2e2",
    date: "",
    description:
      "Performs spherical interpolation between two (or optionally three, including a base) models’ parameters—using configurable tensor‐flattening and row‐wise SLERP via the weight, nuslerp_flatten, and nuslerp_row_wise settings—so you can merge models without needing a dedicated base for angular interpolation of task vectors.",
    tags: ["Model Merging"],
  },
  {
    title: "Drop and rEscaLe via sampLing with mAgnitude (DELLA)",
    link: "d0db49797b90#2e71",
    date: "June 2024",
    description:
      "Reduces interference in model merging by using MAGPRUNE, a novel magnitude-based pruning technique to drop less important delta parameters (differences between fine-tuned and pre-trained weights) and rescale the remaining ones, followed by selecting parameters with agreeing signs and averaging them.",
    tags: ["Model Merging"],
  },
  {
    title: "Select, Calculate, and Erase (SCE)",
    link: "d0db49797b90#5ed3",
    date: "August 2024",
    description:
      "First selects the top τ% most variant elements within each parameter matrix across multiple target LLMs' fusion vectors (difference between fine-tuned and pivot model weights). Then, it calculates parameter matrix-level merging coefficients for each target LLM proportional to the sum of squares of the selected elements. Finally, it resolves sign conflicts within each parameter across fusion vectors by erasing elements with minority signs before merging the remaining parameters with their calculated coefficients, adding the result to the pivot model's weights.",
    tags: ["Model Merging"],
  },
  {
    title: "Mix Data or Merge Models",
    link: "75a3373c7f30",
    date: "October 2024",
    description:
      "Explores model merging as an alternative to data mixing for improving the safety and general performance of multilingual language models. The study finds that objective-based and language-based merging outperform data mixing, with specific merging algorithms like SLERP achieving the best balance between harm reduction and general performance across multiple languages.",
    tags: ["Model Merging", "Cohere"],
  },
  {
    title: "Long-To-Short LLM Reasoning With Model Merging",
    link: "03a212b0ccad",
    date: "March 2025",
    description:
      "Explores model merging as an efficient method for Long-to-Short reasoning in LLMs, aiming to reduce verbose reasoning steps without sacrificing accuracy. The study found that task-vector based merging, effectively reduced response length by ~50% while maintaining or slightly improving accuracy on 7B parameter models; activation-based merging showed even greater promise but is sensitive to calibration data; and merging efficacy was correlated with model scale, with smaller models struggling to learn complex reasoning and larger models posing challenges for significant length reduction.",
    tags: ["Model Merging", "Language Models"],
  },
  {
    title: "Model Interpolation for Efficient Reasoning",
    link: "9029e3301a8b",
    date: "October 2025",
    description:
      "Revisits model interpolation for efficient reasoning in LLMs, revealing a three-stage evolutionary paradigm as the interpolation coefficient changes.",
    tags: ["Model Merging", "Language Models"],
  }
];
