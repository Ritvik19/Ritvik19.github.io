const peft = [
  {
    title: "LoRA",
    link: "a48359cecbfa",
    date: "July 2021",
    description:
      "Introduces trainable rank decomposition matrices into each layer of a pre-trained Transformer model, significantly reducing the number of trainable parameters for downstream tasks.",
    tags: ["Parameter Efficient Fine Tuning"],
  },
  {
    title: "DyLoRA",
    link: "6934fafa74e5#7fb6",
    date: "October 2022",
    description:
      "Allows for flexible rank size by randomly truncating low-rank matrices during training, enabling adaptation to different rank values without retraining.",
    tags: ["Parameter Efficient Fine Tuning"],
  },
  {
    title: "AdaLoRA",
    link: "6934fafa74e5#620f",
    date: "March 2023",
    description:
      "Dynamically allocates a parameter budget based on an importance metric to prune less important singular values during training.",
    tags: ["Parameter Efficient Fine Tuning"],
  },
  {
    title: "QLoRA",
    link: "a6e7273bc630",
    date: "May 2023",
    description:
      "Allows efficient training of large models on limited GPU memory, through innovations like 4-bit NormalFloat (NF4), double quantization and paged optimisers.",
    tags: ["Parameter Efficient Fine Tuning"],
  },
  {
    title: "LoRA-FA",
    link: "6934fafa74e5#c229",
    date: "August 2023",
    description:
      "Freezes one of the low-rank matrices and only trains a scaling vector for the other, further reducing the number of trainable parameters compared to standard LoRA.",
    tags: ["Parameter Efficient Fine Tuning"],
  },
  {
    title: "Delta-LoRA",
    link: "6934fafa74e5#a4ec",
    date: "September 2023",
    description:
      "Utilizes the delta of the low-rank matrix updates to refine the pre-trained weights directly, removing the Dropout layer for accurate backpropagation.",
    tags: ["Parameter Efficient Fine Tuning"],
  },
  {
    title: "LongLoRA",
    link: "24f095b93611",
    date: "September 2023",
    description:
      "Enables context extension for large language models, achieving significant computation savings through sparse local attention and parameter-efficient fine-tuning.",
    tags: ["Parameter Efficient Fine Tuning"],
  },
  {
    title: "VeRA",
    link: "6934fafa74e5#5bb3",
    date: "October 2023",
    description:
      "Utilizes frozen, shared random matrices across all layers and trains scaling vectors to adapt those matrices for each layer, reducing the number of trainable parameters compared to LoRA.",
    tags: ["Parameter Efficient Fine Tuning"],
  },
  {
    title: "LoRA+",
    link: "6934fafa74e5#fd31",
    date: "February 2024",
    description:
      "Enhances LoRA by setting different learning rates for the A and B matrices based on a fixed ratio, promoting better feature learning and improved performance.",
    tags: ["Parameter Efficient Fine Tuning"],
  },
  {
    title: "MoRA",
    link: "6934fafa74e5#21a4",
    date: "May 2024",
    description:
      "Introduces a square matrix and non-parameterized operators to achieve high-rank updating with the same number of trainable parameters as LoRA, improving knowledge memorization capabilities.",
    tags: ["Parameter Efficient Fine Tuning"],
  },
  {
    title: "DoRA",
    link: "6934fafa74e5#028e",
    date: "May 2024",
    description:
      "Decomposes the high-rank LoRA matrix into multiple single-rank components, allowing dynamic pruning of less important components during training for a more efficient parameter budget allocation.",
    tags: ["Parameter Efficient Fine Tuning"],
  },
];
