const encoder_only_transformers = [
  {
    title: "BERT",
    link: "https://ritvik19.medium.com/papers-explained-02-bert-31e59abc0615",
    date: "October 2018",
    description:
      "Introduced pre-training for Encoder Transformers. Uses unified architecture across different tasks.",
    tags: ["Language Models", "Transformer Encoder"],
  },
  {
    title: "RoBERTa",
    link: "https://ritvik19.medium.com/papers-explained-03-roberta-81db014e35b9",
    date: "July 2019",
    description:
      "Built upon BERT, by carefully optimizing hyperparameters and training data size to improve performance on various language tasks .",
    tags: ["Language Models", "Transformer Encoder"],
  },
  {
    title: "Sentence BERT",
    link: "https://ritvik19.medium.com/papers-explained-04-sentence-bert-5159b8e07f21",
    date: "August 2019",
    description:
      "A modification of BERT that uses siamese and triplet network structures to derive sentence embeddings that can be compared using cosine-similarity.",
    tags: ["Language Models", "Transformer Encoder"],
  },
  {
    title: "Tiny BERT",
    link: "https://ritvik19.medium.com/papers-explained-05-tiny-bert-5e36fe0ee173",
    date: "September 2019",
    description:
      "Uses attention transfer, and task specific distillation for distilling BERT.",
    tags: ["Language Models", "Transformer Encoder"],
  },
  {
    title: "ALBERT",
    link: "https://ritvik19.medium.com/papers-explained-07-albert-46a2a0563693",
    date: "September 2019",
    description:
      "Presents certain parameter reduction techniques to lower memory consumption and increase the training speed of BERT.",
    tags: ["Language Models", "Transformer Encoder"],
  },
  {
    title: "Distil BERT",
    link: "https://ritvik19.medium.com/papers-explained-06-distil-bert-6f138849f871",
    date: "October 2019",
    description:
      "Distills BERT on very large batches leveraging gradient accumulation, using dynamic masking and without the next sentence prediction objective.",
    tags: ["Language Models", "Transformer Encoder"],
  },
  {
    title: "Distil RoBERTa",
    link: "https://medium.com/dair-ai/papers-explained-06-distil-bert-6f138849f871#a260",
    date: "October 2019",
    description:
      "Distillation of RoBERTa, using the same techniques as Distil BERT.",
    tags: ["Language Models", "Transformer Encoder"],
  },
  {
    title: "FastBERT",
    link: "https://ritvik19.medium.com/papers-explained-37-fastbert-5bd246c1b432",
    date: "April 2020",
    description:
      "A speed-tunable encoder with adaptive inference time having branches at each transformer output to enable early outputs.",
    tags: ["Language Models", "Transformer Encoder"],
  },
  {
    title: "MobileBERT",
    link: "https://ritvik19.medium.com/papers-explained-36-mobilebert-933abbd5aaf1",
    date: "April 2020",
    description:
      "Compressed and faster version of the BERT, featuring bottleneck structures, optimized attention mechanisms, and knowledge transfer.",
    tags: ["Language Models", "Transformer Encoder"],
  },
  {
    title: "DeBERTa",
    link: "https://ritvik19.medium.com/papers-explained-08-deberta-a808d9b2c52d",
    date: "June 2020",
    description:
      "Enhances BERT and RoBERTa through disentangled attention mechanisms, an enhanced mask decoder, and virtual adversarial training.",
    tags: ["Language Models", "Transformer Encoder"],
  },
  {
    title: "DeBERTa v2",
    link: "https://ritvik19.medium.com/papers-explained-08-deberta-a808d9b2c52d#f5e1",
    date: "June 2020",
    description:
      "Enhanced version of the DeBERTa featuring a new vocabulary, nGiE integration, optimized attention mechanisms, additional model sizes, and improved tokenization.",
    tags: ["Language Models", "Transformer Encoder"],
  },
  {
    title: "DeBERTa v3",
    link: "https://ritvik19.medium.com/papers-explained-182-deberta-v3-65347208ce03",
    date: "November 2021",
    description:
      "Enhances the DeBERTa architecture by introducing replaced token detection (RTD) instead of mask language modeling (MLM), along with a novel gradient-disentangled embedding sharing method, exhibiting superior performance across various natural language understanding tasks.",
    tags: ["Language Models", "Transformer Encoder"],
  },
  {
    title: "ModernBERT",
    link: "https://ritvik19.medium.com/papers-explained-277-modernbert-59f25989f685",
    date: "December 2024",
    description:
      "Modernized encoder-only transformer model trained on 2 trillion tokens with a native 8192 sequence length, incorporating architectural improvements like GeGLU activations, RoPE embeddings, alternating attention, and unpadding, resulting in state-of-the-art performance across diverse classification and retrieval tasks (including code) and superior inference speed and memory efficiency compared to existing encoder models.",
    tags: ["Language Models", "Transformer Encoder", "HuggingFace", "Nvidia"],
  },
  {
    title: "It’s All in The [MASK]",
    link: "https://ritvik19.medium.com/papers-explained-312-its-all-in-the-mask-8c010744924e",
    date: "February 2025",
    description:
      "Introduces ModernBERT-Large-Instruct, a 0.4B parameter encoder-only model using its masked language modeling (MLM) head for generative classification, achieving strong zero-shot performance on classification and knowledge tasks, rivaling larger LLMs.",
    tags: ["Language Models", "Transformer Encoder"],
  },
  {
    title: "NeoBERT",
    link: "https://ritvik19.medium.com/papers-explained-327-neobert-b209f06dfc73",
    date: "February 2025",
    description:
      "A next-generation encoder model incorporating advancements in architecture, data, and pre-training strategies. It features an optimal depth-to-width ratio, an extended context length of 4,096 tokens, and a compact 250M parameter footprint, while also being fully open-source with released code, data, training scripts, and model checkpoints.",
    tags: ["Language Models", "Transformer Encoder"],
  },
  {
    title: "Should We Still Pretrain Encoders with Masked Language Modeling?",
    link: "https://ritvik19.medium.com/papers-explained-407-should-we-still-pretrain-encoders-with-masked-language-modeling-27c25b39e3f0",
    date: "July 2025",
    description:
      "Investigates the impact of Masked Language Modeling (MLM) and Causal Language Modeling (CLM) objectives on learning text representations. The study compares models trained with MLM, CLM, and a combination of both, controlling for factors like model size, architecture, and data volume. The study finds that while MLM generally performs better across text representation tasks, CLM is more data-efficient and offers improved fine-tuning stability; a biphasic training strategy (CLM followed by MLM) achieves optimal performance, especially when initializing from pretrained CLM models.",
    tags: ["Language Models", "Transformer Encoder"]
  }
];