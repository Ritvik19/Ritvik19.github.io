const layout_aware_language_models = [
  {
    title: "Layout LM",
    link: "32ec4bad6406",
    date: "December 2019",
    description:
      "Utilises BERT as the backbone, adds two new input embeddings: 2-D position embedding and image embedding (Only for downstream tasks).",
    tags: ["Layout Aware Language Models"],
  },
  {
    title: "LamBERT",
    link: "8f52d28f20d9",
    date: "February 2020",
    description:
      "Utilises RoBERTa as the backbone and adds Layout embeddings along with relative bias.",
    tags: ["Layout Aware Language Models"],
  },
  {
    title: "Layout LM v2",
    link: "9531a983e659",
    date: "December 2020",
    description:
      "Uses a multi-modal Transformer model, to integrate text, layout, and image in the pre-training stage, to learn end-to-end cross-modal interaction.",
    tags: ["Layout Aware Language Models"],
  },
  {
    title: "Structural LM",
    link: "36e9df91e7c1",
    date: "May 2021",
    description:
      "Utilises BERT as the backbone and feeds text, 1D and (2D cell level) embeddings to the transformer model.",
    tags: ["Layout Aware Language Models"],
  },
  {
    title: "Doc Former",
    link: "228ce27182a0",
    date: "June 2021",
    description:
      "Encoder-only transformer with a CNN backbone for visual feature extraction, combines text, vision, and spatial features through a multi-modal self-attention layer.",
    tags: ["Layout Aware Language Models"],
  },
  {
    title: "BROS",
    link: "1f1127476f73",
    date: "August 2021",
    description:
      "Built upon BERT, encodes relative positions of texts in 2D space and learns from unlabeled documents with area masking strategy.",
    tags: ["Layout Aware Language Models"],
  },
  {
    title: "LiLT",
    link: "701057ec6d9e",
    date: "February 2022",
    description:
      "Introduced Bi-directional attention complementation mechanism (BiACM) to accomplish the cross-modal interaction of text and layout.",
    tags: ["Layout Aware Language Models"],
  },
  {
    title: "Layout LM V3",
    link: "3b54910173aa",
    date: "April 2022",
    description:
      "A unified text-image multimodal Transformer to learn cross-modal representations, that imputs concatenation of text embedding and image embedding.",
    tags: ["Layout Aware Language Models"],
  },
  {
    title: "ERNIE Layout",
    link: "47a5a38e321b",
    date: "October 2022",
    description:
      "Reorganizes tokens using layout information, combines text and visual embeddings, utilizes multi-modal transformers with spatial aware disentangled attention.",
    tags: ["Layout Aware Language Models"],
  },
];
