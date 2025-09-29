const retrieval_and_representation_learning = [
  {
    title: "SimCLR",
    link: "https://ritvik19.medium.com/papers-explained-200-simclr-191ecf19d2fc",
    date: "February 2020",
    description:
      "A simplified framework for contrastive learning that optimizes data augmentation composition, introduces learnable nonlinear transformations, and leverages larger batch sizes and more training steps.",
    tags: ["Representation Learning"],
  },
  {
    title: "Dense Passage Retriever",
    link: "https://ritvik19.medium.com/papers-explained-86-dense-passage-retriever-c4742fdf27ed",
    date: "April 2020",
    description:
      "Shows that retrieval can be practically implemented using dense representations alone, where embeddings are learned from a small number of questions and passages by a simple dual encoder framework.",
    tags: ["Retrieval"],
  },
  {
    title: "ColBERT",
    link: "https://medium.com/@ritvik19/papers-explained-88-colbert-fe2fd0509649",
    date: "April 2020",
    description:
      "Introduces a late interaction architecture that adapts deep LMs (in particular, BERT) for efficient retrieval.",
    tags: ["Retrieval"],
  },
  {
    title: "SimCLRv2",
    link: "https://ritvik19.medium.com/papers-explained-201-simclrv2-bc3fe72b8b48",
    date: "June 2020",
    description:
      "A Semi-supervised learning framework which uses unsupervised pre training followed by supervised fine-tuning and distillation with unlabeled examples.",
    tags: ["Representation Learning"],
  },
  {
    title: "CLIP",
    link: "https://ritvik19.medium.com/papers-explained-100-clip-f9873c65134",
    date: "February 2021",
    description:
      "A vision system that learns image representations from raw text-image pairs through pre-training, enabling zero-shot transfer to various downstream tasks.",
    tags: ["Representation Learning"],
  },
  {
    title: "ColBERTv2",
    link: "https://ritvik19.medium.com/papers-explained-89-colbertv2-7d921ee6e0d9",
    date: "December 2021",
    description:
      "Couples an aggressive residual compression mechanism with a denoised supervision strategy to simultaneously improve the quality and space footprint of late interaction.",
    tags: ["Retrieval"],
  },
  {
    title: "CLAP",
    link: "https://ritvik19.medium.com/3d13ce1e0b40",
    date: "June 2022",
    description:
      "Learns audio concepts from natural language supervision by connecting language and audio using two encoders namely CNN14 and BERT and contrastive learning g to bring audio and text descriptions into a joint multimodal space.",
    tags: ["Representation Learning", "Multimodal Models", "Audio Models"],
  },
  {
    title: "Matryoshka Representation Learning",
    link: "https://ritvik19.medium.com/papers-explained-matryoshka-representation-learning-e7a139f6ad27",
    date: "May 2022",
    description:
      "Encodes information at different granularities and allows a flexible representation that can adapt to multiple downstream tasks with varying computational resources using a single embedding.",
    tags: ["Representation Learning"],
  },
  {
    title: "E5",
    link: "https://ritvik19.medium.com/papers-explained-90-e5-75ea1519efad",
    date: "December 2022",
    description:
      "A family of text embeddings trained in a contrastive manner with weak supervision signals from a curated large-scale text pair dataset CCPairs.",
    tags: ["Representation Learning"],
  },
  {
    title: "SigLIP",
    link: "https://ritvik19.medium.com/papers-explained-152-siglip-011c48f9d448",
    date: "March 2023",
    description:
      "A simple pairwise Sigmoid loss function for Language-Image Pre-training that operates solely on image-text pairs, allowing for larger batch sizes and better performance at smaller batch sizes.",
    tags: ["Representation Learning", "Multimodal Models"],
  },
  {
    title: "Jina Embeddings v1",
    link: "https://ritvik19.medium.com/papers-explained-263-jina-embeddings-v1-33336e9efb0f",
    date: "July 2023",
    description:
      "Contrastively fine tuned T5 encoder on curated high quality pairwise and triplet data specifically to sensitize the models to distinguish negations of statements from confirming statements.",
    tags: ["Representation Learning"],
  },
  {
    title: "Jina Embeddings v2",
    link: "https://ritvik19.medium.com/papers-explained-264-jina-embeddings-v2-c5d540a9154f",
    date: "October 2023",
    description:
      "An open-source text embedding model capable of accommodating up to 8192 tokens, trained by pre-training a modified BERT from scratch before fine tuning for embeddings objectives.",
    tags: ["Representation Learning"],
  },
  {
    title: "SynCLR",
    link: "https://ritvik19.medium.com/papers-explained-202-synclr-85b50ef0081b",
    date: "December 2023",
    description:
      "A visual representation learning method that leverages generative models to synthesize large-scale curated datasets without relying on any real data.",
    tags: ["Representation Learning"],
  },
  {
    title: "E5 Mistral 7B",
    link: "https://ritvik19.medium.com/papers-explained-91-e5-mistral-7b-23890f40f83a",
    date: "December 2023",
    description:
      "Leverages proprietary LLMs to generate diverse synthetic data to fine tune open-source decoder-only LLMs for hundreds of thousands of text embedding tasks.",
    tags: ["Representation Learning"],
  },
  {
    title: "Nomic Embed Text v1",
    link: "https://ritvik19.medium.com/papers-explained-110-nomic-embed-8ccae819dac2",
    date: "February 2024",
    description:
      "A 137M parameter, open-source English text embedding model with an 8192 context length that outperforms OpenAI's models on both short and long-context tasks.",
    tags: ["Representation Learning"],
  },
  {
    title: "Nomic Embed Text v1.5",
    link: "https://ritvik19.medium.com/papers-explained-110-nomic-embed-8ccae819dac2#2119",
    date: "February 2024",
    description:
      "An advanced text embedding model that utilizes Matryoshka Representation Learning to offer flexible embedding sizes with minimal performance trade-offs",
    tags: ["Representation Learning"],
  },
  {
    title: "Jina Bilingual Embeddings",
    link: "https://ritvik19.medium.com/papers-explained-265-jina-bilingual-embeddings-39960d6f7a7c",
    date: "February 2024",
    description:
      "A suite of bilingual text embedding models that support up to 8192 tokens, trained by pre-training a modified bilingual BERT from scratch before fine tuning for embeddings objectives.",
    tags: ["Representation Learning"],
  },
  {
    title: "Jina Reranker",
    link: "https://ritvik19.medium.com/papers-explained-267-jina-reranker-daf6fdf8b2a6",
    date: "February 2024",
    description:
      "A neural reranking model that enhances search and RAG systems by reordering retrieved documents for better alignment with search query terms.",
    tags: ["Retrieval", "Reranking"],
  },
  {
    title: "Gecko",
    link: "https://ritvik19.medium.com/papers-explained-203-gecko-8889158b17e6",
    date: "March 2024",
    description:
      "A 1.2B versatile text embedding model achieving strong retrieval performance by distilling knowledge from LLMs into a retriever.",
    tags: ["Representation Learning", "Retrieval"],
  },
  {
    title: "NV Embed",
    link: "https://ritvik19.medium.com/papers-explained-168-nv-embed-48bd25d83258",
    date: "May 2024",
    description:
      "Introduces architectural innovations and training recipe to significantly enhance LLMs performance in general-purpose text embedding tasks.",
    tags: ["Representation Learning", "Nvidia"],
  },
  {
    title: "Nomic Embed Vision v1 and v1.5",
    link: "https://ritvik19.medium.com/papers-explained-110-nomic-embed-8ccae819dac2#486b",
    date: "June 2024",
    description:
      "Aligns a Vision Encoder with the existing text encoders without destroying the downstream performance of the text encoder, to attain a unified multimodal latent space.",
    tags: ["Representation Learning", "Multimodal Models"],
  },
  {
    title: "Document Screenshot Embedding",
    link: "https://ritvik19.medium.com/papers-explained-313-document-screenshot-embedding-792cb286643c",
    date: "June 2024",
    description:
      "A novel retrieval method that uses large vision-language models (like the fine-tuned Phi-3-vision) to encode document screenshots directly into dense vectors, bypassing the need for content extraction and preserving all information (text, images, layout), allowing unified retrieval across diverse document formats and modalities.",
    tags: ["Retrieval", "Multimodal Models"],
  },
  {
    title: "ColPali",
    link: "https://ritvik19.medium.com/papers-explained-198-colpali-b3be70cbe252",
    date: "June 2024",
    description:
      "A retrieval model based on PaliGemma to produce high-quality contextualized embeddings solely from images of document pages, and employees late interaction allowing for efficient and effective visually rich document retrieval.",
    tags: ["Retrieval", "Multimodal Models"],
  },
  {
    title: "Jina Reranker v2",
    link: "https://ritvik19.medium.com/papers-explained-267-jina-reranker-daf6fdf8b2a6#4405",
    date: "June 2024",
    description:
      "Builds upon Jina Reranker v1 by adding multilingual support, function-calling capabilities, structured data querying, code retrieval, and ultra-fast inference.",
    tags: ["Retrieval", "Reranking"],
  },
  {
    title: "E5-V",
    link: "https://ritvik19.medium.com/papers-explained-172-e5-v-9947d3925802",
    date: "July 2024",
    description:
      "A framework that adapts Multimodal Large Language Models for achieving universal multimodal embeddings by leveraging prompts and single modality training on text pairs, which demonstrates strong performance in multimodal embeddings without fine-tuning and eliminates the need for costly multimodal training data collection.",
    tags: ["Representation Learning", "Multimodal Models"],
  },
  {
    title: "Matryoshka Adaptor",
    link: "https://ritvik19.medium.com/papers-explained-204-matryoshka-adaptor-c22f76488959",
    date: "July 2024",
    description:
      "A framework designed for the customization of LLM embeddings, facilitating substantial dimensionality reduction while maintaining comparable performance levels.",
    tags: ["Representation Learning"],
  },
  {
    title: "Jina Embeddings v3",
    link: "https://ritvik19.medium.com/papers-explained-266-jina-embeddings-v3-9c38c9f69766",
    date: "September 2024",
    description:
      "A text embedding model with 570 million parameters that supports long-context retrieval tasks up to 8192 tokens, includes LoRA adapters for various NLP tasks, and allows flexible output dimension reduction from 1024 down to 32 using Matryoshka Representation Learning.",
    tags: ["Representation Learning"],
  },
  {
    title: "vdr embeddings",
    link: "https://ritvik19.medium.com/papers-explained-314-vdr-embeddings-1482b79e12a4",
    date: "January 2025",
    description:
      "Embedding models designed for visual document retrieval. Trained on a large synthetic dataset using a DSE approach, improving retrieval quality, in cross-lingual scenarios and for visual-heavy documents, and support Matryoshka Representation Learning for reduced vector size with minimal performance impact.",
    tags: [
      "Retrieval",
      "Representation Learning",
      "Multimodal Models",
      "Multilingual Models",
    ],
  },
  {
    title: "mmE5",
    link: "https://ritvik19.medium.com/papers-explained-315-mme5-3839eed789fe",
    date: "February 2025",
    description:
      "A multimodal multilingual E5 model trained on synthetic datasets generated by a novel framework focusing on broad scope (diverse tasks, modalities, and 93 languages), robust cross-modal alignment (deep thinking process within a single MLLM pass), and high fidelity (real images, self-evaluation, and refinement).",
    tags: [
      "Representation Learning",
      "Multimodal Models",
      "Multilingual Models",
    ],
  },
  {
    title: "SigLIP 2",
    link: "https://ritvik19.medium.com/papers-explained-320-siglip-2-dba08ff09559",
    date: "February 2025",
    description:
      "A family of multilingual vision-language encoders improving upon the original SigLIP by incorporating captioning-based pretraining, self-supervised losses (self-distillation, masked prediction), and online data curation, offering various sizes (ViT-B/32, B/16, L, So400m, g), native aspect ratio preservation (NaFlex variant).",
    tags: [
      "Representation Learning",
      "Multimodal Models",
      "Multilingual Models",
    ],
  },
  {
    title: "Gemini Embedding",
    link: "https://ritvik19.medium.com/papers-explained-330-gemini-embedding-324982aeb756",
    date: "March 2025",
    description:
      "Initialized from Google's Gemini LLM, generates generalizable embeddings for multilingual text and code by leveraging Gemini's knowledge and a curated training dataset enhanced with Gemini-generated synthetic data and filtering.",
    tags: ["Representation Learning", "Multilingual Models"],
  },
  {
    title: "ReasonIR",
    link: "https://ritvik19.medium.com/papers-explained-371-reasonir-7ae7a6ceb54b",
    date: "April 2025",
    description:
      "A novel bi-encoder retriever specifically trained for reasoning-intensive tasks using a synthetic data generation pipeline, to create challenging queries paired with relevant documents and plausibly related but unhelpful hard negatives. It is trained on this synthetic data and existing public datasets.",
    tags: ["Retrieval", "Representation Learning", "Reasoning"],
  },
  {
    title: "Hard Negative Mining for Domain-Specific Retrieval",
    link: "https://ritvik19.medium.com/papers-explained-392-hard-negative-mining-for-domain-specific-retrieval-a334df3c97fa",
    date: "May 2025",
    description:
      "Addresses the challenge of retrieving accurate, domain-specific information in enterprise search systems, by dynamically selecting semantically challenging but contextually irrelevant documents to improve re-ranking models. The method integrates diverse embedding models, performs dimensionality reduction, and employs a unique hard negative selection process to ensure computational efficiency and semantic precision.",
    tags: ["Retrieval"],
  },
  {
    title: "Jina Embeddings v4",
    link: "https://ritvik19.medium.com/papers-explained-409-jina-embeddings-v4-9d266f0a6138",
    date: "June 2025",
    description:
      "A 3.8B multimodal embedding model based on Qwen2.5-VL that unifies text and image representations using a novel architecture supporting both single-vector and multi-vector embeddings. It incorporates task-specific LoRA adapters to optimize performance across diverse retrieval scenarios.",
    tags: ["Representation Learning", "Multimodal Models"]
  },
  {
    title: "GloVe 2024",
    link: "https://ritvik19.medium.com/papers-explained-glove-2024-8c935a8ac58a",
    date: "July 2025",
    description:
      "This report documents and evaluates new 2024 English GloVe models, addressing the need for updated word embeddings due to language evolution since the original 2014 models. The 2024 models, trained using Wikipedia, Gigaword, and a subset of Dolma, incorporate new culturally and linguistically relevant words, perform comparably on structural tasks, and demonstrate improved performance on recent NER datasets.",
    tags: ["Representation Learning"]
  },
  {
    title: "Jina Code Embeddings",
    link: "https://ritvik19.medium.com/0a6c9ad05bbd",
    date: "August 2025",
    description:
      "A novel code embedding model suite designed to retrieve code from natural language queries, perform technical question-answering, and identify semantically similar code snippets across programming languages. It makes use of an autoregressive backbone (Qwen2.5-Coder) pre-trained on both text and code, generating embeddings via last-token pooling",
    tags: ["Representation Learning", "Code Models"]
  },
  {
    title: "EmbeddingGemma",
    link: "https://ritvik19.medium.com/076b2bc8b460",
    date: "September 2025",
    description:
      "A lightweight (300M parameter) open text embedding model based on the Gemma 3 language model family using encoder-decoder initialization, geometric embedding distillation, a spread-out regularizer, and checkpoint merging from varied, optimized mixtures.",
    tags: ["Representation Learning"]
  }
];
