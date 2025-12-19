const language_models = [
  {
    title: "Sparsely-Gated Mixture-of-Experts Layer",
    link: "e3462e8c0232",
    date: "Jan 2017",
    description:
      "Introduces the Sparsely-Gated Mixture-of-Experts (MoE) layer consisting of feed-forward sub-networks and a trainable gating network addresses the challenges of conditional computation and dramatically increase model capacity without a proportional increase in computation.",
    tags: ["Neural Network Layers", "Mixture of Experts"],
  },
  {
    title: "Transformer",
    link: "474bb60a33f7",
    date: "June 2017",
    description:
      "An Encoder Decoder model, that introduced multihead attention mechanism for language translation task.",
    tags: ["Language Models"],
  },
  {
    title: "Elmo",
    link: "76362a43e4",
    date: "February 2018",
    description:
      "Deep contextualized word representations that captures both intricate aspects of word usage and contextual variations across language contexts.",
    tags: ["Language Models"],
  },
  {
    title: "Marian MT",
    link: "1b44479b0fd9",
    date: "April 2018",
    description:
      "A Neural Machine Translation framework written entirely in C++ with minimal dependencies, designed for high training and translation speed.",
    tags: ["Language Models"],
  },
  {
    title: "Transformer XL",
    link: "2e407e780e8",
    date: "January 2019",
    description:
      "Extends the original Transformer model to handle longer sequences of text by introducing recurrence into the self-attention mechanism.",
    tags: ["Language Models"],
  },
  {
    title: "XLM",
    link: "42a175e93caf",
    date: "January 2019",
    description:
      "Proposes two methods to learn cross-lingual language models (XLMs): one unsupervised that only relies on monolingual data, and one supervised that leverages parallel data with a new cross-lingual language model objective.",
    tags: ["Language Models"],
  },
  {
    title: "Sparse Transformer",
    link: "906a0be1e4e7",
    date: "April 2019",
    description:
      "Introduced sparse factorizations of the attention matrix to reduce the time and memory consumption to O(n√ n) in terms of sequence lengths.",
    tags: ["Language Models", "OpenAI"],
  },
  {
    title: "UniLM",
    link: "672f0ecc6a4a",
    date: "May 2019",
    description:
      "Utilizes a shared Transformer network and specific self-attention masks to excel in both language understanding and generation tasks.",
    tags: ["Language Models"],
  },
  {
    title: "XLNet",
    link: "ea0c3af96d49",
    date: "June 2019",
    description:
      "Extension of the Transformer-XL, pre-trained using a new method that combines ideas from AR and AE objectives.",
    tags: ["Language Models"],
  },
  {
    title: "CTRL",
    link: "146fcd18a566",
    date: "September 2019",
    description:
      "A 1.63B language model that can generate text conditioned on control codes that govern style, content, and task-specific behavior, allowing for more explicit control over text generation.",
    tags: ["Language Models"],
  },
  {
    title: "BART",
    link: "7f56138175bd",
    date: "October 2019",
    description:
      "An encoder-decoder network pretrained to reconstruct the original text from corrupted versions of it.",
    tags: ["Language Models"],
  },
  {
    title: "T5",
    link: "9d974a3b7957",
    date: "October 2019",
    description:
      "A unified encoder-decoder framework that converts all text-based language problems into a text-to-text format.",
    tags: ["Language Models"],
  },
  {
    title: "XLM-Roberta",
    link: "2da91fc24059",
    date: "November 2019",
    description:
      "A multilingual masked language model pre-trained on text in 100 languages, shows that pretraining multilingual language models at scale leads to significant performance gains for a wide range of crosslingual transfer tasks.",
    tags: ["Language Models"],
  },
  {
    title: "Pegasus",
    link: "1cb16f572553",
    date: "December 2019",
    description:
      "A self-supervised pre-training objective for abstractive text summarization, proposes removing/masking important sentences from an input document and generating them together as one output sequence.",
    tags: ["Language Models"],
  },
  {
    title: "Reformer",
    link: "4445ad305191",
    date: "January 2020",
    description:
      "Improves the efficiency of Transformers by replacing dot-product attention with locality-sensitive hashing (O(Llog L) complexity), using reversible residual layers to store activations only once, and splitting feed-forward layer activations into chunks, allowing it to perform on par with Transformer models while being much more memory-efficient and faster on long sequences.",
    tags: ["Language Models"],
  },
  {
    title: "mBART",
    link: "98432ef6fec",
    date: "January 2020",
    description:
      "A multilingual sequence-to-sequence denoising auto-encoder that pre-trains a complete autoregressive model on large-scale monolingual corpora across many languages using the BART objective, achieving significant performance gains in machine translation tasks.",
    tags: ["Language Models"],
  },
  {
    title: "UniLMv2",
    link: "5a044ca7c525",
    date: "February 2020",
    description:
      "Utilizes a pseudo-masked language model (PMLM) for both autoencoding and partially autoregressive language modeling tasks,significantly advancing the capabilities of language models in diverse NLP tasks.",
    tags: ["Language Models"],
  },
  {
    title: "ELECTRA",
    link: "501c175ae9d8",
    date: "March 2020",
    description:
      "Proposes a sample-efficient pre-training task called replaced token detection, which corrupts input by replacing some tokens with plausible alternatives and trains a discriminative model to predict whether each token was replaced or no.",
    tags: ["Language Models"],
  },
  {
    title: "Longformer",
    link: "9a08416c532e",
    date: "April 2020",
    description:
      "Introduces a linearly scalable attention mechanism, allowing handling texts of exteded length.",
    tags: ["Language Models"],
  },
  {
    title: "T5 v1.1",
    link: "9d974a3b7957#773b",
    date: "July 2020",
    description:
      "An enhanced version of the original T5 model, featuring improvements such as GEGLU activation, no dropout in pre-training, exclusive pre-training on C4, no parameter sharing between embedding and classifier layers.",
    tags: ["Language Models"],
  },
  {
    title: "mT5",
    link: "c61e03bc9218",
    date: "October 2020",
    description:
      "A multilingual variant of T5 based on T5 v1.1, pre-trained on a new Common Crawl-based dataset covering 101 languages (mC4).",
    tags: ["Language Models"],
  },
  {
    title: "Switch Transformers",
    link: "5c3d3d877fb7",
    date: "January 2021",
    description:
      "Sparsely-activated expert models that simplify and improve upon Mixture of Experts (MoE) by using a simplified routing algorithm and improved models with reduced communication and computational costs. They enable training large sparse models with lower precision formats, achieve significant speedups in pre-training.",
    tags: ["Language Models", "Mixture of Experts"],
  },
  {
    title: "FLAN",
    link: "1c5e0d5db7c9",
    date: "September 2021",
    description:
      "An instruction-tuned language model developed through finetuning on various NLP datasets described by natural language instructions.",
    tags: ["Language Models"],
  },
  {
    title: "T0",
    link: "643a53079fe",
    date: "October 2021",
    description:
      "A fine tuned encoder-decoder model on a multitask mixture covering a wide variety of tasks, attaining strong zero-shot performance on several standard datasets.",
    tags: ["Language Models"],
  },
  {
    title: "GLaM",
    link: "c02044027ba0",
    date: "December 2021",
    description:
      "A family of language models that utilizes a sparsely activated mixture-of-experts architecture to scale model capacity while reducing training costs compared to dense models. The largest GLaM has 1.2 trillion parameters, approximately 7x larger than GPT-3, but consumes only 1/3 of the energy and requires half the computation FLOPs for inference.",
    tags: ["Language Models", "Mixture of Experts"],
  },
  {
    title: "BERTopic",
    link: "f9aec10cd5a6",
    date: "March 2022",
    description:
      "Utilizes Sentence-BERT for document embeddings, UMAP, HDBSCAN (soft-clustering), and an adjusted class-based TF-IDF, addressing multiple topics per document and dynamic topics' linear evolution.",
    tags: ["Language Models"],
  },
  {
    title: "Flan T5, Flan PaLM",
    link: "caf168b6f76",
    date: "October 2022",
    description:
      "Explores instruction fine tuning with a particular focus on scaling the number of tasks, scaling the model size, and fine tuning on chain-of-thought data.",
    tags: ["Language Models"],
  },
  {
    title: "BLOOMZ, mT0",
    link: "8932577dcd1d",
    date: "November 2022",
    description:
      "Applies Multitask prompted fine tuning to the pretrained multilingual models on English tasks with English prompts to attain task generalization to non-English languages that appear only in the pretraining corpus.",
    tags: ["Language Models"],
  },
  {
    title: "CodeFusion",
    link: "fee6aba0149a",
    date: "October 2023",
    description:
      "A diffusion code generation model that iteratively refines entire programs based on encoded natural language, overcoming the limitation of auto-regressive models in code generation by allowing reconsideration of earlier tokens.",
    tags: ["Language Models", "Code Generation", "Diffusion Models"],
  },
  {
    title: "Aya 101",
    link: "d813ba17b83a",
    date: "February 2024",
    description:
      "A massively multilingual generative language model that follows instructions in 101 languages,trained by finetuning mT5.",
    tags: ["Language Models", "Multilingual Models", "Cohere"],
  },
  {
    title: "Hawk, Griffin",
    link: "dfc8c77f5dcd",
    date: "February 2024",
    description:
      "Introduces Real Gated Linear Recurrent Unit Layer that forms the core of the new recurrent block, replacing Multi-Query Attention for better efficiency and scalability",
    tags: ["Language Models"],
  },
  {
    title: "RecurrentGemma",
    link: "52732d0f4273",
    date: "April 2024",
    description:
      "Based on Griffin, uses a combination of linear recurrences and local attention instead of global attention to model long sequences efficiently.",
    tags: ["Language Models", "Gemma"],
  },
  {
    title: "Encoder-Decoder Gemma",
    link: "a6f9ee73a5f4",
    date: "April 2025",
    description:
      "Explores adapting pre-trained decoder-only LLMs to encoder-decoder models to leverage the strengths of both approaches for a better quality-efficiency trade-off.",
    tags: ["Language Models", "Gemma"],
  },
  {
    title: "Nemotron-H",
    link: "bc40f4b899cb",
    date: "April 2025",
    description:
      "A family of hybrid Mamba-Transformer models (8B and 56B/47B parameters) designed to reduce inference costs while maintaining or improving accuracy compared to similarly sized open-source Transformer models, achieving up to 3x faster inference speeds; it utilizes techniques like MiniPuzzle for compression and FP8-based training to further enhance efficiency.",
    tags: ["Language Models", "Hybrid Models"],
  },
  {
    title: "VaultGemma",
    link: "f738ba8705dd",
    date: "September 2025",
    description:
      "A 1 billion parameter model within the Gemma family, fully trained with differential privacy (DP) on the same data mixture used for the Gemma 2 series. VaultGemma addresses the privacy risks inherent in LLMs, which are susceptible to memorizing and extracting training data, potentially disclosing sensitive information.",
    tags: ["Language Models", "Gemma", "Differential Privacy"],
  },
  {
    title: "T5Gemma 2",
    link: "c406dbdd3839",
    date: "December 2025",
    description:
      "A family of lightweight open encoder-decoder LLMs featuring three model sizes: 270M-270M, 1B-1B, and 4B-4B, designed for multilingual, multimodal, and long-context understanding. It builds upon the T5Gemma architecture, adapting pretrained decoder-only models with the UL2 objective and incorporating innovations like tied word embeddings and merged attention for efficiency.",
    tags: ["Language Models", "Gemma"],
  }
];
