const nav_data = [
  "Language Models",
  "Encoder Only Transformers",
  "Decoder Only Transformers",
  "Small LLMs",
  "Multi Modal LMs",
  "LLM for Math",
  "Retrieval and Representation Learning",
  "LLM Training",
  "PEFT",
  "Vision Transformers",
  "CNNs",
  "Object Detection",
  "RCNNs",
  "Document Understanding",
  "Layout Aware LMs",
  "GANs",
  "Tabular Data",
  "Datasets",
  "Neural Network Layers",
  "Autoencoders",
  "Miscellaneous Studies",
];

const papers_data = [
  [
    // Language Models
    {
      title: "Transformer",
      link: "https://ritvik19.medium.com/papers-explained-01-transformer-474bb60a33f7",
      date: "June 2017",
      description:
        "An Encoder Decoder model, that introduced multihead attention mechanism for language translation task.",
      tags: ["Language Models"],
    },
    {
      title: "Elmo",
      link: "https://ritvik19.medium.com/papers-explained-33-elmo-76362a43e4",
      date: "February 2018",
      description:
        "Deep contextualized word representations that captures both intricate aspects of word usage and contextual variations across language contexts.",
      tags: ["Language Models"],
    },
    {
      title: "Marian MT",
      link: "https://ritvik19.medium.com/papers-explained-150-marianmt-1b44479b0fd9",
      date: "April 2018",
      description:
        "A Neural Machine Translation framework written entirely in C++ with minimal dependencies, designed for high training and translation speed.",
      tags: ["Language Models"],
    },
    {
      title: "Transformer XL",
      link: "https://ritvik19.medium.com/papers-explained-34-transformerxl-2e407e780e8",
      date: "January 2019",
      description:
        "Extends the original Transformer model to handle longer sequences of text by introducing recurrence into the self-attention mechanism.",
      tags: ["Language Models"],
    },
    {
      title: "XLM",
      link: "https://ritvik19.medium.com/papers-explained-158-xlm-42a175e93caf",
      date: "January 2019",
      description:
        "Proposes two methods to learn cross-lingual language models (XLMs): one unsupervised that only relies on monolingual data, and one supervised that leverages parallel data with a new cross-lingual language model objective.",
      tags: ["Language Models"],
    },
    {
      title: "Sparse Transformer",
      link: "https://ritvik19.medium.com/papers-explained-122-sparse-transformer-906a0be1e4e7",
      date: "April 2019",
      description:
        "Introduced sparse factorizations of the attention matrix to reduce the time and memory consumption to O(n√ n) in terms of sequence lengths.",
      tags: ["Language Models", "OpenAI"],
    },
    {
      title: "UniLM",
      link: "https://ritvik19.medium.com/papers-explained-72-unilm-672f0ecc6a4a",
      date: "May 2019",
      description:
        "Utilizes a shared Transformer network and specific self-attention masks to excel in both language understanding and generation tasks.",
      tags: ["Language Models"],
    },
    {
      title: "XLNet",
      link: "https://ritvik19.medium.com/papers-explained-35-xlnet-ea0c3af96d49",
      date: "June 2019",
      description:
        "Extension of the Transformer-XL, pre-trained using a new method that combines ideas from AR and AE objectives.",
      tags: ["Language Models"],
    },
    {
      title: "CTRL",
      link: "https://ritvik19.medium.com/papers-explained-153-ctrl-146fcd18a566",
      date: "September 2019",
      description:
        "A 1.63B language model that can generate text conditioned on control codes that govern style, content, and task-specific behavior, allowing for more explicit control over text generation.",
      tags: ["Language Models"],
    },
    {
      title: "BART",
      link: "https://ritvik19.medium.com/papers-explained-09-bart-7f56138175bd",
      date: "October 2019",
      description:
        "An encoder-decoder network pretrained to reconstruct the original text from corrupted versions of it.",
      tags: ["Language Models"],
    },
    {
      title: "T5",
      link: "https://ritvik19.medium.com/papers-explained-44-t5-9d974a3b7957",
      date: "October 2019",
      description:
        "A unified encoder-decoder framework that converts all text-based language problems into a text-to-text format.",
      tags: ["Language Models"],
    },
    {
      title: "XLM-Roberta",
      link: "https://ritvik19.medium.com/papers-explained-159-xlm-roberta-2da91fc24059",
      date: "November 2019",
      description:
        "A multilingual masked language model pre-trained on text in 100 languages, shows that pretraining multilingual language models at scale leads to significant performance gains for a wide range of crosslingual transfer tasks.",
      tags: ["Language Models"],
    },
    {
      title: "Pegasus",
      link: "https://ritvik19.medium.com/papers-explained-162-pegasus-1cb16f572553",
      date: "December 2019",
      description:
        "A self-supervised pre-training objective for abstractive text summarization, proposes removing/masking important sentences from an input document and generating them together as one output sequence.",
      tags: ["Language Models"],
    },
    {
      title: "Reformer",
      link: "https://ritvik19.medium.com/papers-explained-165-reformer-4445ad305191",
      date: "January 2020",
      description:
        "Improves the efficiency of Transformers by replacing dot-product attention with locality-sensitive hashing (O(Llog L) complexity), using reversible residual layers to store activations only once, and splitting feed-forward layer activations into chunks, allowing it to perform on par with Transformer models while being much more memory-efficient and faster on long sequences.",
      tags: ["Language Models"],
    },
    {
      title: "mBART",
      link: "https://ritvik19.medium.com/papers-explained-169-mbart-98432ef6fec",
      date: "January 2020",
      description:
        "A multilingual sequence-to-sequence denoising auto-encoder that pre-trains a complete autoregressive model on large-scale monolingual corpora across many languages using the BART objective, achieving significant performance gains in machine translation tasks.",
      tags: ["Language Models"],
    },
    {
      title: "UniLMv2",
      link: "https://ritvik19.medium.com/papers-explained-unilmv2-5a044ca7c525",
      date: "February 2020",
      description:
        "Utilizes a pseudo-masked language model (PMLM) for both autoencoding and partially autoregressive language modeling tasks,significantly advancing the capabilities of language models in diverse NLP tasks.",
      tags: ["Language Models"],
    },
    {
      title: "ELECTRA",
      link: "https://ritvik19.medium.com/papers-explained-173-electra-501c175ae9d8",
      date: "March 2020",
      description:
        "Proposes a sample-efficient pre-training task called replaced token detection, which corrupts input by replacing some tokens with plausible alternatives and trains a discriminative model to predict whether each token was replaced or no.",
      tags: ["Language Models"],
    },
    {
      title: "Longformer",
      link: "https://ritvik19.medium.com/papers-explained-38-longformer-9a08416c532e",
      date: "April 2020",
      description:
        "Introduces a linearly scalable attention mechanism, allowing handling texts of exteded length.",
      tags: ["Language Models"],
    },
    {
      title: "T5 v1.1",
      link: "https://ritvik19.medium.com/papers-explained-44-t5-9d974a3b7957#773b",
      date: "July 2020",
      description:
        "An enhanced version of the original T5 model, featuring improvements such as GEGLU activation, no dropout in pre-training, exclusive pre-training on C4, no parameter sharing between embedding and classifier layers.",
      tags: ["Language Models"],
    },
    {
      title: "mT5",
      link: "https://ritvik19.medium.com/papers-explained-113-mt5-c61e03bc9218",
      date: "October 2020",
      description:
        "A multilingual variant of T5 based on T5 v1.1, pre-trained on a new Common Crawl-based dataset covering 101 languages (mC4).",
      tags: ["Language Models"],
    },
    {
      title: "FLAN",
      link: "https://ritvik19.medium.com/papers-explained-46-flan-1c5e0d5db7c9",
      date: "September 2021",
      description:
        "An instruction-tuned language model developed through finetuning on various NLP datasets described by natural language instructions.",
      tags: ["Language Models"],
    },
    {
      title: "T0",
      link: "https://ritvik19.medium.com/papers-explained-74-t0-643a53079fe",
      date: "October 2021",
      description:
        "A fine tuned encoder-decoder model on a multitask mixture covering a wide variety of tasks, attaining strong zero-shot performance on several standard datasets.",
      tags: ["Language Models"],
    },
    {
      title: "BERTopic",
      link: "https://ritvik19.medium.com/papers-explained-193-bertopic-f9aec10cd5a6",
      date: "March 2022",
      description:
        "Utilizes Sentence-BERT for document embeddings, UMAP, HDBSCAN (soft-clustering), and an adjusted class-based TF-IDF, addressing multiple topics per document and dynamic topics' linear evolution.",
      tags: ["Language Models"],
    },
    {
      title: "Flan T5, Flan PaLM",
      link: "https://ritvik19.medium.com/papers-explained-75-flan-t5-flan-palm-caf168b6f76",
      date: "October 2022",
      description:
        "Explores instruction fine tuning with a particular focus on scaling the number of tasks, scaling the model size, and fine tuning on chain-of-thought data.",
      tags: ["Language Models"],
    },
    {
      title: "BLOOMZ, mT0",
      link: "https://ritvik19.medium.com/papers-explained-99-bloomz-mt0-8932577dcd1d",
      date: "November 2022",
      description:
        "Applies Multitask prompted fine tuning to the pretrained multilingual models on English tasks with English prompts to attain task generalization to non-English languages that appear only in the pretraining corpus.",
      tags: ["Language Models"],
    },
    {
      title: "CodeFusion",
      link: "https://ritvik19.medium.com/papers-explained-70-codefusion-fee6aba0149a",
      date: "October 2023",
      description:
        "A diffusion code generation model that iteratively refines entire programs based on encoded natural language, overcoming the limitation of auto-regressive models in code generation by allowing reconsideration of earlier tokens.",
      tags: ["Language Models", "Code Generation", "Diffusion Models"],
    },
    {
      title: "Aya 101",
      link: "https://ritvik19.medium.com/papers-explained-aya-101-d813ba17b83a",
      date: "February 2024",
      description:
        "A massively multilingual generative language model that follows instructions in 101 languages,trained by finetuning mT5.",
      tags: ["Language Models", "Multilingual Models", "Cohere"],
    },
    {
      title: "Hawk, Griffin",
      link: "https://ritvik19.medium.com/papers-explained-131-hawk-griffin-dfc8c77f5dcd",
      date: "February 2024",
      description:
        "Introduces Real Gated Linear Recurrent Unit Layer that forms the core of the new recurrent block, replacing Multi-Query Attention for better efficiency and scalability",
      tags: ["Language Models"],  
    },
    {
      title: "RecurrentGemma",
      link: "https://ritvik19.medium.com/papers-explained-132-recurrentgemma-52732d0f4273",
      date: "April 2024",
      description:
        "Based on Griffin, uses a combination of linear recurrences and local attention instead of global attention to model long sequences efficiently.",
      tags: ["Language Models", "Gemma"],
    },
  ],
  [
    // Encoder Only Transformers
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
    }
  ],
  [
    // Decoder Only Transformers
    {
      title: "GPT",
      link: "https://ritvik19.medium.com/papers-explained-43-gpt-30b6f1e6d226",
      date: "June 2018",
      description:
        "A Decoder only transformer which is autoregressively pretrained and then finetuned for specific downstream tasks using task-aware input transformations.",
      tags: ["Language Models", "Transformer Decoder", "OpenAI", "GPT"],
    },
    {
      title: "GPT 2",
      link: "https://ritvik19.medium.com/papers-explained-65-gpt-2-98d0a642e520",
      date: "February 2019",
      description:
        "Demonstrates that language models begin to learn various language processing tasks without any explicit supervision.",
      tags: ["Language Models", "Transformer Decoder", "OpenAI", "GPT"],
    },
    {
      title: "GPT 3",
      link: "https://ritvik19.medium.com/papers-explained-66-gpt-3-352f5a1b397",
      date: "May 2020",
      description:
        "Demonstrates that scaling up language models greatly improves task-agnostic, few-shot performance.",
      tags: ["Language Models", "Transformer Decoder", "OpenAI", "GPT"],
    },
    {
      title: "Codex",
      link: "https://ritvik19.medium.com/papers-explained-45-codex-caca940feb31",
      date: "July 2021",
      description:
        "A GPT language model finetuned on publicly available code from GitHub.",
      tags: ["Language Models", "Transformer Decoder", "Code Generation", "OpenAI", "GPT"],
    },
    {
      title: "WebGPT",
      link: "https://ritvik19.medium.com/papers-explained-123-webgpt-5bb0dd646b32",
      date: "December 2021",
      description:
        "A fine-tuned GPT-3 model utilizing text-based web browsing, trained via imitation learning and human feedback, enhancing its ability to answer long-form questions with factual accuracy.",
      tags: ["Language Models", "Transformer Decoder", "OpenAI", "GPT"],
    },
    {
      title: "Gopher",
      link: "https://ritvik19.medium.com/papers-explained-47-gopher-2e71bbef9e87",
      date: "December 2021",
      description:
        "Provides a comprehensive analysis of the performance of various Transformer models across different scales upto 280B on 152 tasks.",
      tags: ["Language Models", "Transformer Decoder"],
    },
    {
      title: "LaMDA",
      link: "https://ritvik19.medium.com/papers-explained-76-lamda-a580ebba1ca2",
      date: "January 2022",
      description:
        "Transformer based models specialized for dialog, which are pre-trained on public dialog data and web text.",
      tags: ["Language Models", "Transformer Decoder"],
    },
    {
      title: "Instruct GPT",
      link: "https://ritvik19.medium.com/papers-explained-48-instructgpt-e9bcd51f03ec",
      date: "March 2022",
      description:
        "Fine-tuned GPT using supervised learning (instruction tuning) and reinforcement learning from human feedback to align with user intent.",
      tags: ["Language Models", "Transformer Decoder", "OpenAI", "GPT"],
    },
    {
      title: "Chinchilla",
      link: "https://ritvik19.medium.com/papers-explained-49-chinchilla-a7ad826d945e",
      date: "March 2022",
      description:
        "Investigated the optimal model size and number of tokens for training a transformer LLM within a given compute budget (Scaling Laws).",
      tags: ["Language Models", "Transformer Decoder"],
    },
    {
      title: "CodeGen",
      link: "https://ritvik19.medium.com/papers-explained-125-codegen-a6bae5c1f7b5",
      date: "March 2022",
      description:
        "An LLM trained for program synthesis using input-output examples and natural language descriptions.",
      tags: ["Language Models", "Transformer Decoder", "Code Generation"],
    },
    {
      title: "PaLM",
      link: "https://ritvik19.medium.com/papers-explained-50-palm-480e72fa3fd5",
      date: "April 2022",
      description:
        "A 540-B parameter, densely activated, Transformer, trained using Pathways, (ML system that enables highly efficient training across multiple TPU Pods).",
      tags: ["Language Models", "Transformer Decoder"],
    },
    {
      title: "GPT-NeoX-20B",
      link: "https://ritvik19.medium.com/papers-explained-78-gpt-neox-20b-fe39b6d5aa5b",
      date: "April 2022",
      description:
        "An autoregressive LLM trained on the Pile, and the largest dense model that had publicly available weights at the time of submission.",
      tags: ["Language Models", "Transformer Decoder"],
    },
    {
      title: "OPT",
      link: "https://ritvik19.medium.com/papers-explained-51-opt-dacd9406e2bd",
      date: "May 2022",
      description:
        "A suite of decoder-only pre-trained transformers with parameter ranges from 125M to 175B. OPT-175B being comparable to GPT-3.",
      tags: ["Language Models", "Transformer Decoder"],
    },
    {
      title: "BLOOM",
      link: "https://ritvik19.medium.com/papers-explained-52-bloom-9654c56cd2",
      date: "November 2022",
      description:
        "A 176B-parameter open-access decoder-only transformer, collaboratively developed by hundreds of researchers, aiming to democratize LLM technology.",
      tags: ["Language Models", "Transformer Decoder"],
    },
    {
      title: "Galactica",
      link: "https://ritvik19.medium.com/papers-explained-53-galactica-1308dbd318dc",
      date: "November 2022",
      description:
        "An LLM trained on scientific data thus specializing in scientific knowledge.",
      tags: ["Language Models", "Transformer Decoder", "Scientific Data"],
    },
    {
      title: "ChatGPT",
      link: "https://ritvik19.medium.com/papers-explained-54-chatgpt-78387333268f",
      date: "November 2022",
      description:
        "An interactive model designed to engage in conversations, built on top of GPT 3.5.",
      tags: ["Language Models", "Transformer Decoder", "OpenAI", "GPT"],
    },
    {
      title: "LLaMA",
      link: "https://ritvik19.medium.com/papers-explained-55-llama-c4f302809d6b",
      date: "February 2023",
      description:
        "A collection of foundation LLMs by Meta ranging from 7B to 65B parameters, trained using publicly available datasets exclusively.",
      tags: ["Language Models", "Transformer Decoder", "Llama"],
    },
    {
      title: "Toolformer",
      link: "https://ritvik19.medium.com/papers-explained-140-toolformer-d21d496b6812",
      date: "February 2023",
      description:
        "An LLM trained to decide which APIs to call, when to call them, what arguments to pass, and how to best incorporate the results into future token prediction.",
      tags: ["Language Models", "Transformer Decoder"],
    },
    {
      title: "Alpaca",
      link: "https://ritvik19.medium.com/papers-explained-56-alpaca-933c4d9855e5",
      date: "March 2023",
      description:
        "A fine-tuned LLaMA 7B model, trained on instruction-following demonstrations generated in the style of self-instruct using text-davinci-003.",
      tags: ["Language Models", "Transformer Decoder"],
    },
    {
      title: "GPT 4",
      link: "https://ritvik19.medium.com/papers-explained-67-gpt-4-fc77069b613e",
      date: "March 2023",
      description:
        "A multimodal transformer model pre-trained to predict the next token in a document, which can accept image and text inputs and produce text outputs.",
      tags: ["Language Models", "Transformer Decoder", "OpenAI", "GPT"],
    },
    {
      title: "Vicuna",
      link: "https://ritvik19.medium.com/papers-explained-101-vicuna-daed99725c7e",
      date: "March 2023",
      description:
        "A 13B LLaMA chatbot fine tuned on user-shared conversations collected from ShareGPT, capable of generating more detailed and well-structured answers compared to Alpaca.",
      tags: ["Language Models", "Transformer Decoder"],
    },
    {
      title: "Bloomberg GPT",
      link: "https://ritvik19.medium.com/papers-explained-120-bloomberggpt-4bedd52ef54b",
      date: "March 2023",
      description:
        "A 50B language model train on general purpose and domain specific data to support a wide range of tasks within the financial industry.",
      tags: ["Language Models", "Transformer Decoder"],
    },
    {
      title: "Pythia",
      link: "https://ritvik19.medium.com/papers-explained-121-pythia-708284c32964",
      date: "April 2023",
      description:
        "A suite of 16 LLMs all trained on public data seen in the exact same order and ranging in size from 70M to 12B parameters.",
      tags: ["Language Models", "Transformer Decoder"],
    },
    {
      title: "WizardLM",
      link: "https://ritvik19.medium.com/papers-explained-127-wizardlm-65099705dfa3",
      date: "April 2023",
      description:
        "Introduces Evol-Instruct, a method to generate large amounts of instruction data with varying levels of complexity using LLM instead of humans to fine tune a Llama model ",
      tags: ["Language Models", "Transformer Decoder", "Synthetic Data", "WizardLM"],
    },
    {
      title: "CodeGen 2",
      link: "https://ritvik19.medium.com/papers-explained-codegen2-d2690d7eb831",
      date: "May 2023",
      description:
        "Proposes an approach to make the training of LLMs for program synthesis more efficient by unifying key components of model architectures, learning methods, infill sampling, and data distributions",
      tags: ["Language Models", "Transformer Decoder", "Code Generation"],
    },
    {
      title: "PaLM 2",
      link: "https://ritvik19.medium.com/papers-explained-58-palm-2-1a9a23f20d6c",
      date: "May 2023",
      description:
        "Successor of PALM, trained on a mixture of different pre-training objectives in order to understand different aspects of language.",
      tags: ["Language Models", "Transformer Decoder"],
    },
    {
      title: "LIMA",
      link: "https://ritvik19.medium.com/papers-explained-57-lima-f9401a5760c3",
      date: "May 2023",
      description:
        "A LLaMa model fine-tuned on only 1,000 carefully curated prompts and responses, without any reinforcement learning or human preference modeling.",
      tags: ["Language Models", "Transformer Decoder"],
    },
    {
      title: "Gorilla",
      link: "https://ritvik19.medium.com/papers-explained-139-gorilla-79f4730913e9",
      date: "May 2023",
      description:
        "A retrieve-aware finetuned LLaMA-7B model, specifically for API calls.",
      tags: ["Language Models", "Transformer Decoder"],
    },
    {
      title: "Falcon",
      link: "https://ritvik19.medium.com/papers-explained-59-falcon-26831087247f",
      date: "June 2023",
      description:
        "An Open Source LLM trained on properly filtered and deduplicated web data alone.",
      tags: ["Language Models", "Transformer Decoder"],
    },
    {
      title: "WizardCoder",
      link: "https://ritvik19.medium.com/papers-explained-wizardcoder-a12ecb5b93b6",
      date: "June 2023",
      description:
        "Enhances the performance of the open-source Code LLM, StarCoder, through the application of Code Evol-Instruct.",
      tags: ["Language Models", "Transformer Decoder", "Code Generation", "Synthetic Data", "WizardLM"],
    },
    {
      title: "Tulu",
      link: "https://ritvik19.medium.com/papers-explained-181-tulu-ee85648cbf1b",
      date: "June 2023",
      description:
        "Explores instruction-tuning of language models ranging from 6.7B to 65B parameters on 12 different instruction datasets.",
      tags: ["Language Models", "Transformer Decoder"],
    },
    {
      title: "LLaMA 2",
      link: "https://ritvik19.medium.com/papers-explained-60-llama-v2-3e415c5b9b17",
      date: "July 2023",
      description:
        "Successor of LLaMA. LLaMA 2-Chat is optimized for dialogue use cases.",
      tags: ["Language Models", "Transformer Decoder", "Llama"],
    },
    {
      title: "Tool LLM",
      link: "https://ritvik19.medium.com/papers-explained-141-tool-llm-856f99e79f55",
      date: "July 2023",
      description:
        "A LLaMA model finetuned on an instruction-tuning dataset for tool use, automatically created using ChatGPT.",
      tags: ["Language Models", "Transformer Decoder"],
    },
    {
      title: "Humpback",
      link: "https://ritvik19.medium.com/papers-explained-61-humpback-46992374fc34",
      date: "August 2023",
      description: "LLaMA finetuned using Instruction backtranslation.",
      tags: ["Language Models", "Transformer Decoder", "Synthetic Data"],
    },
    {
      title: "Code LLaMA",
      link: "https://ritvik19.medium.com/papers-explained-62-code-llama-ee266bfa495f",
      date: "August 2023",
      description: "LLaMA 2 based LLM for code.",
      tags: ["Language Models", "Transformer Decoder", "Code Generation", "Llama"],
    },
    {
      title: "WizardMath",
      link: "https://ritvik19.medium.com/papers-explained-129-wizardmath-265e6e784341",
      date: "August 2023",
      description:
        "Proposes Reinforcement Learning from Evol-Instruct Feedback (RLEIF) method, applied to Llama-2 to enhance the mathematical reasoning abilities.",
      tags: ["Language Models", "Transformer Decoder", "Scientific Data", "WizardLM"],
    },
    {
      title: "LLaMA 2 Long",
      link: "https://ritvik19.medium.com/papers-explained-63-llama-2-long-84d33c26d14a",
      date: "September 2023",
      description:
        "A series of long context LLMs s that support effective context windows of up to 32,768 tokens.",
      tags: ["Language Models", "Transformer Decoder", "Llama"],
    },
    {
      title: "MAmmoTH",
      link: "https://ritvik19.medium.com/papers-explained-230-mammoth-06189e929910",
      date: "September 2023",
      description: 
        "A series of LLMs specifically designed for general math problem-solving, trained on MathInstruct, a dataset compiled from 13 math datasets with intermediate rationales that combines chain-of-thought and program-of-thought approaches to accommodate different thought processes for various math problems.",
      tags: ["Language Models", "Transformer Decoder", "Scientific Data"],
    },
    {
      title: "Llemma",
      link: "https://ritvik19.medium.com/papers-explained-69-llemma-0a17287e890a",
      date: "October 2023",
      description:
        "An LLM for mathematics, formed by continued pretraining of Code Llama on a mixture of scientific papers, web data containing mathematics, and mathematical code.",
      tags: ["Language Models", "Transformer Decoder", "Scientific Data"],
    },
    {
      title: "Grok 1",
      link: "https://ritvik19.medium.com/papers-explained-186-grok-0d9f1aef69be",
      date: "November 2023",
      description:
        "A 314B Mixture-of-Experts model, modeled after the Hitchhiker's Guide to the Galaxy, designed to be witty.",
      tags: ["Language Models", "Transformer Decoder", "Mixtures of Experts", "Grok"],
    },
    {
      title: "Tulu v2",
      link: "https://ritvik19.medium.com/papers-explained-182-tulu-v2-ff38ab1f37f2",
      date: "November 2023",
      description:
        "An updated version of Tulu covering the open resources for instruction tuning om better base models to new finetuning techniques.",
      tags: ["Language Models", "Transformer Decoder"],
    },
    {
      title: "Mixtral 8x7B",
      link: "https://ritvik19.medium.com/papers-explained-95-mixtral-8x7b-9e9f40ebb745",
      date: "January 2024",
      description:
        "A Sparse Mixture of Experts language model based on Mistral 7B trained with multilingual data using a context size of 32k tokens.",
      tags: ["Language Models", "Transformer Decoder", "Mixtures of Experts", "Mistral"],
    },
    {
      title: "Nemotron-4 15B",
      link: "https://ritvik19.medium.com/papers-explained-206-nemotron-4-15b-7d895fb56134",
      date: "February 2024",
      description:
        "A 15B multilingual language model trained on 8T text tokens by Nvidia.",
      tags: ["Language Models", "Transformer Decoder", "Nvidia"],
    },
    {
      title: "DBRX",
      link: "https://ritvik19.medium.com/papers-explained-119-dbrx-17c61739983c",
      date: "March 2024",
      description:
        "A 132B open, general-purpose fine grained Sparse MoE LLM surpassing GPT-3.5 and competitive with Gemini 1.0 Pro.",
      tags: ["Language Models", "Transformer Decoder", "Mixtures of Experts"],
    },
    {
      title: "Command R",
      link: "https://ritvik19.medium.com/papers-explained-166-command-r-models-94ba068ebd2b",
      date: "March 2024",
      description:
        "An LLM optimized for retrieval-augmented generation and tool use, across multiple languages.",
      tags: ["Language Models", "Transformer Decoder", "Cohere"],
    },
    {
      title: "Grok 1.5",
      link: "https://ritvik19.medium.com/papers-explained-186-grok-0d9f1aef69be",
      date: "March 2024",
      description:
        "An advancement over grok, capable of long context understanding up to 128k tokens and advanced reasoning.",
      tags: ["Language Models", "Transformer Decoder", "Mixtures of Experts", "Grok"],
    },
    {
      title: "Mixtral 8x22B",
      link: "https://ritvik19.medium.com/papers-explained-95-mixtral-8x7b-9e9f40ebb745#20f3",
      date: "April 2024",
      description:
        "A open-weight AI model optimised for performance and efficiency, with capabilities such as fluency in multiple languages, strong mathematics and coding abilities, and precise information recall from large documents.",
      tags: ["Language Models", "Transformer Decoder", "Mixtures of Experts", "Mistral"],
    },
    {
      title: "Llama 3",
      link: "https://ritvik19.medium.com/papers-explained-187a-llama-3-51e2b90f63bb",
      date: "April 2024",
      description:
        "A family of 8B and 70B parameter models trained on 15T tokens with a focus on data quality, demonstrating state-of-the-art performance on various benchmarks, improved reasoning capabilities.",
      tags: ["Language Models", "Transformer Decoder", "Llama"],
    },
    {
      title: "Command R+",
      link: "https://ritvik19.medium.com/papers-explained-166-command-r-models-94ba068ebd2b#c2b5",
      date: "April 2024",
      description:
        "Successor of Command R+ with improved performance for retrieval-augmented generation and tool use, across multiple languages.",
      tags: ["Language Models", "Transformer Decoder", "Cohere"],
    },
    {
      title: "Rho-1",
      link: "https://ritvik19.medium.com/papers-explained-132-rho-1-788125e42241",
      date: "April 2024",
      description:
        "Introduces Selective Language Modelling that optimizes the loss only on tokens that align with a desired distribution, utilizing a reference model to score and select tokens.",
      tags: ["Language Models", "Transformer Decoder"],
    },
    {
      title: "MAmmoTH 2",
      link: "https://ritvik19.medium.com/papers-explained-231-mammoth2-e9c0e6fb9795",
      date: "May 2024",
      description: 
        "LLMs fine tuned on a dataset curated through the proposed paradigm that efficiently harvest 10M naturally existing instruction data from the pre-training web corpus to enhance LLM reasoning. It involves recalling relevant documents, extracting instruction-response pairs, and refining the extracted pairs using open-source LLMs.",
      tags: ["Language Models", "Transformer Decoder", "Scientific Data"],
    },
    {
      title: "Codestral 22B",
      link: "https://ritvik19.medium.com/papers-explained-mistral-7b-b9632dedf580#057b",
      date: "May 2024",
      description:
        "An open-weight model designed for code generation tasks, trained on over 80 programming languages, and licensed under the Mistral AI Non-Production License, allowing developers to use it for research and testing purposes.",
      tags: ["Language Models", "Transformer Decoder", "Code Generation", "Mistral"],
    },
    {
      title: "Aya 23",
      link: "https://ritvik19.medium.com/papers-explained-151-aya-23-d01605c3ee80",
      date: "May 2024",
      description:
        "A family of multilingual language models supporting 23 languages, designed to balance breadth and depth by allocating more capacity to fewer languages during pre-training.",
      tags: ["Language Models", "Transformer Decoder", "Multilingual Models", "Cohere"],
    },
    {
      title: "Nemotron-4 340B",
      link: "https://ritvik19.medium.com/papers-explained-207-nemotron-4-340b-4cfe268439f8",
      date: "June 2024",
      description:
        "340B models, along with a reward model by Nvidia, suitable for generating synthetic data to train smaller language models, with over 98% of the data used in model alignment being synthetically generated.",
      tags: ["Language Models", "Transformer Decoder", "Nvidia"],
    },
    {
      title: "LLama 3.1",
      link: "https://ritvik19.medium.com/papers-explained-187b-llama-3-1-f0fb06898c59",
      date: "July 2024",
      description:
        "A family of multilingual language models ranging from 8B to 405B parameters, trained on a massive dataset of 15T tokens and achieving comparable performance to leading models like GPT-4 on various tasks.",
      tags: ["Language Models", "Transformer Decoder", "Llama"],
    },
    {
      title: "LLama 3.1 - Multimodal Experiments",
      link: "https://ritvik19.medium.com/papers-explained-187c-llama-3-1-multimodal-experiments-a1940dd45575",
      date: "July 2024",
      description:
        "Additional experiments of adding multimodal capabilities to Llama3.",
      tags: ["Language Models", "Transformer Decoder", "Multimodal Models", "Llama"],
    },
    {
      title: "Mistral Large 2",
      link: "https://ritvik19.medium.com/papers-explained-mistral-7b-b9632dedf580#301d",
      date: "July 2024",
      description:
        "A 123B model, offers significant improvements in code generation, mathematics, and reasoning capabilities, advanced function calling, a 128k context window, and supports dozens of languages and over 80 coding languages.",
      tags: ["Language Models", "Transformer Decoder", "Mistral"],
    },
    {
      title: "LLM Compiler",
      link: "https://ritvik19.medium.com/papers-explained-223-llm-compiler-15b1ddb9a1b0",
      date: "July 2024",
      description:
        "A suite of pre-trained models designed for code optimization tasks, built upon Code Llama, with two sizes (7B and 13B), trained on LLVM-IR and assembly code to optimize compiler intermediate representations, assemble/disassemble, and achieve high accuracy in optimizing code size and disassembling from x86_64 and ARM assembly back into LLVM-IR.",
      tags: ["Language Models", "Transformer Decoder", "Code Generation", "Llama"],
    },
    {
      title: "Apple Intelligence Foundation Language Models",
      link: "https://ritvik19.medium.com/papers-explained-222-apple-intelligence-foundation-language-models-2b8a41371a42",
      date: "July 2024",
      description:
        "Two foundation language models, AFM-on-device (a ~3 B parameter model) and AFM-server (a larger server-based model), designed to power Apple Intelligence features efficiently, accurately, and responsibly, with a focus on Responsible AI principles that prioritize user empowerment, representation, design care, and privacy protection.",
      tags: ["Language Models", "Transformer Decoder"],
    },
    {
      title: "Hermes 3",
      link: "https://ritvik19.medium.com/papers-explained-188-hermes-3-67d36cfe07d8",
      date: "August 2024",
      description:
        "Neutrally generalist instruct and tool use models, created by fine-tuning Llama 3.1 models with strong reasoning and creative abilities, and are designed to follow prompts neutrally without moral judgment or personal opinions.",
      tags: ["Language Models", "Transformer Decoder"],
    },
    {
      title: "OLMoE",
      link: "https://ritvik19.medium.com/papers-explained-270-olmoe-38832ff4f9bd",
      date: "September 2024",
      description:
        "An open source language model based on sparse Mixture-of-Experts architecture with 7B parameters, out of which only 1B parameters are active per input token. Conducted extensive experiments on MoE training, analyzing routing strategies, expert specialization, and the impact of design choices like routing algorithms and expert size.",
      tags: ["Language Models", "Transformer Decoder", "Mixtures of Experts"],
    },
    {
      title: "o1",
      link: "https://ritvik19.medium.com/papers-explained-211-o1-163fd9c7308e",
      date: "September 2024",
      description:
        "A large language model trained with reinforcement learning to think before answering, producing a long internal chain of thought before responding.",
      tags: ["Language Models", "Transformer Decoder", "OpenAI", "GPT"],
    },
    {
      title: "o1-mini",
      link: "https://ritvik19.medium.com/papers-explained-211-o1-163fd9c7308e#f16a",
      date: "September 2024",
      description:
        "A cost-efficient reasoning model, excelling at STEM, especially math and coding, nearly matching the performance of OpenAI o1 on evaluation benchmarks.",
      tags: ["Language Models", "Transformer Decoder", "OpenAI", "GPT"],
    },
    {
      title: "Llama 3.1-Nemotron-51B",
      link: "https://ritvik19.medium.com/papers-explained-209-minitron-approach-in-practice-6b473f67328d#5df9",
      date: "September 2024",
      description:
        "Uses knowledge distillation and NAS to optimize various constraints, resulting in a model that achieves 2.2x faster inference compared to the reference model while maintaining nearly the same accuracy, with an irregular block structure that reduces or prunes attention and FFN layers for better utilization of H100 and improved LLMs for inference.",
      tags: ["Language Models", "Transformer Decoder", "Pruning", "Knowledge Distillation", "Nvidia"],
    },
    {
      title: "LLama 3.2",
      link: "https://ritvik19.medium.com/papers-explained-187d-llama-3-2-e517fa1f2528",
      date: "September 2024",
      description:
        "Small and medium-sized vision LLMs (11B and 90B), and lightweight, text-only models (1B and 3B).",
      tags: ["Language Models", "Transformer Decoder", "Llama", "Small LLMs", "Multimodal Models"],
    },
    {
      title: "Aya Expanse",
      link: "https://ritvik19.medium.com/papers-explained-151-aya-23-d01605c3ee80#c4a1",
      date: "October 2024",
      description:
        "A family of 8B and 32B highly performant multilingual models that excel across 23 languages.",
      tags: ["Language Models", "Transformer Decoder", "Multilingual Models", "Cohere"],
    },
    {
      title: "Tulu v3",
      link: "https://ritvik19.medium.com/papers-explained-183-tulu-v3-fc7758b18724",
      date: "November 2024",
      description:
        "A family of post-trained models based on Llama 3.1 that outperform instruct versions of other models, including closed models like GPT-4o-mini and Claude 3.5-Haiku, using training methods like supervised finetuning, Direct Preference Optimization, and Reinforcement Learning with Verifiable Rewards.",
      tags: ["Language Models", "Transformer Decoder"],
    },
    {
      title: "Llama 3.3",
      link: "https://ritvik19.medium.com/papers-explained-187e-quantized-llama-3-2-cc6965f61370#bd2c",
      date: "December 2024",
      description:
        "A multilingual, instruction-tuned generative language model with 70B parameters, optimized for dialogue use cases and trained on 15 trillion tokens of public data, incorporating both human-generated and synthetic data for safety and quality control.",
      tags: ["Language Models", "Transformer Decoder", "Llama"],
    },
    {
      title: "OLMo 2",
      link: "https://ritvik19.medium.com/papers-explained-olmo-2-f4d34e886503",
      date: "January 2025",
      description: "A family of open-source language models featuring improved architecture, training recipes, and pre-training data mixtures. It incorporates a new specialized data mix (Dolmino Mix 1124) introduced via late-stage curriculum training, and best practices from Tülu 3 are incorporated to develop OLMo 2-Instruct.",
      tags: ["Language Models", "Transformer Decoder"],
    },
    {
      title: "o3-mini",
      link: "https://ritvik19.medium.com/papers-explained-211-o1-163fd9c7308e#9875",
      date: "January 2025",
      description:
        "A cost-efficient reasoning model, excelling in STEM fields, while maintaining low latency. It supports features like function calling, structured outputs, and developer messages, and offers adjustable reasoning effort levels (low, medium, high) for optimized performance.",
      tags: ["Language Models", "Transformer Decoder", "OpenAI", "GPT"],
    }
  ],
  [
    // Small LLMs
    {
      title: "Phi-1",
      link: "https://ritvik19.medium.com/papers-explained-114-phi-1-14a8dcc77ce5",
      date: "June 2023",
      description:
        "An LLM for code, trained using a textbook quality data from the web and synthetically generated textbooks and exercises with GPT-3.5.",
        tags: ["Language Models", "Transformer Decoder", "Small LLMs", "Synthetic Data", "Phi"],
    },
    {
      title: "Orca",
      link: "https://ritvik19.medium.com/papers-explained-160-orca-928eff06e7f9",
      date: "June 2023",
      description:
        "Presents a novel approach that addresses the limitations of instruction tuning by leveraging richer imitation signals, scaling tasks and instructions, and utilizing a teacher assistant to help with progressive learning.",
      tags: ["Language Models", "Transformer Decoder", "Small LLMs", "Synthetic Data", "Orca"],
    },
    {
      title: "Phi-1.5",
      link: "https://ritvik19.medium.com/papers-explained-phi-1-5-2857e56dbd2a",
      date: "September 2023",
      description:
        "Follows the phi-1 approach, focusing this time on common sense reasoning in natural language.",
      tags: ["Language Models", "Transformer Decoder", "Small LLMs", "Synthetic Data", "Phi"],
    },
    {
      title: "Mistral 7B",
      link: "https://ritvik19.medium.com/papers-explained-mistral-7b-b9632dedf580",
      date: "October 2023",
      description:
        "Leverages grouped-query attention for faster inference, coupled with sliding window attention to effectively handle sequences of arbitrary length with a reduced inference cost.",
      tags: ["Language Models", "Transformer Decoder", "Small LLMs", "Mistral"],
    },
    {
      title: "Zephyr 7B",
      link: "https://ritvik19.medium.com/papers-explained-71-zephyr-7ec068e2f20b",
      date: "October 2023",
      description:
        "Utilizes dDPO and AI Feedback (AIF) preference data to achieve superior intent alignment in chat-based language modeling.",
      tags: ["Language Models", "Transformer Decoder", "Small LLMs", "HuggingFace"],
    },
    {
      title: "Orca 2",
      link: "https://ritvik19.medium.com/papers-explained-161-orca-2-b6ffbccd1eef",
      date: "November 2023",
      description:
        "Introduces Cautious Reasoning for training smaller models to select the most effective solution strategy based on the problem at hand, by crafting data with task-specific system instruction(s) corresponding to the chosen strategy in order to obtain teacher responses for each task and replacing the student’s system instruction with a generic one vacated of details of how to approach the task.",
      tags: ["Language Models", "Transformer Decoder", "Small LLMs", "Synthetic Data", "Orca"],
    },
    {
      title: "Phi-2",
      link: "https://ritvik19.medium.com/papers-explained-phi-1-5-2857e56dbd2a#8230",
      date: "December 2023",
      description:
        "A 2.7B model, developed to explore whether emergent abilities achieved by large-scale language models can also be achieved at a smaller scale using strategic choices for training, such as data selection.",
      tags: ["Language Models", "Transformer Decoder", "Small LLMs", "Synthetic Data", "Phi"],
    },
    {
      title: "TinyLlama",
      link: "https://ritvik19.medium.com/papers-explained-93-tinyllama-6ef140170da9",
      date: "January 2024",
      description:
        "A  1.1B language model built upon the architecture and tokenizer of Llama 2, pre-trained on around 1 trillion tokens for approximately 3 epochs, leveraging FlashAttention and Grouped Query Attention, to achieve better computational efficiency.",
      tags: ["Language Models", "Transformer Decoder", "Small LLMs"],
    },
    {
      title: "Danube",
      link: "https://ritvik19.medium.com/papers-explained-111-h2o-danube-1-8b-b790c073d257",
      date: "January 2024",
      description:
        "A language model trained on 1T tokens following the core principles of LLama 2 and Mistral, leveraging and refining various techniques for pre-training large language models.",
      tags: ["Language Models", "Transformer Decoder", "Small LLMs", "Danube", "H2O"],
    },
    {
      title: "OLMo",
      link: "https://ritvik19.medium.com/papers-explained-98-olmo-fdc358326f9b",
      date: "February 2024",
      description:
        "A state-of-the-art, truly open language model and framework that includes training data, code, and tools for building, studying, and advancing language models.",
      tags: ["Language Models", "Transformer Decoder", "Small LLMs", "Olmo"],
    },
    {
      title: "Mobile LLM",
      link: "https://ritvik19.medium.com/papers-explained-216-mobilellm-2d7fdd5acd86",
      date: "February 2024",
      description:
        "Leverages various architectures and attention mechanisms to achieve a strong baseline network, which is then improved upon by introducing an immediate block-wise weight-sharing approach, resulting in a further accuracy boost.",
      tags: ["Language Models", "Transformer Decoder", "Small LLMs"],
    },
    {
      title: "Orca Math",
      link: "https://ritvik19.medium.com/papers-explained-163-orca-math-ae6a157ce48d",
      date: "February 2024",
      description:
        "A fine tuned Mistral-7B that excels at math problems without external tools, utilizing a high-quality synthetic dataset of 200K problems created through multi-agent collaboration and an iterative learning process that involves practicing problem-solving, receiving feedback, and learning from preference pairs incorporating the model's solutions and feedback.",
      tags: ["Language Models", "Transformer Decoder", "Small LLMs", "Synthetic Data", "Scientific Data", "Orca"],
    },
    {
      title: "Gemma",
      link: "https://ritvik19.medium.com/papers-explained-106-gemma-ca2b449321ac",
      date: "February 2024",
      description:
        "A family of 2B and 7B, state-of-the-art language models based on Google's Gemini models, offering advancements in language understanding, reasoning, and safety.",
      tags: ["Language Models", "Transformer Decoder", "Small LLMs", "Gemma"],
    },
    {
      title: "CodeGemma",
      link: "https://ritvik19.medium.com/papers-explained-124-codegemma-85faa98af20d",
      date: "April 2024",
      description:
        "Open code models based on Gemma models by further training on over 500 billion tokens of primarily code.",
      tags: ["Language Models", "Transformer Decoder", "Small LLMs", "Code Generation", "Gemma"],
    },
    {
      title: "Phi-3",
      link: "https://ritvik19.medium.com/papers-explained-130-phi-3-0dfc951dc404",
      date: "April 2024",
      description:
        "A series of language models trained on heavily filtered web and synthetic data set, achieving performance comparable to much larger models like Mixtral 8x7B and GPT-3.5.",
      tags: ["Language Models", "Transformer Decoder", "Small LLMs", "Synthetic Data", "Phi"],
    },
    {
      title: "Open ELM",
      link: "https://ritvik19.medium.com/papers-explained-133-open-elm-864f6b28a6ab",
      date: "April 2024",
      description:
        "A fully open language model designed to enhance accuracy while using fewer parameters and pre-training tokens. Utilizes a layer-wise scaling strategy to allocate smaller dimensions in early layers, expanding in later layers.",
      tags: ["Language Models", "Transformer Decoder", "Small LLMs"],
    },
    {
      title: "Danube 2",
      link: "https://ritvik19.medium.com/papers-explained-111-h2o-danube-1-8b-b790c073d257#00d8",
      date: "April 2024",
      description:
        "An updated version of the original H2O-Danube model, with improvements including removal of sliding window attention, changes to the tokenizer, and adjustments to the training data, resulting in significant performance enhancements.",
      tags: ["Language Models", "Transformer Decoder", "Small LLMs", "Danube", "H2O"],
    },
    {
      title: "Granite Code Models",
      link: "https://ritvik19.medium.com/paper-explained-144-granite-code-models-e1a92678739b",
      date: "May 2024",
      description:
        "A family of code models ranging from 3B to 34B trained on 3.5-4.5T tokens of code written in 116 programming languages.",
      tags: ["Language Models", "Transformer Decoder", "Small LLMs", "Code Generation"],
    },
    {
      title: "Gemma 2",
      link: "https://ritvik19.medium.com/papers-explained-157-gemma-2-f1b75b56b9f2",
      date: "June 2024",
      description:
        "Utilizes interleaving local-global attentions and group-query attention, trained with knowledge distillation instead of next token prediction to achieve competitive performance comparable with larger models.",
      tags: ["Language Models", "Transformer Decoder", "Small LLMs", "Gemma"],
    },
    {
      title: "Orca 3 (Agent Instruct)",
      link: "https://ritvik19.medium.com/papers-explained-164-orca-3-agent-instruct-41340505af36",
      date: "July 2024",
      description:
        "A fine tuned Mistral-7B through Generative Teaching via synthetic data generated through the proposed AgentInstruct framework, which generates both the prompts and responses, using only raw data sources like text documents and code files as seeds.",
      tags: ["Language Models", "Transformer Decoder", "Small LLMs", "Synthetic Data", "Orca"],
    },
    {
      title: "Mathstral",
      link: "https://ritvik19.medium.com/papers-explained-mistral-7b-b9632dedf580#0fbe",
      date: "July 2024",
      description:
        "A 7B model designed for math reasoning and scientific discovery based on Mistral 7B specializing in STEM subjects.",
      tags: ["Language Models", "Transformer Decoder", "Small LLMs", "Scientific Data", "Mistral"],
    },
    {
      title: "Smol LM",
      link: "https://ritvik19.medium.com/papers-explained-176-smol-lm-a166d5f1facc",
      date: "July 2024",
      description:
        "A family of small models with 135M, 360M, and 1.7B parameters, utilizes Grouped-Query Attention (GQA), embedding tying, and a context length of 2048 tokens, trained on a new open source high-quality dataset.",
      tags: ["Language Models", "Transformer Decoder", "Small LLMs", "HuggingFace"],
    },
    {
      title: "Mistral Nemo",
      link: "https://ritvik19.medium.com/papers-explained-mistral-7b-b9632dedf580#37cd",
      date: "July 2024",
      description:
        "A 12B Language Model built in collaboration between Mistral and NVIDIA, featuring a context window of 128K, an efficient tokenizer and trained with quantization awareness, enabling FP8 inference without any performance loss.",
      tags: ["Language Models", "Transformer Decoder", "Small LLMs", "Mistral"],
    },
    {
      title: "Minitron",
      link: "https://ritvik19.medium.com/papers-explained-208-minitron-e55ea374d9dd",
      date: "July 2024",
      description:
        "Prunes an existing Nemotron model and re-trains it with a fraction of the original training data, achieving compression factors of 2-4×, compute cost savings of up to 40×, and improved performance on various language modeling tasks.",
      tags: ["Language Models", "Transformer Decoder", "Small LLMs", "Pruning", "Knowledge Distillation", "Nvidia"],
    },
    {
      title: "Danube 3",
      link: "https://ritvik19.medium.com/papers-explained-217-h2o-danube-3-917a7b40a79f",
      date: "July 2024",
      description:
        "A series of 4B and 500M language models, trained on high-quality Web data in three stages with different data mixes before being fine-tuned for chat version.",
      tags: ["Language Models", "Transformer Decoder", "Small LLMs", "Danube", "H2O"],
    },
    {
      title: "Smol LM v0.2",
      link: "https://ritvik19.medium.com/papers-explained-176-smol-lm-a166d5f1facc#fdb2",
      date: "August 2024",
      description:
        "An advancement over SmolLM, better at staying on topic and responding appropriately to standard prompts, such as greetings and questions about their role as AI assistants.",
      tags: ["Language Models", "Transformer Decoder", "Small LLMs", "HuggingFace"],
    },
    {
      title: "Phi-3.5",
      link: "https://ritvik19.medium.com/papers-explained-192-phi-3-5-a95429ea26c9",
      date: "August 2024",
      description:
        "A family of models consisting of three variants - MoE (16x3.8B), mini (3.8B), and vision (4.2B) - which are lightweight, multilingual, and trained on synthetic and filtered publicly available documents - with a focus on very high-quality, reasoning dense data.",
      tags: ["Language Models", "Transformer Decoder", "Small LLMs", "Synthetic Data", "Phi"],
    },
    {
      title: "Minitron Approach in Practice",
      link: "https://ritvik19.medium.com/papers-explained-209-minitron-approach-in-practice-6b473f67328d",
      date: "August 2024",
      description:
        "Applies the minitron approach to Llama 3.1 8B and Mistral-Nemo 12B, additionally applies teacher correction to align with the new data distribution.",
      tags: ["Language Models", "Transformer Decoder", "Small LLMs", "Pruning", "Knowledge Distillation", "Nvidia"],
    },
    {
      title: "Mistral Small",
      link: "https://medium.com/dair-ai/papers-explained-mistral-7b-b9632dedf580#5662",
      date: "September 2024",
      description:
        "A 22B model with significant improvements in human alignment, reasoning capabilities, and code over the previous model.",
      tags: ["Language Models", "Transformer Decoder", "Small LLMs", "Mistral"],
    },
    {
      title: "Nemotron-Mini-Hindi",
      link: "https://ritvik19.medium.com/papers-explained-252-nemotron-mini-hindi-c7adc3b2f759",
      date: "October 2024",
      description:
        "A bilingual language model based on Nemotron-Mini 4B, specifically trained to improve Hindi and English performance using continuous pre-training on 400B real and synthetic tokens.",
      tags: ["Language Models", "Transformer Decoder", "Small LLMs", "Nvidia", "Multilingual Models"],
    },
    {
      title: "Ministral",
      link: "https://ritvik19.medium.com/papers-explained-mistral-7b-b9632dedf580#1f34",
      date: "October 2024",
      description:
        "3B and 8B models with support up to 128k context length having a special interleaved sliding-window attention pattern for faster and memory-efficient inference.",
      tags: ["Language Models", "Transformer Decoder", "Small LLMs", "Mistral"],
    },
    {
      title: "Quantized Llama 3.2",
      link: "https://ritvik19.medium.com/papers-explained-187e-quantized-llama-3-2-cc6965f61370",
      date: "October 2024",
      description:
        "Optimized versions of the Llama, using techniques like Quantization-Aware Training with LoRA Adapters and SpinQuant, to reduce model size and memory usage while maintaining accuracy and performance, enabling deployment on resource-constrained devices like mobile phones.",
      tags: ["Language Models", "Transformer Decoder", "Small LLMs", "Quantization"],
    },
    {
      title: "Smol LM v2",
      link: "https://ritvik19.medium.com/papers-explained-176-smol-lm-a166d5f1facc#aa17",
      date: "November 2024",
      description:
        "A family of language models (135M, 360M, and 1.7B parameters), trained on 2T, 4T, and 11T tokens respectively from datasets including FineWeb-Edu, DCLM, The Stack, and curated math and coding datasets, with instruction-tuned versions created using Smol Talk dataset and DPO using UltraFeedback.",
      tags: ["Language Models", "Transformer Decoder", "Small LLMs", "HuggingFace"],
    },
    {
      title: "Command R 7B",
      link: "https://ritvik19.medium.com/papers-explained-166-command-r-models-94ba068ebd2b#0836",
      date: "December 2024",
      description:
        "The smallest, fastest, and final model in the R series of enterprise-focused LLMs. It offers a context length of 128k and delivers a powerful combination of multilingual support, citation verified retrieval-augmented generation (RAG), reasoning, tool use, and agentic behavior.",
      tags: ["Language Models", "Transformer Decoder", "Small LLMs", "Cohere"],
    },
    {
      title: "Phi-4",
      link: "https://ritvik19.medium.com/papers-explained-278-phi-4-ea59220f3f88",
      date: "December 2024",
      description:
        "A 14B language model prioritizing data quality through a training process incorporating synthetic data for pretraining and midtraining, curated organic data seeds, and innovative post-training techniques like pivotal token search for DPO, resulting in strong performance on reasoning-focused benchmarks, especially in STEM, comparable to much larger models, while also addressing overfitting and data contamination concerns.",
      tags: ["Language Models", "Transformer Decoder", "Small LLMs", "Synthetic Data", "Phi"],
    },
    {
      title: "Mistral Small 3",
      link: "https://ritvik19.medium.com/papers-explained-mistral-7b-b9632dedf580#9c9b",
      date: "January 2025",
      description:
        "A latency-optimized, 24B parameter language model, designed for efficient handling of common generative AI tasks requiring strong language understanding and instruction following.",
      tags: ["Language Models", "Transformer Decoder", "Small LLMs", "Mistral"],
    }
  ],
  [
    // Multi Modal Language Models
    {
      title: "Florence",
      link: "https://ritvik19.medium.com/papers-explained-213-florence-f93a3a7d9ef0",
      date: "November 2021",
      description:
        "A computer vision foundation model that can be adapted to various tasks by expanding representations from coarse (scene) to fine (object), static (images) to dynamic (videos), and RGB to multiple modalities.",
      tags: ["Multimodal Models"],
    },
    {
      title: "BLIP",
      link: "https://ritvik19.medium.com/papers-explained-154-blip-6d85c80a744d",
      date: "February 2022",
      description:
        "A Vision-Language Pre-training (VLP) framework that introduces Multimodal mixture of Encoder-Decoder (MED) and Captioning and Filtering (CapFilt), a new dataset bootstrapping method for learning from noisy image-text pairs.",
      tags: ["Multimodal Models"],
    },
    {
      title: "Flamingo",
      link: "https://ritvik19.medium.com/papers-explained-82-flamingo-8c124c394cdb",
      date: "April 2022",
      description:
        "Visual Language Models enabling seamless handling of interleaved visual and textual data, and facilitating few-shot learning on large-scale web corpora.",
      tags: ["Multimodal Models"],
    },
    {
      title: "PaLI",
      link: "https://ritvik19.medium.com/papers-explained-194-pali-c1fffc14068c",
      date: "September 2022",
      description:
        "A joint language-vision model that generates multilingual text based on visual and textual inputs, trained using large pre-trained encoder-decoder language models and Vision Transformers, specifically mT5 and ViT-e.",
      tags: ["Multimodal Models"],
    },
    {
      title: "BLIP 2",
      link: "https://ritvik19.medium.com/papers-explained-155-blip-2-135fff70bf65",
      date: "January 2023",
      description:
        "A Vision-Language Pre-training (VLP) framework that proposes Q-Former, a trainable module to bridge the gap between a frozen image encoder and a frozen LLM to bootstrap vision-language pre-training.",
      tags: ["Multimodal Models"],
    },
    {
      title: "LLaVA 1",
      link: "https://ritvik19.medium.com/papers-explained-102-llava-1-eb0a3db7e43c",
      date: "April 2023",
      description:
        "A large multimodal model connecting CLIP and Vicuna trained end-to-end on instruction-following data generated through GPT-4 from image-text pairs.",
      tags: ["Multimodal Models"],
    },
    {
      title: "PaLI-X",
      link: "https://ritvik19.medium.com/papers-explained-195-pali-x-f9859e73fd97",
      date: "May 2023",
      description: "A multilingual vision and language model with scaled-up components, specifically ViT-22 B and UL2 32B, exhibits emergent properties such as complex counting and multilingual object detection, and demonstrates improved performance across various tasks.",
      tags: ["Multimodal Models"],
    },
    {
      title: "InstructBLIP",
      link: "https://ritvik19.medium.com/papers-explained-156-instructblip-c3cf3291a823",
      date: "May 2023",
      description:
        "Introduces instruction-aware Query Transformer to extract informative features tailored to the given instruction to study vision-language instruction tuning based on the pretrained BLIP-2 models.",
      tags: ["Multimodal Models"],
    },
    {
      title: "Idefics",
      link: "https://ritvik19.medium.com/papers-explained-179-obelics-idefics-a581f8d909b6",
      date: "June 2023",
      description:
        "9B and 80B multimodal models trained on Obelics, an open web-scale dataset of interleaved image-text documents, curated in this work.",
      tags: ["Multimodal Models", "HuggingFace"],
    },
    {
      title: "GPT-4V",
      link: "https://ritvik19.medium.com/papers-explained-68-gpt-4v-6e27c8a1d6ea",
      date: "September 2023",
      description:
        "A multimodal model that combines text and vision capabilities, allowing users to instruct it to analyze image inputs.",
      tags: ["Multimodal Models", "OpenAI", "GPT"],
    },
    {
      title: "PaLI-3",
      link: "https://ritvik19.medium.com/papers-explained-196-pali-3-2f5cf92f60a8",
      date: "October 2023",
      description:
        "A 5B vision language model, built upon a 2B SigLIP Vision Model and UL2 3B Language Model outperforms larger models on various benchmarks and achieves SOTA on several video QA benchmarks despite not being pretrained on any video data.",
      tags: ["Multimodal Models"],
    },
    {
      title: "LLaVA 1.5",
      link: "https://ritvik19.medium.com/papers-explained-103-llava-1-5-ddcb2e7f95b4",
      date: "October 2023",
      description:
        "An enhanced version of the LLaVA model that incorporates a CLIP-ViT-L-336px with an MLP projection and academic-task-oriented VQA data to set new benchmarks in large multimodal models (LMM) research.",
      tags: ["Multimodal Models"],
    },
    {
      title: "Florence-2",
      link: "https://ritvik19.medium.com/papers-explained-214-florence-2-c4e17246d14b",
      date: "November 2023",
      description:
        "A vision foundation model with a unified, prompt-based representation for a variety of computer vision and vision-language tasks.",
      tags: ["Multimodal Models"],
    },
    {
      title: "CogVLM",
      link: "https://ritvik19.medium.com/papers-explained-235-cogvlm-9f3aa657f9b1",
      date: "November 2023",
      description:
        "Bridges the gap between the frozen pretrained language model and image encoder by a trainable visual expert module in the attention and FFN layers.",
      tags: ["Multimodal Models"],
    },
    {
      title: "Gemini 1.0",
      link: "https://ritvik19.medium.com/papers-explained-80-gemini-1-0-97308ef96fcd",
      date: "December 2023",
      description:
        "A family of highly capable multi-modal models, trained jointly across image, audio, video, and text data for the purpose of building a model with strong generalist capabilities across modalities.",
      tags: ["Multimodal Models", "Gemini"],
    },
    {
      title: "MoE-LLaVA",
      link: "https://ritvik19.medium.com/papers-explained-104-moe-llava-cf14fda01e6f",
      date: "January 2024",
      description:
        "A MoE-based sparse LVLM framework that activates only the top-k experts through routers during deployment, maintaining computational efficiency while achieving comparable performance to larger models.",
      tags: ["Multimodal Models", "Mixtures of Experts"],
    },
    {
      title: "LLaVA 1.6",
      link: "https://ritvik19.medium.com/papers-explained-107-llava-1-6-a312efd496c5",
      date: "January 2024",
      description:
        "An improved version of a LLaVA 1.5 with enhanced reasoning, OCR, and world knowledge capabilities, featuring increased image resolution",
      tags: ["Multimodal Models"],
    },
    {
      title: "Gemini 1.5 Pro",
      link: "https://ritvik19.medium.com/papers-explained-105-gemini-1-5-pro-029bbce3b067",
      date: "February 2024",
      description:
        "A highly compute-efficient multimodal mixture-of-experts model that excels in long-context retrieval tasks and understanding across text, video, and audio modalities.",
      tags: ["Multimodal Models", "Mixtures of Experts", "Gemini"],
    },
    {
      title: "Claude 3",
      link: "https://ritvik19.medium.com/papers-explained-181-claude-89dd45e35d92",
      date: "March 2024",
      description:
        "A family of VLMs consisting of Haiku, Sonnet, and Opus models, sets new industry standards for cognitive tasks, offering varying levels of intelligence, speed, and cost-efficiency.",
      tags: ["Multimodal Models", "Anthropic", "Claude"],
    },
    {
      title: "MM-1",
      link: "https://ritvik19.medium.com/papers-explained-117-mm1-c579142bcdc0",
      date: "March 2024",
      description:
        "Studies the importance of various architecture components and data choices. Through comprehensive ablations of the image encoder, the vision language connector, and various pre-training data choices, and identifies several crucial design lessons.",
      tags: ["Multimodal Models"],
    },
    {
      title: "Grok 1.5 V",
      link: "https://ritvik19.medium.com/papers-explained-186-grok-0d9f1aef69be",
      date: "April 2024",
      description: "The first multimodal model in the grok series.",
      tags: ["Multimodal Models", "Grok"],
    },
    {
      title: "Idefics2",
      link: "https://ritvik19.medium.com/papers-explained-180-idefics-2-0adf35cef4ee",
      date: "April 2024",
      description:
        "Improvement upon Idefics1 with enhanced OCR capabilities, simplified architecture, and better pre-trained backbones, trained on a mixture of openly available datasets and fine-tuned on task-oriented data.",
      tags: ["Multimodal Models", "HuggingFace"],
    },
    {
      title: "Phi-3 Vision",
      link: "https://ritvik19.medium.com/papers-explained-130-phi-3-0dfc951dc404#7ba6",
      date: "May 2024",
      description:
        "First multimodal model in the Phi family, bringing the ability to reason over images and extract and reason over text from images.",
      tags: ["Multimodal Models", "Synthetic Data", "Phi"],
    },
    {
      title: "An Introduction to Vision-Language Modeling",
      link: "https://ritvik19.medium.com/papers-explained-an-introduction-to-vision-language-modeling-89e7697da6e3",
      date: "May 2024",
      description:
        "Provides a comprehensive introduction to VLMs, covering their definition, functionality, training methods, and evaluation approaches, aiming to help researchers and practitioners enter the field and advance the development of VLMs for various applications.",
      tags: ["Multimodal Models"],
    },
    {
      title: "GPT-4o",
      link: "https://ritvik19.medium.com/papers-explained-185-gpt-4o-a234bccfd662",
      date: "May 2024",
      description:
        "An omni model accepting and generating various types of inputs and outputs, including text, audio, images, and video.",
      tags: ["Multimodal Models", "OpenAI", "GPT"],
    },
    {
      title: "Gemini 1.5 Flash",
      link: "https://ritvik19.medium.com/papers-explained-142-gemini-1-5-flash-415e2dc6a989",
      date: "May 2024",
      description:
        "A more lightweight variant of the Gemini 1.5 pro, designed for efficiency with minimal regression in quality, making it suitable for applications where compute resources are limited.",
      tags: ["Multimodal Models", "Gemini"],
    },
    {
      title: "Chameleon",
      link: "https://ritvik19.medium.com/papers-explained-143-chameleon-6cddfdbceaa8",
      date: "May 2024",
      description:
        "A family of early-fusion token-based mixed-modal models capable of understanding and generating images and text in any arbitrary sequence.",
      tags: ["Multimodal Models"],
    },
    {
      title: "Claude 3.5 Sonnet",
      link: "https://ritvik19.medium.com/papers-explained-181-claude-89dd45e35d92#2a14",
      date: "June 2024",
      description:
        "Surpasses previous versions and competitors in intelligence, speed, and cost-efficiency, excelling in graduate-level reasoning, undergraduate-level knowledge, coding proficiency, and visual reasoning.",
      tags: ["Multimodal Models", "Anthropic", "Claude"],
    },
    {
      title: "Pali Gemma",
      link: "https://ritvik19.medium.com/papers-explained-197-pali-gemma-6899e871998e",
      date: "July 2024",
      description:
        "Combines SigLIP vision model and the Gemma language model and follows the PaLI-3 training recipe to achieve strong performance on various vision-language tasks.",
        tags: ["Multimodal Models", "Gemma"],
    },
    {
      title: "GPT-4o mini",
      link: "https://ritvik19.medium.com/papers-explained-185-gpt-4o-a234bccfd662#08b9",
      date: "July 2024",
      description:
        "A cost-efficient small model that outperforms GPT-4 on chat preferences, enabling a broad range of tasks with low latency and supporting text, vision, and multimodal inputs and outputs.",
      tags: ["Multimodal Models", "OpenAI", "GPT"],
    },
    {
      title: "Grok 2",
      link: "https://ritvik19.medium.com/papers-explained-186-grok-0d9f1aef69be",
      date: "August 2024",
      description:
        "A frontier language model with state-of-the-art capabilities in chat, coding, and reasoning on par with Claude 3.5 Sonnet and GPT-4-Turbo.",
      tags: ["Multimodal Models", "Grok"],
    },
    {
      title: "BLIP-3 (xGen-MM)",
      link: "https://ritvik19.medium.com/papers-explained-190-blip-3-xgen-mm-6a9c04a3892d",
      date: "August 2024",
      description:
        "A comprehensive system for developing Large Multimodal Models, comprising curated datasets, training recipes, model architectures, and pre-trained models that demonstrate strong in-context learning capabilities and competitive performance on various tasks.",
      tags: ["Multimodal Models"],
    },
    {
      title: "Idefics 3",
      link: "https://ritvik19.medium.com/papers-explained-218-idefics-3-81791c4cde3f",
      date: "August 2024",
      description:
        "A VLM based on Llama 3.1 and SigLIP-SO400M trained efficiently, using only open datasets and a straightforward pipeline, significantly outperforming in document understanding tasks.",
      tags: ["Multimodal Models", "HuggingFace"],
    },
    {
      title: "CogVLM2",
      link: "https://ritvik19.medium.com/papers-explained-236-cogvlm2-db0261745cf5",
      date: "August 2024",
      description:
        "A family of visual language models that enables image and video understanding with improved training recipes, exploring enhanced vision-language fusion, higher input resolution, and broader modalities and applications.",
      tags: ["Multimodal Models"],
    },
    {
      title: "Eagle",
      link: "https://ritvik19.medium.com/papers-explained-269-eagle-09c21e549395",
      date: "August 2024",
      description:
        "Provides an extensive exploration of the design space for MLLMs using a mixture of vision encoders and resolutions, and reveals several underlying principles common to various existing strategies, leading to a streamlined yet effective design approach.",
      tags: ["Multimodal Models", "Nvidia"],
    },
    {
      title: "Pixtral",
      link: "https://ritvik19.medium.com/papers-explained-219-pixtral-a714f94e59ac",
      date: "September 2024",
      description:
        "A 12B parameter natively multimodal vision-language model, trained with interleaved image and text data demonstrating strong performance on multimodal tasks, and excels in instruction following.",
      tags: ["Multimodal Models", "Mistral"],
    },
    {
      title: "NVLM",
      link: "https://ritvik19.medium.com/papers-explained-240-nvlm-7ad201bfbfc2",
      date: "September 2024",
      description:
        "A family of multimodal large language models, provides a comparison between decoder-only multimodal LLMs and cross-attention based models and proposes a hybrid architecture, it further introduces a 1-D title-tagging design for tile-based dynamic high resolution images.",
      tags: ["Multimodal Models", "Nvidia"],
    },
    {
      title: "Molmo",
      link: "https://ritvik19.medium.com/papers-explained-241-pixmo-and-molmo-239d70abebff",
      date: "September 2024",
      description:
        "A family of open-weight vision-language models that achieve state-of-the-art performance by leveraging a novel, human-annotated image caption dataset called PixMo.",
      tags: ["Multimodal Models"],
    },
    {
      title: "MM-1.5",
      link: "https://ritvik19.medium.com/papers-explained-261-mm-1-5-d0dd01a9b68b",
      date: "September 2024",
      description:
        "A family of multimodal large language models designed to enhance capabilities in text-rich image understanding, visual referring and grounding, and multi-image reasoning, achieved through a data-centric approach involving diverse data mixtures, And specialized variants for video and mobile UI understanding.",
      tags: ["Multimodal Models"],
    },
    {
      title: "Mississippi",
      link: "https://ritvik19.medium.com/papers-explained-251-h2ovl-mississippi-1508e9c8e862",
      date: "October 2024",
      description:
        "A collection of small, efficient, open-source vision-language models built on top of Danube, trained on 37 million image-text pairs, specifically designed to perform well on document analysis and OCR tasks while maintaining strong performance on general vision-language benchmarks.",
      tags: ["Multimodal Models", "Danube", "H2O"],
    },
    {
      title: "Claude 3.5 Haiku",
      link: "https://ritvik19.medium.com/papers-explained-181-claude-89dd45e35d92#9637",
      date: "October 2024",
      description:
        "A fast and affordable language model that excels in tasks such as coding, reasoning, and content creation.",
      tags: ["Multimodal Models", "Anthropic", "Claude"],
    },
    {
      title: "Pixtral Large",
      link: "https://ritvik19.medium.com/papers-explained-219-pixtral-a714f94e59ac#6123",
      date: "November 2024",
      description: 
        "A 124 billion parameter open-weight multimodal model built upon Mistral Large 2 and a 1B parameter vision encoder, excelling in understanding documents, charts and natural images. It supports a Context Window of 128K tokens, accommodating at least 30 high-resolution images.",
      tags: ["Multimodal Models", "Mistral"],  
    },
    {
      title: "Smol VLM",
      link: "https://ritvik19.medium.com/papers-explained-176-smol-lm-a166d5f1facc#6245",
      date: "November 2024",
      description:
        "A 2B vision-language model, built using a modified Idefics3 architecture with a smaller language backbone (SmolLM2 1.7B), aggressive pixel shuffle compression, 384x384 image patches, and a shape-optimized SigLIP vision backbone, featuring a 16k token context window.",
      tags: ["Multimodal Models", "HuggingFace"],
    },
    {
      title: "MAmmoTH-VL",
      link: "https://ritvik19.medium.com/papers-explained-296-mammoth-vl-6abec7a58831",
      date: "December 2024",
      description:
        "Curated a large-scale, multimodal instruction-tuning dataset with 12M instruction-response pairs using a cost-effective method involving open-source data collection and categorization, task-specific data augmentation and rewriting using open models (Llama-3-70B-Instruct for caption-based data and InternVL2-Llama3-76B for other data types), and self-filtering with InternVL2-Llama3-76B to remove hallucinations and ensure data quality.",
      tags: ["Multimodal Models", "Synthetic Data"],
    },
    {
      title: "Maya",
      link: "https://ritvik19.medium.com/papers-explained-297-maya-0a38799daa43",
      date: "December 2024",
      description:
        "An open-source multilingual multimodal model designed to improve vision-language understanding in eight languages (English, Chinese, French, Spanish, Russian, Hindi, Japanese, and Arabic). It leverages a newly created, toxicity-filtered multilingual image-text dataset based on LLaVA, incorporating a SigLIP vision encoder and the Aya-23 8B language model, and is fine-tuned on the PALO 150K instruction-tuning dataset.",
      tags: ["Multimodal Models", "Multilingual Models"],
    },
    {
      title: "Llava-Mini",
      link: "https://ritvik19.medium.com/papers-explained-298-llava-mini-fe3a25b9e747",
      date: "January 2025",
      description:
        "An efficient large multimodal model that minimizes vision tokens by pre-fusing visual information from a CLIP vision encoder into text tokens before feeding them, along with a small number of compressed vision tokens (achieved via query-based compression), to an LLM backbone, allowing for efficient processing of standard and high-resolution images, as well as videos, by significantly reducing the number of tokens the LLM needs to handle while preserving visual understanding.",
      tags: ["Multimodal Models"],
    }
  ],
  [
    // LLMS for Math
    {
      title: "Wizard Math",
      link: "https://ritvik19.medium.com/papers-explained-review-12-llms-for-maths-1597e3c7251c#ecf7",
      date: "August 2023",
      description:
        "Proposes Reinforcement Learning from Evol-Instruct Feedback (RLEIF) method, applied to Llama-2 to enhance the mathematical reasoning abilities.",
      tags: ["LLM for Math"],
    },
    {
      title: "MAmmoTH",
      link: "https://ritvik19.medium.com/papers-explained-review-12-llms-for-maths-1597e3c7251c#8cd9",
      date: "September 2023",
      description:
        "A series of LLMs specifically designed for general math problem-solving, trained on MathInstruct, a dataset compiled from 13 math datasets with intermediate rationales that combines chain-of-thought and program-of-thought approaches to accommodate different thought processes for various math problems.",
      tags: ["LLM for Math"],
    },
    {
      title: "MetaMath",
      link: "https://ritvik19.medium.com/papers-explained-review-12-llms-for-maths-1597e3c7251c#bd74",
      date: "September 2023",
      description:
        "A fine-tuned language model that specializes in mathematical reasoning, achieved by bootstrapping mathematical questions from multiple perspectives without extra knowledge and then fine-tuning an LLaMA-2 model on the resulting dataset.",
      tags: ["LLM for Math"],
    },
    {
      title: "ToRA",
      link: "https://ritvik19.medium.com/papers-explained-review-12-llms-for-maths-1597e3c7251c#feb4",
      date: "September 2023",
      description:
        "A series of large language models that seamlessly integrate natural language reasoning with external tools to solve complex mathematical problems.",
      tags: ["LLM for Math"],
    },
    {
      title: "Math Coder",
      link: "https://ritvik19.medium.com/papers-explained-review-12-llms-for-maths-1597e3c7251c#102e",
      date: "October 2023",
      description:
        "The first systematic study that explicitly integrates natural language reasoning, code generation, and feedback from execution results into open-source pre-trained large language models.",
      tags: ["LLM for Math"],
    },
    {
      title: "MuggleMath",
      link: "https://ritvik19.medium.com/papers-explained-review-12-llms-for-maths-1597e3c7251c#c53e",
      date: "October 2023",
      description:
        "GSM8K and MATH datasets are augmented using query augmentation and response augmentation to fine tune Llama 2 models.",
      tags: ["LLM for Math"],
    },
    {
      title: "Llemma",
      link: "https://ritvik19.medium.com/papers-explained-review-12-llms-for-maths-1597e3c7251c#1b38",
      date: "October 2023",
      description:
        "An LLM for mathematics, formed by continued pretraining of Code Llama on a mixture of scientific papers, web data containing mathematics, and mathematical code.",
      tags: ["LLM for Math"],
    },
    {
      title: "MuMath",
      link: "https://ritvik19.medium.com/papers-explained-review-12-llms-for-maths-1597e3c7251c#25c2",
      date: "December 2023",
      description:
        "A multi-perspective augmentation dataset for mathematics that combines strengths from tool-free methods, broadening the scope of augmentation techniques to enhance mathematical reasoning capabilities.",
      tags: ["LLM for Math"],
    },
    {
      title: "MMIQC",
      link: "https://ritvik19.medium.com/papers-explained-review-12-llms-for-maths-1597e3c7251c#3a88",
      date: "January 2024",
      description:
        "Curates a data set by sampling Meta Math, Iterative Question Composition, Utilizing Stack Exchange, Augmenting Similar Problems, Answer Augmentation and Question Bootstrapping to finetune Mistral, Llemma, DeepSeek and Qwen.",
      tags: ["LLM for Math"],
    },
    {
      title: "DeepSeek Math",
      link: "https://ritvik19.medium.com/papers-explained-review-12-llms-for-maths-1597e3c7251c#7760",
      date: "February 2024",
      description:
        "A language model for mathematics leveraging  a variant of Proximal Policy Optimization (PPO) called Group Relative Policy Optimization (GRPO), without relying on external toolkits or voting techniques.",
      tags: ["LLM for Math"],
    },
    {
      title: "Open Math Instruct 1",
      link: "https://ritvik19.medium.com/papers-explained-review-12-llms-for-maths-1597e3c7251c#8709",
      date: "February 2024",
      description:
        "A math instruction tuning dataset containing 1.8M code interpreter styled problem-solution pairs for GSM8K and MATH dataset, generated using Mixtral.",
      tags: ["LLM for Math"],
    },
    {
      title: "Math Orca",
      link: "https://ritvik19.medium.com/papers-explained-review-12-llms-for-maths-1597e3c7251c#80d0",
      date: "February 2024",
      description:
        "A fine tuned Mistral-7B that excels at math problems without external tools, utilizing a high-quality synthetic dataset of 200K problems created through multi-agent collaboration and an iterative learning process that involves practicing problem-solving, receiving feedback, and learning from preference pairs incorporating the model's solutions and feedback.",
      tags: ["LLM for Math"],
    },
    {
      title: "Math Genie",
      link: "https://ritvik19.medium.com/papers-explained-review-12-llms-for-maths-1597e3c7251c#8894",
      date: "February 2024",
      description:
        "A framework that generates diverse and reliable math problems by iteratively augmenting small-scale problem-solution datasets and back-translating, followed by generating and verifying code-integrated solutions",
      tags: ["LLM for Math"],
    },
    {
      title: "Xwin-Math",
      link: "https://ritvik19.medium.com/papers-explained-review-12-llms-for-maths-1597e3c7251c#f71c",
      date: "March 2024",
      description:
        "Scales up GSM 8K and MATH dataset to 1.44M questions using data synthesis and then Llama 2, Mistral, Llemma are finetuned on the curated datasets.",
      tags: ["LLM for Math"],
    },
    {
      title: "MuMath Code",
      link: "https://ritvik19.medium.com/papers-explained-review-12-llms-for-maths-1597e3c7251c#9fb8",
      date: "May 2024",
      description:
        "Integrates tool use and data augmentation by finetuning Llama-2 on a dataset of math questions with code-nested solutions.",
      tags: ["LLM for Math"],
    },
    {
      title: "Numina Math",
      link: "https://ritvik19.medium.com/papers-explained-review-12-llms-for-maths-1597e3c7251c#b743",
      date: "July 2024",
      description:
        "Winner of the 1st AIMO Progress Prize, based on DeepSeek Math, finetuned in two stages: CoT and ToRA.",
      tags: ["LLM for Math"],
    },
    {
      title: "Qwen 2 Math",
      link: "https://ritvik19.medium.com/papers-explained-review-12-llms-for-maths-1597e3c7251c#e37a",
      date: "August 2024",
      description:
        "A model series focused on mathematical capabilities, built upon the Qwen2, outperforming proprietary models such as GPT-4o and Claude 3.5 in math-related tasks.",
      tags: ["LLM for Math"],
    },
    {
      title: "Qwen 2.5 Math",
      link: "https://ritvik19.medium.com/papers-explained-review-12-llms-for-maths-1597e3c7251c#eb40",
      date: "September 2024",
      description:
        "An upgrade of Qwen 2 Math series with improved performance and expanded support to Tool Integrated Reasoning.",
      tags: ["LLM for Math"],
    },
    {
      title: "Open Math Instruct 2",
      link: "https://ritvik19.medium.com/papers-explained-review-12-llms-for-maths-1597e3c7251c#ff3a",
      date: "October 2024",
      description:
        "A math instruction tuning dataset containing 14M question-solution pairs (≈ 600K unique questions) augmented from GSM8K and MATH dataset, generated using Llama 3.2 405B.",
      tags: ["LLM for Math"],
    },
    {
      title: "Math Coder 2",
      link: "https://ritvik19.medium.com/papers-explained-review-12-llms-for-maths-1597e3c7251c#97c9",
      date: "October 2024",
      description:
        "Creates Math Code Pile by filtering web data and then extracting and translating mathematical data from it to create a corpus of interleaved reasoning and code data.",
      tags: ["LLM for Math"],
    },
    {
      title: "AceMath",
      link: "https://ritvik19.medium.com/papers-explained-review-12-llms-for-maths-1597e3c7251c#65b1",
      date: "December 2024",
      description:
        "A suite of math instruction-following models, trained through a two-stage SFT process focusing on general and math-specific reasoning, utilizing high-quality synthetic data and a specialized reward model (AceMath-RM) trained with diverse responses.",
      tags: ["LLM for Math"],
    }
  ],
  [
    // Retrieval and Representation Learning
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
      title: "SigLip",
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
      link: "",
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
      link: "",
      date: "January 2025",
      description:
        "Embedding models designed for visual document retrieval. Trained on a large synthetic dataset using a DSE approach, improving retrieval quality, in cross-lingual scenarios and for visual-heavy documents, and support Matryoshka Representation Learning for reduced vector size with minimal performance impact.",
      tags: ["Retrieval", "Representation Learning", "Multimodal Models", "Multilingual Models"],
    }
  ],
  [
    // LLM Training
    {
      title: "Self-Taught Reasoner (STaR)",
      link: "https://ritvik19.medium.com/papers-explained-288-star-cf485a5b117e",
      date: "May 2022",
      description:
        "A bootstrapping method that iteratively improves a language model's reasoning abilities by generating rationales for a dataset, filtering for rationales that lead to correct answers, fine-tuning the model on these successful rationales, and repeating this process, optionally augmented by 'rationalization' where the model generates rationales given the correct answer as a hint.",
      tags: [],
    },
    {
      title: "Self Instruct",
      link: "https://ritvik19.medium.com/papers-explained-112-self-instruct-5c192580103a",
      date: "December 2022",
      description:
        "A framework for improving the instruction-following capabilities of pretrained language models by bootstrapping off their own generations.",
      tags: ["Synthetic Data"],
    },
    {
      title: "Reinforced Self-Training (ReST)",
      link: "https://ritvik19.medium.com/papers-explained-301-rest-6389371a68ac",
      date: "April 2023",
      description:
        "Iteratively improves a language model by generating a dataset of samples from the current policy (Grow step), filtering those samples based on a reward model derived from human preferences (Improve step), and then fine-tuning the model on the filtered data using an offline RL objective, repeating this process with increasing filtering thresholds to continually refine the model's output quality.",
      tags: [],
    },
    {
      title: "Reward rAnked FineTuning (RAFT)",
      link: "https://ritvik19.medium.com/papers-explained-303-reward-ranked-finetuning-raft-791154585908",
      date: "April 2023",
      description:
        "Generative foundation models are susceptible to implicit biases that can arise from extensive unsupervised training data. Such biases can produce suboptimal samples, skewed outcomes, and unfairness, with potentially serious consequences. Consequently, aligning these models with human ethics and preferences is an essential step toward ensuring their responsible and effective deployment in real-world applications. Prior research has primarily employed Reinforcement Learning from Human Feedback (RLHF) to address this problem, where generative models are fine-tuned with RL algorithms guided by a human-feedback-informed reward model. However, the inefficiencies and instabilities associated with RL algorithms frequently present substantial obstacles to the successful alignment, necessitating the development of a more robust and streamlined approach. To this end, we introduce a new framework, Reward rAnked FineTuning (RAFT), designed to align generative models effectively. Utilizing a reward model and a sufficient number of samples, our approach selects the high-quality samples, discarding those that exhibit undesired behavior, and subsequently enhancing the model by fine-tuning on these filtered samples. Our studies show that RAFT can effectively improve the model performance in both reward learning and other automated metrics in both large language models and diffusion models.",
      tags: [],
    },
    {
      title: "ReST^EM",
      link: "https://ritvik19.medium.com/papers-explained-302-rest-em-9abe7c76936e",
      date: "December 2023",
      description:
        "A self-training method based on expectation-maximization for reinforcement learning with language models. It iteratively generates samples from the model, filters them using binary feedback (E-step), and fine-tunes the base pretrained model on these filtered samples (M-step). Unlike the original ReST, ReST^EM doesn't augment with human data and fine-tunes the base model each iteration, improving transfer performance.",
      tags: [],
    },
    {
      title: "Direct Preference Optimization",
      link: "https://ritvik19.medium.com/papers-explained-148-direct-preference-optimization-d3e031a41be1",
      date: "December 2023",
      description:
        "A stable, performant, and computationally lightweight algorithm that fine-tunes llms to align with human preferences without the need for reinforcement learning, by directly optimizing for the policy best satisfying the preferences with a simple classification objective.",
      tags: [],
    },
    {
      title: "V-STaR",
      link: "https://ritvik19.medium.com/papers-explained-289-v-star-4d2aeedab861",
      date: "February 2024",
      description:
        "Iteratively improves a language model's reasoning abilities by training a verifier with Direct Preference Optimization (DPO) on both correct and incorrect solutions generated by the model, while simultaneously fine-tuning the generator on only the correct solutions, ultimately using the verifier at inference time to select the best solution among multiple candidates.",
      tags: [],
    },
    {
      title: "Retrieval Augmented Fine Tuning (RAFT)",
      link: "https://ritvik19.medium.com/papers-explained-272-raft-5049520bcc26",
      date: "March 2024",
      description:
        "A training method that enhances the performance of LLMs for open-book in-domain question answering by training them to ignore irrelevant documents, cite verbatim relevant passages, and promote logical reasoning.",
      tags: [],
    },
    {
      title: "WRAP",
      link: "https://ritvik19.medium.com/papers-explained-118-wrap-e563e009fe56",
      date: "March 2024",
      description:
        "Uses an off-the-shelf instruction-tuned model prompted to paraphrase documents on the web in specific styles to jointly pre-train LLMs on real and synthetic rephrases.",
      tags: ["Synthetic Data"],
    },
    {
      title: "RLHF Workflow",
      link: "https://ritvik19.medium.com/papers-explained-149-rlhf-workflow-56b4e00019ed",
      date: "May 2024",
      description:
        "Provides a detailed recipe for  online iterative RLHF and achieves state-of-the-art performance on various benchmarks using fully open-source datasets.",
      tags: [],
    },
    {
      title: "Magpie",
      link: "https://ritvik19.medium.com/papers-explained-183-magpie-0603cbdc69c3",
      date: "June 2024",
      description:
        "A self-synthesis method that extracts high-quality instruction data at scale by prompting an aligned LLM with left-side templates, generating 4M instructions and their corresponding responses.",
      tags: ["Synthetic Data"],
    },
    {
      title: "Instruction Pre-Training",
      link: "https://ritvik19.medium.com/papers-explained-184-instruction-pretraining-ee0466f0fd33",
      date: "June 2024",
      description:
        "A framework to augment massive raw corpora with instruction-response pairs enabling supervised multitask pretraining of LMs.",
      tags: ["Synthetic Data"],
    },
    {
      title: "Self-Taught Evaluators",
      link: "https://ritvik19.medium.com/papers-explained-276-self-taught-evaluators-8270905392ed",
      date: "August 2024",
      description:
        "An iterative training scheme that uses only synthetically generated preference data, without human annotations, to improve an LLM's ability to judge the quality of model responses by iteratively generating contrasting model outputs, training an LLM-as-a-Judge to produce reasoning traces and judgments, and using the improved predictions in subsequent iterations.",
      tags: [],
    },
    {
      title: "Direct Judgement Preference Optimization",
      link: "https://ritvik19.medium.com/papers-explained-228-direct-judgement-preference-optimization-6915425402bf",
      date: "September 2024",
      description:
        "Proposes learning through preference optimization to enhance the evaluation capabilities of LLM judges which are trained on three approaches: Chain-of-Thought Critique, Standard Judgement, and Response Deduction across various use cases, including single rating, pairwise comparison, and classification.",
      tags: ["LLM Evaluation"],
    },
    {
      title: "Constrained Generative Policy Optimization (Mixture of Judges)",
      link: "https://ritvik19.medium.com/papers-explained-304-constrained-generative-policy-optimization-mixture-of-judges-71ae4b508b74",
      date: "September 2024",
      description:
        "An LLM post-training paradigm using a Mixture of Judges (MoJ) and cost-efficient constrained policy optimization with stratification to address reward hacking and multi-objective optimization challenges in RLHF. It achieves this by employing rule-based and LLM-based judges to identify and constrain undesirable generation patterns while maximizing calibrated rewards, using tailored optimization strategies for each task in a multi-task setting to avoid conflicting objectives and improve the Pareto frontier.",
      tags: [],
    },
    {
      title: "LongCite",
      link: "https://ritvik19.medium.com/papers-explained-273-longcite-4800340e51d7",
      date: "October 2024",
      description:
        "A system comprising LongBench-Cite benchmark, CoF pipeline for generating cited QA instances, LongCite-45k dataset, and LongCite-8B/9B models trained on this dataset to improve the trustworthiness of long-context LLMs by enabling them to generate responses with fine-grained sentence-level citations.",
      tags: [],
    },
    {
      title: "Thought Preference Optimization",
      link: "https://ritvik19.medium.com/papers-explained-274-thought-preference-optimization-4f365380ae74",
      date: "October 2024",
      description:
        "Iteratively trains LLMs to generate useful 'thoughts' that improve response quality by prompting the model to produce thought-response pairs, scoring the responses with a judge model, creating preference pairs from the highest and lowest-scoring responses and their associated thoughts, and then using these pairs with DPO or IRPO loss to optimize the thought generation process while mitigating judge model length bias through score normalization.",
      tags: [],
    },
    {
      title: "Self-Consistency Preference Optimization",
      link: "https://ritvik19.medium.com/papers-explained-275-self-consistency-preference-optimization-ccd08f5acafb",
      date: "November 2024",
      description:
        "An unsupervised iterative training method for LLMs that leverages the concept of self-consistency to create preference pairs by selecting the most consistent response as the chosen response and the least consistent one as the rejected response, and then optimizes a weighted loss function that prioritizes pairs with larger vote margins, reflecting the model's confidence in the preference.",
      tags: [],
    },
    {
      title: "Hyperfitting",
      link: "",
      date: "December 2024",
      description:
        "Involves fine-tuning a pre-trained LLM on a small dataset until near-zero training loss, significantly improving greedy decoding generation quality despite worsening validation loss. This counter-intuitive process sharpens the model's prediction space, often favoring single tokens, and enhances long-sequence generation even with citation blocking, suggesting the improvement isn't simply memorization.",
      tags: [],
    },
    {
      title: "rStar-Math",
      link: "https://ritvik19.medium.com/papers-explained-290-rstar-math-4b3317a2c2c6",
      date: "January 2025",
      description:
        "Uses a deep thinking approach with Monte Carlo Tree Search and smaller language models to achieve state-of-the-art math reasoning, rivaling or surpassing larger models like OpenAI's. It employs a novel code-augmented CoT data synthesis, a process preference model (PPM) trained with pairwise ranking, and a self-evolution recipe to iteratively improve SLM performance on complex math problems, including Olympiad-level questions.",
      tags: ["LLM for Math"],
    },
    {
      title: "Multiagent Finetuning",
      link: "https://ritvik19.medium.com/papers-explained-292-multiagent-finetuning-a199fc4d8446",
      date: "January 2025",
      description:
        "Improves large language models by training a 'society' of specialized models (generation and critic agents) on data generated through multiagent debate. Generation agents are fine-tuned on their own correct initial responses, while critic agents are fine-tuned on debate sequences showing both initial incorrect and final corrected answers, fostering diversity and enabling iterative self-improvement over multiple rounds.",
      tags: [],
    },
    {
      title: "Critique Fine-Tuning",
      link: "",
      date: "January 2025",
      description:
        "Trains language models to critique noisy responses to questions, rather than simply imitating correct answers, leading to deeper understanding and improved reasoning.",
      tags: [],
    },
    {
      title: "Diverse Preference Optimization",
      link: "",
      date: "January 2025",
      description:
        "Enhances response diversity in language models by selecting preference pairs based on both reward and a diversity criterion. Instead of contrasting the highest and lowest rewarded responses, DivPO contrasts the most diverse response above a reward threshold with the least diverse response below the threshold, promoting a wider range of high-quality outputs.",
      tags: [],
    }, 
    {
      title: "SFT Memorizes, RL Generalizes",
      link: "",
      date: "January 2025",
      description:
        "Investigates the comparative effects of SFT and RL on foundation model generalization and memorization in text and visual tasks (GeneralPoints and V-IRL), finding that RL significantly improves generalization in both rule-based and visual out-of-distribution scenarios while SFT primarily memorizes training data; SFT stabilizes output format for subsequent RL gains, and scaling inference-time compute (verification steps) further improves RL generalization.",
      tags: [],
    }
  ],
  [
    // Parameter Efficient Fine Tuning
    {
      title: "LoRA",
      link: "https://ritvik19.medium.com/papers-explained-lora-a48359cecbfa",
      date: "July 2021",
      description:
        "Introduces trainable rank decomposition matrices into each layer of a pre-trained Transformer model, significantly reducing the number of trainable parameters for downstream tasks.",
        tags: ["Parameter Efficient Fine Tuning"],
    },
    {
      title: "DyLoRA",
      link: "https://ritvik19.medium.com/papers-explained-review-06-parameter-efficient-finetuning-6934fafa74e5#7fb6",
      date: "October 2022",
      description:
        "Allows for flexible rank size by randomly truncating low-rank matrices during training, enabling adaptation to different rank values without retraining.",
        tags: ["Parameter Efficient Fine Tuning"],
    },
    {
      title: "AdaLoRA",
      link: "https://ritvik19.medium.com/papers-explained-review-06-parameter-efficient-finetuning-6934fafa74e5#620f",
      date: "March 2023",
      description:
        "Dynamically allocates a parameter budget based on an importance metric to prune less important singular values during training.",
        tags: ["Parameter Efficient Fine Tuning"],
    },
    {
      title: "QLoRA",
      link: "https://ritvik19.medium.com/papers-explained-146-qlora-a6e7273bc630",
      date: "May 2023",
      description:
        "Allows efficient training of large models on limited GPU memory, through innovations like 4-bit NormalFloat (NF4), double quantization and paged optimisers.",
        tags: ["Parameter Efficient Fine Tuning"],
    },
    {
      title: "LoRA-FA",
      link: "https://ritvik19.medium.com/papers-explained-review-06-parameter-efficient-finetuning-6934fafa74e5#c229",
      date: "August 2023",
      description:
        "Freezes one of the low-rank matrices and only trains a scaling vector for the other, further reducing the number of trainable parameters compared to standard LoRA.",
        tags: ["Parameter Efficient Fine Tuning"],
    },
    {
      title: "Delta-LoRA",
      link: "https://ritvik19.medium.com/papers-explained-review-06-parameter-efficient-finetuning-6934fafa74e5#a4ec",
      date: "September 2023",
      description:
        "Utilizes the delta of the low-rank matrix updates to refine the pre-trained weights directly, removing the Dropout layer for accurate backpropagation.",
        tags: ["Parameter Efficient Fine Tuning"],
    },
    {
      title: "LongLoRA",
      link: "https://ritvik19.medium.com/papers-explained-147-longlora-24f095b93611",
      date: "September 2023",
      description:
        "Enables context extension for large language models, achieving significant computation savings through sparse local attention and parameter-efficient fine-tuning.",
        tags: ["Parameter Efficient Fine Tuning"],
    },
    {
      title: "VeRA",
      link: "https://ritvik19.medium.com/papers-explained-review-06-parameter-efficient-finetuning-6934fafa74e5#5bb3",
      date: "October 2023",
      description:
        "Utilizes frozen, shared random matrices across all layers and trains scaling vectors to adapt those matrices for each layer, reducing the number of trainable parameters compared to LoRA.",
        tags: ["Parameter Efficient Fine Tuning"],
    },
    {
      title: "LoRA+",
      link: "https://ritvik19.medium.com/papers-explained-review-06-parameter-efficient-finetuning-6934fafa74e5#fd31",
      date: "February 2024",
      description:
        "Enhances LoRA by setting different learning rates for the A and B matrices based on a fixed ratio, promoting better feature learning and improved performance.",
        tags: ["Parameter Efficient Fine Tuning"],
    }, 
    {
      title: "MoRA",
      link: "https://ritvik19.medium.com/papers-explained-review-06-parameter-efficient-finetuning-6934fafa74e5#21a4",
      date: "May 2024",
      description:
        "Introduces a square matrix and non-parameterized operators to achieve high-rank updating with the same number of trainable parameters as LoRA, improving knowledge memorization capabilities.",
        tags: ["Parameter Efficient Fine Tuning"],
    },
    {
      title: "DoRA",
      link: "https://ritvik19.medium.com/papers-explained-review-06-parameter-efficient-finetuning-6934fafa74e5#028e",
      date: "May 2024",
      description:
        "Decomposes the high-rank LoRA matrix into multiple single-rank components, allowing dynamic pruning of less important components during training for a more efficient parameter budget allocation.",
        tags: ["Parameter Efficient Fine Tuning"],
    }
  ],
  [
    // Vision Transformers
    {
      title: "Vision Transformer",
      link: "https://ritvik19.medium.com/papers-explained-25-vision-transformers-e286ee8bc06b",
      date: "October 2020",
      description:
        "Images are segmented into patches, which are treated as tokens and a sequence of linear embeddings of these patches are input to a Transformer",
      tags: ["Vision Transformers"],
    },
    {
      title: "Data Efficient Image Transformer (DeiT)",
      link: "https://ritvik19.medium.com/papers-explained-39-deit-3d78dd98c8ec",
      date: "December 2020",
      description:
        "A convolution-free vision transformer that uses a teacher-student strategy with attention-based distillation tokens.",
      tags: ["Vision Transformers"],
    },
    {
      title: "Swin Transformer",
      link: "https://ritvik19.medium.com/papers-explained-26-swin-transformer-39cf88b00e3e",
      date: "March 2021",
      description:
        "A hierarchical vision transformer that uses shifted windows to addresses the challenges of adapting the transformer model to computer vision.",
      tags: ["Vision Transformers"],
    },
    {
      title: "Convolutional Vision Transformer (CvT)",
      link: "https://ritvik19.medium.com/papers-explained-199-cvt-fb4a5c05882e",
      date: "March 2021",
      description:
        "Improves Vision Transformer (ViT) in performance and efficiency by introducing convolutions, to yield the best of both designs.",
      tags: ["Vision Transformers"],
    },
    {
      title: "LeViT",
      link: "https://ritvik19.medium.com/papers-explained-205-levit-89a2defc2d18",
      date: "April 2021",
      description:
        "A hybrid neural network built upon the ViT architecture and DeiT training method, for fast inference image classification.",
      tags: ["Vision Transformers"],
    },
    {
      title: "DINO",
      link: "https://ritvik19.medium.com/papers-explained-249-dino-f7e2c7f438ab",
      date: "April 2021",
      description:
        "Investigates whether self-supervised learning provides new properties to Vision Transformer that stand out compared to convolutional networks and finds that self-supervised ViT features contain explicit information about the semantic segmentation of an image, and are also excellent k-NN classifiers.",
      tags: ["Vision Transformers"],
    },
    {
      title: "BEiT",
      link: "https://ritvik19.medium.com/papers-explained-27-beit-b8c225496c01",
      date: "June 2021",
      description:
        "Utilizes a masked image modeling task inspired by BERT in, involving image patches and visual tokens to pretrain vision Transformers.",
      tags: ["Vision Transformers"],
    },
    {
      title: "MobileViT",
      link: "https://ritvik19.medium.com/papers-explained-40-mobilevit-4793f149c434",
      date: "October 2021",
      description:
        "A lightweight vision transformer designed for mobile devices, effectively combining the strengths of CNNs and ViTs.",
      tags: ["Vision Transformers"],
    },
    {
      title: "Masked AutoEncoder",
      link: "https://ritvik19.medium.com/papers-explained-28-masked-autoencoder-38cb0dbed4af",
      date: "November 2021",
      description:
        "An encoder-decoder architecture that reconstructs input images by masking random patches and leveraging a high proportion of masking for self-supervision.",
      tags: ["Vision Transformers"],
    },
    {
      title: "DINOv2",
      link: "https://ritvik19.medium.com/papers-explained-250-dino-v2-e1e6d12a5c85",
      date: "April 2022",
      description:
        "Demonstrates that existing self-supervised pre-training methods can produce general-purpose visual features by training on curated data from diverse sources, and proposes a new approach that combines techniques to scale pre-training with larger models and datasets.",
      tags: ["Vision Transformers"],
    },
    {
      title: "Multi-Axis Vision Transformer (MaxViT)",
      link: "https://ritvik19.medium.com/papers-explained-210-maxvit-6c68cc515413",
      date: "April 2022",
      description:
        "Introduces multi-axis attention, allowing global-local spatial interactions on arbitrary input resolutions with only linear complexity.",
      tags: ["Vision Transformers"],
    },
    {
      title: "Swin Transformer V2",
      link: "https://ritvik19.medium.com/papers-explained-215-swin-transformer-v2-53bee16ab668",
      date: "April 2022",
      description:
        "A successor to Swin Transformer, addressing challenges like training stability, resolution gaps, and labeled data scarcity.",
      tags: ["Vision Transformers"],
    },
    {
      title: "EfficientFormer",
      link: "https://ritvik19.medium.com/papers-explained-220-efficientformer-97c91540af19",
      date: "June 2022",
      description:
        "Revisits the design principles of ViT and its variants through latency analysis and identifies inefficient designs and operators in ViT to propose a new dimension consistent design paradigm for vision transformers and a simple yet effective latency-driven slimming method to optimize for inference speed.",
      tags: ["Vision Transformers"],
    },
    {
      title: "FastVit",
      link: "https://ritvik19.medium.com/papers-explained-225-fastvit-f1568536ed34",
      date: "March 2023",
      description:
        "A hybrid vision transformer architecture featuring a novel token mixing operator called RepMixer, which significantly improves model efficiency.",
      tags: ["Vision Transformers"],
    },
    {
      title: "Efficient ViT",
      link: "https://ritvik19.medium.com/papers-explained-229-efficient-vit-cc87fbefbe49",
      date: "May 2023",
      description:
        "Employs a single memory-bound MHSA between efficient FFN layers, improves memory efficiency while enhancing channel communication.",
      tags: ["Vision Transformers"],
    },
    {
      title: "Shape-Optimized Vision Transformer (SoViT)",
      link: "https://ritvik19.medium.com/papers-explained-234-sovit-a0ce3c7ef480",
      date: "May 2023",
      description:
        "A shape-optimized vision transformer that achieves competitive results with models twice its size, while being pre-trained with an equivalent amount of compute.",
      tags: ["Vision Transformers"],
    },
    {
      title: "Autoregressive Image Models (AIM)",
      link: "",
      date: "January 2024",
      description:
        "8B Vision models pre-trained using an autoregressive objective, similar to Large Language Models, on 2B images, demonstrating scaling properties utilizing architectural modifications like prefix attention and a parameterized prediction head.",
      tags: ["Vision Transformers", "Autoregressive Image Models"],
    },
    {
      title: "Autoregressive Image Models V2",
      link: "",
      date: "November 2024",
      description:
        "A family of open vision encoders, ranging from 300M to 3B parameters, extending the AIM framework to images and text, pre-trained with a multimodal autoregressive approach, generating both image patches and text tokens using a causal decoder.",
      tags: ["Vision Transformers", "Multimodal Models", "Autoregressive Image Models"],  
    }
  ],
  [
    // Convolutional Neural Networks
    {
      title: "Lenet",
      link: "https://ritvik19.medium.com/papers-explained-review-01-convolutional-neural-networks-78aeff61dcb3#4f26",
      date: "December 1998",
      description: "Introduced Convolutions.",
      tags: ["Convolutional Neural Networks"],
    },
    {
      title: "Alex Net",
      link: "https://ritvik19.medium.com/papers-explained-review-01-convolutional-neural-networks-78aeff61dcb3#f7c6",
      date: "September 2012",
      description:
        "Introduced ReLU activation and Dropout to CNNs. Winner ILSVRC 2012.",
        tags: ["Convolutional Neural Networks"],
    },
    {
      title: "VGG",
      link: "https://ritvik19.medium.com/papers-explained-review-01-convolutional-neural-networks-78aeff61dcb3#c122",
      date: "September 2014",
      description:
        "Used large number of filters of small size in each layer to learn complex features. Achieved SOTA in ILSVRC 2014.",
        tags: ["Convolutional Neural Networks"],
    },
    {
      title: "Inception Net",
      link: "https://ritvik19.medium.com/papers-explained-review-01-convolutional-neural-networks-78aeff61dcb3#d7b3",
      date: "September 2014",
      description:
        "Introduced Inception Modules consisting of multiple parallel convolutional layers, designed to recognize different features at multiple scales.",
        tags: ["Convolutional Neural Networks"],
    },
    {
      title: "Inception Net v2 / Inception Net v3",
      link: "https://ritvik19.medium.com/papers-explained-review-01-convolutional-neural-networks-78aeff61dcb3#d7b3",
      date: "December 2015",
      description:
        "Design Optimizations of the Inception Modules which improved performance and accuracy.",
        tags: ["Convolutional Neural Networks"],
    },
    {
      title: "Res Net",
      link: "https://ritvik19.medium.com/papers-explained-review-01-convolutional-neural-networks-78aeff61dcb3#f761",
      date: "December 2015",
      description:
        "Introduced residual connections, which are shortcuts that bypass one or more layers in the network. Winner ILSVRC 2015.",
        tags: ["Convolutional Neural Networks"],
    },
    {
      title: "Inception Net v4 / Inception ResNet",
      link: "https://ritvik19.medium.com/papers-explained-review-01-convolutional-neural-networks-78aeff61dcb3#83ad",
      date: "February 2016",
      description: "Hybrid approach combining Inception Net and ResNet.",
      tags: ["Convolutional Neural Networks"],
    },
    {
      title: "Dense Net",
      link: "https://ritvik19.medium.com/papers-explained-review-01-convolutional-neural-networks-78aeff61dcb3#65e8",
      date: "August 2016",
      description:
        "Each layer receives input from all the previous layers, creating a dense network of connections between the layers, allowing to learn more diverse features.",
      tags: ["Convolutional Neural Networks"],
    },
    {
      title: "Xception",
      link: "https://ritvik19.medium.com/papers-explained-review-01-convolutional-neural-networks-78aeff61dcb3#bc70",
      date: "October 2016",
      description:
        "Based on InceptionV3 but uses depthwise separable convolutions instead on inception modules.",
      tags: ["Convolutional Neural Networks"],
    },
    {
      title: "Res Next",
      link: "https://ritvik19.medium.com/papers-explained-review-01-convolutional-neural-networks-78aeff61dcb3#90bd",
      date: "November 2016",
      description:
        "Built over ResNet, introduces the concept of grouped convolutions, where the filters in a convolutional layer are divided into multiple groups.",
      tags: ["Convolutional Neural Networks"],
    },
    {
      title: "Mobile Net V1",
      link: "https://ritvik19.medium.com/papers-explained-review-01-convolutional-neural-networks-78aeff61dcb3#3cb5",
      date: "April 2017",
      description:
        "Uses depthwise separable convolutions to reduce the number of parameters and computation required.",
      tags: ["Convolutional Neural Networks"],
    },
    {
      title: "Mobile Net V2",
      link: "https://ritvik19.medium.com/papers-explained-review-01-convolutional-neural-networks-78aeff61dcb3#4440",
      date: "January 2018",
      description:
        "Built upon the MobileNetv1 architecture, uses inverted residuals and linear bottlenecks.",
      tags: ["Convolutional Neural Networks"],
    },
    {
      title: "Mobile Net V3",
      link: "https://ritvik19.medium.com/papers-explained-review-01-convolutional-neural-networks-78aeff61dcb3#8eb6",
      date: "May 2019",
      description:
        "Uses AutoML to find the best possible neural network architecture for a given problem.",
      tags: ["Convolutional Neural Networks"],
    },
    {
      title: "Efficient Net",
      link: "https://ritvik19.medium.com/papers-explained-review-01-convolutional-neural-networks-78aeff61dcb3#560a",
      date: "May 2019",
      description:
        "Uses a compound scaling method to scale the network's depth, width, and resolution to achieve a high accuracy with a relatively low computational cost.",
      tags: ["Convolutional Neural Networks"],
    },
    {
      title: "NF Net",
      link: "https://ritvik19.medium.com/papers-explained-84-nf-net-b8efa03d6b26",
      date: "February 2021",
      description:
        "An improved class of Normalizer-Free ResNets that implement batch-normalized networks, offer faster training times, and introduce an adaptive gradient clipping technique to overcome instabilities associated with deep ResNets.",
      tags: ["Convolutional Neural Networks"],
    },
    {
      title: "Conv Mixer",
      link: "https://ritvik19.medium.com/papers-explained-29-convmixer-f073f0356526",
      date: "January 2022",
      description:
        "Processes image patches using standard convolutions for mixing spatial and channel dimensions.",
      tags: ["Convolutional Neural Networks"],
    },
    {
      title: "ConvNeXt",
      link: "https://ritvik19.medium.com/papers-explained-92-convnext-d13385d9177d",
      date: "January 2022",
      description:
        "A pure ConvNet model, evolved from standard ResNet design, that competes well with Transformers in accuracy and scalability.",
      tags: ["Convolutional Neural Networks"],
    },
    {
      title: "ConvNeXt V2",
      link: "https://ritvik19.medium.com/papers-explained-94-convnext-v2-2ecdabf2081c",
      date: "January 2023",
      description:
        "Incorporates a fully convolutional MAE framework and a Global Response Normalization (GRN) layer, boosting performance across multiple benchmarks.",
      tags: ["Convolutional Neural Networks"],
    },
    {
      title: "Efficient Net V2",
      link: "https://ritvik19.medium.com/papers-explained-efficientnetv2-a7a1e4113b89",
      date: "April 2024",
      description:
        "A new family of convolutional networks, achieves faster training speed and better parameter efficiency than previous models through neural architecture search and scaling, with progressive learning allowing for improved accuracy on various datasets while training up to 11x faster.",
    },
    {
      title: "Mobile Net V4",
      link: "https://ritvik19.medium.com/papers-explained-232-mobilenetv4-83a526887c30",
      date: "April 2024",
      description: 
        "Features a universally efficient architecture design, including the Universal Inverted Bottleneck (UIB) search block, Mobile MQA attention block, and an optimized neural architecture search recipe, which enables it to achieve high accuracy and efficiency on various mobile devices and accelerators.",
      tags: ["Convolutional Neural Networks"],
    },
  ],
  [
    // Object Detection
    {
      title: "SSD",
      link: "https://ritvik19.medium.com/papers-explained-31-single-shot-multibox-detector-14b0aa2f5a97",
      date: "December 2015",
      description:
        "Discretizes bounding box outputs over a span of various scales and aspect ratios per feature map.",
      tags: ["Object Detection"],
    },
    {
      title: "Feature Pyramid Network",
      link: "https://ritvik19.medium.com/papers-explained-21-feature-pyramid-network-6baebcb7e4b8",
      date: "December 2016",
      description:
        "Leverages the inherent multi-scale hierarchy of deep convolutional networks to efficiently construct feature pyramids.",
      tags: ["Object Detection"],
    },
    {
      title: "Focal Loss",
      link: "https://ritvik19.medium.com/papers-explained-22-focal-loss-for-dense-object-detection-retinanet-733b70ce0cb1",
      date: "August 2017",
      description:
        "Addresses class imbalance in dense object detectors by down-weighting the loss assigned to well-classified examples.",
      tags: ["Object Detection"],
    },
    {
      title: "DETR",
      link: "https://ritvik19.medium.com/papers-explained-79-detr-bcdd53355d9f",
      date: "May 2020",
      description:
        "A novel transformers based object detection model that treats object detection as a set prediction problem, eliminating the need for hand-designed components.",
      tags: ["Object Detection", "Vision Transformers"],
    },
    {
      title: "OWL ViT",
      link: "https://ritvik19.medium.com/papers-explained-237-owl-vit-ea58a142de68",
      date: "May 2022",
      description:
        "Employs Vision Transformers, CLIP-based contrastive pre-training, and bipartite matching loss for open-vocabulary detection, utilizing image-level pre-training, multihead attention pooling, and mosaic image augmentation.",
      tags: ["Object Detection", "Vision Transformers"],
    },
    {
      title: "Segment Anything Model (SAM)",
      link: "https://ritvik19.medium.com/papers-explained-238-segment-anything-model-b3960b569fce",
      date: "April 2023",
      description:
        "Introduces a novel image segmentation task, model, and dataset, aiming to enable prompt-able, zero-shot transfer learning in computer vision.",
      tags: ["Object Detection", "Segmentation"],
    },
    {
      title: "Segment Anything Model (SAM) 2",
      link: "https://ritvik19.medium.com/papers-explained-239-sam-2-6ffb7f187281",
      date: "July 2024",
      description:
        "A foundation model towards solving promptable visual segmentation in images and videos based on a simple transformer architecture with streaming memory for real-time video processing.",
      tags: ["Object Detection", "Segmentation"],
    }
  ],
  [
    // Region Based Convolutional Neural Networks
    {
      title: "RCNN",
      link: "https://ritvik19.medium.com/papers-explained-14-rcnn-ede4db2de0ab",
      date: "November 2013",
      description:
        "Uses selective search for region proposals, CNNs for feature extraction, SVM for classification followed by box offset regression.",
      tags: ["Object Detection", "Convolutional Neural Networks"],
    },
    {
      title: "Fast RCNN",
      link: "https://ritvik19.medium.com/papers-explained-15-fast-rcnn-28c1792dcee0",
      date: "April 2015",
      description:
        "Processes entire image through CNN, employs RoI Pooling to extract feature vectors from ROIs, followed by classification and BBox regression.",
      tags: ["Object Detection", "Convolutional Neural Networks"],
    },
    {
      title: "Faster RCNN",
      link: "https://ritvik19.medium.com/papers-explained-16-faster-rcnn-a7b874ffacd9",
      date: "June 2015",
      description:
        "A region proposal network (RPN) and a Fast R-CNN detector, collaboratively predict object regions by sharing convolutional features.",
      tags: ["Object Detection", "Convolutional Neural Networks"],
    },
    {
      title: "Mask RCNN",
      link: "https://ritvik19.medium.com/papers-explained-17-mask-rcnn-82c64bea5261",
      date: "March 2017",
      description:
        "Extends Faster R-CNN to solve instance segmentation tasks, by adding a branch for predicting an object mask in parallel with the existing branch.",
      tags: ["Object Detection", "Convolutional Neural Networks"],
    },
    {
      title: "Cascade RCNN",
      link: "https://ritvik19.medium.com/papers-explained-77-cascade-rcnn-720b161d86e4",
      date: "December 2017",
      description:
        "Proposes a multi-stage approach where detectors are trained with progressively higher IoU thresholds, improving selectivity against false positives.",
      tags: ["Object Detection", "Convolutional Neural Networks"],
    },
  ],
  [
    // Document Understanding
    {
      title: "Table Net",
      link: "https://ritvik19.medium.com/papers-explained-18-tablenet-3d4c62269bb3",
      date: "January 2020",
      description:
        "An end-to-end deep learning model designed for both table detection and structure recognition.",
      tags: ["Document Understanding"],
    },
    {
      title: "SPADE",
      link: "https://ritvik19.medium.com/f564ce612501",
      date: "May 2020",
      description:
        "Formulates Information Extraction (IE) as a spatial dependency parsing problem.",
      tags: ["Document Understanding"],
    },
    {
      title: "Layout Parser",
      link: "https://ritvik19.medium.com/papers-explained-245-layout-parser-d29bb291890c",
      date: "March 2021",
      description:
        "A library integrating Detectron2, CNN-RNN OCR, layout structures, TensorFlow/PyTorch, and a Model Zoo. The toolkit features Tesseract, Google Cloud Vision for OCR, active learning tools and community platform ensures efficiency and adaptability.",
      tags: ["Document Understanding", "Object Detection"],
    },
    {
      title: "Layout Reader",
      link: "https://ritvik19.medium.com/papers-explained-247-layout-reader-248b27db1234",
      date: "August 2021",
      description:
        "A seq2seq model that accurately predicts reading order, text, and layout information from document images.",
      tags: ["Document Understanding"],
    },
    {
      title: "Donut",
      link: "https://ritvik19.medium.com/papers-explained-20-donut-cb1523bf3281",
      date: "November 2021",
      description:
        "An OCR-free Encoder-Decoder Transformer model. The encoder takes in images, decoder takes in prompts & encoded images to generate the required text.",
      tags: ["Document Understanding", "Multimodal Models"],
    },
    {
      title: "DiT",
      link: "https://ritvik19.medium.com/papers-explained-19-dit-b6d6eccd8c4e",
      date: "March 2022",
      description:
        "An Image Transformer pre-trained (self-supervised) on document images",
      tags: ["Document Understanding", "Vision Transformers"],
    },
    {
      title: "Pix2Struct",
      link: "https://ritvik19.medium.com/papers-explained-254-pix2struct-6adc95a01586",
      date: "October 2022",
      description:
        "A pretrained image-to-text model designed for visual language understanding, particularly in tasks involving visually-situated language.",
      tags: ["Document Understanding", "Vision Transformers", "Multimodal Models"],
    },
    {
      title: "Matcha",
      link: "https://ritvik19.medium.com/papers-explained-255-matcha-d5d5fe66b039",
      date: "December 2022",
      description:
        "Leverages Pix2Struct, and introduces pretraining tasks focused on math reasoning and chart derendering to improve chart and plot comprehension, enhancing understanding in diverse visual language tasks.",
      tags: ["Document Understanding", "Vision Transformers", "Multimodal Models"],
    },
    {
      title: "DePlot",
      link: "https://ritvik19.medium.com/papers-explained-256-deplot-3e8a02eefc94",
      date: "December 2022",
      description: 
        "Built upon MatCha, standardises plot to table task, translating plots into linearized tables (markdown) for processing by LLMs.",
      tags: ["Document Understanding", "Vision Transformers", "Multimodal Models"],
    },
    {
      title: "UDoP",
      link: "https://ritvik19.medium.com/papers-explained-42-udop-719732358ab4",
      date: "December 2022",
      description:
        "Integrates text, image, and layout information through a Vision-Text-Layout Transformer, enabling unified representation.",
      tags: ["Document Understanding", "Multimodal Models"],
    },
    {
      title: "GeoLayoutLM",
      link: "https://ritvik19.medium.com/papers-explained-258-geolayoutlm-f581eec8c8a2",
      date: "April 2023",
      description:
        "Explicitly models geometric relations in pre-training and enhances feature representation.",
      tags: ["Document Understanding", "Language Models"],
    },
    {
      title: "Nougat",
      link: "https://ritvik19.medium.com/papers-explained-257-nougat-bb304f7af0a3",
      date: "August 2023",
      description:
        "A Visual Transformer model that performs an Optical Character Recognition (OCR) task for processing scientific documents into a markup language.",
      tags: ["Document Understanding", "Vision Transformers", "Multimodal Models"],
    },
    {
      title: "LMDX",
      link: "https://ritvik19.medium.com/papers-explained-248-lmdx-854c0bc771f0",
      date: "September 2023",
      description:
        "A methodology to adapt arbitrary LLMs for document information extraction.",
      tags: ["Document Understanding", "Language Models"],
    },
    {
      title: "DocLLM",
      link: "https://ritvik19.medium.com/papers-explained-87-docllm-93c188edfaef",
      date: "January 2024",
      description:
        "A lightweight extension to traditional LLMs that focuses on reasoning over visual documents, by incorporating textual semantics and spatial layout without expensive image encoders.",
      tags: ["Document Understanding", "Transformer Decoder"],
    },
    {
      title: "Spreadsheet LLM",
      link: "https://ritvik19.medium.com/papers-explained-271-spreadsheet-llm-25b9d70f06e3",
      date: "July 2024",
      description:
        "An efficient encoding method that utilizes SheetCompressor, a framework comprising structural anchor based compression, inverse index translation, and data format aware aggregation, to effectively compress spreadsheets for LLMs, and Chain of Spreadsheet for spreadsheet understanding and spreadsheet QA task.",
      tags: ["Document Understanding", "Transformer Decoder"],
    }
  ],
  [
    // Layout Aware Language Models
    {
      title: "Layout LM",
      link: "https://ritvik19.medium.com/papers-explained-10-layout-lm-32ec4bad6406",
      date: "December 2019",
      description:
        "Utilises BERT as the backbone, adds two new input embeddings: 2-D position embedding and image embedding (Only for downstream tasks).",
      tags: ["Layout Aware Language Models"],
    },
    {
      title: "LamBERT",
      link: "https://ritvik19.medium.com/papers-explained-41-lambert-8f52d28f20d9",
      date: "February 2020",
      description:
        "Utilises RoBERTa as the backbone and adds Layout embeddings along with relative bias.",
        tags: ["Layout Aware Language Models"],
    },
    {
      title: "Layout LM v2",
      link: "https://ritvik19.medium.com/papers-explained-11-layout-lm-v2-9531a983e659",
      date: "December 2020",
      description:
        "Uses a multi-modal Transformer model, to integrate text, layout, and image in the pre-training stage, to learn end-to-end cross-modal interaction.",
        tags: ["Layout Aware Language Models"],
    },
    {
      title: "Structural LM",
      link: "https://ritvik19.medium.com/papers-explained-23-structural-lm-36e9df91e7c1",
      date: "May 2021",
      description:
        "Utilises BERT as the backbone and feeds text, 1D and (2D cell level) embeddings to the transformer model.",
        tags: ["Layout Aware Language Models"],
    },
    {
      title: "Doc Former",
      link: "https://ritvik19.medium.com/papers-explained-30-docformer-228ce27182a0",
      date: "June 2021",
      description:
        "Encoder-only transformer with a CNN backbone for visual feature extraction, combines text, vision, and spatial features through a multi-modal self-attention layer.",
        tags: ["Layout Aware Language Models"],
    },
    {
      title: "BROS",
      link: "https://ritvik19.medium.com/papers-explained-246-bros-1f1127476f73",
      date: "August 2021",
      description:
        "Built upon BERT, encodes relative positions of texts in 2D space and learns from unlabeled documents with area masking strategy.",
      tags: ["Layout Aware Language Models"],
    },
    {
      title: "LiLT",
      link: "https://ritvik19.medium.com/papers-explained-12-lilt-701057ec6d9e",
      date: "February 2022",
      description:
        "Introduced Bi-directional attention complementation mechanism (BiACM) to accomplish the cross-modal interaction of text and layout.",
        tags: ["Layout Aware Language Models"],
    },
    {
      title: "Layout LM V3",
      link: "https://ritvik19.medium.com/papers-explained-13-layout-lm-v3-3b54910173aa",
      date: "April 2022",
      description:
        "A unified text-image multimodal Transformer to learn cross-modal representations, that imputs concatenation of text embedding and image embedding.",
        tags: ["Layout Aware Language Models"],
    },
    {
      title: "ERNIE Layout",
      link: "https://ritvik19.medium.com/papers-explained-24-ernie-layout-47a5a38e321b",
      date: "October 2022",
      description:
        "Reorganizes tokens using layout information, combines text and visual embeddings, utilizes multi-modal transformers with spatial aware disentangled attention.",
        tags: ["Layout Aware Language Models"],
    },
  ],
  [
    // Generative Adversarial Networks
    {
      title: "Generative Adversarial Networks",
      link: "https://ritvik19.medium.com/papers-explained-review-05-generative-adversarial-networks-bbb51b160d5e#7041",
      date: "June 2014",
      description:
        "Introduces a framework where, a generative and a discriminative model, are trained simultaneously in a minimax game.",
      tags: ["Generative Adversarial Networks"],

    },
    {
      title: "Conditional Generative Adversarial Networks",
      link: "https://ritvik19.medium.com/papers-explained-review-05-generative-adversarial-networks-bbb51b160d5e#86aa",
      date: "November 2014",
      description:
        "A method for training GANs, enabling the generation based on specific conditions, by feeding them to both the generator and discriminator networks.",
      tags: ["Generative Adversarial Networks"],
    },
    {
      title: "Deep Convolutional Generative Adversarial Networks",
      link: "https://ritvik19.medium.com/papers-explained-review-05-generative-adversarial-networks-bbb51b160d5e#fe42",
      date: "November 2015",
      description:
        "Demonstrates the ability of CNNs for unsupervised learning using specific architectural constraints designed.",
      tags: ["Generative Adversarial Networks"],
    },
    {
      title: "Improved GAN",
      link: "https://ritvik19.medium.com/papers-explained-review-05-generative-adversarial-networks-bbb51b160d5e#9a55",
      date: "June 2016",
      description:
        "Presents a variety of new architectural features and training procedures that can be applied to the generative adversarial networks (GANs) framework.",
      tags: ["Generative Adversarial Networks"],
    },
    {
      title: "Wasserstein Generative Adversarial Networks",
      link: "https://ritvik19.medium.com/papers-explained-review-05-generative-adversarial-networks-bbb51b160d5e#6f8f",
      date: "January 2017",
      description:
        "An alternative GAN training algorithm that enhances learning stability, mitigates issues like mode collapse.",
      tags: ["Generative Adversarial Networks"],
    },
    {
      title: "Cycle GAN",
      link: "https://ritvik19.medium.com/papers-explained-review-05-generative-adversarial-networks-bbb51b160d5e#7f8b",
      date: "March 2017",
      description:
        "An approach for learning to translate an image from a source domain X to a target domain Y in the absence of paired examples by leveraging adversarial losses and cycle consistency constraints, using two GANs",
      tags: ["Generative Adversarial Networks"],
    },
  ],
  [
    // Tabular Data
    {
      title: "Entity Embeddings",
      link: "https://ritvik19.medium.com/papers-explained-review-04-tabular-deep-learning-776db04f965b#932e",
      date: "April 2016",
      description:
        "Maps categorical variables into continuous vector spaces through neural network learning, revealing intrinsic properties.",
      tags: ["Tabular Data"],
    },
    {
      title: "Wide and Deep Learning",
      link: "https://ritvik19.medium.com/papers-explained-review-04-tabular-deep-learning-776db04f965b#bfdc",
      date: "June 2016",
      description:
        "Combines memorization of specific patterns with generalization of similarities.",
      tags: ["Tabular Data"],
    },
    {
      title: "Deep and Cross Network",
      link: "https://ritvik19.medium.com/papers-explained-review-04-tabular-deep-learning-776db04f965b#0017",
      date: "August 2017",
      description:
        "Combines the  a novel cross network with deep neural networks (DNNs) to efficiently learn feature interactions without manual feature engineering.",
      tags: ["Tabular Data"],
    },
    {
      title: "Tab Transformer",
      link: "https://ritvik19.medium.com/papers-explained-review-04-tabular-deep-learning-776db04f965b#48c4",
      date: "December 2020",
      description:
        "Employs multi-head attention-based Transformer layers to convert categorical feature embeddings into robust contextual embeddings.",
      tags: ["Tabular Data"],
    },
    {
      title: "Tabular ResNet",
      link: "https://ritvik19.medium.com/papers-explained-review-04-tabular-deep-learning-776db04f965b#46af",
      date: "June 2021",
      description: "An MLP with skip connections.",
      tags: ["Tabular Data"],
    },
    {
      title: "Feature Tokenizer Transformer",
      link: "https://ritvik19.medium.com/papers-explained-review-04-tabular-deep-learning-776db04f965b#1ab8",
      date: "June 2021",
      description:
        "Transforms all features (categorical and numerical) to embeddings and applies a stack of Transformer layers to the embeddings.",
      tags: ["Tabular Data"],
    },
  ],
  [
    // Datasets
    {
      title: "Obelics",
      link: "https://ritvik19.medium.com/papers-explained-179-obelics-idefics-a581f8d909b6",
      date: "June 2023",
      description:
        "An open web-scale filtered dataset of interleaved image-text documents comprising 141M web pages, 353M associated images, and 115B text tokens, extracted from CommonCrawl.",
      tags: ["Datasets", "Multimodal Datasets", "HuggingFace"],
    },
    {
      title: "Dolma",
      link: "https://ritvik19.medium.com/papers-explained-97-dolma-a656169269cb",
      date: "January 2024",
      description:
        "An open corpus of three trillion tokens designed to support language model pretraining research.",
      tags : ["Datasets", "Language Model Datasets", "Olmo"],
    },
    {
      title: "Aya Dataset",
      link: "https://ritvik19.medium.com/papers-explained-108-aya-dataset-9e299ac74a19",
      date: "February 2024",
      description:
        "A human-curated instruction-following dataset that spans 65 languages, created to bridge the language gap in datasets for natural language processing.",
      tags: ["Datasets", "Multilingual Datasets", "Cohere"],
    },
    {
      title: "WebSight",
      link: "https://ritvik19.medium.com/papers-explained-177-websight-2905d0e14233",
      date: "March 2024",
      description:
        "A synthetic dataset consisting of 2M pairs of HTML codes and their corresponding screenshots, generated through LLMs, aimed to accelerate research for converting a screenshot into a corresponding HTML. ",
      tags: ["Datasets", "Multimodal Datasets", "HuggingFace"],
    },
    {
      title: "Cosmopedia",
      link: "https://ritvik19.medium.com/papers-explained-175-cosmopedia-5f7e81c76d14",
      date: "March 2024",
      description:
        "Synthetic Data containing over 30M files and 25B tokens, generated by Mixtral-8x7B-Instruct-v0., aimed to reproduce the training data for Phi-1.5.",
      tags: ["Datasets", "Language Model Datasets", "HuggingFace", "Synthetic Data"],
    },
    {
      title: "RewardBench",
      link: "A benchmark dataset and code-base designed to evaluate reward models used in RLHF",
      date: "March 2024",
      description:
        "A benchmark dataset and code-base designed to evaluate reward models used in RLHF.",
      tags: ["Datasets", "LLM Evaluation"],
    },
    {
      title: "Fine Web",
      link: "https://ritvik19.medium.com/papers-explained-174-fineweb-280bbc08068b",
      date: "May 2024",
      description:
        "A large-scale dataset for pretraining LLMs, consisting of 15T tokens, shown to produce better-performing models than other open pretraining datasets.",
      tags: ["Datasets", "Language Model Datasets", "HuggingFace"],
    },
    {
      title: "Cosmopedia v2",
      link: "https://ritvik19.medium.com/papers-explained-175-cosmopedia-5f7e81c76d14#5bab",
      date: "July 2024",
      description:
        "An enhanced version of Cosmopedia, with a lot of emphasis on prompt optimization.",
      tags: ["Datasets", "Language Model Datasets", "HuggingFace", "Synthetic Data"],
    },
    {
      title: "Docmatix",
      link: "https://ritvik19.medium.com/papers-explained-178-docmatix-9f2731ff1654",
      date: "July 2024",
      description:
        "A massive dataset for DocVQA containing 2.4M images, 9.5M question-answer pairs, and 1.3M PDF documents, generated by taking transcriptions from the PDFA OCR dataset and using a Phi-3-small model to generate Q/A pairs. ",
      tags: ["Datasets", "Document Understanding", "HuggingFace"],
    },
    {
      title: "PixMo",
      link: "https://ritvik19.medium.com/papers-explained-241-pixmo-and-molmo-239d70abebff",
      date: "September 2024",
      description:
        "A high-quality dataset of detailed image descriptions collected through speech-based annotations, enabling the creation of more robust and accurate VLMs.",
      tags: ["Datasets", "Multimodal Datasets"],
    },
    {
      title: "Smol Talk",
      link: "https://ritvik19.medium.com/papers-explained-176-smol-lm-a166d5f1facc#b5e3",
      date: "November 2024",
      description:
        "A synthetic instruction-following dataset comprising 1 million samples, built using a fine-tuned LLM on a diverse range of instruction-following datasets and then generating synthetic conversations using various prompts and instructions to improve instruction following, chat, and reasoning capabilities.",
      tags: ["Datasets", "Synthetic Data", "HuggingFace"],
    },
    {
      title: "Red Pajama V1",
      link: "https://ritvik19.medium.com/papers-explained-299-red-pajama-4aced4a3ff72",
      date: "November 2024",
      description:
        "A reproduction of the LLaMA training dataset, built from seven sources (CommonCrawl, C4, GitHub, Wikipedia, Books, ArXiv, and Stack Exchange) totaling 1.2 trillion tokens. The reproduction process involved addressing gaps and ambiguities in the original LLaMA documentation, with some differences in data processing choices.",
      tags: ["Datasets"],
    },
    {
      title: "Red Pajama V2",
      link: "https://ritvik19.medium.com/papers-explained-299-red-pajama-4aced4a3ff72#9376",
      date: "November 2024",
      description:
        "A massive, unfiltered web-based dataset derived from CommonCrawl, comprising over 100 trillion tokens in multiple languages. It includes various quality signals (natural language metrics, repetitiveness, content flags, ML heuristics, and deduplication data) as metadata, enabling flexible filtering and dataset creation for diverse downstream tasks.",
      tags: ["Datasets"],
    }
  ],
  [
    // Neural Network Layers
    {
      title: "Convolution Layer",
      link: "https://ritvik19.medium.com/papers-explained-review-07-convolution-layers-c083e7410cd3#4176",
      date: "",
      description: "",
      tags: ["Neural Network Layers", "Convolution"],
    },
    {
      title: "Pointwise Convolution",
      link: "https://ritvik19.medium.com/papers-explained-review-07-convolution-layers-c083e7410cd3#8f24",
      date: "",
      description: "",
      tags: ["Neural Network Layers", "Convolution"],
    },
    {
      title: "Depthwise Convolution",
      link: "https://ritvik19.medium.com/papers-explained-review-07-convolution-layers-c083e7410cd3#20e4",
      date: "",
      description: "",
      tags: ["Neural Network Layers", "Convolution"],
    },
    {
      title: "Separable Convolution",
      link: "https://ritvik19.medium.com/papers-explained-review-07-convolution-layers-c083e7410cd3#539f",
      date: "",
      description: "",
      tags: ["Neural Network Layers", "Convolution"],
    },
    {
      title: "Convolution Transpose",
      link: "https://ritvik19.medium.com/papers-explained-review-07-convolution-layers-c083e7410cd3#a302",
      date: "",
      description: "",
      tags: ["Neural Network Layers", "Convolution"],
    },
    {
      title: "Simple Recurrent",
      link: "https://ritvik19.medium.com/papers-explained-review-08-recurrent-layers-ff2f224af059#e405",
      date: "",
      description: "",
      tags: ["Neural Network Layers", "Recurrent"],
    },
    {
      title: "LSTM",
      link: "https://ritvik19.medium.com/papers-explained-review-08-recurrent-layers-ff2f224af059#0947",
      date: "",
      description: "",
      tags: ["Neural Network Layers", "Recurrent"],
    },
    {
      title: "GRU",
      link: "https://ritvik19.medium.com/papers-explained-review-08-recurrent-layers-ff2f224af059#4571",
      date: "",
      description: "",
      tags: ["Neural Network Layers", "Recurrent"],
    },
    {
      title: "Scaled Dot Product Attention",
      link: "https://ritvik19.medium.com/papers-explained-review-09-attention-layers-beeef323e7f5#c18c",
      date: "",
      description: "",
      tags: ["Neural Network Layers", "Attention"],
    },
    {
      title: "Multi Head Attention",
      link: "https://ritvik19.medium.com/papers-explained-review-09-attention-layers-beeef323e7f5#be63",
      date: "",
      description: "",
      tags: ["Neural Network Layers", "Attention"],
    },
    {
      title: "Cross Attention",
      link: "https://ritvik19.medium.com/papers-explained-review-09-attention-layers-beeef323e7f5#0f28",
      date: "",
      description: "",
      tags: ["Neural Network Layers", "Attention"],
    },
    {
      title: "Causal Attention",
      link: "https://ritvik19.medium.com/papers-explained-review-09-attention-layers-beeef323e7f5#14c7",
      date: "",
      description: "",
      tags: ["Neural Network Layers", "Attention"],
    },
    {
      title: "Sliding Window Attention",
      link: "https://ritvik19.medium.com/papers-explained-review-09-attention-layers-beeef323e7f5#324c",
      date: "",
      description: "",
      tags: ["Neural Network Layers", "Attention"],
    },
    {
      title: "Multi Query Attention",
      link: "https://ritvik19.medium.com/papers-explained-review-09-attention-layers-beeef323e7f5#0bfd",
      date: "",
      description: "",
      tags: ["Neural Network Layers", "Attention"],
    },
    {
      title: "Grouped Query Attention",
      link: "https://ritvik19.medium.com/papers-explained-review-09-attention-layers-beeef323e7f5#d5fb",
      date: "",
      description: "",
      tags: ["Neural Network Layers", "Attention"],
    },
    {
      title: "Batch Normalisation",
      link: "https://ritvik19.medium.com/papers-explained-review-10-normalization-layers-56b556c9646e#00ea",
      date: "",
      description: "",
      tags: ["Neural Network Layers", "Normalization"],
    },
    {
      title: "Layer Normalisation",
      link: "https://ritvik19.medium.com/papers-explained-review-10-normalization-layers-56b556c9646e#9439",
      date: "",
      description: "",
      tags: ["Neural Network Layers", "Normalization"],
    },
    {
      title: "Instance Normalisation",
      link: "https://ritvik19.medium.com/papers-explained-review-10-normalization-layers-56b556c9646e#7783",
      date: "",
      description: "",
      tags: ["Neural Network Layers", "Normalization"],
    },
    {
      title: "Group Normalisation",
      link: "https://ritvik19.medium.com/papers-explained-review-10-normalization-layers-56b556c9646e#cd7f",
      date: "",
      description: "",
      tags: ["Neural Network Layers", "Normalization"],
    },
    {
      title: "Weight Standardisation",
      link: "https://ritvik19.medium.com/papers-explained-review-10-normalization-layers-56b556c9646e#3944",
      date: "",
      description: "",
      tags: ["Neural Network Layers", "Normalization"],
    },
    {
      title: "Batch Channel Normalisation",
      link: "https://ritvik19.medium.com/papers-explained-review-10-normalization-layers-56b556c9646e#3944",
      date: "",
      description: "",
      tags: ["Neural Network Layers", "Normalization"],
    }
  ],
  [
    // Autoencoders
    {
      title: "Auto Encoders",
      link: "https://ritvik19.medium.com/papers-explained-review-11-auto-encoders-3b8f08b4eac0#b8a0",
      date: "",
      description: "",
      tags: ["Autoencoders"],
    },
    {
      title: "Sparse Auto Encoders",
      link: "https://ritvik19.medium.com/papers-explained-review-11-auto-encoders-3b8f08b4eac0#f605",
      date: "",
      description: "",
      tags: ["Autoencoders"],
    },
    {
      title: "K Sparse Auto Encoders",
      link: "https://ritvik19.medium.com/papers-explained-review-11-auto-encoders-3b8f08b4eac0#23b0",
      date: "",
      description: "",
      tags: ["Autoencoders"],
    },
    {
      title: "Contractive Auto Encoders",
      link: "https://ritvik19.medium.com/papers-explained-review-11-auto-encoders-3b8f08b4eac0#23b3",
      date: "",
      description: "",
      tags: ["Autoencoders"],
    },
    {
      title: "Convolutional Auto Encoders",
      link: "https://ritvik19.medium.com/papers-explained-review-11-auto-encoders-3b8f08b4eac0#59a1",
      date: "",
      description: "",
      tags: ["Autoencoders"],
    },
    {
      title: "Sequence to Sequence Auto Encoders",
      link: "https://ritvik19.medium.com/papers-explained-review-11-auto-encoders-3b8f08b4eac0#73e0",
      date: "",
      description: "",
      tags: ["Autoencoders"],
    },
    {
      title: "Denoising Auto Encoders",
      link: "https://ritvik19.medium.com/papers-explained-review-11-auto-encoders-3b8f08b4eac0#1829",
      date: "",
      description: "",
      tags: ["Autoencoders"],
    },
    {
      title: "Variational Auto Encoders",
      link: "https://ritvik19.medium.com/papers-explained-review-11-auto-encoders-3b8f08b4eac0#a626",
      date: "",
      description: "",
      tags: ["Autoencoders"],
    },
    {
      title: "Masked Auto Encoders",
      link: "https://ritvik19.medium.com/papers-explained-review-11-auto-encoders-3b8f08b4eac0#2247",
      date: "",
      description: "",
      tags: ["Autoencoders"],
    }
  ],
  [
    // Miscellaneous Studies
    {
      title: "TLDR",
      link: "https://ritvik19.medium.com/papers-explained-293-tldr-a31d787cd365",
      date: "April 2020",
      description:
        "Extreme summarization for scientific papers, creating concise, single-sentence summaries of key contributions.",
      tags: [],
    },
    {
      title: "ColD Fusion",
      link: "https://ritvik19.medium.com/papers-explained-32-cold-fusion-452f33101a91",
      date: "December 2022",
      description:
        "A method enabling the benefits of multitask learning through distributed computation without data sharing and improving model performance.",
      tags: [],
    },
    
    {
      title: "Are Emergent Abilities of Large Language Models a Mirage?",
      link: "https://ritvik19.medium.com/papers-explained-are-emergent-abilities-of-large-language-models-a-mirage-4160cf0e44cb",
      date: "April 2023",
      description:
        "Presents an alternative explanation for emergent abilities, i.e. emergent abilities are created by the researcher’s choice of metrics, not fundamental changes in model family behaviour on specific tasks with scale.",
      tags: [],
    },
    {
      title: "Scaling Data-Constrained Language Models",
      link: "https://ritvik19.medium.com/papers-explained-85-scaling-data-constrained-language-models-2a4c18bcc7d3",
      date: "May 2023",
      description:
        "This study investigates scaling language models in data-constrained regimes.",
      tags: [],
    },
    {
      title: "Multiagent Debate",
      link: "https://ritvik19.medium.com/papers-explained-291-multiagent-debate-1a1693d5fa5e",
      date: "May 2023",
      description:
        "Involves multiple language model instances independently generating answers to a query, then iteratively critiquing and updating their responses based on others' answers over multiple rounds, ultimately converging on a single, refined answer. This process mimics multi-threaded reasoning and multi-source fact-checking, leveraging the collective intelligence of the models to improve factual accuracy and reasoning capabilities.",
      tags: ["Agentic Systems"],
    },
    {
      title: "RAGAS",
      link: "https://ritvik19.medium.com/papers-explained-227-ragas-4594fc4d96b9",
      date: "September 2023",
      description:
        "A framework for reference-free evaluation of RAG systems, assessing the retrieval system's ability to find relevant context, the LLM's faithfulness in using that context, and the overall quality of the generated response.",
      tags: ["LLM Evaluation"],
    },
    {
      title: "DSPy",
      link: "https://ritvik19.medium.com/papers-explained-135-dspy-fe8af7e35091",
      date: "October 2023",
      description:
        "A programming model that abstracts LM pipelines as text transformation graphs, i.e. imperative computation graphs where LMs are invoked through declarative modules, optimizing their use through a structured framework of signatures, modules, and teleprompters to automate and enhance text transformation tasks.",
      tags: ["Prompt Optimization"],
    },
    {
      title: "LLMLingua",
      link: "https://ritvik19.medium.com/papers-explained-136-llmlingua-f9b2f53f5f9b",
      date: "October 2023",
      description:
        "A novel coarse-to-fine prompt compression method, incorporating a budget controller, an iterative token-level compression algorithm, and distribution alignment, achieving up to 20x compression with minimal performance loss.",
        tags: ["Prompt Compression"],
    },
    {
      title: "LongLLMLingua",
      link: "https://ritvik19.medium.com/papers-explained-137-longllmlingua-45961fa703dd",
      date: "October 2023",
      description:
        "A novel approach for prompt compression to enhance performance in long context scenarios using question-aware compression and document reordering.",
        tags: ["Prompt Compression"],
    },
    {
      title: "Prometheus",
      link: "https://ritvik19.medium.com/papers-explained-170-prometheus-5e72b8054729",
      date: "October 2023",
      description:
        "A 13B fully open source evaluation LLM trained on Feedback Collection curated using GPT-4 (in this work).",
        tags: ["LLM Evaluation"],
    },
    {
      title: "An In-depth Look at Gemini's Language Abilities",
      link: "https://ritvik19.medium.com/papers-explained-81-an-in-depth-look-at-geminis-language-abilities-540ca9046d8e",
      date: "December 2023",
      description:
        "A third-party, objective comparison of the abilities of the OpenAI GPT and Google Gemini models with reproducible code and fully transparent results.",
      tags: [],
    },
    {
      title: "STORM",
      link: "https://ritvik19.medium.com/papers-explained-242-storm-2c55270d3150",
      date: "February 2024",
      description:
        "A writing system that addresses the prewriting stage of long-form article generation by researching diverse perspectives, simulating multi-perspective question-asking, and curating information to create an outline, ultimately leading to more organized and comprehensive articles compared to baseline methods.",
      tags: ["Agentic Systems"],
    },
    {
      title: "NuNER",
      link: "https://ritvik19.medium.com/papers-explained-186-nuner-03e092dfb6ff",
      date: "February 2024",
      description: 
        "A foundation model for Named Entity Recognition (NER) created by further pre-training RoBERTa, using contrastive training on a large dataset annotated by GPT-3.5, derived from a subset of C4.",
      tags: ["Language Models","Named Entity Recognition", "Synthetic Data"],
    },
    {
      title: "LLMLingua2",
      link: "https://ritvik19.medium.com/papers-explained-138-llmlingua-2-510c752368a8",
      date: "March 2024",
      description:
        "A novel approach to task-agnostic prompt compression, aiming to enhance generalizability, using  data distillation and leveraging a Transformer encoder for token classification.",
        tags: ["Prompt Compression"],
    },
    {
      title: "Prometheus 2",
      link: "https://ritvik19.medium.com/papers-explained-171-prometheus-2-324e9c162e18",
      date: "May 2024",
      description:
        "7B & 8x7B evaluation LLMs that score high correlations with both human evaluators and proprietary LM-based judges on both direct assessment and pairwise ranking, obtained by merging Mistral models trained on Feedback Collection and Preference Collection (curated in this work.",
        tags: ["LLM Evaluation"],
    },
    {
      title: "PromptWizard",
      link: "https://ritvik19.medium.com/papers-explained-262-promptwizard-228568783085",
      date: "May 2024",
      description: 
        "A  framework that leverages LLMs to iteratively synthesize and refine prompts tailored to specific tasks by optimizing both prompt instructions and in-context examples, maximizing model performance.",
      tags: ["Prompt Optimization"],
    },
    {
      title: "LearnLM Tutor",
      link: "https://ritvik19.medium.com/papers-explained-279-learnlm-tutor-77503db05415",
      date: "May 2024",
      description:
        "A text-based gen AI tutor based on Gemini 1.0, further fine- tuned for 1:1 conversational tutoring with improved education-related capabilities over a prompt tuned Gemini 1.0.",
      tags: ["Language Models", "Gemini"],
    },
    {
      title: "Monte Carlo Tree Self-refine",
      link: "https://ritvik19.medium.com/papers-explained-167-monte-carlo-tree-self-refine-79bffb070c1a",
      date: "June 2024",
      description:
        "Integrates LLMs with Monte Carlo Tree Search to enhance performance in complex mathematical reasoning tasks, leveraging systematic exploration and heuristic self-refine mechanisms to improve decision-making frameworks.",
      tags: ["Scientific Data"],
    },
    {
      title: "NuExtract",
      link: "https://ritvik19.medium.com/papers-explained-287-nuextract-f722082999b5",
      date: "June 2024",
      description:
        "Small language models fine-tuned on a synthetic dataset of C4 text passages and LLM-generated structured extraction templates and outputs, achieving comparable or superior performance to much larger LLMs like GPT-4 on complex extraction tasks while being significantly smaller and offering zero-shot, hybrid few-shot (using output examples), and fine-tuning capabilities.",
      tags: ["Language Models", "Named Entity Recognition", "Synthetic Data"],
    },
    {
      title: "Proofread",
      link: "https://ritvik19.medium.com/papers-explained-189-proofread-4e1fe4eccf01",
      date: "June 2024",
      description:
        "A Gboard feature powered by a server-side LLM, enabling seamless sentence-level and paragraph-level corrections with a single tap.",
      tags: ["Language Models"],
    },
    {
      title: "CriticGPT",
      link: "https://ritvik19.medium.com/papers-explained-224-criticgpt-6d9af57451fa",
      date: "June 2024",
      description:
        "A model based on GPT-4 trained with RLHF to catch errors in ChatGPT's code output, accepts a question-answer pair as input and outputs a structured critique that highlights potential problems in the answer.",
      tags: ["Language Models", "OpenAI", "GPT"],
    },
    {
      title: "Gemma APS",
      link: "https://ritvik19.medium.com/papers-explained-244-gemma-aps-8fac1838b9ef",
      date: "June 2024",
      description:
        "Proposes a scalable, yet accurate, proposition segmentation model by modeling Proposition segmentation as a supervised task by training LLMs on existing annotated datasets.",
      tags: ["Language Models", "Gemma"],
    },
    {
      title: "Shield Gemma",
      link: "https://ritvik19.medium.com/papers-explained-243-shieldgemma-d779fd66ee3e",
      date: "July 2024",
      description:
        "A comprehensive suite of LLM-based safety content moderation models ranging from 2B to 27B parameters built upon Gemma2 that provide predictions of safety risks across key harm types (sexually explicit, dangerous content, harassment, hate speech) in both user input and LLM-generated output.",
      tags: ["Language Models", "Gemma", "LLM Safety"],
    },
    {
      title: "OmniParser",
      link: "https://ritvik19.medium.com/papers-explained-259-omniparser-2e895f6f2c15",
      date: "August 2024",
      description:
        "A method for parsing user interface screenshots into structured elements, enhancing the ability of GPT-4V to generate actions grounded in the interface by accurately identifying interactable icons and understanding element semantics.",
      tags: ["Language Models", "Multimodal Models", "BLIP"],
    },
    {
      title: "Reader-LM",
      link: "https://ritvik19.medium.com/papers-explained-221-reader-lm-7382b9eb6ed9",
      date: "September 2024",
      description:
        "Small multilingual models specifically trained to generate clean markdown directly from noisy raw HTML, with a context length of up to 256K tokens.",
      tags: ["Language Models"],
    },
    {
      title: "DataGemma",
      link: "https://ritvik19.medium.com/papers-explained-212-datagemma-cf0d2f40d867",
      date: "September 2024",
      description:
        "A set of models that aims to reduce hallucinations in LLMs by grounding them in the factual data of Google's Data Commons, allowing users to ask questions in natural language and receive responses based on verified information from trusted sources.",
      tags: ["Language Models", "Transformer Decoder", "Small LLMs", "Gemma"],
    },
    {
      title: "GSM-Symbolic",
      link: "https://ritvik19.medium.com/papers-explained-260-gsm-symbolic-759d379052c7",
      date: "October 2024",
      description:
        "Investigates the true mathematical reasoning capabilities of LLMs by introducing GSM-Symbolic, a new benchmark based on symbolic templates, revealing that LLMs exhibit inconsistent performance, struggle with complex questions, and appear to rely on pattern recognition rather than genuine logical reasoning.",
      tags: ["Scientific Data"],  
    },
    {
      title: "NuExtract 1.5",
      link: "https://ritvik19.medium.com/papers-explained-287-nuextract-f722082999b5#83ce",
      date: "October 2024",
      description:
        "An upgrade of NuExtract trained on a multilingual synthetic dataset derived from C4 documents and LLM-generated structured extraction templates and outputs, incorporating longer documents and a 'continuation' training methodology for handling documents exceeding typical context windows, resulting in a model capable of multilingual, long-document structured extraction with performance comparable or superior to much larger LLMs.",
      tags: ["Language Models", "Named Entity Recognition", "Synthetic Data"],
    },
    {
      title: "LearnLM",
      link: "https://ritvik19.medium.com/papers-explained-280-learnlm-df8cdc2fed45",
      date: "December 2024",
      description:
        "Combines Supervised Fine-Tuning (SFT) with Reinforcement Learning from Human Feedback (RLHF) to enhance pedagogical instruction in conversational AI, co-trained with Gemini's SFT, Reward Model (RM), and RL stages.",
      tags: ["Language Models", "Gemini"],
    },
    {
      title: "Open Scholar",
      link: "https://ritvik19.medium.com/papers-explained-185-openscholar-76b1b2df7b99",
      date: "December 2024",
      description:
        "A retrieval-augmented large language model specializing in scientific literature synthesis. It uses a large datastore of 45 million open-access papers, specialized retrievers, an open-source 8B model trained on synthetically generated data and iterative self-feedback generation to answer scientific queries with accurate citations.",
      tags: ["Scientific Data", "Agentic Systems"],
    },
    {
      title: "Shiksha",
      link: "https://ritvik19.medium.com/papers-explained-300-shiksha-ad09f6db7f38",
      date: "December 2024",
      description:
        "Addresses the challenge of training machine translation models to effectively handle scientific and technical language, particularly for low-resource Indian languages, by creating a multilingual parallel corpus containing over 2.8 million translation pairs across eight Indian languages by mining human-translated transcriptions of NPTEL video lectures.",
      tags: ["Language Models", "Multilingual Models"],
    },
    {
      title: "Multi-LLM Text Summarization",
      link: "https://ritvik19.medium.com/papers-explained-294-multi-llm-text-summarization-6141d1276772",
      date: "December 2024",
      description:
        "Introduces a novel multi-LLM summarization framework with centralized and decentralized approaches. Multiple LLMs generate diverse summaries, and either a single LLM (centralized) or all LLMs (decentralized) evaluate them iteratively.",
      tags: ["Language Models", "Agentic Systems"],
    },
    {
      title: "Reader LM v2",
      link: "https://ritvik19.medium.com/papers-explained-295-readerlm-v2-7103e9b25a10",
      date: "December 2024",
      description:
        "A 1.5B parameter language model specializing in converting raw HTML to markdown or JSON, handling up to 512K tokens and supporting 29 languages. Trained with a new paradigm and higher-quality data, it treats HTML-to-markdown as translation, and addresses the degeneration issues of its predecessor through contrastive loss.",
      tags: ["Language Models"],
    }
  ],
];

const surveys_data = [
  {
    title: "Best Practices and Lessons Learned on Synthetic Data",
    link: "2404.07503",
    date: "April 2024",
    description: "Provides an overview of synthetic data research, discussing its applications, challenges, and future directions.",
    tags: ["Survey", "Synthetic Data"],
  },
  {
    title: "The Prompt Report: A Systematic Survey of Prompting Techniques",
    link: "2406.06608",
    date: "June 2024",
    description: 
      "Establishes a structured understanding of prompts, by assembling a taxonomy of prompting techniques and analyzing their use.",
    tags: ["Survey", "Prompt Optimization"],
  },
  {
    title: "What is the Role of Small Models in the LLM Era",
    link: "2409.06857",
    date: "September 2024",
    description:
      "Systematically examines the relationship between LLMs and SMs from two key perspectives: Collaboration and Competition.",
    tags: ["Survey", "Small Models"],
  },
  {
    title: "Small Language Models: Survey, Measurements, and Insights",
    link: "2409.15790",
    date: "September 2024",
    description:
      "Surveys 59 SoTA open-source SLMs, analyzing their technical innovations across three axes: architectures, training datasets, and training algorithms, evaluate their capabilities, and benchmark their inference latency and memory footprints.",
    tags: ["Survey", "Small Models"],
  },
  {
    title: "A Survey of Small Language Models",
    link: "2410.20011",
    date: "October 2024",
    description:
      "A comprehensive survey on SLMs, focusing on their architectures, training techniques, and model compression techniques",
    tags: ["Survey", "Small Models"],
  }
]

const journeys_data = [
  {
    title: "Encoder Only Transformers",
    link: "transformer-encoders",
    papers: [
      "BERT", "RoBERTa", "Sentence BERT", "Tiny BERT", "ALBERT", "Distil BERT", "Distil RoBERTa", "FastBERT", "MobileBERT", 
      "ColBERT", "DeBERTa", "DeBERTa v2", "DeBERTa v3", "ColBERT v2"
    ],
  },
  {
    title: "Vision Transformers",
    link: "vision-transformers",
    papers: [
      "Vision Transformer", "What do Vision Transformers Learn?", "CNNs Match ViTs at Scale",
      "DeiT", "Swin Transformer", "CvT", "LeViT", "BEiT", 
      "MobileViT", "Masked AutoEncoder", "Max ViT", "Swin Transformer v2", "EfficientFormer",
      "FastVit", "Efficient ViT", "SoViT"
    ],
  },
  {
    title: "Low Rank Adaptors",
    link: "low-rank-adaptors",
    papers: [
      "LoRA", "DyLoRA", "AdaLoRA", "QLoRA", "LoRA-FA", "Delta-LoRA", "LongLoRA", "VeRA", "LoRA+", "MoRA", "DoRA"
    ],
  }
]

const literature_review_data = [
  {
    title: "Convolutional Neural Networks",
    link: "https://ritvik19.medium.com/papers-explained-review-01-convolutional-neural-networks-78aeff61dcb3",
    papers: [
      "LeNet", "AlexNet", "VGG", "Inception Net", "ResNet", "Inception Net v2 / v3", "ResNext", "DenseNet", "Xception", "MobileNet V1", "MobileNet V2", "MobileNet V3", "EfficientNet"
    ]
  },
  {
    title: "Layout Transformers",
    link: "https://ritvik19.medium.com/papers-explained-review-02-layout-transformers-b2d165c94ad5",
    papers: [
      "Layout LM", "LamBERT", "Layout LM v2", "Structural LM", "Doc Former",  "BROS", "LiLT", "Layout LM V3", "ERNIE Layout"
    ]
  },
  {
    title: "Region Based Convolutional Neural Networks",
    link: "https://ritvik19.medium.com/papers-explained-review-03-rcnns-42c0a3974493",
    papers: [
      "RCNN", "Fast RCNN", "Faster RCNN", "Mask RCNN", "Cascade RCNN"
    ]
  },
  {
    title: "Tabular Deep Learning",
    link: "https://ritvik19.medium.com/papers-explained-review-04-tabular-deep-learning-776db04f965b",
    papers: [
      "Entity Embeddings", "Wide and Deep Learning", "Deep and Cross Network", "Tab Transformer", "Tabular ResNet", "Feature Tokenizer Transformer"
    ]
  },
  {
    title: "Generative Adversarial Networks",
    link: "https://ritvik19.medium.com/papers-explained-review-05-generative-adversarial-networks-bbb51b160d5e",
    papers: [
      "GAN", "Conditional GAN", "Deep Convolutional GAN", "Improved GAN", "Wasserstein GAN", "Cycle GAN"
    ]
  },
  {
    title: "Parameter Efficient FineTuning",
    link: "https://ritvik19.medium.com/papers-explained-review-06-parameter-efficient-finetuning-6934fafa74e5",
    papers: [
      "LoRA", "DyLoRA", "AdaLoRA", "QLoRA", "LoRA-FA", "Delta-LoRA", "LongLoRA", "VeRA", "LoRA+", "MoRA", "DoRA"
    ]
  },
  {
    title: "Convolution Layers",
    link: "https://ritvik19.medium.com/papers-explained-review-07-convolution-layers-c083e7410cd3",
    papers: [
      "Convolution Layer", "Separable Convolution", "Pointwise Convolution", "Depthwise Convolution", "Convolution Transpose"
    ]
  },
  {
    title: "Recurrent Layers",
    link: "https://ritvik19.medium.com/papers-explained-review-08-recurrent-layers-ff2f224af059",
    papers: [
      "Simple Recurrent", "LSTM", "GRU"
    ]
  },
  {
    title: "Attention Layers",
    link: "https://ritvik19.medium.com/papers-explained-review-09-attention-layers-beeef323e7f5",
    papers: [
      "Scaled Dot Product Attention", "Multi Head Attention", "Cross Attention", "Causal Attention", "Sliding Window Attention", "Multi Query Attention", "Grouped Query Attention"
    ]
  },
  {
    title: "Normalization Layers",
    link: "https://ritvik19.medium.com/papers-explained-review-10-normalization-layers-56b556c9646e",
    papers: [
      "Batch Normalisation", "Layer Normalisation", "Instance Normalisation", "Group Normalisation", "Weight Standardisation", "Batch Channel Normalisation"
    ]
  },
  {
    title: "Auto Encoders",
    link: "https://ritvik19.medium.com/papers-explained-review-11-auto-encoders-3b8f08b4eac0",
    papers: [
      "Auto Encoders", "Sparse Auto Encoders", "K Sparse Auto Encoders", "Contractive Auto Encoders", "Convolutional Auto Encoders", "Sequence to Sequence Auto Encoders", "Denoising Auto Encoders", "Variational Auto Encoders", "Masked Auto Encoders"
    ]
  },
  {
    title: "LLMs for Maths",
    link: "https://ritvik19.medium.com/papers-explained-review-12-llms-for-maths-1597e3c7251c",
    papers: [
      "Wizard Math", "MAmmoTH", "MetaMath", "ToRA", "Math Coder", "MuggleMath", "Llemma", "MuMath", "MMIQC", "DeepSeek Math", "Open Math Instruct 1", "Math Orca", "Math Genie", "Xwin-Math", "MuMath Code", "Numina Math", "Qwen 2 Math", "Qwen 2.5 Math", "Open Math Instruct 2", "Math Coder 2", "AceMath"
    ]
  }
];

const reading_list_data = [
  {
    title: "Language Models",
    link: "https://ritvik19.medium.com/list/language-models-11b008ddc292",
  },
  {
    title: "Encoder Only Language Transformers",
    link: "https://ritvik19.medium.com/list/encoderonly-language-transformers-0f2ff06e0309",
  },
  {
    title: "Decoder Only Language Transformers",
    link: "https://ritvik19.medium.com/list/decoderonly-language-transformers-5448110c6046",
  },
  {
    title: "LLMs for Code",
    link: "https://ritvik19.medium.com/list/llms-for-code-e5360a1b353a",
  },
  {
    title: "Small LLMs",
    link: "https://ritvik19.medium.com/list/small-llms-41124d5c7c80",
  },
  {
    title: "GPT Models",
    link: "https://ritvik19.medium.com/list/gpt-models-fa2cc801d840",
  },
  {
    title: "LLaMA Models",
    link: "https://ritvik19.medium.com/list/llama-models-5b8ea07308cb",
  },
  {
    title: "Gemini / Gemma Models",
    link: "https://ritvik19.medium.com/list/gemini-gemma-models-4cb7dfc50d42",
  },
  {
    title: "Wizard Models",
    link: "https://ritvik19.medium.com/list/wizard-models-9b972e860683",
  },
  {
    title: "Orca Series",
    link: "https://ritvik19.medium.com/list/orca-series-1c87367458fe",
  },
  {
    title: "BLIP Series",
    link: "https://ritvik19.medium.com/list/blip-series-4b0831017c5b",
  },
  {
    title: "LLM Lingua Series",
    link: "https://ritvik19.medium.com/list/llm-lingua-series-2f61b47d0343",
  },
  {
    title: "Multi Task Language Models",
    link: "https://ritvik19.medium.com/list/multi-task-language-models-e6a2a1e517e6",
  },
  {
    title: "Layout Aware Transformers",
    link: "https://ritvik19.medium.com/list/layout-transformers-1ce4f291a9f0",
  },
  {
    title: "Retrieval and Representation Learning",
    link: "https://ritvik19.medium.com/list/retrieval-and-representation-learning-bcd23de0bd8e",
  },
  {
    title: "Vision Transformers",
    link: "https://ritvik19.medium.com/list/vision-transformers-61e6836230f1",
  },
  {
    title: "Multi Modal Transformers",
    link: "https://ritvik19.medium.com/list/multi-modal-transformers-67453f215ecf",
  },
  {
    title: "LLM Evaluation",
    link: "https://ritvik19.medium.com/list/llm-evaluation-a011ddd1a546",
  },
  {
    title: "Convolutional Neural Networks",
    link: "https://ritvik19.medium.com/list/convolutional-neural-networks-5b875ce3b689",
  },
  {
    title: "Object Detection",
    link: "https://ritvik19.medium.com/list/object-detection-bd9e6e21ca3e",
  },
  {
    title: "Region Based Convolutional Neural Networks",
    link: "https://ritvik19.medium.com/list/rcnns-b51467f53dc9",
  },
  {
    title: "Document Information Processing",
    link: "https://ritvik19.medium.com/list/document-information-processing-3cd900a34972",
  },
];
