const nav_data = [
  "Language Models",
  "Encoder Only Transformers",
  "Decoder Only Transformers",
  "Small LLMs",
  "Multi Modal LMs",
  "Retrieval and Representation Learning",
  "PEFT",
  "LLM Evaluation",
  "Compression, Pruning, Quantization",
  "Vision Transformers",
  "CNNs",
  "Object Detection",
  "RCNNs",
  "Document Understanding",
  "Layout Aware LMs",
  "GANs",
  "Tabular Data",
  "Datasets",
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
    },
    {
      title: "Elmo",
      link: "https://ritvik19.medium.com/papers-explained-33-elmo-76362a43e4",
      date: "February 2018",
      description:
        "Deep contextualized word representations that captures both intricate aspects of word usage and contextual variations across language contexts.",
    },
    {
      title: "Marian MT",
      link: "https://ritvik19.medium.com/papers-explained-150-marianmt-1b44479b0fd9",
      date: "April 2018",
      description:
        "A Neural Machine Translation framework written entirely in C++ with minimal dependencies, designed for high training and translation speed.",
    },
    {
      title: "Transformer XL",
      link: "https://ritvik19.medium.com/papers-explained-34-transformerxl-2e407e780e8",
      date: "January 2019",
      description:
        "Extends the original Transformer model to handle longer sequences of text by introducing recurrence into the self-attention mechanism.",
    },
    {
      title: "XLM",
      link: "https://ritvik19.medium.com/papers-explained-158-xlm-42a175e93caf",
      date: "January 2019",
      description:
        "Proposes two methods to learn cross-lingual language models (XLMs): one unsupervised that only relies on monolingual data, and one supervised that leverages parallel data with a new cross-lingual language model objective.",
    },
    {
      title: "Sparse Transformer",
      link: "https://ritvik19.medium.com/papers-explained-122-sparse-transformer-906a0be1e4e7",
      date: "April 2019",
      description:
        "Introduced sparse factorizations of the attention matrix to reduce the time and memory consumption to O(n√ n) in terms of sequence lengths.",
    },
    {
      title: "UniLM",
      link: "https://ritvik19.medium.com/papers-explained-72-unilm-672f0ecc6a4a",
      date: "May 2019",
      description:
        "Utilizes a shared Transformer network and specific self-attention masks to excel in both language understanding and generation tasks.",
    },
    {
      title: "XLNet",
      link: "https://ritvik19.medium.com/papers-explained-35-xlnet-ea0c3af96d49",
      date: "June 2019",
      description:
        "Extension of the Transformer-XL, pre-trained using a new method that combines ideas from AR and AE objectives.",
    },
    {
      title: "CTRL",
      link: "https://ritvik19.medium.com/papers-explained-153-ctrl-146fcd18a566",
      date: "September 2019",
      description:
        "A 1.63B language model that can generate text conditioned on control codes that govern style, content, and task-specific behavior, allowing for more explicit control over text generation.",
    },
    {
      title: "BART",
      link: "https://ritvik19.medium.com/papers-explained-09-bart-7f56138175bd",
      date: "October 2019",
      description:
        "An encoder-decoder network pretrained to reconstruct the original text from corrupted versions of it.",
    },
    {
      title: "T5",
      link: "https://ritvik19.medium.com/papers-explained-44-t5-9d974a3b7957",
      date: "October 2019",
      description:
        "A unified encoder-decoder framework that converts all text-based language problems into a text-to-text format.",
    },
    {
      title: "XLM-Roberta",
      link: "https://ritvik19.medium.com/papers-explained-159-xlm-roberta-2da91fc24059",
      date: "November 2019",
      description:
        "A multilingual masked language model pre-trained on text in 100 languages, shows that pretraining multilingual language models at scale leads to significant performance gains for a wide range of crosslingual transfer tasks.",
    },
    {
      title: "Pegasus",
      link: "https://ritvik19.medium.com/papers-explained-162-pegasus-1cb16f572553",
      date: "December 2019",
      description:
        "A self-supervised pre-training objective for abstractive text summarization, proposes removing/masking important sentences from an input document and generating them together as one output sequence.",
    },
    {
      title: "Reformer",
      link: "https://ritvik19.medium.com/papers-explained-165-reformer-4445ad305191",
      date: "January 2020",
      description:
        "Improves the efficiency of Transformers by replacing dot-product attention with locality-sensitive hashing (O(Llog L) complexity), using reversible residual layers to store activations only once, and splitting feed-forward layer activations into chunks, allowing it to perform on par with Transformer models while being much more memory-efficient and faster on long sequences.",
    },
    {
      title: "mBART",
      link: "https://ritvik19.medium.com/papers-explained-169-mbart-98432ef6fec",
      date: "January 2020",
      description:
        "A multilingual sequence-to-sequence denoising auto-encoder that pre-trains a complete autoregressive model on large-scale monolingual corpora across many languages using the BART objective, achieving significant performance gains in machine translation tasks.",
    },
    {
      title: "UniLMv2",
      link: "https://ritvik19.medium.com/papers-explained-unilmv2-5a044ca7c525",
      date: "February 2020",
      description:
        "Utilizes a pseudo-masked language model (PMLM) for both autoencoding and partially autoregressive language modeling tasks,significantly advancing the capabilities of language models in diverse NLP tasks.",
    },
    {
      title: "ELECTRA",
      link: "https://ritvik19.medium.com/papers-explained-173-electra-501c175ae9d8",
      date: "March 2020",
      description:
        "Proposes a sample-efficient pre-training task called replaced token detection, which corrupts input by replacing some tokens with plausible alternatives and trains a discriminative model to predict whether each token was replaced or no.",
    },
    {
      title: "Longformer",
      link: "https://ritvik19.medium.com/papers-explained-38-longformer-9a08416c532e",
      date: "April 2020",
      description:
        "Introduces a linearly scalable attention mechanism, allowing handling texts of exteded length.",
    },
    {
      title: "T5 v1.1",
      link: "https://ritvik19.medium.com/papers-explained-44-t5-9d974a3b7957#773b",
      date: "July 2020",
      description:
        "An enhanced version of the original T5 model, featuring improvements such as GEGLU activation, no dropout in pre-training, exclusive pre-training on C4, no parameter sharing between embedding and classifier layers.",
    },
    {
      title: "mT5",
      link: "https://ritvik19.medium.com/papers-explained-113-mt5-c61e03bc9218",
      date: "October 2020",
      description:
        "A multilingual variant of T5 based on T5 v1.1, pre-trained on a new Common Crawl-based dataset covering 101 languages (mC4).",
    },
    {
      title: "FLAN",
      link: "https://ritvik19.medium.com/papers-explained-46-flan-1c5e0d5db7c9",
      date: "September 2021",
      description:
        "An instruction-tuned language model developed through finetuning on various NLP datasets described by natural language instructions.",
    },
    {
      title: "T0",
      link: "https://ritvik19.medium.com/papers-explained-74-t0-643a53079fe",
      date: "October 2021",
      description:
        "A fine tuned encoder-decoder model on a multitask mixture covering a wide variety of tasks, attaining strong zero-shot performance on several standard datasets.",
    },
    {
      title: "BERTopic",
      link: "https://ritvik19.medium.com/papers-explained-193-bertopic-f9aec10cd5a6",
      date: "March 2022",
      description:
        "Utilizes Sentence-BERT for document embeddings, UMAP, HDBSCAN (soft-clustering), and an adjusted class-based TF-IDF, addressing multiple topics per document and dynamic topics' linear evolution.",
    },
    {
      title: "Flan T5, Flan PaLM",
      link: "https://ritvik19.medium.com/papers-explained-75-flan-t5-flan-palm-caf168b6f76",
      date: "October 2022",
      description:
        "Explores instruction fine tuning with a particular focus on scaling the number of tasks, scaling the model size, and fine tuning on chain-of-thought data.",
    },
    {
      title: "BLOOMZ, mT0",
      link: "https://ritvik19.medium.com/papers-explained-99-bloomz-mt0-8932577dcd1d",
      date: "November 2022",
      description:
        "Applies Multitask prompted fine tuning to the pretrained multilingual models on English tasks with English prompts to attain task generalization to non-English languages that appear only in the pretraining corpus.",
    },
    {
      title: "Self Instruct",
      link: "https://ritvik19.medium.com/papers-explained-112-self-instruct-5c192580103a",
      date: "December 2022",
      description:
        "A framework for improving the instruction-following capabilities of pretrained language models by bootstrapping off their own generations.",
    },
    {
      title: "CodeFusion",
      link: "https://ritvik19.medium.com/papers-explained-70-codefusion-fee6aba0149a",
      date: "October 2023",
      description:
        "A diffusion code generation model that iteratively refines entire programs based on encoded natural language, overcoming the limitation of auto-regressive models in code generation by allowing reconsideration of earlier tokens.",
    },
    {
      title: "Aya 101",
      link: "https://ritvik19.medium.com/papers-explained-aya-101-d813ba17b83a",
      date: "February 2024",
      description:
        "A massively multilingual generative language model that follows instructions in 101 languages,trained by finetuning mT5.",
    },
    {
      title: "Hawk, Griffin",
      link: "https://ritvik19.medium.com/papers-explained-131-hawk-griffin-dfc8c77f5dcd",
      date: "February 2024",
      description:
        "Introduces Real Gated Linear Recurrent Unit Layer that forms the core of the new recurrent block, replacing Multi-Query Attention for better efficiency and scalability",
    },
    {
      title: "WRAP",
      link: "https://ritvik19.medium.com/papers-explained-118-wrap-e563e009fe56",
      date: "March 2024",
      description:
        "Uses an off-the-shelf instruction-tuned model prompted to paraphrase documents on the web in specific styles to jointly pre-train LLMs on real and synthetic rephrases.",
    },
    {
      title: "RecurrentGemma",
      link: "https://ritvik19.medium.com/papers-explained-132-recurrentgemma-52732d0f4273",
      date: "April 2024",
      description:
        "Based on Griffin, uses a combination of linear recurrences and local attention instead of global attention to model long sequences efficiently.",
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
    },
    {
      title: "RoBERTa",
      link: "https://ritvik19.medium.com/papers-explained-03-roberta-81db014e35b9",
      date: "July 2019",
      description:
        "Built upon BERT, by carefully optimizing hyperparameters and training data size to improve performance on various language tasks .",
    },
    {
      title: "Sentence BERT",
      link: "https://ritvik19.medium.com/papers-explained-04-sentence-bert-5159b8e07f21",
      date: "August 2019",
      description:
        "A modification of BERT that uses siamese and triplet network structures to derive sentence embeddings that can be compared using cosine-similarity.",
    },
    {
      title: "Tiny BERT",
      link: "https://ritvik19.medium.com/papers-explained-05-tiny-bert-5e36fe0ee173",
      date: "September 2019",
      description:
        "Uses attention transfer, and task specific distillation for distilling BERT.",
    },
    {
      title: "ALBERT",
      link: "https://ritvik19.medium.com/papers-explained-07-albert-46a2a0563693",
      date: "September 2019",
      description:
        "Presents certain parameter reduction techniques to lower memory consumption and increase the training speed of BERT.",
    },
    {
      title: "Distil BERT",
      link: "https://ritvik19.medium.com/papers-explained-06-distil-bert-6f138849f871",
      date: "October 2019",
      description:
        "Distills BERT on very large batches leveraging gradient accumulation, using dynamic masking and without the next sentence prediction objective.",
    },
    {
      title: "FastBERT",
      link: "https://ritvik19.medium.com/papers-explained-37-fastbert-5bd246c1b432",
      date: "April 2020",
      description:
        "A speed-tunable encoder with adaptive inference time having branches at each transformer output to enable early outputs.",
    },
    {
      title: "MobileBERT",
      link: "https://ritvik19.medium.com/papers-explained-36-mobilebert-933abbd5aaf1",
      date: "April 2020",
      description:
        "Compressed and faster version of the BERT, featuring bottleneck structures, optimized attention mechanisms, and knowledge transfer.",
    },
    {
      title: "DeBERTa",
      link: "https://ritvik19.medium.com/papers-explained-08-deberta-a808d9b2c52d",
      date: "June 2020",
      description:
        "Enhances BERT and RoBERTa through disentangled attention mechanisms, an enhanced mask decoder, and virtual adversarial training.",
    },
    {
      title: "DeBERTa v2",
      link: "https://ritvik19.medium.com/papers-explained-08-deberta-a808d9b2c52d#f5e1",
      date: "June 2020",
      description:
        "Enhanced version of the DeBERTa featuring a new vocabulary, nGiE integration, optimized attention mechanisms, additional model sizes, and improved tokenization.",
    },
    {
      title: "DeBERTa v3",
      link: "https://ritvik19.medium.com/papers-explained-182-deberta-v3-65347208ce03",
      date: "November 2021",
      description:
        "Enhances the DeBERTa architecture by introducing replaced token detection (RTD) instead of mask language modeling (MLM), along with a novel gradient-disentangled embedding sharing method, exhibiting superior performance across various natural language understanding tasks.",
    },
  ],
  [
    // Decoder Only Transformers
    {
      title: "GPT",
      link: "https://ritvik19.medium.com/papers-explained-43-gpt-30b6f1e6d226",
      date: "June 2018",
      description:
        "A Decoder only transformer which is autoregressively pretrained and then finetuned for specific downstream tasks using task-aware input transformations.",
    },
    {
      title: "GPT 2",
      link: "https://ritvik19.medium.com/papers-explained-65-gpt-2-98d0a642e520",
      date: "February 2019",
      description:
        "Demonstrates that language models begin to learn various language processing tasks without any explicit supervision.",
    },
    {
      title: "GPT 3",
      link: "https://ritvik19.medium.com/papers-explained-66-gpt-3-352f5a1b397",
      date: "May 2020",
      description:
        "Demonstrates that scaling up language models greatly improves task-agnostic, few-shot performance.",
    },
    {
      title: "Codex",
      link: "https://ritvik19.medium.com/papers-explained-45-codex-caca940feb31",
      date: "July 2021",
      description:
        "A GPT language model finetuned on publicly available code from GitHub.",
    },
    {
      title: "WebGPT",
      link: "https://ritvik19.medium.com/papers-explained-123-webgpt-5bb0dd646b32",
      date: "December 2021",
      description:
        "A fine-tuned GPT-3 model utilizing text-based web browsing, trained via imitation learning and human feedback, enhancing its ability to answer long-form questions with factual accuracy.",
    },
    {
      title: "Gopher",
      link: "https://ritvik19.medium.com/papers-explained-47-gopher-2e71bbef9e87",
      date: "December 2021",
      description:
        "Provides a comprehensive analysis of the performance of various Transformer models across different scales upto 280B on 152 tasks.",
    },
    {
      title: "LaMDA",
      link: "https://ritvik19.medium.com/papers-explained-76-lamda-a580ebba1ca2",
      date: "January 2022",
      description:
        "Transformer based models specialized for dialog, which are pre-trained on public dialog data and web text.",
    },
    {
      title: "Instruct GPT",
      link: "https://ritvik19.medium.com/papers-explained-48-instructgpt-e9bcd51f03ec",
      date: "March 2022",
      description:
        "Fine-tuned GPT using supervised learning (instruction tuning) and reinforcement learning from human feedback to align with user intent.",
    },
    {
      title: "Chinchilla",
      link: "https://ritvik19.medium.com/papers-explained-49-chinchilla-a7ad826d945e",
      date: "March 2022",
      description:
        "Investigated the optimal model size and number of tokens for training a transformer LLM within a given compute budget (Scaling Laws).",
    },
    {
      title: "CodeGen",
      link: "https://ritvik19.medium.com/papers-explained-125-codegen-a6bae5c1f7b5",
      date: "March 2022",
      description:
        "An LLM trained for program synthesis using input-output examples and natural language descriptions.",
    },
    {
      title: "PaLM",
      link: "https://ritvik19.medium.com/papers-explained-50-palm-480e72fa3fd5",
      date: "April 2022",
      description:
        "A 540-B parameter, densely activated, Transformer, trained using Pathways, (ML system that enables highly efficient training across multiple TPU Pods).",
    },
    {
      title: "GPT-NeoX-20B",
      link: "https://ritvik19.medium.com/papers-explained-78-gpt-neox-20b-fe39b6d5aa5b",
      date: "April 2022",
      description:
        "An autoregressive LLM trained on the Pile, and the largest dense model that had publicly available weights at the time of submission.",
    },
    {
      title: "OPT",
      link: "https://ritvik19.medium.com/papers-explained-51-opt-dacd9406e2bd",
      date: "May 2022",
      description:
        "A suite of decoder-only pre-trained transformers with parameter ranges from 125M to 175B. OPT-175B being comparable to GPT-3.",
    },
    {
      title: "BLOOM",
      link: "https://ritvik19.medium.com/papers-explained-52-bloom-9654c56cd2",
      date: "November 2022",
      description:
        "A 176B-parameter open-access decoder-only transformer, collaboratively developed by hundreds of researchers, aiming to democratize LLM technology.",
    },
    {
      title: "Galactica",
      link: "https://ritvik19.medium.com/papers-explained-53-galactica-1308dbd318dc",
      date: "November 2022",
      description:
        "An LLM trained on scientific data thus specializing in scientific knowledge.",
    },
    {
      title: "ChatGPT",
      link: "https://ritvik19.medium.com/papers-explained-54-chatgpt-78387333268f",
      date: "November 2022",
      description:
        "An interactive model designed to engage in conversations, built on top of GPT 3.5.",
    },
    {
      title: "LLaMA",
      link: "https://ritvik19.medium.com/papers-explained-55-llama-c4f302809d6b",
      date: "February 2023",
      description:
        "A collection of foundation LLMs by Meta ranging from 7B to 65B parameters, trained using publicly available datasets exclusively.",
    },
    {
      title: "Toolformer",
      link: "https://ritvik19.medium.com/papers-explained-140-toolformer-d21d496b6812",
      date: "February 2023",
      description:
        "An LLM trained to decide which APIs to call, when to call them, what arguments to pass, and how to best incorporate the results into future token prediction.",
    },
    {
      title: "Alpaca",
      link: "https://ritvik19.medium.com/papers-explained-56-alpaca-933c4d9855e5",
      date: "March 2023",
      description:
        "A fine-tuned LLaMA 7B model, trained on instruction-following demonstrations generated in the style of self-instruct using text-davinci-003.",
    },
    {
      title: "GPT 4",
      link: "https://ritvik19.medium.com/papers-explained-67-gpt-4-fc77069b613e",
      date: "March 2023",
      description:
        "A multimodal transformer model pre-trained to predict the next token in a document, which can accept image and text inputs and produce text outputs.",
    },
    {
      title: "Vicuna",
      link: "https://ritvik19.medium.com/papers-explained-101-vicuna-daed99725c7e",
      date: "March 2023",
      description:
        "A 13B LLaMA chatbot fine tuned on user-shared conversations collected from ShareGPT, capable of generating more detailed and well-structured answers compared to Alpaca.",
    },
    {
      title: "Bloomberg GPT",
      link: "https://ritvik19.medium.com/papers-explained-120-bloomberggpt-4bedd52ef54b",
      date: "March 2023",
      description:
        "A 50B language model train on general purpose and domain specific data to support a wide range of tasks within the financial industry.",
    },
    {
      title: "Pythia",
      link: "https://ritvik19.medium.com/papers-explained-121-pythia-708284c32964",
      date: "April 2023",
      description:
        "A suite of 16 LLMs all trained on public data seen in the exact same order and ranging in size from 70M to 12B parameters.",
    },
    {
      title: "WizardLM",
      link: "https://ritvik19.medium.com/papers-explained-127-wizardlm-65099705dfa3",
      date: "April 2023",
      description:
        "Introduces Evol-Instruct, a method to generate large amounts of instruction data with varying levels of complexity using LLM instead of humans to fine tune a Llama model ",
    },
    {
      title: "CodeGen 2",
      link: "https://ritvik19.medium.com/papers-explained-codegen2-d2690d7eb831",
      date: "May 2023",
      description:
        "Proposes an approach to make the training of LLMs for program synthesis more efficient by unifying key components of model architectures, learning methods, infill sampling, and data distributions",
    },
    {
      title: "PaLM 2",
      link: "https://ritvik19.medium.com/papers-explained-58-palm-2-1a9a23f20d6c",
      date: "May 2023",
      description:
        "Successor of PALM, trained on a mixture of different pre-training objectives in order to understand different aspects of language.",
    },
    {
      title: "LIMA",
      link: "https://ritvik19.medium.com/papers-explained-57-lima-f9401a5760c3",
      date: "May 2023",
      description:
        "A LLaMa model fine-tuned on only 1,000 carefully curated prompts and responses, without any reinforcement learning or human preference modeling.",
    },
    {
      title: "Gorilla",
      link: "https://ritvik19.medium.com/papers-explained-139-gorilla-79f4730913e9",
      date: "May 2023",
      description:
        "A retrieve-aware finetuned LLaMA-7B model, specifically for API calls.",
    },
    {
      title: "Falcon",
      link: "https://ritvik19.medium.com/papers-explained-59-falcon-26831087247f",
      date: "June 2023",
      description:
        "An Open Source LLM trained on properly filtered and deduplicated web data alone.",
    },
    {
      title: "WizardCoder",
      link: "https://ritvik19.medium.com/papers-explained-wizardcoder-a12ecb5b93b6",
      date: "June 2023",
      description:
        "Enhances the performance of the open-source Code LLM, StarCoder, through the application of Code Evol-Instruct.",
    },
    {
      title: "LLaMA 2",
      link: "https://ritvik19.medium.com/papers-explained-60-llama-v2-3e415c5b9b17",
      date: "July 2023",
      description:
        "Successor of LLaMA. LLaMA 2-Chat is optimized for dialogue use cases.",
    },
    {
      title: "Tool LLM",
      link: "https://ritvik19.medium.com/papers-explained-141-tool-llm-856f99e79f55",
      date: "July 2023",
      description:
        "A LLaMA model finetuned on an instruction-tuning dataset for tool use, automatically created using ChatGPT.",
    },
    {
      title: "Humpback",
      link: "https://ritvik19.medium.com/papers-explained-61-humpback-46992374fc34",
      date: "August 2023",
      description: "LLaMA finetuned using Instruction backtranslation.",
    },
    {
      title: "Code LLaMA",
      link: "https://ritvik19.medium.com/papers-explained-62-code-llama-ee266bfa495f",
      date: "August 2023",
      description: "LLaMA 2 based LLM for code.",
    },
    {
      title: "WizardMath",
      link: "https://ritvik19.medium.com/papers-explained-129-wizardmath-265e6e784341",
      date: "August 2023",
      description:
        "Proposes Reinforcement Learning from Evol-Instruct Feedback (RLEIF) method, applied to Llama-2 to enhance the mathematical reasoning abilities.",
    },
    {
      title: "LLaMA 2 Long",
      link: "https://ritvik19.medium.com/papers-explained-63-llama-2-long-84d33c26d14a",
      date: "September 2023",
      description:
        "A series of long context LLMs s that support effective context windows of up to 32,768 tokens.",
    },
    {
      title: "Llemma",
      link: "https://ritvik19.medium.com/papers-explained-69-llemma-0a17287e890a",
      date: "October 2023",
      description:
        "An LLM for mathematics, formed by continued pretraining of Code Llama on a mixture of scientific papers, web data containing mathematics, and mathematical code.",
    },
    {
      title: "Grok 1",
      link: "https://ritvik19.medium.com/papers-explained-186-grok-0d9f1aef69be",
      date: "November 2023",
      description:
        "A 314B Mixture-of-Experts model, modeled after the Hitchhiker's Guide to the Galaxy, designed to be witty.",
    },
    {
      title: "Mixtral 8x7B",
      link: "https://ritvik19.medium.com/papers-explained-95-mixtral-8x7b-9e9f40ebb745",
      date: "January 2024",
      description:
        "A Sparse Mixture of Experts language model based on Mistral 7B trained with multilingual data using a context size of 32k tokens.",
    },
    {
      title: "DBRX",
      link: "https://ritvik19.medium.com/papers-explained-119-dbrx-17c61739983c",
      date: "March 2024",
      description:
        "A 132B open, general-purpose fine grained Sparse MoE LLM surpassing GPT-3.5 and competitive with Gemini 1.0 Pro.",
    },
    {
      title: "Command R",
      link: "https://ritvik19.medium.com/papers-explained-166-command-r-models-94ba068ebd2b",
      date: "March 2024",
      description:
        "An LLM optimized for retrieval-augmented generation and tool use, across multiple languages.",
    },
    {
      title: "Grok 1.5",
      link: "https://ritvik19.medium.com/papers-explained-186-grok-0d9f1aef69be",
      date: "March 2024",
      description:
        "An advancement over grok, capable of long context understanding up to 128k tokens and advanced reasoning.",
    },
    {
      title: "Mixtral 8x22B",
      link: "https://ritvik19.medium.com/papers-explained-95-mixtral-8x7b-9e9f40ebb745#20f3",
      date: "April 2024",
      description:
        "A open-weight AI model optimised for performance and efficiency, with capabilities such as fluency in multiple languages, strong mathematics and coding abilities, and precise information recall from large documents.",
    },
    {
      title: "Llama 3",
      link: "https://ritvik19.medium.com/papers-explained-187a-llama-3-51e2b90f63bb",
      date: "April 2024",
      description:
        "A family of 8B and 70B parameter models trained on 15T tokens with a focus on data quality, demonstrating state-of-the-art performance on various benchmarks, improved reasoning capabilities.",
    },
    {
      title: "Command R+",
      link: "https://ritvik19.medium.com/papers-explained-166-command-r-models-94ba068ebd2b#c2b5",
      date: "April 2024",
      description:
        "Successor of Command R+ with improved performance for retrieval-augmented generation and tool use, across multiple languages.",
    },
    {
      title: "Rho-1",
      link: "https://ritvik19.medium.com/papers-explained-132-rho-1-788125e42241",
      date: "April 2024",
      description:
        "Introduces Selective Language Modelling that optimizes the loss only on tokens that align with a desired distribution, utilizing a reference model to score and select tokens.",
    },
    {
      title: "Codestral 22B",
      link: "https://ritvik19.medium.com/papers-explained-mistral-7b-b9632dedf580#057b",
      date: "May 2024",
      description:
        "An open-weight model designed for code generation tasks, trained on over 80 programming languages, and licensed under the Mistral AI Non-Production License, allowing developers to use it for research and testing purposes.",
    },
    {
      title: "Aya 23",
      link: "https://ritvik19.medium.com/papers-explained-151-aya-23-d01605c3ee80",
      date: "May 2024",
      description:
        "A family of multilingual language models supporting 23 languages, designed to balance breadth and depth by allocating more capacity to fewer languages during pre-training.",
    },
    {
      title: "LLama 3.1",
      link: "https://ritvik19.medium.com/papers-explained-187b-llama-3-1-f0fb06898c59",
      date: "July 2024",
      description:
        "A family of multilingual language models ranging from 8B to 405B parameters, trained on a massive dataset of 15T tokens and achieving comparable performance to leading models like GPT-4 on various tasks.",
    },
    {
      title: "LLama 3.1 - Multimodal Experiments",
      link: "https://ritvik19.medium.com/papers-explained-187c-llama-3-1-multimodal-experiments-a1940dd45575",
      date: "July 2024",
      description:
        "Additional experiments of adding multimodal capabilities to Llama3.",
    },
    {
      title: "Mistral Large 2",
      link: "https://ritvik19.medium.com/papers-explained-mistral-7b-b9632dedf580#301d",
      date: "July 2024",
      description:
        "A 123B model, offers significant improvements in code generation, mathematics, and reasoning capabilities, advanced function calling, a 128k context window, and supports dozens of languages and over 80 coding languages.",
    },
    {
      title: "Hermes 3",
      link: "https://ritvik19.medium.com/papers-explained-188-hermes-3-67d36cfe07d8",
      date: "August 2024",
      description:
        "Neutrally generalist instruct and tool use models, created by fine-tuning Llama 3.1 models with strong reasoning and creative abilities, and are designed to follow prompts neutrally without moral judgment or personal opinions.",
    },
  ],
  [
    // Small LLMs
    {
      title: "Phi-1",
      link: "https://ritvik19.medium.com/papers-explained-114-phi-1-14a8dcc77ce5",
      date: "June 2023",
      description:
        "An LLM for code, trained using a textbook quality data from the web and synthetically generated textbooks and exercises with GPT-3.5.",
    },
    {
      title: "Orca",
      link: "https://ritvik19.medium.com/papers-explained-160-orca-928eff06e7f9",
      date: "June 2023",
      description:
        "Presents a novel approach that addresses the limitations of instruction tuning by leveraging richer imitation signals, scaling tasks and instructions, and utilizing a teacher assistant to help with progressive learning.",
    },
    {
      title: "Phi-1.5",
      link: "https://ritvik19.medium.com/papers-explained-phi-1-5-2857e56dbd2a",
      date: "September 2023",
      description:
        "Follows the phi-1 approach, focusing this time on common sense reasoning in natural language.",
    },
    {
      title: "Mistral 7B",
      link: "https://ritvik19.medium.com/papers-explained-mistral-7b-b9632dedf580",
      date: "October 2023",
      description:
        "Leverages grouped-query attention for faster inference, coupled with sliding window attention to effectively handle sequences of arbitrary length with a reduced inference cost.",
    },
    {
      title: "Zephyr 7B",
      link: "https://ritvik19.medium.com/papers-explained-71-zephyr-7ec068e2f20b",
      date: "October 2023",
      description:
        "Utilizes dDPO and AI Feedback (AIF) preference data to achieve superior intent alignment in chat-based language modeling.",
    },
    {
      title: "Orca 2",
      link: "https://ritvik19.medium.com/papers-explained-161-orca-2-b6ffbccd1eef",
      date: "November 2023",
      description:
        "Introduces Cautious Reasoning for training smaller models to select the most effective solution strategy based on the problem at hand, by crafting data with task-specific system instruction(s) corresponding to the chosen strategy in order to obtain teacher responses for each task and replacing the student’s system instruction with a generic one vacated of details of how to approach the task.",
    },
    {
      title: "Phi-2",
      link: "https://ritvik19.medium.com/papers-explained-phi-1-5-2857e56dbd2a#8230",
      date: "December 2023",
      description:
        "A 2.7B model, developed to explore whether emergent abilities achieved by large-scale language models can also be achieved at a smaller scale using strategic choices for training, such as data selection.",
    },
    {
      title: "TinyLlama",
      link: "https://ritvik19.medium.com/papers-explained-93-tinyllama-6ef140170da9",
      date: "January 2024",
      description:
        "A  1.1B language model built upon the architecture and tokenizer of Llama 2, pre-trained on around 1 trillion tokens for approximately 3 epochs, leveraging FlashAttention and Grouped Query Attention, to achieve better computational efficiency.",
    },
    {
      title: "H2O Danube 1.8B",
      link: "https://ritvik19.medium.com/papers-explained-111-h2o-danube-1-8b-b790c073d257",
      date: "January 2024",
      description:
        "A language model trained on 1T tokens following the core principles of LLama 2 and Mistral, leveraging and refining various techniques for pre-training large language models.",
    },
    {
      title: "OLMo",
      link: "https://ritvik19.medium.com/papers-explained-98-olmo-fdc358326f9b",
      date: "February 2024",
      description:
        "A state-of-the-art, truly open language model and framework that includes training data, code, and tools for building, studying, and advancing language models.",
    },
    {
      title: "Orca Math",
      link: "https://ritvik19.medium.com/papers-explained-163-orca-math-ae6a157ce48d",
      date: "February 2024",
      description:
        "A fine tuned Mistral-7B that excels at math problems without external tools, utilizing a high-quality synthetic dataset of 200K problems created through multi-agent collaboration and an iterative learning process that involves practicing problem-solving, receiving feedback, and learning from preference pairs incorporating the model's solutions and feedback.",
    },
    {
      title: "Gemma",
      link: "https://ritvik19.medium.com/papers-explained-106-gemma-ca2b449321ac",
      date: "February 2024",
      description:
        "A family of 2B and 7B, state-of-the-art language models based on Google's Gemini models, offering advancements in language understanding, reasoning, and safety.",
    },
    {
      title: "CodeGemma",
      link: "https://ritvik19.medium.com/papers-explained-124-codegemma-85faa98af20d",
      date: "April 2024",
      description:
        "Open code models based on Gemma models by further training on over 500 billion tokens of primarily code.",
    },
    {
      title: "Phi-3",
      link: "https://ritvik19.medium.com/papers-explained-130-phi-3-0dfc951dc404",
      date: "April 2024",
      description:
        "A series of language models trained on heavily filtered web and synthetic data set, achieving performance comparable to much larger models like Mixtral 8x7B and GPT-3.5.",
    },
    {
      title: "Open ELM",
      link: "https://ritvik19.medium.com/papers-explained-133-open-elm-864f6b28a6ab",
      date: "April 2024",
      description:
        "A fully open language model designed to enhance accuracy while using fewer parameters and pre-training tokens. Utilizes a layer-wise scaling strategy to allocate smaller dimensions in early layers, expanding in later layers.",
    },
    {
      title: "H2O Danube2 1.8B",
      link: "https://ritvik19.medium.com/papers-explained-111-h2o-danube-1-8b-b790c073d257#00d8",
      date: "April 2024",
      description:
        "An updated version of the original H2O-Danube model, with improvements including removal of sliding window attention, changes to the tokenizer, and adjustments to the training data, resulting in significant performance enhancements.",
    },
    {
      title: "Granite Code Models",
      link: "https://ritvik19.medium.com/paper-explained-144-granite-code-models-e1a92678739b",
      date: "May 2024",
      description:
        "A family of code models ranging from 3B to 34B trained on 3.5-4.5T tokens of code written in 116 programming languages.",
    },
    {
      title: "Gemma 2",
      link: "https://ritvik19.medium.com/papers-explained-157-gemma-2-f1b75b56b9f2",
      date: "June 2024",
      description:
        "Utilizes interleaving local-global attentions and group-query attention, trained with knowledge distillation instead of next token prediction to achieve competitive performance comparable with larger models.",
    },
    {
      title: "Orca 3 (Agent Instruct)",
      link: "https://ritvik19.medium.com/papers-explained-164-orca-3-agent-instruct-41340505af36",
      date: "July 2024",
      description:
        "A fine tuned Mistral-7B through Generative Teaching via synthetic data generated through the proposed AgentInstruct framework, which generates both the prompts and responses, using only raw data sources like text documents and code files as seeds.",
    },
    {
      title: "Mathstral",
      link: "https://ritvik19.medium.com/papers-explained-mistral-7b-b9632dedf580#0fbe",
      date: "July 2024",
      description:
        "A 7B model designed for math reasoning and scientific discovery based on Mistral 7B specializing in STEM subjects.",
    },
    {
      title: "Smol LM",
      link: "https://ritvik19.medium.com/papers-explained-176-smol-lm-a166d5f1facc",
      date: "July 2024",
      description:
        "A family of small models with 135M, 360M, and 1.7B parameters, utilizes Grouped-Query Attention (GQA), embedding tying, and a context length of 2048 tokens, trained on a new open source high-quality dataset.",
    },
    {
      title: "Mistral Nemo",
      link: "https://ritvik19.medium.com/papers-explained-mistral-7b-b9632dedf580#37cd",
      date: "July 2024",
      description:
        "A 12B Language Model built in collaboration between Mistral and NVIDIA, featuring a context window of 128K, an efficient tokenizer and trained with quantization awareness, enabling FP8 inference without any performance loss.",
    },
    {
      title: "Smol LM v0.2",
      link: "https://ritvik19.medium.com/papers-explained-176-smol-lm-a166d5f1facc#fdb2",
      date: "August 2024",
      description:
        "An advancement over SmolLM, better at staying on topic and responding appropriately to standard prompts, such as greetings and questions about their role as AI assistants.",
    },
    {
      title: "Phi-3.5",
      link: "https://ritvik19.medium.com/papers-explained-192-phi-3-5-a95429ea26c9",
      date: "August 2024",
      description:
        "A family of models consisting of three variants - MoE (16x3.8B), mini (3.8B), and vision (4.2B) - which are lightweight, multilingual, and trained on synthetic and filtered publicly available documents - with a focus on very high-quality, reasoning dense data.",
    },
  ],
  [
    // Multi Modal Language Models
    {
      title: "BLIP",
      link: "https://ritvik19.medium.com/papers-explained-154-blip-6d85c80a744d",
      date: "February 2022",
      description:
        "A Vision-Language Pre-training (VLP) framework that introduces Multimodal mixture of Encoder-Decoder (MED) and Captioning and Filtering (CapFilt), a new dataset bootstrapping method for learning from noisy image-text pairs.",
    },
    {
      title: "Flamingo",
      link: "https://ritvik19.medium.com/papers-explained-82-flamingo-8c124c394cdb",
      date: "April 2022",
      description:
        "Visual Language Models enabling seamless handling of interleaved visual and textual data, and facilitating few-shot learning on large-scale web corpora.",
    },
    {
      title: "PaLI",
      link: "https://ritvik19.medium.com/papers-explained-194-pali-c1fffc14068c",
      date: "September 2022",
      description:
        "A joint language-vision model that generates multilingual text based on visual and textual inputs, trained using large pre-trained encoder-decoder language models and Vision Transformers, specifically mT5 and ViT-e.",
    },
    {
      title: "BLIP 2",
      link: "https://ritvik19.medium.com/papers-explained-155-blip-2-135fff70bf65",
      date: "January 2023",
      description:
        "A Vision-Language Pre-training (VLP) framework that proposes Q-Former, a trainable module to bridge the gap between a frozen image encoder and a frozen LLM to bootstrap vision-language pre-training.",
    },
    {
      title: "LLaVA 1",
      link: "https://ritvik19.medium.com/papers-explained-102-llava-1-eb0a3db7e43c",
      date: "April 2023",
      description:
        "A large multimodal model connecting CLIP and Vicuna trained end-to-end on instruction-following data generated through GPT-4 from image-text pairs.",
    },
    {
      title: "PaLI-X",
      link: "https://ritvik19.medium.com/papers-explained-195-pali-x-f9859e73fd97",
      date: "May 2023",
      description: "A multilingual vision and language model with scaled-up components, specifically ViT-22 B and UL2 32B, exhibits emergent properties such as complex counting and multilingual object detection, and demonstrates improved performance across various tasks."
    },
    {
      title: "InstructBLIP",
      link: "https://ritvik19.medium.com/papers-explained-156-instructblip-c3cf3291a823",
      date: "May 2023",
      description:
        "Introduces instruction-aware Query Transformer to extract informative features tailored to the given instruction to study vision-language instruction tuning based on the pretrained BLIP-2 models.",
    },
    {
      title: "Idefics",
      link: "https://ritvik19.medium.com/papers-explained-179-obelics-idefics-a581f8d909b6",
      date: "June 2023",
      description:
        "9B and 80B multimodal models trained on Obelics, an open web-scale dataset of interleaved image-text documents, curated in this work.",
    },
    {
      title: "GPT-4V",
      link: "https://ritvik19.medium.com/papers-explained-68-gpt-4v-6e27c8a1d6ea",
      date: "September 2023",
      description:
        "A multimodal model that combines text and vision capabilities, allowing users to instruct it to analyze image inputs.",
    },
    {
      title: "PaLI-3",
      link: "https://ritvik19.medium.com/papers-explained-196-pali-3-2f5cf92f60a8",
      date: "October 2023",
      description:
        "A 5B vision language model, built upon a 2B SigLIP Vision Model and UL2 3B Language Model outperforms larger models on various benchmarks and achieves SOTA on several video QA benchmarks despite not being pretrained on any video data.",
    },
    {
      title: "LLaVA 1.5",
      link: "https://ritvik19.medium.com/papers-explained-103-llava-1-5-ddcb2e7f95b4",
      date: "October 2023",
      description:
        "An enhanced version of the LLaVA model that incorporates a CLIP-ViT-L-336px with an MLP projection and academic-task-oriented VQA data to set new benchmarks in large multimodal models (LMM) research.",
    },
    {
      title: "Gemini 1.0",
      link: "https://ritvik19.medium.com/papers-explained-80-gemini-1-0-97308ef96fcd",
      date: "December 2023",
      description:
        "A family of highly capable multi-modal models, trained jointly across image, audio, video, and text data for the purpose of building a model with strong generalist capabilities across modalities.",
    },
    {
      title: "MoE-LLaVA",
      link: "https://ritvik19.medium.com/papers-explained-104-moe-llava-cf14fda01e6f",
      date: "January 2024",
      description:
        "A MoE-based sparse LVLM framework that activates only the top-k experts through routers during deployment, maintaining computational efficiency while achieving comparable performance to larger models.",
    },
    {
      title: "LLaVA 1.6",
      link: "https://ritvik19.medium.com/papers-explained-107-llava-1-6-a312efd496c5",
      date: "January 2024",
      description:
        "An improved version of a LLaVA 1.5 with enhanced reasoning, OCR, and world knowledge capabilities, featuring increased image resolution",
    },
    {
      title: "Gemini 1.5 Pro",
      link: "https://ritvik19.medium.com/papers-explained-105-gemini-1-5-pro-029bbce3b067",
      date: "February 2024",
      description:
        "A highly compute-efficient multimodal mixture-of-experts model that excels in long-context retrieval tasks and understanding across text, video, and audio modalities.",
    },
    {
      title: "Claude 3",
      link: "https://ritvik19.medium.com/papers-explained-181-claude-89dd45e35d92",
      date: "March 2024",
      description:
        "A family of VLMs consisting of Haiku, Sonnet, and Opus models, sets new industry standards for cognitive tasks, offering varying levels of intelligence, speed, and cost-efficiency.",
    },
    {
      title: "MM1",
      link: "https://ritvik19.medium.com/papers-explained-117-mm1-c579142bcdc0",
      date: "March 2024",
      description:
        "A multimodal llm that combines a ViT-H image encoder with 378x378px resolution, pretrained on a data mix of image-text documents and text-only documents, scaled up to 3B, 7B, and 30B parameters for enhanced performance across various tasks.",
    },
    {
      title: "Grok 1.5 V",
      link: "https://ritvik19.medium.com/papers-explained-186-grok-0d9f1aef69be",
      date: "April 2024",
      description: "The first multimodal model in the grok series.",
    },
    {
      title: "Idefics2",
      link: "https://ritvik19.medium.com/papers-explained-180-idefics-2-0adf35cef4ee",
      date: "April 2024",
      description:
        "Improvement upon Idefics1 with enhanced OCR capabilities, simplified architecture, and better pre-trained backbones, trained on a mixture of openly available datasets and fine-tuned on task-oriented data.",
    },
    {
      title: "Phi-3 Vision",
      link: "https://ritvik19.medium.com/papers-explained-130-phi-3-0dfc951dc404#7ba6",
      date: "May 2024",
      description:
        "First multimodal model in the Phi family, bringing the ability to reason over images and extract and reason over text from images.",
    },
    {
      title: "An Introduction to Vision-Language Modeling",
      link: "https://ritvik19.medium.com/papers-explained-an-introduction-to-vision-language-modeling-89e7697da6e3",
      date: "May 2024",
      description:
        "Provides a comprehensive introduction to VLMs, covering their definition, functionality, training methods, and evaluation approaches, aiming to help researchers and practitioners enter the field and advance the development of VLMs for various applications.",
    },
    {
      title: "GPT-4o",
      link: "https://ritvik19.medium.com/papers-explained-185-gpt-4o-a234bccfd662",
      date: "May 2024",
      description:
        "An omni model accepting and generating various types of inputs and outputs, including text, audio, images, and video.",
    },
    {
      title: "Gemini 1.5 Flash",
      link: "https://ritvik19.medium.com/papers-explained-142-gemini-1-5-flash-415e2dc6a989",
      date: "May 2024",
      description:
        "A more lightweight variant of the Gemini 1.5 pro, designed for efficiency with minimal regression in quality, making it suitable for applications where compute resources are limited.",
    },
    {
      title: "Chameleon",
      link: "https://ritvik19.medium.com/papers-explained-143-chameleon-6cddfdbceaa8",
      date: "May 2024",
      description:
        "A family of early-fusion token-based mixed-modal models capable of understanding and generating images and text in any arbitrary sequence.",
    },
    {
      title: "Claude 3.5 Sonnet",
      link: "https://ritvik19.medium.com/papers-explained-181-claude-89dd45e35d92#2a14",
      date: "June 2024",
      description:
        "Surpasses previous versions and competitors in intelligence, speed, and cost-efficiency, excelling in graduate-level reasoning, undergraduate-level knowledge, coding proficiency, and visual reasoning.",
    },
    {
      title: "Pali Gemma",
      link: "https://ritvik19.medium.com/papers-explained-197-pali-gemma-6899e871998e",
      date: "July 2024",
      description:
        "Combines SigLIP vision model and the Gemma language model and follows the PaLI-3 training recipe to achieve strong performance on various vision-language tasks.",
    },
    {
      title: "GPT-4o mini",
      link: "https://ritvik19.medium.com/papers-explained-185-gpt-4o-a234bccfd662#08b9",
      date: "July 2024",
      description:
        "A cost-efficient small model that outperforms GPT-4 on chat preferences, enabling a broad range of tasks with low latency and supporting text, vision, and multimodal inputs and outputs.",
    },
    {
      title: "Grok 2",
      link: "https://ritvik19.medium.com/papers-explained-186-grok-0d9f1aef69be",
      date: "August 2024",
      description:
        "A frontier language model with state-of-the-art capabilities in chat, coding, and reasoning on par with Claude 3.5 Sonnet and GPT-4-Turbo.",
    },
    {
      title: "BLIP-3 (xGen-MM)",
      link: "https://ritvik19.medium.com/papers-explained-190-blip-3-xgen-mm-6a9c04a3892d",
      date: "August 2024",
      description:
        "A comprehensive system for developing Large Multimodal Models, comprising curated datasets, training recipes, model architectures, and pre-trained models that demonstrate strong in-context learning capabilities and competitive performance on various tasks.",
    },
  ],
  [
    // Retrieval and Representation Learning
    {
      title: "Dense Passage Retriever",
      link: "https://ritvik19.medium.com/papers-explained-86-dense-passage-retriever-c4742fdf27ed",
      date: "April 2020",
      description:
        "Shows that retrieval can be practically implemented using dense representations alone, where embeddings are learned from a small number of questions and passages by a simple dual encoder framework.",
    },
    {
      title: "ColBERT",
      link: "https://medium.com/@ritvik19/papers-explained-88-colbert-fe2fd0509649",
      date: "April 2020",
      description:
        "Introduces a late interaction architecture that adapts deep LMs (in particular, BERT) for efficient retrieval.",
    },
    {
      title: "CLIP",
      link: "https://ritvik19.medium.com/papers-explained-100-clip-f9873c65134",
      date: "February 2021",
      description:
        "A vision system that learns image representations from raw text-image pairs through pre-training, enabling zero-shot transfer to various downstream tasks.",
    },
    {
      title: "ColBERTv2",
      link: "https://ritvik19.medium.com/papers-explained-89-colbertv2-7d921ee6e0d9",
      date: "December 2021",
      description:
        "Couples an aggressive residual compression mechanism with a denoised supervision strategy to simultaneously improve the quality and space footprint of late interaction.",
    },
    {
      title: "Matryoshka Representation Learning",
      link: "https://ritvik19.medium.com/papers-explained-matryoshka-representation-learning-e7a139f6ad27",
      date: "May 2022",
      description:
        "Encodes information at different granularities and allows a flexible representation that can adapt to multiple downstream tasks with varying computational resources using a single embedding.",
    },
    {
      title: "E5",
      link: "https://ritvik19.medium.com/papers-explained-90-e5-75ea1519efad",
      date: "December 2022",
      description:
        "A family of text embeddings trained in a contrastive manner with weak supervision signals from a curated large-scale text pair dataset CCPairs.",
    },
    {
      title: "SigLip",
      link: "https://ritvik19.medium.com/papers-explained-152-siglip-011c48f9d448",
      date: "March 2023",
      description:
        "A simple pairwise Sigmoid loss function for Language-Image Pre-training that operates solely on image-text pairs, allowing for larger batch sizes and better performance at smaller batch sizes.",
    },
    {
      title: "E5 Mistral 7B",
      link: "https://ritvik19.medium.com/papers-explained-91-e5-mistral-7b-23890f40f83a",
      date: "December 2023",
      description:
        "Leverages proprietary LLMs to generate diverse synthetic data to fine tune open-source decoder-only LLMs for hundreds of thousands of text embedding tasks.",
    },
    {
      title: "Nomic Embed Text v1",
      link: "https://ritvik19.medium.com/papers-explained-110-nomic-embed-8ccae819dac2",
      date: "February 2024",
      description:
        "A 137M parameter, open-source English text embedding model with an 8192 context length that outperforms OpenAI's models on both short and long-context tasks.",
    },
    {
      title: "Nomic Embed Text v1.5",
      link: "https://ritvik19.medium.com/papers-explained-110-nomic-embed-8ccae819dac2#2119",
      date: "February 2024",
      description:
        "An advanced text embedding model that utilizes Matryoshka Representation Learning to offer flexible embedding sizes with minimal performance trade-offs",
    },
    {
      title: "NV Embed",
      link: "https://ritvik19.medium.com/papers-explained-168-nv-embed-48bd25d83258",
      date: "May 2024",
      description:
        "Introduces architectural innovations and training recipe to significantly enhance LLMs performance in general-purpose text embedding tasks.",
    },
    {
      title: "Nomic Embed Vision v1 and v1.5",
      link: "https://ritvik19.medium.com/papers-explained-110-nomic-embed-8ccae819dac2#486b",
      date: "June 2024",
      description:
        "Aligns a Vision Encoder with the existing text encoders without destroying the downstream performance of the text encoder, to attain a unified multimodal latent space.",
    },
    {
      title: "ColPali",
      link: "https://ritvik19.medium.com/papers-explained-198-colpali-b3be70cbe252",
      date: "June 2024",
      description:
        "A retrieval model based on PaliGemma to produce high-quality contextualized embeddings solely from images of document pages, and employees late interaction allowing for efficient and effective visually rich document retrieval.",
    },
    {
      title: "E5-V",
      link: "https://ritvik19.medium.com/papers-explained-172-e5-v-9947d3925802",
      date: "July 2024",
      description:
        "A framework that adapts Multimodal Large Language Models for achieving universal multimodal embeddings by leveraging prompts and single modality training on text pairs, which demonstrates strong performance in multimodal embeddings without fine-tuning and eliminates the need for costly multimodal training data collection.",
    },
  ],
  [
    // Parameter Efficient Fine Tuning
    {
      title: "LoRA",
      link: "https://ritvik19.medium.com/papers-explained-lora-a48359cecbfa",
      date: "July 2021",
      description:
        "Introduces trainable rank decomposition matrices into each layer of a pre-trained Transformer model, significantly reducing the number of trainable parameters for downstream tasks.",
    },
    {
      title: "QLoRA",
      link: "https://ritvik19.medium.com/papers-explained-146-qlora-a6e7273bc630",
      date: "May 2023",
      description:
        "Allows efficient training of large models on limited GPU memory, through innovations like 4-bit NormalFloat (NF4), double quantization and paged optimisers.",
    },
    {
      title: "LongLoRA",
      link: "https://ritvik19.medium.com/papers-explained-147-longlora-24f095b93611",
      date: "September 2023",
      description:
        "Enables context extension for large language models, achieving significant computation savings through sparse local attention and parameter-efficient fine-tuning.",
    },
  ],
  [
    // LLM Evaluation
    {
      title: "Prometheus",
      link: "https://ritvik19.medium.com/papers-explained-170-prometheus-5e72b8054729",
      date: "October 2023",
      description:
        "A 13B fully open source evaluation LLM trained on Feedback Collection curated using GPT-4 (in this work).",
    },
    {
      title: "Prometheus 2",
      link: "https://ritvik19.medium.com/papers-explained-171-prometheus-2-324e9c162e18",
      date: "May 2024",
      description:
        "7B & 8x7B evaluation LLMs that score high correlations with both human evaluators and proprietary LM-based judges on both direct assessment and pairwise ranking, obtained by merging Mistral models trained on Feedback Collection and Preference Collection (curated in this work.",
    },
  ],
  [
    // Compression, Pruning, Quantization
    {
      title: "LLMLingua",
      link: "https://ritvik19.medium.com/papers-explained-136-llmlingua-f9b2f53f5f9b",
      date: "October 2023",
      description:
        "A novel coarse-to-fine prompt compression method, incorporating a budget controller, an iterative token-level compression algorithm, and distribution alignment, achieving up to 20x compression with minimal performance loss.",
    },
    {
      title: "LongLLMLingua",
      link: "https://ritvik19.medium.com/papers-explained-137-longllmlingua-45961fa703dd",
      date: "October 2023",
      description:
        "A novel approach for prompt compression to enhance performance in long context scenarios using question-aware compression and document reordering.",
    },
    {
      title: "LLMLingua2",
      link: "https://ritvik19.medium.com/papers-explained-138-llmlingua-2-510c752368a8",
      date: "March 2024",
      description:
        "A novel approach to task-agnostic prompt compression, aiming to enhance generalizability, using  data distillation and leveraging a Transformer encoder for token classification.",
    },
  ],
  [
    // Vision Transformers
    {
      title: "Vision Transformer",
      link: "https://ritvik19.medium.com/papers-explained-25-vision-transformers-e286ee8bc06b",
      date: "October 2020",
      description:
        "Images are segmented into patches, which are treated as tokens and a sequence of linear embeddings of these patches are input to a Transformer",
    },
    {
      title: "DeiT",
      link: "https://ritvik19.medium.com/papers-explained-39-deit-3d78dd98c8ec",
      date: "December 2020",
      description:
        "A convolution-free vision transformer that uses a teacher-student strategy with attention-based distillation tokens.",
    },
    {
      title: "Swin Transformer",
      link: "https://ritvik19.medium.com/papers-explained-26-swin-transformer-39cf88b00e3e",
      date: "March 2021",
      description:
        "A hierarchical vision transformer that uses shifted windows to addresses the challenges of adapting the transformer model to computer vision.",
    },
    {
      title: "Convolutional vision Transformer",
      link: "https://ritvik19.medium.com/papers-explained-199-cvt-fb4a5c05882e",
      date: "March 2021",
      description:
        "Improves Vision Transformer (ViT) in performance and efficiency by introducing convolutions, to yield the best of both designs.",
    },
    {
      title: "BEiT",
      link: "https://ritvik19.medium.com/papers-explained-27-beit-b8c225496c01",
      date: "June 2021",
      description:
        "Utilizes a masked image modeling task inspired by BERT in, involving image patches and visual tokens to pretrain vision Transformers.",
    },
    {
      title: "MobileViT",
      link: "https://ritvik19.medium.com/papers-explained-40-mobilevit-4793f149c434",
      date: "October 2021",
      description:
        "A lightweight vision transformer designed for mobile devices, effectively combining the strengths of CNNs and ViTs.",
    },
    {
      title: "Masked AutoEncoder",
      link: "https://ritvik19.medium.com/papers-explained-28-masked-autoencoder-38cb0dbed4af",
      date: "November 2021",
      description:
        "An encoder-decoder architecture that reconstructs input images by masking random patches and leveraging a high proportion of masking for self-supervision.",
    },
  ],
  [
    // Convolutional Neural Networks
    {
      title: "Lenet",
      link: "https://ritvik19.medium.com/papers-explained-review-01-convolutional-neural-networks-78aeff61dcb3#4f26",
      date: "December 1998",
      description: "Introduced Convolutions.",
    },
    {
      title: "Alex Net",
      link: "https://ritvik19.medium.com/papers-explained-review-01-convolutional-neural-networks-78aeff61dcb3#f7c6",
      date: "September 2012",
      description:
        "Introduced ReLU activation and Dropout to CNNs. Winner ILSVRC 2012.",
    },
    {
      title: "VGG",
      link: "https://ritvik19.medium.com/papers-explained-review-01-convolutional-neural-networks-78aeff61dcb3#c122",
      date: "September 2014",
      description:
        "Used large number of filters of small size in each layer to learn complex features. Achieved SOTA in ILSVRC 2014.",
    },
    {
      title: "Inception Net",
      link: "https://ritvik19.medium.com/papers-explained-review-01-convolutional-neural-networks-78aeff61dcb3#d7b3",
      date: "September 2014",
      description:
        "Introduced Inception Modules consisting of multiple parallel convolutional layers, designed to recognize different features at multiple scales.",
    },
    {
      title: "Inception Net v2 / Inception Net v3",
      link: "https://ritvik19.medium.com/papers-explained-review-01-convolutional-neural-networks-78aeff61dcb3#d7b3",
      date: "December 2015",
      description:
        "Design Optimizations of the Inception Modules which improved performance and accuracy.",
    },
    {
      title: "Res Net",
      link: "https://ritvik19.medium.com/papers-explained-review-01-convolutional-neural-networks-78aeff61dcb3#f761",
      date: "December 2015",
      description:
        "Introduced residual connections, which are shortcuts that bypass one or more layers in the network. Winner ILSVRC 2015.",
    },
    {
      title: "Inception Net v4 / Inception ResNet",
      link: "https://ritvik19.medium.com/papers-explained-review-01-convolutional-neural-networks-78aeff61dcb3#83ad",
      date: "February 2016",
      description: "Hybrid approach combining Inception Net and ResNet.",
    },
    {
      title: "Dense Net",
      link: "https://ritvik19.medium.com/papers-explained-review-01-convolutional-neural-networks-78aeff61dcb3#65e8",
      date: "August 2016",
      description:
        "Each layer receives input from all the previous layers, creating a dense network of connections between the layers, allowing to learn more diverse features.",
    },
    {
      title: "Xception",
      link: "https://ritvik19.medium.com/papers-explained-review-01-convolutional-neural-networks-78aeff61dcb3#bc70",
      date: "October 2016",
      description:
        "Based on InceptionV3 but uses depthwise separable convolutions instead on inception modules.",
    },
    {
      title: "Res Next",
      link: "https://ritvik19.medium.com/papers-explained-review-01-convolutional-neural-networks-78aeff61dcb3#90bd",
      date: "November 2016",
      description:
        "Built over ResNet, introduces the concept of grouped convolutions, where the filters in a convolutional layer are divided into multiple groups.",
    },
    {
      title: "Mobile Net V1",
      link: "https://ritvik19.medium.com/papers-explained-review-01-convolutional-neural-networks-78aeff61dcb3#3cb5",
      date: "April 2017",
      description:
        "Uses depthwise separable convolutions to reduce the number of parameters and computation required.",
    },
    {
      title: "Mobile Net V2",
      link: "https://ritvik19.medium.com/papers-explained-review-01-convolutional-neural-networks-78aeff61dcb3#4440",
      date: "January 2018",
      description:
        "Built upon the MobileNetv1 architecture, uses inverted residuals and linear bottlenecks.",
    },
    {
      title: "Mobile Net V3",
      link: "https://ritvik19.medium.com/papers-explained-review-01-convolutional-neural-networks-78aeff61dcb3#8eb6",
      date: "May 2019",
      description:
        "Uses AutoML to find the best possible neural network architecture for a given problem.",
    },
    {
      title: "Efficient Net",
      link: "https://ritvik19.medium.com/papers-explained-review-01-convolutional-neural-networks-78aeff61dcb3#560a",
      date: "May 2019",
      description:
        "Uses a compound scaling method to scale the network's depth, width, and resolution to achieve a high accuracy with a relatively low computational cost.",
    },
    {
      title: "NF Net",
      link: "https://ritvik19.medium.com/papers-explained-84-nf-net-b8efa03d6b26",
      date: "February 2021",
      description:
        "An improved class of Normalizer-Free ResNets that implement batch-normalized networks, offer faster training times, and introduce an adaptive gradient clipping technique to overcome instabilities associated with deep ResNets.",
    },
    {
      title: "Conv Mixer",
      link: "https://ritvik19.medium.com/papers-explained-29-convmixer-f073f0356526",
      date: "January 2022",
      description:
        "Processes image patches using standard convolutions for mixing spatial and channel dimensions.",
    },
    {
      title: "ConvNeXt",
      link: "https://ritvik19.medium.com/papers-explained-92-convnext-d13385d9177d",
      date: "January 2022",
      description:
        "A pure ConvNet model, evolved from standard ResNet design, that competes well with Transformers in accuracy and scalability.",
    },
    {
      title: "ConvNeXt V2",
      link: "https://ritvik19.medium.com/papers-explained-94-convnext-v2-2ecdabf2081c",
      date: "January 2023",
      description:
        "Incorporates a fully convolutional MAE framework and a Global Response Normalization (GRN) layer, boosting performance across multiple benchmarks.",
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
    },
    {
      title: "Feature Pyramid Network",
      link: "https://ritvik19.medium.com/papers-explained-21-feature-pyramid-network-6baebcb7e4b8",
      date: "December 2016",
      description:
        "Leverages the inherent multi-scale hierarchy of deep convolutional networks to efficiently construct feature pyramids.",
    },
    {
      title: "Focal Loss",
      link: "https://ritvik19.medium.com/papers-explained-22-focal-loss-for-dense-object-detection-retinanet-733b70ce0cb1",
      date: "August 2017",
      description:
        "Addresses class imbalance in dense object detectors by down-weighting the loss assigned to well-classified examples.",
    },
    {
      title: "DETR",
      link: "https://ritvik19.medium.com/papers-explained-79-detr-bcdd53355d9f",
      date: "May 2020",
      description:
        "A novel transformers based object detection model that treats object detection as a set prediction problem, eliminating the need for hand-designed components.",
    },
  ],
  [
    // Region Based Convolutional Neural Networks
    {
      title: "RCNN",
      link: "https://ritvik19.medium.com/papers-explained-14-rcnn-ede4db2de0ab",
      date: "November 2013",
      description:
        "Uses selective search for region proposals, CNNs for feature extraction, SVM for classification followed by box offset regression.",
    },
    {
      title: "Fast RCNN",
      link: "https://ritvik19.medium.com/papers-explained-15-fast-rcnn-28c1792dcee0",
      date: "April 2015",
      description:
        "Processes entire image through CNN, employs RoI Pooling to extract feature vectors from ROIs, followed by classification and BBox regression.",
    },
    {
      title: "Faster RCNN",
      link: "https://ritvik19.medium.com/papers-explained-16-faster-rcnn-a7b874ffacd9",
      date: "June 2015",
      description:
        "A region proposal network (RPN) and a Fast R-CNN detector, collaboratively predict object regions by sharing convolutional features.",
    },
    {
      title: "Mask RCNN",
      link: "https://ritvik19.medium.com/papers-explained-17-mask-rcnn-82c64bea5261",
      date: "March 2017",
      description:
        "Extends Faster R-CNN to solve instance segmentation tasks, by adding a branch for predicting an object mask in parallel with the existing branch.",
    },
    {
      title: "Cascade RCNN",
      link: "https://ritvik19.medium.com/papers-explained-77-cascade-rcnn-720b161d86e4",
      date: "December 2017",
      description:
        "Proposes a multi-stage approach where detectors are trained with progressively higher IoU thresholds, improving selectivity against false positives.",
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
    },
    {
      title: "Donut",
      link: "https://ritvik19.medium.com/papers-explained-20-donut-cb1523bf3281",
      date: "November 2021",
      description:
        "An OCR-free Encoder-Decoder Transformer model. The encoder takes in images, decoder takes in prompts & encoded images to generate the required text.",
    },
    {
      title: "DiT",
      link: "https://ritvik19.medium.com/papers-explained-19-dit-b6d6eccd8c4e",
      date: "March 2022",
      description:
        "An Image Transformer pre-trained (self-supervised) on document images",
    },
    {
      title: "UDoP",
      link: "https://ritvik19.medium.com/papers-explained-42-udop-719732358ab4",
      date: "December 2022",
      description:
        "Integrates text, image, and layout information through a Vision-Text-Layout Transformer, enabling unified representation.",
    },
    {
      title: "DocLLM",
      link: "https://ritvik19.medium.com/papers-explained-87-docllm-93c188edfaef",
      date: "January 2024",
      description:
        "A lightweight extension to traditional LLMs that focuses on reasoning over visual documents, by incorporating textual semantics and spatial layout without expensive image encoders.",
    },
  ],
  [
    // Layout Aware Language Models
    {
      title: "Layout LM",
      link: "https://ritvik19.medium.com/papers-explained-10-layout-lm-32ec4bad6406",
      date: "December 2019",
      description:
        "Utilises BERT as the backbone, adds two new input embeddings: 2-D position embedding and image embedding (Only for downstream tasks).",
    },
    {
      title: "LamBERT",
      link: "https://ritvik19.medium.com/papers-explained-41-lambert-8f52d28f20d9",
      date: "February 2020",
      description:
        "Utilises RoBERTa as the backbone and adds Layout embeddings along with relative bias.",
    },
    {
      title: "Layout LM v2",
      link: "https://ritvik19.medium.com/papers-explained-11-layout-lm-v2-9531a983e659",
      date: "December 2020",
      description:
        "Uses a multi-modal Transformer model, to integrate text, layout, and image in the pre-training stage, to learn end-to-end cross-modal interaction.",
    },
    {
      title: "Structural LM",
      link: "https://ritvik19.medium.com/papers-explained-23-structural-lm-36e9df91e7c1",
      date: "May 2021",
      description:
        "Utilises BERT as the backbone and feeds text, 1D and (2D cell level) embeddings to the transformer model.",
    },
    {
      title: "Doc Former",
      link: "https://ritvik19.medium.com/papers-explained-30-docformer-228ce27182a0",
      date: "June 2021",
      description:
        "Encoder-only transformer with a CNN backbone for visual feature extraction, combines text, vision, and spatial features through a multi-modal self-attention layer.",
    },
    {
      title: "LiLT",
      link: "https://ritvik19.medium.com/papers-explained-12-lilt-701057ec6d9e",
      date: "February 2022",
      description:
        "Introduced Bi-directional attention complementation mechanism (BiACM) to accomplish the cross-modal interaction of text and layout.",
    },
    {
      title: "Layout LM V3",
      link: "https://ritvik19.medium.com/papers-explained-13-layout-lm-v3-3b54910173aa",
      date: "April 2022",
      description:
        "A unified text-image multimodal Transformer to learn cross-modal representations, that imputs concatenation of text embedding and image embedding.",
    },
    {
      title: "ERNIE Layout",
      link: "https://ritvik19.medium.com/papers-explained-24-ernie-layout-47a5a38e321b",
      date: "October 2022",
      description:
        "Reorganizes tokens using layout information, combines text and visual embeddings, utilizes multi-modal transformers with spatial aware disentangled attention.",
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
    },
    {
      title: "Conditional Generative Adversarial Networks",
      link: "https://ritvik19.medium.com/papers-explained-review-05-generative-adversarial-networks-bbb51b160d5e#86aa",
      date: "November 2014",
      description:
        "A method for training GANs, enabling the generation based on specific conditions, by feeding them to both the generator and discriminator networks.",
    },
    {
      title: "Deep Convolutional Generative Adversarial Networks",
      link: "https://ritvik19.medium.com/papers-explained-review-05-generative-adversarial-networks-bbb51b160d5e#fe42",
      date: "November 2015",
      description:
        "Demonstrates the ability of CNNs for unsupervised learning using specific architectural constraints designed.",
    },
    {
      title: "Improved GAN",
      link: "https://ritvik19.medium.com/papers-explained-review-05-generative-adversarial-networks-bbb51b160d5e#9a55",
      date: "June 2016",
      description:
        "Presents a variety of new architectural features and training procedures that can be applied to the generative adversarial networks (GANs) framework.",
    },
    {
      title: "Wasserstein Generative Adversarial Networks",
      link: "https://ritvik19.medium.com/papers-explained-review-05-generative-adversarial-networks-bbb51b160d5e#6f8f",
      date: "January 2017",
      description:
        "An alternative GAN training algorithm that enhances learning stability, mitigates issues like mode collapse.",
    },
    {
      title: "Cycle GAN",
      link: "https://ritvik19.medium.com/papers-explained-review-05-generative-adversarial-networks-bbb51b160d5e#7f8b",
      date: "March 2017",
      description:
        "An approach for learning to translate an image from a source domain X to a target domain Y in the absence of paired examples by leveraging adversarial losses and cycle consistency constraints, using two GANs",
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
    },
    {
      title: "Wide and Deep Learning",
      link: "https://ritvik19.medium.com/papers-explained-review-04-tabular-deep-learning-776db04f965b#bfdc",
      date: "June 2016",
      description:
        "Combines memorization of specific patterns with generalization of similarities.",
    },
    {
      title: "Deep and Cross Network",
      link: "https://ritvik19.medium.com/papers-explained-review-04-tabular-deep-learning-776db04f965b#0017",
      date: "August 2017",
      description:
        "Combines the  a novel cross network with deep neural networks (DNNs) to efficiently learn feature interactions without manual feature engineering.",
    },
    {
      title: "Tab Transformer",
      link: "https://ritvik19.medium.com/papers-explained-review-04-tabular-deep-learning-776db04f965b#48c4",
      date: "December 2020",
      description:
        "Employs multi-head attention-based Transformer layers to convert categorical feature embeddings into robust contextual embeddings.",
    },
    {
      title: "Tabular ResNet",
      link: "https://ritvik19.medium.com/papers-explained-review-04-tabular-deep-learning-776db04f965b#46af",
      date: "June 2021",
      description: "An MLP with skip connections.",
    },
    {
      title: "Feature Tokenizer Transformer",
      link: "https://ritvik19.medium.com/papers-explained-review-04-tabular-deep-learning-776db04f965b#1ab8",
      date: "June 2021",
      description:
        "Transforms all features (categorical and numerical) to embeddings and applies a stack of Transformer layers to the embeddings.",
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
    },
    {
      title: "Dolma",
      link: "https://ritvik19.medium.com/papers-explained-97-dolma-a656169269cb",
      date: "January 2024",
      description:
        "An open corpus of three trillion tokens designed to support language model pretraining research.",
    },
    {
      title: "Aya Dataset",
      link: "https://ritvik19.medium.com/papers-explained-108-aya-dataset-9e299ac74a19",
      date: "February 2024",
      description:
        "A human-curated instruction-following dataset that spans 65 languages, created to bridge the language gap in datasets for natural language processing.",
    },
    {
      title: "WebSight",
      link: "https://ritvik19.medium.com/papers-explained-177-websight-2905d0e14233",
      date: "March 2024",
      description:
        "A synthetic dataset consisting of 2M pairs of HTML codes and their corresponding screenshots, generated through LLMs, aimed to accelerate research for converting a screenshot into a corresponding HTML. ",
    },
    {
      title: "Cosmopedia",
      link: "https://ritvik19.medium.com/papers-explained-175-cosmopedia-5f7e81c76d14",
      date: "March 2024",
      description:
        "Synthetic Data containing over 30M files and 25B tokens, generated by Mixtral-8x7B-Instruct-v0., aimed to reproduce the training data for Phi-1.5.",
    },
    {
      title: "Fine Web",
      link: "https://ritvik19.medium.com/papers-explained-174-fineweb-280bbc08068b",
      date: "May 2024",
      description:
        "A large-scale dataset for pretraining LLMs, consisting of 15T tokens, shown to produce better-performing models than other open pretraining datasets.",
    },
    {
      title: "Cosmopedia v2",
      link: "https://ritvik19.medium.com/papers-explained-175-cosmopedia-5f7e81c76d14#5bab",
      date: "July 2024",
      description:
        "An enhanced version of Cosmopedia, with a lot of emphasis on prompt optimization.",
    },
    {
      title: "Docmatix",
      link: "https://ritvik19.medium.com/papers-explained-178-docmatix-9f2731ff1654",
      date: "July 2024",
      description:
        "A massive dataset for DocVQA containing 2.4M images, 9.5M question-answer pairs, and 1.3M PDF documents, generated by taking transcriptions from the PDFA OCR dataset and using a Phi-3-small model to generate Q/A pairs. ",
    },
  ],
  [
    // Miscellaneous Studies
    {
      title: "ColD Fusion",
      link: "https://ritvik19.medium.com/papers-explained-32-cold-fusion-452f33101a91",
      date: "December 2022",
      description:
        "A method enabling the benefits of multitask learning through distributed computation without data sharing and improving model performance.",
    },
    {
      title: "Are Emergent Abilities of Large Language Models a Mirage?",
      link: "https://ritvik19.medium.com/papers-explained-are-emergent-abilities-of-large-language-models-a-mirage-4160cf0e44cb",
      date: "April 2023",
      description:
        "Presents an alternative explanation for emergent abilities, i.e. emergent abilities are created by the researcher’s choice of metrics, not fundamental changes in model family behaviour on specific tasks with scale.",
    },
    {
      title: "Scaling Data-Constrained Language Models",
      link: "https://ritvik19.medium.com/papers-explained-85-scaling-data-constrained-language-models-2a4c18bcc7d3",
      date: "May 2023",
      description:
        "This study investigates scaling language models in data-constrained regimes.",
    },
    {
      title: "DSPy",
      link: "https://ritvik19.medium.com/papers-explained-135-dspy-fe8af7e35091",
      date: "October 2023",
      description:
        "A programming model that abstracts LM pipelines as text transformation graphs, i.e. imperative computation graphs where LMs are invoked through declarative modules, optimizing their use through a structured framework of signatures, modules, and teleprompters to automate and enhance text transformation tasks.",
    },
    {
      title: "An In-depth Look at Gemini's Language Abilities",
      link: "https://ritvik19.medium.com/papers-explained-81-an-in-depth-look-at-geminis-language-abilities-540ca9046d8e",
      date: "December 2023",
      description:
        "A third-party, objective comparison of the abilities of the OpenAI GPT and Google Gemini models with reproducible code and fully transparent results.",
    },
    {
      title: "Direct Preference Optimization",
      link: "https://ritvik19.medium.com/papers-explained-148-direct-preference-optimization-d3e031a41be1",
      date: "December 2023",
      description:
        "A stable, performant, and computationally lightweight algorithm that fine-tunes llms to align with human preferences without the need for reinforcement learning, by directly optimizing for the policy best satisfying the preferences with a simple classification objective.",
    },
    {
      title: "RLHF Workflow",
      link: "https://ritvik19.medium.com/papers-explained-149-rlhf-workflow-56b4e00019ed",
      date: "May 2024",
      description:
        "Provides a detailed recipe for  online iterative RLHF and achieves state-of-the-art performance on various benchmarks using fully open-source datasets.",
    },
    {
      title: "Monte Carlo Tree Self-refine",
      link: "https://ritvik19.medium.com/papers-explained-167-monte-carlo-tree-self-refine-79bffb070c1a",
      date: "June 2024",
      description:
        "Integrates LLMs with Monte Carlo Tree Search to enhance performance in complex mathematical reasoning tasks, leveraging systematic exploration and heuristic self-refine mechanisms to improve decision-making frameworks.",
    },
    {
      title: "Magpie",
      link: "https://ritvik19.medium.com/papers-explained-183-magpie-0603cbdc69c3",
      date: "June 2024",
      description:
        "A self-synthesis method that extracts high-quality instruction data at scale by prompting an aligned LLM with left-side templates, generating 4M instructions and their corresponding responses.",
    },
    {
      title: "Instruction Pre-Training",
      link: "https://ritvik19.medium.com/papers-explained-184-instruction-pretraining-ee0466f0fd33",
      date: "June 2024",
      description:
        "A framework to augment massive raw corpora with instruction-response pairs enabling supervised multitask pretraining of LMs.",
    },
    {
      title: "Proofread",
      link: "https://ritvik19.medium.com/papers-explained-189-proofread-4e1fe4eccf01",
      date: "June 2024",
      description:
        "A Gboard feature powered by a server-side LLM, enabling seamless sentence-level and paragraph-level corrections with a single tap.",
    },
  ],
];

const literature_review_data = [
  {
    title: "Convolutional Neural Networks",
    link: "https://ritvik19.medium.com/papers-explained-review-01-convolutional-neural-networks-78aeff61dcb3",
  },
  {
    title: "Layout Transformers",
    link: "https://ritvik19.medium.com/papers-explained-review-02-layout-transformers-b2d165c94ad5",
  },
  {
    title: "Region Based Convolutional Neural Networks",
    link: "https://ritvik19.medium.com/papers-explained-review-03-rcnns-42c0a3974493",
  },
  {
    title: "Tabular Deep Learning",
    link: "https://ritvik19.medium.com/papers-explained-review-04-tabular-deep-learning-776db04f965b",
  },
  {
    title: "Generative Adversarial Networks",
    link: "https://ritvik19.medium.com/papers-explained-review-05-generative-adversarial-networks-bbb51b160d5e",
  },
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
