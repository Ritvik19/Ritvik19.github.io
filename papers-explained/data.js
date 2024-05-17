const nav_data = [
  "Language Models",
  "Encoder Only Transformers",
  "Decoder Only Transformers",
  "Small LLMs",
  "Multi Modal LMs",
  "LMs for Retrieval",
  "Representation Learning",
  "Compression, Pruning, Quantization",
  "Vision Transformers",
  "CNNs",
  "Object Detection",
  "RCNNs",
  "Document Understanding",
  "Layout Aware LMs",
  "GANs",
  "Tabular Data",
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
      title: "Transformer XL",
      link: "https://ritvik19.medium.com/papers-explained-34-transformerxl-2e407e780e8",
      date: "January 2019",
      description:
        "Extends the original Transformer model to handle longer sequences of text by introducing recurrence into the self-attention mechanism.",
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
      title: "UniLMv2",
      link: "https://ritvik19.medium.com/papers-explained-unilmv2-5a044ca7c525",
      date: "February 2020",
      description:
        "Utilizes a pseudo-masked language model (PMLM) for both autoencoding and partially autoregressive language modeling tasks,significantly advancing the capabilities of language models in diverse NLP tasks.",
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
      title: "Rho-1",
      link: "https://ritvik19.medium.com/papers-explained-132-rho-1-788125e42241",
      date: "April 2024",
      description:
        "Introduces Selective Language Modelling that optimizes the loss only on tokens that align with a desired distribution, utilizing a reference model to score and select tokens.",
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
      title: "Phi-2",
      link: "https://ritvik19.medium.com/papers-explained-phi-1-5-2857e56dbd2a",
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
  ],
  [
    // Multi Modal Language Models
    {
      title: "Flamingo",
      link: "https://ritvik19.medium.com/papers-explained-82-flamingo-8c124c394cdb",
      date: "April 2022",
      description:
        "Visual Language Models enabling seamless handling of interleaved visual and textual data, and facilitating few-shot learning on large-scale web corpora.",
    },
    {
      title: "LLaVA 1",
      link: "https://ritvik19.medium.com/papers-explained-102-llava-1-eb0a3db7e43c",
      date: "April 2023",
      description:
        "A large multimodal model connecting CLIP and Vicuna trained end-to-end on instruction-following data generated through GPT-4 from image-text pairs.",
    },
    {
      title: "GPT-4V",
      link: "https://ritvik19.medium.com/papers-explained-68-gpt-4v-6e27c8a1d6ea",
      date: "September 2023",
      description:
        "A multimodal model that combines text and vision capabilities, allowing users to instruct it to analyze image inputs.",
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
      title: "MM1",
      link: "https://ritvik19.medium.com/papers-explained-117-mm1-c579142bcdc0",
      date: "March 2024",
      description:
        "A multimodal llm that combines a ViT-H image encoder with 378x378px resolution, pretrained on a data mix of image-text documents and text-only documents, scaled up to 3B, 7B, and 30B parameters for enhanced performance across various tasks",
    },
  ],
  [
    // Language Models for Retrieval
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
      title: "ColBERTv2",
      link: "https://ritvik19.medium.com/papers-explained-89-colbertv2-7d921ee6e0d9",
      date: "December 2021",
      description:
        "Couples an aggressive residual compression mechanism with a denoised supervision strategy to simultaneously improve the quality and space footprint of late interaction.",
    },
    {
      title: "E5",
      link: "https://ritvik19.medium.com/papers-explained-90-e5-75ea1519efad",
      date: "December 2022",
      description:
        "A family of text embeddings trained in a contrastive manner with weak supervision signals from a curated large-scale text pair dataset CCPairs.",
    },
    {
      title: "E5 Mistral 7B",
      link: "https://ritvik19.medium.com/papers-explained-91-e5-mistral-7b-23890f40f83a",
      date: "December 2023",
      description:
        "Leverages proprietary LLMs to generate diverse synthetic data to fine tune open-source decoder-only LLMs for hundreds of thousands of text embedding tasks.",
    },
  ],
  [
    // Representation Learning
    {
      title: "CLIP",
      link: "https://ritvik19.medium.com/papers-explained-100-clip-f9873c65134",
      date: "February 2021",
      description:
        "A vision system that learns image representations from raw text-image pairs through pre-training, enabling zero-shot transfer to various downstream tasks.",
    },
    {
      title: "Matryoshka Representation Learning",
      link: "https://ritvik19.medium.com/papers-explained-matryoshka-representation-learning-e7a139f6ad27",
      date: "May 2022",
      description:
        "Encodes information at different granularities and allows a flexible representation that can adapt to multiple downstream tasks with varying computational resources using a single embedding.",
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
    title: "Language Models for Retrieval",
    link: "https://ritvik19.medium.com/list/language-models-for-retrieval-3b6e14887105",
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
    title: "Representation Learning",
    link: "https://ritvik19.medium.com/list/representation-learning-bd438198713c",
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
