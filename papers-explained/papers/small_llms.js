const small_llms = [
  {
    title: "Phi-1",
    link: "14a8dcc77ce5",
    date: "June 2023",
    description:
      "An LLM for code, trained using a textbook quality data from the web and synthetically generated textbooks and exercises with GPT-3.5.",
    tags: [
      "Language Models",
      "Transformer Decoder",
      "Small LLMs",
      "Synthetic Data",
      "Phi",
    ],
  },
  {
    title: "Orca",
    link: "928eff06e7f9",
    date: "June 2023",
    description:
      "Presents a novel approach that addresses the limitations of instruction tuning by leveraging richer imitation signals, scaling tasks and instructions, and utilizing a teacher assistant to help with progressive learning.",
    tags: [
      "Language Models",
      "Transformer Decoder",
      "Small LLMs",
      "Synthetic Data",
      "Orca",
    ],
  },
  {
    title: "Phi-1.5",
    link: "2857e56dbd2a",
    date: "September 2023",
    description:
      "Follows the phi-1 approach, focusing this time on common sense reasoning in natural language.",
    tags: [
      "Language Models",
      "Transformer Decoder",
      "Small LLMs",
      "Synthetic Data",
      "Phi",
    ],
  },
  {
    title: "Mistral 7B",
    link: "b9632dedf580",
    date: "October 2023",
    description:
      "Leverages grouped-query attention for faster inference, coupled with sliding window attention to effectively handle sequences of arbitrary length with a reduced inference cost.",
    tags: ["Language Models", "Transformer Decoder", "Small LLMs", "Mistral"],
  },
  {
    title: "Zephyr 7B",
    link: "7ec068e2f20b",
    date: "October 2023",
    description:
      "Utilizes dDPO and AI Feedback (AIF) preference data to achieve superior intent alignment in chat-based language modeling.",
    tags: [
      "Language Models",
      "Transformer Decoder",
      "Small LLMs",
      "HuggingFace",
    ],
  },
  {
    title: "Orca 2",
    link: "b6ffbccd1eef",
    date: "November 2023",
    description:
      "Introduces Cautious Reasoning for training smaller models to select the most effective solution strategy based on the problem at hand, by crafting data with task-specific system instruction(s) corresponding to the chosen strategy in order to obtain teacher responses for each task and replacing the student’s system instruction with a generic one vacated of details of how to approach the task.",
    tags: [
      "Language Models",
      "Transformer Decoder",
      "Small LLMs",
      "Synthetic Data",
      "Orca",
    ],
  },
  {
    title: "Phi-2",
    link: "2857e56dbd2a#8230",
    date: "December 2023",
    description:
      "A 2.7B model, developed to explore whether emergent abilities achieved by large-scale language models can also be achieved at a smaller scale using strategic choices for training, such as data selection.",
    tags: [
      "Language Models",
      "Transformer Decoder",
      "Small LLMs",
      "Synthetic Data",
      "Phi",
    ],
  },
  {
    title: "TinyLlama",
    link: "6ef140170da9",
    date: "January 2024",
    description:
      "A  1.1B language model built upon the architecture and tokenizer of Llama 2, pre-trained on around 1 trillion tokens for approximately 3 epochs, leveraging FlashAttention and Grouped Query Attention, to achieve better computational efficiency.",
    tags: ["Language Models", "Transformer Decoder", "Small LLMs"],
  },
  {
    title: "Danube",
    link: "b790c073d257",
    date: "January 2024",
    description:
      "A language model trained on 1T tokens following the core principles of LLama 2 and Mistral, leveraging and refining various techniques for pre-training large language models.",
    tags: [
      "Language Models",
      "Transformer Decoder",
      "Small LLMs",
      "Danube",
      "H2O",
    ],
  },
  {
    title: "OLMo",
    link: "fdc358326f9b",
    date: "February 2024",
    description:
      "A state-of-the-art, truly open language model and framework that includes training data, code, and tools for building, studying, and advancing language models.",
    tags: ["Language Models", "Transformer Decoder", "Small LLMs", "Olmo"],
  },
  {
    title: "Mobile LLM",
    link: "2d7fdd5acd86",
    date: "February 2024",
    description:
      "Leverages various architectures and attention mechanisms to achieve a strong baseline network, which is then improved upon by introducing an immediate block-wise weight-sharing approach, resulting in a further accuracy boost.",
    tags: ["Language Models", "Transformer Decoder", "Small LLMs"],
  },
  {
    title: "Orca Math",
    link: "ae6a157ce48d",
    date: "February 2024",
    description:
      "A fine tuned Mistral-7B that excels at math problems without external tools, utilizing a high-quality synthetic dataset of 200K problems created through multi-agent collaboration and an iterative learning process that involves practicing problem-solving, receiving feedback, and learning from preference pairs incorporating the model's solutions and feedback.",
    tags: [
      "Language Models",
      "Transformer Decoder",
      "Small LLMs",
      "Synthetic Data",
      "Scientific Data",
      "Orca",
    ],
  },
  {
    title: "Gemma",
    link: "ca2b449321ac",
    date: "February 2024",
    description:
      "A family of 2B and 7B, state-of-the-art language models based on Google's Gemini models, offering advancements in language understanding, reasoning, and safety.",
    tags: ["Language Models", "Transformer Decoder", "Small LLMs", "Gemma"],
  },
  {
    title: "CodeGemma",
    link: "85faa98af20d",
    date: "April 2024",
    description:
      "Open code models based on Gemma models by further training on over 500 billion tokens of primarily code.",
    tags: [
      "Language Models",
      "Transformer Decoder",
      "Small LLMs",
      "Code Generation",
      "Gemma",
    ],
  },
  {
    title: "Phi-3",
    link: "0dfc951dc404",
    date: "April 2024",
    description:
      "A series of language models trained on heavily filtered web and synthetic data set, achieving performance comparable to much larger models like Mixtral 8x7B and GPT-3.5.",
    tags: [
      "Language Models",
      "Transformer Decoder",
      "Small LLMs",
      "Synthetic Data",
      "Phi",
    ],
  },
  {
    title: "Open ELM",
    link: "864f6b28a6ab",
    date: "April 2024",
    description:
      "A fully open language model designed to enhance accuracy while using fewer parameters and pre-training tokens. Utilizes a layer-wise scaling strategy to allocate smaller dimensions in early layers, expanding in later layers.",
    tags: ["Language Models", "Transformer Decoder", "Small LLMs"],
  },
  {
    title: "Danube 2",
    link: "b790c073d257#00d8",
    date: "April 2024",
    description:
      "An updated version of the original H2O-Danube model, with improvements including removal of sliding window attention, changes to the tokenizer, and adjustments to the training data, resulting in significant performance enhancements.",
    tags: [
      "Language Models",
      "Transformer Decoder",
      "Small LLMs",
      "Danube",
      "H2O",
    ],
  },
  {
    title: "Granite Code Models",
    link: "e1a92678739b",
    date: "May 2024",
    description:
      "A family of code models ranging from 3B to 34B trained on 3.5-4.5T tokens of code written in 116 programming languages.",
    tags: [
      "Language Models",
      "Transformer Decoder",
      "Small LLMs",
      "Code Generation",
    ],
  },
  {
    title: "Gemma 2",
    link: "f1b75b56b9f2",
    date: "June 2024",
    description:
      "Utilizes interleaving local-global attentions and group-query attention, trained with knowledge distillation instead of next token prediction to achieve competitive performance comparable with larger models.",
    tags: ["Language Models", "Transformer Decoder", "Small LLMs", "Gemma"],
  },
  {
    title: "Orca 3 (Agent Instruct)",
    link: "41340505af36",
    date: "July 2024",
    description:
      "A fine tuned Mistral-7B through Generative Teaching via synthetic data generated through the proposed AgentInstruct framework, which generates both the prompts and responses, using only raw data sources like text documents and code files as seeds.",
    tags: [
      "Language Models",
      "Transformer Decoder",
      "Small LLMs",
      "Synthetic Data",
      "Orca",
    ],
  },
  {
    title: "Mathstral",
    link: "b9632dedf580#0fbe",
    date: "July 2024",
    description:
      "A 7B model designed for math reasoning and scientific discovery based on Mistral 7B specializing in STEM subjects.",
    tags: [
      "Language Models",
      "Transformer Decoder",
      "Small LLMs",
      "Scientific Data",
      "Mistral",
    ],
  },
  {
    title: "Smol LM",
    link: "a166d5f1facc",
    date: "July 2024",
    description:
      "A family of small models with 135M, 360M, and 1.7B parameters, utilizes Grouped-Query Attention (GQA), embedding tying, and a context length of 2048 tokens, trained on a new open source high-quality dataset.",
    tags: [
      "Language Models",
      "Transformer Decoder",
      "Small LLMs",
      "HuggingFace",
    ],
  },
  {
    title: "Mistral Nemo",
    link: "b9632dedf580#37cd",
    date: "July 2024",
    description:
      "A 12B Language Model built in collaboration between Mistral and NVIDIA, featuring a context window of 128K, an efficient tokenizer and trained with quantization awareness, enabling FP8 inference without any performance loss.",
    tags: ["Language Models", "Transformer Decoder", "Small LLMs", "Mistral"],
  },
  {
    title: "Minitron",
    link: "e55ea374d9dd",
    date: "July 2024",
    description:
      "Prunes an existing Nemotron model and re-trains it with a fraction of the original training data, achieving compression factors of 2-4×, compute cost savings of up to 40×, and improved performance on various language modeling tasks.",
    tags: [
      "Language Models",
      "Transformer Decoder",
      "Small LLMs",
      "Pruning",
      "Knowledge Distillation",
      "Nvidia",
    ],
  },
  {
    title: "Danube 3",
    link: "917a7b40a79f",
    date: "July 2024",
    description:
      "A series of 4B and 500M language models, trained on high-quality Web data in three stages with different data mixes before being fine-tuned for chat version.",
    tags: [
      "Language Models",
      "Transformer Decoder",
      "Small LLMs",
      "Danube",
      "H2O",
    ],
  },
  {
    title: "Smol LM v0.2",
    link: "a166d5f1facc#fdb2",
    date: "August 2024",
    description:
      "An advancement over SmolLM, better at staying on topic and responding appropriately to standard prompts, such as greetings and questions about their role as AI assistants.",
    tags: [
      "Language Models",
      "Transformer Decoder",
      "Small LLMs",
      "HuggingFace",
    ],
  },
  {
    title: "Phi-3.5",
    link: "a95429ea26c9",
    date: "August 2024",
    description:
      "A family of models consisting of three variants - MoE (16x3.8B), mini (3.8B), and vision (4.2B) - which are lightweight, multilingual, and trained on synthetic and filtered publicly available documents - with a focus on very high-quality, reasoning dense data.",
    tags: [
      "Language Models",
      "Transformer Decoder",
      "Small LLMs",
      "Synthetic Data",
      "Phi",
    ],
  },
  {
    title: "Minitron Approach in Practice",
    link: "6b473f67328d",
    date: "August 2024",
    description:
      "Applies the minitron approach to Llama 3.1 8B and Mistral-Nemo 12B, additionally applies teacher correction to align with the new data distribution.",
    tags: [
      "Language Models",
      "Transformer Decoder",
      "Small LLMs",
      "Pruning",
      "Knowledge Distillation",
      "Nvidia",
    ],
  },
  {
    title: "Mistral Small",
    link: "https://medium.com/dair-ai/b9632dedf580#5662",
    date: "September 2024",
    description:
      "A 22B model with significant improvements in human alignment, reasoning capabilities, and code over the previous model.",
    tags: ["Language Models", "Transformer Decoder", "Small LLMs", "Mistral"],
  },
  {
    title: "Nemotron-Mini-Hindi",
    link: "c7adc3b2f759",
    date: "October 2024",
    description:
      "A bilingual language model based on Nemotron-Mini 4B, specifically trained to improve Hindi and English performance using continuous pre-training on 400B real and synthetic tokens.",
    tags: [
      "Language Models",
      "Transformer Decoder",
      "Small LLMs",
      "Nvidia",
      "Multilingual Models",
    ],
  },
  {
    title: "Ministral",
    link: "b9632dedf580#1f34",
    date: "October 2024",
    description:
      "3B and 8B models with support up to 128k context length having a special interleaved sliding-window attention pattern for faster and memory-efficient inference.",
    tags: ["Language Models", "Transformer Decoder", "Small LLMs", "Mistral"],
  },
  {
    title: "Quantized Llama 3.2",
    link: "cc6965f61370",
    date: "October 2024",
    description:
      "Optimized versions of the Llama, using techniques like Quantization-Aware Training with LoRA Adapters and SpinQuant, to reduce model size and memory usage while maintaining accuracy and performance, enabling deployment on resource-constrained devices like mobile phones.",
    tags: [
      "Language Models",
      "Transformer Decoder",
      "Small LLMs",
      "Quantization",
    ],
  },
  {
    title: "Smol LM v2",
    link: "53991a485d7b",
    date: "November 2024",
    description:
      "A family of language models (135M, 360M, and 1.7B parameters), trained on 2T, 4T, and 11T tokens respectively from datasets including FineWeb-Edu, DCLM, The Stack, and curated math and coding datasets, with instruction-tuned versions created using Smol Talk dataset and DPO using UltraFeedback.",
    tags: [
      "Language Models",
      "Transformer Decoder",
      "Small LLMs",
      "HuggingFace",
    ],
  },
  {
    title: "Command R 7B",
    link: "94ba068ebd2b#0836",
    date: "December 2024",
    description:
      "The smallest, fastest, and final model in the R series of enterprise-focused LLMs. It offers a context length of 128k and delivers a powerful combination of multilingual support, citation verified retrieval-augmented generation (RAG), reasoning, tool use, and agentic behavior.",
    tags: ["Language Models", "Transformer Decoder", "Small LLMs", "Cohere"],
  },
  {
    title: "Phi-4",
    link: "ea59220f3f88",
    date: "December 2024",
    description:
      "A 14B language model prioritizing data quality through a training process incorporating synthetic data for pretraining and midtraining, curated organic data seeds, and innovative post-training techniques like pivotal token search for DPO, resulting in strong performance on reasoning-focused benchmarks, especially in STEM, comparable to much larger models, while also addressing overfitting and data contamination concerns.",
    tags: [
      "Language Models",
      "Transformer Decoder",
      "Small LLMs",
      "Synthetic Data",
      "Phi",
    ],
  },
  {
    title: "Mistral Small 3",
    link: "b9632dedf580#9c9b",
    date: "January 2025",
    description:
      "A latency-optimized, 24B parameter language model, designed for efficient handling of common generative AI tasks requiring strong language understanding and instruction following.",
    tags: ["Language Models", "Transformer Decoder", "Small LLMs", "Mistral"],
  },
  {
    title: "Mistral Saba",
    link: "b9632dedf580#9420",
    date: "February 2025",
    description:
      "A 24B parameter regional language model specializing in Middle Eastern and South Asian languages, particularly Arabic and South Indian languages like Tamil. It outperforms much larger models in regional accuracy and relevance while offering lower latency.",
    tags: [
      "Language Models",
      "Transformer Decoder",
      "Small LLMs",
      "Multilingual Models",
      "Mistral",
    ],
  },
  {
    title: "Phi-4 Mini",
    link: "2be1a69be78c",
    date: "February 2025",
    description:
      "A 3.8B parameter language model excelling in math and coding, utilizing high-quality web and synthetic data, and featuring a 200K token vocabulary and group query attention.",
    tags: [
      "Language Models",
      "Transformer Decoder",
      "Small LLMs",
      "Synthetic Data",
      "Phi",
    ],
  },
  {
    title: "Mistral Small 3.1",
    link: "b9632dedf580#1065",
    date: "March 2025",
    description:
      "A multimodal language model built upon Mistral Small 3 with improved text performance and an expanded context window of up to 128k tokens.",
    tags: ["Multimodal Models", "Mistral"],
  },
  {
    title: "Phi-4-Reasoning",
    link: "98c1d3b5e52d",
    date: "April 2025",
    description:
      "Phi-4-reasoning, and Phi-4-reasoning-plus, its RL-enhanced variant, are smaller yet highly performant reasoning models trained on a curated dataset of complex prompts and demonstrations generated by o3-mini. They demonstrate strong performance across diverse reasoning tasks including STEM, coding, and algorithmic problem-solving, often exceeding larger models while offering accuracy/token length tradeoffs and highlighting the importance of data curation, SFT/RL combination.",
    tags: [
      "Language Models",
      "Transformer Decoder",
      "Small LLMs",
      "Synthetic Data",
      "Phi",
      "Reasoning",
    ],
  },
  {
    title: "Phi-4-Mini-Reasoning",
    link: "251652be3e39",
    date: "April 2025",
    description:
      "A 3.8B parameter language model trained using a four-step process: large-scale mid-training and supervised fine-tuning on distilled long-CoT data, Rollout DDPO and Reinforcement Learning with verifiable rewards. This model outperforms larger reasoning models on math reasoning tasks, demonstrating the effectiveness of this training recipe for strong reasoning capabilities in small language models.",
    tags: [
      "Language Models",
      "Transformer Decoder",
      "Small LLMs",
      "Synthetic Data",
      "Phi",
      "Reasoning",
    ],
  },
  {
    title: "Gemma 3n",
    link: "153803a2c591#f2b2",
    date: "May 2025",
    description:
      "model optimized for on-device use, featuring innovations like PLE caching and MatFormer architecture for efficient performance.",
    tags: ["Language Models", "Transformer Decoder", "Small LLMs", "Gemma"],
  },
  {
    title: "Sarvam - M",
    link: "594e1bfb6b6c",
    date: "May 2025",
    description:
      "A large language model built on top of Mistral Small, enhanced through supervised finetuning (SFT), reinforcement learning with verifiable rewards (RLVR), and inference optimization, with a focus on Indian languages and reasoning tasks like math and programming.",
    tags: ["Language Models", "Transformer Decoder", "Small LLMs"],
  },
  {
    title: "Sarvam - Translate",
    link: "fb96bd865054",
    date: "June 2025",
    description:
      "A translation model trained by fine-tuning Gemma3–4B-IT. It supports 22 Indian languages - Hindi, Bengali, Marathi, Telugu, Tamil, Gujarati, Urdu, Kannada, Odia, Malayalam, Punjabi, Assamese, Maithili, Santali, Kashmiri, Nepali, Sindhi, Dogri, Konkani, Manipuri (Meitei), Bodo, Sanskrit.",
    tags: ["Language Models", "Transformer Decoder", "Small LLMs", "Multilingual Models"],
  },
  {
    title: "Magistral",
    link: "d79523e0df6b",
    date: "June 2025",
    description:
      "Mistral AI's first reasoning model, available in both open-source (Magistral Small, 24B parameters) and enterprise (Magistral Medium) versions, designed for domain-specific, transparent, and multilingual reasoning across various applications like business strategy, regulated industries, and software engineering.",
    tags: ["Language Models", "Transformer Decoder", "Small LLMs", "Mistral"],
  },
  {
    title: "Arcee Foundation Model",
    link: "5751b562dc7b",
    date: "June 2025",
    description:
      "A family of generative AI models built from the ground up for enterprise reality. Combined with built-in support for function calling and agentic reasoning, AFM-4.5B is ready to automate complex workflows immediately i.e. no fragile prompt engineering required.",
    tags: ["Language Models", "Transformer Decoder", "Small LLMs"],
  },
  {
    title: "SmolLM3",
    link: "a166d5f1facc#e4be",
    date: "July 2025",
    description:
      "A 3B language model featuring dual-mode reasoning (think/no_think), multilingual support for six languages (English, French, Spanish, German, Italian, Portuguese), and long context capabilities up to 128k.",
    tags: ["Language Models", "Transformer Decoder", "Small LLMs", "HuggingFace"],
  },
  {
    title: "MobileLLM-R1",
    link: "00d456b46e5e",
    date: "September 2025",
    description:
      "A series of sub-billion-parameter reasoning models that achieve strong reasoning abilities with significantly less data (∼2T tokens of high-quality data resampled to 4.2T tokens) compared to other models.",
    tags: ["Language Models", "Transformer Decoder", "Small LLMs", "Reasoning"],
  },
  {
    title: "Rnj-1",
    link: "43be78ab4b40",
    date: "December 2025",
    description:
      "An 8B model following the open-source Gemma 3 architecture, employs global self-attention and YaRN to extend the context to 32k, it is a part of the open-source AI movement and aims to advance AI technologies equitably.",
    tags: ["Language Models", "Transformer Decoder", "Small LLMs"],
  },
  {
    title: "Nemotron 3 Nano",
    link: "8c95d44b0540",
    date: "December 2025",
    description:
      "A 31.6B Mixture-of-Experts hybrid Mamba-Transformer (Activating 6 of 128 experts, only ~3.2B params, ~3.6B with embeddings) pretrained on 25 trillion tokens (including >3T new tokens) and supporting context lengths up to 1M tokens; it uses Mamba-2 and Grouped-Query-Attention blocks. The model was post-trained with supervised fine-tuning, multi-environment RLVR, and RLHF.",
    tags: ["Language Models", "Transformer Decoder", "Small LLMs", "Nvidia"],
  },
  {
    title: "TranslateGemma",
    link: "016d61e37245",
    date: "January 2026",
    description:
      "A suite of open machine translation models built on the Gemma 3 foundation models, enhanced via a two-stage process: supervised fine-tuning on a rich mix of high-quality synthetic and human-translated parallel data, followed by reinforcement learning using an ensemble of reward models (including MetricX-QE and AutoMQM) to directly optimize translation quality.",
    tags: ["Language Models", "Transformer Decoder", "Small LLMs", "Multilingual Models", "Gemma"]  
  },
  {
    title: "Ministral 3",
    link: "81f394784f92",
    date: "January 2026",
    description:
      "A family of parameter-efficient dense language models derived from the 24B-parameter Mistral Small 3.1 using a compute-efficient Cascade Distillation recipe (iterative pruning plus continued training with logit distillation). The series includes three sizes (3B, 8B, 14B), each released in three variants (base, instruction-finetuned, and reasoning), all with image understanding, long context windows (up to 256k tokens, 128k for reasoning models).",
    tags: ["Language Models", "Transformer Decoder", "Small LLMs", "Mistral",]
  },
  {
    title: "Tiny Aya",
    link: "5eccbb462932",
    date: "February 2026",
    description:
      "A family of 3.35B open-weight multilingual language models, designed for balanced performance across 70+ languages through deliberate data curation, training design, and a comprehensive multilingual evaluation framework. The release includes Tiny Aya Base (pretrained model), Tiny Aya Global (instruction-tuned model optimized for consistent multilingual performance), and three region-specialized variants: Tiny Aya Earth (Africa and West Asia), Tiny Aya Fire (South Asia), and Tiny Aya Water (Asia-Pacific and Europe).",
    tags: ["Language Models", "Transformer Decoder", "Small LLMs", "Multilingual Models"],
  }
];
