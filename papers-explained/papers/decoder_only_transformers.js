const decoder_only_transformers = [
  {
    title: "GPT",
    link: "30b6f1e6d226",
    date: "June 2018",
    description:
      "A Decoder only transformer which is autoregressively pretrained and then finetuned for specific downstream tasks using task-aware input transformations.",
    tags: ["Language Models", "Transformer Decoder", "OpenAI", "GPT"],
  },
  {
    title: "GPT 2",
    link: "98d0a642e520",
    date: "February 2019",
    description:
      "Demonstrates that language models begin to learn various language processing tasks without any explicit supervision.",
    tags: ["Language Models", "Transformer Decoder", "OpenAI", "GPT"],
  },
  {
    title: "GPT 3",
    link: "352f5a1b397",
    date: "May 2020",
    description:
      "Demonstrates that scaling up language models greatly improves task-agnostic, few-shot performance.",
    tags: ["Language Models", "Transformer Decoder", "OpenAI", "GPT"],
  },
  {
    title: "Codex",
    link: "caca940feb31",
    date: "July 2021",
    description:
      "A GPT language model finetuned on publicly available code from GitHub.",
    tags: [
      "Language Models",
      "Transformer Decoder",
      "Code Generation",
      "OpenAI",
      "GPT",
    ],
  },
  {
    title: "WebGPT",
    link: "5bb0dd646b32",
    date: "December 2021",
    description:
      "A fine-tuned GPT-3 model utilizing text-based web browsing, trained via imitation learning and human feedback, enhancing its ability to answer long-form questions with factual accuracy.",
    tags: ["Language Models", "Transformer Decoder", "OpenAI", "GPT"],
  },
  {
    title: "Gopher",
    link: "2e71bbef9e87",
    date: "December 2021",
    description:
      "Provides a comprehensive analysis of the performance of various Transformer models across different scales upto 280B on 152 tasks.",
    tags: ["Language Models", "Transformer Decoder"],
  },
  {
    title: "LaMDA",
    link: "a580ebba1ca2",
    date: "January 2022",
    description:
      "Transformer based models specialized for dialog, which are pre-trained on public dialog data and web text.",
    tags: ["Language Models", "Transformer Decoder"],
  },
  {
    title: "Instruct GPT",
    link: "e9bcd51f03ec",
    date: "March 2022",
    description:
      "Fine-tuned GPT using supervised learning (instruction tuning) and reinforcement learning from human feedback to align with user intent.",
    tags: ["Language Models", "Transformer Decoder", "OpenAI", "GPT"],
  },
  {
    title: "Chinchilla",
    link: "a7ad826d945e",
    date: "March 2022",
    description:
      "Investigated the optimal model size and number of tokens for training a transformer LLM within a given compute budget (Scaling Laws).",
    tags: ["Language Models", "Transformer Decoder"],
  },
  {
    title: "CodeGen",
    link: "a6bae5c1f7b5",
    date: "March 2022",
    description:
      "An LLM trained for program synthesis using input-output examples and natural language descriptions.",
    tags: ["Language Models", "Transformer Decoder", "Code Generation"],
  },
  {
    title: "PaLM",
    link: "480e72fa3fd5",
    date: "April 2022",
    description:
      "A 540-B parameter, densely activated, Transformer, trained using Pathways, (ML system that enables highly efficient training across multiple TPU Pods).",
    tags: ["Language Models", "Transformer Decoder"],
  },
  {
    title: "GPT-NeoX-20B",
    link: "fe39b6d5aa5b",
    date: "April 2022",
    description:
      "An autoregressive LLM trained on the Pile, and the largest dense model that had publicly available weights at the time of submission.",
    tags: ["Language Models", "Transformer Decoder"],
  },
  {
    title: "OPT",
    link: "dacd9406e2bd",
    date: "May 2022",
    description:
      "A suite of decoder-only pre-trained transformers with parameter ranges from 125M to 175B. OPT-175B being comparable to GPT-3.",
    tags: ["Language Models", "Transformer Decoder"],
  },
  {
    title: "BLOOM",
    link: "9654c56cd2",
    date: "November 2022",
    description:
      "A 176B-parameter open-access decoder-only transformer, collaboratively developed by hundreds of researchers, aiming to democratize LLM technology.",
    tags: ["Language Models", "Transformer Decoder"],
  },
  {
    title: "Galactica",
    link: "1308dbd318dc",
    date: "November 2022",
    description:
      "An LLM trained on scientific data thus specializing in scientific knowledge.",
    tags: ["Language Models", "Transformer Decoder", "Scientific Data"],
  },
  {
    title: "LLaMA",
    link: "c4f302809d6b",
    date: "February 2023",
    description:
      "A collection of foundation LLMs by Meta ranging from 7B to 65B parameters, trained using publicly available datasets exclusively.",
    tags: ["Language Models", "Transformer Decoder", "Llama"],
  },
  {
    title: "Toolformer",
    link: "d21d496b6812",
    date: "February 2023",
    description:
      "An LLM trained to decide which APIs to call, when to call them, what arguments to pass, and how to best incorporate the results into future token prediction.",
    tags: ["Language Models", "Transformer Decoder"],
  },
  {
    title: "Alpaca",
    link: "933c4d9855e5",
    date: "March 2023",
    description:
      "A fine-tuned LLaMA 7B model, trained on instruction-following demonstrations generated in the style of self-instruct using text-davinci-003.",
    tags: ["Language Models", "Transformer Decoder"],
  },
  {
    title: "Vicuna",
    link: "daed99725c7e",
    date: "March 2023",
    description:
      "A 13B LLaMA chatbot fine tuned on user-shared conversations collected from ShareGPT, capable of generating more detailed and well-structured answers compared to Alpaca.",
    tags: ["Language Models", "Transformer Decoder"],
  },
  {
    title: "Bloomberg GPT",
    link: "4bedd52ef54b",
    date: "March 2023",
    description:
      "A 50B language model train on general purpose and domain specific data to support a wide range of tasks within the financial industry.",
    tags: ["Language Models", "Transformer Decoder"],
  },
  {
    title: "Pythia",
    link: "708284c32964",
    date: "April 2023",
    description:
      "A suite of 16 LLMs all trained on public data seen in the exact same order and ranging in size from 70M to 12B parameters.",
    tags: ["Language Models", "Transformer Decoder"],
  },
  {
    title: "WizardLM",
    link: "65099705dfa3",
    date: "April 2023",
    description:
      "Introduces Evol-Instruct, a method to generate large amounts of instruction data with varying levels of complexity using LLM instead of humans to fine tune a Llama model ",
    tags: [
      "Language Models",
      "Transformer Decoder",
      "Synthetic Data",
      "WizardLM",
    ],
  },
  {
    title: "CodeGen 2",
    link: "d2690d7eb831",
    date: "May 2023",
    description:
      "Proposes an approach to make the training of LLMs for program synthesis more efficient by unifying key components of model architectures, learning methods, infill sampling, and data distributions",
    tags: ["Language Models", "Transformer Decoder", "Code Generation"],
  },
  {
    title: "PaLM 2",
    link: "1a9a23f20d6c",
    date: "May 2023",
    description:
      "Successor of PALM, trained on a mixture of different pre-training objectives in order to understand different aspects of language.",
    tags: ["Language Models", "Transformer Decoder"],
  },
  {
    title: "LIMA",
    link: "f9401a5760c3",
    date: "May 2023",
    description:
      "A LLaMa model fine-tuned on only 1,000 carefully curated prompts and responses, without any reinforcement learning or human preference modeling.",
    tags: ["Language Models", "Transformer Decoder", "Less is More"],
  },
  {
    title: "Gorilla",
    link: "79f4730913e9",
    date: "May 2023",
    description:
      "A retrieve-aware finetuned LLaMA-7B model, specifically for API calls.",
    tags: ["Language Models", "Transformer Decoder"],
  },
  {
    title: "Falcon",
    link: "26831087247f",
    date: "June 2023",
    description:
      "An Open Source LLM trained on properly filtered and deduplicated web data alone.",
    tags: ["Language Models", "Transformer Decoder"],
  },
  {
    title: "WizardCoder",
    link: "a12ecb5b93b6",
    date: "June 2023",
    description:
      "Enhances the performance of the open-source Code LLM, StarCoder, through the application of Code Evol-Instruct.",
    tags: [
      "Language Models",
      "Transformer Decoder",
      "Code Generation",
      "Synthetic Data",
      "WizardLM",
    ],
  },
  {
    title: "Tulu",
    link: "ee85648cbf1b",
    date: "June 2023",
    description:
      "Explores instruction-tuning of language models ranging from 6.7B to 65B parameters on 12 different instruction datasets.",
    tags: ["Language Models", "Transformer Decoder"],
  },
  {
    title: "LLaMA 2",
    link: "3e415c5b9b17",
    date: "July 2023",
    description:
      "Successor of LLaMA. LLaMA 2-Chat is optimized for dialogue use cases.",
    tags: ["Language Models", "Transformer Decoder", "Llama"],
  },
  {
    title: "Tool LLM",
    link: "856f99e79f55",
    date: "July 2023",
    description:
      "A LLaMA model finetuned on an instruction-tuning dataset for tool use, automatically created using ChatGPT.",
    tags: ["Language Models", "Transformer Decoder"],
  },
  {
    title: "Humpback",
    link: "46992374fc34",
    date: "August 2023",
    description: "LLaMA finetuned using Instruction backtranslation.",
    tags: ["Language Models", "Transformer Decoder", "Synthetic Data"],
  },
  {
    title: "Code LLaMA",
    link: "ee266bfa495f",
    date: "August 2023",
    description: "LLaMA 2 based LLM for code.",
    tags: [
      "Language Models",
      "Transformer Decoder",
      "Code Generation",
      "Llama",
    ],
  },
  {
    title: "WizardMath",
    link: "265e6e784341",
    date: "August 2023",
    description:
      "Proposes Reinforcement Learning from Evol-Instruct Feedback (RLEIF) method, applied to Llama-2 to enhance the mathematical reasoning abilities.",
    tags: [
      "Language Models",
      "Transformer Decoder",
      "Scientific Data",
      "WizardLM",
    ],
  },
  {
    title: "LLaMA 2 Long",
    link: "84d33c26d14a",
    date: "September 2023",
    description:
      "A series of long context LLMs s that support effective context windows of up to 32,768 tokens.",
    tags: ["Language Models", "Transformer Decoder", "Llama"],
  },
  {
    title: "MAmmoTH",
    link: "06189e929910",
    date: "September 2023",
    description:
      "A series of LLMs specifically designed for general math problem-solving, trained on MathInstruct, a dataset compiled from 13 math datasets with intermediate rationales that combines chain-of-thought and program-of-thought approaches to accommodate different thought processes for various math problems.",
    tags: ["Language Models", "Transformer Decoder", "Scientific Data"],
  },
  {
    title: "Llemma",
    link: "0a17287e890a",
    date: "October 2023",
    description:
      "An LLM for mathematics, formed by continued pretraining of Code Llama on a mixture of scientific papers, web data containing mathematics, and mathematical code.",
    tags: ["Language Models", "Transformer Decoder", "Scientific Data"],
  },
  {
    title: "Tulu v2",
    link: "ff38ab1f37f2",
    date: "November 2023",
    description:
      "An updated version of Tulu covering the open resources for instruction tuning om better base models to new finetuning techniques.",
    tags: ["Language Models", "Transformer Decoder"],
  },
  {
    title: "Mixtral 8x7B",
    link: "9e9f40ebb745",
    date: "January 2024",
    description:
      "A Sparse Mixture of Experts language model based on Mistral 7B trained with multilingual data using a context size of 32k tokens.",
    tags: [
      "Language Models",
      "Transformer Decoder",
      "Mixtures of Experts",
      "Mistral",
    ],
  },
  {
    title: "Nemotron-4 15B",
    link: "7d895fb56134",
    date: "February 2024",
    description:
      "A 15B multilingual language model trained on 8T text tokens by Nvidia.",
    tags: ["Language Models", "Transformer Decoder", "Nvidia"],
  },
  {
    title: "DBRX",
    link: "17c61739983c",
    date: "March 2024",
    description:
      "A 132B open, general-purpose fine grained Sparse MoE LLM surpassing GPT-3.5 and competitive with Gemini 1.0 Pro.",
    tags: ["Language Models", "Transformer Decoder", "Mixtures of Experts"],
  },
  {
    title: "Command R",
    link: "94ba068ebd2b",
    date: "March 2024",
    description:
      "An LLM optimized for retrieval-augmented generation and tool use, across multiple languages.",
    tags: ["Language Models", "Transformer Decoder", "Cohere"],
  },
  {
    title: "Mixtral 8x22B",
    link: "9e9f40ebb745#20f3",
    date: "April 2024",
    description:
      "A open-weight AI model optimised for performance and efficiency, with capabilities such as fluency in multiple languages, strong mathematics and coding abilities, and precise information recall from large documents.",
    tags: [
      "Language Models",
      "Transformer Decoder",
      "Mixtures of Experts",
      "Mistral",
    ],
  },
  {
    title: "Llama 3",
    link: "51e2b90f63bb",
    date: "April 2024",
    description:
      "A family of 8B and 70B parameter models trained on 15T tokens with a focus on data quality, demonstrating state-of-the-art performance on various benchmarks, improved reasoning capabilities.",
    tags: ["Language Models", "Transformer Decoder", "Llama"],
  },
  {
    title: "Command R+",
    link: "94ba068ebd2b#c2b5",
    date: "April 2024",
    description:
      "Successor of Command R+ with improved performance for retrieval-augmented generation and tool use, across multiple languages.",
    tags: ["Language Models", "Transformer Decoder", "Cohere"],
  },
  {
    title: "Rho-1",
    link: "788125e42241",
    date: "April 2024",
    description:
      "Introduces Selective Language Modelling that optimizes the loss only on tokens that align with a desired distribution, utilizing a reference model to score and select tokens.",
    tags: ["Language Models", "Transformer Decoder"],
  },
  {
    title: "MAmmoTH 2",
    link: "e9c0e6fb9795",
    date: "May 2024",
    description:
      "LLMs fine tuned on a dataset curated through the proposed paradigm that efficiently harvest 10M naturally existing instruction data from the pre-training web corpus to enhance LLM reasoning. It involves recalling relevant documents, extracting instruction-response pairs, and refining the extracted pairs using open-source LLMs.",
    tags: ["Language Models", "Transformer Decoder", "Scientific Data"],
  },
  {
    title: "Codestral 22B",
    link: "b9632dedf580#057b",
    date: "May 2024",
    description:
      "An open-weight model designed for code generation tasks, trained on over 80 programming languages, and licensed under the Mistral AI Non-Production License, allowing developers to use it for research and testing purposes.",
    tags: [
      "Language Models",
      "Transformer Decoder",
      "Code Generation",
      "Mistral",
    ],
  },
  {
    title: "Aya 23",
    link: "d01605c3ee80",
    date: "May 2024",
    description:
      "A family of multilingual language models supporting 23 languages, designed to balance breadth and depth by allocating more capacity to fewer languages during pre-training.",
    tags: [
      "Language Models",
      "Transformer Decoder",
      "Multilingual Models",
      "Cohere",
    ],
  },
  {
    title: "Nemotron-4 340B",
    link: "4cfe268439f8",
    date: "June 2024",
    description:
      "340B models, along with a reward model by Nvidia, suitable for generating synthetic data to train smaller language models, with over 98% of the data used in model alignment being synthetically generated.",
    tags: ["Language Models", "Transformer Decoder", "Nvidia"],
  },
  {
    title: "LLama 3.1",
    link: "f0fb06898c59",
    date: "July 2024",
    description:
      "A family of multilingual language models ranging from 8B to 405B parameters, trained on a massive dataset of 15T tokens and achieving comparable performance to leading models like GPT-4 on various tasks.",
    tags: ["Language Models", "Transformer Decoder", "Llama"],
  },
  {
    title: "LLama 3.1 - Multimodal Experiments",
    link: "a1940dd45575",
    date: "July 2024",
    description:
      "Additional experiments of adding multimodal capabilities to Llama3.",
    tags: [
      "Language Models",
      "Transformer Decoder",
      "Multimodal Models",
      "Llama",
    ],
  },
  {
    title: "LLM Compiler",
    link: "15b1ddb9a1b0",
    date: "July 2024",
    description:
      "A suite of pre-trained models designed for code optimization tasks, built upon Code Llama, with two sizes (7B and 13B), trained on LLVM-IR and assembly code to optimize compiler intermediate representations, assemble/disassemble, and achieve high accuracy in optimizing code size and disassembling from x86_64 and ARM assembly back into LLVM-IR.",
    tags: [
      "Language Models",
      "Transformer Decoder",
      "Code Generation",
      "Llama",
    ],
  },
  {
    title: "Apple Intelligence Foundation Language Models",
    link: "2b8a41371a42",
    date: "July 2024",
    description:
      "Two foundation language models, AFM-on-device (a ~3 B parameter model) and AFM-server (a larger server-based model), designed to power Apple Intelligence features efficiently, accurately, and responsibly, with a focus on Responsible AI principles that prioritize user empowerment, representation, design care, and privacy protection.",
    tags: ["Language Models", "Transformer Decoder"],
  },
  {
    title: "Hermes 3",
    link: "67d36cfe07d8",
    date: "August 2024",
    description:
      "Neutrally generalist instruct and tool use models, created by fine-tuning Llama 3.1 models with strong reasoning and creative abilities, and are designed to follow prompts neutrally without moral judgment or personal opinions.",
    tags: ["Language Models", "Transformer Decoder"],
  },
  {
    title: "OLMoE",
    link: "38832ff4f9bd",
    date: "September 2024",
    description:
      "An open source language model based on sparse Mixture-of-Experts architecture with 7B parameters, out of which only 1B parameters are active per input token. Conducted extensive experiments on MoE training, analyzing routing strategies, expert specialization, and the impact of design choices like routing algorithms and expert size.",
    tags: ["Language Models", "Transformer Decoder", "Mixtures of Experts"],
  },
  {
    title: "Llama 3.1-Nemotron-51B",
    link: "6b473f67328d#5df9",
    date: "September 2024",
    description:
      "Uses knowledge distillation and NAS to optimize various constraints, resulting in a model that achieves 2.2x faster inference compared to the reference model while maintaining nearly the same accuracy, with an irregular block structure that reduces or prunes attention and FFN layers for better utilization of H100 and improved LLMs for inference.",
    tags: [
      "Language Models",
      "Transformer Decoder",
      "Pruning",
      "Knowledge Distillation",
      "Nvidia",
    ],
  },
  {
    title: "LLama 3.2",
    link: "e517fa1f2528",
    date: "September 2024",
    description:
      "Small and medium-sized vision LLMs (11B and 90B), and lightweight, text-only models (1B and 3B).",
    tags: [
      "Language Models",
      "Transformer Decoder",
      "Llama",
      "Small LLMs",
      "Multimodal Models",
    ],
  },
  {
    title: "Aya Expanse",
    link: "d01605c3ee80#c4a1",
    date: "October 2024",
    description:
      "A family of 8B and 32B highly performant multilingual models that excel across 23 languages.",
    tags: [
      "Language Models",
      "Transformer Decoder",
      "Multilingual Models",
      "Cohere",
    ],
  },
  {
    title: "Tulu v3",
    link: "fc7758b18724",
    date: "November 2024",
    description:
      "A family of post-trained models based on Llama 3.1 that outperform instruct versions of other models, including closed models like GPT-4o-mini and Claude 3.5-Haiku, using training methods like supervised finetuning, Direct Preference Optimization, and Reinforcement Learning with Verifiable Rewards.",
    tags: ["Language Models", "Transformer Decoder"],
  },
  {
    title: "Llama 3.3",
    link: "cc6965f61370#bd2c",
    date: "December 2024",
    description:
      "A multilingual, instruction-tuned generative language model with 70B parameters, optimized for dialogue use cases and trained on 15 trillion tokens of public data, incorporating both human-generated and synthetic data for safety and quality control.",
    tags: ["Language Models", "Transformer Decoder", "Llama"],
  },
  {
    title: "OLMo 2",
    link: "f4d34e886503",
    date: "January 2025",
    description:
      "A family of open-source language models featuring improved architecture, training recipes, and pre-training data mixtures. It incorporates a new specialized data mix (Dolmino Mix 1124) introduced via late-stage curriculum training, and best practices from Tülu 3 are incorporated to develop OLMo 2-Instruct.",
    tags: ["Language Models", "Transformer Decoder"],
  },
  {
    title: "AceCoder",
    link: "2611b3feef6c",
    date: "February 2025",
    description:
      "Leverages automated large-scale test-case synthesis to enhance code model training via reinforcement learning. It creates a dataset (AceCode-89K) of questions and test cases, trains reward models (AceCode-RM) using pass rates, and then uses these reward models and test-case pass rewards for reinforcement learning, significantly improving code generation performance across various benchmarks.",
    tags: ["Language Models", "Transformer Decoder", "Code Generation"],
  },
  {
    title: "Command A",
    link: "4e0512baee56",
    date: "March 2025",
    description:
      "A 111 billion parameter open-weights research release model optimized for business-critical agentic and multilingual tasks. It features a 256K context length, is trained on 23 languages, and is specifically designed for RAG and tool use,offering verifiable citations for both.",
    tags: ["Language Models", "Transformer Decoder", "Cohere"],
  },
  {
    title: "UltraLong",
    link: "981e997e4e19",
    date: "April 2025",
    description:
      "Introduces a training method for developing ultra-long context LLMs with context windows extending up to 4 million tokens, achieved through efficient continued pretraining with YaRN-based scaling, followed by instruction tuning.",
    tags: ["Language Models", "Transformer Decoder", "Nvidia"],
  },
  {
    title: "Llama-Nemotron",
    link: "d6b64f407e28",
    date: "May 2025",
    description:
      "An open-source family of heterogeneous reasoning models (Nano (8B), Super (49B), and Ultra (253B)) designed for exceptional reasoning, and efficient inference. Trained using neural architecture search, knowledge distillation, continued pretraining, supervised fine-tuning, and reinforcement learning, these models offer a dynamic reasoning toggle for switching between standard chat and detailed reasoning modes, achieving state-of-the-art performance, especially LN-Ultra which surpasses DeepSeek-R1 in scientific reasoning.",
    tags: ["Language Models", "Transformer Decoder", "Nvidia", "Reasoning"],
  },
  {
    title: "Devstral",
    link: "b9632dedf580#d26b",
    date: "May 2025",
    description:
      "An agentic LLM for software engineering tasks developed through a collaboration between Mistral AI and All Hands AI. It is finetuned from Mistral-Small-3.1",
    tags: ["Language Models", "Transformer Decoder", "Small LLMs", "Mistral"],
  },
  {
    title: "Kimi K2",
    link: "05663a5ee4aa",
    date: "July 2025",
    description:
      "A 1.04T parameter MoE LLM with 32B activated parameters, pre-trained on 15.5 trillion tokens using the MuonClip optimizer, with a novel QK-clip technique to address training instability while enjoying the advanced token efficiency of Muon, and post-trained with a large-scale agentic data synthesis pipeline and reinforcement learning.",
    tags: ["Language Models", "Transformer Decoder", "Mixtures of Experts"],
  },
  {
    title: "LIMI", 
    link: "f696e12fdb3f/",
    date: "September 2025",
    description:
      "Demonstrates that sophisticated agentic intelligence can emerge from minimal but strategically curated demonstrations of autonomous behavior. This challenges the traditional paradigm that more data yields better agency, using only 78 carefully designed training samples.",
    tags: ["Language Models", "Transformer Decoder", "Less is More"],
  }
];
