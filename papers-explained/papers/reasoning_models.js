const reasoning_models = [
  {
    title: "Kimi k1.5",
    link: "c41e914cae7b",
    date: "January 2025",
    description:
      "A multimodal LLM trained with reinforcement learning (RL) focused on long context scaling and improved policy optimization, achieving state-of-the-art reasoning performance in various benchmarks and modalities, matching OpenAI's o1 in long-context tasks and significantly outperforming short-context models like GPT-4o and Claude Sonnet 3.5 through effective long2short context compression methods.",
    tags: ["Multimodal Models"],
  },
  {
    title: "Fathom R1",
    link: "ba93dd040cbf",
    date: "May 2025",
    description:
      "A 14B parameter reasoning language model, derived from Deepseek-R1-Distilled-Qwen-14B, trained using supervised fine-tuning (SFT) on curated datasets and model merging, and the models, post-training recipes, and datasets are open-sourced.",
    tags: ["LLM for Math"],
  },
  {
    title: "AceReason-Nemotron",
    link: "0b3bd6495890",
    date: "May 2025",
    description:
      "Demonstrates that large-scale reinforcement learning can significantly enhance the reasoning capabilities of strong small- and mid-sized models by first training on math-only prompts, then on code-only prompts.",
    tags: ["LLM for Math", "Reinforcement Learning"],
  },
  {
    title: "ProRL",
    link: "261c9ac00bc7",
    date: "May 2025",
    description:
      "Challenges the idea that RL only amplifies existing outputs and demonstrates that prolonged RL training can uncover novel reasoning strategies not accessible to base models. ProRL is designed to enable extended reinforcement learning training periods, facilitating deeper exploration of reasoning strategies by incorporating KL divergence control, reference policy resetting, and a diverse suite of tasks.",
    tags: ["Language Models", "LLM for Math"],
  },
  {
    title: "Adaptive Reasoning Model",
    link: "9aacae6918a9",
    date: "May 2025",
    description:
      "A reasoning model that adaptively selects appropriate reasoning formats (Direct Answer, Short CoT, Code, and Long CoT) based on task difficulty to balance performance and computational efficiency. It is trained using Ada-GRPO, an adaptation of GRPO, to address format collapse and improve training speed.",
    tags: ["Language Models", "LLM for Math"],
  },
  {
    title: "General-Reasoner",
    link: "d3b1c11bfc9a",
    date: "May 2025",
    description:
      "Enhances LLM reasoning across diverse domains by constructing a large-scale, high-quality dataset of verifiable questions curated by web crawling and developing a generative model-based answer verifier with chain-of-thought.",
    tags: ["Language Models", "Reinforcement Learning"],
  },
  {
    title: "AceReason-Nemotron 1.1",
    link: "f2747af699a5",
    date: "June 2025",
    description:
      "Leverages the synergy between SFT and RL through scaling SFT training data, carefully selecting the sampling temperature during RL training to balance exploration and exploitation, and employing a stage-wise RL approach on math-only and code-only prompts.",
    tags: ["LLM for Math", "Reinforcement Learning"],
  },
  {
    title: "ReCode",
    link: "a0a63c7705fe",
    date: "June 2025",
    description:
      "A framework that uses rule-based Reinforcement Learning for Code Update to improve LLMs' code generation performance in dynamic API scenarios by mimicking how human programmers adapt to API changes.",
    tags: ["LLM for Code", "Reinforcement Learning"]
  },
  {
    title: "POLARIS",
    link: "d0384cf030c5",
    date: "June 2025",
    description:
      "A post-training recipe focused on calibrated data difficulty, enhanced data diversity, inference-time length scaling, and efficient training, designed to scale reinforcement learning on advanced reasoning models.",
    tags: ["LLM for Math", "Reinforcement Learning"]
  },
  {
    title: "MiroMind-M1",
    link: "d2206c0b1b1e",
    date: "July 2025",
    description:
      "A series of Reasoning Language Models built on the Qwen-2.5 backbone,trained in two stages: SFT on a curated dataset of math problems with chain-of-thought trajectories, followed by reinforcement learning with verifiable reward, utilizing a novel Context-Aware Multi-Stage Policy Optimization algorithm to enhance robustness and efficiency.",
    tags: ["LLM for Math", "Reinforcement Learning"]
  },
  {
    title: "Klear-Reasoner",
    link: "985624c5348e",
    date: "August 2025",
    description:
      "A reasoning model that achieves high performance on multiple benchmarks by using long Chain-of-Thought supervised fine-tuning and reinforcement learning. It addresses issues with current clipping mechanisms in RL by proposing Gradient-Preserving clipping Policy Optimization (GPPO), which enhances exploration and improves learning from negative samples.",
    tags: ["LLM for Math", "Reinforcement Learning"]
  },
  {
    title: "gpt oss",
    link: "e1aed3d15afe",
    date: "August 2025",
    description:
      "A series of open-weight language models released by OpenAI, including gpt-oss-120b and gpt-oss-20b, designed for reasoning, agentic tasks, and versatile developer use cases, optimized for efficient deployment, tool use, and customization, with a focus on safety and alignment with OpenAI's standards.",
    tags: ["Language Models", "OpenAI"]
  },
  {
    title: "ProRL V2",
    link: "261c9ac00bc7#06a7",
    date: "August 2025",
    description:
      "The updated iteration of the ProRL regime, designed to test the effects of extended RL training on LLMs, pushing beyond typical training schedules with advanced algorithms, regularization, and domain coverage.",
    tags: ["Language Models", "LLM for Math"]
  },
  {
    title: "Nemotron Nano 2",
    link: "d3cc3326fe5c",
    date: "August 2025",
    description:
      "A hybrid Mamba-Transformer language model designed to increase throughput for reasoning workloads, achieving accuracy compared to similarly-sized models like Qwen3-8B, with up to 6x higher inference throughput in reasoning settings. It builds on the Nemotron-H architecture, utilizes new datasets and recipes for pre-training, alignment, pruning, and distillation.",
    tags: ["LLM for Math", "Reinforcement Learning", "Hybrid Models"]
  },
  {
    title: "Apriel-Nemotron-15B-Thinker",
    link : "463f8f4b5045",
    date: "August 2025",
    description:
      "A 15B parameter model in the ServiceNow Apriel SLM series. It is trained in a four stage training pipeline including: Base Model upscaling, Continual Pre-training, Supervised Fine-tuning (SFT), Reinforcement Learning using GRPO.",
    tags: ["LLM for Math", "Reinforcement Learning"]
  },
  {
    title: "Command A Reasoning",
    link: "94ba068ebd2b#7840",
    date: "August 2025",
    description:
      "A 111B reasoning model for enterprise tasks, offering secure, efficient, and scalable deployment options, a long context length, and a user-controlled token budget, making it ideal for agentic workflows and complex multi-step use cases.",
    tags: ["Language Models"]
  },
  {
    title: "Hermes 4",
    link: "2fba381f0c0a",
    date: "August 2025",
    description:
      "A family of hybrid reasoning models that combines structured, multi-step reasoning with broad instruction-following ability. The report details the data synthesis and curation strategy, training methodology incorporating loss-masking and efficient packing, demonstrating that open-weight reasoning models can be effectively trained and evaluated to achieve performance comparable to frontier systems.",
    tags: ["Language Models", "Transformer Decoder", "Reasoning"],
  },
  {
    title: "rStar2-Agent",
    link: "a3e7f451ddb7",
    date: "August 2025",
    description:
      "A 14B math reasoning model trained with agentic reinforcement learning, utilizing advanced cognitive behaviors like careful tool use and reflection on code execution feedback. It incorporates innovations such as an efficient RL infrastructure, the GRPO-RoC algorithm, and an efficient agent training recipe, enabling strong generalization to other tasks.",
    tags: ["LLM for Math", "Reinforcement Learning", "Agentic Models"]
  },
  {
    title: "Apriel-1.5-15B-Thinker",
    link: "228b6fab1efd",
    date: "October 2025",
    description:
      "A 15-billion parameter open-weights multimodal reasoning model that achieves frontier-level performance through a three-stage training methodology involving depth upscaling, staged continual pre-training, and high-quality supervised fine-tuning. The model's design focuses on maximizing the potential of the base model through mid-training, without employing reinforcement learning or preference optimization, making it suitable for organizations with limited infrastructure.",
    tags: ["Multimodal Models", "Language Models"],
  },
  {
    title: "gpt oss safeguard",
    link: "d8d36703a63e",
    date: "October 2025",
    description:
      "A set of open-weight reasoning models based on gpt oss designed for safety classification tasks, allowing developers to classify content based on their own policies by providing both the policy and the content to the model at inference time. It uses chain-of-thought reasoning, enabling developers to understand how the model reaches its decisions, and offers flexibility in adapting to evolving risks.",
    tags: ["Language Models", "OpenAI"]
  },
  {
    title: "P1",
    link: "15520a79edd3",
    date: "November 2025",
    description:
      "A family of open-source physics reasoning models, including P1-235B-A22B (Gold-medal performance at IPhO 2025) and P1-30B-A3B (Silver-medal performance), trained via reinforcement learning and designed for Olympiad-level physics problem-solving. Combined with the PhysicsMinions agent framework, P1-235B-A22B achieves the top score on IPhO 2025.",
    tags: ["Reinforcement Learning"]
  }
];
