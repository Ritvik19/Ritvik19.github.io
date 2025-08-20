const reasoning_models = [
  {
    title: "Kimi k1.5",
    link: "https://ritvik19.medium.com/papers-explained-334-kimi-k1-5-c41e914cae7b",
    date: "January 2025",
    description:
      "A multimodal LLM trained with reinforcement learning (RL) focused on long context scaling and improved policy optimization, achieving state-of-the-art reasoning performance in various benchmarks and modalities, matching OpenAI's o1 in long-context tasks and significantly outperforming short-context models like GPT-4o and Claude Sonnet 3.5 through effective long2short context compression methods.",
    tags: ["Multimodal Models"],
  },
  {
    title: "Fathom R1",
    link: "https://ritvik19.medium.com/papers-explainedv377-fathom-r1-ba93dd040cbf",
    date: "May 2025",
    description:
      "A 14B parameter reasoning language model, derived from Deepseek-R1-Distilled-Qwen-14B, trained using supervised fine-tuning (SFT) on curated datasets and model merging, and the models, post-training recipes, and datasets are open-sourced.",
    tags: ["LLM for Math"],
  },
  {
    title: "AceReason-Nemotron",
    link: "https://ritvik19.medium.com/papers-explained-381-acereason-nemotron-0b3bd6495890",
    date: "May 2025",
    description:
      "Demonstrates that large-scale reinforcement learning can significantly enhance the reasoning capabilities of strong small- and mid-sized models by first training on math-only prompts, then on code-only prompts.",
    tags: ["LLM for Math", "Reinforcement Learning"],
  },
  {
    title: "ProRL",
    link: "https://ritvik19.medium.com/papers-explained-386-prorl-261c9ac00bc7",
    date: "May 2025",
    description:
      "Challenges the idea that RL only amplifies existing outputs and demonstrates that prolonged RL training can uncover novel reasoning strategies not accessible to base models. ProRL is designed to enable extended reinforcement learning training periods, facilitating deeper exploration of reasoning strategies by incorporating KL divergence control, reference policy resetting, and a diverse suite of tasks.",
    tags: ["Language Models", "LLM for Math"],
  },
  {
    title: "Adaptive Reasoning Model",
    link: "https://ritvik19.medium.com/papers-explained-391-adaptive-reasoning-model-9aacae6918a9",
    date: "May 2025",
    description:
      "A reasoning model that adaptively selects appropriate reasoning formats (Direct Answer, Short CoT, Code, and Long CoT) based on task difficulty to balance performance and computational efficiency. It is trained using Ada-GRPO, an adaptation of GRPO, to address format collapse and improve training speed.",
    tags: ["Language Models", "LLM for Math"],
  },
  {
    title: "AceReason-Nemotron 1.1",
    link: "https://ritvik19.medium.com/papers-explained-395-acereason-nemotron-1-1-f2747af699a5",
    date: "June 2025",
    description:
      "Leverages the synergy between SFT and RL through scaling SFT training data, carefully selecting the sampling temperature during RL training to balance exploration and exploitation, and employing a stage-wise RL approach on math-only and code-only prompts.",
    tags: ["LLM for Math", "Reinforcement Learning"],
  },
  {
    title: "ReCode",
    link: "https://ritvik19.medium.com/papers-explained-425-recode-a0a63c7705fe",
    date: "June 2025",
    description:
      "A framework that uses rule-based Reinforcement Learning for Code Update to improve LLMs' code generation performance in dynamic API scenarios by mimicking how human programmers adapt to API changes.",
    tags: ["LLM for Code", "Reinforcement Learning"]
  },
  {
    title: "POLARIS",
    link: "",
    date: "June 2025",
    description:
      "A post-training recipe focused on calibrated data difficulty, enhanced data diversity, inference-time length scaling, and efficient training, designed to scale reinforcement learning on advanced reasoning models.",
    tags: ["LLM for Math", "Reinforcement Learning"]
  },
  {
    title: "MiroMind-M1",
    link: "",
    date: "July 2025",
    description:
      "A series of Reasoning Language Models built on the Qwen-2.5 backbone,trained in two stages: SFT on a curated dataset of math problems with chain-of-thought trajectories, followed by reinforcement learning with verifiable reward, utilizing a novel Context-Aware Multi-Stage Policy Optimization algorithm to enhance robustness and efficiency.",
    tags: ["LLM for Math", "Reinforcement Learning"]
  },
  {
    title: "gpt oss",
    link: "https://ritvik19.medium.com/papers-explained-428-gpt-oss-e1aed3d15afe",
    date: "August 2025",
    description:
      "A series of open-weight language models released by OpenAI, including gpt-oss-120b and gpt-oss-20b, designed for reasoning, agentic tasks, and versatile developer use cases, optimized for efficient deployment, tool use, and customization, with a focus on safety and alignment with OpenAI's standards.",
    tags: ["Language Models", "OpenAI"]
  },
  {
    title: "ProRL V2",
    link: "https://ritvik19.medium.com/papers-explained-386-prorl-261c9ac00bc7#06a7",
    date: "August 2025",
    description:
      "The updated iteration of the ProRL regime, designed to test the effects of extended RL training on LLMs, pushing beyond typical training schedules with advanced algorithms, regularization, and domain coverage.",
    tags: ["Language Models", "LLM for Math"]
  },
  {
    title: "Apriel-Nemotron-15B-Thinker",
    link : "",
    date: "August 2025",
    description:
      "A 15B parameter model in the ServiceNow Apriel SLM series. It is trained in a four stage training pipeline including: Base Model upscaling, Continual Pre-training, Supervised Fine-tuning (SFT), Reinforcement Learning using GRPO.",
    tags: ["LLM for Math", "Reinforcement Learning"]
  }
];
