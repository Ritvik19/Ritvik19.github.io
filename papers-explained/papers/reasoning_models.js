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
    title: "OctThinker",
    link: "bdec24e27301",
    date: "June 2025",
    description:
      "Explores the effects of midtraining on RL performance and emphasises the importance of high-quality mathematical corpora, QA-styled data, particularly long chain-of-thought (CoT) reasoning examples, and instruction data. While long-CoT improves reasoning depth, it can also induce verbosity of model responses and instability of RL training.",
    tags: ["LLM for Math", "Reinforcement Learning", "Mid Training"]
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
    title: "Rubrics as Rewards",
    link: "229ff69f7355",
    date: "July 2025",
    description:
      "A reinforcement learning framework that uses structured, instance-specific rubrics as reward signals to train models for complex reasoning tasks in real-world domains like medicine and science. By decomposing evaluation criteria into modular, interpretable subgoals, RaR overcomes the limitations of traditional reward models and enables on-policy learning with transparent and aligned supervision.",
    tags: ["Reinforcement Learning"]
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
    link: "463f8f4b5045",
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
    title: "Composer",
    link: "9bc921210902#ecbf",
    date: "October 2025",
    description:
      "A fast, software engineering-focused mixture-of-experts (MoE) language model, trained with reinforcement learning (RL) to efficiently solve real-world coding challenges in large codebases using production search and editing tools.",
    tags: ["LLM for Code", "Reinforcement Learning", "Mixture-of-Experts"]
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
    title: "BroRL",
    link: "da46c4caec8f",
    date: "October 2025",
    description:
      "A reinforcement learning scaling paradigm for LLMs within the RLVR framework that focuses on broadening exploration by dramatically increasing the number of rollouts per example (to hundreds or thousands) instead of just adding more training steps.",
    tags: ["Language Models", "Reinforcement Learning"],
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
    title: "ScaleRL",
    link: "14ccd31f7d63",
    date: "October 2025",
    description:
      "The first large-scale systematic study, amounting to more than 400,000 GPU-hours, defines a principled framework for analyzing and predicting RL scaling in LLMs. Sigmoidal compute-performance curves are fit for RL training and a wide range of common design choices are ablated to analyze their effects on asymptotic performance and compute efficiency.",
    tags: ["Reinforcement Learning"]
  },
  {
    title: "DR Tulu",
    link: "123b031776c5",
    date: "November 2025",
    description:
      "The first open model directly trained for open-ended, long-form deep research, using a new method called Reinforcement Learning with Evolving Rubrics (RLER), where evaluation rubrics co-evolve with the policy model and are grounded in live, searched knowledge rather than static, closed-book criteria.",
    tags: ["Reinforcement Learning", "Deep Research"]
  },
  {
    title: "P1",
    link: "15520a79edd3",
    date: "November 2025",
    description:
      "A family of open-source physics reasoning models, including P1-235B-A22B (Gold-medal performance at IPhO 2025) and P1-30B-A3B (Silver-medal performance), trained via reinforcement learning and designed for Olympiad-level physics problem-solving. Combined with the PhysicsMinions agent framework, P1-235B-A22B achieves the top score on IPhO 2025.",
    tags: ["Reinforcement Learning"]
  },
  {
    title: "ToolOrchestra",
    link: "fc50eb47177f",
    date: "November 2025",
    description:
      "A reinforcement-learning-based method for training a small 8B-parameter Orchestrator model that coordinates a diverse set of tools, including web search, deterministic functions, specialized LLMs, and powerful generalist LLMs, the Orchestrator alternates between reasoning and tool calls over multiple turns.",
    tags: ["Reinforcement Learning", "Tool Use"]
  },
  {
    title: "VibeThinker-1.5B",
    link: "ef56288a0329",
    date: "November 2025",
    description:
      "A 1.5B-parameter dense language model developed using the innovative Spectrum-to-Signal Principle, which separates SFT and RL into distinct phases: “Diversity-Exploring Distillation” in SFT generates diverse solutions, while “MaxEnt-Guided Policy Optimization” in RL amplifies correct reasoning paths.",
    tags: ["Reinforcement Learning", "Language Models"]
  },
  {
    title: "Nemotron Cascade",
    link: "4fc163e3bdbf",
    date: "December 2025",
    description:
      "A family of general-purpose reasoning models trained with a cascaded, domain-wise reinforcement learning (Cascade RL) framework, starting from Qwen3-8B/14B base models. Instead of mixing heterogeneous prompts from many domains, it applies RL sequentially across domains, which simplifies infrastructure, allows domain-specific hyperparameters and curricula.",
    tags: ["Reinforcement Learning"]
  },
  {
    title: "Composition RL",
    link: "f537cd947a10",
    date: "February 2026",
    description:
      "Combats the growing number of too-easy (pass-rate-1) verifiable prompts in RLVR by automatically composing multiple existing problems into new, more challenging verifiable questions. It can follow a curriculum that gradually increases the compositional depth K and can improve reasoning performance in cross-domain settings.",
    tags: ["Reinforcement Learning", "Curriculum Learning", "Synthetic Data"]
  },
  {
    title: "QED Nano",
    link: "dd7f19dec9d7",
    date: "February 2026",
    description:
      "A compact 4B language model specifically trained to write mathematical proofs at the level of Olympiad-level problems. It achieves this through a three-stage process: supervised fine-tuning, reinforcement learning with rubric-based rewards, and training with a reasoning cache for iterative proof construction.",
    tags: ["LLM for Math", "Reinforcement Learning", "HuggingFace"]
  },
  {
    title: "Composer 1.5",
    link: "9bc921210902#f259",
    date: "February 2026",
    description:
      "An improved agentic coding model that builds on the original Composer 1 by scaling reinforcement learning 20x further and using more compute in post-training than in pretraining. It introduces self-summarization for handling long tasks.",
    tags: ["LLM for Code", "Reinforcement Learning"]
  },
  {
    title: "Likelihood-Based Reward Designs for General LLM Reasoning",
    link: "e889106eff08",
    date: "February 2026",
    description:
      "This paper systematically investigates likelihood-based reward functions, specifically probability and log-probability of reference answers, for fine-tuning LLMs on reasoning across both verifiable and non-verifiable domains. The authors find that using the log-probability of the reference answer as the reward consistently outperforms other methods, achieving comparable or better success rates than standard binary rewards in verifiable settings and performing similarly to supervised fine-tuning in non-verifiable settings.",
    tags: ["Reinforcement Learning", "Reward Design"]
  },
  {
    title: "Leanstral",
    link: "b9632dedf580#3b3a",
    date: "March 2026",
    description:
      "An open-source, highly efficient Lean 4 focused code and proof agent that uses a sparse architecture and parallel inference with Lean as a verifier to generate and formally prove implementations in realistic formal repositories. ",
    tags: ["LLM for Code", "Formal Methods"]
  },
  {
    title: "Nemotron Cascade 2",
    link: "1ac869c28c8c",
    date: "March 2026",
    description:
      "An open 30B Mixture-of-Experts model with 3B activated parameters that uses a Cascade RL framework plus multi-domain on-policy distillation to achieve best-in-class reasoning, strong agentic capabilities, and gold-medal-level performance on the 2025 IMO, IOI, and ICPC World Finals, approaching frontier open models.",
    tags: ["Reinforcement Learning"]
  },
  {
    title: "Composer 2",
    link: "9bc921210902",
    date: "March 2026",
    description:
      "A specialized coding model designed for agentic software engineering, excelling in long-term planning, multi-step execution, and coding intelligence. Trained through continued pretraining and reinforcement learning in a real-world environment, it achieves high performance on both internal and public benchmarks while being more cost-effective than general-purpose models.",
    tags: ["LLM for Code", "Reinforcement Learning"]
  },
  {
    title: "Apriel-1.5-OpenReasoner",
    link: "5826103aac57",
    date: "April 2026",
    description:
      "A 15B open-weight reasoning model trained using a multi-domain RL post-training recipe across five diverse domains: mathematics, code generation, instruction following, logical puzzles, and function calling. It incorporates adaptive domain sampling to maintain target domain ratios during asynchronous training and a difficulty-aware length penalty to encourage concise reasoning on easier problems while allowing longer traces for harder ones.",
    tags: ["Language Models", "Reinforcement Learning"]
  },
  {
    title: "Advancing Search Augmented Language Models",
    link: "bceb21866e26",
    date: "April 2026",
    description:
      "Perplexity develops search agents using a two-stage post-training pipeline: first, Supervised Fine-Tuning establishes deployment-critical behaviors like guardrails and stylistic consistency, then Reinforcement Learning refines accuracy and tool-use efficiency using a blend of verifiable QA and rubric-based chat data.",
    tags: ["Language Models", "Reinforcement Learning", "Search Agents", "Perplexity"]
  },
  {
    title: "Aryabhata 2",
    link: "9d3d23738731",
    date: "May 2026",
    description:
      "A competitive exams focused model, post-trained from GPT-OSS-20B by PhysicsWallah using three phased reinforcement learning on a rigorously cleaned and verified curriculum of Physics, Chemistry, Mathematics, and Reasoning questions.",
    tags: ["Reasoning Models", "Reinforcement Learning", "Competitive Exams"]
  },
  {
    title: "Reward Hacking in Rubric-Based RL",
    link: "cfefd83ed729",
    date: "May 2026",
    description:
      "Shows that in rubric-based reinforcement learning, reward hacking persists even with stronger verifiers, as gains concentrate on presence-based criteria like completeness but broader quality declines, highlighting the limitations of rubrics and the need for better diagnostics such as the self-internalization gap.",
    tags: ["Reinforcement Learning"]
  },
  {
    title: "Composer 2.5",
    link: "9bc921210902#4db4",
    date: "May 2026",
    description:
      "Built on Moonshot's Kimi K2.5 checkpoint, improves intelligence and usability over Composer 2 by scaling training, expanding synthetic task diversity (25x increase), and introducing targeted RL with textual feedback to adjust behavior at problem points.",
    tags: ["LLM for Code", "Reinforcement Learning"]
  },
  {
    title: "Policy-Aware Rubric Reward (POW3R)",
    link: "6fa98f57e4f9",
    date: "May 2026",
    description:
      "A policy-aware rubric reward framework that dynamically reallocates training pressure toward rubric criteria that currently distinguish the model's outputs, based on rollout-level contrastiveness measured by the standard deviation of judge verdicts. It blends and clips this signal to maintain a learning floor for saturated or failed criteria, then renormalizes within rubric categories to preserve the original human weights and category balance, resulting in more informative reward signals and faster, more robust learning.",
    tags: ["Reinforcement Learning", "Reward Design"]
  },
  {
    title: "MAI Thinking-1",
    link: "e5afeca9bbfc",
    date: "June 2026",
    description:
      "A 35B active/1T total parameter MoE model developed from scratch trained solely on 30T tokens of high-quality, exclusively human-generated data with no synthetic, open-source, or third-party model distillation. Its development uses empirically-driven, scalable pre-training, robust reinforcement learning and a hill-climbing optimization process.",
    tags: ["Language Models", "Reinforcement Learning", "Mixture-of-Experts"]
  },
  {
    title: "Rubric Guided Self Distillation",
    link: "bd61a188450f",
    date: "June 2026",
    description:
      "A verifier-free training method for open-ended, rubric-graded tasks, where the teacher is conditioned on the rubric and its outputs are distilled, token-by-token, into an unconditioned student model, thus internalizing rubric criteria without relying on expensive LLM verifier calls.",
    tags: ["Reinforcement Learning", "Reward Design", "Self-Distillation"]
  },
  {
    title: "VibeThinker-3B",
    link: "a82a4fe1299f",
    date: "June 2026",
    description:
      "A compact dense model, developed to investigate how far verifiable reasoning can be pushed within a strictly small-model regime. Building upon the Spectrum-to-Signal post-training paradigm, the model is systematically enhanced through an optimized pipeline that includes curriculum-based supervised fine-tuning, multi-domain reinforcement learning, and offline self-distillation.",
    tags: ["Language Models", "Reinforcement Learning"]
  }
];
