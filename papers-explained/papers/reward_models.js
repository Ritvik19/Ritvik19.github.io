const reward_models = [
  {
    title: "Shepherd",
    link: "",
    date: "August 2023",
    description:
      "A language model tuned to critique model responses and suggest refinements, trained on a high-quality feedback dataset curated from community feedback and human annotations. The model is named Shepherd, as it guides Llamas.",
    tags: ["LLM Evaluation"],
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
    title: "Prometheus Vision",
    link: "https://ritvik19.medium.com/papers-explained-401-prometheus-vision-10ce3fbdf5a0",
    date: "January 2024",
    description:
      "The first open-source VLM evaluator model, trained using the Perception Collection dataset, which contains 15K fine-grained score rubrics. It excels at assessing VLMs based on user-defined, fine-grained criteria and demonstrates a high correlation with human evaluators and GPT-4V, making it a cost-effective and transparent alternative for VLM evaluation.",
    tags: ["VLM Evaluation"],
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
    title: "CriticGPT",
    link: "https://ritvik19.medium.com/papers-explained-224-criticgpt-6d9af57451fa",
    date: "June 2024",
    description:
      "A model based on GPT-4 trained with RLHF to catch errors in ChatGPT's code output, accepts a question-answer pair as input and outputs a structured critique that highlights potential problems in the answer.",
    tags: ["Language Models", "OpenAI", "GPT"],
  },
  {
    title: "J1",
    link: "https://ritvik19.medium.com/papers-explained-385-j1-8245b6ee64df",
    date: "May 2025",
    description:
      "A reinforcement learning approach for training LLM-as-a-Judge models that converts both verifiable and non-verifiable prompts into judgment tasks with verifiable rewards to incentivize thinking and mitigate judgment bias.",
    tags: ["Outcome Reward Model"],
  },
  {
    title: "ThinkPRM",
    link: "https://ritvik19.medium.com/papers-explained-368-thinkprm-0f530cc98ca4",
    date: "April 2025",
    description:
      "A generative PRM that verifies step-by-step solutions using a long CoT reasoning process. Trained on a small amount of synthetic data, it outperforms both discriminative PRMs trained on significantly more data and LLM-as-a-Judge baselines in various reasoning tasks and verification scenarios.",
    tags: ["LLM for Math", "Process Reward Model"],
  },
  {
    title: "RM-R1",
    link: "https://ritvik19.medium.com/papers-explained-369-rm-r1-5a1b5f7ff27a",
    date: "May 2025",
    description:
      "A family Generative Reward Models, called as Reasoning Reward Models that formulates reward modeling as a reasoning task, enhancing interpretability and performance. Trained via a reasoning-oriented pipeline involving structured reasoning distillation and reinforcement learning with verifiable rewards, RM-R1 generates reasoning traces or chat-specific rubrics to evaluate candidate responses.",
    tags: ["LLM for Math", "Outcome Reward Model"],
  },
  {
    title: "Reward Reasoning Model",
    link: "https://ritvik19.medium.com/papers-explained-400-reward-reasoning-model-109dba633c27",
    date: "May 2025",
    description:
        "Enhance reward model performance by executing a deliberate reasoning process before generating final rewards, leveraging chain-of-thought reasoning and additional test-time compute for complex queries.",
    tags: ["Outcome Reward Model"],
  },
  {
    title: "RewardAnything",
    link: "https://ritvik19.medium.com/papers-explained-399-rewardanything-f100d1d97c8b",
    date: "June 2025",
    description:
      "A generalizable reward model designed to explicitly follow natural language principles, addressing the limitations of current RMs that are rigidly aligned to fixed preference datasets trained using GRPO.",
    tags: ["Reward Models", "Language Models"],
  },
  {
    title: "One Token to Fool LLM-as-a-Judge",
    link: "https://ritvik19.medium.com/papers-explained-424-one-token-to-fool-llm-as-a-judge-b8d30ed4d281",
    date: "July 2025",
    description:
      "Investigates that generative reward models are susceptible to superficial manipulations like non-word symbols or reasoning openers, leading to false positive rewards. To address this, it introduces a data augmentation strategy and trains a more robust generative reward model, highlighting the need for more reliable LLM-based evaluation methods.",
    tags: ["LLM Evaluation"],
  }
];
