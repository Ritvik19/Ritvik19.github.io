const reward_models = [
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
    link: "",
    date: "May 2025",
    description:
        "Enhance reward model performance by executing a deliberate reasoning process before generating final rewards, leveraging chain-of-thought reasoning and additional test-time compute for complex queries.",
    tags: ["Outcome Reward Model"],
  },
  {
    title: "RewardAnything",
    link: "",
    date: "June 2025",
    description:
      "A generalizable reward model designed to explicitly follow natural language principles, addressing the limitations of current RMs that are rigidly aligned to fixed preference datasets trained using GRPO.",
    tags: ["Reward Models", "Language Models"],
  },
];
