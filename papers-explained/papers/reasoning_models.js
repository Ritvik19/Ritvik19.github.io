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
];
