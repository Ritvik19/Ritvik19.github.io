const llm_for_math = [
  {
    title: "Wizard Math",
    link: "https://ritvik19.medium.com/papers-explained-review-12-llms-for-maths-1597e3c7251c#ecf7",
    date: "August 2023",
    description:
      "Proposes Reinforcement Learning from Evol-Instruct Feedback (RLEIF) method, applied to Llama-2 to enhance the mathematical reasoning abilities.",
    tags: ["LLM for Math"],
  },
  {
    title: "MAmmoTH",
    link: "https://ritvik19.medium.com/papers-explained-review-12-llms-for-maths-1597e3c7251c#8cd9",
    date: "September 2023",
    description:
      "A series of LLMs specifically designed for general math problem-solving, trained on MathInstruct, a dataset compiled from 13 math datasets with intermediate rationales that combines chain-of-thought and program-of-thought approaches to accommodate different thought processes for various math problems.",
    tags: ["LLM for Math"],
  },
  {
    title: "MetaMath",
    link: "https://ritvik19.medium.com/papers-explained-review-12-llms-for-maths-1597e3c7251c#bd74",
    date: "September 2023",
    description:
      "A fine-tuned language model that specializes in mathematical reasoning, achieved by bootstrapping mathematical questions from multiple perspectives without extra knowledge and then fine-tuning an LLaMA-2 model on the resulting dataset.",
    tags: ["LLM for Math"],
  },
  {
    title: "ToRA",
    link: "https://ritvik19.medium.com/papers-explained-review-12-llms-for-maths-1597e3c7251c#feb4",
    date: "September 2023",
    description:
      "A series of large language models that seamlessly integrate natural language reasoning with external tools to solve complex mathematical problems.",
    tags: ["LLM for Math"],
  },
  {
    title: "Math Coder",
    link: "https://ritvik19.medium.com/papers-explained-review-12-llms-for-maths-1597e3c7251c#102e",
    date: "October 2023",
    description:
      "The first systematic study that explicitly integrates natural language reasoning, code generation, and feedback from execution results into open-source pre-trained large language models.",
    tags: ["LLM for Math"],
  },
  {
    title: "MuggleMath",
    link: "https://ritvik19.medium.com/papers-explained-review-12-llms-for-maths-1597e3c7251c#c53e",
    date: "October 2023",
    description:
      "GSM8K and MATH datasets are augmented using query augmentation and response augmentation to fine tune Llama 2 models.",
    tags: ["LLM for Math"],
  },
  {
    title: "Llemma",
    link: "https://ritvik19.medium.com/papers-explained-review-12-llms-for-maths-1597e3c7251c#1b38",
    date: "October 2023",
    description:
      "An LLM for mathematics, formed by continued pretraining of Code Llama on a mixture of scientific papers, web data containing mathematics, and mathematical code.",
    tags: ["LLM for Math"],
  },
  {
    title: "MuMath",
    link: "https://ritvik19.medium.com/papers-explained-review-12-llms-for-maths-1597e3c7251c#25c2",
    date: "December 2023",
    description:
      "A multi-perspective augmentation dataset for mathematics that combines strengths from tool-free methods, broadening the scope of augmentation techniques to enhance mathematical reasoning capabilities.",
    tags: ["LLM for Math"],
  },
  {
    title: "Math Shepherd",
    link: "https://ritvik19.medium.com/papers-explained-366-math-shepherd-234b1bdfbcae",
    date: "December 2023",
    description:
      "A process reward model that automatically scores the correctness of each step in a math problem solution, using this to rerank LLM outputs and to reinforce LLMs via step-by-step PPO, without human annotations or external tools. It leverages a Monte Carlo Tree Search inspired approach where an LLM decodes multiple subsequent paths from each step and the step's score reflects how many lead to the correct final answer.",
    tags: ["LLM for Math", "Process Reward Model"],
  },
  {
    title: "MMIQC",
    link: "https://ritvik19.medium.com/papers-explained-review-12-llms-for-maths-1597e3c7251c#3a88",
    date: "January 2024",
    description:
      "Curates a data set by sampling Meta Math, Iterative Question Composition, Utilizing Stack Exchange, Augmenting Similar Problems, Answer Augmentation and Question Bootstrapping to finetune Mistral, Llemma, DeepSeek and Qwen.",
    tags: ["LLM for Math"],
  },
  {
    title: "DeepSeek Math",
    link: "https://ritvik19.medium.com/papers-explained-review-12-llms-for-maths-1597e3c7251c#7760",
    date: "February 2024",
    description:
      "A language model for mathematics leveraging  a variant of Proximal Policy Optimization (PPO) called Group Relative Policy Optimization (GRPO), without relying on external toolkits or voting techniques.",
    tags: ["LLM for Math"],
  },
  {
    title: "Open Math Instruct 1",
    link: "https://ritvik19.medium.com/papers-explained-review-12-llms-for-maths-1597e3c7251c#8709",
    date: "February 2024",
    description:
      "A math instruction tuning dataset containing 1.8M code interpreter styled problem-solution pairs for GSM8K and MATH dataset, generated using Mixtral.",
    tags: ["LLM for Math"],
  },
  {
    title: "Math Orca",
    link: "https://ritvik19.medium.com/papers-explained-review-12-llms-for-maths-1597e3c7251c#80d0",
    date: "February 2024",
    description:
      "A fine tuned Mistral-7B that excels at math problems without external tools, utilizing a high-quality synthetic dataset of 200K problems created through multi-agent collaboration and an iterative learning process that involves practicing problem-solving, receiving feedback, and learning from preference pairs incorporating the model's solutions and feedback.",
    tags: ["LLM for Math"],
  },
  {
    title: "Math Genie",
    link: "https://ritvik19.medium.com/papers-explained-review-12-llms-for-maths-1597e3c7251c#8894",
    date: "February 2024",
    description:
      "A framework that generates diverse and reliable math problems by iteratively augmenting small-scale problem-solution datasets and back-translating, followed by generating and verifying code-integrated solutions",
    tags: ["LLM for Math"],
  },
  {
    title: "Xwin-Math",
    link: "https://ritvik19.medium.com/papers-explained-review-12-llms-for-maths-1597e3c7251c#f71c",
    date: "March 2024",
    description:
      "Scales up GSM 8K and MATH dataset to 1.44M questions using data synthesis and then Llama 2, Mistral, Llemma are finetuned on the curated datasets.",
    tags: ["LLM for Math"],
  },
  {
    title: "MuMath Code",
    link: "https://ritvik19.medium.com/papers-explained-review-12-llms-for-maths-1597e3c7251c#9fb8",
    date: "May 2024",
    description:
      "Integrates tool use and data augmentation by finetuning Llama-2 on a dataset of math questions with code-nested solutions.",
    tags: ["LLM for Math"],
  },
  {
    title: "Skywork-Math",
    link: "https://ritvik19.medium.com/papers-explained-352-skywork-math-d8f2aa59ddcf",
    date: "July 2024",
    description:
      "A series of 7B parameter LLMs, fine-tuned solely on a new 2.5M instance dataset called Skywork-MathQA generated through a novel two-stage pipeline employing diverse seed problems and augmentation of hard problems.",
    tags: ["LLM for Math"],
  },
  {
    title: "Numina Math",
    link: "https://ritvik19.medium.com/papers-explained-review-12-llms-for-maths-1597e3c7251c#b743",
    date: "July 2024",
    description:
      "Winner of the 1st AIMO Progress Prize, based on DeepSeek Math, finetuned in two stages: CoT and ToRA.",
    tags: ["LLM for Math"],
  },
  {
    title: "Qwen 2 Math",
    link: "https://ritvik19.medium.com/papers-explained-review-12-llms-for-maths-1597e3c7251c#e37a",
    date: "August 2024",
    description:
      "A model series focused on mathematical capabilities, built upon the Qwen2, outperforming proprietary models such as GPT-4o and Claude 3.5 in math-related tasks.",
    tags: ["LLM for Math"],
  },
  {
    title: "Qwen 2.5 Math",
    link: "https://ritvik19.medium.com/papers-explained-review-12-llms-for-maths-1597e3c7251c#eb40",
    date: "September 2024",
    description:
      "An upgrade of Qwen 2 Math series with improved performance and expanded support to Tool Integrated Reasoning.",
    tags: ["LLM for Math"],
  },
  {
    title: "Open Math Instruct 2",
    link: "https://ritvik19.medium.com/papers-explained-review-12-llms-for-maths-1597e3c7251c#ff3a",
    date: "October 2024",
    description:
      "A math instruction tuning dataset containing 14M question-solution pairs (≈ 600K unique questions) augmented from GSM8K and MATH dataset, generated using Llama 3.2 405B.",
    tags: ["LLM for Math"],
  },
  {
    title: "Math Coder 2",
    link: "https://ritvik19.medium.com/papers-explained-review-12-llms-for-maths-1597e3c7251c#97c9",
    date: "October 2024",
    description:
      "Creates Math Code Pile by filtering web data and then extracting and translating mathematical data from it to create a corpus of interleaved reasoning and code data.",
    tags: ["LLM for Math"],
  },
  {
    title: "AceMath",
    link: "https://ritvik19.medium.com/papers-explained-review-12-llms-for-maths-1597e3c7251c#65b1",
    date: "December 2024",
    description:
      "A suite of math instruction-following models, trained through a two-stage SFT process focusing on general and math-specific reasoning, utilizing high-quality synthetic data and a specialized reward model (AceMath-RM) trained with diverse responses.",
    tags: ["LLM for Math"],
  },
  {
    title: "s1",
    link: "https://ritvik19.medium.com/papers-explained-353-s1-f071ec558fa0",
    date: "January 2025",
    description:
      "Proposes a new test-time scaling approach that achieves strong reasoning performance after supervised finetuning on a small, curated 1,000-sample dataset (s1K) of diverse, difficult questions with reasoning traces, and employs 'budget forcing' to control test-time compute by limiting or extending the model's thinking process.",
    tags: ["LLM for Math"],
  },
  {
    title: "LIMO",
    link: "https://ritvik19.medium.com/papers-explained-328-limo-261765a5616c",
    date: "February 2025",
    description:
      "Challenges the notion that complex reasoning requires massive datasets by achieving state-of-the-art performance on mathematical reasoning benchmarks using only 817 curated training examples and emphasizing high-quality reasoning chains.",
    tags: ["LLM for Math"],
  },
  {
    title: "MathFusion",
    link: "https://ritvik19.medium.com/papers-explained-351-mathfusion-f612d91667c4",
    date: "March 2025",
    description:
      "MathFusion is a novel framework that improves mathematical reasoning in LLMs by synthesizing new problems from existing ones using three fusion strategies: sequential, parallel, and conditional, to capture relational structures in mathematical knowledge.",
    tags: ["LLM for Math", "Synthetic Data"],
  },
  {
    title: "Nemotron CrossThink",
    link: "https://ritvik19.medium.com/papers-explained-360-nemotron-crossthink-3e804e878541",
    date: "April 2025",
    description:
      "A framework that uses reinforcement learning to improve large language models' reasoning abilities across beyond math to diverse tasks by incorporating multi-domain data (STEM, humanities, social sciences, etc.) with varied formats (multiple-choice, open-ended) and verifiable answers, optimizing data blending strategies for effective training.",
    tags: ["LLM for Math"],
  },
  {
    title: "OpenMath Nemotron",
    link: "https://ritvik19.medium.com/papers-explained-355-openmath-nemotron-d73c6000148a",
    date: "April 2025",
    description:
      "A series of mathematical reasoning models (1.5B, 7B, 14B, and 32B parameters), including a winning submission to the AI Mathematical Olympiad - Progress Prize 2 (AIMO-2) competition, trained on a massive dataset of 540K unique math problems and 3.2M solutions (OpenMathReasoning dataset) capable of CoT and TIR with Python code execution.",
    tags: ["LLM for Math"],
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
];
