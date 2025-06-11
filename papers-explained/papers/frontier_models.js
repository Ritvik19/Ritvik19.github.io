const frontier_models = [
  {
    title: "ChatGPT",
    link: "https://ritvik19.medium.com/papers-explained-54-chatgpt-78387333268f",
    date: "November 2022",
    description:
      "An interactive model designed to engage in conversations, built on top of GPT 3.5.",
    tags: ["Language Models", "Transformer Decoder", "OpenAI", "GPT"],
  },
  {
    title: "GPT 4",
    link: "https://ritvik19.medium.com/papers-explained-67-gpt-4-fc77069b613e",
    date: "March 2023",
    description:
      "A multimodal transformer model pre-trained to predict the next token in a document, which can accept image and text inputs and produce text outputs.",
    tags: ["Language Models", "Transformer Decoder", "OpenAI", "GPT"],
  },
  {
    title: "GPT-4V",
    link: "https://ritvik19.medium.com/papers-explained-68-gpt-4v-6e27c8a1d6ea",
    date: "September 2023",
    description:
      "A multimodal model that combines text and vision capabilities, allowing users to instruct it to analyze image inputs.",
    tags: ["Multimodal Models", "OpenAI", "GPT"],
  },
  {
    title: "Grok 1",
    link: "https://ritvik19.medium.com/papers-explained-186-grok-0d9f1aef69be",
    date: "November 2023",
    description:
      "A 314B Mixture-of-Experts model, modeled after the Hitchhiker's Guide to the Galaxy, designed to be witty.",
    tags: [
      "Language Models",
      "Transformer Decoder",
      "Mixtures of Experts",
      "Grok",
    ],
  },
  {
    title: "Gemini 1.0",
    link: "https://ritvik19.medium.com/papers-explained-80-gemini-1-0-97308ef96fcd",
    date: "December 2023",
    description:
      "A family of highly capable multi-modal models, trained jointly across image, audio, video, and text data for the purpose of building a model with strong generalist capabilities across modalities.",
    tags: ["Multimodal Models", "Gemini"],
  },
  {
    title: "Gemini 1.5 Pro",
    link: "https://ritvik19.medium.com/papers-explained-105-gemini-1-5-pro-029bbce3b067",
    date: "February 2024",
    description:
      "A highly compute-efficient multimodal mixture-of-experts model that excels in long-context retrieval tasks and understanding across text, video, and audio modalities.",
    tags: ["Multimodal Models", "Mixtures of Experts", "Gemini"],
  },
  {
    title: "Claude 3",
    link: "https://ritvik19.medium.com/papers-explained-181-claude-89dd45e35d92",
    date: "March 2024",
    description:
      "A family of VLMs consisting of Haiku, Sonnet, and Opus models, sets new industry standards for cognitive tasks, offering varying levels of intelligence, speed, and cost-efficiency.",
    tags: ["Multimodal Models", "Anthropic", "Claude"],
  },
  {
    title: "Grok 1.5",
    link: "https://ritvik19.medium.com/papers-explained-186-grok-0d9f1aef69be",
    date: "March 2024",
    description:
      "An advancement over grok, capable of long context understanding up to 128k tokens and advanced reasoning.",
    tags: [
      "Language Models",
      "Transformer Decoder",
      "Mixtures of Experts",
      "Grok",
    ],
  },
  {
    title: "Grok 1.5 V",
    link: "https://ritvik19.medium.com/papers-explained-186-grok-0d9f1aef69be",
    date: "April 2024",
    description: "The first multimodal model in the grok series.",
    tags: ["Multimodal Models", "Grok"],
  },
  {
    title: "GPT-4o",
    link: "https://ritvik19.medium.com/papers-explained-185-gpt-4o-a234bccfd662",
    date: "May 2024",
    description:
      "An omni model accepting and generating various types of inputs and outputs, including text, audio, images, and video.",
    tags: ["Multimodal Models", "OpenAI", "GPT"],
  },
  {
    title: "Gemini 1.5 Flash",
    link: "https://ritvik19.medium.com/papers-explained-142-gemini-1-5-flash-415e2dc6a989",
    date: "May 2024",
    description:
      "A more lightweight variant of the Gemini 1.5 pro, designed for efficiency with minimal regression in quality, making it suitable for applications where compute resources are limited.",
    tags: ["Multimodal Models", "Gemini"],
  },
  {
    title: "Claude 3.5 Sonnet",
    link: "https://ritvik19.medium.com/papers-explained-181-claude-89dd45e35d92#2a14",
    date: "June 2024",
    description:
      "Surpasses previous versions and competitors in intelligence, speed, and cost-efficiency, excelling in graduate-level reasoning, undergraduate-level knowledge, coding proficiency, and visual reasoning.",
    tags: ["Multimodal Models", "Anthropic", "Claude"],
  },
  {
    title: "Mistral Large 2",
    link: "https://ritvik19.medium.com/papers-explained-mistral-7b-b9632dedf580#301d",
    date: "July 2024",
    description:
      "A 123B model, offers significant improvements in code generation, mathematics, and reasoning capabilities, advanced function calling, a 128k context window, and supports dozens of languages and over 80 coding languages.",
    tags: ["Language Models", "Transformer Decoder", "Mistral"],
  },
  {
    title: "GPT-4o mini",
    link: "https://ritvik19.medium.com/papers-explained-185-gpt-4o-a234bccfd662#08b9",
    date: "July 2024",
    description:
      "A cost-efficient small model that outperforms GPT-4 on chat preferences, enabling a broad range of tasks with low latency and supporting text, vision, and multimodal inputs and outputs.",
    tags: ["Multimodal Models", "OpenAI", "GPT"],
  },
  {
    title: "Grok 2",
    link: "https://ritvik19.medium.com/papers-explained-186-grok-0d9f1aef69be",
    date: "August 2024",
    description:
      "A frontier language model with state-of-the-art capabilities in chat, coding, and reasoning on par with Claude 3.5 Sonnet and GPT-4-Turbo.",
    tags: ["Multimodal Models", "Grok"],
  },

  {
    title: "o1",
    link: "https://ritvik19.medium.com/papers-explained-211-o1-163fd9c7308e",
    date: "September 2024",
    description:
      "A large language model trained with reinforcement learning to think before answering, producing a long internal chain of thought before responding.",
    tags: [
      "Language Models",
      "Transformer Decoder",
      "OpenAI",
      "GPT",
      "Reasoning",
    ],
  },
  {
    title: "o1-mini",
    link: "https://ritvik19.medium.com/papers-explained-211-o1-163fd9c7308e#f16a",
    date: "September 2024",
    description:
      "A cost-efficient reasoning model, excelling at STEM, especially math and coding, nearly matching the performance of OpenAI o1 on evaluation benchmarks.",
    tags: [
      "Language Models",
      "Transformer Decoder",
      "OpenAI",
      "GPT",
      "Reasoning",
    ],
  },
  {
    title: "Claude 3.5 Haiku",
    link: "https://ritvik19.medium.com/papers-explained-181-claude-89dd45e35d92#9637",
    date: "October 2024",
    description:
      "A fast and affordable language model that excels in tasks such as coding, reasoning, and content creation.",
    tags: ["Multimodal Models", "Anthropic", "Claude"],
  },
  {
    title: "Gemini 2.0 Flash",
    link: "https://ritvik19.medium.com/papers-explained-367-gemini-models-97b6b71e0054#3a09",
    date: "December 2024",
    description:
      "A faster, more powerful large language model than its predecessor, boasting enhanced multimodal input/output (including native image generation and steerable multilingual audio), native tool use (like Google Search and code execution), and enabling a new class of agentic experiences through improved reasoning and complex instruction following.",
    tags: ["Multimodal Models", "Gemini"],
  },
  {
    title: "o3-mini",
    link: "https://ritvik19.medium.com/papers-explained-211-o1-163fd9c7308e#9875",
    date: "January 2025",
    description:
      "A cost-efficient reasoning model, excelling in STEM fields, while maintaining low latency. It supports features like function calling, structured outputs, and developer messages, and offers adjustable reasoning effort levels (low, medium, high) for optimized performance.",
    tags: [
      "Language Models",
      "Transformer Decoder",
      "OpenAI",
      "GPT",
      "Reasoning",
    ],
  },
  {
    title: "Gemini 2.0 Flash Lite",
    link: "https://ritvik19.medium.com/papers-explained-367-gemini-models-97b6b71e0054#d87a",
    date: "February 2025",
    description:
      "A production-ready language model offering enhanced performance over its predecessor (1.5 Flash) in reasoning, multimodal tasks, math, and factual accuracy, along with cost-effective pricing, making it ideal for applications requiring long context windows.",
    tags: ["Multimodal Models", "Gemini"],
  },
  {
    title: "Grok 3 Beta",
    link: "https://ritvik19.medium.com/papers-explained-186-grok-0d9f1aef69be#2fe9",
    date: "February 2025",
    description:
      "Trained on a massive scale with 10x the compute of previous models, exhibiting advanced reasoning, and a 1 million token context window, with a Think mode and is available in a cost-efficient mini version for STEM tasks. Grok agents like DeepSearch further enhance its capabilities by combining reasoning with tool use like internet access and code interpreters.",
    tags: ["Language Models", "Transformer Decoder", "Grok"],
  },
  {
    title: "Claude 3.7 Sonnet",
    link: "https://ritvik19.medium.com/papers-explained-181-claude-89dd45e35d92#a30f",
    date: "February 2025",
    description:
      "A hybrid reasoning model offering both fast responses and detailed, user-visible step-by-step thinking controllable by a token budget, featuring integration with GitHub and a new coding tool (Claude Code).",
    tags: ["Multimodal Models", "Anthropic", "Claude"],
  },
  {
    title: "GPT-4.5",
    link: "https://ritvik19.medium.com/papers-explained-350-gpt-4-5-dc1d4b097ad1",
    date: "February 2025",
    description:
      "A research preview model representing an advancement in scaling unsupervised learning through increased compute, data, architecture, and optimization innovations, resulting in broader knowledge, deeper understanding, reduced hallucinations, and increased reliability. It excels in tasks requiring natural conversation, creativity, and understanding human intent due to new scalable training techniques derived from smaller models.",
    tags: ["Language Models", "Transformer Decoder", "OpenAI", "GPT"],
  },
  {
    title: "Gemini 2.5 Pro",
    link: "https://ritvik19.medium.com/papers-explained-367-gemini-models-97b6b71e0054#5485",
    date: "March 2025",
    description:
      "Google's most advanced experimental AI model, excelling in complex tasks due to its enhanced reasoning and coding capabilities, achieving state-of-the-art performance on various benchmarks, and building upon Gemini's strengths of native multimodality and a long context window.",
    tags: ["Multimodal Models", "Gemini", "Reasoning"],
  },
  {
    title: "GPT-4.1",
    link: "https://ritvik19.medium.com/papers-explained-350-gpt-4-5-dc1d4b097ad1#a605",
    date: "April 2025",
    description:
      "Advanced language models offering superior coding, instruction following, and long-context comprehension (up to 1 million tokens) compared to predecessors, with lower cost and latency, excelling in real-world tasks and boast enhanced vision capabilities",
    tags: ["Language Models", "Transformer Decoder", "OpenAI", "GPT"],
  },
  {
    title: "Gemini 2.5 Flash",
    link: "https://ritvik19.medium.com/papers-explained-367-gemini-models-97b6b71e0054#b40c",
    date: "April 2025",
    description:
      "A hybrid reasoning model that allows developers to control a 'thinking' process for improved accuracy on complex tasks, offering a customizable thinking budget to balance quality, cost, and speed.",
    tags: ["Multimodal Models", "Gemini", "Reasoning"],
  },
  {
    title: "o3",
    link: "https://ritvik19.medium.com/papers-explained-211-o1-163fd9c7308e#4cfa",
    date: "April 2025",
    description:
      "Reasoning model, excelling in complex tasks requiring multi-faceted analysis, especially those involving visual data like images and charts. It leverages tools like web search, code execution, and image generation to provide comprehensive answers and demonstrates significantly improved performance across various benchmarks and real-world tasks.",
    tags: [
      "Language Models",
      "Transformer Decoder",
      "OpenAI",
      "GPT",
      "Reasoning",
      "Multimodal",
    ],
  },
  {
    title: "o4-mini",
    link: "https://ritvik19.medium.com/papers-explained-211-o1-163fd9c7308e#4cfa",
    date: "April 2025",
    description:
      "A smaller, faster, and more cost-effective reasoning model optimized for speed and efficiency. While powerful in math, coding, and visual tasks, it maintains strong performance in other areas and supports higher usage limits due to its efficiency, making it suitable for high-volume applications.",
    tags: [
      "Language Models",
      "Transformer Decoder",
      "OpenAI",
      "GPT",
      "Reasoning",
      "Multimodal",
    ],
  },
  {
    title: "Mistral Medium 3",
    link: "https://ritvik19.medium.com/papers-explained-mistral-7b-b9632dedf580#0ea6",
    date: "May 2025",
    description:
      "A cost-effective, high-performing language model designed for enterprise use, offering SOTA performance comparable to Claude Sonnet 3.7 at 8x lower cost, excelling in professional tasks like coding and STEM.",
    tags: ["Language Models", "Transformer Decoder", "Mistral"],
  },
  {
    title: "Claude 4",
    link: "https://ritvik19.medium.com/papers-explained-181-claude-89dd45e35d92#90f2",
    date: "May 2025",
    description:
      "Hybrid models offering two modes: near-instant responses and extended thinking for deeper reasoning, setting new standards for coding, advanced reasoning, and AI agents.",
    tags: ["Multimodal Models", "Anthropic", "Claude"],
  },
];
