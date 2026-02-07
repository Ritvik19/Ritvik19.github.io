const frontier_models = [
  {
    title: "ChatGPT",
    link: "78387333268f",
    date: "November 2022",
    description:
      "An interactive model designed to engage in conversations, built on top of GPT 3.5.",
    tags: ["Frontier Model", "OpenAI", "GPT"],
  },
  {
    title: "GPT 4",
    link: "fc77069b613e",
    date: "March 2023",
    description:
      "A multimodal transformer model pre-trained to predict the next token in a document, which can accept image and text inputs and produce text outputs.",
    tags: ["Frontier Model", "OpenAI", "GPT"],
  },
  {
    title: "GPT 4V",
    link: "6e27c8a1d6ea",
    date: "September 2023",
    description:
      "A multimodal model that combines text and vision capabilities, allowing users to instruct it to analyze image inputs.",
    tags: ["Frontier Model", "OpenAI", "GPT"],
  },
  {
    title: "Grok 1",
    link: "0d9f1aef69be",
    date: "November 2023",
    description:
      "A 314B Mixture-of-Experts model, modeled after the Hitchhiker's Guide to the Galaxy, designed to be witty.",
    tags: ["Frontier Model", "Mixtures of Experts", "Grok" ],
  },
  {
    title: "Gemini 1.0",
    link: "97308ef96fcd",
    date: "December 2023",
    description:
      "A family of highly capable multi-modal models, trained jointly across image, audio, video, and text data for the purpose of building a model with strong generalist capabilities across modalities.",
    tags: ["Frontier Model", "Gemini"],
  },
  {
    title: "Gemini 1.5 Pro",
    link: "029bbce3b067",
    date: "February 2024",
    description:
      "A highly compute-efficient multimodal mixture-of-experts model that excels in long-context retrieval tasks and understanding across text, video, and audio modalities.",
    tags: ["Frontier Model", "Mixtures of Experts", "Gemini"],
  },
  {
    title: "Claude 3",
    link: "89dd45e35d92",
    date: "March 2024",
    description:
      "A family of VLMs consisting of Haiku, Sonnet, and Opus models, sets new industry standards for cognitive tasks, offering varying levels of intelligence, speed, and cost-efficiency.",
    tags: ["Frontier Model", "Anthropic", "Claude"],
  },
  {
    title: "Grok 1.5",
    link: "0d9f1aef69be",
    date: "March 2024",
    description:
      "An advancement over grok, capable of long context understanding up to 128k tokens and advanced reasoning.",
    tags: ["Frontier Model", "Mixtures of Experts", "Grok"],
  },
  {
    title: "Grok 1.5 V",
    link: "0d9f1aef69be",
    date: "April 2024",
    description: "The first multimodal model in the grok series.",
    tags: ["Frontier Model", "Grok"],
  },
  {
    title: "GPT 4o",
    link: "a234bccfd662",
    date: "May 2024",
    description:
      "An omni model accepting and generating various types of inputs and outputs, including text, audio, images, and video.",
    tags: ["Frontier Model", "OpenAI", "GPT"],
  },
  {
    title: "Gemini 1.5 Flash",
    link: "415e2dc6a989",
    date: "May 2024",
    description:
      "A more lightweight variant of the Gemini 1.5 pro, designed for efficiency with minimal regression in quality, making it suitable for applications where compute resources are limited.",
    tags: ["Frontier Model", "Gemini"],
  },
  {
    title: "Claude 3.5 Sonnet",
    link: "89dd45e35d92#2a14",
    date: "June 2024",
    description:
      "Surpasses previous versions and competitors in intelligence, speed, and cost-efficiency, excelling in graduate-level reasoning, undergraduate-level knowledge, coding proficiency, and visual reasoning.",
    tags: ["Frontier Model", "Anthropic", "Claude"],
  },
  {
    title: "Mistral Large 2",
    link: "b9632dedf580#301d",
    date: "July 2024",
    description:
      "A 123B model, offers significant improvements in code generation, mathematics, and reasoning capabilities, advanced function calling, a 128k context window, and supports dozens of languages and over 80 coding languages.",
    tags: ["Frontier Model", "Mistral"],
  },
  {
    title: "GPT 4o mini",
    link: "a234bccfd662#08b9",
    date: "July 2024",
    description:
      "A cost-efficient small model that outperforms GPT-4 on chat preferences, enabling a broad range of tasks with low latency and supporting text, vision, and multimodal inputs and outputs.",
    tags: ["Frontier Model", "OpenAI", "GPT"],
  },
  {
    title: "Grok 2",
    link: "0d9f1aef69be",
    date: "August 2024",
    description:
      "A frontier language model with state-of-the-art capabilities in chat, coding, and reasoning on par with Claude 3.5 Sonnet and GPT-4-Turbo.",
    tags: ["Frontier Model", "Grok"],
  },

  {
    title: "o1",
    link: "163fd9c7308e",
    date: "September 2024",
    description:
      "A large language model trained with reinforcement learning to think before answering, producing a long internal chain of thought before responding.",
    tags: ["Frontier Model", "OpenAI", "GPT", "Reasoning"],
  },
  {
    title: "o1-mini",
    link: "163fd9c7308e#f16a",
    date: "September 2024",
    description:
      "A cost-efficient reasoning model, excelling at STEM, especially math and coding, nearly matching the performance of OpenAI o1 on evaluation benchmarks.",
    tags: ["Frontier Model", "OpenAI", "GPT", "Reasoning"],
  },
  {
    title: "Claude 3.5 Haiku",
    link: "89dd45e35d92#9637",
    date: "October 2024",
    description:
      "A fast and affordable language model that excels in tasks such as coding, reasoning, and content creation.",
    tags: ["Frontier Model", "Anthropic", "Claude"],
  },
  {
    title: "Gemini 2.0 Flash",
    link: "97b6b71e0054#3a09",
    date: "December 2024",
    description:
      "A faster, more powerful large language model than its predecessor, boasting enhanced multimodal input/output (including native image generation and steerable multilingual audio), native tool use (like Google Search and code execution), and enabling a new class of agentic experiences through improved reasoning and complex instruction following.",
    tags: ["Frontier Model", "Gemini"],
  },
  {
    title: "o3-mini",
    link: "163fd9c7308e#9875",
    date: "January 2025",
    description:
      "A cost-efficient reasoning model, excelling in STEM fields, while maintaining low latency. It supports features like function calling, structured outputs, and developer messages, and offers adjustable reasoning effort levels (low, medium, high) for optimized performance.",
    tags: ["Frontier Model", "OpenAI", "GPT", "Reasoning"],
  },
  {
    title: "Gemini 2.0 Flash Lite",
    link: "97b6b71e0054#d87a",
    date: "February 2025",
    description:
      "A production-ready language model offering enhanced performance over its predecessor (1.5 Flash) in reasoning, multimodal tasks, math, and factual accuracy, along with cost-effective pricing, making it ideal for applications requiring long context windows.",
    tags: ["Frontier Model", "Gemini"],
  },
  {
    title: "Grok 3 Beta",
    link: "0d9f1aef69be#2fe9",
    date: "February 2025",
    description:
      "Trained on a massive scale with 10x the compute of previous models, exhibiting advanced reasoning, and a 1 million token context window, with a Think mode and is available in a cost-efficient mini version for STEM tasks. Grok agents like DeepSearch further enhance its capabilities by combining reasoning with tool use like internet access and code interpreters.",
    tags: ["Frontier Model", "Grok"],
  },
  {
    title: "Claude 3.7 Sonnet",
    link: "89dd45e35d92#a30f",
    date: "February 2025",
    description:
      "A hybrid reasoning model offering both fast responses and detailed, user-visible step-by-step thinking controllable by a token budget, featuring integration with GitHub and a new coding tool (Claude Code).",
    tags: ["Frontier Model", "Anthropic", "Claude"],
  },
  {
    title: "GPT 4.5",
    link: "dc1d4b097ad1",
    date: "February 2025",
    description:
      "A research preview model representing an advancement in scaling unsupervised learning through increased compute, data, architecture, and optimization innovations, resulting in broader knowledge, deeper understanding, reduced hallucinations, and increased reliability. It excels in tasks requiring natural conversation, creativity, and understanding human intent due to new scalable training techniques derived from smaller models.",
    tags: ["Frontier Model", "OpenAI", "GPT"],
  },
  {
    title: "Gemini 2.5 Pro",
    link: "97b6b71e0054#5485",
    date: "March 2025",
    description:
      "Google's most advanced experimental AI model, excelling in complex tasks due to its enhanced reasoning and coding capabilities, achieving state-of-the-art performance on various benchmarks, and building upon Gemini's strengths of native multimodality and a long context window.",
    tags: ["Frontier Model", "Gemini", "Reasoning"],
  },
  {
    title: "GPT 4.1",
    link: "dc1d4b097ad1#a605",
    date: "April 2025",
    description:
      "Advanced language models offering superior coding, instruction following, and long-context comprehension (up to 1 million tokens) compared to predecessors, with lower cost and latency, excelling in real-world tasks and boast enhanced vision capabilities",
    tags: ["Frontier Model", "OpenAI", "GPT"],
  },
  {
    title: "Gemini 2.5 Flash",
    link: "97b6b71e0054#b40c",
    date: "April 2025",
    description:
      "A hybrid reasoning model that allows developers to control a 'thinking' process for improved accuracy on complex tasks, offering a customizable thinking budget to balance quality, cost, and speed.",
    tags: ["Frontier Model", "Gemini", "Reasoning"],
  },
  {
    title: "o3",
    link: "163fd9c7308e#4cfa",
    date: "April 2025",
    description:
      "Reasoning model, excelling in complex tasks requiring multi-faceted analysis, especially those involving visual data like images and charts. It leverages tools like web search, code execution, and image generation to provide comprehensive answers and demonstrates significantly improved performance across various benchmarks and real-world tasks.",
    tags: ["Frontier Model", "OpenAI", "GPT", "Reasoning", "Multimodal"],
  },
  {
    title: "o4-mini",
    link: "163fd9c7308e#4cfa",
    date: "April 2025",
    description:
      "A smaller, faster, and more cost-effective reasoning model optimized for speed and efficiency. While powerful in math, coding, and visual tasks, it maintains strong performance in other areas and supports higher usage limits due to its efficiency, making it suitable for high-volume applications.",
    tags: ["Frontier Model", "OpenAI", "GPT", "Reasoning", "Multimodal"],
  },
  {
    title: "Mistral Medium 3",
    link: "b9632dedf580#0ea6",
    date: "May 2025",
    description:
      "A cost-effective, high-performing language model designed for enterprise use, offering SOTA performance comparable to Claude Sonnet 3.7 at 8x lower cost, excelling in professional tasks like coding and STEM.",
    tags: ["Frontier Model", "Mistral"],
  },
  {
    title: "Claude 4",
    link: "89dd45e35d92#90f2",
    date: "May 2025",
    description:
      "Hybrid models offering two modes: near-instant responses and extended thinking for deeper reasoning, setting new standards for coding, advanced reasoning, and AI agents.",
    tags: ["Frontier Model", "Anthropic", "Claude"],
  },
  {
    title: "Gemini 2.5",
    link: "3b8877cf4da9",
    date: "June 2025",
    description:
      "Google's next generation of AI models designed for agentic systems, featuring native multimodality, long context input, and tool use support, with each model offering different strengths in reasoning, cost, and latency. Gemini 2.5 Pro excels in reasoning and coding, Gemini 2.5 Flash balances quality, cost, and latency, Gemini 2.0 Flash is fast and cost-efficient, and Gemini 2.0 Flash-Lite is the fastest and most cost-efficient.",
    tags: ["Frontier Model", "Gemini", "Reasoning"],
  },
  {
    title: "Grok 4",
    link: "0d9f1aef69be#8e14",
    date: "July 2025",
    description:
      "Trained using reinforcement learning on a 200k GPU cluster, featuring native tool use, real-time search integration, and an upgraded Voice Mode with visual analysis, achieving state-of-the-art results on benchmarks like ARC-AGI V2 and Vending-Bench.",
    tags: ["Frontier Model", "Grok"],
  },
  {
    title: "Claude Opus 4.1",
    link: "89dd45e35d92#7a6f",
    date: "August 2025",
    description:  
      "An upgrade to Claude Opus 4 on agentic tasks, real-world coding, and reasoning.",
    tags: ["Frontier Model", "Anthropic", "Claude"],
  },
  {
    title: "GPT 5",
    link: "0342672382e7",
    date: "August 2025",
    description:
      "A unified model with a smart, efficient component for quick answers and a deeper reasoning component (GPT-5 thinking) for complex problems, managed by a real-time router.",
    tags: ["Frontier Model", "OpenAI", "GPT"],
  },
  {
    title: "Grok 4 Fast",
    link: "0d9f1aef69be#bb66",
    date: "September 2025",
    description:
      "A cost-efficient reasoning model, built upon the learnings from Grok 4, delivering frontier-level performance in both Enterprise and Consumer domains with exceptional token efficiency, featuring a 2M token context window, native tool use, and SOTA search capabilities.",
    tags: ["Frontier Model", "Grok"],
  },
  {
    title: "Claude 4.5 Sonnet",
    link: "89dd45e35d92#bf64",
    date: "September 2025",
    description:
      "Excels in coding, computer use, reasoning, and math, making it the best coding model and strongest for building complex agents, while also being the most aligned frontier model with improved safety features.",
    tags: ["Frontier Model", "Anthropic", "Claude"],
  },
  {
    title: "Claude Haiku 4.5",
    link: "89dd45e35d92#6ccd",
    date: "October 2025",
    description:
      "Anthropic’s latest small model, offering near-frontier coding performance at one-third the cost and more than twice the speed of Claude Sonnet 4.",
    tags: ["Frontier Model", "Anthropic", "Claude"],
  },
  {
    title: "GPT 5.1",
    link: "0342672382e7#6047",
    date: "November 2025",
    description:
      "An upgrade to the GPT-5 series, featuring GPT-5.1 Instant (warmer, more intelligent, better instruction following) and GPT-5.1 Thinking (more efficient, easier to understand), with improved customization options for ChatGPT's tone and style.",
    tags: ["Frontier Model", "OpenAI", "GPT"],
  },
  {
    title: "Grok 4.1",
    link: "0d9f1aef69be#b913",
    date: "November 2025",
    description:
      "A significant upgrade focusing on enhanced real-world usability, particularly in creative, emotional, and collaborative interactions, while retaining the intelligence and reliability of its predecessor.",
    tags: ["Frontier Model", "Grok"],
  },
  {
    title: "Gemini 3",
    link: "3b8877cf4da9#1277",
    date: "November 2025",
    description:
      "Google's most intelligent AI model, enhancing reasoning and multimodal capabilities, with a Deep Think mode. It outperforms previous models in reasoning, multimodality, and coding benchmarks, enabling users to learn, build, and plan anything with improved reasoning and tool use.",
    tags: ["Frontier Model", "Gemini", "Reasoning"],
  },
  {
    title: "Claude Opus 4.5",
    link: "89dd45e35d92#25d5",
    date: "November 2025",
    description:
      "An enhanced version of Claude Opus 4, excelling in coding, agentic tasks, and general computer use, while also demonstrating significant improvements in research, spreadsheet handling, and overall reasoning capabilities.",
    tags: ["Frontier Model", "Anthropic", "Claude"],
  },
  {
    title: "GPT 5.2",
    link: "0342672382e7#2b77",
    date: "December 2025",
    description:
      "An advancement in the GPT-5 series, designed to enhance professional knowledge work by excelling in creating spreadsheets, building presentations, writing code, perceiving images, understanding long contexts, using tools, and handling complex, multi-step projects.",
    tags: ["Frontier Model", "OpenAI", "GPT"],
  },
  {
    title: "Claude Opus 4.6",
    link: "89dd45e35d92#bd90",
    date: "February 2026",
    description:
      "Improves on Opus 4.5 in coding, long-context reasoning, and autonomous agentic work. It plans more carefully, sustains long-running tasks, operates reliably in large codebases, and excels at code review and debugging, while also handling complex everyday work like financial analysis, research, and creating or editing documents, spreadsheets, and presentations.",
      tags: ["Frontier Model", "Anthropic", "Claude"],      
  }
];
