const miscellaneous_studies = [
  {
    title: "TLDR",
    link: "https://ritvik19.medium.com/papers-explained-293-tldr-a31d787cd365",
    date: "April 2020",
    description:
      "Extreme summarization for scientific papers, creating concise, single-sentence summaries of key contributions.",
    tags: [],
  },
  {
    title: "KL Divergence VS MSE for Knowledge Distillation",
    link: "https://ritvik19.medium.com/papers-explained-381-kl-divergence-vs-mse-for-knowledge-distillation-97988e80de3e",
    date: "May 2021",
    description:
      "Theoretically and empirically demonstrates that MSE loss outperforms the traditional KL divergence loss in knowledge distillation by focusing on logit matching, and further shows that sequential distillation enhances performance and that using KL divergence loss with a small tau can mitigate label noise.",
    tags: [],
  },
  {
    title: "What do Vision Transformers Learn",
    link: "https://ritvik19.medium.com/papers-explained-344-what-do-vision-transformers-learn-ef4a80da46d8",
    date: "January 2022",
    description:
      "Provides insights into the working mechanisms of vision transformers and their similarities and differences from convolutional neural networks.",
    tags: ["Vision Transformers"],
  },
  {
    title: "Are Emergent Abilities of Large Language Models a Mirage?",
    link: "https://ritvik19.medium.com/papers-explained-are-emergent-abilities-of-large-language-models-a-mirage-4160cf0e44cb",
    date: "April 2023",
    description:
      "Presents an alternative explanation for emergent abilities, i.e. emergent abilities are created by the researcher’s choice of metrics, not fundamental changes in model family behaviour on specific tasks with scale.",
    tags: [],
  },
  {
    title: "Scaling Data-Constrained Language Models",
    link: "https://ritvik19.medium.com/papers-explained-85-scaling-data-constrained-language-models-2a4c18bcc7d3",
    date: "May 2023",
    description:
      "This study investigates scaling language models in data-constrained regimes.",
    tags: [],
  },
  {
    title: "Multiagent Debate",
    link: "https://ritvik19.medium.com/papers-explained-291-multiagent-debate-1a1693d5fa5e",
    date: "May 2023",
    description:
      "Involves multiple language model instances independently generating answers to a query, then iteratively critiquing and updating their responses based on others' answers over multiple rounds, ultimately converging on a single, refined answer. This process mimics multi-threaded reasoning and multi-source fact-checking, leveraging the collective intelligence of the models to improve factual accuracy and reasoning capabilities.",
    tags: ["Agentic Systems"],
  },
  {
    title: "GGUF",
    link: "",
    date: "August 2023",
    description:
      "A binary file format designed for storing and loading large language models (LLMs), specifically for inference, primarily within the GGML ecosystem and its derivatives like llama.cpp.",
    tags: ["File Formats"],
  },
  {
    title: "RAGAS",
    link: "https://ritvik19.medium.com/papers-explained-227-ragas-4594fc4d96b9",
    date: "September 2023",
    description:
      "A framework for reference-free evaluation of RAG systems, assessing the retrieval system's ability to find relevant context, the LLM's faithfulness in using that context, and the overall quality of the generated response.",
    tags: ["LLM Evaluation"],
  },
  {
    title: "ConvNets Match Vision Transformers at Scale",
    link: "https://ritvik19.medium.com/papers-explained-345-convnets-match-vision-transformers-at-scale-496690f604c7",
    date: "October 2023",
    description:
      "Challenges the belief that Vision Transformers outperform ConvNets on large datasets by demonstrating that ConvNets, specifically NFNets, achieve comparable performance when pre-trained on a large dataset and fine-tuned on ImageNet.",
    tags: ["Vision Transformers", "Convolutional Neural Networks"],
  },
  {
    title: "DSPy",
    link: "https://ritvik19.medium.com/papers-explained-135-dspy-fe8af7e35091",
    date: "October 2023",
    description:
      "A programming model that abstracts LM pipelines as text transformation graphs, i.e. imperative computation graphs where LMs are invoked through declarative modules, optimizing their use through a structured framework of signatures, modules, and teleprompters to automate and enhance text transformation tasks.",
    tags: ["Prompt Optimization"],
  },
  {
    title: "LLMLingua",
    link: "https://ritvik19.medium.com/papers-explained-136-llmlingua-f9b2f53f5f9b",
    date: "October 2023",
    description:
      "A novel coarse-to-fine prompt compression method, incorporating a budget controller, an iterative token-level compression algorithm, and distribution alignment, achieving up to 20x compression with minimal performance loss.",
    tags: ["Prompt Compression"],
  },
  {
    title: "LongLLMLingua",
    link: "https://ritvik19.medium.com/papers-explained-137-longllmlingua-45961fa703dd",
    date: "October 2023",
    description:
      "A novel approach for prompt compression to enhance performance in long context scenarios using question-aware compression and document reordering.",
    tags: ["Prompt Compression"],
  },
  {
    title: "An In-depth Look at Gemini's Language Abilities",
    link: "https://ritvik19.medium.com/papers-explained-81-an-in-depth-look-at-geminis-language-abilities-540ca9046d8e",
    date: "December 2023",
    description:
      "A third-party, objective comparison of the abilities of the OpenAI GPT and Google Gemini models with reproducible code and fully transparent results.",
    tags: [],
  },
  {
    title: "STORM",
    link: "https://ritvik19.medium.com/papers-explained-242-storm-2c55270d3150",
    date: "February 2024",
    description:
      "A writing system that addresses the prewriting stage of long-form article generation by researching diverse perspectives, simulating multi-perspective question-asking, and curating information to create an outline, ultimately leading to more organized and comprehensive articles compared to baseline methods.",
    tags: ["Agentic Systems"],
  },
  {
    title: "NuNER",
    link: "https://ritvik19.medium.com/papers-explained-186-nuner-03e092dfb6ff",
    date: "February 2024",
    description:
      "A foundation model for Named Entity Recognition (NER) created by further pre-training RoBERTa, using contrastive training on a large dataset annotated by GPT-3.5, derived from a subset of C4.",
    tags: ["Language Models", "Named Entity Recognition", "Synthetic Data"],
  },
  {
    title: "LLMLingua2",
    link: "https://ritvik19.medium.com/papers-explained-138-llmlingua-2-510c752368a8",
    date: "March 2024",
    description:
      "A novel approach to task-agnostic prompt compression, aiming to enhance generalizability, using  data distillation and leveraging a Transformer encoder for token classification.",
    tags: ["Prompt Compression"],
  },
  {
    title: "PromptWizard",
    link: "https://ritvik19.medium.com/papers-explained-262-promptwizard-228568783085",
    date: "May 2024",
    description:
      "A  framework that leverages LLMs to iteratively synthesize and refine prompts tailored to specific tasks by optimizing both prompt instructions and in-context examples, maximizing model performance.",
    tags: ["Prompt Optimization"],
  },
  {
    title: "LearnLM Tutor",
    link: "https://ritvik19.medium.com/papers-explained-279-learnlm-tutor-77503db05415",
    date: "May 2024",
    description:
      "A text-based gen AI tutor based on Gemini 1.0, further fine- tuned for 1:1 conversational tutoring with improved education-related capabilities over a prompt tuned Gemini 1.0.",
    tags: ["Language Models", "Gemini"],
  },
  {
    title: "Monte Carlo Tree Self-refine",
    link: "https://ritvik19.medium.com/papers-explained-167-monte-carlo-tree-self-refine-79bffb070c1a",
    date: "June 2024",
    description:
      "Integrates LLMs with Monte Carlo Tree Search to enhance performance in complex mathematical reasoning tasks, leveraging systematic exploration and heuristic self-refine mechanisms to improve decision-making frameworks.",
    tags: ["Scientific Data"],
  },
  {
    title: "NuExtract",
    link: "https://ritvik19.medium.com/papers-explained-287-nuextract-f722082999b5",
    date: "June 2024",
    description:
      "Small language models fine-tuned on a synthetic dataset of C4 text passages and LLM-generated structured extraction templates and outputs, achieving comparable or superior performance to much larger LLMs like GPT-4 on complex extraction tasks while being significantly smaller and offering zero-shot, hybrid few-shot (using output examples), and fine-tuning capabilities.",
    tags: ["Language Models", "Named Entity Recognition", "Synthetic Data"],
  },
  {
    title: "Proofread",
    link: "https://ritvik19.medium.com/papers-explained-189-proofread-4e1fe4eccf01",
    date: "June 2024",
    description:
      "A Gboard feature powered by a server-side LLM, enabling seamless sentence-level and paragraph-level corrections with a single tap.",
    tags: ["Language Models"],
  },
  {
    title: "Gemma APS",
    link: "https://ritvik19.medium.com/papers-explained-244-gemma-aps-8fac1838b9ef",
    date: "June 2024",
    description:
      "Proposes a scalable, yet accurate, proposition segmentation model by modeling Proposition segmentation as a supervised task by training LLMs on existing annotated datasets.",
    tags: ["Language Models", "Gemma"],
  },
  {
    title: "Shield Gemma",
    link: "https://ritvik19.medium.com/papers-explained-243-shieldgemma-d779fd66ee3e",
    date: "July 2024",
    description:
      "A comprehensive suite of LLM-based safety content moderation models ranging from 2B to 27B parameters built upon Gemma2 that provide predictions of safety risks across key harm types (sexually explicit, dangerous content, harassment, hate speech) in both user input and LLM-generated output.",
    tags: ["Language Models", "Gemma", "LLM Safety"],
  },
  {
    title: "OmniParser",
    link: "https://ritvik19.medium.com/papers-explained-259-omniparser-2e895f6f2c15",
    date: "August 2024",
    description:
      "A method for parsing user interface screenshots into structured elements, enhancing the ability of GPT-4V to generate actions grounded in the interface by accurately identifying interactable icons and understanding element semantics.",
    tags: ["Language Models", "Multimodal Models", "BLIP"],
  },
  {
    title: "Reader-LM",
    link: "https://ritvik19.medium.com/papers-explained-221-reader-lm-7382b9eb6ed9",
    date: "September 2024",
    description:
      "Small multilingual models specifically trained to generate clean markdown directly from noisy raw HTML, with a context length of up to 256K tokens.",
    tags: ["Language Models"],
  },
  {
    title: "DataGemma",
    link: "https://ritvik19.medium.com/papers-explained-212-datagemma-cf0d2f40d867",
    date: "September 2024",
    description:
      "A set of models that aims to reduce hallucinations in LLMs by grounding them in the factual data of Google's Data Commons, allowing users to ask questions in natural language and receive responses based on verified information from trusted sources.",
    tags: ["Language Models", "Transformer Decoder", "Small LLMs", "Gemma"],
  },
  {
    title: "GSM-Symbolic",
    link: "https://ritvik19.medium.com/papers-explained-260-gsm-symbolic-759d379052c7",
    date: "October 2024",
    description:
      "Investigates the true mathematical reasoning capabilities of LLMs by introducing GSM-Symbolic, a new benchmark based on symbolic templates, revealing that LLMs exhibit inconsistent performance, struggle with complex questions, and appear to rely on pattern recognition rather than genuine logical reasoning.",
    tags: ["Scientific Data"],
  },
  {
    title: "NuExtract 1.5",
    link: "https://ritvik19.medium.com/papers-explained-287-nuextract-f722082999b5#83ce",
    date: "October 2024",
    description:
      "An upgrade of NuExtract trained on a multilingual synthetic dataset derived from C4 documents and LLM-generated structured extraction templates and outputs, incorporating longer documents and a 'continuation' training methodology for handling documents exceeding typical context windows, resulting in a model capable of multilingual, long-document structured extraction with performance comparable or superior to much larger LLMs.",
    tags: ["Language Models", "Named Entity Recognition", "Synthetic Data"],
  },
  {
    title: "LearnLM",
    link: "https://ritvik19.medium.com/papers-explained-280-learnlm-df8cdc2fed45",
    date: "December 2024",
    description:
      "Combines Supervised Fine-Tuning (SFT) with Reinforcement Learning from Human Feedback (RLHF) to enhance pedagogical instruction in conversational AI, co-trained with Gemini's SFT, Reward Model (RM), and RL stages.",
    tags: ["Language Models", "Gemini"],
  },
  {
    title: "Open Scholar",
    link: "https://ritvik19.medium.com/papers-explained-185-openscholar-76b1b2df7b99",
    date: "December 2024",
    description:
      "A retrieval-augmented large language model specializing in scientific literature synthesis. It uses a large datastore of 45 million open-access papers, specialized retrievers, an open-source 8B model trained on synthetically generated data and iterative self-feedback generation to answer scientific queries with accurate citations.",
    tags: ["Scientific Data", "Agentic Systems"],
  },
  {
    title: "Shiksha",
    link: "https://ritvik19.medium.com/papers-explained-300-shiksha-ad09f6db7f38",
    date: "December 2024",
    description:
      "Addresses the challenge of training machine translation models to effectively handle scientific and technical language, particularly for low-resource Indian languages, by creating a multilingual parallel corpus containing over 2.8 million translation pairs across eight Indian languages by mining human-translated transcriptions of NPTEL video lectures.",
    tags: ["Language Models", "Multilingual Models"],
  },
  {
    title: "Multi-LLM Text Summarization",
    link: "https://ritvik19.medium.com/papers-explained-294-multi-llm-text-summarization-6141d1276772",
    date: "December 2024",
    description:
      "Introduces a novel multi-LLM summarization framework with centralized and decentralized approaches. Multiple LLMs generate diverse summaries, and either a single LLM (centralized) or all LLMs (decentralized) evaluate them iteratively.",
    tags: ["Language Models", "Agentic Systems"],
  },
  {
    title: "Reader LM v2",
    link: "https://ritvik19.medium.com/papers-explained-348-readerlm-v2-b5bed3f57fbe",
    date: "December 2024",
    description:
      "A 1.5B parameter language model specializing in converting raw HTML to markdown or JSON, handling up to 512K tokens and supporting 29 languages. Trained with a new paradigm and higher-quality data, it treats HTML-to-markdown as translation, and addresses the degeneration issues of its predecessor through contrastive loss.",
    tags: ["Language Models"],
  },
  {
    title: "OmniParser V2",
    link: "https://ritvik19.medium.com/papers-explained-259-omniparser-2e895f6f2c15#6ac5",
    date: "February 2025",
    description:
      "Improves upon OmniParser with higher accuracy on smaller elements, faster inference, achieved through training with larger datasets and a smaller icon caption model.",
    tags: ["Language Models", "Multimodal Models", "BLIP"],
  },
  {
    title: "Competitive Programming with Large Reasoning Models",
    link: "https://ritvik19.medium.com/papers-explained-317-competitive-programming-with-large-reasoning-models-51836dbf584e",
    date: "February 2025",
    description:
      "Explores how reinforcement learning significantly improves large language model performance on competitive programming and software engineering tasks, comparing OpenAI models o1, o1-ioi, and o3.",
    tags: ["Language Models", "OpenAI", "GPT"],
  },
  {
    title: "olmOCR",
    link: "https://ritvik19.medium.com/papers-explained-326-olmocr-bc9158752901",
    date: "February 2025",
    description:
      "An open-source Python toolkit that converts PDFs into linearized plain text while preserving structured content (sections, tables, lists, equations, etc.). It uses a document-anchoring approach, leveraging a fine-tuned 7B VLM.",
    tags: ["Language Models", "Multimodal Models"],
  },
  {
    title: "Rethinking Compute-Optimal Test-Time Scaling",
    link: "https://ritvik19.medium.com/papers-explained-336-rethinking-compute-optimal-test-time-scaling-732ee1134883",
    date: "February 2025",
    description:
      "Investigates compute-optimal Test-Time Scaling (TTS) for Large Language Models (LLMs), focusing on the influence of policy models, Process Reward Models (PRMs), and problem difficulty. Through experiments on MATH-500 and AIME24.",
    tags: ["Language Models"],
  },
  {
    title: "CHallenging AI with Synthetic Evaluations (CHASE)",
    link: "https://ritvik19.medium.com/papers-explained-340-chase-84857503f39c",
    date: "February 2025",
    description:
      "A framework for generating challenging LLM evaluation benchmarks synthetically, using a bottom-up approach, building complex problems from simpler components and hiding solution elements within the context, while decomposing the generation process into verifiable sub-tasks to ensure correctness.",
    tags: ["Synthetic Data", "LLM Evaluation"],
  },
  {
    title: "Large-Scale Data Selection for Instruction Tuning",
    link: "https://ritvik19.medium.com/papers-explained-338-large-scale-data-selection-for-instruction-tuning-72ef9f8221aa",
    date: "March 2025",
    description:
      "Investigates the effectiveness and scalability of automated data selection methods for instruction-tuning LLMs, finding that many existing methods underperform random selection at larger scales. A variant of representation-based data selection (RDS+), using weighted mean pooling of pre-trained LM hidden states, consistently outperforms other methods, including in multi-task settings.",
    tags: [],
  },
  {
    title: "Transformers without Normalization",
    link: "https://ritvik19.medium.com/papers-explained-335-transformers-without-normalization-a1cec27c2c4f",
    date: "March 2025",
    description:
      "Introduces Dynamic Tanh (DyT), a simple element-wise operation replacing normalization layers in Transformers, achieving comparable or better performance across various tasks in vision, language, and speech by emulating Layer Normalization's tanh-like activation mapping and scaling without statistical computation, challenging the perceived indispensability of normalization and potentially improving efficiency.",
    tags: ["Transformers"],
  },
  {
    title: "SmolDocling",
    link: "https://ritvik19.medium.com/papers-explained-333-smoldocling-a788ac739b92",
    date: "March 2025",
    description:
      "A compact 256M parameter vision-language model designed for end-to-end document conversion into a novel universal markup format called DocTags, which captures content, structure, and spatial location of all page elements. It leverages a curriculum learning approach trained on augmented existing and novel datasets for comprehensive document understanding, achieving performance comparable to much larger models while minimizing computational requirements.",
    tags: ["Multimodal Models"],
  },
  {
    title: "MDocAgent",
    link: "https://ritvik19.medium.com/papers-explained-422-mdocagent-db1c83ddc4ee",
    date: "March 2025",
    description:
      "A multi-modal, multi-agent framework for document understanding that leverages RAG with 5 specialized (general, critical, text, image, and summarizing) agents to improve complex question answering on documents with rich textual and visual information.",
    tags: ["Agentic Systems"],
  },
  {
    title: "Long-To-Short LLM Reasoning With Model Merging",
    link: "https://ritvik19.medium.com/papers-explained-357-long-to-short-llm-reasoning-with-model-merging-03a212b0ccad",
    date: "March 2025",
    description:
      "Explores model merging as an efficient method for Long-to-Short reasoning in LLMs, aiming to reduce verbose reasoning steps without sacrificing accuracy. The study found that task-vector based merging, effectively reduced response length by ~50% while maintaining or slightly improving accuracy on 7B parameter models; activation-based merging showed even greater promise but is sensitive to calibration data; and merging efficacy was correlated with model scale, with smaller models struggling to learn complex reasoning and larger models posing challenges for significant length reduction.",
    tags: ["Model Merging", "Language Models"],
  },
  {
    title: "QALIGN",
    link: "https://ritvik19.medium.com/papers-explained-372-qalign-977600e913fb",
    date: "April 2025",
    description:
      "A test-time alignment method that uses Markov Chain Monte Carlo (MCMC) sampling to generate a sequence of increasingly aligned text samples, guided by a reward model. It then selects the final output using Minimum Bayes Risk (MBR) over the generated samples.",
    tags: ["Language Models"],
  },
  {
    title:
      "Does RL Incentivize Reasoning Capacity in LLMs Beyond the Base Model",
    link: "https://ritvik19.medium.com/papers-explained-354-does-rl-incentivize-reasoning-capacity-in-llms-beyond-the-base-model-77ae394a5054",
    date: "April 2025",
    description:
      "Challenges the prevailing belief that RLVR significantly improves LLMs' reasoning abilities, finding instead that while RLVR increases sampling efficiency for correct answers at low k values in pass@k, it actually restricts the overall reasoning capacity boundary at high k values compared to base models due to reduced exploration of potentially successful reasoning paths already present in the base model.",
    tags: ["Language Models"],
  },
  {
    title: "The Ladder of Reasoning",
    link: "https://ritvik19.medium.com/papers-explained-419-the-ladder-of-reasoning-e58f9727abec",
    date: "April 2025",
    description:
      "Analyzes the impact of supervised fine-tuning (SFT) on language models' mathematical reasoning abilities using the AIME24 dataset, revealing a ladder-like structure of problem difficulty (Easy, Medium, Hard, Exh). The study finds that progressing through these tiers requires different capabilities, with R1 reasoning sufficient for Medium-level, stability in deeper exploration needed for Hard-level, and unconventional problem-solving skills required for Exh-level, while also highlighting the importance of scaling SFT datasets and the limitations of SFT alone in achieving higher-level reasoning.",
    tags: ["LLM for Math"],
  },
  {
    title: "Crosslingual Reasoning through Test-Time Scaling",
    link: "https://ritvik19.medium.com/papers-explained-403-crosslingual-reasoning-through-test-time-scaling-ee069fcca028",
    date: "May 2025",
    description:
      "Investigates the crosslingual generalization capabilities of English-centric Reasoning Language Models (RLMs) through test-time scaling. It demonstrates that scaling up inference compute for these models improves multilingual mathematical reasoning, reveals a quote-and-think language-mixing pattern, discovers a strategy to control reasoning language (with better performance in high-resource languages), and observes poor out-of-domain reasoning generalization.",
    tags: ["Language Models"],
  },
  {
    title: "Examining Citation Relationships using LLMs",
    link: "",
    date: "May 2025",
    description:
      "Addresses interpretability of LLMs in document-based tasks through attribution, which involves tracing generated outputs back to their source documents, through two techniques: a zero-shot approach framing attribution as textual entailment (using flan-ul2) and an attention-based binary classification technique (using flan-t5-small)",
    tags: ["Language Models"],
  },
  {
    title: "short-m@k",
    link: "https://ritvik19.medium.com/papers-explained-389-short-m-k-e0b516dbb7c0",
    date: "May 2025",
    description:
      "Challenges the assumption that longer reasoning chains in LLMs lead to better reasoning, demonstrating that shorter chains are often more accurate and propose a new inference method, short-m@k, which executes parallel generations and halts after m chains, using majority voting for the final answer, achieving similar or superior performance with reduced computational costs and inference time.",
    tags: ["Language Models", "LLM for Math"],
  },
  {
    title: "Paper2Poster",
    link: "https://ritvik19.medium.com/papers-explained-427-paper2poster-504437bba4cc",
    date: "May 2025",
    description:
      "A multi-agent pipeline that distills papers into structured assets, plans layouts, and refines panels with visual feedback.",
    tags: ["Agentic Systems"],
  },
  {
    title: "Evaluation is All You Need",
    link: "https://ritvik19.medium.com/papers-explained-398-evaluation-is-all-you-need-648b9ede2d5e",
    date: "June 2025",
    description:
      "The study reveals that the benchmark evaluation results of reasoning models are subject to significant fluctuations caused by various factors. Subtle differences in evaluation conditions can lead to substantial variations in results, making their claimed performance improvements difficult to reproduce reliably.",
    tags: ["LLM Evaluation"],
  },
  {
    title: "Claude Research",
    link: "https://ritvik19.medium.com/papers-explained-412-claude-research-334b1f6a353b",
    date: "June 2025",
    description:
      "A multi-agent system developed by Anthropic that allows Claude to search across the web, Google Workspace, and integrations to accomplish complex tasks. It uses a multi-agent architecture with an orchestrator-worker pattern, where a lead agent coordinates the process and delegates to specialized subagents that operate in parallel.",
    tags: ["Agentic Systems", "Anthropic", "Claude"],
  },
  {
    title: "Kimi Researcher",
    link: "https://ritvik19.medium.com/papers-explained-417-kimi-researcher-baa1c9f4ae68",
    date: "June 2025",
    description:
      "An autonomous agent that excels at multi-turn search and reasoning. It performs an average of 23 reasoning steps and explores over 200 URLs per task. Built on an internal version of the Kimi k-series model and trained entirely through end-to-end agentic reinforcement learning.",
    tags: ["Agentic Systems"],
  },
  {
    title: "Universal Tokenizer",
    link: "https://ritvik19.medium.com/papers-explained-405-universal-tokenizer-1dfd6e76cbd1",
    date: "June 2025",
    description:
      "Explores how to improve the ability of multilingual LLMs to adapt to new languages after pretraining, focusing on tokenizer design and proposes using a universal tokenizer trained on more languages than the pretraining languages, demonstrating that this approach significantly enhances language adaptation capabilities, even for languages unseen during pretraining, with minimal performance compromise on the original pretraining languages.",
    tags: ["Language Models", "Multilingual Models"],
  },
  {
    title: "Vision-Guided Chunking",
    link: "",
    date: "June 2025",
    description:
      "Introduces a vision-guided chunking approach that uses Large Multimodal Models to process PDF documents in batches, preserving semantic coherence and structural integrity across page boundaries. The method addresses the limitations of traditional text-based chunking in RAG systems by effectively handling complex layouts, tables, figures, and multi-page content.",
    tags: ["Multimodal Models", "Document Understanding"],
  },
  {
    title: "Answer Matching",
    link: "https://ritvik19.medium.com/papers-explained-406-answer-matching-0940b4c50570",
    date: "July 2025",
    description:
      "Argues that multiple-choice benchmarks, traditionally used for evaluating language models, suffer from a critical flaw: they allow models to exploit discriminative shortcuts and answer questions without truly understanding or generating the correct response. Proposes 'Answer Matching' as a superior alternative for evaluating the generative capabilities of language models.",
    tags: ["LLM Evaluation"],
  },
  {
    title: "Reasoning or Memorization",
    link: "https://ritvik19.medium.com/papers-explained-423-reasoning-or-memorization-29b5c073ed55",
    date: "July 2025",
    description:
      "Investigates the reasoning capabilities of LLMs enhanced by RL, focusing on discrepancies in reported improvements across different model families like Qwen2.5 and Llama. It reveals that Qwen2.5's superior performance on mathematical benchmarks like MATH-500 may be due to data contamination from its pre-training on web-scale corpora, and demonstrates that only accurate reward signals consistently improve mathematical reasoning performance in RL.",
    tags: ["Language Models", "LLM for Math"],
  },
  {
    title: "Gemini 2.5 Pro Capable of Winning Gold at IMO 2025",
    link: "https://ritvik19.medium.com/papers-explained-415-gemini-2-5-pro-capable-of-winning-gold-at-imo-2025-0e76cc8ae940",
    date: "July 2025",
    description:
      "Presents a novel methodology based on pipeline design and prompt engineering with the Gemini 2.5 Pro model, solving  5 out of the 6 problems of IMO 2025.",
    tags: ["Language Models", "Gemini", "LLM for Math", "Agentic Systems"],
  },
  {
    title: "Multi-Domain Reasoning via Reinforcement Learning",
    link: "",
    date: "July 2025",
    description:
      "Investigates multi-domain reasoning in LLMs using RLVR, focusing on mathematical reasoning, code generation, and logical puzzle solving. The research evaluates in-domain improvements, cross-domain generalization, interactions during combined training, the influence of SFT on RL, and the impacts of curriculum learning, reward design, and language-specific factors, providing insights for optimizing RL methodologies to foster comprehensive, multi-domain reasoning capabilities in LLMs.",
    tags: ["Language Models", "Reasoning"],
  },
  {
    title: "OpenAI Harmony Response Format",
    link: "https://ritvik19.medium.com/papers-explained-428-gpt-oss-e1aed3d15afe",
    date: "August 2025",
    description:
      "A chat format that structures conversations with special tokens and roles (system, developer, user, assistant, tool) to define message types, reasoning effort, instructions, and available tools, while utilizing channels (final, analysis, commentary) to separate user-facing responses from internal reasoning and tool calls, ensuring proper model behavior and safety.",
    tags: ["OpenAI"],
  }
];
