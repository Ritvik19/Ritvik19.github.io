document.getElementById("title").innerHTML = "Best Practices and Lessons Learned on Synthetic Data";

((f, d, h, u) => {
  const g = f();
  window.mm = g.Markmap.create("svg#mindmap", (d || g.deriveOptions)(u), h);
})(
  () => window.markmap,
  null,
  {
    content:
      '<a href="https://arxiv.org/abs/2404.07503">Best Practices and Lessons Learned on Synthetic Data</a>',
    children: [
      {
        content: "Synthetic Data in Training",
        children: [
          {
            content: "Reasoning",
            children: [
              {
                content: "Math",
                children: [
                  {
                    content:
                      '<strong>Math-targeted pre-training</strong>: Models like <a href="">Minerva</a>, <a href="">Llemma</a>, and <a href="">DeepSeekMath</a> are pre-trained on datasets specifically curated for mathematical tasks.',
                    children: [],
                    payload: { lines: "8,9" },
                  },
                  {
                    content:
                      '<strong>Synthetic question generation</strong>: <a href="">WizardMath</a> increases complexity in questions and answers using GPT-3.5 to imitate target benchmarks.',
                    children: [],
                    payload: { lines: "9,10" },
                  },
                  {
                    content:
                      '<strong>Bootstrapping via question rephrasing</strong>: <a href="">MetaMath</a> rewrites questions using techniques like semantic rephrasing, self-verification, and backward reasoning to improve performance.',
                    children: [],
                    payload: { lines: "10,11" },
                  },
                  {
                    content:
                      '<strong>Answer format optimization</strong>: <a href="">GAIR-Abel</a> shows that paraphrasing the question followed by step-by-step solutions outperforms vanilla answer formats.',
                    children: [],
                    payload: { lines: "11,12" },
                  },
                  {
                    content:
                      '<strong>Data scaling</strong>: <a href="">Xwin-Math</a> demonstrates that scaling synthetic data up to one million examples benefits models like LLaMA-2 7B.',
                    children: [],
                    payload: { lines: "12,13" },
                  },
                  {
                    content:
                      '<strong>Dataset bundling</strong>: <a href="">MMIQC</a> combines SFT-style data (rephrased from MetaMath or taken directly) with high-quality pre-training data like OpenWebMath for better performance.',
                    children: [],
                    payload: { lines: "13,14" },
                  },
                  {
                    content:
                      '<strong>Verification of synthetic data</strong>: <a href="">AlphaGeometry</a> employs a symbolic deduction engine and 100 million synthetic data points for complex geometry problem-solving, achieving results comparable to Olympiad-level performance.',
                    children: [],
                    payload: { lines: "14,16" },
                  },
                ],
                payload: { lines: "7,8" },
              },
              {
                content: "Code",
                children: [
                  {
                    content:
                      '<strong><a href="">CodeRL</a>:</strong> An actor-critic approach that improves pretrained language models by using feedback signals on synthetic code samples for better code reasoning.',
                    children: [],
                    payload: { lines: "17,18" },
                  },
                  {
                    content:
                      '<strong><a href="">Language Models Can Teach Themselves to Program Better</a>:</strong> A self-improvement strategy where models generate synthetic puzzle-solution pairs, verified by a real interpreter before being used for fine-tuning.',
                    children: [],
                    payload: { lines: "18,19" },
                  },
                  {
                    content:
                      '<strong><a href="">Learning Performance-Improving Code Edits</a>:</strong> A framework leveraging simulated environments and adaptation strategies like self-improvement synthetic data generation and CoT prompting for code optimization.',
                    children: [],
                    payload: { lines: "19,20" },
                  },
                  {
                    content:
                      '<strong><a href="">InterCode</a>:</strong> A framework designed to enhance interactive code generation using reinforcement learning, where code acts as actions and execution feedback as observations.',
                    children: [],
                    payload: { lines: "20,21" },
                  },
                  {
                    content:
                      '<strong><a href="">Reflexion</a>:</strong> Employs external or simulated linguistic feedback signals to improve code reasoning capabilities of language models.',
                    children: [],
                    payload: { lines: "21,22" },
                  },
                  {
                    content:
                      '<strong><a href="">Code Alpaca</a>:</strong> A synthetic dataset of 20K code instructions generated via SELF-INSTRUCT applied to ChatGPT across 21 seed tasks.',
                    children: [],
                    payload: { lines: "22,23" },
                  },
                  {
                    content:
                      '<strong><a href="">WizardCoder</a>:</strong> Introduces Code Evol-Instruct, using heuristic prompts to guide ChatGPT in creating more complex and diverse synthetic data.',
                    children: [],
                    payload: { lines: "23,24" },
                  },
                  {
                    content:
                      '<strong><a href="">Magicoder</a>:</strong> Developed OSS-INSTRUCT, which generates 75K diverse synthetic instruction samples from open-source code snippets.',
                    children: [],
                    payload: { lines: "24,26" },
                  },
                ],
                payload: { lines: "16,17" },
              },
              {
                content: "Other reasoning tasks",
                children: [
                  {
                    content:
                      '<strong><a href="">Symbol tuning</a></strong>: Augmented natural language datasets by replacing labels with arbitrary symbols, generating 500k+ synthetic examples, improving model performance on unseen in-context learning and algorithmic reasoning tasks.',
                    children: [],
                    payload: { lines: "27,28" },
                  },
                  {
                    content:
                      '<strong><a href="">STaR</a></strong>: Generated synthetic chain-of-thought rationales and filtered out incorrect rationales for finetuning language models to enhance reasoning capabilities.',
                    children: [],
                    payload: { lines: "28,29" },
                  },
                  {
                    content:
                      '<strong><a href="">Mind’s Eye</a></strong>: Trained a text-to-code model with synthetic "text-description → rendering code" data, using a physical engine (MuJoCo) to boost reasoning performance in physics, achieving results comparable to models 100x larger.',
                    children: [],
                    payload: { lines: "29,31" },
                  },
                ],
                payload: { lines: "26,27" },
              },
            ],
            payload: { lines: "5,6" },
          },
          {
            content: "Tool-using and Planning",
            children: [
              {
                content: "Learning tool-using through synthetic trajectories",
                children: [
                  {
                    content:
                      '<strong><a href="">LaMDA</a></strong>: Trained on both web documents and synthetic interaction data between crowdworkers and the model, annotated with tool calls, allowing it to learn calculator, search engine, and machine translator usage.',
                    children: [],
                    payload: { lines: "34,35" },
                  },
                  {
                    content:
                      '<strong><a href="">Toolformer</a></strong>: Learns to decide which APIs to call and what arguments to pass through training on template-generated synthetic data.',
                    children: [],
                    payload: { lines: "35,36" },
                  },
                  {
                    content:
                      '<strong><a href="">Galactica</a></strong>: Integrates API-calling data into its pre-training mixture, enhancing tool-use capabilities during pre-training.',
                    children: [],
                    payload: { lines: "36,37" },
                  },
                  {
                    content:
                      '<strong><a href="">ToolAlpaca</a></strong>: Generates a diverse tool-use corpus by simulating a multi-agent environment where agents iteratively select and use tools.',
                    children: [],
                    payload: { lines: "37,39" },
                  },
                ],
                payload: { lines: "33,34" },
              },
              {
                content: "Learning to plan in synthetic environments",
                children: [
                  {
                    content:
                      '<strong><a href="">Planning in Autonomous Machine Intelligence</a></strong>: The agent decomposes complex tasks into subtasks and completes them in a reward-optimal manner.',
                    children: [],
                    payload: { lines: "40,41" },
                  },
                  {
                    content:
                      '<strong><a href="">Synthetic Data as Feedback</a></strong>: Synthetic data, collected from simulators, serves as feedback to aid the agent in planning tasks.',
                    children: [],
                    payload: { lines: "41,42" },
                  },
                  {
                    content:
                      '<strong><a href="">Affordance Awareness</a></strong>: Learning from synthetic data helps agents become aware of affordances, enhancing their ability to act in environments.',
                    children: [],
                    payload: { lines: "42,43" },
                  },
                  {
                    content:
                      '<strong><a href="">Inner Monologue</a></strong>: Uses natural language feedback from simulated environments to teach LLM-based robots planning, improving instruction completion in both simulated and real-world tasks.',
                    children: [],
                    payload: { lines: "43,44" },
                  },
                  {
                    content:
                      '<strong><a href="">VIMA</a></strong>: Develops a multi-modality simulated environment, VIMA-Bench, to generate realistic planning tasks like object rearrangement, supporting a variety of objects and textures.',
                    children: [],
                    payload: { lines: "44,45" },
                  },
                  {
                    content:
                      '<strong><a href="">Voyager</a></strong>: Deploys GPT-4 based agents in Minecraft, finding that synthetic feedback helps unlock new skills faster and makes planning more efficient.',
                    children: [],
                    payload: { lines: "45,47" },
                  },
                ],
                payload: { lines: "39,40" },
              },
            ],
            payload: { lines: "31,32" },
          },
          {
            content: "Multimodality",
            children: [
              {
                content: "Reverse rendering from vision to text",
                children: [
                  {
                    content:
                      "<strong>Web-scraped Image-Caption Pairs</strong>: Popular for multimodal alignment, but often noisy and provide only coarse-grained correspondence, limiting fine-grained grounding of images in language.",
                    children: [],
                    payload: { lines: "50,51" },
                  },
                  {
                    content:
                      '<strong><a href="">Pix2Struct</a></strong>: Utilizes web servers to render HTML into website screenshots, training the model to derender masked screenshots back into full HTML code.',
                    children: [],
                    payload: { lines: "51,52" },
                  },
                  {
                    content:
                      '<strong><a href="">MatCha</a> and <a href="">DePlot</a></strong>: Render tabular data into charts using Python libraries, pretraining models by pairing rendered images with corresponding code or tabular data.',
                    children: [],
                    payload: { lines: "52,53" },
                  },
                  {
                    content:
                      '<strong><a href="">Design2Code</a> and <a href="">WebSight</a></strong>: Train on synthetically generated HTML and CSS files to convert webpage screenshots into code implementations, achieving good generalization on real-world data.',
                    children: [],
                    payload: { lines: "53,54" },
                  },
                  {
                    content:
                      '<strong><a href="">Unity Perception</a></strong>: Proposes using physics or game engines (e.g., Unity) as synthetic data generators to enhance computer vision research.',
                    children: [],
                    payload: { lines: "54,56" },
                  },
                ],
                payload: { lines: "49,50" },
              },
              {
                content: "Multi-modality instruction following",
                children: [
                  {
                    content:
                      '<strong><a href="">LLaVA</a></strong>: Uses existing image captions to prompt GPT-4 in text-only mode for generating diverse long-form question-response pairs, which can be used in multimodal LLM training.',
                    children: [],
                    payload: { lines: "57,58" },
                  },
                  {
                    content:
                      '<strong><a href="">Svit</a></strong>: Incorporates object bounding box data as image attribute input for multimodal LLMs, fitting into the synthetic data pipeline for image attributes + text.',
                    children: [],
                    payload: { lines: "58,59" },
                  },
                  {
                    content:
                      '<strong><a href="">Llavar</a></strong>: Leverages Optical Character Recognition (OCR) data from images as another source of image attribute information for multimodal LLM training.',
                    children: [],
                    payload: { lines: "59,60" },
                  },
                  {
                    content:
                      '<strong><a href="">UniChart</a></strong>: Integrates data from derendered charts into the multimodal LLM pipeline, using chart attributes and text as inputs for training synthetic data.',
                    children: [],
                    payload: { lines: "60,62" },
                  },
                ],
                payload: { lines: "56,57" },
              },
            ],
            payload: { lines: "47,48" },
          },
          {
            content: "Multilingual",
            children: [
              {
                content: "Back-translation augmentation",
                children: [
                  {
                    content:
                      '<strong>Backtranslation for Data Augmentation</strong>: Many multilingual models utilize backtranslation to generate synthetic parallel training data from monolingual sources, enhancing translation performance (<a href="">Improving Neural Machine Translation Models with Monolingual Data</a>, <a href="">Improving Neural Machine Translation Models with Monolingual Data</a>, <a href="">Tagged Back-Translation</a>, <a href="">Tagged Back-translation Revisited</a>, <a href="">Data Augmentation for Text Generation Without Any Augmented Data</a> <a href="">Back-translation for Large-Scale Multilingual Machine Translation</a> <a href="">Meta Back-Translation</a>, <a href="">On Synthetic Data for Back Translation</a>).',
                    children: [],
                    payload: { lines: "64,65" },
                  },
                  {
                    content:
                      '<strong>Sampling Methods Exploration</strong>: Researchers have investigated various sampling methods for backtranslation (beam search, constrained sampling, unconstrained sampling) to assess their comparative effectiveness in generating high-quality synthetic data (<a href="">Improving Neural Machine Translation Models with Monolingual Data</a>, <a href="">Understanding Back-Translation at Scale</a>, <a href="">Generalizing Back-Translation in Neural Machine Translation</a>, <a href="">Paraphrasing with Bilingual Parallel Corpora</a>).',
                    children: [],
                    payload: { lines: "65,66" },
                  },
                  {
                    content:
                      '<strong>Optimization of Synthetic Data Quality</strong>: <a href="">On Synthetic Data for Back Translation</a> highlighted the importance of balancing the weight and quality of synthetic data for optimal neural machine translation (NMT) performance, proposing a method to optimize the ratio of search methods alongside a gamma score for improved effectiveness.',
                    children: [],
                    payload: { lines: "66,67" },
                  },
                  {
                    content:
                      '<strong>Limitations of Backtranslation</strong>: Backtranslation\'s effectiveness can be limited by the quality and diversity of the synthetic data, as poor performance in backtranslation may lead to noisy or insufficiently diverse data, ultimately restricting performance gains (<a href="">Improving backtranslation with iterative filtering and data selection for sinhala-english nmt.</a>, <a href="">Improved unsupervised neural machine translation with semantically weighted back translation for morphologically rich and low resource languages.</a>)',
                    children: [],
                    payload: { lines: "67,69" },
                  },
                ],
                payload: { lines: "63,64" },
              },
              {
                content:
                  "Generating multilingual questions and answers at scale",
                children: [
                  {
                    content:
                      '<strong>Translation of Monolingual QA</strong>: Translating existing monolingual questions and/or answers into other languages to enhance multilingual capabilities (<a href="">One Question Answering Model for Many Languages with Cross-lingual Dense Passage Retrieval</a>).',
                    children: [],
                    payload: { lines: "70,71" },
                  },
                  {
                    content:
                      '<strong>Cross-lingual Question Generation</strong>: Using Question Generation models to generate synthetic questions in a cross-lingual manner based on provided answers and/or source texts (<a href="">Cross-Lingual Training for Automatic Question Generation</a>, <a href="">Cross-Lingual Natural Language Generation via Pre-Training</a>, <a href="">Synthetic Data Augmentation for Zero-Shot Cross-Lingual Question Answering</a>).',
                    children: [],
                    payload: { lines: "71,72" },
                  },
                  {
                    content:
                      '<strong>Joint Generation of Multilingual QA</strong>: Focusing on the simultaneous generation of questions and answers across multiple languages for improved flexibility in language model training (<a href="">Towards Zero-Shot Multilingual Synthetic Question and Answer Generation for Cross-Lingual Reading Comprehension</a>, <a href="">PAXQA</a>).',
                    children: [],
                    payload: { lines: "72,73" },
                  },
                  {
                    content:
                      '<strong>Fine-tuning Multilingual Models</strong>: Fine-tuning a pretrained multilingual T5 model on a combination of QA generation tasks and multilingual masked language modeling to create synthetic QA pairs in various languages (<a href="">Towards Zero-Shot Multilingual Synthetic Question and Answer Generation for Cross-Lingual Reading Comprehension</a>).',
                    children: [],
                    payload: { lines: "73,75" },
                  },
                ],
                payload: { lines: "69,70" },
              },
            ],
            payload: { lines: "62,63" },
          },
          {
            content: "Alignment",
            children: [
              {
                content: "Instruction Following",
                children: [
                  {
                    content:
                      '<strong><a href="">Self-Instruct</a></strong>: Utilizes LLMs to generate instruction-following data by imitating a small set of seed samples, expanding the variety of training scenarios.',
                    children: [],
                    payload: { lines: "78,79" },
                  },
                  {
                    content:
                      '<strong><a href="">Stanford Alpaca</a></strong>: Similar to Self-Instruct, this approach generates additional instruction-following examples using LLMs based on initial seed data to enhance training coverage.',
                    children: [],
                    payload: { lines: "79,80" },
                  },
                  {
                    content:
                      '<strong><a href="">Evol-Instruct</a></strong>: Introduces complexity to simple instructions through advanced prompting techniques, aiming to enrich the quality of generated data.',
                    children: [],
                    payload: { lines: "80,81" },
                  },
                  {
                    content:
                      '<strong><a href="">FLAN Dataset Revision</a></strong>: Implements an iterative revision process using LLMs to enhance instructions and responses, resulting in improved model performance across various NLP tasks.',
                    children: [],
                    payload: { lines: "81,82" },
                  },
                  {
                    content:
                      '<strong><a href="">UltraChat</a></strong>: Creates a large-scale synthetic dialogue dataset by employing two ChatGPT Turbo API models, simulating user behavior through carefully crafted prompts.',
                    children: [],
                    payload: { lines: "82,83" },
                  },
                  {
                    content:
                      '<strong><a href="">Synthetic Data for Robustness</a></strong>: Generates synthetic data to mitigate sycophantic tendencies in models, incorporating this data in a finetuning step to promote balanced responses regardless of user opinions.',
                    children: [],
                    payload: { lines: "83,85" },
                  },
                ],
                payload: { lines: "77,78" },
              },
              {
                content: "Mitigating hallucination",
                children: [
                  {
                    content:
                      "<strong>Reward Model in GPT-4</strong>: Leveraged synthetic hallucination data for reinforcement learning, significantly improving performance on the TruthfulQA dataset.",
                    children: [],
                    payload: { lines: "86,87" },
                  },
                  {
                    content:
                      '<strong><a href="">Synthetic Task Design</a></strong>: Created a task for evaluating hallucinations, optimizing outputs by learning a continuous postfix via prefixtuning.',
                    children: [],
                    payload: { lines: "87,88" },
                  },
                  {
                    content:
                      '<strong><a href="">Automated Fact-Checking</a></strong>: Utilized confidence scores to rank factuality of model responses, finetuning with DPO to enhance factual accuracy.',
                    children: [],
                    payload: { lines: "88,90" },
                  },
                ],
                payload: { lines: "85,86" },
              },
              {
                content: "Aligning with shared human preference and values",
                children: [
                  {
                    content:
                      "<strong>Direct Finetuning on Human-Preferred Data</strong>: A straightforward alignment method requiring substantial human annotation, often leading to high costs and inconsistent quality in the annotated samples.",
                    children: [],
                    payload: { lines: "91,92" },
                  },
                  {
                    content:
                      "<strong>Reinforcement Learning from Human Feedback (RLHF)</strong>: Trains a reward model with human data to serve as a proxy for human judgment, optimizing the language model's generation policy based on this feedback.",
                    children: [],
                    payload: { lines: "92,93" },
                  },
                  {
                    content:
                      '<strong>Mixture of Synthetic and Real Human Data</strong>: Combines synthetic data with real human data to enhance the robustness of reward models, providing a more diverse training set (<a href="">Scaling laws for reward model overoptimization</a>).',
                    children: [],
                    payload: { lines: "93,94" },
                  },
                  {
                    content:
                      '<strong><a href="">Constitutional AI</a></strong>: Utilizes a small set of guiding principles to generate critiques and feedback, allowing for synthetic data to replace real human data in RLHF pipelines, achieving strong performance similar to RLHF baselines.',
                    children: [],
                    payload: { lines: "94,95" },
                  },
                  {
                    content:
                      '<strong>Synthetic Data for Value Alignment</strong>: Offers a cost-effective way to generate large, diverse datasets that simulate various ethical dilemmas, social interactions, and cultural norms, aiding in comprehensive testing of AI models\' alignment with human values (<a href="">Ultrafeedback</a>, <a href="">Red teaming language models to reduce harms</a>, <a href="">Red Teaming Language Models with Language Models</a>, <a href="">Training Socially Aligned Language Models on Simulated Social Interactions</a>, <a href="">NormBank</a>).',
                    children: [],
                    payload: { lines: "95,96" },
                  },
                  {
                    content:
                      '<strong>Identification and Mitigation of Bias and Fairness Issues</strong>: Enables systematic testing to identify biases and fairness issues before real-world deployment, helping to prevent unintended consequences (<a href="">Mitigating Political Bias in Language Models through Reinforced Calibration</a>, <a href="">Bias in data-driven artificial intelligence systems</a>, <a href="">Gender Bias in Coreference Resolution</a>, <a href="">Auditing the ai auditors</a>, <a href="">ToolSword</a>).',
                    children: [],
                    payload: { lines: "96,97" },
                  },
                  {
                    content:
                      '<strong>Risks of Low-Fidelity Synthetic Data</strong>: Acknowledges limitations in accurately reflecting nuanced human judgment, which may lead to vulnerabilities under specific attacks and deceptive behaviors (<a href="">Out of one, many</a>, <a href="">Is ChatGPT better than Human Annotators? Potential and Limitations of ChatGPT in Explaining Implicit Hate Speech</a>, <a href="">Toxicity in ChatGPT</a>, <a href="">The Effects of Reward Misspecification</a>, <a href="">ML Systems Will Have Weird Failure Modes</a>, <a href="">Reward tampering problems and solutions in reinforcement learning</a>).',
                    children: [],
                    payload: { lines: "97,99" },
                  },
                ],
                payload: { lines: "90,91" },
              },
            ],
            payload: { lines: "75,76" },
          },
        ],
        payload: { lines: "3,4" },
      },
      {
        content: "Synthetic Data in Evaluation",
        children: [
          {
            content: "&nbsp;",
            children: [
              {
                content: "Factuality",
                children: [
                  {
                    content:
                      'Early statistical-based hallucination evaluation methods used n-grams to calculate vocabulary overlap between input and output but failed to account for semantics (<a href="">Handling divergent reference texts when evaluating table-to-text generation</a>, <a href="">Towards faithful neural table-to-text generation with content-matching constraints</a>).',
                    children: [],
                    payload: { lines: "102,103" },
                  },
                  {
                    content:
                      'Statistical methods were limited as they only considered lexical overlap and couldn\'t evaluate complex hallucinations (<a href=""> Survey of hallucination in natural language generation</a>).',
                    children: [],
                    payload: { lines: "103,104" },
                  },
                  {
                    content:
                      'Model-based methods replaced statistical approaches, providing more robustness than token-difference-based methods (<a href="">Evaluating factual consistency in knowledge-grounded dialogues via question generation and question answering</a>).',
                    children: [],
                    payload: { lines: "104,105" },
                  },
                  {
                    content:
                      'Model-based methods can measure hallucination severity but struggle to identify specific factual errors (<a href="">Ranking generated summaries by correctness</a>).',
                    children: [],
                    payload: { lines: "105,106" },
                  },
                  {
                    content:
                      'Combining LLM generation with random walks on knowledge graphs helps generate synthetic evaluation data by focusing on entities and relations (<a href="">FactKB</a>).',
                    children: [],
                    payload: { lines: "106,107" },
                  },
                  {
                    content:
                      'The <a href="">LongFact</a> dataset has been created created for long-form factuality evaluation, using Google Search as grounding and LLM for automated judgement, achieving human-level accuracy at a lower cost ().',
                    children: [],
                    payload: { lines: "107,109" },
                  },
                ],
                payload: { lines: "101,102" },
              },
              {
                content: "Safety",
                children: [
                  {
                    content:
                      '<strong>Red teaming</strong> generates diverse, realistic scenarios to identify unaligned or harmful outputs in AI models (<a href=""> Explore, establish, exploit</a>).',
                    children: [],
                    payload: { lines: "110,111" },
                  },
                  {
                    content:
                      '<strong><a href="">Red teaming language models with language models</a></strong> used LMs to create 154 high-quality datasets to evaluate other LMs, discovering new inverse scaling issues.',
                    children: [],
                    payload: { lines: "111,112" },
                  },
                  {
                    content:
                      '<strong><a href=""> Sleeper agents</a></strong> leveraged synthetic data to trigger backdoor attacks, revealing deceptive behavior in LMs and limitations of standard safety training.',
                    children: [],
                    payload: { lines: "112,113" },
                  },
                  {
                    content:
                      '<strong>AI assistance</strong> can help scale human oversight in addressing complex and unseen domains (<a href="">Measuring progress on scalable oversight for large language models</a>).',
                    children: [],
                    payload: { lines: "113,115" },
                  },
                ],
                payload: { lines: "109,110" },
              },
              {
                content: "Assisting human evaluation",
                children: [
                  {
                    content:
                      '<strong><a href="">Alpaca Eval</a>:</strong> A benchmark that uses GPT-4 as a judge to assess the comprehensive abilities of LM-based chatbots.',
                    children: [],
                    payload: { lines: "116,117" },
                  },
                  {
                    content:
                      '<strong><a href="">MT Bench</a>:</strong> Another benchmark using GPT-4 to evaluate the capabilities of LM-based chatbots, focusing on various aspects of performance.',
                    children: [],
                    payload: { lines: "117,118" },
                  },
                  {
                    content:
                      '<strong><a href="">CRUXEval</a>:</strong> A code execution reasoning benchmark with 800 Python functions generated by CodeLLaMA-34B to evaluate coding task performance.',
                    children: [],
                    payload: { lines: "118,119" },
                  },
                  {
                    content:
                      '<strong><a href="">CodeMind</a>:</strong> A framework that evaluates LLMs on code reasoning abilities across Independent Execution Reasoning (IER), Dependent Execution Reasoning (DER), and Specification Reasoning (SR).',
                    children: [],
                    payload: { lines: "119,122" },
                  },
                ],
                payload: { lines: "115,116" },
              },
            ],
            payload: { lines: "100,101" },
          },
        ],
        payload: { lines: "99,100" },
      },
      {
        content: "Challenges and Limitations of Synthetic Data",
        children: [
          {
            content: "&nbsp;",
            children: [
              {
                content:
                  "Misuse of synthetic data might proliferate misinformation",
                children: [
                  {
                    content:
                      "<strong>AI systems' capability to generate human-like data</strong>: Models can now produce text, images, songs, and videos, raising concerns about their misuse.",
                    children: [],
                    payload: { lines: "125,126" },
                  },
                  {
                    content:
                      "<strong>Risks of synthetic data impersonating real people</strong>: Synthetic data could be exploited for malicious purposes, such as impersonation, public opinion manipulation, and political interference.",
                    children: [],
                    payload: { lines: "126,127" },
                  },
                  {
                    content:
                      "<strong>Erosion of trust through synthetic data-driven misinformation</strong>: The spread of misinformation generated by synthetic data threatens the credibility of legitimate information sources.",
                    children: [],
                    payload: { lines: "127,128" },
                  },
                  {
                    content:
                      "<strong>Need for ethical guidelines and detection mechanisms</strong>: Researchers, developers, and policymakers must implement clear ethical guidelines and technologies to detect and combat synthetic misinformation .",
                    children: [],
                    payload: { lines: "128,130" },
                  },
                ],
                payload: { lines: "124,125" },
              },
              {
                content: "Synthetic data might cause ambiguity in AI alignment",
                children: [
                  {
                    content:
                      "<strong>Constitutional AI</strong>: The use of synthetic data, like in Constitutional AI, may introduce ambiguity and uncertainty, affecting the model's ability to align with human values.",
                    children: [],
                    payload: { lines: "131,132" },
                  },
                  {
                    content:
                      "<strong>Representation of Human Values</strong>: Synthetic data may fail to accurately represent human values and preferences, potentially leading to misaligned AI behaviors.",
                    children: [],
                    payload: { lines: "132,133" },
                  },
                  {
                    content:
                      "<strong>Bias in Synthetic Data</strong>: Models trained on synthetic data may learn biased information, resulting in misrepresentations of real-world scenarios.",
                    children: [],
                    payload: { lines: "133,134" },
                  },
                  {
                    content:
                      "<strong>Ungrounded Learning</strong>: AI models might be trained on ungrounded data, leading to a lack of real-world applicability.",
                    children: [],
                    payload: { lines: "134,135" },
                  },
                  {
                    content:
                      "<strong>Misrepresentation of Real-World Scenarios</strong>: Synthetic data can misrepresent complex real-world situations, causing AI models to exhibit behaviors that diverge from human expectations.",
                    children: [],
                    payload: { lines: "135,136" },
                  },
                  {
                    content:
                      "<strong>Unintended Consequences</strong>: Misalignment due to synthetic data can lead to unintended or harmful behaviors in AI systems.",
                    children: [],
                    payload: { lines: "136,137" },
                  },
                  {
                    content:
                      "<strong>Interpretability Challenges</strong>: The ambiguity introduced by synthetic data complicates the interpretation of AI decision-making processes, making alignment more difficult.",
                    children: [],
                    payload: { lines: "137,139" },
                  },
                ],
                payload: { lines: "130,131" },
              },
              {
                content:
                  "Training with synthetic data makes evaluation decontamination harder",
                children: [
                  {
                    content:
                      "Synthetic data complicates fair evaluation in model training due to potential overlap with public benchmark test cases.",
                    children: [],
                    payload: { lines: "140,141" },
                  },
                  {
                    content:
                      "Publicly available benchmarks, often sourced from online platforms, risk being included in pre-training data, affecting evaluation fairness.",
                    children: [],
                    payload: { lines: "141,142" },
                  },
                  {
                    content:
                      "Synthetic data use worsens evaluation contamination rather than reducing it, particularly when it rephrases existing benchmark data.",
                    children: [],
                    payload: { lines: "142,143" },
                  },
                  {
                    content:
                      "Token-level decontamination methods, such as min-k% prob, are insufficient for models trained with synthetic data.",
                    children: [],
                    payload: { lines: "143,144" },
                  },
                  {
                    content:
                      "More advanced techniques for evaluation contamination detection are necessary, alongside the creation of protected, proprietary benchmarks for reliable evaluation.",
                    children: [],
                    payload: { lines: "144,146" },
                  },
                ],
                payload: { lines: "139,140" },
              },
            ],
            payload: { lines: "123,124" },
          },
        ],
        payload: { lines: "122,123" },
      },
      {
        content: "Directions for Future Work",
        children: [
          {
            content: "&nbsp;",
            children: [
              {
                content: "Synthetic data scaling",
                children: [
                  {
                    content:
                      "<strong>Over-trained small language models:</strong> The performance of models like the Mistral and Gemma series demonstrates that training on large amounts of data, even exceeding the compute-optimal Chinchilla law, leads to impressive results.",
                    children: [],
                    payload: { lines: "149,150" },
                  },
                  {
                    content:
                      "<strong>Synthetic data in training:</strong> It remains an open question whether training with synthetic data can yield similar conclusions, as its quality is often inconsistent compared to real-world data.",
                    children: [],
                    payload: { lines: "150,151" },
                  },
                  {
                    content:
                      "<strong>Scaling laws for synthetic data:</strong> Future research should investigate the optimal balance between the quantity and quality of synthetic data for large-scale language model training.",
                    children: [],
                    payload: { lines: "151,152" },
                  },
                  {
                    content:
                      "<strong>Cost-efficient strategies with synthetic data:</strong> Exploring effective strategies for using synthetic data could lead to more efficient and cost-effective training approaches for large language models.",
                    children: [],
                    payload: { lines: "152,154" },
                  },
                ],
                payload: { lines: "148,149" },
              },
              {
                content:
                  "Further improving quality and diversity of synthetic data",
                children: [
                  {
                    content:
                      "<strong>Generative Adversarial Networks (GANs):</strong> Focus on using GANs to improve the quality of synthetic data by closely mimicking real-world data, with emphasis on controlling specific attributes.",
                    children: [],
                    payload: { lines: "155,156" },
                  },
                  {
                    content:
                      "<strong>Diffusion Models:</strong> Investigate the use of Diffusion Models to generate high-quality synthetic data with more granular control over data attributes.",
                    children: [],
                    payload: { lines: "156,157" },
                  },
                  {
                    content:
                      "<strong>Retrieval Augmented Generation (RAG):</strong> Explore incorporating domain-specific knowledge into synthetic data generation through methods like RAG, ensuring that the data adheres to domain constraints while maintaining quality.",
                    children: [],
                    payload: { lines: "157,158" },
                  },
                  {
                    content:
                      "<strong>Privacy-Preserving Analysis:</strong> Improve synthetic data techniques to support privacy-preserving analysis across fields like healthcare and finance.",
                    children: [],
                    payload: { lines: "158,161" },
                  },
                ],
                payload: { lines: "154,155" },
              },
              {
                content:
                  "Towards high-fidelity and more efficient scalable oversight",
                children: [
                  {
                    content:
                      '<strong><a href="">Debate</a>:</strong> Simulates social iterations by having AI models engage in a structured debate to expose and correct errors.',
                    children: [],
                    payload: { lines: "162,163" },
                  },
                  {
                    content:
                      '<strong><a href="">Reflection</a>:</strong> Uses AI models\' introspection and self-evaluation to generate synthetic data for oversight by reflecting on their decisions.',
                    children: [],
                    payload: { lines: "163,164" },
                  },
                  {
                    content:
                      '<strong><a href="">Revisions</a>:</strong> Focuses on iterative correction of AI model outputs to improve oversight using synthetic data.',
                    children: [],
                    payload: { lines: "164,165" },
                  },
                  {
                    content:
                      '<strong><a href="">Comprehensive Scenarios and Modalities</a>:</strong> Proposes exploring a broader range of scenarios and modalities in synthetic data generation for better oversight.',
                    children: [],
                    payload: { lines: "165,166" },
                  },
                  {
                    content:
                      '<strong><a href="">Narrowed-down Issues</a>:</strong> Highlights problems arising from simulation of overly specific and narrow scenarios.',
                    children: [],
                    payload: { lines: "166,167" },
                  },
                  {
                    content:
                      '<strong><a href="">Over-simplified Scenes</a>:</strong> Identifies issues with simulations that are too simplistic to provide meaningful oversight.',
                    children: [],
                    payload: { lines: "167,169" },
                  },
                ],
                payload: { lines: "161,162" },
              },
              {
                content: "The emergent self-improvement capability",
                children: [
                  {
                    content:
                      "<strong>Synthetic Data Quality</strong>: Using the most capable model to generate synthetic data can lead to higher quality outputs compared to less advanced models.",
                    children: [],
                    payload: { lines: "170,171" },
                  },
                  {
                    content:
                      "<strong>Self-Improvement Potential</strong>: Investigating whether models can generate synthetic data superior to their training data, potentially enabling self-improvement.",
                    children: [],
                    payload: { lines: "171,172" },
                  },
                  {
                    content:
                      "<strong>Iterative Learning</strong>: A model could iteratively improve by learning from enhanced synthetic data, leading to continuous performance gains.",
                    children: [],
                    payload: { lines: "172,173" },
                  },
                ],
                payload: { lines: "169,170" },
              },
            ],
            payload: { lines: "147,148" },
          },
        ],
        payload: { lines: "146,147" },
      },
    ],
    payload: { lines: "1,2" },
  },
  { color: ["#2980b9"], maxWidth: 400, initialExpandLevel: 4,}
);
