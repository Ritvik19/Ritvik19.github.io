document.getElementById("title").innerHTML = "The Prompt Report: A Systematic Survey of Prompting Techniques";

((getMarkmap, getOptions, root2, jsonOptions) => {
  const markmap = getMarkmap();
  window.mm = markmap.Markmap.create(
    "svg#mindmap",
    (getOptions || markmap.deriveOptions)(jsonOptions),
    root2
  );
})(
  () => window.markmap,
  null,
  {
    content:
      '<a href="https://arxiv.org/abs/2406.06608">The Prompt Report: A Systematic Survey of Prompting Techniques</a>',
    children: [
      {
        content: "English Only",
        children: [
          {
            content: "Zero-Shot Prompting",
            children: [
              {
                content: "Role Prompting",
                children: [
                  {
                    content:
                      'Also known as persona prompting assigns a specific role to the GenAI in the prompt. For example to act like "Madonna" or a "travel writer".',
                    children: [],
                    payload: { lines: "7,8" },
                  },
                  {
                    content:
                      "This can create more desirable outputs for open-ended tasks and in some cases improve accuracy on benchmarks.",
                    children: [],
                    payload: { lines: "8,10" },
                  },
                ],
                payload: { lines: "6,7" },
              },
              {
                content: "Style Prompting",
                children: [
                  {
                    content:
                      "It involves specifying the desired style, tone, or genre in the prompt to shape the output of a GenAI.",
                    children: [],
                    payload: { lines: "11,12" },
                  },
                  {
                    content:
                      "A similar effect can be achieved using role prompting.",
                    children: [],
                    payload: { lines: "12,13" },
                  },
                  {
                    content:
                      '<a href="https://arxiv.org/abs/2302.09185">Bounding the Capabilities of Large Language Models in Open Text Generation with Prompt Constraints</a>',
                    children: [],
                    payload: { lines: "13,15" },
                  },
                ],
                payload: { lines: "10,11" },
              },
              {
                content: "Emotion Prompting",
                children: [
                  {
                    content:
                      'It incorporates phrases of psychological relevance to humans (e.g., "This is important to my career") into the prompt, which may lead to improved LLM performance on benchmarks and open-ended text generation.',
                    children: [],
                    payload: { lines: "16,17" },
                  },
                  {
                    content:
                      '<a href="https://arxiv.org/abs/2307.11760">Large Language Models Understand and Can be Enhanced by Emotional Stimuli</a>',
                    children: [],
                    payload: { lines: "17,19" },
                  },
                ],
                payload: { lines: "15,16" },
              },
              {
                content: "System 2 Attention (S2A)",
                children: [
                  {
                    content:
                      "It first asks an LLM to rewrite the prompt and remove any information unrelated to the question therein.",
                    children: [],
                    payload: { lines: "20,21" },
                  },
                  {
                    content:
                      "Then, it passes this new prompt into an LLM to retrieve a final response.",
                    children: [],
                    payload: { lines: "21,22" },
                  },
                  {
                    content:
                      '<a href="https://arxiv.org/abs/2311.11829">System 2 Attention (is something you might need too)</a>',
                    children: [],
                    payload: { lines: "22,24" },
                  },
                ],
                payload: { lines: "19,20" },
              },
              {
                content: "SimToM",
                children: [
                  {
                    content:
                      "It deals with complicated questions which involve multiple people or objects. Given the question, it attempts to establish the set of facts one person knows, then answer the question based only on those facts.",
                    children: [],
                    payload: { lines: "25,26" },
                  },
                  {
                    content:
                      "This is a two prompt process and can help eliminate the effect of irrelevant information in the prompt.",
                    children: [],
                    payload: { lines: "26,27" },
                  },
                  {
                    content:
                      '<a href="https://arxiv.org/abs/2311.10227">Think Twice: Perspective-Taking Improves Large Language Models\' Theory-of-Mind Capabilities</a>',
                    children: [],
                    payload: { lines: "27,29" },
                  },
                ],
                payload: { lines: "24,25" },
              },
              {
                content: "Rephrase and Respond (RaR)",
                children: [
                  {
                    content:
                      "It instructs the LLM to rephrase and expand the question before generating the final answer.",
                    children: [],
                    payload: { lines: "30,31" },
                  },
                  {
                    content:
                      "For example, it might add the following phrase to the<br>\nquestion: <code>Rephrase and expand the question, and respond</code>.",
                    children: [],
                    payload: { lines: "31,33" },
                  },
                  {
                    content:
                      "This could all be done in a single pass or the new question could be passed to the LLM separately.",
                    children: [],
                    payload: { lines: "33,34" },
                  },
                  {
                    content:
                      '<a href="https://arxiv.org/abs/2311.04205">Rephrase and Respond: Let Large Language Models Ask Better Questions for Themselves</a>',
                    children: [],
                    payload: { lines: "34,36" },
                  },
                ],
                payload: { lines: "29,30" },
              },
              {
                content: "Re-reading (RE2)",
                children: [
                  {
                    content:
                      "It adds the phrase <code>Read the question again:</code> to the prompt in addition to repeating the question.",
                    children: [],
                    payload: { lines: "37,38" },
                  },
                  {
                    content:
                      "Although this is such a simple technique, it has shown improvement in reasoning benchmarks, especially with complex questions.",
                    children: [],
                    payload: { lines: "38,40" },
                  },
                ],
                payload: { lines: "36,37" },
              },
              {
                content: "Self-Ask",
                children: [
                  {
                    content:
                      "It prompts LLMs to first decide if they need to ask follow up questions for a given prompt.",
                    children: [],
                    payload: { lines: "41,42" },
                  },
                  {
                    content:
                      "If so, the LLM generates these questions, then answers them and finally answers the original question.",
                    children: [],
                    payload: { lines: "42,44" },
                  },
                ],
                payload: { lines: "40,41" },
              },
            ],
            payload: { lines: "5,6" },
          },
          {
            content: "Few-Shot Prompting",
            children: [
              {
                content: "&nbsp;",
                children: [
                  {
                    content:
                      "In-Context Learning refers to the ability of GenAIs to learn skills and tasks by providing them with exemplars and or relevant instructions within the prompt, without the need for weight updates/retraining.",
                    children: [],
                    payload: { lines: "46,47" },
                  },
                  {
                    content:
                      "Few-Shot Prompting is the paradigm, where the GenAI learns to complete a task with only a few examples (exemplars).",
                    children: [],
                    payload: { lines: "47,48" },
                  },
                  {
                    content:
                      '<a href="https://arxiv.org/abs/2005.14165">Language Models are Few-Shot Learners</a>',
                    children: [],
                    payload: { lines: "48,50" },
                  },
                ],
                payload: { lines: "45,46" },
              },
            ],
            payload: { lines: "44,45" },
          },
          {
            content: "Thought Generation",
            children: [
              {
                content: "Chain-of-Thought (CoT) Prompting",
                children: [
                  {
                    content:
                      "It Prompts LLM to express its thought process before delivering its final answer.",
                    children: [],
                    payload: { lines: "52,53" },
                  },
                  {
                    content:
                      "It has been demonstrated to significantly enhance the LLM’s performance in mathematics and reasoning tasks.",
                    children: [],
                    payload: { lines: "53,54" },
                  },
                  {
                    content:
                      "It involves appending a thought inducing phrase like <code>Let’s think step by step.</code> or <code>Let’s work this out in a step by step way to be sure we have the right answer</code> or <code>First, let’s think about this logically</code> to the prompt.",
                    children: [],
                    payload: { lines: "54,55" },
                  },
                  {
                    content:
                      '<a href="https://arxiv.org/abs/2201.11903">Chain-of-Thought Prompting Elicits Reasoning in Large Language Models</a>',
                    children: [],
                    payload: { lines: "55,57" },
                  },
                ],
                payload: { lines: "51,52" },
              },
              {
                content: "Step-Back Prompting",
                children: [
                  {
                    content:
                      "It is a modification of CoT where the LLM is first asked a generic, high-level question about relevant concepts or facts before delving into reasoning.",
                    children: [],
                    payload: { lines: "58,59" },
                  },
                  {
                    content:
                      "This approach has improved performance significantly on multiple reasoning benchmarks for both PaLM2L and GPT-4.",
                    children: [],
                    payload: { lines: "59,60" },
                  },
                  {
                    content:
                      '<a href="https://arxiv.org/abs/2310.06117">Take a Step Back: Evoking Reasoning via Abstraction in Large Language Models</a>',
                    children: [],
                    payload: { lines: "60,62" },
                  },
                ],
                payload: { lines: "57,58" },
              },
              {
                content: "Analogical Prompting",
                children: [
                  {
                    content:
                      "It automatically generates exemplars that include CoTs.",
                    children: [],
                    payload: { lines: "63,64" },
                  },
                  {
                    content:
                      "It has demonstrated improvements in mathematical reasoning and code generation tasks.",
                    children: [],
                    payload: { lines: "64,65" },
                  },
                  {
                    content:
                      '<a href="https://arxiv.org/abs/2310.01714">Large Language Models as Analogical Reasoners</a>',
                    children: [],
                    payload: { lines: "65,67" },
                  },
                ],
                payload: { lines: "62,63" },
              },
              {
                content: "Thread-of-Thought (ThoT) Prompting",
                children: [
                  {
                    content:
                      "It consists of an improved thought inducer for CoT reasoning: <code>Walk me through this context in manageable parts step by step, summarizing and analyzing as we go.</code>",
                    children: [],
                    payload: { lines: "68,69" },
                  },
                  {
                    content:
                      "This thought inducer works well in question-answering and retrieval settings, especially when dealing with large, complex contexts.",
                    children: [],
                    payload: { lines: "69,70" },
                  },
                  {
                    content:
                      '<a href="https://arxiv.org/abs/2311.08734">Thread of Thought Unraveling Chaotic Contexts</a>',
                    children: [],
                    payload: { lines: "70,72" },
                  },
                ],
                payload: { lines: "67,68" },
              },
              {
                content: "Tabular Chain-of-Thought (Tab-CoT)",
                children: [
                  {
                    content:
                      "It consists of a Zero-Shot CoT prompt that makes the LLM output reasoning as a markdown table.",
                    children: [],
                    payload: { lines: "73,74" },
                  },
                  {
                    content:
                      "This tabular design enables the LLM to improve the structure and thus the reasoning of its output.",
                    children: [],
                    payload: { lines: "74,75" },
                  },
                  {
                    content:
                      '<a href="https://arxiv.org/abs/2305.17812">Tab-CoT: Zero-shot Tabular Chain of Thought</a>',
                    children: [],
                    payload: { lines: "75,77" },
                  },
                ],
                payload: { lines: "72,73" },
              },
              {
                content: "Contrastive CoT Prompting",
                children: [
                  {
                    content:
                      "It adds both exemplars with incorrect and correct explanations to the CoT prompt in order to show the LLM how not to reason.",
                    children: [],
                    payload: { lines: "78,79" },
                  },
                  {
                    content:
                      "This method has shown significant improvement in areas like Arithmetic Reasoning and Factual QA.",
                    children: [],
                    payload: { lines: "79,80" },
                  },
                  {
                    content:
                      '<a href="https://arxiv.org/abs/2311.09277">Contrastive Chain-of-Thought Prompting</a>',
                    children: [],
                    payload: { lines: "80,82" },
                  },
                ],
                payload: { lines: "77,78" },
              },
              {
                content: "Uncertainty-Routed CoT Prompting",
                children: [
                  {
                    content:
                      "It samples multiple CoT reasoning paths, then selects the majority if it is above a certain threshold.",
                    children: [],
                    payload: { lines: "83,84" },
                  },
                  {
                    content:
                      "If not, it samples greedily and selects that response.",
                    children: [],
                    payload: { lines: "84,85" },
                  },
                  {
                    content:
                      "This method demonstrates improvement on the MMLU benchmark for both GPT4 and Gemini Ultra models.",
                    children: [],
                    payload: { lines: "85,86" },
                  },
                  {
                    content:
                      '<a href="https://storage.googleapis.com/deepmind-media/gemini/gemini_1_report.pdf">Gemini: A family of highly capable multimodal models</a>',
                    children: [],
                    payload: { lines: "86,88" },
                  },
                ],
                payload: { lines: "82,83" },
              },
              {
                content: "Complexity-based Prompting",
                children: [
                  {
                    content: "It involves two major modifications to CoT.",
                    children: [],
                    payload: { lines: "89,90" },
                  },
                  {
                    content:
                      "First, it<br>\nselects complex examples for annotation and inclusion in the prompt, based on factors like question length or reasoning steps required.",
                    children: [],
                    payload: { lines: "90,92" },
                  },
                  {
                    content:
                      "Second, during inference, it samples multiple reasoning chains (answers) and uses a majority vote among chains exceeding a certain length threshold, under the premise that longer reasoning indicates higher answer quality.",
                    children: [],
                    payload: { lines: "92,93" },
                  },
                  {
                    content:
                      "This technique has shown improvements on three mathematical reasoning datasets.",
                    children: [],
                    payload: { lines: "93,94" },
                  },
                  {
                    content:
                      '<a href="https://arxiv.org/abs/2210.00720">Complexity-Based Prompting for Multi-Step Reasoning</a>',
                    children: [],
                    payload: { lines: "94,96" },
                  },
                ],
                payload: { lines: "88,89" },
              },
              {
                content: "Active Prompting",
                children: [
                  {
                    content:
                      "It starts with some training questions/exemplars, asks the LLM to solve them, then calculates uncertainty (disagreement in this case) and asks human annotators to rewrite the exemplars with highest uncertainty.",
                    children: [],
                    payload: { lines: "97,98" },
                  },
                  {
                    content:
                      '<a href="https://arxiv.org/abs/2302.12246">Active Prompting with Chain-of-Thought for Large Language Models</a>',
                    children: [],
                    payload: { lines: "98,100" },
                  },
                ],
                payload: { lines: "96,97" },
              },
              {
                content: "Memory-of-Thought Prompting",
                children: [
                  {
                    content:
                      "It leverage unlabeled training exemplars to build Few-Shot CoT prompts at test time.",
                    children: [],
                    payload: { lines: "101,102" },
                  },
                  {
                    content:
                      "Before test time, it performs inference on the unlabeled training exemplars with CoT.",
                    children: [],
                    payload: { lines: "102,103" },
                  },
                  {
                    content:
                      "At test time, it retrieves similar instances to the test sample.",
                    children: [],
                    payload: { lines: "103,104" },
                  },
                  {
                    content:
                      "This technique has shown substantial improvements in benchmarks like Arithmetic, commonsense, and factual reasoning.",
                    children: [],
                    payload: { lines: "104,105" },
                  },
                  {
                    content:
                      '<a href="https://arxiv.org/abs/2305.05181">MoT: Memory-of-Thought Enables ChatGPT to Self-Improve</a>',
                    children: [],
                    payload: { lines: "105,107" },
                  },
                ],
                payload: { lines: "100,101" },
              },
              {
                content: "Automatic Chain-of-Thought (Auto-CoT) Prompting",
                children: [
                  {
                    content:
                      "It uses Zero-Shot prompt to automatically generate chains of thought.",
                    children: [],
                    payload: { lines: "108,109" },
                  },
                  {
                    content:
                      "These are then used to build a Few-Shot CoT prompt for a test sample.",
                    children: [],
                    payload: { lines: "109,110" },
                  },
                  {
                    content:
                      '<a href="https://arxiv.org/abs/2210.03493">Automatic Chain of Thought Prompting in Large Language Models</a>',
                    children: [],
                    payload: { lines: "110,112" },
                  },
                ],
                payload: { lines: "107,108" },
              },
            ],
            payload: { lines: "50,51" },
          },
          {
            content: "Decomposition",
            children: [
              {
                content: "Least-to-Most Prompting",
                children: [
                  {
                    content:
                      "It starts by prompting a LLM to break a given problem into sub-problems without solving them.",
                    children: [],
                    payload: { lines: "115,116" },
                  },
                  {
                    content:
                      "Then, it solves them sequentially, appending model responses to the prompt each time, until it arrives at a final result.",
                    children: [],
                    payload: { lines: "116,117" },
                  },
                  {
                    content:
                      "This method has shown significant improvements in tasks involving symbolic manipulation, compositional generalization, and mathematical reasoning.",
                    children: [],
                    payload: { lines: "117,118" },
                  },
                  {
                    content:
                      '<a href="https://arxiv.org/abs/2205.10625">Least-to-Most Prompting Enables Complex Reasoning in Large Language Models</a>',
                    children: [],
                    payload: { lines: "118,120" },
                  },
                ],
                payload: { lines: "114,115" },
              },
              {
                content: "Decomposed Prompting (DECOMP)",
                children: [
                  {
                    content:
                      "It Few-Shot prompts a LLM to show it how to use certain functions.",
                    children: [],
                    payload: { lines: "121,122" },
                  },
                  {
                    content:
                      "These might include things like string splitting or internet searching; these are often implemented as separate LLM calls.",
                    children: [],
                    payload: { lines: "122,123" },
                  },
                  {
                    content:
                      "Given this, the LLM breaks down its original problem into sub-problems which it sends to different functions.",
                    children: [],
                    payload: { lines: "123,124" },
                  },
                  {
                    content:
                      '<a href="https://arxiv.org/abs/2210.02406">Decomposed Prompting: A Modular Approach for Solving Complex Tasks</a>',
                    children: [],
                    payload: { lines: "124,126" },
                  },
                ],
                payload: { lines: "120,121" },
              },
              {
                content: "Plan-and-Solve Prompting",
                children: [
                  {
                    content:
                      "It consists of an improved Zero-Shot CoT prompt: <code>Let’s first understand the problem and devise a plan to solve it. Then, let’s carry out the plan and solve the problem step by step.</code>",
                    children: [],
                    payload: { lines: "127,128" },
                  },
                  {
                    content:
                      "This method generates more robust reasoning processes than standard Zero-Shot-CoT on multiple reasoning datasets.",
                    children: [],
                    payload: { lines: "128,129" },
                  },
                  {
                    content:
                      '<a href="https://arxiv.org/abs/2305.04091">Plan-and-Solve Prompting: Improving Zero-Shot Chain-of-Thought Reasoning by Large Language Models</a>',
                    children: [],
                    payload: { lines: "129,131" },
                  },
                ],
                payload: { lines: "126,127" },
              },
              {
                content: "Tree-of-Thought (ToT)",
                children: [
                  {
                    content:
                      "Itcreates a tree-like search problem by starting with an initial problem then generating multiple possible steps in the form of thoughts (as from a CoT).",
                    children: [],
                    payload: { lines: "132,133" },
                  },
                  {
                    content:
                      "It evaluates the progress each step makes towards solving the problem (through prompting) and decides which steps to continue with, then keeps creating more thoughts.",
                    children: [],
                    payload: { lines: "133,134" },
                  },
                  {
                    content:
                      '<a href="https://arxiv.org/abs/2305.10601">Tree of Thoughts: Deliberate Problem Solving with Large Language Models</a>',
                    children: [],
                    payload: { lines: "134,135" },
                  },
                  {
                    content:
                      '<a href="https://arxiv.org/abs/2305.08291">Large Language Model Guided Tree-of-Thought</a>',
                    children: [],
                    payload: { lines: "135,137" },
                  },
                ],
                payload: { lines: "131,132" },
              },
              {
                content: "Recursion-of-Thought",
                children: [
                  {
                    content:
                      "Every time it encounters a complicated problem in the middle of its reasoning chain, it sends this problem into another prompt/LLM call.",
                    children: [],
                    payload: { lines: "138,139" },
                  },
                  {
                    content:
                      "After this is completed, the answer is inserted into the original prompt.",
                    children: [],
                    payload: { lines: "139,140" },
                  },
                  {
                    content:
                      "In this way, it can recursively solve complex problems, including ones which might otherwise run over that maximum context length.",
                    children: [],
                    payload: { lines: "140,141" },
                  },
                  {
                    content:
                      "This method has shown improvements on arithmetic and algorithmic tasks.",
                    children: [],
                    payload: { lines: "141,142" },
                  },
                  {
                    content:
                      '<a href="https://arxiv.org/abs/2306.06891">Recursion of Thought: A Divide-and-Conquer Approach to Multi-Context Reasoning with Language Models</a>',
                    children: [],
                    payload: { lines: "142,144" },
                  },
                ],
                payload: { lines: "137,138" },
              },
              {
                content: "Program-of-Thoughts",
                children: [
                  {
                    content:
                      "It uses Code LLMs to generate programming code<br>\nas reasoning steps.",
                    children: [],
                    payload: { lines: "145,147" },
                  },
                  {
                    content:
                      "A code interpreter executes these steps to obtain the final answer.",
                    children: [],
                    payload: { lines: "147,148" },
                  },
                  {
                    content:
                      "It excels in mathematical and programming-related tasks but is less effective for semantic reasoning tasks.",
                    children: [],
                    payload: { lines: "148,149" },
                  },
                  {
                    content:
                      '<a href="https://arxiv.org/abs/2211.12588">Program of Thoughts Prompting: Disentangling Computation from Reasoning for Numerical Reasoning Tasks</a>',
                    children: [],
                    payload: { lines: "149,151" },
                  },
                ],
                payload: { lines: "144,145" },
              },
              {
                content: "Faithful Chain-of-Thought",
                children: [
                  {
                    content:
                      "It generates a CoT that has both natural language and symbolic language (e.g. Python) reasoning, just like Program-of-Thoughts.",
                    children: [],
                    payload: { lines: "152,153" },
                  },
                  {
                    content:
                      "However, it also makes use of different types of symbolic languages in a task-dependent fashion.",
                    children: [],
                    payload: { lines: "153,154" },
                  },
                  {
                    content:
                      '<a href="https://arxiv.org/abs/2301.13379">Faithful Chain-of-Thought Reasoning</a>',
                    children: [],
                    payload: { lines: "154,156" },
                  },
                ],
                payload: { lines: "151,152" },
              },
              {
                content: "Skeleton-of-Thought",
                children: [
                  {
                    content:
                      "It focuses on accelerating answer speed through parallelization.",
                    children: [],
                    payload: { lines: "157,158" },
                  },
                  {
                    content:
                      "Given a problem, it prompts an LLM to create a skeleton of the answer, in a sense, sub-problems to be solved.",
                    children: [],
                    payload: { lines: "158,159" },
                  },
                  {
                    content:
                      "Then, in parallel, it sends these questions to an LLM and concatenates all the outputs to get a final response.",
                    children: [],
                    payload: { lines: "159,160" },
                  },
                  {
                    content:
                      '<a href="https://arxiv.org/abs/2307.15337">Skeleton-of-Thought: Prompting LLMs for Efficient Parallel Generation</a>',
                    children: [],
                    payload: { lines: "160,162" },
                  },
                ],
                payload: { lines: "156,157" },
              },
            ],
            payload: { lines: "112,113" },
          },
          {
            content: "Ensembling",
            children: [
              {
                content: "Demonstration Ensembling (DENSE)",
                children: [
                  {
                    content:
                      "It creates multiple few-shot prompts, each containing a distinct subset of exemplars from the training set.",
                    children: [],
                    payload: { lines: "164,165" },
                  },
                  {
                    content:
                      "Next, it aggregates over their outputs to generate a final response.",
                    children: [],
                    payload: { lines: "165,166" },
                  },
                  {
                    content:
                      '<a href="https://arxiv.org/abs/2308.08780">Exploring Demonstration Ensembling for In-context Learning</a>',
                    children: [],
                    payload: { lines: "166,168" },
                  },
                ],
                payload: { lines: "163,164" },
              },
              {
                content: "Mixture of Reasoning Experts (MoRE)",
                children: [
                  {
                    content:
                      "It creates a set of diverse reasoning experts by using different specialized prompts for different reasoning types (such as retrieval augmentation prompts for factual reasoning, Chain-of-Thought reasoning for multi-hop and math reasoning, and generated knowledge prompting for commonsense reasoning).",
                    children: [],
                    payload: { lines: "169,170" },
                  },
                  {
                    content:
                      "The best answer from all experts is selected based on an agreement score.",
                    children: [],
                    payload: { lines: "170,171" },
                  },
                  {
                    content:
                      '<a href="https://arxiv.org/abs/2305.14628">Getting MoRE out of Mixture of Language Model Reasoning Experts</a>',
                    children: [],
                    payload: { lines: "171,173" },
                  },
                ],
                payload: { lines: "168,169" },
              },
              {
                content: "Max Mutual Information Method",
                children: [
                  {
                    content:
                      "It creates multiple prompt templates with<br>\nvaried styles and exemplars, then selects the optimal template as the one that maximizes mutual<br>\ninformation between the prompt and the LLM’s<br>\noutputs.",
                    children: [],
                    payload: { lines: "174,178" },
                  },
                  {
                    content:
                      '<a href="https://arxiv.org/abs/2203.11364">An Information-theoretic Approach to Prompt Engineering Without Ground Truth Labels</a>',
                    children: [],
                    payload: { lines: "178,180" },
                  },
                ],
                payload: { lines: "173,174" },
              },
              {
                content: "Self-Consistency",
                children: [
                  {
                    content:
                      "It is based on the intuition that multiple different reasoning paths can lead to the same answer.",
                    children: [],
                    payload: { lines: "181,182" },
                  },
                  {
                    content:
                      "This method first prompts the LLM multiple times to perform CoT, crucially with a non-zero temperature to elicit diverse reasoning paths.",
                    children: [],
                    payload: { lines: "182,183" },
                  },
                  {
                    content:
                      "Next, it uses a majority vote over all generated responses to select a final response.",
                    children: [],
                    payload: { lines: "183,184" },
                  },
                  {
                    content:
                      "Self-Consistency has shown improvements on arithmetic, commonsense, and symbolic reasoning tasks.",
                    children: [],
                    payload: { lines: "184,185" },
                  },
                  {
                    content:
                      '<a href="https://arxiv.org/abs/2203.11171">Self-Consistency Improves Chain of Thought Reasoning in Language Models</a>',
                    children: [],
                    payload: { lines: "185,187" },
                  },
                ],
                payload: { lines: "180,181" },
              },
              {
                content: "Universal Self-Consistency",
                children: [
                  {
                    content:
                      "It is similar to Self-Consistency except that rather<br>\nthan selecting the majority response by programmatically counting how often it occurs, it inserts all outputs into a prompt template that selects the majority answer.",
                    children: [],
                    payload: { lines: "188,190" },
                  },
                  {
                    content:
                      "This is helpful for free-form text generation and cases where the same answer may be output slightly differently by different prompts.",
                    children: [],
                    payload: { lines: "190,191" },
                  },
                  {
                    content:
                      '<a href="https://arxiv.org/abs/2311.17311">Universal Self-Consistency for Large Language Model Generation</a>',
                    children: [],
                    payload: { lines: "191,193" },
                  },
                ],
                payload: { lines: "187,188" },
              },
              {
                content: "Meta-Reasoning over Multiple CoTs",
                children: [
                  {
                    content:
                      "It is similar to universal SelfConsistency; it first generates multiple reasoning chains (but not necessarily final answers) for a given problem.",
                    children: [],
                    payload: { lines: "194,195" },
                  },
                  {
                    content:
                      "Next, it inserts all of these chains in a single prompt template then generates a final answer from them.",
                    children: [],
                    payload: { lines: "195,196" },
                  },
                  {
                    content:
                      '<a href="https://arxiv.org/abs/2304.13007">Answering Questions by Meta-Reasoning over Multiple Chains of Thought</a>',
                    children: [],
                    payload: { lines: "196,198" },
                  },
                ],
                payload: { lines: "193,194" },
              },
              {
                content: "DiVeRSe",
                children: [
                  {
                    content:
                      "It creates multiple prompts for a given problem then performs SelfConsistency for each, generating multiple reasoning paths.",
                    children: [],
                    payload: { lines: "199,200" },
                  },
                  {
                    content:
                      "They score reasoning paths based on<br>\neach step in them then select a final response.",
                    children: [],
                    payload: { lines: "200,202" },
                  },
                  {
                    content:
                      '<a href="https://arxiv.org/abs/2206.02336">Making Large Language Models Better Reasoners with Step-Aware Verifier</a>',
                    children: [],
                    payload: { lines: "202,204" },
                  },
                ],
                payload: { lines: "198,199" },
              },
              {
                content: "Consistency-based Self-adaptive Prompting (COSP)",
                children: [
                  {
                    content:
                      "It constructs Few-Shot CoT prompts by running Zero-Shot CoT with Self-Consistency on a set of examples then selecting a high agreement subset of the outputs to be included in the final prompt as exemplars.",
                    children: [],
                    payload: { lines: "205,206" },
                  },
                  {
                    content:
                      "It again performs Self-Consistency with this final prompt.",
                    children: [],
                    payload: { lines: "206,207" },
                  },
                  {
                    content:
                      '<a href="https://arxiv.org/abs/2305.14106">Better Zero-Shot Reasoning with Self-Adaptive Prompting</a>',
                    children: [],
                    payload: { lines: "207,209" },
                  },
                ],
                payload: { lines: "204,205" },
              },
              {
                content: "Universal Self-Adaptive Prompting (USP)",
                children: [
                  {
                    content:
                      "It builds upon the success of COSP, aiming to make it generalizable to all tasks. USP makes use of unlabeled data to generate exemplars and a more complicated scoring function to select them.",
                    children: [],
                    payload: { lines: "210,211" },
                  },
                  {
                    content: "Additionally, USP does not use Self-Consistency.",
                    children: [],
                    payload: { lines: "211,212" },
                  },
                  {
                    content:
                      '<a href="https://arxiv.org/abs/2305.14926">Universal Self-Adaptive Prompting</a>',
                    children: [],
                    payload: { lines: "212,214" },
                  },
                ],
                payload: { lines: "209,210" },
              },
              {
                content: "Prompt Paraphrasing",
                children: [
                  {
                    content:
                      "It transforms an original prompt by changing some of the wording, while still maintaining the overall meaning.",
                    children: [],
                    payload: { lines: "215,216" },
                  },
                  {
                    content:
                      "It is effectively a data augmentation technique that can be used to generate prompts for an ensemble.",
                    children: [],
                    payload: { lines: "216,217" },
                  },
                  {
                    content:
                      '<a href="https://arxiv.org/abs/1911.12543">How Can We Know What Language Models Know?</a>',
                    children: [],
                    payload: { lines: "217,219" },
                  },
                ],
                payload: { lines: "214,215" },
              },
            ],
            payload: { lines: "162,163" },
          },
          {
            content: "Self-Criticism",
            children: [
              {
                content: "Self-Calibration",
                children: [
                  {
                    content: "It first prompts an LLM to answer a question.",
                    children: [],
                    payload: { lines: "222,223" },
                  },
                  {
                    content:
                      "Then, it builds a new prompt that includes the question, the LLM’s answer, and an additional instruction asking whether the answer is correct.",
                    children: [],
                    payload: { lines: "223,224" },
                  },
                  {
                    content:
                      "This can be useful for gauging confidence levels when applying LLMs when deciding when to accept or revise the original answer.",
                    children: [],
                    payload: { lines: "224,225" },
                  },
                  {
                    content:
                      '<a href="https://arxiv.org/abs/2207.05221">Language Models (Mostly) Know What They Know</a>',
                    children: [],
                    payload: { lines: "225,227" },
                  },
                ],
                payload: { lines: "221,222" },
              },
              {
                content: "Self-Refine",
                children: [
                  {
                    content:
                      "It is an iterative framework where, given an initial answer from the LLM, it prompts the same LLM to provide feedback on the answer, and then prompts the LLM to<br>\nimprove the answer based on the feedback.",
                    children: [],
                    payload: { lines: "228,230" },
                  },
                  {
                    content:
                      "This iterative process continues until a stopping condition is met (e.g., max number of steps reached).",
                    children: [],
                    payload: { lines: "230,231" },
                  },
                  {
                    content:
                      "Self-Refine has demonstrated improvement across a range of reasoning, coding, and generation tasks.",
                    children: [],
                    payload: { lines: "231,232" },
                  },
                  {
                    content:
                      '<a href="https://arxiv.org/abs/2303.17651">Self-Refine: Iterative Refinement with Self-Feedback</a>',
                    children: [],
                    payload: { lines: "232,234" },
                  },
                ],
                payload: { lines: "227,228" },
              },
              {
                content: "Reversing Chain-of-Thought (RCoT)",
                children: [
                  {
                    content:
                      "It first prompts LLMs to reconstruct the problem based on generated answer.",
                    children: [],
                    payload: { lines: "235,236" },
                  },
                  {
                    content:
                      "Then, it generates fine-grained comparisons between the original problem and the reconstructed problem as a way to check for any inconsistencies.",
                    children: [],
                    payload: { lines: "236,237" },
                  },
                  {
                    content:
                      "These inconsistencies are then converted to feedback for the LLM to revise the generated answer.",
                    children: [],
                    payload: { lines: "237,238" },
                  },
                  {
                    content:
                      '<a href="https://arxiv.org/abs/2305.11499">RCOT: Detecting and Rectifying Factual Inconsistency in Reasoning by Reversing Chain-of-Thought</a>',
                    children: [],
                    payload: { lines: "238,240" },
                  },
                ],
                payload: { lines: "234,235" },
              },
              {
                content: "Self-Verification",
                children: [
                  {
                    content:
                      "It generates multiple candidate solutions with Chain-ofThought (CoT).<br>\nIt then scores each solution by masking certain parts of the original question and asking an LLM to predict them based on the rest of the question and the generated solution.",
                    children: [],
                    payload: { lines: "241,243" },
                  },
                  {
                    content:
                      "This method has shown improvement on eight reasoning datasets.",
                    children: [],
                    payload: { lines: "243,244" },
                  },
                  {
                    content:
                      '<a href="https://arxiv.org/abs/2212.09561">Large Language Models are Better Reasoners with Self-Verification</a>',
                    children: [],
                    payload: { lines: "244,246" },
                  },
                ],
                payload: { lines: "240,241" },
              },
              {
                content: "Chain-of-Verification (COVE)",
                children: [
                  {
                    content:
                      "It first uses an LLM to generate an answer to a given question.",
                    children: [],
                    payload: { lines: "247,248" },
                  },
                  {
                    content:
                      "Then, it creates a list of related questions that would help verify the correctness of the answer.",
                    children: [],
                    payload: { lines: "248,249" },
                  },
                  {
                    content:
                      "Each question is answered by the LLM, then all the information is given to the LLM to produce the final revised answer.",
                    children: [],
                    payload: { lines: "249,250" },
                  },
                  {
                    content:
                      "This method has shown improvements in various question-answering and text-generation tasks.",
                    children: [],
                    payload: { lines: "250,251" },
                  },
                  {
                    content:
                      '<a href="https://arxiv.org/abs/2309.11495">Chain-of-Verification Reduces Hallucination in Large Language Models</a>',
                    children: [],
                    payload: { lines: "251,253" },
                  },
                ],
                payload: { lines: "246,247" },
              },
              {
                content: "Cumulative Reasoning",
                children: [
                  {
                    content:
                      "It first generates several potential steps in answering<br>\nthe question.",
                    children: [],
                    payload: { lines: "254,256" },
                  },
                  {
                    content:
                      "It then has a LLM evaluate them, deciding to either accept or reject these steps.",
                    children: [],
                    payload: { lines: "256,257" },
                  },
                  {
                    content:
                      "Finally, it checks whether it has arrived at the final answer.",
                    children: [],
                    payload: { lines: "257,258" },
                  },
                  {
                    content:
                      "If so, it terminates the process, but otherwise it repeats it.",
                    children: [],
                    payload: { lines: "258,259" },
                  },
                  {
                    content:
                      "This method has demonstrated improvements in logical inference tasks and mathematical problem.",
                    children: [],
                    payload: { lines: "259,260" },
                  },
                  {
                    content:
                      '<a href="https://arxiv.org/abs/2308.04371">Cumulative Reasoning with Large Language Models</a>',
                    children: [],
                    payload: { lines: "260,262" },
                  },
                ],
                payload: { lines: "253,254" },
              },
            ],
            payload: { lines: "219,220" },
          },
        ],
        payload: { lines: "3,4" },
      },
      {
        content: "Multilingual Prompting",
        children: [
          {
            content: "&nbsp;",
            children: [
              {
                content: "Translate First Prompting",
                children: [
                  {
                    content:
                      "It first translates non-English input examples into English. By translating the inputs into English, the model can utilize its strengths in English to better understand the content. Translation tools vary",
                    children: [],
                    payload: { lines: "267,268" },
                  },
                  {
                    content:
                      '<a href="https://arxiv.org/abs/2210.03057">Language Models are Multilingual Chain-of-Thought Reasoners</a> use an<br>\nexternal MT system.',
                    children: [],
                    payload: { lines: "268,270" },
                  },
                  {
                    content:
                      '<a href="https://arxiv.org/abs/2308.01223">Do Multilingual Language Models Think Better in English?</a>  prompt<br>\nmultilingual LMs.',
                    children: [],
                    payload: { lines: "270,272" },
                  },
                  {
                    content:
                      '<a href="https://arxiv.org/abs/2210.07313">Bootstrapping Multilingual Semantic Parsers using Large Language Models</a>  prompt LLMs to translate non-English inputs.',
                    children: [],
                    payload: { lines: "272,274" },
                  },
                ],
                payload: { lines: "266,267" },
              },
            ],
            payload: { lines: "264,265" },
          },
          {
            content: "Chain-of-Thought",
            children: [
              {
                content: "XLT (Cross-Lingual Thought) Prompting",
                children: [
                  {
                    content:
                      "It  utilizes a prompt template composed of six separate instructions, including role assignment, cross-lingual thinking, and CoT.",
                    children: [],
                    payload: { lines: "277,278" },
                  },
                  {
                    content:
                      '<a href="https://arxiv.org/abs/2305.07004">Not All Languages Are Created Equal in LLMs: Improving Multilingual Capability by Cross-Lingual-Thought Prompting</a>',
                    children: [],
                    payload: { lines: "278,280" },
                  },
                ],
                payload: { lines: "276,277" },
              },
              {
                content: "Cross-Lingual Self Consistent Prompting (CLSP)",
                children: [
                  {
                    content:
                      "It introduces an ensemble technique that constructs reasoning paths in different languages to answer the same question.",
                    children: [],
                    payload: { lines: "281,282" },
                  },
                  {
                    content:
                      '<a href="https://arxiv.org/abs/2310.14799">Cross-lingual Prompting: Improving Zero-shot Chain-of-Thought Reasoning across Languages</a>',
                    children: [],
                    payload: { lines: "282,284" },
                  },
                ],
                payload: { lines: "280,281" },
              },
            ],
            payload: { lines: "274,275" },
          },
          {
            content: "In-Context Learning",
            children: [
              {
                content: "X-InSTA Prompting",
                children: [
                  {
                    content:
                      "It explores three distinct approaches for aligning incontext examples with the input sentence for classification tasks: using semantically similar examples to the input (semantic alignment), examples that share the same label as the input (task-based alignment), and the combination of both semantic and task-based alignments.",
                    children: [],
                    payload: { lines: "287,288" },
                  },
                  {
                    content:
                      '<a href="https://arxiv.org/abs/2305.05940">Multilingual LLMs are Better Cross-lingual In-context Learners with Alignment</a>',
                    children: [],
                    payload: { lines: "288,290" },
                  },
                ],
                payload: { lines: "286,287" },
              },
              {
                content: "In-CLT (Cross-lingual Transfer) Prompting",
                children: [
                  {
                    content:
                      "It leverages both the source and target languages to create in-context examples, diverging from the traditional method of using source language exemplars. This strategy helps stimulate the cross-lingual cognitive capabilities of multilingual LLMs, thus boosting performance on crosslingual tasks.",
                    children: [],
                    payload: { lines: "291,292" },
                  },
                  {
                    content:
                      '<a href="https://arxiv.org/abs/2305.15233">Cross-lingual QA: A Key to Unlocking In-context Cross-lingual Performance</a>',
                    children: [],
                    payload: { lines: "292,294" },
                  },
                ],
                payload: { lines: "290,291" },
              },
            ],
            payload: { lines: "284,285" },
          },
          {
            content: "Prompting for Machine Translation",
            children: [
              {
                content: "Multi-Aspect Prompting and Selection (MAPS)",
                children: [
                  {
                    content:
                      "It mimics the human translation process, which involves multiple preparatory steps to ensure high-quality output.",
                    children: [],
                    payload: { lines: "297,298" },
                  },
                  {
                    content:
                      "This framework starts with knowledge mining from the source sentence (extracting keywords and topics, and generating translation exemplars).",
                    children: [],
                    payload: { lines: "298,299" },
                  },
                  {
                    content:
                      "It integrates this knowledge to generate multiple possible translations, then selects the best one.",
                    children: [],
                    payload: { lines: "299,300" },
                  },
                  {
                    content:
                      '<a href="https://arxiv.org/abs/2305.04118">Exploring Human-Like Translation Strategy with Large Language Models</a>',
                    children: [],
                    payload: { lines: "300,302" },
                  },
                ],
                payload: { lines: "296,297" },
              },
              {
                content: "Chain-of-Dictionary (CoD)",
                children: [
                  {
                    content:
                      "It first extracts words from the source phrase, then<br>\nmakes a list of their meanings in multiple languages, automatically via retrieval from a dictionary (e.g. English: ‘apple’, Spanish: ‘manzana’).",
                    children: [],
                    payload: { lines: "303,305" },
                  },
                  {
                    content:
                      "Then, they prepend these dictionary phrases to the prompt, where it asks a GenAI to use them during translation.",
                    children: [],
                    payload: { lines: "305,306" },
                  },
                  {
                    content:
                      '<a href="https://arxiv.org/abs/2305.06575">Chain-of-Dictionary Prompting Elicits Translation in Large Language Models</a>',
                    children: [],
                    payload: { lines: "306,308" },
                  },
                ],
                payload: { lines: "302,303" },
              },
              {
                content:
                  "Dictionary-based Prompting for Machine Translation (DiPMT)",
                children: [
                  {
                    content:
                      "It works similarly to CoD, but only gives definitions<br>\nin the source and target languages, and formats<br>\nthem slightly differently.",
                    children: [],
                    payload: { lines: "309,312" },
                  },
                  {
                    content:
                      '<a href="https://arxiv.org/abs/2302.07856">Dictionary-based Phrase-level Prompting of Large Language Models for Machine Translation</a>',
                    children: [],
                    payload: { lines: "312,314" },
                  },
                ],
                payload: { lines: "308,309" },
              },
              {
                content: "Decomposed Prompting for MT (DecoMT)",
                children: [
                  {
                    content:
                      "It divides the source text into several chunks and translates them independently using few-shot prompting.",
                    children: [],
                    payload: { lines: "315,316" },
                  },
                  {
                    content:
                      "Then it uses these translations and contextual information between chunks to generate a final translation.",
                    children: [],
                    payload: { lines: "316,317" },
                  },
                  {
                    content:
                      '<a href="https://arxiv.org/abs/2305.13085">Decomposed Prompting for Machine Translation Between Related Languages using Large Language Models</a>',
                    children: [],
                    payload: { lines: "317,319" },
                  },
                ],
                payload: { lines: "314,315" },
              },
            ],
            payload: { lines: "294,295" },
          },
        ],
        payload: { lines: "262,263" },
      },
      {
        content: "Agents",
        children: [
          {
            content: "Tool Use Agents",
            children: [
              {
                content:
                  "Modular Reasoning, Knowledge, and Language (MRKL) System",
                children: [
                  {
                    content:
                      "It contains a LLM router providing access to multiple tools.",
                    children: [],
                    payload: { lines: "324,325" },
                  },
                  {
                    content:
                      "The router can make multiple calls to get information such as weather or the current date.",
                    children: [],
                    payload: { lines: "325,326" },
                  },
                  {
                    content:
                      "It then combines this information to generate a final response.",
                    children: [],
                    payload: { lines: "326,327" },
                  },
                  {
                    content:
                      '<a href="https://arxiv.org/abs/2205.00445">MRKL Systems: A modular, neuro-symbolic architecture that combines large language models, external knowledge sources and discrete reasoning</a>',
                    children: [],
                    payload: { lines: "327,329" },
                  },
                ],
                payload: { lines: "323,324" },
              },
              {
                content:
                  "Self-Correcting with Tool-Interactive Critiquing (CRITIC)",
                children: [
                  {
                    content:
                      "It first generates a response to the prompt, with no external calls.",
                    children: [],
                    payload: { lines: "330,331" },
                  },
                  {
                    content:
                      "Then, the same LLM criticizes this response for possible errors.",
                    children: [],
                    payload: { lines: "331,332" },
                  },
                  {
                    content:
                      "Finally, it uses tools (e.g. Internet search or a code interpreter) accordingly to verify or amend parts of the response.",
                    children: [],
                    payload: { lines: "332,333" },
                  },
                  {
                    content:
                      '<a href="https://arxiv.org/abs/2305.11738">CRITIC: Large Language Models Can Self-Correct with Tool-Interactive Critiquing</a>',
                    children: [],
                    payload: { lines: "333,335" },
                  },
                ],
                payload: { lines: "329,330" },
              },
            ],
            payload: { lines: "321,322" },
          },
          {
            content: "Code-Generation Agents",
            children: [
              {
                content: "Program-aided Language Model (PAL)",
                children: [
                  {
                    content:
                      "It translates a problem directly into code, which is sent to a Python interpreter to generate an answer.",
                    children: [],
                    payload: { lines: "338,339" },
                  },
                  {
                    content:
                      '<a href="https://arxiv.org/abs/2211.10435">PAL: Program-aided Language Models</a>',
                    children: [],
                    payload: { lines: "339,341" },
                  },
                ],
                payload: { lines: "337,338" },
              },
              {
                content: "Tool-Integrated Reasoning Agent (ToRA)",
                children: [
                  {
                    content:
                      "It is similar to PAL, but instead of a single code generation step, it interleaves code and reasoning steps for as long as necessary to solve the problem.",
                    children: [],
                    payload: { lines: "342,343" },
                  },
                  {
                    content:
                      '<a href="https://arxiv.org/abs/2309.17452">ToRA: A Tool-Integrated Reasoning Agent for Mathematical Problem Solving</a>',
                    children: [],
                    payload: { lines: "343,345" },
                  },
                ],
                payload: { lines: "341,342" },
              },
              {
                content: "TaskWeaver",
                children: [
                  {
                    content:
                      "It is also similar to PAL, transforming user requests into code, but can also make use of user-defined plugin.",
                    children: [],
                    payload: { lines: "346,347" },
                  },
                  {
                    content:
                      '<a href="https://arxiv.org/abs/2311.17541">TaskWeaver: A Code-First Agent Framework</a>',
                    children: [],
                    payload: { lines: "347,349" },
                  },
                ],
                payload: { lines: "345,346" },
              },
            ],
            payload: { lines: "335,336" },
          },
          {
            content: "Observation-Based Agents",
            children: [
              {
                content: "Reasoning and Acting (ReAct)",
                children: [
                  {
                    content:
                      "It generates a thought, takes an action, and receives an observation (and repeats this process) when given a problem to solve.",
                    children: [],
                    payload: { lines: "352,353" },
                  },
                  {
                    content:
                      "All of this information is inserted into the prompt so it has a memory of past thoughts, actions, and observations.",
                    children: [],
                    payload: { lines: "353,354" },
                  },
                  {
                    content:
                      '<a href="https://arxiv.org/abs/2210.03629">ReAct: Synergizing Reasoning and Acting in Language Models</a>',
                    children: [],
                    payload: { lines: "354,356" },
                  },
                ],
                payload: { lines: "351,352" },
              },
              {
                content: "Reflexion",
                children: [
                  {
                    content:
                      "It  builds on ReAct, adding a layer of introspection.",
                    children: [],
                    payload: { lines: "357,358" },
                  },
                  {
                    content:
                      "It obtains a trajectory of actions and observations, then is given an evaluation of success/failure.",
                    children: [],
                    payload: { lines: "358,359" },
                  },
                  {
                    content:
                      "Then, it generates a reflection on what it did and what went wrong.",
                    children: [],
                    payload: { lines: "359,360" },
                  },
                  {
                    content:
                      "This reflection is added to its prompt as a working<br>\nmemory, and the process repeats.",
                    children: [],
                    payload: { lines: "360,362" },
                  },
                  {
                    content:
                      '<a href="https://arxiv.org/abs/2303.11366">Reflexion: Language Agents with Verbal Reinforcement Learning</a>',
                    children: [],
                    payload: { lines: "362,364" },
                  },
                ],
                payload: { lines: "356,357" },
              },
            ],
            payload: { lines: "349,350" },
          },
          {
            content: "Retrieval Augmented Generation",
            children: [
              {
                content: "Verify-and-Edit",
                children: [
                  {
                    content:
                      "It improves on self-consistency by generating multiple chains-ofthought, then selecting some to be edited. They do this by retrieving relevant (external) information to the CoTs, and allowing the LLM to augment them accordingly.",
                    children: [],
                    payload: { lines: "367,368" },
                  },
                  {
                    content:
                      '<a href="https://arxiv.org/abs/2305.03268">Verify-and-Edit: A Knowledge-Enhanced Chain-of-Thought Framework</a>',
                    children: [],
                    payload: { lines: "368,371" },
                  },
                ],
                payload: { lines: "366,367" },
              },
              {
                content: "Demonstrate-Search-Predict",
                children: [
                  {
                    content:
                      "It first decomposes a question into subquestions, then uses queries to solve them and combine their responses in a final answer.",
                    children: [],
                    payload: { lines: "372,373" },
                  },
                  {
                    content:
                      "It uses few-shot prompting to decompose the problem and combine responses.",
                    children: [],
                    payload: { lines: "373,374" },
                  },
                  {
                    content:
                      '<a href="https://arxiv.org/abs/2212.14024">Demonstrate-Search-Predict: Composing retrieval and language models for knowledge-intensive NLP</a>',
                    children: [],
                    payload: { lines: "374,376" },
                  },
                ],
                payload: { lines: "371,372" },
              },
              {
                content:
                  "Interleaved Retrieval guided by Chain-ofThought (IRCoT)",
                children: [
                  {
                    content:
                      "It is a technique for multi-hop question answering that<br>\ninterleaves CoT and retrieval.",
                    children: [],
                    payload: { lines: "377,379" },
                  },
                  {
                    content:
                      "IRCoT leverages CoT to guide which documents to retrieve and retrieval to help plan the reasoning steps of CoT.",
                    children: [],
                    payload: { lines: "379,380" },
                  },
                  {
                    content:
                      '<a href="https://arxiv.org/abs/2212.10509">Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions</a>',
                    children: [],
                    payload: { lines: "380,381" },
                  },
                ],
                payload: { lines: "376,377" },
              },
            ],
            payload: { lines: "364,365" },
          },
        ],
        payload: { lines: "319,320" },
      },
    ],
    payload: { lines: "1,2" },
  },
  { color: ["#2980b9"], initialExpandLevel: 3 }
);
