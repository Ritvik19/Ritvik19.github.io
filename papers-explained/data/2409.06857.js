((f, d, h, u) => {
  const g = f();
  window.mm = g.Markmap.create("svg#mindmap", (d || g.deriveOptions)(u), h);
})(
  () => window.markmap,
  null,
  {
    content:
      '<a href="https://www.arxiv.org/abs/2409.06857">Role of Small Models in the LLM Era</a>',
    children: [
      {
        content: "Collaboration",
        children: [
          {
            content: "LLMs Enhance SMs",
            children: [
              {
                content: "Knowledge Distillation",
                children: [
                  {
                    content:
                      "Black-Box Knowledge Distillation: involves generating a distillation dataset through the teacher LLM, which is then used for fine-tuning the student model.",
                    children: [],
                    payload: { lines: "8,9" },
                  },
                  {
                    content:
                      "White-Box Knowledge Distillation: using internal states of the teacher model in the training process of the student model.",
                    children: [],
                    payload: { lines: "9,11" },
                  },
                ],
                payload: { lines: "7,8" },
              },
              {
                content: "Data Synthesis",
                children: [
                  {
                    content:
                      "Generating a dataset from scratch using LLMs, in an unsupervised manner, followed by training a task-specific SM on the synthesized dataset.",
                    children: [],
                    payload: { lines: "12,13" },
                  },
                  {
                    content:
                      "Leveraging LLMs solely to generate labels rather than the entire training dataset.",
                    children: [],
                    payload: { lines: "13,14" },
                  },
                  {
                    content:
                      "Using LLMs to modify existing data points, thereby increasing data diversity, which can then be directly used to train smaller models, Eg: paraphrase or rewrite texts.",
                    children: [],
                    payload: { lines: "14,16" },
                  },
                ],
                payload: { lines: "11,12" },
              },
            ],
            payload: { lines: "5,6" },
          },
          {
            content: "SMs Enhance LLMs",
            children: [
              {
                content: "Data Curation",
                children: [
                  {
                    content:
                      "Small model can be trained specifically to evaluate text quality, enabling the selection of high-quality subsets.",
                    children: [],
                    payload: { lines: "19,20" },
                  },
                  {
                    content:
                      "Perplexity scores can be calculated by a SM to select data that is more likely to be of high quality",
                    children: [],
                    payload: { lines: "20,21" },
                  },
                  {
                    content:
                      "SMs can be used as classifiers to evaluate instruction data based on quality, coverage, and necessity.",
                    children: [],
                    payload: { lines: "21,23" },
                  },
                ],
                payload: { lines: "18,19" },
              },
              {
                content: "Weak-to-Strong Paradigm",
                children: [
                  {
                    content:
                      "LLMs can be fine-tuned on labels generated by a diverse set of specialized SMs, enabling the strong models to generalize beyond the limitations of their weaker supervisors.",
                    children: [],
                    payload: { lines: "24,25" },
                  },
                  {
                    content:
                      "SMs can also collaborate with LLMs during the inference phase to further enhance alignment.",
                    children: [],
                    payload: { lines: "25,27" },
                  },
                ],
                payload: { lines: "23,24" },
              },
              {
                content: "Efficient Inference",
                children: [
                  {
                    content: "Model Cascading",
                    children: [],
                    payload: { lines: "28,29" },
                  },
                  {
                    content: "Model Routing",
                    children: [],
                    payload: { lines: "29,30" },
                  },
                  {
                    content:
                      "Speculative Decoding: The auxiliary SM can quickly generates multiple token candidates in parallel, which are then validated or refined by the LLM.",
                    children: [],
                    payload: { lines: "30,32" },
                  },
                ],
                payload: { lines: "27,28" },
              },
              {
                content: "Evaluating LLMs",
                children: [
                  {
                    content:
                      "Model-based evaluation approaches can use smaller models to assess performance. Eg: BERT Score.",
                    children: [],
                    payload: { lines: "33,35" },
                  },
                ],
                payload: { lines: "32,33" },
              },
              {
                content: "Domain Adaptation",
                children: [
                  {
                    content:
                      "Black-Box Adaptation involves using a domain-specific SM to guide LLMs toward a target domain by providing textual relevant knowledge.",
                    children: [],
                    payload: { lines: "36,37" },
                  },
                  {
                    content:
                      "White-Box Adaptation typically involves fine- tuning a SM to adjust the token distributions of frozen LLMs for a specific target domain.",
                    children: [],
                    payload: { lines: "37,39" },
                  },
                ],
                payload: { lines: "35,36" },
              },
              {
                content: "Retrieval Augmented Generation",
                children: [
                  {
                    content:
                      "Retrievers based on SMs can be used for enhancing generations, Eg ColBERT",
                    children: [],
                    payload: { lines: "40,42" },
                  },
                ],
                payload: { lines: "39,40" },
              },
              {
                content: "Prompt-based Learning",
                children: [
                  {
                    content:
                      "SMs can be employed to enhance prompts, thereby improving the performance of larger models.",
                    children: [],
                    payload: { lines: "43,44" },
                  },
                  {
                    content:
                      "SMs can be used to verify or rewrite the outputs of LLMs, thereby achieving performance gains without the need for fine-tuning.",
                    children: [],
                    payload: { lines: "44,46" },
                  },
                ],
                payload: { lines: "42,43" },
              },
              {
                content: "Deficiency Repair",
                children: [
                  {
                    content:
                      "SMs can leverage contrastive decoding to reduce repetition, hallucinations in LLMs.",
                    children: [],
                    payload: { lines: "47,48" },
                  },
                  {
                    content:
                      "Specialized fine-tuned SM can be used to address some of the shortcomings of the larger model.",
                    children: [],
                    payload: { lines: "48,50" },
                  },
                ],
                payload: { lines: "46,47" },
              },
            ],
            payload: { lines: "16,17" },
          },
        ],
        payload: { lines: "3,4" },
      },
      {
        content: "Competition",
        children: [
          {
            content: ".",
            children: [
              {
                content: "Computation-constrained Environment",
                children: [
                  {
                    content:
                      "Small models are increasingly valuable in scenarios where computational resources are limited.",
                    children: [],
                    payload: { lines: "53,54" },
                  },
                ],
                payload: { lines: "52,53" },
              },
              {
                content: "Task-specific Environment",
                children: [
                  {
                    content:
                      "Small tree-based models can achieve competitive performance compared to large deep learning models for tabular data.",
                    children: [],
                    payload: { lines: "55,56" },
                  },
                  {
                    content:
                      "Fine-tuning SMs on domain-specific datasets can outperform general LLMs.",
                    children: [],
                    payload: { lines: "56,57" },
                  },
                  {
                    content:
                      "SMs can be particularly effective for tasks such as text classification, phrase representation, and entity retrieval.",
                    children: [],
                    payload: { lines: "57,58" },
                  },
                ],
                payload: { lines: "54,55" },
              },
              {
                content: "Interpretability-required Environment",
                children: [
                  {
                    content:
                      "Generally, smaller and simpler models offer better interpretability compared to larger, more complex models.",
                    children: [],
                    payload: { lines: "59,60" },
                  },
                ],
                payload: { lines: "58,59" },
              },
            ],
            payload: { lines: "51,52" },
          },
        ],
        payload: { lines: "50,51" },
      },
    ],
    payload: { lines: "1,2" },
  },
  { color: ["#2980b9"], initialExpandLevel: 4 }
);
