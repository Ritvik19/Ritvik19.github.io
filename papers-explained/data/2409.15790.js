document.getElementById("title").innerHTML = "Small Language Models: Survey, Measurements, and Insights";

((f, d, h, u) => {
  const g = f();
  window.mm = g.Markmap.create("svg#mindmap", (d || g.deriveOptions)(u), h);
})(
  () => window.markmap,
  null,
  {
    content:
      '<a href="https://arxiv.org/abs/2409.15790">Small Language Models: Survey, Measurements, and Insights</a>',
    children: [
      {
        content: "&nbsp;",
        children: [
          {
            content: "Model Architecture",
            children: [
              {
                content:
                  "As of August 2024, a typical SLM architecture tends to use group-query attention, gated FFN with SiLU activation, an intermediate ratio of FFN between 2 and 8, RMS normalization, and a vocabulary size larger than 50K. However, the choice of such settings is mostly empirical, without strict and public validation on the superiority of such model’s capacity. Instead, the architecture innovations have relative larger impacts on the runtime performance on devices.",
                children: [],
                payload: { lines: "5,6" },
              },
              {
                content:
                  "The innovations to the transformer architecture is limited in nowaday SLMs. For the few that did contribute architectural innovation (except embedding-lm head sharing), there is not strong evidence showing them being significantly superior to the vanilla transformer, and they are not generally adopted or studied across different research groups or companies. The significance of those innovations remain to be explored and validated.",
                children: [],
                payload: { lines: "6,8" },
              },
            ],
            payload: { lines: "4,5" },
          },
        ],
        payload: { lines: "3,4" },
      },
      {
        content: "&nbsp;",
        children: [
          {
            content: "Training Datasets",
            children: [
              {
                content:
                  'Data quality is crucial to SLM capability, which receives increasing attentions in recent SLM research. The importance of data quality to the final SLM capability typically outweighs the data quantity and model architecture configurations. A notable trend of dataset research is using model-based filtering, which result in two state-of-the-art open-sourced pre-training datasets: <a href="https://arxiv.org/abs/2406.17557">FineWeb-Edu</a> <a href="https://ritvik19.medium.com/papers-explained-174-fineweb-280bbc08068b">📑</a> (1.3T/5.4T) and <a href="https://arxiv.org/abs/2406.11794">DCLM-baseline</a> (4T). SLMs trained on these two datasets have achieved competitive performance to those trained on closed datasets, which have significantly advanced the SLM research reproducibility.',
                children: [],
                payload: { lines: "10,11" },
              },
              {
                content:
                  'Recent SLMs are trained over large amount of tokens (typically &gt;1.5T), disregarding their parameter sizes. In some cases, smaller SLMs are trained over even more data (e.g., <a href="https://arxiv.org/abs/2407.10671">Qwen2-0.5B</a> at 12T tokens but <a href="https://arxiv.org/abs/2407.10671">Qwen2-1.5B</a> at 7T tokens). It also means those SLMs are significantly “over-trained”, as compared to the Chinchilla law that estimates the parameter-token ratio to be around only 20 (e.g., 1B model with 20B tokens). The incentive of such “over-training” action is to deploy powerful SLMs on resource-constrained devices through investing more training-time compute resources.',
                children: [],
                payload: { lines: "11,13" },
              },
            ],
            payload: { lines: "9,10" },
          },
        ],
        payload: { lines: "8,9" },
      },
      {
        content: "&nbsp;",
        children: [
          {
            content: "Training Algorithms",
            children: [
              {
                content:
                  "Maximal Update Parameterization(µP) controls initialization, layer-wise learning rates, and activation magnitudes to ensure analytically stable training independent of a model’s layer widths. In addition to improving training stability, µP also improves the transferability of training hyperparameters from smaller to larger scale models, which permits directly using the same settings for some optimizer hyperparameters, most notably the learning rate. For example, Cerebras-GPT trains models with Maximal Update Parameterization.",
                children: [],
                payload: { lines: "15,16" },
              },
              {
                content:
                  'Knowledge Distillation is a crucial concept in the realm of Large Language Models (LLM). It involves extracting valuable knowledge from a large and complex teacher model and transferring it to a smaller and more efficient student model. The essence of this technique is to have the student model learn to approximate the behavior and predictions of the teacher. This is achieved by minimizing the difference between their outputs. <a href="https://arxiv.org/abs/2304.14402">LaMini-GPT</a> and <a href="https://arxiv.org/abs/2408.00118">Gemma-2</a> <a href="https://ritvik19.medium.com/papers-explained-157-gemma-2-f1b75b56b9f2">📑</a> adopt Knowledge Distillation.',
                children: [],
                payload: { lines: "16,17" },
              },
              {
                content:
                  'Two Stage Pre-training Strategy is a training strategy that involves training a model in two distinct phases. During the pretraining phase, <a href="https://arxiv.org/abs/2404.06395">MiniCPM</a> only uses large-scale coarse-quality pre-training data, which is abundant and can support continuous training when provided with more computational resources. During the annealing phase, we use diverse and high-quality knowledge and ability-oriented SFT data, mixed into the pre-training data. MninCPM adopts Two Stage Pre-training Strategy',
                children: [],
                payload: { lines: "17,19" },
              },
            ],
            payload: { lines: "14,15" },
          },
        ],
        payload: { lines: "13,14" },
      },
      {
        content: "SLM Capabilities",
        children: [
          {
            content: "Overall Capabilities",
            children: [
              {
                content:
                  'From 2022 to 2024, SLMs exhibited significant performance improvements across various language tasks, outpacing the improvements of the LLaMA-7B series (<a href="https://arxiv.org/abs/2302.13971">1</a> <a href="https://ritvik19.medium.com/papers-explained-55-llama-c4f302809d6b">📑</a> / <a href="https://arxiv.org/abs/2307.09288">2</a> <a href="https://ritvik19.medium.com/papers-explained-60-llama-v2-3e415c5b9b17">📑</a> / <a href="https://ai.meta.com/blog/meta-llama-3/">3</a> <a href="https://ritvik19.medium.com/papers-explained-187a-llama-3-51e2b90f63bb">📑</a> / <a href="https://ai.meta.com/research/publications/the-llama-3-herd-of-models/">3.1</a> <a href="https://ritvik19.medium.com/papers-explained-187b-llama-3-1-f0fb06898c59">📑</a> versions). This paints a promising picture for SLMs’ potential to solve a range of downstream tasks on devices.',
                children: [],
                payload: { lines: "21,22" },
              },
              {
                content:
                  'The Phi family (<a href="https://arxiv.org/abs/2306.11644">1</a> <a href="https://ritvik19.medium.com/papers-explained-114-phi-1-14a8dcc77ce5">📑</a> / <a href="https://arxiv.org/abs/2309.05463">1.5</a> <a href="https://ritvik19.medium.com/papers-explained-phi-1-5-2857e56dbd2a">📑</a> / <a href="https://www.microsoft.com/en-us/research/blog/phi-2-the-surprising-power-of-small-language-models/">2</a> <a href="https://ritvik19.medium.com/papers-explained-phi-1-5-2857e56dbd2a#8230">📑</a> / <a href="https://arxiv.org/abs/2404.14219">3</a> <a href="https://ritvik19.medium.com/papers-explained-130-phi-3-0dfc951dc404">📑</a> / <a href="https://techcommunity.microsoft.com/t5/ai-azure-ai-services-blog/discover-the-new-multi-lingual-high-quality-phi-3-5-slms/ba-p/4225280">3.5</a> <a href="https://ritvik19.medium.com/papers-explained-192-phi-3-5-a95429ea26c9">📑</a> ) consistently achieves state-of-the-art performance across most tasks. In particular, Phi-3.5-mini achieves the highest accuracy as of September 2024, rivaling LLaMA 3.1 8B. While much of its superior performance may be due to careful data engineering by the Microsoft team, part of it may also be attributed to instructive tuning and potential overfitting to specific datasets.',
                children: [],
                payload: { lines: "22,23" },
              },
              {
                content:
                  "Although larger parameter counts generally lead to better performance, exceptions such as Qwen 2 1.5B demonstrate that smaller models can still excel in specific tasks.",
                children: [],
                payload: { lines: "23,24" },
              },
              {
                content:
                  "SLMs trained on open-source datasets are closing the gap with their closed-source counterparts in commonsense tasks. However, the gap remains significant in tasks requiring complex reasoning or logic. This underscores the need for improved datasets focused on mathematical reasoning to address this disparity.",
                children: [],
                payload: { lines: "24,26" },
              },
            ],
            payload: { lines: "20,21" },
          },
          {
            content: "In-context Learning Capabilities",
            children: [
              {
                content:
                  "Generally, most SLMs encompass certain levels of in-context learning ability. However, such ability varies across different tasks: almost all SLMs benefit significantly from in-context learning in arc challenge task while certain tasks show mere benefit from in-context learning across all the models, such as hellaswag and piqa.",
                children: [],
                payload: { lines: "27,28" },
              },
              {
                content:
                  "Larger models tend to exhibit stronger in-context learning capabilities compared to their smaller counterparts. Some small SLMs even show a decrease in performance with in-context learning.",
                children: [],
                payload: { lines: "28,30" },
              },
            ],
            payload: { lines: "26,27" },
          },
        ],
        payload: { lines: "19,20" },
      },
      {
        content: "SLM Runtime Cost",
        children: [
          {
            content: "Memory Footprint",
            children: [
              {
                content:
                  "Apart from the model size, the model architecture also impacts latency. Factors such as the number of layers, the width of the FFN, the size of the vocabulary, and whether parameters are shared play significant roles. For example, Qwen1.5-0.5B has 25.4% more parameters than Qwen2-0.5B, but runs 31.9% faster on Jetson Orin NX 16GB. The correlation is likely hardware-dependent. This indicates that SLM development shall be aligned with the hardware where it will be deployed.",
                children: [],
                payload: { lines: "32,33" },
              },
              {
                content:
                  "The impacts of model architecture on inference speed is more significant at prefill stage than decode stage. This is because that the computational density in the prefill stage is higher, making it more likely to be compute-bound, while the decode stage is primarily memory-bound. Differences in model architecture can more easily affect the compute-bound scenarios; for example, wider and shallower models have higher computational parallelism.",
                children: [],
                payload: { lines: "33,34" },
              },
              {
                content:
                  "Runtime memory usage is generally linearly correlated with the model’s parameter count. A few models have larger memory usage compared to others with similar parameter counts, typically due to their larger vocabulary sizes. For instance, the Bloom series has a vocabulary size of 250,880, which is 5× to 8× larger than that of most models.",
                children: [],
                payload: { lines: "34,36" },
              },
            ],
            payload: { lines: "31,32" },
          },
          {
            content: "Impact of Quantization",
            children: [
              {
                content:
                  "The benefits of quantization during the decode stage are greater than those in the prefill stage. On mobile devices, quantization mainly reduces memory access overhead. Since the decode stage is more bandwidthbound, it gains more from quantization compared to the compute-bound prefill stage.",
                children: [],
                payload: { lines: "37,38" },
              },
              {
                content:
                  "More regular quantization precision leads to better performance. Although 3-bit quantization offers a higher model compression rate, 4-bit quantization performs better in both the prefill and decode stages. The inferior performance of 3-bit quantization is due to its irregular bit-width, which lacks hardware optimization support and incurs additional overhead from data alignment and padding. As a result, despite its lower compression rate, 4-bit quantization is more efficient. Similarly, irregular 5-bit and 6-bit quantization result in inference latency that is comparable to, or even higher than 8-bit quantization, despite offering higher compression rates.",
                children: [],
                payload: { lines: "38,40" },
              },
            ],
            payload: { lines: "36,37" },
          },
          {
            content: "Impact of Hardware",
            children: [
              {
                content:
                  "GPU shows an even greater advantage over the CPU during the prefill phase. The prefill phase involves parallel processing of tokens within the prompt, whereas the decode phase generates each token sequentially. Therefore, the prefill phase has a higher degree of parallelism, making it more suitable for GPUs, which have more parallel computing units.",
                children: [],
                payload: { lines: "41,42" },
              },
              {
                content:
                  "The Jetson demonstrates better performance stability compared to the smartphone. Due to its relatively simple hardware structure, which facilitates better heat dissipation, the Jetson maintains more stable latency during lengthy inference tasks.",
                children: [],
                payload: { lines: "42,44" },
              },
            ],
            payload: { lines: "40,41" },
          },
          {
            content: "Latency and Memory Breakdown",
            children: [
              {
                content:
                  "Matrix by vector multiplication is the most time-consuming operations of SLM, which constitute more than 70% end-to-end inference time.",
                children: [],
                payload: { lines: "45,46" },
              },
              {
                content:
                  "Context length is crucial for model runtime memory usage. When context length gets to 32,000, the KV cache will take up over 80% memory",
                children: [],
                payload: { lines: "46,47" },
              },
            ],
            payload: { lines: "44,45" },
          },
        ],
        payload: { lines: "30,31" },
      },
    ],
    payload: { lines: "1,2" },
  },
  { color: ["#2980b9"], maxWidth: 400, initialExpandLevel: 3 }
);
