document.getElementById("title").innerHTML = "Transformer Encoders";

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
    content: "Transformer Encoders",
    children: [
      {
        content: "&nbsp;",
        children: [
          {
            content:
              '<a href="https://arxiv.org/abs/1810.04805">BERT</a> <a href="https://ritvik19.medium.com/papers-explained-02-bert-31e59abc0615">📑</a>',
            children: [
              {
                content: "Purpose:",
                children: [
                  {
                    content:
                      "Pre-train deep bidirectional representations from unlabeled text for various NLP tasks.",
                    children: [],
                    payload: { lines: "6,7" },
                  },
                  {
                    content:
                      "Improve fine-tuning based approaches by alleviating the unidirectionality constraint.",
                    children: [],
                    payload: { lines: "7,8" },
                  },
                ],
                payload: { lines: "5,6" },
              },
              {
                content: "Method:",
                children: [
                  {
                    content:
                      "Masked Language Model (MLM): Randomly mask tokens and predict them based on context.",
                    children: [],
                    payload: { lines: "9,10" },
                  },
                  {
                    content:
                      "Next Sentence Prediction (NSP): Jointly pre-train text-pair representations.",
                    children: [],
                    payload: { lines: "10,11" },
                  },
                ],
                payload: { lines: "8,9" },
              },
              {
                content: "Advantages:",
                children: [
                  {
                    content: "State-of-the-art results on various NLP tasks.",
                    children: [],
                    payload: { lines: "12,13" },
                  },
                  {
                    content:
                      "Simple architecture and effective for both sentence-level and token-level tasks.",
                    children: [],
                    payload: { lines: "13,14" },
                  },
                  {
                    content:
                      "Fine-tuning allows adaptation to a wide range of tasks without significant modifications.",
                    children: [],
                    payload: { lines: "14,16" },
                  },
                ],
                payload: { lines: "11,12" },
              },
            ],
            payload: { lines: "4,5" },
          },
        ],
        payload: { lines: "3,4" },
      },
      {
        content: "Improving BERT",
        children: [
          {
            content:
              '<a href="https://arxiv.org/abs/1907.11692">RoBERTa</a> <a href="https://ritvik19.medium.com/papers-explained-03-roberta-81db014e35b9">📑</a>',
            children: [
              {
                content: "Purpose:",
                children: [
                  {
                    content:
                      "Improve BERT pre-training by optimizing training parameters and data.",
                    children: [],
                    payload: { lines: "20,21" },
                  },
                  {
                    content:
                      "Address the inconsistencies and missing details in the original BERT implementation.",
                    children: [],
                    payload: { lines: "21,22" },
                  },
                ],
                payload: { lines: "19,20" },
              },
              {
                content: "Method:",
                children: [
                  {
                    content:
                      "Larger batch sizes, longer training time, and more data.",
                    children: [],
                    payload: { lines: "23,24" },
                  },
                  {
                    content: "Dynamic masking instead of static masking.",
                    children: [],
                    payload: { lines: "24,25" },
                  },
                  {
                    content: "Removal of NSP task.",
                    children: [],
                    payload: { lines: "25,26" },
                  },
                ],
                payload: { lines: "22,23" },
              },
              {
                content: "Advantages:",
                children: [
                  {
                    content:
                      "Improved performance and robustness compared to BERT.",
                    children: [],
                    payload: { lines: "27,28" },
                  },
                  {
                    content: "More efficient pre-training process.",
                    children: [],
                    payload: { lines: "28,29" },
                  },
                  {
                    content: "Better generalization to downstream tasks.",
                    children: [],
                    payload: { lines: "29,31" },
                  },
                ],
                payload: { lines: "26,27" },
              },
            ],
            payload: { lines: "18,19" },
          },
          {
            content:
              '<a href="https://arxiv.org/abs/2006.03654">DeBERTa</a> <a href="https://ritvik19.medium.com/papers-explained-08-deberta-a808d9b2c52d">📑</a>',
            children: [
              {
                content: "Purpose:",
                children: [
                  {
                    content:
                      "Improve BERT and RoBERTa by enhancing attention mechanism and decoding.",
                    children: [],
                    payload: { lines: "33,34" },
                  },
                  {
                    content:
                      "Achieve better performance on both NLU and NLG tasks.",
                    children: [],
                    payload: { lines: "34,35" },
                  },
                ],
                payload: { lines: "32,33" },
              },
              {
                content: "Method:",
                children: [
                  {
                    content:
                      "Disentangled attention: Separate vectors for content and position, disentangled attention matrices.",
                    children: [],
                    payload: { lines: "36,37" },
                  },
                  {
                    content:
                      "Enhanced mask decoder: Incorporate absolute position embeddings in the decoding layer.",
                    children: [],
                    payload: { lines: "37,38" },
                  },
                  {
                    content: "Virtual adversarial training for fine-tuning.",
                    children: [],
                    payload: { lines: "38,39" },
                  },
                ],
                payload: { lines: "35,36" },
              },
              {
                content: "Advantages:",
                children: [
                  {
                    content:
                      "Significant performance gains on a wide range of NLP tasks.",
                    children: [],
                    payload: { lines: "40,41" },
                  },
                  {
                    content: "Improved training efficiency.",
                    children: [],
                    payload: { lines: "41,42" },
                  },
                  {
                    content: "Better generalization and robustness.",
                    children: [],
                    payload: { lines: "42,44" },
                  },
                ],
                payload: { lines: "39,40" },
              },
            ],
            payload: { lines: "31,32" },
          },
        ],
        payload: { lines: "16,17" },
      },
      {
        content: "Compressing BERT",
        children: [
          {
            content:
              '<a href="https://arxiv.org/abs/1909.10351">TinyBERT</a> <a href="https://ritvik19.medium.com/papers-explained-05-tiny-bert-5e36fe0ee173">📑</a>',
            children: [
              {
                content: "Purpose:",
                children: [
                  {
                    content:
                      "Compress BERT to a smaller size and accelerate inference for resource-constrained devices.",
                    children: [],
                    payload: { lines: "48,49" },
                  },
                  {
                    content:
                      "Maintain accuracy while reducing model complexity.",
                    children: [],
                    payload: { lines: "49,50" },
                  },
                ],
                payload: { lines: "47,48" },
              },
              {
                content: "Method:",
                children: [
                  {
                    content: "Knowledge distillation from a larger BERT model.",
                    children: [],
                    payload: { lines: "51,52" },
                  },
                  {
                    content:
                      "Two-stage learning: general distillation on a large corpus and task-specific distillation.",
                    children: [],
                    payload: { lines: "52,53" },
                  },
                  {
                    content:
                      "Transformer distillation with multiple loss functions to fit different representations.",
                    children: [],
                    payload: { lines: "53,54" },
                  },
                ],
                payload: { lines: "50,51" },
              },
              {
                content: "Advantages:",
                children: [
                  {
                    content:
                      "Significant reduction in size and inference time compared to BERT.",
                    children: [],
                    payload: { lines: "55,56" },
                  },
                  {
                    content:
                      "Retains a high percentage of the teacher BERT's performance.",
                    children: [],
                    payload: { lines: "56,57" },
                  },
                  {
                    content:
                      "Flexible model configuration due to the two-stage distillation framework.",
                    children: [],
                    payload: { lines: "57,59" },
                  },
                ],
                payload: { lines: "54,55" },
              },
            ],
            payload: { lines: "46,47" },
          },
          {
            content:
              '<a href="https://arxiv.org/abs/1909.11942">ALBERT</a> <a href="https://ritvik19.medium.com/papers-explained-07-albert-46a2a0563693">📑</a>',
            children: [
              {
                content: "Purpose:",
                children: [
                  {
                    content:
                      "Reduce BERT's memory consumption and training time for larger models.",
                    children: [],
                    payload: { lines: "61,62" },
                  },
                  {
                    content: "Improve parameter efficiency and scalability.",
                    children: [],
                    payload: { lines: "62,63" },
                  },
                ],
                payload: { lines: "60,61" },
              },
              {
                content: "Method:",
                children: [
                  {
                    content:
                      "Factorized embedding parameterization to decouple embedding size from hidden size.",
                    children: [],
                    payload: { lines: "64,65" },
                  },
                  {
                    content:
                      "Cross-layer parameter sharing to reduce the number of parameters.",
                    children: [],
                    payload: { lines: "65,66" },
                  },
                  {
                    content:
                      "Sentence order prediction (SOP) as a self-supervised loss to improve inter-sentence coherence.",
                    children: [],
                    payload: { lines: "66,67" },
                  },
                ],
                payload: { lines: "63,64" },
              },
              {
                content: "Advantages:",
                children: [
                  {
                    content:
                      "Significant reduction in parameters and memory footprint.",
                    children: [],
                    payload: { lines: "68,69" },
                  },
                  {
                    content: "Faster training speed.",
                    children: [],
                    payload: { lines: "69,70" },
                  },
                  {
                    content:
                      "Better performance on downstream tasks, especially those with multi-sentence inputs.",
                    children: [],
                    payload: { lines: "70,72" },
                  },
                ],
                payload: { lines: "67,68" },
              },
            ],
            payload: { lines: "59,60" },
          },
          {
            content:
              '<a href="https://arxiv.org/abs/2004.02178">FastBERT</a> <a href="https://ritvik19.medium.com/papers-explained-37-fastbert-5bd246c1b432">📑</a>',
            children: [
              {
                content: "Purpose:",
                children: [
                  {
                    content:
                      "Accelerate BERT inference by adapting the inference time based on input complexity.",
                    children: [],
                    payload: { lines: "74,75" },
                  },
                  {
                    content:
                      "Maintain accuracy while reducing computational cost.",
                    children: [],
                    payload: { lines: "75,76" },
                  },
                ],
                payload: { lines: "73,74" },
              },
              {
                content: "Method:",
                children: [
                  {
                    content:
                      "Self-distillation with a student-teacher framework.",
                    children: [],
                    payload: { lines: "77,78" },
                  },
                  {
                    content:
                      "Adaptive inference with early exit based on uncertainty estimation.",
                    children: [],
                    payload: { lines: "78,79" },
                  },
                  {
                    content: "Training with both original and distilled data.",
                    children: [],
                    payload: { lines: "79,80" },
                  },
                ],
                payload: { lines: "76,77" },
              },
              {
                content: "Advantages:",
                children: [
                  {
                    content:
                      "Significant speedup in inference time compared to BERT.",
                    children: [],
                    payload: { lines: "81,82" },
                  },
                  {
                    content: "Minimal accuracy loss.",
                    children: [],
                    payload: { lines: "82,83" },
                  },
                  {
                    content:
                      "Suitable for industrial scenarios with varying input complexity.",
                    children: [],
                    payload: { lines: "83,85" },
                  },
                ],
                payload: { lines: "80,81" },
              },
            ],
            payload: { lines: "72,73" },
          },
          {
            content:
              '<a href="https://arxiv.org/abs/2004.02984">MobileBERT</a> <a href="https://ritvik19.medium.com/papers-explained-36-mobilebert-933abbd5aaf1">📑</a>',
            children: [
              {
                content: "Purpose:",
                children: [
                  {
                    content:
                      "Compress and accelerate BERT for deployment on resource-limited mobile devices.",
                    children: [],
                    payload: { lines: "87,88" },
                  },
                  {
                    content:
                      "Maintain task-agnostic capabilities for fine-tuning on various NLP tasks.",
                    children: [],
                    payload: { lines: "88,89" },
                  },
                ],
                payload: { lines: "86,87" },
              },
              {
                content: "Method:",
                children: [
                  {
                    content:
                      "Bottleneck structure with reduced width but similar depth to BERTLARGE.",
                    children: [],
                    payload: { lines: "90,91" },
                  },
                  {
                    content:
                      "Knowledge transfer from a specially designed teacher model (IB-BERT).",
                    children: [],
                    payload: { lines: "91,92" },
                  },
                  {
                    content:
                      "Feature map transfer and attention transfer for layer-wise knowledge distillation.",
                    children: [],
                    payload: { lines: "92,93" },
                  },
                ],
                payload: { lines: "89,90" },
              },
              {
                content: "Advantages:",
                children: [
                  {
                    content: "4.3x smaller and 5.5x faster than BERTBASE.",
                    children: [],
                    payload: { lines: "94,95" },
                  },
                  {
                    content:
                      "Competitive performance on GLUE and SQuAD benchmarks.",
                    children: [],
                    payload: { lines: "95,96" },
                  },
                  {
                    content: "Low latency on mobile devices.",
                    children: [],
                    payload: { lines: "96,98" },
                  },
                ],
                payload: { lines: "93,94" },
              },
            ],
            payload: { lines: "85,86" },
          },
        ],
        payload: { lines: "44,45" },
      },
      {
        content: "Distilled Versions of BERT/RoBERTa",
        children: [
          {
            content:
              '<a href="https://arxiv.org/abs/1910.01108">DistilBERT</a> <a href="https://ritvik19.medium.com/papers-explained-06-distil-bert-6f138849f871">📑</a>',
            children: [
              {
                content: "Purpose:",
                children: [
                  {
                    content:
                      "Create a smaller and faster version of BERT while retaining most of its capabilities.",
                    children: [],
                    payload: { lines: "102,103" },
                  },
                  {
                    content:
                      "Enable efficient operation on edge devices and under resource constraints.",
                    children: [],
                    payload: { lines: "103,104" },
                  },
                ],
                payload: { lines: "101,102" },
              },
              {
                content: "Method:",
                children: [
                  {
                    content:
                      "Knowledge distillation from a larger BERT model during pre-training.",
                    children: [],
                    payload: { lines: "105,106" },
                  },
                  {
                    content:
                      "Triple loss combining language modeling, distillation, and cosine-distance losses.",
                    children: [],
                    payload: { lines: "106,107" },
                  },
                  {
                    content: "Removal of token-type embeddings and pooler.",
                    children: [],
                    payload: { lines: "107,108" },
                  },
                ],
                payload: { lines: "104,105" },
              },
              {
                content: "Advantages:",
                children: [
                  {
                    content:
                      "40% smaller and 60% faster than BERT while maintaining 97% of its performance.",
                    children: [],
                    payload: { lines: "109,110" },
                  },
                  {
                    content:
                      "Cheaper to pre-train and suitable for on-device computations.",
                    children: [],
                    payload: { lines: "110,111" },
                  },
                  {
                    content:
                      "Good performance on various tasks, including classification and question answering.",
                    children: [],
                    payload: { lines: "111,113" },
                  },
                ],
                payload: { lines: "108,109" },
              },
            ],
            payload: { lines: "100,101" },
          },
          {
            content:
              '<a href="https://huggingface.co/distilbert/distilroberta-base">DistilRoBERTa</a> <a href="https://medium.com/dair-ai/papers-explained-06-distil-bert-6f138849f871#a260">📑</a>',
            children: [
              {
                content: "Purpose:",
                children: [
                  {
                    content:
                      "Create a distilled version of RoBERTa with reduced size and faster inference.",
                    children: [],
                    payload: { lines: "115,116" },
                  },
                  {
                    content:
                      "Maintain the benefits of RoBERTa while improving efficiency.",
                    children: [],
                    payload: { lines: "116,117" },
                  },
                ],
                payload: { lines: "114,115" },
              },
              {
                content: "Method:",
                children: [
                  {
                    content:
                      "Knowledge distillation from RoBERTa-base using a similar procedure to DistilBERT.",
                    children: [],
                    payload: { lines: "118,119" },
                  },
                  {
                    content:
                      "Reduced number of layers compared to the teacher model.",
                    children: [],
                    payload: { lines: "119,120" },
                  },
                ],
                payload: { lines: "117,118" },
              },
              {
                content: "Advantages:",
                children: [
                  {
                    content:
                      "82M parameters compared to 125M parameters for RoBERTa-base.",
                    children: [],
                    payload: { lines: "121,122" },
                  },
                  {
                    content: "Twice as fast as RoBERTa-base on average.",
                    children: [],
                    payload: { lines: "122,123" },
                  },
                  {
                    content:
                      "Good performance on GLUE tasks after fine-tuning.",
                    children: [],
                    payload: { lines: "123,125" },
                  },
                ],
                payload: { lines: "120,121" },
              },
            ],
            payload: { lines: "113,114" },
          },
        ],
        payload: { lines: "98,99" },
      },
      {
        content: "Improving DeBERTa",
        children: [
          {
            content:
              '<a href="https://huggingface.co/docs/transformers/en/model_doc/deberta-v2">DeBERTa v2</a> <a href="https://ritvik19.medium.com/papers-explained-08-deberta-a808d9b2c52d#f5e1">📑</a>',
            children: [
              {
                content: "Purpose:",
                children: [
                  {
                    content:
                      "Build upon DeBERTa to further improve performance and efficiency.",
                    children: [],
                    payload: { lines: "129,130" },
                  },
                ],
                payload: { lines: "128,129" },
              },
              {
                content: "Method:",
                children: [
                  {
                    content:
                      "Introduces several optimization techniques, including improved parameter initialization and training procedures.",
                    children: [],
                    payload: { lines: "131,132" },
                  },
                  {
                    content:
                      "The sources do not provide specific details on the new methods employed by DeBERTa V2.",
                    children: [],
                    payload: { lines: "132,133" },
                  },
                ],
                payload: { lines: "130,131" },
              },
              {
                content: "Advantages:",
                children: [
                  {
                    content: "Outperforms DeBERTa on various NLP tasks.",
                    children: [],
                    payload: { lines: "134,135" },
                  },
                  {
                    content: "More efficient training with less data.",
                    children: [],
                    payload: { lines: "135,136" },
                  },
                  {
                    content:
                      "The sources do not explicitly list the advantages of DeBERTa V2.",
                    children: [],
                    payload: { lines: "136,138" },
                  },
                ],
                payload: { lines: "133,134" },
              },
            ],
            payload: { lines: "127,128" },
          },
          {
            content:
              '<a href="https://arxiv.org/abs/2111.09543">DeBERTa v3</a> <a href="https://ritvik19.medium.com/papers-explained-182-deberta-v3-65347208ce03">📑</a>',
            children: [
              {
                content: "Purpose:",
                children: [
                  {
                    content:
                      "Enhance DeBERTa with ELECTRA-style pre-training and gradient-disentangled embedding sharing.",
                    children: [],
                    payload: { lines: "140,141" },
                  },
                  {
                    content:
                      "Further improve efficiency and performance, particularly for smaller models.",
                    children: [],
                    payload: { lines: "141,142" },
                  },
                ],
                payload: { lines: "139,140" },
              },
              {
                content: "Method:",
                children: [
                  {
                    content:
                      "Replaced masked language modeling (MLM) with replaced token detection (RTD).",
                    children: [],
                    payload: { lines: "143,144" },
                  },
                  {
                    content:
                      "Gradient-disentangled embedding sharing (GDES) for more efficient knowledge transfer.",
                    children: [],
                    payload: { lines: "144,145" },
                  },
                  {
                    content:
                      "Combines RTD, GDES, and the disentangled attention mechanism from DeBERTa.",
                    children: [],
                    payload: { lines: "145,146" },
                  },
                ],
                payload: { lines: "142,143" },
              },
              {
                content: "Advantages:",
                children: [
                  {
                    content:
                      "Improved training efficiency and performance compared to DeBERTa.",
                    children: [],
                    payload: { lines: "147,148" },
                  },
                  {
                    content:
                      "Especially effective for smaller models, achieving significant gains with reduced parameters.",
                    children: [],
                    payload: { lines: "148,149" },
                  },
                  {
                    content:
                      "Demonstrates strong performance even with less data compared to RoBERTa and XLNet.",
                    children: [],
                    payload: { lines: "149,151" },
                  },
                ],
                payload: { lines: "146,147" },
              },
            ],
            payload: { lines: "138,139" },
          },
        ],
        payload: { lines: "125,126" },
      },
      {
        content: "Sentence Embeddings",
        children: [
          {
            content:
              '<a href="https://arxiv.org/abs/1908.10084">Sentence BERT</a> <a href="https://ritvik19.medium.com/papers-explained-04-sentence-bert-5159b8e07f21">📑</a>',
            children: [
              {
                content: "Purpose:",
                children: [
                  {
                    content:
                      "Generate semantically meaningful sentence embeddings for tasks like similarity comparison.",
                    children: [],
                    payload: { lines: "155,156" },
                  },
                  {
                    content:
                      "Improve efficiency compared to using BERT for pairwise sentence scoring.",
                    children: [],
                    payload: { lines: "156,157" },
                  },
                ],
                payload: { lines: "154,155" },
              },
              {
                content: "Method:",
                children: [
                  {
                    content:
                      "Siamese BERT network architecture to encode two sentences simultaneously.",
                    children: [],
                    payload: { lines: "158,159" },
                  },
                  {
                    content:
                      "Fine-tuning on various objective functions, including classification and regression.",
                    children: [],
                    payload: { lines: "159,160" },
                  },
                  {
                    content:
                      "Different pooling strategies to derive sentence embeddings from BERT outputs.",
                    children: [],
                    payload: { lines: "160,161" },
                  },
                ],
                payload: { lines: "157,158" },
              },
              {
                content: "Advantages:",
                children: [
                  {
                    content:
                      "State-of-the-art performance on sentence similarity and other tasks.",
                    children: [],
                    payload: { lines: "162,163" },
                  },
                  {
                    content:
                      "Significantly faster inference compared to pairwise sentence scoring.",
                    children: [],
                    payload: { lines: "163,164" },
                  },
                  {
                    content: "Adaptable to specific tasks through fine-tuning.",
                    children: [],
                    payload: { lines: "164,166" },
                  },
                ],
                payload: { lines: "161,162" },
              },
            ],
            payload: { lines: "153,154" },
          },
          {
            content:
              '<a href="https://arxiv.org/abs/2004.12832">ColBERT</a> <a href="https://medium.com/@ritvik19/papers-explained-88-colbert-fe2fd0509649">📑</a>',
            children: [
              {
                content: "Purpose:",
                children: [
                  {
                    content:
                      "Achieve efficient and effective passage retrieval using contextualized late interaction over BERT.",
                    children: [],
                    payload: { lines: "168,169" },
                  },
                  {
                    content:
                      "Improve the efficiency of BERT-based ranking models while maintaining high accuracy.",
                    children: [],
                    payload: { lines: "169,170" },
                  },
                ],
                payload: { lines: "167,168" },
              },
              {
                content: "Method:",
                children: [
                  {
                    content:
                      "Generates token-level embeddings for both queries and documents.",
                    children: [],
                    payload: { lines: "171,172" },
                  },
                  {
                    content:
                      "Late interaction between query and document embeddings using a MaxSim operator.",
                    children: [],
                    payload: { lines: "172,173" },
                  },
                  {
                    content:
                      "Efficient scoring and ranking based on the interaction matrix.",
                    children: [],
                    payload: { lines: "173,174" },
                  },
                ],
                payload: { lines: "170,171" },
              },
              {
                content: "Advantages:",
                children: [
                  {
                    content:
                      "Significantly faster than traditional BERT-based re-ranking models.",
                    children: [],
                    payload: { lines: "175,176" },
                  },
                  {
                    content:
                      "Comparable or better retrieval effectiveness compared to more complex models.",
                    children: [],
                    payload: { lines: "176,177" },
                  },
                  {
                    content:
                      "Suitable for large-scale retrieval tasks with high efficiency requirements.",
                    children: [],
                    payload: { lines: "177,179" },
                  },
                ],
                payload: { lines: "174,175" },
              },
            ],
            payload: { lines: "166,167" },
          },
          {
            content:
              '<a href="https://arxiv.org/abs/2112.01488">ColBERT v2</a> <a href="https://ritvik19.medium.com/papers-explained-89-colbertv2-7d921ee6e0d9">📑</a>',
            children: [
              {
                content: "Purpose",
                children: [
                  {
                    content:
                      "Improve Efficiency and Effectiveness of Late Interaction Retrieval",
                    children: [],
                    payload: { lines: "181,182" },
                  },
                ],
                payload: { lines: "180,181" },
              },
              {
                content: "Method",
                children: [
                  {
                    content:
                      "Denoised Supervision: distillation from a cross-encoder (MiniLM) and hard negative mining, to improve the model's robustness and accuracy by providing more informative training signals.",
                    children: [],
                    payload: { lines: "183,184" },
                  },
                  {
                    content:
                      "Residual Compression: leverages the observation that ColBERT token embeddings tend to cluster around specific semantic centroids.",
                    children: [],
                    payload: { lines: "184,185" },
                  },
                  {
                    content:
                      "It compresses embeddings by storing the nearest centroid ID and a quantized residual vector, drastically reducing storage requirements.",
                    children: [],
                    payload: { lines: "185,186" },
                  },
                ],
                payload: { lines: "182,183" },
              },
              {
                content: "Advantages",
                children: [
                  {
                    content:
                      "Achieves state-of-the-art results on various in-domain and out-of-domain benchmarks.",
                    children: [],
                    payload: { lines: "187,188" },
                  },
                  {
                    content:
                      "Residual compression achieves a 6-10x reduction in storage size compared to the original ColBERT.",
                    children: [],
                    payload: { lines: "188,189" },
                  },
                ],
                payload: { lines: "186,187" },
              },
            ],
            payload: { lines: "179,180" },
          },
        ],
        payload: { lines: "151,152" },
      },
    ],
    payload: { lines: "1,2" },
  },
  { color: ["#2980b9"], maxWidth: 400, initialExpandLevel: 3 }
);
