document.getElementById("title").innerHTML = "Vision Transformers";

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
    content: "Vision Transformers",
    children: [
      {
        content: "&nbsp;",
        children: [
          {
            content:
              '<a href="https://arxiv.org/abs/2010.11929">Vision Transformer (ViT)</a> <a href="https://ritvik19.medium.com/papers-explained-25-vision-transformers-e286ee8bc06b">📑</a>',
            children: [
              {
                content: "Purpose",
                children: [
                  {
                    content:
                      "Apply the Transformer architecture directly to images for image classification.",
                    children: [],
                    payload: { lines: "6,7" },
                  },
                ],
                payload: { lines: "5,6" },
              },
              {
                content: "Method",
                children: [
                  {
                    content: "Splits an image into fixed-size patches.",
                    children: [],
                    payload: { lines: "8,9" },
                  },
                  {
                    content:
                      "Linearly embeds each patch and adds position embeddings.",
                    children: [],
                    payload: { lines: "9,10" },
                  },
                  {
                    content:
                      "Feeds the sequence of embedded patches into a standard Transformer encoder.",
                    children: [],
                    payload: { lines: "10,11" },
                  },
                  {
                    content:
                      'Uses an extra learnable "classification token" for classification.',
                    children: [],
                    payload: { lines: "11,12" },
                  },
                ],
                payload: { lines: "7,8" },
              },
              {
                content: "Advantages",
                children: [
                  {
                    content:
                      "Achieves excellent results on image classification tasks, comparable to state-of-the-art convolutional networks, while requiring fewer computational resources.",
                    children: [],
                    payload: { lines: "13,14" },
                  },
                  {
                    content:
                      "Benefits from the scalability and efficiency of NLP Transformer architectures and their implementations.",
                    children: [],
                    payload: { lines: "14,15" },
                  },
                  {
                    content:
                      "Self-attention allows for global information integration from the lowest layers, increasing attention distance with network depth.",
                    children: [],
                    payload: { lines: "15,17" },
                  },
                ],
                payload: { lines: "12,13" },
              },
            ],
            payload: { lines: "4,5" },
          },
        ],
        payload: { lines: "3,4" },
      },
      {
        content: "Studies Related to Vision Transformer",
        children: [
          {
            content:
              '<a href="https://arxiv.org/abs/2212.06727">What do Vision Transformers Learn? A Visual Exploration</a> <a href="">📑</a>',
            children: [
              {
                content: "Purpose",
                children: [
                  {
                    content:
                      "<strong>Understanding ViT Mechanisms:</strong> Investigate the internal workings of Vision Transformers (ViTs) and the features they learn, as limited understanding exists compared to Convolutional Neural Networks (CNNs).",
                    children: [],
                    payload: { lines: "22,23" },
                  },
                  {
                    content:
                      "<strong>Visualize ViT Features:</strong> Overcome the challenges of visualizing ViT representations and establish effective methods for visual exploration.",
                    children: [],
                    payload: { lines: "23,24" },
                  },
                  {
                    content:
                      "<strong>Compare ViTs and CNNs:</strong> Explore the similarities and differences in behavior between ViTs and CNNs, particularly regarding feature extraction and reliance on image components.",
                    children: [],
                    payload: { lines: "24,25" },
                  },
                  {
                    content:
                      "<strong>Analyze Impact of Language Supervision:</strong> Study the effect of training ViTs with language model supervision, such as CLIP, on the types of features they extract.",
                    children: [],
                    payload: { lines: "25,27" },
                  },
                ],
                payload: { lines: "21,22" },
              },
              {
                content: "Method",
                children: [
                  {
                    content: "<strong>Visualization Framework:</strong>",
                    children: [
                      {
                        content:
                          "<strong>Activation Maximization:</strong> Apply gradient descent to synthesize images that maximally activate specific neurons, similar to techniques used for CNNs.",
                        children: [],
                        payload: { lines: "29,30" },
                      },
                      {
                        content:
                          "<strong>Improved Visualization Techniques:</strong> Incorporate methods like total variation regularization, jitter augmentation, color shift augmentation, augmentation ensembling, and Gaussian smoothing to enhance image quality.",
                        children: [],
                        payload: { lines: "30,31" },
                      },
                      {
                        content:
                          "<strong>Pairing with ImageNet Examples:</strong> Supplement visualizations with images from the ImageNet dataset that strongly activate the corresponding feature, providing context and interpretability.",
                        children: [],
                        payload: { lines: "31,32" },
                      },
                      {
                        content:
                          "<strong>Patch-wise Activation Maps:</strong> Generate activation maps by passing images through the network, revealing how different patches contribute to feature activation and highlighting spatial information preservation.",
                        children: [],
                        payload: { lines: "32,33" },
                      },
                    ],
                    payload: { lines: "28,33" },
                  },
                  {
                    content: "<strong>Model Selection:</strong>",
                    children: [
                      {
                        content:
                          "<strong>ViT-B16 as Primary Model:</strong> Focus on ViT-B16, a widely used variant, for detailed demonstrations and analysis.",
                        children: [],
                        payload: { lines: "34,35" },
                      },
                      {
                        content:
                          "<strong>Extensive Visualization of Other Variants:</strong> Validate the visualization method on a wide range of ViT architectures, including DeiT, CoaT, ConViT, PiT, Swin, and Twin, to ensure generalizability.",
                        children: [],
                        payload: { lines: "35,36" },
                      },
                    ],
                    payload: { lines: "33,36" },
                  },
                  {
                    content: "<strong>Targeted Analysis:</strong>",
                    children: [
                      {
                        content:
                          "<strong>Spatial Information Preservation:</strong> Investigate how patch tokens maintain spatial information throughout the network layers, examining activation patterns and comparing with CNN behavior.",
                        children: [],
                        payload: { lines: "37,38" },
                      },
                      {
                        content:
                          "<strong>Last-Layer Token Mixing:</strong> Analyze the role of the last attention block in globalizing information and its similarity to average pooling in CNNs.",
                        children: [],
                        payload: { lines: "38,39" },
                      },
                      {
                        content:
                          "<strong>Background and Foreground Feature Reliance:</strong>  Compare ViT and CNN performance when selectively masking foreground or background information in images, using ImageNet bounding boxes to assess dependence on different image components.",
                        children: [],
                        payload: { lines: "39,40" },
                      },
                      {
                        content:
                          "<strong>Impact of High-Frequency Information:</strong> Evaluate the sensitivity of ViTs and CNNs to the removal of high-frequency image content through low-pass filtering, revealing differences in reliance on texture information.",
                        children: [],
                        payload: { lines: "40,41" },
                      },
                      {
                        content:
                          "<strong>CLIP-Trained ViT Feature Analysis:</strong> Examine the emergence of abstract concept detectors in ViTs trained with language supervision, contrasting them with object-specific features in image-classifier ViTs.",
                        children: [],
                        payload: { lines: "41,43" },
                      },
                    ],
                    payload: { lines: "36,43" },
                  },
                ],
                payload: { lines: "27,28" },
              },
              {
                content: "Findings",
                children: [
                  {
                    content:
                      "<strong>Spatial Information Preservation (Except Last Layer):</strong> ViTs preserve spatial information in patch tokens throughout all layers except the final attention block. This behavior contrasts with CNNs where spatial information is restricted by the receptive field size.",
                    children: [],
                    payload: { lines: "44,45" },
                  },
                  {
                    content:
                      "<strong>Last Layer as Learned Global Pooling:</strong> The final attention block in ViTs functions similarly to average pooling in CNNs, mixing information from all patches and discarding spatial localization.",
                    children: [],
                    payload: { lines: "45,46" },
                  },
                  {
                    content:
                      "<strong>Effective Background Information Usage:</strong> ViTs leverage background information more effectively than CNNs for classification and are more resilient to foreground removal.",
                    children: [],
                    payload: { lines: "46,47" },
                  },
                  {
                    content:
                      "<strong>Reduced Reliance on High-Frequency Information:</strong> ViTs are less sensitive to the loss of high-frequency image content compared to CNNs, indicating less dependence on texture for prediction.",
                    children: [],
                    payload: { lines: "47,48" },
                  },
                  {
                    content:
                      "<strong>Abstract Concept Detection in CLIP-Trained Models:</strong> Language model supervision leads ViTs to develop features that detect abstract concepts, parts of speech, and conceptual categories, going beyond object-specific features found in image classifiers.",
                    children: [],
                    payload: { lines: "48,50" },
                  },
                ],
                payload: { lines: "43,44" },
              },
            ],
            payload: { lines: "19,20" },
          },
          {
            content:
              '<a href="https://arxiv.org/abs/2310.16764">ConvNets Match Vision Transformers at Scale</a> <a href="">📑</a>',
            children: [
              {
                content: "Purpose",
                children: [
                  {
                    content:
                      "<strong>Challenge the Belief that Vision Transformers Outperform ConvNets at Scale:</strong> Many believe that while ConvNets excel on small to moderate datasets, they fall short compared to Vision Transformers when trained on web-scale datasets. This research aims to challenge this assumption.",
                    children: [],
                    payload: { lines: "53,54" },
                  },
                  {
                    content:
                      "<strong>Evaluate the Scaling Properties of ConvNets on Large Datasets:</strong> The study seeks to determine if ConvNets can achieve comparable performance to Vision Transformers when provided with similar computational resources and training data.",
                    children: [],
                    payload: { lines: "54,55" },
                  },
                  {
                    content:
                      "<strong>Re-examine the Effectiveness of ConvNets in the Era of Large-Scale Pre-Training:</strong> The research investigates whether ConvNets can benefit significantly from pre-training on massive datasets like Vision Transformers do.",
                    children: [],
                    payload: { lines: "55,57" },
                  },
                ],
                payload: { lines: "52,53" },
              },
              {
                content: "Method",
                children: [
                  {
                    content:
                      "<strong>Model Selection: NFNets:</strong>  The study utilizes the NFNet family of models, a purely convolutional architecture known for its high performance. The choice stems from the fact that NFNets were among the last ConvNets to achieve state-of-the-art results on ImageNet, demonstrating their potential.",
                    children: [],
                    payload: { lines: "58,59" },
                  },
                  {
                    content:
                      "<strong>Dataset: JFT-4B:</strong>  NFNets are trained on the JFT-4B dataset, a large-scale dataset containing approximately 4 billion labeled images across 30,000 classes. This dataset is commonly used for training foundation models, allowing for a fair comparison with Vision Transformers trained on similar data.",
                    children: [],
                    payload: { lines: "59,60" },
                  },
                  {
                    content:
                      "<strong>Compute Budgets:</strong>  The research explores a range of compute budgets for pre-training, from 0.4k to 110k TPU-v4 core compute hours. This approach enables a comparison of model performance across various levels of computational resources.",
                    children: [],
                    payload: { lines: "60,61" },
                  },
                  {
                    content: "<strong>Training Procedure:</strong>",
                    children: [
                      {
                        content:
                          "<strong>Varying Depth and Width:</strong> NFNets of different depths and widths are trained to investigate the impact of model size on performance.",
                        children: [],
                        payload: { lines: "62,63" },
                      },
                      {
                        content:
                          "<strong>Epoch Budgets and Learning Rate Tuning:</strong> Models are trained with various epoch budgets, and the learning rate is meticulously tuned for each budget to optimize performance.",
                        children: [],
                        payload: { lines: "63,64" },
                      },
                      {
                        content:
                          "<strong>Training Optimizations:</strong> The training leverages techniques like Stochastic Gradient Descent with Momentum, Adaptive Gradient Clipping (AGC), and removal of near-duplicate images from ImageNet in JFT-4B.",
                        children: [],
                        payload: { lines: "64,65" },
                      },
                    ],
                    payload: { lines: "61,65" },
                  },
                  {
                    content:
                      "<strong>Fine-tuning on ImageNet:</strong> To assess the generalizability of the pre-trained models, they are fine-tuned on the ImageNet dataset using techniques like sharpness aware minimization (SAM), stochastic depth, and dropout.",
                    children: [],
                    payload: { lines: "65,67" },
                  },
                ],
                payload: { lines: "57,58" },
              },
              {
                content: "Findings",
                children: [
                  {
                    content:
                      "<strong>Scaling Laws:</strong>  A clear log-log scaling law is observed between validation loss on JFT-4B and the compute budget used for pre-training. This finding mirrors the scaling laws previously observed in language modeling with transformers, suggesting that ConvNets exhibit similar scaling behavior.",
                    children: [],
                    payload: { lines: "68,69" },
                  },
                  {
                    content:
                      "<strong>Model Size and Epoch Budget:</strong>  The optimal model size and epoch budget, leading to the lowest validation loss, increase with the available compute budget. This observation suggests that larger models and longer training durations are beneficial when more computational resources are available.",
                    children: [],
                    payload: { lines: "69,70" },
                  },
                  {
                    content:
                      "<strong>Learning Rate Trends:</strong>  The optimal learning rate, minimizing validation loss, exhibits predictable behavior. It tends to be consistent for small epoch budgets but decreases gradually as the epoch budget and model size increase.",
                    children: [],
                    payload: { lines: "70,71" },
                  },
                  {
                    content:
                      "<strong>Competitive Performance on ImageNet:</strong>  Fine-tuned NFNets achieve ImageNet Top-1 accuracy levels comparable to Vision Transformers trained with similar compute budgets. This key finding challenges the notion of Vision Transformers being superior at scale.",
                    children: [],
                    payload: { lines: "71,72" },
                  },
                  {
                    content:
                      "<strong>Impact of Pre-training:</strong>  Large-scale pre-training significantly boosts the performance of NFNets on ImageNet, highlighting the importance of large datasets for training ConvNets.",
                    children: [],
                    payload: { lines: "72,73" },
                  },
                  {
                    content:
                      "<strong>Fine-tuning Considerations:</strong>  The study observes that pre-trained models with the lowest validation loss on JFT-4B don't always translate to the highest Top-1 accuracy on ImageNet after fine-tuning.  Fine-tuning often favors slightly larger models and shorter epoch budgets, potentially due to the larger models' capacity to adapt to the new task.",
                    children: [],
                    payload: { lines: "73,75" },
                  },
                ],
                payload: { lines: "67,68" },
              },
            ],
            payload: { lines: "50,51" },
          },
        ],
        payload: { lines: "17,18" },
      },
      {
        content: "Enhancing Efficiency and Performance",
        children: [
          {
            content:
              '<a href="https://arxiv.org/abs/2012.12877">Data-efficient Image Transformer (DeiT)</a> <a href="https://ritvik19.medium.com/papers-explained-39-deit-3d78dd98c8ec">📑</a>',
            children: [
              {
                content: "Purpose:",
                children: [
                  {
                    content:
                      "Improve data efficiency of ViT and achieve competitive performance with less training data.",
                    children: [],
                    payload: { lines: "79,80" },
                  },
                ],
                payload: { lines: "78,79" },
              },
              {
                content: "Method:",
                children: [
                  {
                    content:
                      "Employs improved training strategies and distillation.",
                    children: [],
                    payload: { lines: "81,82" },
                  },
                ],
                payload: { lines: "80,81" },
              },
              {
                content: "Advantages:",
                children: [
                  {
                    content:
                      "Reduces training time and data requirements compared to ViT.",
                    children: [],
                    payload: { lines: "83,84" },
                  },
                  {
                    content:
                      "Achieves performance comparable to CNNs with similar complexity and efficiency.",
                    children: [],
                    payload: { lines: "84,86" },
                  },
                ],
                payload: { lines: "82,83" },
              },
            ],
            payload: { lines: "77,78" },
          },
          {
            content:
              '<a href="https://arxiv.org/abs/2305.07027">Efficient ViT</a> <a href="https://ritvik19.medium.com/papers-explained-229-efficient-vit-cc87fbefbe49">📑</a>',
            children: [
              {
                content: "Purpose:",
                children: [
                  {
                    content:
                      "Improve the efficiency of vision transformers by addressing memory access and computation redundancy.",
                    children: [],
                    payload: { lines: "88,89" },
                  },
                ],
                payload: { lines: "87,88" },
              },
              {
                content: "Method:",
                children: [
                  {
                    content:
                      "Designs a new building block with a sandwich layout, using a single memory-bound MHSA layer between efficient FFN layers.",
                    children: [],
                    payload: { lines: "90,91" },
                  },
                  {
                    content:
                      "Introduces a cascaded group attention (CGA) module that feeds attention heads with different splits of the full feature to enhance diversity and reduce computation.",
                    children: [],
                    payload: { lines: "91,92" },
                  },
                  {
                    content:
                      "Redistributes parameters to prioritize critical network components like value projections while shrinking less important ones like hidden dimensions in FFNs.",
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
                    content:
                      "Achieves a good balance between speed and accuracy, outperforming existing efficient models.",
                    children: [],
                    payload: { lines: "94,95" },
                  },
                  {
                    content:
                      "Demonstrates good transferability to various downstream tasks.",
                    children: [],
                    payload: { lines: "95,97" },
                  },
                ],
                payload: { lines: "93,94" },
              },
            ],
            payload: { lines: "86,87" },
          },
          {
            content:
              '<a href="https://arxiv.org/abs/2305.13035">Shape-Optimized Vision Transformer (SoViT)</a> <a href="https://ritvik19.medium.com/papers-explained-234-sovit-a0ce3c7ef480">📑</a>',
            children: [
              {
                content: "Purpose:",
                children: [
                  {
                    content:
                      "Optimize the shape (width and depth) of vision transformer models to achieve compute-optimal designs.",
                    children: [],
                    payload: { lines: "99,100" },
                  },
                ],
                payload: { lines: "98,99" },
              },
              {
                content: "Method:",
                children: [
                  {
                    content:
                      "Extends and refines existing scaling law methods to optimize multiple shape dimensions jointly, requiring fewer experiments.",
                    children: [],
                    payload: { lines: "101,102" },
                  },
                  {
                    content:
                      "Applies this method to ViT, optimizing the model for a compute budget equivalent to ViT-g/14.",
                    children: [],
                    payload: { lines: "102,103" },
                  },
                ],
                payload: { lines: "100,101" },
              },
              {
                content: "Advantages:",
                children: [
                  {
                    content:
                      "Results in a smaller, faster model with the same quality as a much larger ViT model trained with the same compute.",
                    children: [],
                    payload: { lines: "104,105" },
                  },
                  {
                    content:
                      "Achieves competitive results across multiple tasks, including image classification, captioning, VQA, and zero-shot transfer.",
                    children: [],
                    payload: { lines: "105,107" },
                  },
                ],
                payload: { lines: "103,104" },
              },
            ],
            payload: { lines: "97,98" },
          },
        ],
        payload: { lines: "75,76" },
      },
      {
        content: "Adapting ViT for Diverse Vision Tasks",
        children: [
          {
            content:
              '<a href="https://arxiv.org/abs/2103.14030">Swin Transformer</a> <a href="https://ritvik19.medium.com/papers-explained-26-swin-transformer-39cf88b00e3e">📑</a>',
            children: [
              {
                content: "Purpose:",
                children: [
                  {
                    content:
                      "Adapt Transformer for a wider range of vision tasks, including dense prediction tasks.",
                    children: [],
                    payload: { lines: "111,112" },
                  },
                ],
                payload: { lines: "110,111" },
              },
              {
                content: "Method:",
                children: [
                  {
                    content:
                      "Introduces a hierarchical structure with shifted windows for multi-scale processing.",
                    children: [],
                    payload: { lines: "113,114" },
                  },
                  {
                    content:
                      "Employs shifted window attention to limit self-attention computation to local windows while allowing for cross-window connections.",
                    children: [],
                    payload: { lines: "114,115" },
                  },
                ],
                payload: { lines: "112,113" },
              },
              {
                content: "Advantages",
                children: [
                  {
                    content:
                      "Achieves linear computational complexity with respect to image size.",
                    children: [],
                    payload: { lines: "116,117" },
                  },
                  {
                    content:
                      "Suitable as a general-purpose backbone for various vision tasks, including image classification and dense prediction tasks.",
                    children: [],
                    payload: { lines: "117,118" },
                  },
                  {
                    content:
                      "Achieves state-of-the-art performance on various vision benchmarks.",
                    children: [],
                    payload: { lines: "118,120" },
                  },
                ],
                payload: { lines: "115,116" },
              },
            ],
            payload: { lines: "109,110" },
          },
          {
            content:
              '<a href="https://arxiv.org/abs/2111.09883">Swin Transformer V2</a> <a href="https://ritvik19.medium.com/papers-explained-215-swin-transformer-v2-53bee16ab668">📑</a>',
            children: [
              {
                content: "Purpose:",
                children: [
                  {
                    content:
                      "Scale up Swin Transformer to handle larger model capacity and higher resolution images.",
                    children: [],
                    payload: { lines: "122,123" },
                  },
                ],
                payload: { lines: "121,122" },
              },
              {
                content: "Method:",
                children: [
                  {
                    content:
                      "Introduces a residual-post-norm method and cosine attention to improve training stability.",
                    children: [],
                    payload: { lines: "124,125" },
                  },
                  {
                    content:
                      "Employs a log-spaced continuous position bias (Log-CPB) to transfer models pre-trained with low-resolution images to downstream tasks with high-resolution inputs.",
                    children: [],
                    payload: { lines: "125,126" },
                  },
                  {
                    content:
                      "Uses a self-supervised pre-training method, SimMIM, to reduce the need for large amounts of labeled data.",
                    children: [],
                    payload: { lines: "126,127" },
                  },
                  {
                    content:
                      "Incorporates techniques like zero-optimizer, activation checkpointing, and sequential self-attention computation to address memory issues.",
                    children: [],
                    payload: { lines: "127,128" },
                  },
                ],
                payload: { lines: "123,124" },
              },
              {
                content: "Advantages:",
                children: [
                  {
                    content:
                      "Enables training of larger Swin Transformer models (up to 3 billion parameters) with higher resolution images (up to 1536x1536).",
                    children: [],
                    payload: { lines: "129,130" },
                  },
                  {
                    content:
                      "Achieves state-of-the-art performance on various vision benchmarks with less training data and time compared to other large vision models.",
                    children: [],
                    payload: { lines: "130,133" },
                  },
                ],
                payload: { lines: "128,129" },
              },
            ],
            payload: { lines: "120,121" },
          },
        ],
        payload: { lines: "107,108" },
      },
      {
        content: "Hybridizing with Convolutional Techniques",
        children: [
          {
            content:
              '<a href="https://arxiv.org/abs/2103.15808">Convolutional Vision Transformer (CvT)</a> <a href="https://ritvik19.medium.com/papers-explained-199-cvt-fb4a5c05882e">📑</a>',
            children: [
              {
                content: "Purpose:",
                children: [
                  {
                    content:
                      "Improve ViT by combining the strengths of CNNs and Transformers.",
                    children: [],
                    payload: { lines: "137,138" },
                  },
                ],
                payload: { lines: "136,137" },
              },
              {
                content: "Method:",
                children: [
                  {
                    content:
                      "Introduces convolutional operations for token embedding, incorporating shift, scale, and distortion invariance.",
                    children: [],
                    payload: { lines: "139,140" },
                  },
                  {
                    content:
                      "Replaces linear projections with depth-wise separable convolutions in the transformer block.",
                    children: [],
                    payload: { lines: "140,141" },
                  },
                ],
                payload: { lines: "138,139" },
              },
              {
                content: "Advantages:",
                children: [
                  {
                    content:
                      "Achieves state-of-the-art performance on ImageNet-1k with fewer parameters and lower FLOPs compared to other ViTs and ResNets.",
                    children: [],
                    payload: { lines: "142,143" },
                  },
                  {
                    content:
                      "Maintains performance gains when pre-trained on larger datasets and fine-tuned to downstream tasks.",
                    children: [],
                    payload: { lines: "143,144" },
                  },
                  {
                    content:
                      "Eliminates the need for positional encoding, simplifying the design and allowing for variable input resolutions.",
                    children: [],
                    payload: { lines: "144,146" },
                  },
                ],
                payload: { lines: "141,142" },
              },
            ],
            payload: { lines: "135,136" },
          },
          {
            content:
              '<a href="https://arxiv.org/abs/2104.01136">LeViT (Vision Transformer in ConvNet\'s Clothing)</a> <a href="https://ritvik19.medium.com/papers-explained-205-levit-89a2defc2d18">📑</a>',
            children: [
              {
                content: "Purpose:",
                children: [
                  {
                    content:
                      "Optimize the trade-off between accuracy and efficiency for high-speed inference in ViTs.",
                    children: [],
                    payload: { lines: "148,149" },
                  },
                ],
                payload: { lines: "147,148" },
              },
              {
                content: "Method:",
                children: [
                  {
                    content:
                      "Adopts a pyramid structure with pooling, inspired by LeNet, for efficient multi-resolution processing.",
                    children: [],
                    payload: { lines: "150,151" },
                  },
                  {
                    content:
                      "Introduces the attention bias, a new way to integrate positional information.",
                    children: [],
                    payload: { lines: "151,152" },
                  },
                ],
                payload: { lines: "149,150" },
              },
              {
                content: "Advantages:",
                children: [
                  {
                    content:
                      "Offers better speed-accuracy trade-offs than ViT/DeiT models, particularly for small and medium-sized architectures.",
                    children: [],
                    payload: { lines: "153,154" },
                  },
                  {
                    content:
                      "Achieves better inference speed on GPUs, CPUs, and ARM hardware.",
                    children: [],
                    payload: { lines: "154,156" },
                  },
                ],
                payload: { lines: "152,153" },
              },
            ],
            payload: { lines: "146,147" },
          },
          {
            content:
              '<a href="https://arxiv.org/abs/2303.14189">FastViT</a> <a href="https://ritvik19.medium.com/papers-explained-225-fastvit-f1568536ed34">📑</a>',
            children: [
              {
                content: "Purpose:",
                children: [
                  {
                    content:
                      "Design a hybrid vision transformer model with a strong latency-accuracy trade-off.",
                    children: [],
                    payload: { lines: "158,159" },
                  },
                ],
                payload: { lines: "157,158" },
              },
              {
                content: "Method:",
                children: [
                  {
                    content:
                      "Combines transformers and convolutional layers to efficiently capture both local and global information.",
                    children: [],
                    payload: { lines: "160,161" },
                  },
                  {
                    content:
                      "Uses RepMixer, a token mixer with structural reparameterization, to reduce memory access costs.",
                    children: [],
                    payload: { lines: "161,162" },
                  },
                  {
                    content:
                      "Employs factorized convolutions with train-time overparameterization in the stem and patch embedding layers for improved efficiency and performance.",
                    children: [],
                    payload: { lines: "162,163" },
                  },
                ],
                payload: { lines: "159,160" },
              },
              {
                content: "Advantages:",
                children: [
                  {
                    content:
                      "Aims for state-of-the-art latency-accuracy trade-off in vision transformer models.",
                    children: [],
                    payload: { lines: "164,166" },
                  },
                ],
                payload: { lines: "163,164" },
              },
            ],
            payload: { lines: "156,157" },
          },
          {
            content:
              '<a href="https://arxiv.org/abs/2204.01697">Multi-Axis Vision Transformer (MaxViT)</a> <a href="https://ritvik19.medium.com/papers-explained-210-maxvit-6c68cc515413">📑</a>',
            children: [
              {
                content: "Purpose:",
                children: [
                  {
                    content:
                      "Design an efficient and scalable vision transformer that can handle high-resolution images.",
                    children: [],
                    payload: { lines: "168,169" },
                  },
                ],
                payload: { lines: "167,168" },
              },
              {
                content: "Method:",
                children: [
                  {
                    content:
                      "Introduces multi-axis attention, which combines blocked local and dilated global attention for capturing both local and global spatial interactions.",
                    children: [],
                    payload: { lines: "170,171" },
                  },
                  {
                    content:
                      "Blends multi-axis attention with convolutions in a simple hierarchical backbone architecture.",
                    children: [],
                    payload: { lines: "171,172" },
                  },
                ],
                payload: { lines: "169,170" },
              },
              {
                content: "Advantages:",
                children: [
                  {
                    content:
                      "Achieves linear complexity with respect to image size while allowing for global-local spatial interactions at arbitrary input resolutions.",
                    children: [],
                    payload: { lines: "173,174" },
                  },
                  {
                    content:
                      "Excels on a wide range of vision tasks, including image classification, object detection, and visual aesthetic assessment.",
                    children: [],
                    payload: { lines: "174,175" },
                  },
                  {
                    content:
                      "Demonstrates strong generative modeling capabilities.",
                    children: [],
                    payload: { lines: "175,177" },
                  },
                ],
                payload: { lines: "172,173" },
              },
            ],
            payload: { lines: "166,167" },
          },
        ],
        payload: { lines: "133,134" },
      },
      {
        content: "Exploring Self-Supervised Learning",
        children: [
          {
            content:
              '<a href="https://arxiv.org/abs/2106.08254">BERT Pre-Training of Image Transformers (BEiT)</a> <a href="https://ritvik19.medium.com/papers-explained-27-beit-b8c225496c01">📑</a>',
            children: [
              {
                content: "Purpose:",
                children: [
                  {
                    content:
                      "Pre-train vision transformers effectively using a masked image modeling task inspired by BERT in NLP.",
                    children: [],
                    payload: { lines: "181,182" },
                  },
                ],
                payload: { lines: "180,181" },
              },
              {
                content: "Method:",
                children: [
                  {
                    content:
                      "Proposes a masked image modeling (MIM) pre-training task.",
                    children: [],
                    payload: { lines: "183,184" },
                  },
                  {
                    content:
                      "Uses two views of each image: image patches and visual tokens (discrete tokens) obtained from a discrete variational autoencoder (VAE).",
                    children: [],
                    payload: { lines: "184,185" },
                  },
                  {
                    content:
                      "Masks some image patches and tasks the model with predicting the original visual tokens based on the corrupted image patches.",
                    children: [],
                    payload: { lines: "185,186" },
                  },
                ],
                payload: { lines: "182,183" },
              },
              {
                content: "Advantages:",
                children: [
                  {
                    content:
                      "Achieves competitive results on image classification and semantic segmentation tasks compared to previous pre-training methods.",
                    children: [],
                    payload: { lines: "187,188" },
                  },
                  {
                    content:
                      "Enables self-supervised pre-training of vision transformers, leveraging large-scale image data without labels.",
                    children: [],
                    payload: { lines: "188,189" },
                  },
                  {
                    content:
                      "Demonstrates the ability to distinguish semantic regions and object boundaries without human annotation.",
                    children: [],
                    payload: { lines: "189,191" },
                  },
                ],
                payload: { lines: "186,187" },
              },
            ],
            payload: { lines: "179,180" },
          },
          {
            content:
              '<a href="https://arxiv.org/abs/2111.06377">Masked AutoEncoder (MAE)</a> <a href="https://ritvik19.medium.com/papers-explained-28-masked-autoencoder-38cb0dbed4af">📑</a>',
            children: [
              {
                content: "Purpose:",
                children: [
                  {
                    content:
                      "Explore self-supervised learning for vision using a masked autoencoding approach.",
                    children: [],
                    payload: { lines: "193,194" },
                  },
                ],
                payload: { lines: "192,193" },
              },
              {
                content: "Method:",
                children: [
                  {
                    content:
                      "Masks a high portion of random patches in an image.",
                    children: [],
                    payload: { lines: "195,196" },
                  },
                  {
                    content:
                      "Tasks the model with reconstructing the missing patches based on the visible ones.",
                    children: [],
                    payload: { lines: "196,197" },
                  },
                ],
                payload: { lines: "194,195" },
              },
              {
                content: "Advantages:",
                children: [
                  {
                    content:
                      "Encourages holistic understanding of images by requiring the model to infer missing information.",
                    children: [],
                    payload: { lines: "198,199" },
                  },
                  {
                    content:
                      "Reduces redundancy in the input data and creates a challenging self-supervisory task.",
                    children: [],
                    payload: { lines: "199,200" },
                  },
                  {
                    content:
                      "The design of the decoder plays a crucial role in determining the semantic level of the learned representations.",
                    children: [],
                    payload: { lines: "200,202" },
                  },
                ],
                payload: { lines: "197,198" },
              },
            ],
            payload: { lines: "191,192" },
          },
        ],
        payload: { lines: "177,178" },
      },
      {
        content: "Optimizing for Mobile and Edge Devices",
        children: [
          {
            content:
              '<a href="https://arxiv.org/abs/2110.02178">MobileViT</a> <a href="https://ritvik19.medium.com/papers-explained-40-mobilevit-4793f149c434">📑</a>',
            children: [
              {
                content: "Purpose:",
                children: [
                  {
                    content:
                      "Design an efficient and scalable vision transformer that can handle high-resolution images.",
                    children: [],
                    payload: { lines: "206,207" },
                  },
                ],
                payload: { lines: "205,206" },
              },
              {
                content: "Method:",
                children: [
                  {
                    content:
                      "Introduces multi-axis attention, which combines blocked local and dilated global attention for capturing both local and global spatial interactions.",
                    children: [],
                    payload: { lines: "208,209" },
                  },
                  {
                    content:
                      "Blends multi-axis attention with convolutions in a simple hierarchical backbone architecture.",
                    children: [],
                    payload: { lines: "209,210" },
                  },
                ],
                payload: { lines: "207,208" },
              },
              {
                content: "Advantages:",
                children: [
                  {
                    content:
                      "Achieves linear complexity with respect to image size while allowing for global-local spatial interactions at arbitrary input resolutions.",
                    children: [],
                    payload: { lines: "211,212" },
                  },
                  {
                    content:
                      "Excels on a wide range of vision tasks, including image classification, object detection, and visual aesthetic assessment.",
                    children: [],
                    payload: { lines: "212,213" },
                  },
                  {
                    content:
                      "Demonstrates strong generative modeling capabilities.",
                    children: [],
                    payload: { lines: "213,215" },
                  },
                ],
                payload: { lines: "210,211" },
              },
            ],
            payload: { lines: "204,205" },
          },
          {
            content:
              '<a href="https://arxiv.org/abs/2206.01191">EfficientFormer</a> <a href="https://ritvik19.medium.com/papers-explained-220-efficientformer-97c91540af19">📑</a>',
            children: [
              {
                content: "Purpose:",
                children: [
                  {
                    content:
                      "Optimize vision transformer models for fast inference speed on resource-constrained edge devices.",
                    children: [],
                    payload: { lines: "217,218" },
                  },
                ],
                payload: { lines: "216,217" },
              },
              {
                content: "Method:",
                children: [
                  {
                    content:
                      "Performs a comprehensive latency analysis of ViT and its variants on mobile devices.",
                    children: [],
                    payload: { lines: "219,220" },
                  },
                  {
                    content:
                      "Introduces a dimension-consistent design paradigm, using 4D MetaBlocks and 3D MHSA blocks.",
                    children: [],
                    payload: { lines: "220,221" },
                  },
                  {
                    content:
                      "Employs a latency-driven slimming method to search for optimal model configurations based on mobile inference speed.",
                    children: [],
                    payload: { lines: "221,222" },
                  },
                ],
                payload: { lines: "218,219" },
              },
              {
                content: "Advantages:",
                children: [
                  {
                    content:
                      "Achieves ultra-fast inference speed while maintaining competitive performance on image classification, object detection, and segmentation tasks.",
                    children: [],
                    payload: { lines: "223,224" },
                  },
                  {
                    content:
                      "Outperforms existing transformer models in terms of speed and accuracy.",
                    children: [],
                    payload: { lines: "224,225" },
                  },
                ],
                payload: { lines: "222,223" },
              },
            ],
            payload: { lines: "215,216" },
          },
        ],
        payload: { lines: "202,203" },
      },
    ],
    payload: { lines: "1,2" },
  },
  { color: ["#2980b9"], maxWidth: 400, initialExpandLevel: 3 }
);
