const vision_transformers = [
  {
    title: "Vision Transformer",
    link: "https://ritvik19.medium.com/papers-explained-25-vision-transformers-e286ee8bc06b",
    date: "October 2020",
    description:
      "Images are segmented into patches, which are treated as tokens and a sequence of linear embeddings of these patches are input to a Transformer",
    tags: ["Vision Transformers"],
  },
  {
    title: "Data Efficient Image Transformer (DeiT)",
    link: "https://ritvik19.medium.com/papers-explained-39-deit-3d78dd98c8ec",
    date: "December 2020",
    description:
      "A convolution-free vision transformer that uses a teacher-student strategy with attention-based distillation tokens.",
    tags: ["Vision Transformers"],
  },
  {
    title: "Swin Transformer",
    link: "https://ritvik19.medium.com/papers-explained-26-swin-transformer-39cf88b00e3e",
    date: "March 2021",
    description:
      "A hierarchical vision transformer that uses shifted windows to addresses the challenges of adapting the transformer model to computer vision.",
    tags: ["Vision Transformers"],
  },
  {
    title: "Convolutional Vision Transformer (CvT)",
    link: "https://ritvik19.medium.com/papers-explained-199-cvt-fb4a5c05882e",
    date: "March 2021",
    description:
      "Improves Vision Transformer (ViT) in performance and efficiency by introducing convolutions, to yield the best of both designs.",
    tags: ["Vision Transformers"],
  },
  {
    title: "LeViT",
    link: "https://ritvik19.medium.com/papers-explained-205-levit-89a2defc2d18",
    date: "April 2021",
    description:
      "A hybrid neural network built upon the ViT architecture and DeiT training method, for fast inference image classification.",
    tags: ["Vision Transformers"],
  },
  {
    title: "DINO",
    link: "https://ritvik19.medium.com/papers-explained-249-dino-f7e2c7f438ab",
    date: "April 2021",
    description:
      "Investigates whether self-supervised learning provides new properties to Vision Transformer that stand out compared to convolutional networks and finds that self-supervised ViT features contain explicit information about the semantic segmentation of an image, and are also excellent k-NN classifiers.",
    tags: ["Vision Transformers"],
  },
  {
    title: "BEiT",
    link: "https://ritvik19.medium.com/papers-explained-27-beit-b8c225496c01",
    date: "June 2021",
    description:
      "Utilizes a masked image modeling task inspired by BERT in, involving image patches and visual tokens to pretrain vision Transformers.",
    tags: ["Vision Transformers"],
  },
  {
    title: "MobileViT",
    link: "https://ritvik19.medium.com/papers-explained-40-mobilevit-4793f149c434",
    date: "October 2021",
    description:
      "A lightweight vision transformer designed for mobile devices, effectively combining the strengths of CNNs and ViTs.",
    tags: ["Vision Transformers"],
  },
  {
    title: "Masked AutoEncoder",
    link: "https://ritvik19.medium.com/papers-explained-28-masked-autoencoder-38cb0dbed4af",
    date: "November 2021",
    description:
      "An encoder-decoder architecture that reconstructs input images by masking random patches and leveraging a high proportion of masking for self-supervision.",
    tags: ["Vision Transformers"],
  },
  {
    title: "DINOv2",
    link: "https://ritvik19.medium.com/papers-explained-250-dino-v2-e1e6d12a5c85",
    date: "April 2022",
    description:
      "Demonstrates that existing self-supervised pre-training methods can produce general-purpose visual features by training on curated data from diverse sources, and proposes a new approach that combines techniques to scale pre-training with larger models and datasets.",
    tags: ["Vision Transformers"],
  },
  {
    title: "Multi-Axis Vision Transformer (MaxViT)",
    link: "https://ritvik19.medium.com/papers-explained-210-maxvit-6c68cc515413",
    date: "April 2022",
    description:
      "Introduces multi-axis attention, allowing global-local spatial interactions on arbitrary input resolutions with only linear complexity.",
    tags: ["Vision Transformers"],
  },
  {
    title: "Swin Transformer V2",
    link: "https://ritvik19.medium.com/papers-explained-215-swin-transformer-v2-53bee16ab668",
    date: "April 2022",
    description:
      "A successor to Swin Transformer, addressing challenges like training stability, resolution gaps, and labeled data scarcity.",
    tags: ["Vision Transformers"],
  },
  {
    title: "EfficientFormer",
    link: "https://ritvik19.medium.com/papers-explained-220-efficientformer-97c91540af19",
    date: "June 2022",
    description:
      "Revisits the design principles of ViT and its variants through latency analysis and identifies inefficient designs and operators in ViT to propose a new dimension consistent design paradigm for vision transformers and a simple yet effective latency-driven slimming method to optimize for inference speed.",
    tags: ["Vision Transformers"],
  },
  {
    title: "FastVit",
    link: "https://ritvik19.medium.com/papers-explained-225-fastvit-f1568536ed34",
    date: "March 2023",
    description:
      "A hybrid vision transformer architecture featuring a novel token mixing operator called RepMixer, which significantly improves model efficiency.",
    tags: ["Vision Transformers"],
  },
  {
    title: "Efficient ViT",
    link: "https://ritvik19.medium.com/papers-explained-229-efficient-vit-cc87fbefbe49",
    date: "May 2023",
    description:
      "Employs a single memory-bound MHSA between efficient FFN layers, improves memory efficiency while enhancing channel communication.",
    tags: ["Vision Transformers"],
  },
  {
    title: "Shape-Optimized Vision Transformer (SoViT)",
    link: "https://ritvik19.medium.com/papers-explained-234-sovit-a0ce3c7ef480",
    date: "May 2023",
    description:
      "A shape-optimized vision transformer that achieves competitive results with models twice its size, while being pre-trained with an equivalent amount of compute.",
    tags: ["Vision Transformers"],
  },
  {
    title: "Autoregressive Image Models (AIM)",
    link: "https://ritvik19.medium.com/papers-explained-318-autoregressive-image-models-aim-d6a90f93876f",
    date: "January 2024",
    description:
      "8B Vision models pre-trained using an autoregressive objective, similar to Large Language Models, on 2B images, demonstrating scaling properties utilizing architectural modifications like prefix attention and a parameterized prediction head.",
    tags: ["Vision Transformers", "Autoregressive Image Models"],
  },
  {
    title: "Autoregressive Image Models V2",
    link: "https://ritvik19.medium.com/papers-explained-319-autoregressive-image-models-v2-aim-v2-e28eadf5ba9b",
    date: "November 2024",
    description:
      "A family of open vision encoders, ranging from 300M to 3B parameters, extending the AIM framework to images and text, pre-trained with a multimodal autoregressive approach, generating both image patches and text tokens using a causal decoder.",
    tags: [
      "Vision Transformers",
      "Multimodal Models",
      "Autoregressive Image Models",
    ],
  },
  {
    title: "Perception Encoder",
    link: "",
    date: "April 2025",
    description:
      "A vision encoder trained via vision-language learning that achieves state-of-the-art results on various tasks, including zero-shot image and video classification/retrieval, document/image/video Q&A, and spatial tasks like detection and depth estimation. It leverages contrastive vision-language training, language alignment, and spatial alignment to produce strong, general embeddings from intermediate layers, outperforming existing models.",
    tags: ["Vision Transformers"],
  },
];
