const multi_modal_lms = [
  {
    title: "Florence",
    link: "https://ritvik19.medium.com/papers-explained-213-florence-f93a3a7d9ef0",
    date: "November 2021",
    description:
      "A computer vision foundation model that can be adapted to various tasks by expanding representations from coarse (scene) to fine (object), static (images) to dynamic (videos), and RGB to multiple modalities.",
    tags: ["Multimodal Models"],
  },
  {
    title: "BLIP",
    link: "https://ritvik19.medium.com/papers-explained-154-blip-6d85c80a744d",
    date: "February 2022",
    description:
      "A Vision-Language Pre-training (VLP) framework that introduces Multimodal mixture of Encoder-Decoder (MED) and Captioning and Filtering (CapFilt), a new dataset bootstrapping method for learning from noisy image-text pairs.",
    tags: ["Multimodal Models"],
  },
  {
    title: "Flamingo",
    link: "https://ritvik19.medium.com/papers-explained-82-flamingo-8c124c394cdb",
    date: "April 2022",
    description:
      "Visual Language Models enabling seamless handling of interleaved visual and textual data, and facilitating few-shot learning on large-scale web corpora.",
    tags: ["Multimodal Models"],
  },
  {
    title: "PaLI",
    link: "https://ritvik19.medium.com/papers-explained-194-pali-c1fffc14068c",
    date: "September 2022",
    description:
      "A joint language-vision model that generates multilingual text based on visual and textual inputs, trained using large pre-trained encoder-decoder language models and Vision Transformers, specifically mT5 and ViT-e.",
    tags: ["Multimodal Models"],
  },
  {
    title: "BLIP 2",
    link: "https://ritvik19.medium.com/papers-explained-155-blip-2-135fff70bf65",
    date: "January 2023",
    description:
      "A Vision-Language Pre-training (VLP) framework that proposes Q-Former, a trainable module to bridge the gap between a frozen image encoder and a frozen LLM to bootstrap vision-language pre-training.",
    tags: ["Multimodal Models"],
  },
  {
    title: "LLaVA 1",
    link: "https://ritvik19.medium.com/papers-explained-102-llava-1-eb0a3db7e43c",
    date: "April 2023",
    description:
      "A large multimodal model connecting CLIP and Vicuna trained end-to-end on instruction-following data generated through GPT-4 from image-text pairs.",
    tags: ["Multimodal Models"],
  },
  {
    title: "PaLI-X",
    link: "https://ritvik19.medium.com/papers-explained-195-pali-x-f9859e73fd97",
    date: "May 2023",
    description:
      "A multilingual vision and language model with scaled-up components, specifically ViT-22 B and UL2 32B, exhibits emergent properties such as complex counting and multilingual object detection, and demonstrates improved performance across various tasks.",
    tags: ["Multimodal Models"],
  },
  {
    title: "InstructBLIP",
    link: "https://ritvik19.medium.com/papers-explained-156-instructblip-c3cf3291a823",
    date: "May 2023",
    description:
      "Introduces instruction-aware Query Transformer to extract informative features tailored to the given instruction to study vision-language instruction tuning based on the pretrained BLIP-2 models.",
    tags: ["Multimodal Models"],
  },
  {
    title: "Idefics",
    link: "https://ritvik19.medium.com/papers-explained-179-obelics-idefics-a581f8d909b6",
    date: "June 2023",
    description:
      "9B and 80B multimodal models trained on Obelics, an open web-scale dataset of interleaved image-text documents, curated in this work.",
    tags: ["Multimodal Models", "HuggingFace"],
  },
  {
    title: "PaLI-3",
    link: "https://ritvik19.medium.com/papers-explained-196-pali-3-2f5cf92f60a8",
    date: "October 2023",
    description:
      "A 5B vision language model, built upon a 2B SigLIP Vision Model and UL2 3B Language Model outperforms larger models on various benchmarks and achieves SOTA on several video QA benchmarks despite not being pretrained on any video data.",
    tags: ["Multimodal Models"],
  },
  {
    title: "LLaVA 1.5",
    link: "https://ritvik19.medium.com/papers-explained-103-llava-1-5-ddcb2e7f95b4",
    date: "October 2023",
    description:
      "An enhanced version of the LLaVA model that incorporates a CLIP-ViT-L-336px with an MLP projection and academic-task-oriented VQA data to set new benchmarks in large multimodal models (LMM) research.",
    tags: ["Multimodal Models"],
  },
  {
    title: "Florence-2",
    link: "https://ritvik19.medium.com/papers-explained-214-florence-2-c4e17246d14b",
    date: "November 2023",
    description:
      "A vision foundation model with a unified, prompt-based representation for a variety of computer vision and vision-language tasks.",
    tags: ["Multimodal Models"],
  },
  {
    title: "CogVLM",
    link: "https://ritvik19.medium.com/papers-explained-235-cogvlm-9f3aa657f9b1",
    date: "November 2023",
    description:
      "Bridges the gap between the frozen pretrained language model and image encoder by a trainable visual expert module in the attention and FFN layers.",
    tags: ["Multimodal Models"],
  },
  {
    title: "MoE-LLaVA",
    link: "https://ritvik19.medium.com/papers-explained-104-moe-llava-cf14fda01e6f",
    date: "January 2024",
    description:
      "A MoE-based sparse LVLM framework that activates only the top-k experts through routers during deployment, maintaining computational efficiency while achieving comparable performance to larger models.",
    tags: ["Multimodal Models", "Mixtures of Experts"],
  },
  {
    title: "LLaVA 1.6",
    link: "https://ritvik19.medium.com/papers-explained-107-llava-1-6-a312efd496c5",
    date: "January 2024",
    description:
      "An improved version of a LLaVA 1.5 with enhanced reasoning, OCR, and world knowledge capabilities, featuring increased image resolution",
    tags: ["Multimodal Models"],
  },
  {
    title: "MM-1",
    link: "https://ritvik19.medium.com/papers-explained-117-mm1-c579142bcdc0",
    date: "March 2024",
    description:
      "Studies the importance of various architecture components and data choices. Through comprehensive ablations of the image encoder, the vision language connector, and various pre-training data choices, and identifies several crucial design lessons.",
    tags: ["Multimodal Models"],
  },
  {
    title: "Idefics2",
    link: "https://ritvik19.medium.com/papers-explained-180-idefics-2-0adf35cef4ee",
    date: "April 2024",
    description:
      "Improvement upon Idefics1 with enhanced OCR capabilities, simplified architecture, and better pre-trained backbones, trained on a mixture of openly available datasets and fine-tuned on task-oriented data.",
    tags: ["Multimodal Models", "HuggingFace"],
  },
  {
    title: "Phi-3 Vision",
    link: "https://ritvik19.medium.com/papers-explained-130-phi-3-0dfc951dc404#7ba6",
    date: "May 2024",
    description:
      "First multimodal model in the Phi family, bringing the ability to reason over images and extract and reason over text from images.",
    tags: ["Multimodal Models", "Synthetic Data", "Phi"],
  },
  {
    title: "An Introduction to Vision-Language Modeling",
    link: "https://ritvik19.medium.com/papers-explained-an-introduction-to-vision-language-modeling-89e7697da6e3",
    date: "May 2024",
    description:
      "Provides a comprehensive introduction to VLMs, covering their definition, functionality, training methods, and evaluation approaches, aiming to help researchers and practitioners enter the field and advance the development of VLMs for various applications.",
    tags: ["Multimodal Models"],
  },
  {
    title: "Chameleon",
    link: "https://ritvik19.medium.com/papers-explained-143-chameleon-6cddfdbceaa8",
    date: "May 2024",
    description:
      "A family of early-fusion token-based mixed-modal models capable of understanding and generating images and text in any arbitrary sequence.",
    tags: ["Multimodal Models"],
  },
  {
    title: "Pali Gemma",
    link: "https://ritvik19.medium.com/papers-explained-197-pali-gemma-6899e871998e",
    date: "July 2024",
    description:
      "Combines SigLIP vision model and the Gemma language model and follows the PaLI-3 training recipe to achieve strong performance on various vision-language tasks.",
    tags: ["Multimodal Models", "Gemma"],
  },
  {
    title: "BLIP-3 (xGen-MM)",
    link: "https://ritvik19.medium.com/papers-explained-190-blip-3-xgen-mm-6a9c04a3892d",
    date: "August 2024",
    description:
      "A comprehensive system for developing Large Multimodal Models, comprising curated datasets, training recipes, model architectures, and pre-trained models that demonstrate strong in-context learning capabilities and competitive performance on various tasks.",
    tags: ["Multimodal Models"],
  },
  {
    title: "Idefics 3",
    link: "https://ritvik19.medium.com/papers-explained-218-idefics-3-81791c4cde3f",
    date: "August 2024",
    description:
      "A VLM based on Llama 3.1 and SigLIP-SO400M trained efficiently, using only open datasets and a straightforward pipeline, significantly outperforming in document understanding tasks.",
    tags: ["Multimodal Models", "HuggingFace"],
  },
  {
    title: "CogVLM2",
    link: "https://ritvik19.medium.com/papers-explained-236-cogvlm2-db0261745cf5",
    date: "August 2024",
    description:
      "A family of visual language models that enables image and video understanding with improved training recipes, exploring enhanced vision-language fusion, higher input resolution, and broader modalities and applications.",
    tags: ["Multimodal Models"],
  },
  {
    title: "Eagle",
    link: "https://ritvik19.medium.com/papers-explained-269-eagle-09c21e549395",
    date: "August 2024",
    description:
      "Provides an extensive exploration of the design space for MLLMs using a mixture of vision encoders and resolutions, and reveals several underlying principles common to various existing strategies, leading to a streamlined yet effective design approach.",
    tags: ["Multimodal Models", "Nvidia"],
  },
  {
    title: "Pixtral",
    link: "https://ritvik19.medium.com/papers-explained-219-pixtral-a714f94e59ac",
    date: "September 2024",
    description:
      "A 12B parameter natively multimodal vision-language model, trained with interleaved image and text data demonstrating strong performance on multimodal tasks, and excels in instruction following.",
    tags: ["Multimodal Models", "Mistral"],
  },
  {
    title: "NVLM",
    link: "https://ritvik19.medium.com/papers-explained-240-nvlm-7ad201bfbfc2",
    date: "September 2024",
    description:
      "A family of multimodal large language models, provides a comparison between decoder-only multimodal LLMs and cross-attention based models and proposes a hybrid architecture, it further introduces a 1-D title-tagging design for tile-based dynamic high resolution images.",
    tags: ["Multimodal Models", "Nvidia"],
  },
  {
    title: "Molmo",
    link: "https://ritvik19.medium.com/papers-explained-241-pixmo-and-molmo-239d70abebff",
    date: "September 2024",
    description:
      "A family of open-weight vision-language models that achieve state-of-the-art performance by leveraging a novel, human-annotated image caption dataset called PixMo.",
    tags: ["Multimodal Models"],
  },
  {
    title: "MM-1.5",
    link: "https://ritvik19.medium.com/papers-explained-261-mm-1-5-d0dd01a9b68b",
    date: "September 2024",
    description:
      "A family of multimodal large language models designed to enhance capabilities in text-rich image understanding, visual referring and grounding, and multi-image reasoning, achieved through a data-centric approach involving diverse data mixtures, And specialized variants for video and mobile UI understanding.",
    tags: ["Multimodal Models"],
  },
  {
    title: "Mississippi",
    link: "https://ritvik19.medium.com/papers-explained-251-h2ovl-mississippi-1508e9c8e862",
    date: "October 2024",
    description:
      "A collection of small, efficient, open-source vision-language models built on top of Danube, trained on 37 million image-text pairs, specifically designed to perform well on document analysis and OCR tasks while maintaining strong performance on general vision-language benchmarks.",
    tags: ["Multimodal Models", "Danube", "H2O"],
  },
  {
    title: "Pangea",
    link: "",
    date: "October 2024",
    description:
      "An open-source multilingual multimodal LLM designed to bridge linguistic and cultural gaps in visual understanding tasks, trained on PangeaIns, a 6 million sample instruction dataset spanning 39 languages, and evaluated using PangeaBench, a holistic evaluation suite encompassing 14 datasets covering 47 languages.",
    tags: ["Multimodal Models", "Multilingual Models"],
  },
  {
    title: "Pixtral Large",
    link: "https://ritvik19.medium.com/papers-explained-219-pixtral-a714f94e59ac#6123",
    date: "November 2024",
    description:
      "A 124 billion parameter open-weight multimodal model built upon Mistral Large 2 and a 1B parameter vision encoder, excelling in understanding documents, charts and natural images. It supports a Context Window of 128K tokens, accommodating at least 30 high-resolution images.",
    tags: ["Multimodal Models", "Mistral"],
  },
  {
    title: "Smol VLM",
    link: "https://ritvik19.medium.com/papers-explained-346-smolvlm-9b4e208fa66b",
    date: "November 2024",
    description:
      "A 2B vision-language model, built using a modified Idefics3 architecture with a smaller language backbone (SmolLM2 1.7B), aggressive pixel shuffle compression, 384x384 image patches, and a shape-optimized SigLIP vision backbone, featuring a 16k token context window.",
    tags: ["Multimodal Models", "HuggingFace"],
  },
  {
    title: "MAmmoTH-VL",
    link: "https://ritvik19.medium.com/papers-explained-296-mammoth-vl-6abec7a58831",
    date: "December 2024",
    description:
      "Curated a large-scale, multimodal instruction-tuning dataset with 12M instruction-response pairs using a cost-effective method involving open-source data collection and categorization, task-specific data augmentation and rewriting using open models (Llama-3-70B-Instruct for caption-based data and InternVL2-Llama3-76B for other data types), and self-filtering with InternVL2-Llama3-76B to remove hallucinations and ensure data quality.",
    tags: ["Multimodal Models", "Synthetic Data"],
  },
  {
    title: "Maya",
    link: "https://ritvik19.medium.com/papers-explained-297-maya-0a38799daa43",
    date: "December 2024",
    description:
      "An open-source multilingual multimodal model designed to improve vision-language understanding in eight languages (English, Chinese, French, Spanish, Russian, Hindi, Japanese, and Arabic). It leverages a newly created, toxicity-filtered multilingual image-text dataset based on LLaVA, incorporating a SigLIP vision encoder and the Aya-23 8B language model, and is fine-tuned on the PALO 150K instruction-tuning dataset.",
    tags: ["Multimodal Models", "Multilingual Models"],
  },
  {
    title: "Llava-Mini",
    link: "https://ritvik19.medium.com/papers-explained-298-llava-mini-fe3a25b9e747",
    date: "January 2025",
    description:
      "An efficient large multimodal model that minimizes vision tokens by pre-fusing visual information from a CLIP vision encoder into text tokens before feeding them, along with a small number of compressed vision tokens (achieved via query-based compression), to an LLM backbone, allowing for efficient processing of standard and high-resolution images, as well as videos, by significantly reducing the number of tokens the LLM needs to handle while preserving visual understanding.",
    tags: ["Multimodal Models"],
  },
  {
    title: "Kimi k1.5",
    link: "https://ritvik19.medium.com/papers-explained-334-kimi-k1-5-c41e914cae7b",
    date: "January 2025",
    description:
      "A multimodal LLM trained with reinforcement learning (RL) focused on long context scaling and improved policy optimization, achieving state-of-the-art reasoning performance in various benchmarks and modalities, matching OpenAI's o1 in long-context tasks and significantly outperforming short-context models like GPT-4o and Claude Sonnet 3.5 through effective long2short context compression methods.",
    tags: ["Multimodal Models"],
  },
  {
    title: "Eagle 2",
    link: "https://ritvik19.medium.com/papers-explained-378-eagle-2-cda1e612c0b4",
    date: "January 2025",
    description:
      "A family of performant vision-language models (VLMs) developed by building a post-training data strategy from scratch, focusing on data collection, filtering, selection, and augmentation, alongside a vision-centric architecture design with tiled mixture of vision encoders and a three-stage training recipe.",
    tags: ["Multimodal Models", "Nvidia"],
  },
  {
    title: "Phi-4 Multimodal",
    link: "https://ritvik19.medium.com/papers-explained-322-phi-4-mini-phi-4-multimodal-2be1a69be78c",
    date: "February 2025",
    description:
      "Extends Phi-4-Mini with vision and speech/audio modalities via a novel mixture of LoRAs approach, enabling combined modality inference without interference and achieving state-of-the-art performance in various multimodal tasks while maintaining the base language model's capabilities.",
    tags: ["Multimodal Models", "Synthetic Data", "Phi"],
  },
  {
    title: "Aya Vision",
    link: "https://ritvik19.medium.com/papers-explained-332-aya-vision-5aec8dce396e",
    date: "March 2025",
    description:
      "A family of open-weight 8B and 32B parameter multilingual vision-language models (VLMs) supporting 23 languages, built upon Aya Expanse and incorporating techniques like synthetic annotations, data translation/rephrasing, and multimodal model merging.",
    tags: ["Multimodal Models", "Multilingual Models"],
  },
  {
    title: "Gemma 3",
    link: "https://ritvik19.medium.com/papers-explained-329-gemma-3-153803a2c591",
    date: "March 2025",
    description:
      "A multimodal language model with vision understanding, wider language coverage, and a longer context (128k tokens) than its predecessors. It utilizes a modified architecture with increased local attention to reduce KV-cache memory demands, is trained with distillation, and employs a novel post-training recipe improving performance in areas like math, chat, and multilingual tasks.",
    tags: ["Multimodal Models", "Gemma"],
  },
  {
    title: "MAmmoTH-VL 2",
    link: "https://ritvik19.medium.com/papers-explained-331-mammoth-vl-2-108ac94dc3b3",
    date: "March 2025",
    description:
      "Introduces VisualWebInstruct, a novel approach leveraging Google Image Search and LLMs (Gemini and GPT-4) to create a large-scale, diverse multimodal instruction dataset of ~900K question-answer pairs (40% visual) focused on complex reasoning across various scientific disciplines. Fine-tuning existing VLMs on this dataset leads to significant performance gains on reasoning benchmarks.",
    tags: ["Multimodal Models", "Synthetic Data"],
  },
  {
    title: "Eagle 2.5",
    link: "",
    date: "April 2025",
    description:
      "A family of vision-language models designed for long-context multimodal learning, addressing challenges in long video comprehension and high-resolution image understanding through a generalist framework. It incorporates Automatic Degrade Sampling, Image Area Preservation, efficiency optimizations, and the Eagle-Video-110K dataset, achieving substantial improvements on long-context multimodal benchmarks.",
    tags: ["Multimodal Models", "Nvidia"],
  },
  {
    title: "PerceptionLM",
    link: "",
    date: "April 2025",
    description:
      "An open and reproducible framework for image and video understanding research, addressing the limitations of closed-source VLMs and the issues with distillation from proprietary models.",
    tags: ["Multimodal Models"],
  },
];
