const nav_data = [
  "Language Models",
  "Vision Models",
  "Convolutional Neural Networks",
  "Single Stage Object Detection",
  "Region Based Convolutional Neural Networks",
  "Document Understanding",
  "Layout Aware Language Models",
  "Tabular Data",
  "Miscellaneous",
];

const papers_data = [
  [
    // Language Models
    {
      title: "Transformer",
      link: "https://ritvik19.medium.com/papers-explained-01-transformer-474bb60a33f7",
      date: "June 2017",
      description:
        "An Encoder Decoder model, that introduced multihead attention mechanism for language translation task.",
    },
    {
      title: "Elmo",
      link: "https://ritvik19.medium.com/papers-explained-33-elmo-76362a43e4",
      date: "February 2018",
      description:
        "Deep contextualized word representations that captures both intricate aspects of word usage and contextual variations across language contexts.",
    },
    {
      title: "GPT",
      link: "https://ritvik19.medium.com/papers-explained-43-gpt-30b6f1e6d226",
      date: "June 2018",
      description:
        "A Decoder only transformer which is autoregressively pretrained and then finetuned for specific downstream tasks using task-aware input transformations.",
    },
    {
      title: "BERT",
      link: "https://ritvik19.medium.com/papers-explained-02-bert-31e59abc0615",
      date: "October 2018",
      description:
        "Introduced pre-training for Encoder Transformers. Uses unified architecture across different tasks.",
    },
    {
      title: "Transformer XL",
      link: "https://ritvik19.medium.com/papers-explained-34-transformerxl-2e407e780e8",
      date: "January 2019",
      description:
        "Extends the original Transformer model to handle longer sequences of text by introducing recurrence into the self-attention mechanism.",
    },
    {
      title: "GPT 2",
      link: "https://ritvik19.medium.com/papers-explained-65-gpt-2-98d0a642e520",
      date: "February 2019",
      description:
        "Demonstrates that language models begin to learn various language processing tasks without any explicit supervision.",
    },
    {
      title: "UniLM",
      link: "https://ritvik19.medium.com/papers-explained-72-unilm-672f0ecc6a4a",
      date: "May 2019",
      description:
        "Utilizes a shared Transformer network and specific self-attention masks to excel in both language understanding and generation tasks.",
    },
    {
      title: "XLNet",
      link: "https://ritvik19.medium.com/papers-explained-35-xlnet-ea0c3af96d49",
      date: "June 2019",
      description:
        "Extension of the Transformer-XL, pre-trained using a new method that combines ideas from AR and AE objectives.",
    },
    {
      title: "RoBERTa",
      link: "https://ritvik19.medium.com/papers-explained-03-roberta-81db014e35b9",
      date: "July 2019",
      description:
        "Built upon BERT, by carefully optimizing hyperparameters and training data size to improve performance on various language tasks .",
    },
    {
      title: "Sentence BERT",
      link: "https://ritvik19.medium.com/papers-explained-04-sentence-bert-5159b8e07f21",
      date: "August 2019",
      description:
        "A modification of BERT that uses siamese and triplet network structures to derive sentence embeddings that can be compared using cosine-similarity.",
    },
    {
      title: "Tiny BERT",
      link: "https://ritvik19.medium.com/papers-explained-05-tiny-bert-5e36fe0ee173",
      date: "September 2019",
      description:
        "Uses attention transfer, and task specific distillation for distilling BERT.",
    },
    {
      title: "ALBERT",
      link: "https://ritvik19.medium.com/papers-explained-07-albert-46a2a0563693",
      date: "September 2019",
      description:
        "Presents certain parameter reduction techniques to lower memory consumption and increase the training speed of BERT.",
    },
    {
      title: "Distil BERT",
      link: "https://ritvik19.medium.com/papers-explained-06-distil-bert-6f138849f871",
      date: "October 2019",
      description:
        "Distills BERT on very large batches leveraging gradient accumulation, using dynamic masking and without the next sentence prediction objective.",
    },
    {
      title: "T5",
      link: "https://ritvik19.medium.com/papers-explained-44-t5-9d974a3b7957",
      date: "October 2019",
      description:
        "A unified encoder-decoder framework that converts all text-based language problems into a text-to-text format.",
    },
    {
      title: "BART",
      link: "https://ritvik19.medium.com/papers-explained-09-bart-7f56138175bd",
      date: "October 2019",
      description:
        "A Decoder pretrained to reconstruct the original text from corrupted versions of it.",
    },
    {
      title: "UniLMv2",
      link: "https://ritvik19.medium.com/papers-explained-unilmv2-5a044ca7c525",
      date: "February 2020",
      description:
        "Utilizes a pseudo-masked language model (PMLM) for both autoencoding and partially autoregressive language modeling tasks,significantly advancing the capabilities of language models in diverse NLP tasks.",
    },
    {
      title: "FastBERT",
      link: "https://ritvik19.medium.com/papers-explained-37-fastbert-5bd246c1b432",
      date: "April 2020",
      description:
        "A speed-tunable encoder with adaptive inference time having branches at each transformer output to enable early outputs.",
    },
    {
      title: "MobileBERT",
      link: "https://ritvik19.medium.com/papers-explained-36-mobilebert-933abbd5aaf1",
      date: "April 2020",
      description:
        "Compressed and faster version of the BERT, featuring bottleneck structures, optimized attention mechanisms, and knowledge transfer.",
    },
    {
      title: "Longformer",
      link: "https://ritvik19.medium.com/papers-explained-38-longformer-9a08416c532e",
      date: "April 2020",
      description:
        "Introduces a linearly scalable attention mechanism, allowing handling texts of exteded length.",
    },
    {
      title: "GPT 3",
      link: "https://ritvik19.medium.com/papers-explained-66-gpt-3-352f5a1b397",
      date: "May 2020",
      description:
        "Demonstrates that scaling up language models greatly improves task-agnostic, few-shot performance.",
    },
    {
      title: "DeBERTa",
      link: "https://ritvik19.medium.com/papers-explained-08-deberta-a808d9b2c52d",
      date: "June 2020",
      description:
        "Enhances BERT and RoBERTa through disentangled attention mechanisms, an enhanced mask decoder, and virtual adversarial training.",
    },
    {
      title: "DeBERTa v2",
      link: "https://ritvik19.medium.com/papers-explained-08-deberta-a808d9b2c52d#f5e1",
      date: "June 2020",
      description:
        "Enhanced version of the DeBERTa featuring a new vocabulary, nGiE integration, optimized attention mechanisms, additional model sizes, and improved tokenization.",
    },
    {
      title: "Codex",
      link: "https://ritvik19.medium.com/papers-explained-45-codex-caca940feb31",
      date: "July 2021",
      description:
        "A GPT language model finetuned on publicly available code from GitHub.",
    },
    {
      title: "FLAN",
      link: "https://ritvik19.medium.com/papers-explained-46-flan-1c5e0d5db7c9",
      date: "September 2021",
      description:
        "An instruction-tuned language model developed through finetuning on various NLP datasets described by natural language instructions.",
    },
    {
      title: "Gopher",
      link: "https://ritvik19.medium.com/papers-explained-47-gopher-2e71bbef9e87",
      date: "December 2021",
      description:
        "Provides a comprehensive analysis of the performance of various Transformer models across different scales upto 280B on 152 tasks.",
    },
    {
      title: "Instruct GPT",
      link: "https://ritvik19.medium.com/papers-explained-48-instructgpt-e9bcd51f03ec",
      date: "March 2022",
      description:
        "Fine-tuned GPT using supervised learning (instruction tuning) and reinforcement learning from human feedback to align with user intent.",
    },
    {
      title: "Chinchilla",
      link: "https://ritvik19.medium.com/papers-explained-49-chinchilla-a7ad826d945e",
      date: "March 2022",
      description:
        "Investigated the optimal model size and number of tokens for training a transformer LLM within a given compute budget (Scaling Laws).",
    },
    {
      title: "PaLM",
      link: "https://ritvik19.medium.com/papers-explained-50-palm-480e72fa3fd5",
      date: "April 2022",
      description:
        "A 540-B parameter, densely activated, Transformer, trained using Pathways, (ML system that enables highly efficient training across multiple TPU Pods).",
    },
    {
      title: "OPT",
      link: "https://ritvik19.medium.com/papers-explained-51-opt-dacd9406e2bd",
      date: "May 2022",
      description:
        "A suite of decoder-only pre-trained transformers with parameter ranges from 125M to 175B. OPT-175B being comparable to GPT-3.",
    },
    {
      title: "BLOOM",
      link: "https://ritvik19.medium.com/papers-explained-52-bloom-9654c56cd2",
      date: "November 2022",
      description:
        "A 176B-parameter open-access decoder-only transformer, collaboratively developed by hundreds of researchers, aiming to democratize LLM technology.",
    },
    {
      title: "Galactica",
      link: "https://ritvik19.medium.com/papers-explained-53-galactica-1308dbd318dc",
      date: "November 2022",
      description:
        "An LLM trained on scientific data thus specializing in scientific knowledge.",
    },
    {
      title: "ChatGPT",
      link: "https://ritvik19.medium.com/papers-explained-54-chatgpt-78387333268f",
      date: "November 2022",
      description:
        "An interactive model designed to engage in conversations, built on top of GPT 3.5.",
    },
    {
      title: "LLaMA",
      link: "https://ritvik19.medium.com/papers-explained-55-llama-c4f302809d6b",
      date: "February 2023",
      description:
        "A collection of foundation LLMs by Meta ranging from 7B to 65B parameters, trained using publicly available datasets exclusively.",
    },
    {
      title: "Alpaca",
      link: "https://ritvik19.medium.com/papers-explained-56-alpaca-933c4d9855e5",
      date: "March 2023",
      description:
        "A fine-tuned LLaMA 7B model, trained on instruction-following demonstrations generated in the style of self-instruct using text-davinci-003.",
    },
    {
      title: "GPT 4",
      link: "https://ritvik19.medium.com/papers-explained-67-gpt-4-fc77069b613e",
      date: "March 2023",
      description:
        "A multimodal transformer model pre-trained to predict the next token in a document, which can accept image and text inputs and produce text outputs.",
    },
    {
      title: "PaLM 2",
      link: "https://ritvik19.medium.com/papers-explained-58-palm-2-1a9a23f20d6c",
      date: "May 2023",
      description:
        "Successor of PALM, trained on a mixture of different pre-training objectives in order to understand different aspects of language.",
    },
    {
      title: "LIMA",
      link: "https://ritvik19.medium.com/papers-explained-57-lima-f9401a5760c3",
      date: "May 2023",
      description:
        "A LLaMa model fine-tuned on only 1,000 carefully curated prompts and responses, without any reinforcement learning or human preference modeling.",
    },
    {
      title: "Falcon",
      link: "https://ritvik19.medium.com/papers-explained-59-falcon-26831087247f",
      date: "June 2023",
      description:
        "An Open Source LLM trained on properly filtered and deduplicated web data alone.",
    },
    {
      title: "LLaMA 2",
      link: "https://ritvik19.medium.com/papers-explained-60-llama-v2-3e415c5b9b17",
      date: "July 2023",
      description:
        "Successor of LLaMA. LLaMA 2-Chat is optimized for dialogue use cases.",
    },
    {
      title: "Humpback",
      link: "https://ritvik19.medium.com/papers-explained-61-humpback-46992374fc34",
      date: "August 2023",
      description: "LLaMA finetuned using Instrustion backtranslation.",
    },
    {
      title: "Code LLaMA",
      link: "https://ritvik19.medium.com/papers-explained-62-code-llama-ee266bfa495f",
      date: "August 2023",
      description: "LLaMA 2 based LLM for code.",
    },
    {
      title: "GPT-4V",
      link: "https://ritvik19.medium.com/papers-explained-68-gpt-4v-6e27c8a1d6ea",
      date: "September 2023",
      description:
        "A multimodal model that combines text and vision capabilities, allowing users to instruct it to analyze image inputs.",
    },
    {
      title: "LLaMA 2 Long",
      link: "https://ritvik19.medium.com/papers-explained-63-llama-2-long-84d33c26d14a",
      date: "September 2023",
      description:
        "A series of long context LLMs s that support effective context windows of up to 32,768 tokens.",
    },
    {
      title: "Mistral 7B",
      link: "https://ritvik19.medium.com/papers-explained-mistral-7b-b9632dedf580",
      date: "October 2023",
      description:
        "Leverages grouped-query attention for faster inference, coupled with sliding window attention to effectively handle sequences of arbitrary length with a reduced inference cost.",
    },
    {
      title: "Llemma",
      link: "https://ritvik19.medium.com/papers-explained-69-llemma-0a17287e890a",
      date: "October 2023",
      description:
        "An LLM for mathematics, formed by continued pretraining of Code Llama on a mixture of scientific papers, web data containing mathematics, and mathematical code.",
    },
    {
      title: "CodeFusion",
      link: "https://ritvik19.medium.com/papers-explained-70-codefusion-fee6aba0149a",
      date: "October 2023",
      description:
        "A diffusion code generation model that iteratively refines entire programs based on encoded natural language, overcoming the limitation of auto-regressive models in code generation by allowing reconsideration of earlier tokens.",
    },
    {
      title: "Zephyr 7B",
      link: "https://ritvik19.medium.com/papers-explained-71-zephyr-7ec068e2f20b",
      date: "October 2023",
      description:
        "Utilizes dDPO and AI Feedback (AIF) preference data to achieve superior intent alignment in chat-based language modeling.",
    },
  ],
  [
    // Vision Models
    {
      title: "Vision Transformer",
      link: "https://ritvik19.medium.com/papers-explained-25-vision-transformers-e286ee8bc06b",
      date: "October 2020",
      description:
        "Images are segmented into patches, which are treated as tokens and a sequence of linear embeddings of these patches are input to a Transformer",
    },
    {
      title: "DeiT",
      link: "https://ritvik19.medium.com/papers-explained-39-deit-3d78dd98c8ec",
      date: "December 2020",
      description:
        "A convolution-free vision transformer that uses a teacher-student strategy with attention-based distillation tokens.",
    },
    {
      title: "Swin Transformer",
      link: "https://ritvik19.medium.com/papers-explained-26-swin-transformer-39cf88b00e3e",
      date: "March 2021",
      description:
        "A hierarchical vision transformer that uses shifted windows to addresses the challenges of adapting the transformer model to computer vision.",
    },
    {
      title: "BEiT",
      link: "https://ritvik19.medium.com/papers-explained-27-beit-b8c225496c01",
      date: "June 2021",
      description:
        "Utilizes a masked image modeling task inspired by BERT in, involving image patches and visual tokens to pretrain vision Transformers.",
    },
    {
      title: "MobileViT",
      link: "https://ritvik19.medium.com/papers-explained-40-mobilevit-4793f149c434",
      date: "October 2021",
      description:
        "A lightweight vision transformer designed for mobile devices, effectively combining the strengths of CNNs and ViTs.",
    },
    {
      title: "Masked AutoEncoder",
      link: "https://ritvik19.medium.com/papers-explained-28-masked-autoencoder-38cb0dbed4af",
      date: "November 2021",
      description:
        "An encoder-decoder architecture that reconstructs input images by masking random patches and leveraging a high proportion of masking for self-supervision.",
    },
  ],
  [
    // Convolutional Neural Networks
    {
      title: "Lenet",
      link: "https://ritvik19.medium.com/papers-explained-review-01-convolutional-neural-networks-78aeff61dcb3#4f26",
      date: "December 1998",
      description: "Introduced Convolutions.",
    },
    {
      title: "Alex Net",
      link: "https://ritvik19.medium.com/papers-explained-review-01-convolutional-neural-networks-78aeff61dcb3#f7c6",
      date: "September 2012",
      description:
        "Introduced ReLU activation and Dropout to CNNs. Winner ILSVRC 2012.",
    },
    {
      title: "VGG",
      link: "https://ritvik19.medium.com/papers-explained-review-01-convolutional-neural-networks-78aeff61dcb3#c122",
      date: "September 2014",
      description:
        "Used large number of filters of small size in each layer to learn complex features. Achieved SOTA in ILSVRC 2014.",
    },
    {
      title: "Inception Net",
      link: "https://ritvik19.medium.com/papers-explained-review-01-convolutional-neural-networks-78aeff61dcb3#d7b3",
      date: "September 2014",
      description:
        "Introduced Inception Modules consisting of multiple parallel convolutional layers, designed to recognize different features at multiple scales.",
    },
    {
      title: "Inception Net v2 / Inception Net v3",
      link: "https://ritvik19.medium.com/papers-explained-review-01-convolutional-neural-networks-78aeff61dcb3#d7b3",
      date: "December 2015",
      description:
        "Design Optimizations of the Inception Modules which improved performance and accuracy.",
    },
    {
      title: "Res Net",
      link: "https://ritvik19.medium.com/papers-explained-review-01-convolutional-neural-networks-78aeff61dcb3#f761",
      date: "December 2015",
      description:
        "Introduced residual connections, which are shortcuts that bypass one or more layers in the network. Winner ILSVRC 2015.",
    },
    {
      title: "Inception Net v4 / Inception ResNet",
      link: "https://ritvik19.medium.com/papers-explained-review-01-convolutional-neural-networks-78aeff61dcb3#83ad",
      date: "February 2016",
      description: "Hybrid approach combining Inception Net and ResNet.",
    },
    {
      title: "Dense Net",
      link: "https://ritvik19.medium.com/papers-explained-review-01-convolutional-neural-networks-78aeff61dcb3#65e8",
      date: "August 2016",
      description:
        "Each layer receives input from all the previous layers, creating a dense network of connections between the layers, allowing to learn more diverse features.",
    },
    {
      title: "Xception",
      link: "https://ritvik19.medium.com/papers-explained-review-01-convolutional-neural-networks-78aeff61dcb3#bc70",
      date: "October 2016",
      description:
        "Based on InceptionV3 but uses depthwise separable convolutions instead on inception modules.",
    },
    {
      title: "Res Next",
      link: "https://ritvik19.medium.com/papers-explained-review-01-convolutional-neural-networks-78aeff61dcb3#90bd",
      date: "November 2016",
      description:
        "Built over ResNet, introduces the concept of grouped convolutions, where the filters in a convolutional layer are divided into multiple groups.",
    },
    {
      title: "Mobile Net V1",
      link: "https://ritvik19.medium.com/papers-explained-review-01-convolutional-neural-networks-78aeff61dcb3#3cb5",
      date: "April 2017",
      description:
        "Uses depthwise separable convolutions to reduce the number of parameters and computation required.",
    },
    {
      title: "Mobile Net V2",
      link: "https://ritvik19.medium.com/papers-explained-review-01-convolutional-neural-networks-78aeff61dcb3#4440",
      date: "January 2018",
      description:
        "Built upon the MobileNetv1 architecture, uses inverted residuals and linear bottlenecks.",
    },
    {
      title: "Mobile Net V3",
      link: "https://ritvik19.medium.com/papers-explained-review-01-convolutional-neural-networks-78aeff61dcb3#8eb6",
      date: "May 2019",
      description:
        "Uses AutoML to find the best possible neural network architecture for a given problem.",
    },
    {
      title: "Efficient Net",
      link: "https://ritvik19.medium.com/papers-explained-review-01-convolutional-neural-networks-78aeff61dcb3#560a",
      date: "May 2019",
      description:
        "Uses a compound scaling method to scale the network's depth, width, and resolution to achieve a high accuracy with a relatively low computational cost.",
    },
    {
      title: "Conv Mixer",
      link: "https://ritvik19.medium.com/papers-explained-29-convmixer-f073f0356526",
      date: "January 2022",
      description:
        "Processes image patches using standard convolutions for mixing spatial and channel dimensions.",
    },
  ],
  [
    // Single Stage Object Detection
    {
      title: "SSD",
      link: "https://ritvik19.medium.com/papers-explained-31-single-shot-multibox-detector-14b0aa2f5a97",
      date: "December 2015",
      description:
        "Discretizes bounding box outputs over a span of various scales and aspect ratios per feature map.",
    },
    {
      title: "Feature Pyramid Network",
      link: "https://ritvik19.medium.com/papers-explained-21-feature-pyramid-network-6baebcb7e4b8",
      date: "December 2016",
      description:
        "Leverages the inherent multi-scale hierarchy of deep convolutional networks to efficiently construct feature pyramids.",
    },
    {
      title: "Focal Loss",
      link: "https://ritvik19.medium.com/papers-explained-22-focal-loss-for-dense-object-detection-retinanet-733b70ce0cb1",
      date: "August 2017",
      description:
        "Addresses class imbalance in dense object detectors by down-weighting the loss assigned to well-classified examples.",
    },
  ],
  [
    // Region Based Convolutional Neural Networks
    {
      title: "RCNN",
      link: "https://ritvik19.medium.com/papers-explained-14-rcnn-ede4db2de0ab",
      date: "November 2013",
      description:
        "Uses selective search for region proposals, CNNs for feature extraction, SVM for classification followed by box offset regression.",
    },
    {
      title: "Fast RCNN",
      link: "https://ritvik19.medium.com/papers-explained-15-fast-rcnn-28c1792dcee0",
      date: "April 2015",
      description:
        "Processes entire image through CNN, employs RoI Pooling to extract feature vectors from ROIs, followed by classification and BBox regression.",
    },
    {
      title: "Faster RCNN",
      link: "https://ritvik19.medium.com/papers-explained-16-faster-rcnn-a7b874ffacd9",
      date: "June 2015",
      description:
        "A region proposal network (RPN) and a Fast R-CNN detector, collaboratively predict object regions by sharing convolutional features.",
    },
    {
      title: "Mask RCNN",
      link: "https://ritvik19.medium.com/papers-explained-17-mask-rcnn-82c64bea5261",
      date: "March 2017",
      description:
        "Extends Faster R-CNN to solve instance segmentation tasks, by adding a branch for predicting an object mask in parallel with the existing branch.",
    },
  ],
  [
    // Document Understanding
    {
      title: "Table Net",
      link: "https://ritvik19.medium.com/papers-explained-18-tablenet-3d4c62269bb3",
      date: "January 2020",
      description:
        "An end-to-end deep learning model designed for both table detection and structure recognition.",
    },
    {
      title: "Donut",
      link: "https://ritvik19.medium.com/papers-explained-20-donut-cb1523bf3281",
      date: "November 2021",
      description:
        "An OCR-free Encoder-Decoder Transformer model. The encoder takes in images, decoder takes in prompts & encoded images to generate the required text.",
    },
    {
      title: "DiT",
      link: "https://ritvik19.medium.com/papers-explained-19-dit-b6d6eccd8c4e",
      date: "March 2022",
      description:
        "An Image Transformer pre-trained (self-supervised) on document images",
    },
    {
      title: "UDoP",
      link: "https://ritvik19.medium.com/papers-explained-42-udop-719732358ab4",
      date: "December 2022",
      description:
        "Integrates text, image, and layout information through a Vision-Text-Layout Transformer, enabling unified representation.",
    },
  ],
  [
    // Layout Aware Language Models
    {
      title: "Layout LM",
      link: "https://ritvik19.medium.com/papers-explained-10-layout-lm-32ec4bad6406",
      date: "December 2019",
      description:
        "Utilises BERT as the backbone, adds two new input embeddings: 2-D position embedding and image embedding (Only for downstream tasks).",
    },
    {
      title: "LamBERT",
      link: "https://ritvik19.medium.com/papers-explained-41-lambert-8f52d28f20d9",
      date: "February 2020",
      description:
        "Utilises RoBERTa as the backbone and adds Layout embeddings along with relative bias.",
    },
    {
      title: "Layout LM v2",
      link: "https://ritvik19.medium.com/papers-explained-11-layout-lm-v2-9531a983e659",
      date: "December 2020",
      description:
        "Uses a multi-modal Transformer model, to integrate text, layout, and image in the pre-training stage, to learn end-to-end cross-modal interaction.",
    },
    {
      title: "Structural LM",
      link: "https://ritvik19.medium.com/papers-explained-23-structural-lm-36e9df91e7c1",
      date: "May 2021",
      description:
        "Utilises BERT as the backbone and feeds text, 1D and (2D cell level) embeddings to the transformer model.",
    },
    {
      title: "Doc Former",
      link: "https://ritvik19.medium.com/papers-explained-30-docformer-228ce27182a0",
      date: "June 2021",
      description:
        "Encoder-only transformer with a CNN backbone for visual feature extraction, combines text, vision, and spatial features through a multi-modal self-attention layer.",
    },
    {
      title: "LiLT",
      link: "https://ritvik19.medium.com/papers-explained-12-lilt-701057ec6d9e",
      date: "February 2022",
      description:
        "Introduced Bi-directional attention complementation mechanism (BiACM) to accomplish the cross-modal interaction of text and layout.",
    },
    {
      title: "Layout LM V3",
      link: "https://ritvik19.medium.com/papers-explained-13-layout-lm-v3-3b54910173aa",
      date: "April 2022",
      description:
        "A unified text-image multimodal Transformer to learn cross-modal representations, that imputs concatenation of text embedding and image embedding.",
    },
    {
      title: "ERNIE Layout",
      link: "https://ritvik19.medium.com/papers-explained-24-ernie-layout-47a5a38e321b",
      date: "October 2022",
      description:
        "Reorganizes tokens using layout information, combines text and visual embeddings, utilizes multi-modal transformers with spatial aware disentangled attention.",
    },
  ],
  [
    // Tabular Data
    {
      title: "Entity Embeddings",
      link: "https://ritvik19.medium.com/papers-explained-review-04-tabular-deep-learning-776db04f965b#932e",
      date: "April 2016",
      description:
        "Maps categorical variables into continuous vector spaces through neural network learning, revealing intrinsic properties.",
    },
    {
      title: "Wide and Deep Learning",
      link: "https://ritvik19.medium.com/papers-explained-review-04-tabular-deep-learning-776db04f965b#bfdc",
      date: "June 2016",
      description:
        "Combines memorization of specific patterns with generalization of similarities.",
    },
    {
      title: "Deep and Cross Network",
      link: "https://ritvik19.medium.com/papers-explained-review-04-tabular-deep-learning-776db04f965b#0017",
      date: "August 2017",
      description:
        "Combines the  a novel cross network with deep neural networks (DNNs) to efficiently learn feature interactions without manual feature engineering.",
    },
    {
      title: "Tab Transformer",
      link: "https://ritvik19.medium.com/papers-explained-review-04-tabular-deep-learning-776db04f965b#48c4",
      date: "December 2020",
      description:
        "Employs multi-head attention-based Transformer layers to convert categorical feature embeddings into robust contextual embeddings.",
    },
    {
      title: "Tabular ResNet",
      link: "https://ritvik19.medium.com/papers-explained-review-04-tabular-deep-learning-776db04f965b#46af",
      date: "June 2021",
      description: "An MLP with skip connections.",
    },
    {
      title: "Feature Tokenizer Transformer",
      link: "https://ritvik19.medium.com/papers-explained-review-04-tabular-deep-learning-776db04f965b#1ab8",
      date: "June 2021",
      description:
        "Transforms all features (categorical and numerical) to embeddings and applies a stack of Transformer layers to the embeddings.",
    },
  ],
  [
    // Miscellaneous
    {
      title: "ColD Fusion",
      link: "https://ritvik19.medium.com/papers-explained-32-cold-fusion-452f33101a91",
      date: "December 2022",
      description:
        "A method enabling the benefits of multitask learning through distributed computation without data sharing and improving model performance.",
    },
  ],
];

const literature_review_data = [
  {
    title: "Convolutional Neural Networks",
    link: "https://ritvik19.medium.com/papers-explained-review-01-convolutional-neural-networks-78aeff61dcb3",
  },
  {
    title: "Layout Transformers",
    link: "https://ritvik19.medium.com/papers-explained-review-02-layout-transformers-b2d165c94ad5",
  },
  {
    title: "Region Based Convolutional Neural Networks",
    link: "https://ritvik19.medium.com/papers-explained-review-03-rcnns-42c0a3974493",
  },
  {
    title: "Tabular Deep Learning",
    link: "https://ritvik19.medium.com/papers-explained-review-04-tabular-deep-learning-776db04f965b",
  },
];
