const convolutional_neural_networks = [
  {
    title: "Lenet",
    link: "https://ritvik19.medium.com/78aeff61dcb3#4f26",
    date: "December 1998",
    description: "Introduced Convolutions.",
    tags: ["Convolutional Neural Networks"],
  },
  {
    title: "Alex Net",
    link: "https://ritvik19.medium.com/78aeff61dcb3#f7c6",
    date: "September 2012",
    description:
      "Introduced ReLU activation and Dropout to CNNs. Winner ILSVRC 2012.",
    tags: ["Convolutional Neural Networks"],
  },
  {
    title: "VGG",
    link: "https://ritvik19.medium.com/78aeff61dcb3#c122",
    date: "September 2014",
    description:
      "Used large number of filters of small size in each layer to learn complex features. Achieved SOTA in ILSVRC 2014.",
    tags: ["Convolutional Neural Networks"],
  },
  {
    title: "Inception Net",
    link: "https://ritvik19.medium.com/78aeff61dcb3#d7b3",
    date: "September 2014",
    description:
      "Introduced Inception Modules consisting of multiple parallel convolutional layers, designed to recognize different features at multiple scales.",
    tags: ["Convolutional Neural Networks"],
  },
  {
    title: "Inception Net v2 / Inception Net v3",
    link: "https://ritvik19.medium.com/78aeff61dcb3#d7b3",
    date: "December 2015",
    description:
      "Design Optimizations of the Inception Modules which improved performance and accuracy.",
    tags: ["Convolutional Neural Networks"],
  },
  {
    title: "Res Net",
    link: "https://ritvik19.medium.com/78aeff61dcb3#f761",
    date: "December 2015",
    description:
      "Introduced residual connections, which are shortcuts that bypass one or more layers in the network. Winner ILSVRC 2015.",
    tags: ["Convolutional Neural Networks"],
  },
  {
    title: "Inception Net v4 / Inception ResNet",
    link: "https://ritvik19.medium.com/78aeff61dcb3#83ad",
    date: "February 2016",
    description: "Hybrid approach combining Inception Net and ResNet.",
    tags: ["Convolutional Neural Networks"],
  },
  {
    title: "Dense Net",
    link: "https://ritvik19.medium.com/78aeff61dcb3#65e8",
    date: "August 2016",
    description:
      "Each layer receives input from all the previous layers, creating a dense network of connections between the layers, allowing to learn more diverse features.",
    tags: ["Convolutional Neural Networks"],
  },
  {
    title: "Xception",
    link: "https://ritvik19.medium.com/78aeff61dcb3#bc70",
    date: "October 2016",
    description:
      "Based on InceptionV3 but uses depthwise separable convolutions instead on inception modules.",
    tags: ["Convolutional Neural Networks"],
  },
  {
    title: "Res Next",
    link: "https://ritvik19.medium.com/78aeff61dcb3#90bd",
    date: "November 2016",
    description:
      "Built over ResNet, introduces the concept of grouped convolutions, where the filters in a convolutional layer are divided into multiple groups.",
    tags: ["Convolutional Neural Networks"],
  },
  {
    title: "Mobile Net V1",
    link: "https://ritvik19.medium.com/78aeff61dcb3#3cb5",
    date: "April 2017",
    description:
      "Uses depthwise separable convolutions to reduce the number of parameters and computation required.",
    tags: ["Convolutional Neural Networks"],
  },
  {
    title: "Mobile Net V2",
    link: "https://ritvik19.medium.com/78aeff61dcb3#4440",
    date: "January 2018",
    description:
      "Built upon the MobileNetv1 architecture, uses inverted residuals and linear bottlenecks.",
    tags: ["Convolutional Neural Networks"],
  },
  {
    title: "Mobile Net V3",
    link: "https://ritvik19.medium.com/78aeff61dcb3#8eb6",
    date: "May 2019",
    description:
      "Uses AutoML to find the best possible neural network architecture for a given problem.",
    tags: ["Convolutional Neural Networks"],
  },
  {
    title: "Efficient Net",
    link: "https://ritvik19.medium.com/78aeff61dcb3#560a",
    date: "May 2019",
    description:
      "Uses a compound scaling method to scale the network's depth, width, and resolution to achieve a high accuracy with a relatively low computational cost.",
    tags: ["Convolutional Neural Networks"],
  },
  {
    title: "NF Net",
    link: "https://ritvik19.medium.com/b8efa03d6b26",
    date: "February 2021",
    description:
      "An improved class of Normalizer-Free ResNets that implement batch-normalized networks, offer faster training times, and introduce an adaptive gradient clipping technique to overcome instabilities associated with deep ResNets.",
    tags: ["Convolutional Neural Networks"],
  },
  {
    title: "Conv Mixer",
    link: "https://ritvik19.medium.com/f073f0356526",
    date: "January 2022",
    description:
      "Processes image patches using standard convolutions for mixing spatial and channel dimensions.",
    tags: ["Convolutional Neural Networks"],
  },
  {
    title: "ConvNeXt",
    link: "https://ritvik19.medium.com/d13385d9177d",
    date: "January 2022",
    description:
      "A pure ConvNet model, evolved from standard ResNet design, that competes well with Transformers in accuracy and scalability.",
    tags: ["Convolutional Neural Networks"],
  },
  {
    title: "ConvNeXt V2",
    link: "https://ritvik19.medium.com/2ecdabf2081c",
    date: "January 2023",
    description:
      "Incorporates a fully convolutional MAE framework and a Global Response Normalization (GRN) layer, boosting performance across multiple benchmarks.",
    tags: ["Convolutional Neural Networks"],
  },
  {
    title: "Efficient Net V2",
    link: "https://ritvik19.medium.com/a7a1e4113b89",
    date: "April 2024",
    description:
      "A new family of convolutional networks, achieves faster training speed and better parameter efficiency than previous models through neural architecture search and scaling, with progressive learning allowing for improved accuracy on various datasets while training up to 11x faster.",
  },
  {
    title: "Mobile Net V4",
    link: "https://ritvik19.medium.com/83a526887c30",
    date: "April 2024",
    description:
      "Features a universally efficient architecture design, including the Universal Inverted Bottleneck (UIB) search block, Mobile MQA attention block, and an optimized neural architecture search recipe, which enables it to achieve high accuracy and efficiency on various mobile devices and accelerators.",
    tags: ["Convolutional Neural Networks"],
  },
  {
    title: "LS Net",
    link: "https://ritvik19.medium.com/22fc08fec1ae",
    date: "March 2025",
    description:
      "Mimics the human vision system's 'See Large, Focus Small' strategy by using a novel Large-Small (LS) convolution that combines large-kernel static convolution for broad contextual perception and small-kernel dynamic convolution with a group mechanism for precise, adaptive feature aggregation within a small visual field.",
    tags: ["Convolutional Neural Networks"],
  },
];
