const data = [
  {
    title: "papers-explained",
    description:
      " A series of articles on Medium Explaining various Research Papers",
    actions: [
      {
        title: "View papers-explained Documentation",
        link: "/papers-explained",
      },
      {
        title: "View papers-explained on Medium",
        link: "https://ritvik19.medium.com/",
      },
    ],
  },
  {
    title: "implemented-data-science",
    description:
      "A collection of various implementations of various data science techniques and research papers." +
      "This project garnered attention from Elvis Saravia, (ex Cofounder of paperswithcode.com) and he added some of my implementations to his project ML-Notebooks",
    actions: [
      {
        title: "View implemented-data-science Documentation",
        link: "/implemented-data-science",
      },
      {
        title: "",
        link: "",
      },
      {
        title: "View implemented-data-science on GitHub",
        link: "https://github.com/Ritvik19/Implemented-Data-Science",
      },
      {
        title: "View ML-Notebooks",
        link: "https://github.com/dair-ai/ML-Notebooks",
      },
    ],
  },
  {
    title: "VidScripter",
    description: "A One Stop Solution to Video Transcription. It provides you with options to download, convert to audio, transcribe and summarize any video.",
    actions: [
      {
        title: "View VidScripter in Action",
        link: "https://huggingface.co/spaces/Ritvik19/VidScripter",
      }]
  },
  {
    title: "pyradox",
    description:
      "A library that helps with implementing various state of the art neural networks in a totally customizable fashion using TensorFlow 2." +
      "It is a high level API that provides you with a lot of flexibility and customizability. " +
      "It has two extensions pyradox-generative and pyradox-tabular. " +
      "pyradox-generative provides lLight weight trainers for various state of the art Ggenerative Adversarial Networks" +
      "pyradox-tabular provides implementations for various state of the art neural networks for tabular data",

    actions: [
      {
        title: "View pradox Documentation",
        link: "/pyradox",
      },
      {
        title: "View pyradox on GitHub",
        link: "https://github.com/Ritvik19/pyradox",
      },
      {
        title: "View pyradox-generative Documentation",
        link: "/pyradox-generative",
      },
      {
        title: "View pyradox-generative on GitHub",
        link: "https://github.com/Ritvik19/pyradox-generative",
      },
      {
        title: "View pyradox-tabular Documentation",
        link: "/pyradox-tabular",
      },
      {
        title: "View pyradox-tabular on GitHub",
        link: "https://github.com/Ritvik19/pyradox-tabular",
      },
    ],
  },
  {
    title: "text-data-augmentation",
    description:
      "A library that implements various State of the Art Text Data Augmentation Techniques for Natural Language Processing Applications",
    actions: [
      {
        title: "View text-data-augmentation Documentation",
        link: "/text-data-augmentation",
      },
      {
        title: "View text-data-augmentation on GitHub",
        link: "https://github.com/Ritvik19/text-data-augmentation",
      },
    ],
  },
  {
    title: "vizard",
    description:
      "A library that provides you with Low Code Data Visualisations for any Tabular Data Science Project",
    actions: [
      {
        title: "View vizard Documentation",
        link: "/vizard",
      },
      {
        title: "View vizard on GitHub",
        link: "https://github.com/Ritvik19/vizard",
      },
    ],
  },
  // {
  //   title: "ai-sudoku-solver",
  //   description:
  //     "A library that helps Solving Sudoku Puzzles using Artificial Neural Networks",
  // },
];

const skills = {
  "Python-Programming": 98,
  "Machine-Learning": 95,
  "Deep-Learning": 95,
  "Natural-Language-Processing": 95,
  "Computer-Vision": 92,
  "Web-Scraping": 95,
};

const models = [
  {
    title: "SudokuNet",
    description:
      "Sudoku-Net is a neural network model that solves sudoku puzzles. It was developed to see the performance of machine learning applications on solving sudokus.",
    linked_models: [
      [
        "sudoku-net-v1",
        "It is trained on 1 million Sudoku games dataset provided on kaggle by Kyubyong Park",
      ],
      [
        "sudoku-net-v2",
        "It is trained on 1 million Sudoku games dataset provided on kaggle by Kyubyong Park and 9 Million Sudoku Puzzles and Solutions dataset provided on kaggle by Vopani",
      ],
    ],
  },
  // {
  //   title: "SentiNet",
  //   description:
  //     "Senti-Net is an intelligent sentiment analyzer to thoroughly analyze the sentiments, emotions of a given text.",
  //   linked_models: [
  //     [
  //       "sentinet-v1",
  //       "The underlying algorithm is TF-IDF Vectorization followed by Logistic Regression",
  //     ],
  //   ],
  // },
  {
    title: "ScAi-Fi",
    description:
      "ScAi-Fi is a language model designed to assist you in comprehensively analyzing machine learning research paper abstracts. With its various functionalities, it simplifies understanding and extracting key information from academic abstracts.",
    linked_models: [
      ["ScAi-Fi", "A FLAN T5 fine tuned model on Papers Explained Data"],
    ],
  },
];
