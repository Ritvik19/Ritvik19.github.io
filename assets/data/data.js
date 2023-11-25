const data = [
  {
    title: "pyradox",
    description:
      "A library that helps you with implementing various state of the art neural networks in a totally customizable fashion using TensorFlow 2",
  },
  {
    title: "pyradox-generative",
    description:
      "A library that provides Light Weight Trainers For The State Of The Art Generative Neural Networks Using A TensorFlow Backend",
  },
  {
    title: "pyradox-tabular",
    description:
      "A library that helps you train various state of the art neural networks for tabular data using TensorFlow 2",
  },
  {
    title: "text-data-augmentation",
    description:
      "A library that implements various State of the Art Text Data Augmentation Techniques for Natural Language Processing Applications",
  },
  {
    title: "vizard",
    description:
      "A library that provides you with Low Code Data Visualisations for any Tabular Data Science Project",
  },
  // {
  //   title: "ai-sudoku-solver",
  //   description:
  //     "A library that helps Solving Sudoku Puzzles using Artificial Neural Networks",
  // },
  {
    title: "implemented-data-science",
    description:
      "This Library of Implementations of various data science techniques and research papers",
  },
  {
    title: "ml-notebooks",
    description:
      "A project in collaboration with Elvis Saravia (founder, paperswithcode), containing a series of code examples for all sorts of machine learning tasks and applications",
    github: "dair-ai",
    no_documentation: true,
  },
  {
    title: "papers-explained",
    description:
      " A blog series on Kaggle Explaining various Research Papers followed by their implementation",
    no_github: true,
  },
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
