const nav_data = [
  "Abstractive Summarization",
  "Back Translation",
  "Character Noise",
  "Contextual Word Replacement",
  "Easy Data Augmentation",
  "KeyBoard Noise",
  "OCR Noise",
  "Paraphrase",
  "Similar Word Replacement",
  "Synonym Replacement",
  "Word Split",
];

const references = [
  [
    "Data Expansion Using Back Translation and Paraphrasing for Hate Speech Detection",
    "https://arxiv.org/pdf/2106.04681.pdf",
  ],
  [
    "A Survey on Data Augmentation for Text Classification",
    "https://arxiv.org/ftp/arxiv/papers/2107/2107.03158.pdf",
  ],
  [
    "Contextual Augmentation: Data Augmentation by Words with Paradigmatic Relations",
    "https://arxiv.org/pdf/1805.06201.pdf",
  ],
  [
    "EDA: Easy Data Augmentation Techniques for Boosting Performance on Text Classification Tasks",
    "https://arxiv.org/pdf/1901.11196.pdf",
  ],
  [
    "An Analysis of Simple Data Augmentation for Named Entity Recognition",
    "https://aclanthology.org/2020.coling-main.343.pdf",
  ],
  [
    "Deep Statistical Analysis of OCR Errors for Effective Post-OCR Processing",
    "https://zenodo.org/record/3245169/files/JCDL2019_Deep_Analysis.pdf",
  ],
  [
    "A Study of Various Text Augmentation Techniques for Relation Classification in Free Text",
    "https://www.researchgate.net/publication/331784439_A_Study_of_Various_Text_Augmentation_Techniques_for_Relation_Classification_in_Free_Text",
  ],
  [
    "Text Augmentation for Neural Networks",
    "http://ceur-ws.org/Vol-2268/paper11.pdf",
  ],
  [
    "Synthetic And Natural Noise Both Break Neural Machine Translation",
    "https://arxiv.org/pdf/1711.02173.pdf",
  ],
  [
    "Improving Neural Machine Translation Models with Monolingual Data",
    "https://arxiv.org/pdf/1511.06709.pdf",
  ],
  [
    "Data Augmentation Using Pre-trained Transformer Models",
    "https://arxiv.org/pdf/2003.02245.pdf",
  ],
  [
    "Data Augmentation via Dependency Tree Morphing for Low-Resource Languages",
    "https://arxiv.org/pdf/1903.09460.pdf",
  ],
  [
    "Adversarial Over-Sensitivity and Over-Stability Strategies for Dialogue Models",
    "https://arxiv.org/pdf/1809.02079.pdf",
  ],
  [
    "TextBugger: Generating Adversarial Text Against Real-world Applications",
    "https://arxiv.org/pdf/1812.05271v1.pdf",
  ],
  [
    "Generating Natural Language Adversarial Examples",
    "https://arxiv.org/pdf/1804.07998.pdf",
  ],
  [
    "Character-level Convolutional Networks for Text Classification",
    "https://arxiv.org/pdf/1509.01626.pdf",
  ],
  [
    "Neural Abstractive Text Summarization with Sequence-to-Sequence Models",
    "https://arxiv.org/pdf/1812.02303.pdf",
  ],
  [
    "BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension",
    "https://arxiv.org/pdf/1910.13461v1.pdf",
  ],
  [
    "Unsupervised Data Augmentation for Consistency Training",
    "https://arxiv.org/pdf/1904.12848.pdf",
  ],
  [
    "Text Data Augmentation: Towards better detection of spear-phishing emails",
    "https://arxiv.org/pdf/2007.02033.pdf",
  ],
];

const usage = [
  {
    title: "Abstractive Summarization",
    content: [
      {
        type: "p",
        text: 'Abstractive Summarization Augmentation summarizes the model using transformer models. <a href="#ref-17">[17]</a> <a href="#ref-18">[18]</a>',
      },
      {
        type: "code",
        text: `>>> from text_data_augmentation import AbstractiveSummarization
>>> aug = AbstractiveSummarization()
>>> aug(['Abstractive Summarization is a task in Natural Language Processing (NLP) that aims to generate a concise summary of a source text. Unlike extractive summarization, abstractive summarization does not simply copy important phrases from the source text but also potentially come up with new phrases that are relevant, which can be seen as paraphrasing. Abstractive summarization yields a number of applications in different domains, from books and literature, to science and R&D, to financial research and legal documents analysis.'])
['Abstractive Summarization is a task in Natural Language Processing (NLP) that aims to generate a concise summary of a source text. Unlike extractive summarization, abstractive summarization does not simply copy important phrases from the source text but also potentially come up with new phrases that are relevant, which can be seen as paraphrasing. Abstractive summarization yields a number of applications in different domains, from books and literature, to science and R&D, to financial research and legal documents analysis.', 'Abstractive Summarization is a task in Natural Language Processing (NLP) that aims to generate a concise summary of a source text . Unlike extractive summarization, it does not copy important phrases from the source text but also potentially come up with new phrases thatare relevant, which can be seen as paraphrasing .']`,
      },
    ],
  },
  {
    title: "Back Translation",
    content: [
      {
        type: "p",
        text: 'Back Translation Augmentation relies on translating text data to another language and then translating it back to the original language. This technique allows generating textual data of distinct wording to original text while preserving the original context and meaning. <a href="#ref-1">[1]</a> <a href="#ref-2">[2]</a> <a href="#ref-10">[10]</a>',
      },
      {
        type: "code",
        text: `>>> from text_data_augmentation import BackTranslation
>>> aug = BackTranslation()
>>> aug(['A quick brown fox jumps over the lazy dog'])
['A quick brown fox jumps over the lazy dog', 'A quick brown fox jumps on the lazy dog']`,
      },
    ],
  },
  {
    title: "Character Noise",
    content: [
      {
        type: "p",
        text: 'Character Noise Augmentation adds character level noise by randomly inserting, deleting, swaping or replacing some charaters in the input text. <a href="#ref-2">[2]</a> <a href="#ref-9">[9]</a>',
      },
      {
        type: "code",
        text: `>>> from text_data_augmentation import CharacterNoise
>>> aug = CharacterNoise(alpha=0.1, n_aug=1)
>>> aug(['A quick brown fox jumps over the lazy dog'])
['A quick brown fox jumps over the lazy dog', 'A quick brown fox jumps ovr the lazy dog']`,
      },
    ],
  },
  {
    title: "Contextual Word Replacement",
    content: [
      {
        type: "p",
        text: 'Contextual Word Replacement Augmentation creates Augmented Samples by randomly replacing some words with a mask and then using a Masked Language Model to fill it. Sampling of words can be weighted using TFIDF values as well. <a href="#ref-2">[2]</a> <a href="#ref-3">[3]</a> <a href="#ref-11">[11]</a> <a href="#ref-19">[19]</a>',
      },
      {
        type: "code",
        text: `>>> from text_data_augmentation import ContextualWordReplacement
>>> aug = ContextualWordReplacement(n_aug=1)
>>> aug(['A quick brown fox jumps over the lazy dog'])
['A quick brown fox jumps over the lazy dog', 'A quick brown fox jumps over his lazy dog']`,
      },
    ],
  },
  {
    title: "Easy Data Augmentation",
    content: [
      {
        type: "p",
        text: 'Easy Data Augmentation adds word level noise by randomly inserting, deleting, swaping some words in the input text or by shuffling the sentences in the input text. <a href="#ref-4">[4]</a> <a href="#ref-5">[5]</a> <a href="#ref-9">[9]</a> <a href="#ref-12">[12]</a> <a href="#ref-13">[13]</a>',
      },
      {
        type: "code",
        text: `>>> from text_data_augmentation import EasyDataAugmentation
>>> aug = EasyDataAugmentation(n_aug=1)
>>> aug(['A quick brown fox jumps over the lazy dog'])
['A quick brown fox jumps over the lazy dog', 'A quick brown fox jumps over the dog']`,
      },
    ],
  },
  {
    title: "KeyBoard Noise",
    content: [
      {
        type: "p",
        text: 'KeyBoard Noise Augmentation adds character level spelling mistake noise by mimicing typographical errors made using a qwerty keyboard in the input text. <a href="#ref-2">[2]</a> <a href="#ref-9">[9]</a>',
      },
      {
        type: "code",
        text: `>>> from text_data_augmentation import Paraphrase
>>> aug = Paraphrase("&lt;T5 Model&gt;", n_aug=1)
>>> aug(['A quick brown fox jumps over the lazy dog'])
['A quick brown fox jumps over the lazy dog', 'A quick brown fox has jumped on the lazy dog.']`,
      },
    ],
  },
  {
    title: "OCR Noise",
    content: [
      {
        type: "p",
        text: 'OCR Noise Augmentation adds character level spelling mistake noise by mimicing ocr errors in the input text. <a href="#ref-6">[6]</a>',
      },
      {
        type: "code",
        text: `>>> from text_data_augmentation import OCRNoise
>>> aug = OCRNoise(alpha=0.1, n_aug=1)
>>> aug(['A quick brown fox jumps over the lazy dog'])
['A quick brown fox jumps over the lazy dog', 'A quick hrown lox jumps over the lazy dog']`,
      },
    ],
  },
  {
    title: "Paraphrase",
    content: [
      {
        type: "p",
        text: 'Paraphrase Augmentation rephrases the input sentences using T5 models. <a href="#ref-2">[2]</a>',
      },
      {
        type: "code",
        text: `>>> from text_data_augmentation import Paraphrase
>>> aug = Paraphrase("&lt;T5 Model&gt;", n_aug=1)
>>> aug(['A quick brown fox jumps over the lazy dog'])
['A quick brown fox jumps over the lazy dog', 'A quick brown fox has jumped on the lazy dog.']`,
      },
    ],
  },
  {
    title: "Similar Word Replacement",
    content: [
      {
        type: "p",
        text: 'Similar Word Replacement Augmentation creates Augmented Samples by randomly replacing some words with a word having the most similar vector to it. Sampling of words can be weighted using TFIDF values as well. <a href="#ref-2">[2]</a> <a href="#ref-7">[7]</a> <a href="#ref-15">[15]</a> <a href="#ref-16">[16]</a> <a href="#ref-19">[19]</a>',
      },
      {
        type: "code",
        text: `>>> from text_data_augmentation import SimilarWordReplacement
>>> aug = SimilarWordReplacement("en_core_web_lg",  alpha=0.1, n_aug=1)
>>> aug(['A quick brown fox jumps over the lazy dog'])
['A quick brown fox jumps over the lazy dog', 'A quick White Wolf jumps over the lazy Cat.']`,
      },
    ],
  },
  {
    title: "Synonym Replacement",
    content: [
      {
        type: "p",
        text: 'Synonym Replacement Augmentation creates Augmented Samples by randomly replacing some words with their synonyms based on the word net data base. Sampling of words can be weighted using TFIDF values as well. <a href="#ref-2">[2]</a> <a href="#ref-4">[4]</a> <a href="#ref-8">[8]</a> <a href="#ref-13">[13]</a> <a href="#ref-19">[19]</a>',
      },
      {
        type: "code",
        text: `>>> from text_data_augmentation import SynonymReplacement
>>> aug = SynonymReplacement(alpha=0.1, n_aug=1)
>>> aug(['A quick brown fox jumps over the lazy dog'])
['A quick brown fox jumps over the lazy dog', 'A quick brown fox jumps over the lethargic dog']`,
      },
    ],
  },
  {
    title: "Word Split",
    content: [
      {
        type: "p",
        text: 'Word Split Augmentation adds word level spelling mistake noise by spliting words randomly in the input text. <a href="#ref-2">[2]</a> <a href="#ref-14">[14]</a>',
      },
      {
        type: "code",
        text: `>>> from text_data_augmentation import WordSplit
>>> aug = WordSplit(alpha=0.1, n_aug=1)
>>> aug(['A quick brown fox jumps over the lazy dog'])
['A quick brown fox jumps over the lazy dog', 'A quick brown fox jumps over th e lazy dog']`,
      },
    ],
  },
];
