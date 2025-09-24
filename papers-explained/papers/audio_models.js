const audio_models = [
    {
        title: "wav2vec",
        link: "https://ritvik19.medium.com/f82a0cbde202",
        date: "April 2019",
        description:
            "An unsupervised pre-training method for speech recognition that learns representations of raw audio using a multi-layer convolutional neural network. It is trained on large amounts of unlabeled audio data via a noise contrastive binary classification task, and the resulting representations are used to improve acoustic model training.",
        tags: ["Audio Models", "Self-Supervised Learning"],
    },
    {
        title: "Pretrained Audio Neural Networks (PANNs)",
        link: "https://ritvik19.medium.com/a4baa79c5139",
        date: "December 2019",
        description:
            "Addresses the limited research on pretraining systems on large-scale datasets for audio pattern recognition and transferring them to other audio-related tasks. Trained on AudioSet, which contains 1.9 million audio clips with an ontology of 527 sound classes. Explores various convolutional neural network architectures, including a proposed Wavegram-Logmel-CNN that uses both log-mel spectrogram and waveform as input features.",
        tags: ["Audio Models", "Transfer Learning"],
    },
    {
        title: "wav2vec 2.0",
        link: "https://ritvik19.medium.com/fe05d2379da1",
        date: "June 2020",
        description:
            "A framework for self-supervised learning of speech representations that learns powerful representations from speech audio alone and then fine-tunes on transcribed speech. It masks the speech input in the latent space and solves a contrastive task defined over a quantization of the latent representations, which are jointly learned, demonstrating the feasibility of speech recognition with limited amounts of labeled data.",
        tags: ["Audio Models", "Self-Supervised Learning"],
    }
]