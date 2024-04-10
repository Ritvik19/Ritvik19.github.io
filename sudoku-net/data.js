let title = "SudokuNet: Neural Network for Sudoku Puzzle Solving";
let project_date = "March 2024"
let links = {
    "paper": "",
    "demo": "https://huggingface.co/spaces/Ritvik19/SudokuNetDemo",
    "code": "https://github.com/Ritvik19/SudokuNet",
    "model": "https://github.com/Ritvik19/SudokuNet",
    "data": "https://huggingface.co/datasets/Ritvik19/Sudoku-Dataset",
}
let link2icon = {
    "code": "fas fa-code",
    "demo": "fas fa-terminal",
    "model": "fas fa-cogs",
    "data": "fas fa-database",
    "paper": "fas fa-file-pdf",
}
let project_contents = {
    "Abstract": [{"type": "text", "content": "This work studies the effectiveness of Neural Network models in solving Sudoku Puzzles. The study presents SudokuNet, a feed forward network. Leveraging a dataset comprising 17 million puzzles sourced from various sources. The model achieved a remarkable accuracy of 92.932% on the evaluation set, demonstrating its efficacy in solving Sudoku puzzles. Additionally, interactive demos and pretrained models are provided, facilitating easy integration and deployment in various applications."}],
    "Introduction": [{"type": "text", "content": "Sudoku puzzles have long been a popular recreational activity, challenging players to fill a 9x9 grid with digits so that each column, row, and 3x3 subgrid contains all of the digits from 1 to 9 without repetition. With the rise of artificial intelligence and machine learning, there has been growing interest in developing automated solvers for Sudoku puzzles. In this study, we introduce SudokuNet, a Neural Network model trained to solve Sudoku puzzles efficiently. Leveraging large-scale datasets and advanced neural network architectures, SudokuNet aims to demonstrate the effectiveness of machine learning techniques in tackling combinatorial optimization problems."}],
    "Dataset": [
        {"type": "text", "content": "The SudokuNet model is trained on a comprehensive dataset comprising 17 million Sudoku puzzles. The dataset includes puzzles of varying difficulty levels sourced from multiple sources, including Kaggle datasets and miscellaneous scraped puzzles. Each puzzle in the dataset is accompanied by its solution, difficulty level, and source identifier. The dataset is split into training and evaluation sets, with 16.95 million puzzles used for training and 50K puzzles reserved for evaluation purposes. The data is stored in Parquet files, facilitating efficient processing and storage."},
        {"type": "heading", "content": "Dataset Attributes"},
        {"type": "list", "content": ["<b>Puzzle:</b> Sudoku puzzle configurations.", "<b>Solution:</b> Corresponding solutions to the puzzles.", "<b>Difficulty:</b> Indicates the difficulty level of the puzzles.", "<b>Source:</b> Identifier for the puzzle's source."]},
        {"type": "heading", "content": "Dataset Statistics"},
        {"type": "table", "columns": ["# Puzzles", "Easy", "Hard"], "rows": [["Training", "16.93M", "14K"], ["Evaluation", "49K", "1K"]]},
        {"type": "text", "content": "Number of Difficult puzzles by source"},
        {"type": "table", "columns": ["Source", "# Puzzles"], "rows": [["1m", "0"], ["3m", "78"], ["4m", "15169"], ["9m", "67"], ["challenge", "7"]]},
        {"type": "text", "content": "Number of Empty cells in puzzles"},
        {"type": "table", "columns": ["% of the dataset", "Training", "Evaluation"], "rows": [["[1, 10]", "3.67", "4.96"], ["[11, 20]", "3.85", "5.31"], ["[21, 30]", "6.05", "8.15"], ["[31, 40]", "19.31", "25.36"], ["[41, 50]", "41.79", "52.77"], ["[51, 64]", "25.29", "3.45"]]},
    ],   
    "Method": [
        {"type": "heading", "content": "Models"},
        {"type": "carousel", "images": [{"src": "sudokunet-architecture-ffn.png", "caption": "FFN Architecture"}, {"src": "sudokunet-architecture-cnn.png", "caption": "CNN Architecture"}, {"src": "sudokunet-architecture-rnn.png", "caption": "RNN Architecture"}, {"src": "sudokunet-architecture-lstm.png", "caption": "LSTM Architecture"}, {"src": "sudokunet-architecture-gru.png", "caption": "GRU"}]},
        {"type": "text", "content": "SudokuNet explores various neural network architectures, including Feed Forward Networks (FFN), Convolutional Neural Networks (CNN), Recurrent Neural Networks (RNN), Long Short-Term Memory (LSTM), and Gated Recurrent Unit (GRU). The models are designed to predict the missing digits in Sudoku puzzles based on the available information. The architectures are optimized to capture the spatial dependencies and patterns present in Sudoku puzzles, enabling accurate predictions."},
        {"type": "text", "content": "The Sudoku puzzles are represented as 9x9 grids, with each cell containing a digit from 0 to 9. The input data is preprocessed by converting the digits to one-hot encoded vectors, resulting in a 9x9x10 tensor representation. In the case of RNN, LSTM, and GRU architectures, the input is then reshaped to a tensor of shape (81, 10)."},
        {"type": "text", "content": "The neural network then consists of 2 or 4 hidden layers with 64 or 128 units each followed by a dropout layer each. the outputs of the hidden layers are then flattened by a flatten layer."},
        {"type": "text", "content": "The output layer is a collection of 81 dense layers with 9 units and a softmax activation function, which predicts the probability distribution of digits for each cell in the Sudoku grid."},
        {"type": "heading", "content": "Model Training"},
        {"type": "text", "content": "The models are trained using the Adam optimizer with a learning rate of 0.001. A batch size of 64K is used for training, and the model is trained in two stages. The first stage involves training the model on synthetic data generated by randomly deleting digits from the solved puzzles. This process helps the model learn to predict missing digits in the Sudoku grid. In the second stage, the model is fine-tuned on the original dataset to improve its performance on real-world puzzles."},
        {"type": "heading", "content": "Hyperparameters"},
        {"type": "text", "content": "All the models is trained using the Adam optimizer with a learning rate of 0.001. A batch size of 64K is used for training, and the model is trained in two stages. The loss function employed is the categorical cross-entropy loss, which is suitable for multi-class classification tasks. Early stopping is applied to prevent overfitting, and the model checkpoints are saved based on the validation loss."},
        {"type": "table", "columns": ["Parameter", "Value"], "rows": [["Optimizer", "Adam"], ["Learning Rate", "0.001"], ["Batch Size", "64K"], ["Loss Function", "Categorical Cross-Entropy"], ["Early Stopping", "Enabled"]]},
        {"type": "heading", "content": "Training Recipe"},
        {"type": "text", "content": "The first stage of training involves training the model on synthetic data generated by randomly deleting digits from the solved puzzles. This process helps the model learn to predict missing digits in the Sudoku grid. In the second stage, the model is fine-tuned on the original dataset to improve its performance on real-world puzzles."},
        {"type": "table", "columns": ["Stage", "# Deletions", "Epochs"], "rows": [
            ["1.00", "0", "3"], ["1.01", "1", "5"], ["1.02", "2", "5"], ["1.04", "4", "5"], ["1.08", "8", "10"], ["1.16", "16", "10"], ["1.32", "32", "15"], ["1.64", "64", "15"],
            ["2", "Original Data", "30"]]},
    ],
    "Results": [
        {"type": "text", "content": "SudokuNet achieves a high accuracy of 92.932% on the evaluation set, demonstrating its effectiveness in solving Sudoku puzzles. This accuracy metric is defined as the number of puzzles correctly solved by the model. The model's performance is further validated through interactive demos, allowing users to input Sudoku puzzles and observe the model's solutions in real-time. Additionally, pretrained models are made available for download, facilitating easy integration and deployment in various applications."},
        {"type": "table", "columns": ["Model", "Accuracy"], "rows": [["ffn__64x2.keras", "92.932%"], ["ffn__64x4.keras", "92.928%"], ["ffn__128x2.keras", "92.968%"], ["ffn__128x4.keras", "92.892%"], ["cnn__64x2.keras", "92.796%"], ["cnn__64x4.keras", "93.030%"]]},
    ],
    "Conclusion": [
        {"type": "text", "content": "In conclusion, SudokuNet presents a promising approach to solving Sudoku puzzles using Neural Networks. By leveraging large-scale datasets and advanced machine learning techniques, SudokuNet demonstrates high accuracy and efficiency in solving Sudoku puzzles of varying difficulty levels. The model's performance opens up possibilities for applications in puzzle-solving, educational tools, and game development. Future research directions may involve exploring more complex puzzle types, optimizing model architectures, and extending the applicability of SudokuNet to other combinatorial optimization problems."},
    ],
}

