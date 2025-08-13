let title = "Aryabhatta 1.0 : An exam-focused language model for JEE Math";
let project_date = "July 2025"
let links = {
    "paper": "https://arxiv.org/abs/2508.08665",
    "demo": "https://huggingface.co/spaces/PhysicsWallahAI/Aryabhata-Demo",
    "code": "",
    "model": "https://huggingface.co/PhysicsWallahAI/Aryabhata-1.0",
    "data": "",
}
let link2icon = {
    "code": "fas fa-code",
    "demo": "fas fa-terminal",
    "model": "fas fa-cogs",
    "data": "fas fa-database",
    "paper": "fas fa-file-pdf",
}
let project_contents = {
    "Overview": [{"type": "text", "content": "<b>Aryabhata 1.0</b> is a 7B parameter small language model for mathematics developed by <b>Physics Wallah AI Research</b>, optimized for high-stakes Indian competitive exams like <b>JEE Mains</b>. Despite its compact size, Aryabhata 1.0 achieves <b>state-of-the-art performance</b> on exam-centric reasoning tasks with impressive <b>token efficiency</b> and low inference cost."}],
    "Key Features": [{"type": "list", "content": [
        "<b>Architecture:</b> 7B parameter causal decoder-based model.",
        "<b>Exam-Centric Optimization:</b> Specifically tuned for JEE-level Mathematics reasoning.",
        "<b>High Accuracy:</b> 86% on JEE Mains January 2025 session, 90.2% on JEE Mains April 2025 session.",
        "<b>Token Efficiency:</b> Operates effectively around a ~2K token window, compared to ~8K required by other reasoning models.",
        "<b>Compute Efficient:</b> Trained on a 1x2 NVIDIA H100 GPU using optimized pipeline."
    ]}],
    "Training Overview": [
        {"type": "heading", "content": "Training Data"},
        {"type": "text", "content": "The training dataset consists of approximately 130,000 problem-solution pairs curated from proprietary Physics Wallah exam datasets. The data is specifically designed to cover a wide range of JEE-level Mathematics topics and problem types."},
        {"type": "heading", "content": "Training Pipeline"},
        {"type": "list", "content": [
            "<b>Model Merging:</b> Combining multiple model checkpoints to create a robust base model.",
            "<b>Rejection Sampling:</b> Filtering out low-quality responses to ensure high-quality outputs.",
            "<b>Supervised Fine-Tuning (SFT):</b> Fine-tuning the model on the curated dataset to improve its performance on JEE-level problems.",
            "<b>Reinforcement Learning with Verifiable Rewards (RLVR):</b> Applying reinforcement learning techniques to further enhance the model's reasoning capabilities."
        ]},
        {"type": "heading", "content": "Training Setup"},
        {"type": "text", "content": "The model was trained using a single NVIDIA H100 GPU with 80GB memory, utilizing an optimized training pipeline that allowed for efficient training within a limited compute budget."}
    ],
    "Method": [
        {"type": "heading", "content": "Model Merging"},
        {"type": "text", "content": "We began with model merging (weighted average) to build a strong initialization (Aryabhata 0.5) by combining diverse model capabilities:"},
        {"type": "list", "content": [
            "<b>Qwen 2.5 Math:</b> A robust math-centric LLM with solid symbolic math foundations.",
            "<b>Ace Math:</b> An enhanced version of Qwen 2.5 Math, fine-tuned by NVIDIA for improved accuracy in mathematics benchmarks.",
            "<b>DeepSeek R1 Distill Qwen:</b> A long-form reasoning model, fine-tuned on reasoning traces distilled from DeepSeek R1."
        ]},
        {"type": "heading", "content": "Data Curation + Rejection Sampling"},
        {"type": "text", "content": "We extracted ~250K raw questions from Physics Wallah's internal database and applied aggressive filtering and cleaning:"},
        {"type": "list", "content": [
            "<b>Removed:</b> Diagram-based, non-English, and option-heavy questions.",
            "<b>Kept:</b> Questions matching the distribution of JEE Main 2019–2024.",
            "<b>Final curated dataset:</b> ~130K high-quality questions."
        ]},
        {"type": "text", "content": "For each question, we generated 4 Chains of Thought (CoTs) using Aryabhata 0.5 and retained only those leading to correct final answers."},
        {"type": "text", "content": "Resulting Dataset: ~100K questions, ~350K high-quality CoTs."},
        {"type": "text", "content": "We used this dataset for Supervised Fine-Tuning (SFT)."},
        
        {"type": "heading", "content": "Reinforcement Learning with Verifiable Rewards (RLVR)"},
        {"type": "text", "content": "We used a custom in-house variant of Group Relative Policy Optimization (GRPO), adapted for math-specific reward functions."},
        {"type": "list", "content": [
            "<b>Removed:</b> KL-divergence penalty",
            "<b>Removed:</b> Clipping"
        ]},
        {"type": "text", "content": "We used RLVR on the remaining ~30K questions."},
        {"type": "text", "content": "This multi-phase training strategy allows Aryabhata 1.0 to capture pedagogy-aligned reasoning patterns, making it highly effective for solving real student queries in mathematics."},
    ],
    "Results": [
        {"type": "heading", "content": "Evaluation Setup"},
        {"type": "text", "content": "All evaluations were performed with greedy decoding (temperature = 0.0), and we report pass@1 accuracy."},
        
        {"type": "heading", "content": "Evaluation Datasets"},
        {"type": "text", "content": "We evaluated the model on two sets of official JEE Mains 2025 mathematics papers:"},
        {"type": "list", "content": [
            "<b>January Session:</b> 10 question papers containing 250 questions.",
            "<b>April Session:</b> 9 question papers containing 225 questions."
        ]},
        {"type": "text", "content": "Each paper includes a mix of Multiple Choice Questions (MCQs) with one correct option and Numeric Answer Type (NAT) questions requiring precise numerical responses."},
        
        {"type": "heading", "content": "Evaluation Metric"},
        {"type": "text", "content": "We used a composite evaluation metric to reflect real-world grading rigor and reduce false positives:"},
        {"type": "list", "content": [
            "<b>Float Match:</b> Compares predicted and target answers within a tolerance (±1e-9), handling rounding artifacts and small numerical errors robustly.",
            "<b>String Match:</b> Used for symbolic answers (e.g., fractions, radicals), requiring strict exact match — predictions must match ground truth character-for-character.",
            "<b>LLM-as-Judge (GPT-4o-mini):</b> Used for mathematical equivalence for ambiguous formats."
        ]},
        {
            type: "carousel", images: [{"src": "benchmark.png", "caption": "Benchmark Results"}, {"src": "accuracy.png", "caption": "Accuracy Breakdown"}, {"src": "accuracy-token.jpeg", "caption": "Accuracy vs. Token Count"}]
        },
        // *Aryabhata has the best accuracy on JEE Main Maths, on par with frontier models*
        // *Aryabhata is on par with frontier models in terms of accuracy vs token usage*
        {"type": "list", "content": [
            "<b>Aryabhata</b> has the best accuracy on JEE Main Maths, on par with frontier models.",
            "<b>Aryabhata</b> is on par with frontier models in terms of accuracy vs token usage."
        ]
        }
    ],
    "Roadmap": [
        {"type": "list", "content": [
            "Extending domain coverage to <b>Physics</b> and <b>Chemistry</b>.",
            "Supporting <b>JEE Advanced</b>, <b>NEET</b>, and <b>Foundation syllabus</b>.",
            "Further optimization for affordability and accuracy in real-time deployments."
        ]}
    ],
    "Citation": [
        {"type": "text", "content": "If you use this model, please cite:"},
        {"type": "code", "content": "@misc{Aryabhata2025,\n  title = {Aryabhata 1.0: A compact, exam-focused language model tailored for mathematics in Indian competitive exams, especially JEE Main.},\n  author = {Physics Wallah AI Research},\n  year = {2025},\n  note = {\\url{https://huggingface.co/PhysicsWallahAI/Aryabhata-1.0}},\n}"}
    ]
}

