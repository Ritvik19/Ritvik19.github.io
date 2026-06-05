let title = "Aryabhata 2 : Scaling Reinforcement Learning for Advanced STEM Reasoning";
let project_date = "May 2026"
let links = {
    "paper": "https://arxiv.org/abs/2605.28829",
    "demo": "",
    "code": "",
    "model": "https://huggingface.co/PhysicsWallahAI/Aryabhata-2.0",
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
    "Overview": [{"type": "text", "content": "<b>Aryabhata 2</b> is a reasoning-focused language model developed by <b>Physics Wallah AI Research</b> for competitive STEM examinations including <b>JEE Main</b>, <b>JEE Advanced</b>, and <b>NEET</b>. Built by post-training <b>GPT-OSS-20B</b> (20B-parameter MoE, 3.6B active parameters) via reinforcement learning on a curated curriculum from Physics Wallah's internal question banks, Aryabhata 2 achieves <b>88.95% average Pass@1</b> on in-distribution exam benchmarks while using up to <b>64% fewer output tokens</b> than its base model."}],
    "Key Features": [{"type": "list", "content": [
        "<b>Base Model:</b> GPT-OSS-20B with LoRA adapters (rank 64, α=128); only 0.15% of parameters are trainable.",
        "<b>Training Method:</b> Group Relative Policy Optimization (GRPO) with verifiable rewards.",
        "<b>Multi-Subject Coverage:</b> Physics, Chemistry, Mathematics, and General Reasoning.",
        "<b>Strong In-Distribution Performance:</b> 92.99% on JEE Main 2026, 87.80% on JEE Main 2025, 86.51% on JEE Advanced 2025, 84.66% on NEET 2025.",
        "<b>Token Efficiency:</b> 42.31 Acc./1K tokens on in-distribution tasks (best among evaluated models).",
        "<b>Compute Efficient:</b> Trained on 2× NVIDIA H100 NVL GPUs."
    ]}],
    "Training Overview": [
        {"type": "heading", "content": "Training Data"},
        {"type": "text", "content": "The training corpus is derived from Physics Wallah's internal question banks covering Physics, Chemistry, Mathematics, and General Reasoning. The raw dataset contains 1.78M questions; after multi-stage cleaning and answer verification, a 100K-question difficulty-aware curriculum is constructed for reinforcement learning."},
        {"type": "heading", "content": "Three-Phase RL Pipeline"},
        {"type": "table", "columns": ["Phase", "Steps", "Group Size", "Data", "Focus"], "rows": [
            ["Format Alignment", "300", "8", "~5K (trivial)", "Output format"],
            ["Prolonged RL (ProRL)", "~5,000", "8 → 16", "~80K (learnable)", "Reasoning accuracy"],
            ["Broadened RL (BroRL)", "~700", "64 → 128", "~15K (challenging)", "Exploration & generalization"]
        ]},
        {"type": "heading", "content": "Training Setup"},
        {"type": "text", "content": "Training uses GRPO with LoRA applied to attention projection and token embedding layers (q_proj, k_proj, v_proj, o_proj, embed_tokens). Reward function: R = R_accuracy × R_format, combining a cascade of string, numeric, and symbolic matchers with a format reward for well-structured responses."}
    ],
    "Method": [
        {"type": "heading", "content": "Base Model"},
        {"type": "text", "content": "Aryabhata 2 is built on GPT-OSS-20B, an open-source 20B-parameter Mixture of Experts model released by OpenAI. Parameter-efficient fine-tuning with LoRA (rank 64, scaling factor 128) reduces memory usage while preserving adaptation capacity for reasoning improvements through RL."},
        {"type": "heading", "content": "Data Preparation"},
        {"type": "text", "content": "A deterministic cleaning pipeline removes malformed or unsuitable questions before reinforcement learning:"},
        {"type": "list", "content": [
            "<b>HTML removal:</b> Discard diagram-dependent questions with image tags.",
            "<b>LaTeX validation:</b> Render expressions with pdflatex; discard failures.",
            "<b>Completeness check:</b> LLM classifier removes ill-posed or incomplete questions.",
            "<b>Domain filtering:</b> Retain only Physics, Chemistry, Mathematics, and General Reasoning.",
            "<b>Result:</b> ~24% of the original dataset is removed."
        ]},
        {"type": "heading", "content": "Answer Verification"},
        {"type": "text", "content": "An automated verification pipeline uses GPT-OSS-120B as the policy model and Qwen3-30B-A3B-Thinking as the judge to validate ground-truth answer keys:"},
        {"type": "list", "content": [
            "<b>Single-sample stage:</b> 1 CoT generation; ~80% of dataset verified.",
            "<b>Four-sample stage:</b> 4 independent generations; additional ~8% verified.",
            "<b>Sixteen-sample stage:</b> 16 independent generations; additional ~4% verified."
        ]},
        {"type": "heading", "content": "Curriculum Construction"},
        {"type": "text", "content": "After verification, questions are categorized by empirical difficulty based on 4-sample correctness rates: <b>trivial</b> (4/4 correct), <b>learnable</b> (1–3/4 correct), and <b>challenging</b> (0/4 correct). The final 100K curriculum balances subjects (30K Physics, 30K Chemistry, 30K Mathematics, 10K General Reasoning) across difficulty levels."},
        {"type": "heading", "content": "Reinforcement Learning"},
        {"type": "text", "content": "Training proceeds in three on-policy RL phases: format alignment on trivial questions, prolonged RL (ProRL) on learnable questions with gradually increased group sizes, and broadened RL (BroRL) on challenging questions with large rollout groups (64→128) to improve exploration. This unified RL-only pipeline enables stable training under constrained compute while preserving strong reasoning quality on exam-style problems."}
    ],
    "Results": [
        {"type": "heading", "content": "Evaluation Setup"},
        {"type": "text", "content": "We report Pass@1 accuracy (4-sample mean) on in-distribution competitive exam benchmarks and out-of-distribution reasoning datasets. Answer correctness is determined via a multi-stage pipeline: string matching, numeric matching, symbolic matching (math-verify), option matching, and LLM-as-judge fallback."},
        {"type": "heading", "content": "In-Distribution Benchmarks (Pass@1, %)"},
        {"type": "table", "columns": ["Model", "JEE Adv. 2025", "NEET 2025", "JEE Main 2025", "JEE Main 2026", "Avg."], "rows": [
            ["Gemini 2.5 Flash", "96.81", "90.00", "87.26", "96.22", "90.23"],
            ["GPT-5 Mini", "93.65", "87.33", "87.07", "95.83", "89.71"],
            ["Qwen3-30B-A3B (Thinking)", "90.48", "86.00", "84.89", "97.26", "88.55"],
            ["GPT-OSS-120B", "84.13", "85.33", "85.61", "95.42", "88.28"],
            ["<b>Aryabhata 2 (ours)</b>", "<b>86.51</b>", "<b>84.66</b>", "<b>87.80</b>", "<b>92.99</b>", "<b>88.95</b>"],
            ["Nemotron 3 Nano 30B A3B", "90.87", "84.00", "82.89", "94.84", "86.51"],
            ["GPT-OSS-20B", "77.38", "81.33", "79.27", "92.46", "83.00"]
        ]},
        {"type": "heading", "content": "Out-of-Distribution Benchmarks (Pass@1, %)"},
        {"type": "table", "columns": ["Model", "AIME", "HMMT", "GPQA", "MMLU-Pro", "MMLU-Redux 2.0", "Avg."], "rows": [
            ["GPT-OSS-120B", "90.00", "80.01", "77.06", "90.11", "95.94", "89.50"],
            ["Qwen3-30B-A3B (Thinking)", "84.58", "51.88", "73.31", "90.80", "97.77", "89.42"],
            ["Gemini 2.5 Flash", "66.61", "59.13", "75.09", "90.44", "96.85", "89.13"],
            ["GPT-5 Mini", "83.33", "70.97", "75.46", "89.64", "96.40", "88.85"],
            ["<b>Aryabhata 2 (ours)</b>", "<b>86.67</b>", "<b>78.96</b>", "<b>74.86</b>", "<b>88.49</b>", "<b>92.92</b>", "<b>87.64</b>"],
            ["GPT-OSS-20B", "86.67", "77.42", "70.51", "85.42", "93.32", "84.95"],
            ["Nemotron 3 Nano 30B A3B", "77.08", "65.86", "65.38", "84.33", "94.10", "83.48"]
        ]},
        {"type": "heading", "content": "Token Efficiency"},
        {"type": "table", "columns": ["Model", "In-Dist. Pass@1", "In-Dist. Tokens", "In-Dist. Acc./1K↑", "OOD Pass@1", "OOD Tokens", "OOD Acc./1K↑"], "rows": [
            ["<b>Aryabhata 2 (ours)</b>", "<b>88.95</b>", "<b>2,102</b>", "<b>42.31</b>", "<b>87.64</b>", "<b>2,214</b>", "<b>39.58</b>"],
            ["GPT-OSS-120B", "88.28", "3,312", "26.66", "89.50", "3,661", "24.44"],
            ["Qwen3-30B-A3B (Thinking)", "88.55", "4,556", "19.44", "89.42", "4,299", "20.80"],
            ["GPT-OSS-20B", "83.00", "5,293", "15.68", "84.95", "4,860", "17.48"]
        ]},
        {
            type: "carousel", images: [
                {"src": "in-distribution-benchmark.png", "caption": "In-Distribution Accuracy vs. Tokens"},
                {"src": "ood-benchmark.png", "caption": "Out-of-Distribution Accuracy vs. Tokens"}
            ]
        },
        {"type": "list", "content": [
            "<b>Aryabhata 2</b> achieves the strongest open-source aggregate in-distribution performance (88.95% avg), improving over GPT-OSS-20B (83.00%) and matching or exceeding larger baselines.",
            "<b>Aryabhata 2</b> delivers the best accuracy-per-token ratio among evaluated models, using up to 64% fewer output tokens than GPT-OSS-20B on in-distribution exams.",
            "On OOD benchmarks, Aryabhata 2 improves over GPT-OSS-20B across AIME, HMMT, GPQA, and MMLU-Pro while remaining competitive with frontier models."
        ]}
    ],
    "Citation": [
        {"type": "text", "content": "If you use this model, please cite:"},
        {"type": "code", "content": "@misc{aryabhata2,\n  author       = {Rastogi, Ritvik and Singh, Vishal and Chaudhari, Tejas and Varma, Sandeep},\n  title        = {Aryabhata 2},\n  year         = {2025},\n  publisher    = {PhysicsWallah},\n  howpublished = {\\url{https://huggingface.co/PhysicsWallahAI/Aryabhata-2.0}},\n}"}
    ]
}
