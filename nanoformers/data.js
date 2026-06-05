let title = "Nanoformers";
let project_date = "Open Source"
let links = {
    "paper": "",
    "demo": "https://wandb.ai/ritvik19/nanoformers",
    "code": "https://github.com/Ritvik19/Nanoformers",
    "model": "",
    "data": ""
}
let link2icon = {
    "code": "fas fa-code",
    "demo": "fas fa-terminal",
    "model": "fas fa-cogs",
    "data": "fas fa-database",
    "paper": "fas fa-file-pdf",
}
let project_contents = {
    "Overview": [
        {
            "type": "text",
            "content": "Miniature implementations of key LLMs — a minimal playground for building and training transformer models from scratch. Covers self-supervised, supervised, reinforcement, and contrastive learning with tiny transformer architectures, PEFT (LoRA/QLoRA), and 20+ training runs logged on Weights &amp; Biases."
        }
    ],
    "Installation": [
        {
            "type": "code",
            "content": "git clone https://github.com/Ritvik19/Nanoformers.git\ncd Nanoformers\npip install -r requirements.txt"
        }
    ],
    "Training Paradigms": [
        {
            "type": "text",
            "content": "Nanoformers implements tiny transformer architectures from scratch and provides training loops across four paradigms:"
        },
        {
            "type": "html",
            "content": "<h4>Self-Supervised</h4>\n<div class=\"table-wrap\"><table class=\"data-table\">\n<thead><tr><th>Task</th><th>Architecture</th><th>Status</th></tr></thead>\n<tbody>\n<tr><td>Causal Language Modeling</td><td>Decoder-only</td><td>✅</td></tr>\n<tr><td>Masked Language Modeling</td><td>Encoder-only</td><td>✅</td></tr>\n<tr><td>Span Corruption</td><td>Encoder-Decoder</td><td>✅</td></tr>\n</tbody></table></div>"
        },
        {
            "type": "html",
            "content": "<h4>Supervised</h4>\n<div class=\"table-wrap\"><table class=\"data-table\">\n<thead><tr><th>Task</th><th>Architecture</th><th>Status</th></tr></thead>\n<tbody>\n<tr><td>Instruction Fine-Tuning</td><td>Decoder-only</td><td>✅</td></tr>\n<tr><td>Direct Preference Optimization</td><td>Decoder-only</td><td>✅</td></tr>\n<tr><td>Sequence Classification</td><td>Encoder-only</td><td>✅</td></tr>\n<tr><td>Token Classification</td><td>Encoder-only</td><td>✅</td></tr>\n<tr><td>Extractive Question Answering</td><td>Encoder-only</td><td>✅</td></tr>\n<tr><td>Sequence-to-Sequence Modeling</td><td>Encoder-Decoder</td><td>✅</td></tr>\n</tbody></table></div>"
        },
        {
            "type": "html",
            "content": "<h4>Reinforcement Learning</h4>\n<div class=\"table-wrap\"><table class=\"data-table\">\n<thead><tr><th>Task</th><th>Architecture</th><th>Status</th></tr></thead>\n<tbody>\n<tr><td>REINFORCE (baseline, KL penalty, length normalization)</td><td>Decoder-only</td><td>✅</td></tr>\n<tr><td>Proximal Policy Optimization</td><td>Decoder-only</td><td>✅</td></tr>\n<tr><td>GRPO (DAPO, Dr. GRPO, GSPO variants)</td><td>Decoder-only</td><td>✅</td></tr>\n</tbody></table></div>"
        },
        {
            "type": "html",
            "content": "<h4>Contrastive</h4>\n<div class=\"table-wrap\"><table class=\"data-table\">\n<thead><tr><th>Task</th><th>Architecture</th><th>Status</th></tr></thead>\n<tbody>\n<tr><td>Contrastive Loss</td><td>Encoder-only</td><td>✅</td></tr>\n<tr><td>Triplet Loss</td><td>Encoder-only</td><td>✅</td></tr>\n<tr><td>InfoNCE Loss</td><td>Encoder-only</td><td>✅</td></tr>\n<tr><td>Image-Text Contrastive</td><td>Dual Encoder (Vision + Text)</td><td>✅</td></tr>\n<tr><td>Image-Text Sigmoid Contrastive</td><td>Dual Encoder (Vision + Text)</td><td>✅</td></tr>\n</tbody></table></div>"
        }
    ],
    "Models Trained": [
        {
            "type": "text",
            "content": "All training runs are logged on <a href=\"https://wandb.ai/ritvik19/nanoformers\" target=\"_blank\" rel=\"noopener\">Weights &amp; Biases</a>."
        },
        {
            "type": "html",
            "content": "<div class=\"table-wrap\"><table class=\"data-table\">\n<thead><tr><th>Model</th><th>Dataset</th><th>Task</th><th>Logs</th></tr></thead>\n<tbody>\n<tr><td>Qwen/Qwen3-0.6B</td><td>Ritvik19/gsm8k-onpolicy-Qwen3-0.6B-ift</td><td>Instruction Fine-Tuning</td><td><a href=\"https://wandb.ai/ritvik19/nanoformers/runs/6njm3m9q\" target=\"_blank\" rel=\"noopener\">wandb</a></td></tr>\n<tr><td>Qwen/Qwen3-0.6B</td><td>Ritvik19/gsm8k-onpolicy-Qwen3-0.6B-dpo</td><td>Direct Preference Optimization</td><td><a href=\"https://wandb.ai/ritvik19/nanoformers/runs/3hsxkyfp\" target=\"_blank\" rel=\"noopener\">wandb</a></td></tr>\n<tr><td>Qwen/Qwen3-0.6B-Base</td><td>Ritvik19/gsm8k-onpolicy-Qwen3-0.6B-cpt</td><td>Causal Language Modeling</td><td><a href=\"https://wandb.ai/ritvik19/nanoformers/runs/twq18n69\" target=\"_blank\" rel=\"noopener\">wandb</a></td></tr>\n<tr><td>distilbert-base-uncased</td><td>Ritvik19/dair-ai-emotion</td><td>Sequence Classification</td><td><a href=\"https://wandb.ai/ritvik19/nanoformers/runs/6wlqge7k\" target=\"_blank\" rel=\"noopener\">wandb</a></td></tr>\n<tr><td>distilbert-base-uncased</td><td>Ritvik19/conll-2003-ner</td><td>Token Classification</td><td><a href=\"https://wandb.ai/ritvik19/nanoformers/runs/hrpxlrfe\" target=\"_blank\" rel=\"noopener\">wandb</a></td></tr>\n<tr><td>distilbert-base-uncased</td><td>Ritvik19/squad-v2</td><td>Extractive Question Answering</td><td><a href=\"https://wandb.ai/ritvik19/nanoformers/runs/x00m7mfb\" target=\"_blank\" rel=\"noopener\">wandb</a></td></tr>\n<tr><td>google/flan-t5-base</td><td>Ritvik19/gsm8k-seq2seq</td><td>Sequence-to-Sequence Modeling</td><td><a href=\"https://wandb.ai/ritvik19/nanoformers/runs/n5iq698c\" target=\"_blank\" rel=\"noopener\">wandb</a></td></tr>\n<tr><td>distilbert-base-uncased</td><td>Ritvik19/qqp-contrastive</td><td>Contrastive Loss</td><td><a href=\"https://wandb.ai/ritvik19/nanoformers/runs/o8y891t1\" target=\"_blank\" rel=\"noopener\">wandb</a></td></tr>\n<tr><td>distilbert-base-uncased</td><td>Ritvik19/qqp-triplet</td><td>Triplet Loss</td><td><a href=\"https://wandb.ai/ritvik19/nanoformers/runs/2s72jin4\" target=\"_blank\" rel=\"noopener\">wandb</a></td></tr>\n<tr><td>distilbert-base-uncased</td><td>Ritvik19/qqp-info_nce</td><td>InfoNCE Loss</td><td><a href=\"https://wandb.ai/ritvik19/nanoformers/runs/jpl7ndhk\" target=\"_blank\" rel=\"noopener\">wandb</a></td></tr>\n<tr><td>roberta-base + vit-base-patch16-224</td><td>Ritvik19/flickr30k</td><td>Image-Text Contrastive</td><td><a href=\"https://wandb.ai/ritvik19/nanoformers/runs/1s18t8h4\" target=\"_blank\" rel=\"noopener\">wandb</a></td></tr>\n<tr><td>roberta-base + vit-base-patch16-224</td><td>Ritvik19/flickr30k</td><td>Image-Text Sigmoid Contrastive</td><td><a href=\"https://wandb.ai/ritvik19/nanoformers/runs/6q89d7xe\" target=\"_blank\" rel=\"noopener\">wandb</a></td></tr>\n<tr><td>bert-base-uncased</td><td>Ritvik19/open-web-text</td><td>Masked Language Modeling</td><td><a href=\"https://wandb.ai/ritvik19/nanoformers/runs/1jiqkkmp\" target=\"_blank\" rel=\"noopener\">wandb</a></td></tr>\n<tr><td>t5-base</td><td>Ritvik19/open-web-text</td><td>Span Corruption</td><td><a href=\"https://wandb.ai/ritvik19/nanoformers/runs/tnotg2s3\" target=\"_blank\" rel=\"noopener\">wandb</a></td></tr>\n<tr><td>Qwen/Qwen3-0.6B</td><td>Ritvik19/math-rl</td><td>REINFORCE</td><td><a href=\"https://wandb.ai/ritvik19/nanoformers/runs/a2ttdud6\" target=\"_blank\" rel=\"noopener\">wandb</a></td></tr>\n<tr><td>Qwen/Qwen3-0.6B</td><td>Ritvik19/math-rl</td><td>Proximal Policy Optimization</td><td><a href=\"https://wandb.ai/ritvik19/nanoformers/runs/23891ele\" target=\"_blank\" rel=\"noopener\">wandb</a></td></tr>\n<tr><td>Qwen/Qwen3-0.6B</td><td>Ritvik19/math-rl</td><td>GRPO</td><td><a href=\"https://wandb.ai/ritvik19/nanoformers/runs/of3lqzyc\" target=\"_blank\" rel=\"noopener\">wandb</a></td></tr>\n<tr><td>Qwen/Qwen3-0.6B</td><td>Ritvik19/math-rl</td><td>DAPO</td><td><a href=\"https://wandb.ai/ritvik19/nanoformers/runs/j0rdxxeg\" target=\"_blank\" rel=\"noopener\">wandb</a></td></tr>\n<tr><td>Qwen/Qwen3-0.6B</td><td>Ritvik19/math-rl</td><td>Dr. GRPO</td><td><a href=\"https://wandb.ai/ritvik19/nanoformers/runs/p9w90aoz\" target=\"_blank\" rel=\"noopener\">wandb</a></td></tr>\n<tr><td>Qwen/Qwen3-0.6B</td><td>Ritvik19/math-rl</td><td>GSPO</td><td><a href=\"https://wandb.ai/ritvik19/nanoformers/runs/1ihsqjgn\" target=\"_blank\" rel=\"noopener\">wandb</a></td></tr>\n</tbody></table></div>"
        }
    ],
    "PEFT & Parallelism": [
        {
            "type": "html",
            "content": "<h4>Parameter-Efficient Fine-Tuning</h4>\n<div class=\"table-wrap\"><table class=\"data-table\">\n<thead><tr><th>Method</th><th>Description</th><th>Status</th></tr></thead>\n<tbody>\n<tr><td>LoRA</td><td>Low-Rank Adaptation (W + BA) on a frozen base model — pass <code>--peft-config configs/peft/lora.yaml</code></td><td>✅</td></tr>\n<tr><td>QLoRA</td><td>LoRA on top of a 4-bit NF4 quantized base model — pass <code>--peft-config configs/peft/qlora.yaml</code></td><td>✅</td></tr>\n</tbody></table></div>"
        },
        {
            "type": "html",
            "content": "<h4>Parallelization Strategies</h4>\n<div class=\"table-wrap\"><table class=\"data-table\">\n<thead><tr><th>Strategy</th><th>Scope</th><th>Status</th></tr></thead>\n<tbody>\n<tr><td>Gradient Accumulation</td><td>Single GPU, larger effective batch</td><td>✅</td></tr>\n<tr><td>Mixed Precision (fp16 / bf16)</td><td>Single GPU, memory &amp; speed</td><td>✅</td></tr>\n<tr><td>Data Parallelism (DDP)</td><td>Replicate model, shard batch</td><td>⬜️</td></tr>\n<tr><td>Fully Sharded Data Parallelism (FSDP / ZeRO-3)</td><td>Shard params, grads, optimizer states</td><td>⬜️</td></tr>\n<tr><td>Tensor Parallelism (TP)</td><td>Shard individual matmuls within a layer</td><td>⬜️</td></tr>\n<tr><td>Pipeline Parallelism (PP)</td><td>Shard layers across GPUs with micro-batching</td><td>⬜️</td></tr>\n<tr><td>Context / Sequence Parallelism (CP / SP)</td><td>Shard along sequence length</td><td>⬜️</td></tr>\n<tr><td>Expert Parallelism (EP)</td><td>Shard MoE experts across GPUs</td><td>⬜️</td></tr>\n</tbody></table></div>"
        }
    ],
    "Usage Examples": [
        {
            "type": "text",
            "content": "Every training script accepts <code>--config</code> (required) and an optional <code>--peft-config</code> for LoRA / QLoRA. Omit <code>--peft-config</code> for full fine-tuning."
        },
        {
            "type": "code",
            "content": "# Full fine-tuning\npython -m src.cli.train_ift --config configs/ift_qwen_gsm8k.yaml\n\n# LoRA\npython -m src.cli.train_ift --config configs/ift_qwen_gsm8k.yaml \\\n    --peft-config configs/peft/lora.yaml\n\n# Instruction fine-tuning\npython -m src.cli.train_ift --config configs/ift_qwen_gsm8k.yaml\n\n# GRPO reinforcement learning\npython -m src.cli.train_grpo --config configs/grpo_qwen_math.yaml\n\n# Causal language modeling\npython -m src.cli.train_clm --config configs/clm_qwen_gsm8k.yaml"
        }
    ],
    "References": [
        {
            "type": "list",
            "content": [
                "<a href=\"https://github.com/Ritvik19/Nanoformers\" target=\"_blank\" rel=\"noopener\">Nanoformers on GitHub</a>",
                "<a href=\"https://wandb.ai/ritvik19/nanoformers\" target=\"_blank\" rel=\"noopener\">Weights & Biases Project</a>"
            ]
        }
    ],
};
