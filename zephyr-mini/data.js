let title = "Zephyr Mini: Aligning Small LLMs";
let project_date = "May 2024"
let links = {
    "paper": "",
    "demo": "",
    "code": "https://github.com/Ritvik19/alignment-handbook",
    "model": "https://huggingface.co/collections/Ritvik19/zephyr-mini-66322633e39731d65a0567bd",
    "data": "",
}
let link2icon = {
    "code": "fas fa-code",
    "demo": "fas fa-terminal",
    "model": "fas fa-cogs",
    "data": "fas fa-database",
    "paper": "fas fa-file-pdf",
}

let code_snippets = {
    "imports": `
import torch
from transformers import AutoModelForCausalLM, GenerationConfig, AutoTokenizer
from peft import PeftModelForCausalLM
`.trim(),
    
        "checkpoint": `
base_model_id = "base_model_id"
finetuned_model_id = "finetuned_model_id"
tokenizer_id = "tokenizer_id"
model_revison = "model_revision"
`.trim(),

        "model": `
tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
pretrained_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.float16,
    load_in_8bit=False,
    device_map="auto",
    trust_remote_code=True
)
peft_model = PeftModelForCausalLM.from_pretrained(
    pretrained_model,
    finetuned_model_id,
    from_transformers=True,
    device_map="auto"
    revision=model_revision,
)

model = peft_model.merge_and_unload()
`.trim(),

        "generate": `
generation_config = GenerationConfig(
    pad_token_id=tokenizer.eos_token_id,
    max_new_tokens=1024, 
)

def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to('cuda')
    outputs = model.generate(**inputs, generation_config=generation_config)
    generated_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_response
`.trim(),

        "prompt": `
chat = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello, how are you?"},
]
prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
response = generate_response(prompt)
print(response)
`.trim(),

}

function create_model_details_table(details) {
    return table = [
        ["Model", ...details['titles'].map((title) => title.replace(/_/g, " "))],
        ["Fine-Tuned Model", ...details['fine_tuned_models'].map((fine_tuned_model) => `<a href="https://huggingface.co/${fine_tuned_model}" target="_blank">${fine_tuned_model}</a>`)],
        ["Training Approach", ...details['training_approaches'].map((training_approach) => training_approach.replace(/_/g, " "))],
        ["Base Model", ...details['base_models'].map((base_model) => `<a href="https://huggingface.co/${base_model}" target="_blank">${base_model}</a>`)],
        ["Training Config", ...details['training_configs'].map((training_config) => `<a href="${training_config}" target="_blank">View Config</a>`)],
        ["Model Revision", ...details['revisions']]
        
    ];
}

let model_details = {
    'TinyLlama 1.1B': {
        'titles': ['Zephyr TinyLlama SFT Qlora v0.1'],
        'training_approaches': ['sft'],
        'base_models': ['TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T'],
        'fine_tuned_models': ['Ritvik19/zephyr-tinyllama-sft-qlora'],
        'training_configs': ['https://github.com/Ritvik19/alignment-handbook/blob/main/recipes/zephyr-tinyllama/sft/config_qlora.yaml'],
        'revisions': ['002c335a34c5176e14c92875084764cee0f7be98']
    },
    'Danube 1.8B': {
        'titles': ['Zephyr Danube SFT Qlora v0.1'],
        'training_approaches': ['sft'],
        'base_models': ['h2oai/h2o-danube-1.8b-base'],
        'fine_tuned_models': ['Ritvik19/zephyr-danube-sft-qlora'],
        'training_configs': ['https://github.com/Ritvik19/alignment-handbook/blob/main/recipes/zephyr-danube/sft/config_qlora.yaml'],
        'revisions': ['f89870505f24b92988e35543132bab68529ea1dd']
    },
    'Gemma 2B':{
        'titles': ['Zephyr Gemma 2B SFT Qlora v0.1'],
        'training_approaches': ['sft'],
        'base_models': ['google/gemma-2b'],
        'fine_tuned_models': ['Ritvik19/zephyr-2b-gemma-sft-qlora'],
        'training_configs': ['https://github.com/Ritvik19/alignment-handbook/blob/main/recipes/zephyr-2b-gemma/sft/config_qlora.yaml'],
        'revisions': ['2a2c01f40945e4104a022f9edd0a764b70bb88a7']
    }
    
}

let project_contents = {
    "TL;DR": [
        {"type": "text", "content": "This project centers on the alignment and fine-tuning of small language models (LlMs), drawing on insights from the Zephyr paper by Hugging Face and utilising the resources from the Alignment-Handbook repository. Our objective is to create detailed training recipes for small LLMs and provide aligned checkpoints for open-source distribution."},
        {"type": "text", "content": "We aim to fine-tune smaller LLMs utilizing a combination of tried-and-true methods and novel approaches. Our project embraces the principles of open-source development, ensuring that all training recipes and aligned checkpoints are accessible to the broader community. This approach promotes collaboration, enabling a continuous cycle of improvement and knowledge sharing among researchers and developers."},
        {"type": "text", "content": "Through this work, we offer a flexible framework for aligning small LLMs, designed for easy integration with various applications. The project's overarching goal is to advance the conversation on AI alignment while setting a benchmark for community-driven AI development."},

    ],
    "TinyLlama 1.1B": [
        {"type": "table", "columns": ["", ""], "rows": create_model_details_table(model_details['TinyLlama 1.1B'])},
        {"type": "heading", "content": "Example Prompt"},
        {"type": "code", "content": '<|system|>\nYou are a helpful assistant. \n<|user|>\nHello, how are you? \n<|assistant|>\nI am fine, thank you. How are you?'},
    ],
    "Danube 1.8B": [
        {"type": "table", "columns": ["", ""], "rows": create_model_details_table(model_details['Danube 1.8B'])},
        {"type": "heading", "content": "Example Prompt"},
        {"type": "code", "content": '<|system|>\nYou are a helpful assistant. \n<|user|>\nHello, how are you? \n<|assistant|>\nI am doing well, thank you for asking.'},
    ],
    "Gemma 2B": [
        {"type": "table", "columns": ["", ""], "rows": create_model_details_table(model_details['Gemma 2B'])},
    ],    
    "Evaluation": [
        {"type": "text", "content": "We evaluate models on 6 key benchmarks using the Open LLM Leaderboard which utilises Eleuther AI Language Model Evaluation Harness, a unified framework to test generative language models on a large number of different evaluation tasks."},
        {"type": "text", "content": "The benchmarks include:"},
        {"type": "bullet", "content": [
            "AI2 Reasoning Challenge (25-shot) - a set of grade-school science questions.", 
            "HellaSwag (10-shot) - a test of commonsense inference, which is easy for humans (~95%) but challenging for SOTA models.",
            "MMLU (5-shot) - a test to measure a text model's multitask accuracy. The test covers 57 tasks including elementary mathematics, US history, computer science, law, and more.",
            "TruthfulQA (0-shot) - a test to measure a model's propensity to reproduce falsehoods commonly found online. Note: TruthfulQA is technically a 6-shot task in the Harness because each example is prepended with 6 Q/A pairs, even in the 0-shot setting.",
            "Winogrande (5-shot) - an adversarial and difficult Winograd benchmark at scale, for commonsense reasoning.",
            "GSM8k (5-shot) - diverse grade school math word problems to measure a model's ability to solve multi-step mathematical reasoning problems.",
        ]},
        {"type": "text", "content": "For all these evaluations, a higher score is a better score. We chose these benchmarks as they test a variety of reasoning and general knowledge across a wide variety of fields in 0-shot and few-shot settings."},
        {"type": "heading", "content": "Results"},
        {"type": "table", "columns": ["Model",  "Average", "ARC", "HellaSwag", "MMLU", "TruthfulQA", "Winogrande", "GSM8k"], "rows": [
            ["TinyLlama 1.1B 3T", "36.42", "33.87", "60.31", "26.04", "37.32", "59.51", "1.44"],
            ["Zephyr TinyLlama SFT Qlora v0.1", "36.64", "34.64", "59.84", "25.85", "36.57", "61.17", "1.74"],
            ["Danube 1.8B Base", "39.12", "39.42", "69.58", "25.94", "33.86", "64.48", "1.44",],
            ["Gemma 2B", "46.37", "48.38", "71.77", "41.77", "33.08", "66.3", "16.91"],
            // ["Zephyr Danube SFT Qlora v0.1", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0"],
            // ["Zephyr Gemma 2B SFT Qlora v0.1", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0", "0.0"],
        ]},
    ],
    "Usage": [
        {"type": "text", "content": "First, you need to install the required dependencies. You can install them using the following command:"},
        {"type": "code", "content": "!pip install torch transformers peft"},
        {"type": "text", "content": "Next, import the necessary libraries:"},
        {"type": "code", "content": code_snippets['imports']},
        {"type": "text", "content": "Define the model checkpoint IDs:"},
        {"type": "code", "content": code_snippets['checkpoint']},
        {"type": "text", "content": "Load the model and tokenizer:"},
        {"type": "code", "content": code_snippets['model']},
        {"type": "text", "content": "Here is the generate function. You can also experiment with different generation configurations to explore further."},
        {"type": "code", "content": code_snippets['generate']},
        {"type": "text", "content": "Finally, you can generate responses to prompts by calling the above function and using the model specific prompt template:"},
        {"type": "code", "content": code_snippets['prompt']},
    ]
}

