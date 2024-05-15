let title = "Aligning Small LLMs";
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
model_revison = "model_revision"
`.trim(),

        "model": `
tokenizer = AutoTokenizer.from_pretrained(finetuned_model_id)
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
prompt = tokenizer.apply_chat_template(
    chat, tokenize=False, add_generation_prompt=True
)
response = generate_response(prompt)
print(response)
`.trim(),

}

function create_model_details_table(details) {
    return table = [
        ["Model", ...details['titles'].map((title) => title.replace(/_/g, " "))],
        ["Fine-Tuned Model", ...details['fine_tuned_models'].map((fine_tuned_model) => `<a href="https://huggingface.co/${fine_tuned_model}" target="_blank">${fine_tuned_model}</a>`)],
        ["Base Model", ...details['base_models'].map((base_model) => `<a href="https://huggingface.co/${base_model}" target="_blank">${base_model}</a>`)],
        ["Training Config", ...details['training_configs'].map((training_config) => `<a href="${training_config}" target="_blank">View Config</a>`)],
        ["Model Revision", ...details['revisions']]
        
    ];
}

let model_details = {
    'OLMo 1B' : {
        'titles': ['Zephyr OLMo 1B v0.1', 'OpenHermes OLMo 1B v0.1'],
        'base_models': ['allenai/OLMo-1B-hf', 'allenai/OLMo-1B-hf'],
        'fine_tuned_models': ['Ritvik19/zephyr-1b-olmo-sft-qlora', 'Ritvik19/openhermes-1b-olmo-sft-qlora'],
        'training_configs': [
            'https://github.com/Ritvik19/alignment-handbook/blob/main/recipes/zephyr-1b-olmo/sft/config_qlora.yaml',
            'https://github.com/Ritvik19/alignment-handbook/blob/main/recipes/openhermes-1b-olmo/sft/config_qlora.yaml'
    ],
        'revisions': ['b21cdb8d10251d5dc28156b5749ba2957ecfbe4d', 'a8403864e6c15759e116bd8a0acc35be3169afb3']
    },
    'TinyLlama 1.1B': {
        'titles': ['Zephyr TinyLlama v0.1', 'OpenHermes TinyLlama v0.1'],
        'base_models': ['TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T', 'TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T'],
        'fine_tuned_models': ['Ritvik19/zephyr-tinyllama-sft-qlora', 'Ritvik19/openhermes-tinyllama-sft-qlora'],
        'training_configs': [
            'https://github.com/Ritvik19/alignment-handbook/blob/main/recipes/zephyr-tinyllama/sft/config_qlora.yaml',
            'https://github.com/Ritvik19/alignment-handbook/blob/main/recipes/openhermes-tinyllama/sft/config_qlora.yaml'
        ],
        'revisions': ['002c335a34c5176e14c92875084764cee0f7be98', '078bd472c485667f13f833e6ea655984cf36d970']
    },
    'Phi-1.5 1.4B':{
        'titles': ['Zephyr Phi-1.5 v0.1', 'OpenHermes Phi-1.5 v0.1'],
        'base_models': ['microsoft/phi-1_5', 'microsoft/phi-1_5'],
        'fine_tuned_models': ['Ritvik19/zephyr-phi-1_5-sft-qlora', 'Ritvik19/openhermes-phi-1_5-sft-qlora'],
        'training_configs': [
            'https://github.com/Ritvik19/alignment-handbook/blob/main/recipes/zephyr-phi-1_5/sft/config_qlora.yaml',
            'https://github.com/Ritvik19/alignment-handbook/blob/main/recipes/openhermes-phi-1_5/sft/config_qlora.yaml',
        ],
        'revisions': ['a2d8bbc54949f5f580ac2e9f8c193bc8c3101304', '7d7e60ec181cc7f287683258b937bbb630aea9c7']
    },
    'Danube 1.8B': {
        'titles': ['Zephyr Danube v0.1', 'OpenHermes Danube v0.1'],
        'base_models': ['h2oai/h2o-danube-1.8b-base', 'h2oai/h2o-danube-1.8b-base'],
        'fine_tuned_models': ['Ritvik19/zephyr-danube-sft-qlora', 'Ritvik19/openhermes-danube-sft-qlora'],
        'training_configs': [
            'https://github.com/Ritvik19/alignment-handbook/blob/main/recipes/zephyr-danube/sft/config_qlora.yaml',
            'https://github.com/Ritvik19/alignment-handbook/blob/main/recipes/openhermes-danube/sft/config_qlora.yaml',
        ],
        'revisions': ['f89870505f24b92988e35543132bab68529ea1dd', '4812426de6ba1a70edf23f207c955106982cf29c']
    },
    'Danube2 1.8B': {
        'titles': ['Zephyr Danube v0.2', 'OpenHermes Danube v0.2'],
        'base_models': ['h2oai/h2o-danube2-1.8b-base', 'h2oai/h2o-danube2-1.8b-base'],
        'fine_tuned_models': ['Ritvik19/zephyr-danube2-sft-qlora', 'Ritvik19/openhermes-danube2-sft-qlora'],
        'training_configs': [
            'https://github.com/Ritvik19/alignment-handbook/blob/main/recipes/zephyr-danube/sft/config_qlora.yaml',
            'https://github.com/Ritvik19/alignment-handbook/blob/main/recipes/openhermes-danube/sft/config_qlora.yaml',
        ],
        'revisions': ['879fd07de6eeee108421b67bc431ad3cb6191b3c', 'bcec71aab0ec267d688a38dbb55751f27306bced']
    },
    'Gemma 2B':{
        'titles': ['Zephyr Gemma 2B v0.1', 'OpenHermes Gemma 2B v0.1'],
        'base_models': ['google/gemma-2b', 'google/gemma-2b'],
        'fine_tuned_models': ['Ritvik19/zephyr-2b-gemma-sft-qlora', 'Ritvik19/openhermes-2b-gemma-sft-qlora'],
        'training_configs': [
            'https://github.com/Ritvik19/alignment-handbook/blob/main/recipes/zephyr-2b-gemma/sft/config_qlora.yaml',
            'https://github.com/Ritvik19/alignment-handbook/blob/main/recipes/openhermes-2b-gemma/sft/config_qlora.yaml'
        ],
        'revisions': ['bd70703a95b854373446188b8033926913092470', '57fe9f69c0187be481170d884d1ababe0d0901ba']
    }
    
}

model_details_tables = {
    "OLMo 1B": create_model_details_table(model_details['OLMo 1B']),
    "TinyLlama 1.1B": create_model_details_table(model_details['TinyLlama 1.1B']),
    "Phi-1.5 1.4B": create_model_details_table(model_details['Phi-1.5 1.4B']),
    "Danube 1.8B": create_model_details_table(model_details['Danube 1.8B']),
    "Danube2 1.8B": create_model_details_table(model_details['Danube2 1.8B']),
    "Gemma 2B": create_model_details_table(model_details['Gemma 2B']),

}

let project_contents = {
    "TL;DR": [
        {"type": "text", "content": "This project centers on the alignment and fine-tuning of small language models (LlMs), drawing on insights from the Zephyr paper by Hugging Face and utilising the resources from the Alignment-Handbook repository. Our objective is to create detailed training recipes for small LLMs and provide aligned checkpoints for open-source distribution."},
        {"type": "text", "content": "We aim to fine-tune smaller LLMs utilizing a combination of tried-and-true methods and novel approaches. Our project embraces the principles of open-source development, ensuring that all training recipes and aligned checkpoints are accessible to the broader community. This approach promotes collaboration, enabling a continuous cycle of improvement and knowledge sharing among researchers and developers."},
        {"type": "text", "content": "Through this work, we offer a flexible framework for aligning small LLMs, designed for easy integration with various applications. The project's overarching goal is to advance the conversation on AI alignment while setting a benchmark for community-driven AI development."},

    ],
    "OLMo 1B" : [
        {"type": "text", "content": "OLMo is a 1B truly open language model and framework that includes training data, code, and tools for building, studying, and advancing language models."},
        {"type": "text", "content": "Read more about the model in my article <a href='https://ritvik19.medium.com/papers-explained-98-olmo-fdc358326f9b' target='_blank'>here</a>."},
        {"type": "table", "columns": model_details_tables["OLMo 1B"][0], "rows": model_details_tables["OLMo 1B"].slice(1)},
        {"type": "heading", "content": "Example Prompt"},
        {"type": "code", "content": '<|system|>\nYou are a helpful assistant.\n<|user|>\nHello, how are you?\n<|assistant|>\nI am well, thank you for asking. How are you doing?'},
    ],
    "TinyLlama 1.1B": [
        {"type": "text", "content": "TinyLlama is a 1.1B language model built upon the architecture and tokenizer of Llama 2, pre-trained on around 1 trillion tokens for approximately 3 epochs, leveraging FlashAttention and Grouped Query Attention, to achieve better computational efficiency."},
        {"type": "text", "content": "Read more about the model in my article <a href='https://ritvik19.medium.com/papers-explained-93-tinyllama-6ef140170da9' target='_blank'>here</a>."},
        {"type": "table", "columns": model_details_tables["TinyLlama 1.1B"][0], "rows": model_details_tables["TinyLlama 1.1B"].slice(1)},
        {"type": "heading", "content": "Example Prompt"},
        {"type": "code", "content": '<|system|>\nYou are a helpful assistant. \n<|user|>\nHello, how are you? \n<|assistant|>\nI am fine, thank you. How are you?'},
    ],
    "Phi-1.5 1.4B": [
        {"type": "text", "content": "Phi-1.5 is a 1.4B language model by Microsoft, focusing on common sense reasoning in natural language, trained using a textbook quality data from the web and synthetically generated textbooks and exercises with GPT-3.5."},
        {"type": "text", "content": "Read more about the model in my article <a href='https://ritvik19.medium.com/papers-explained-phi-1-5-2857e56dbd2a' target='_blank'>here</a>."},
        {"type": "table", "columns": model_details_tables["Phi-1.5 1.4B"][0], "rows": model_details_tables["Phi-1.5 1.4B"].slice(1)},
        {"type": "heading", "content": "Example Prompt"},
        {"type": "code", "content": '<|system|>\nYou are a helpful assistant.\n<|user|>\nHello, how are you?\n<|assistant|>\nI am doing well, thank you for asking. How can I assist you further?'},
    ],
    "Danube 1.8B": [
        {"type": "text", "content": "Danube is a 1.8B language model, by h20 ai, trained on 1T tokens following the core principles of LLama 2 and Mistral, leveraging and refining various techniques for pre-training large language models."},
        {"type": "text", "content": "Danube 2 is an updated version of the Danube model, with improvements including removal of sliding window attention, changes to the tokenizer, and adjustments to the training data, resulting in significant performance enhancements."},
        {"type": "text", "content": "Read more about the model in my article <a href='https://ritvik19.medium.com/papers-explained-111-h2o-danube-1-8b-b790c073d257' target='_blank'>here</a>."},
        {"type": "table", "columns": model_details_tables["Danube 1.8B"][0], "rows": model_details_tables["Danube 1.8B"].slice(1)},
        {"type": "table", "columns": model_details_tables["Danube2 1.8B"][0], "rows": model_details_tables["Danube2 1.8B"].slice(1)},
        {"type": "heading", "content": "Example Prompt"},
        {"type": "code", "content": '<|system|>\nYou are a helpful assistant. \n<|user|>\nHello, how are you? \n<|assistant|>\nI am doing well, thank you for asking.'},
    ],
    "Gemma 2B": [
        {"type": "text", "content": "Gemma is a 2B language models based on Google's Gemini models, offering advancements in language understanding, reasoning, and safety."},
        {"type": "text", "content": "Read more about the model in my article <a href='https://ritvik19.medium.com/papers-explained-106-gemma-ca2b449321ac' target='_blank'>here</a>."},
        {"type": "table", "columns": model_details_tables["Gemma 2B"][0], "rows": model_details_tables["Gemma 2B"].slice(1)},
        {"type": "heading", "content": "Example Prompt"},
        {"type": "code", "content": '<|system|>\nYou are a helpful assistant.\n<|user|>\nHello, how are you?\n<|assistant|>\nI am doing well, thank you. How are you?'},
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
            ["OLMo 1B", "36.73", "34.56", "63.6", "26.31", "32.92", "61.09", "1.9"],
            ["Zephyr OLMo 1B v0.1", "37.47", "36.26", "63.48", "27.28", "35.05", "60.14", "2.58"],
            ["OpenHermes OLMo 1B v0.1", "37.15", "33.19", "63.9", "25.67", "39.19", "60.93", "0.00"],
            ["TinyLlama 1.1B 3T", "36.42", "33.87", "60.31", "26.04", "37.32", "59.51", "1.44"],
            ["Zephyr TinyLlama v0.1", "36.64", "34.64", "59.84", "25.85", "36.57", "61.17", "1.74"],
            ["OpenHermes TinyLlama v0.1", "36.72", "32.34", "60.45", "27.67", "38.29", "61.56", "0.00"],
            ["Phi-1.5", "47.69", "52.9", "63.79", "43.89", "40.89", "72.22", "12.43"],
            ["Zephyr Phi-1.5 v0.1", "50.14", "51.96", "62.22", "43.09", "42.87", "73.09", "27.6"],
            ["OpenHermes Phi-1.5 v0.1", "49.49", "48.98", "62.14", "41.15", "42.36", "71.43", "30.86"],
            ["Danube 1.8B Base", "39.12", "39.42", "69.58", "25.94", "33.86", "64.48", "1.44"],
            ["Zephyr Danube v0.1", "40.11", "40.44", "69.4", "27", "37.08", "64.72", "2.05"],
            ["OpenHermes Danube v0.1", "38.75", "37.37", "69.45", "25.08", "35.28", "65.35", "0.00"],
            ["Danube 2 1.8B Base", "48.72", "43.34", "72.95", "40.2", "38.01", "68.03", "29.8"],
            ["Gemma 2B", "46.37", "48.38", "71.77", "41.77", "33.08", "66.3", "16.91"],
            ["Zephyr Gemma 2B v0.1", "47.26", "49.15", "71.94", "41.88", "35.77", "66.61", "18.2"],
            ["OpenHermes Gemma 2B v0.1", "43.87", "44.37", "71.58", "39.64", "40.09", "67.56", "0.00"],
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

