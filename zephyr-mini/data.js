let title = "Zephyr Mini: Aligning Small LLMs";
let project_date = "May 2024"
let links = {
    "paper": "",
    "demo": "",
    "code": "https://github.com/Ritvik19/alignment-handbook",
    "model": "",
    "data": "",
}
let link2icon = {
    "code": "fas fa-code",
    "demo": "fas fa-terminal",
    "model": "fas fa-cogs",
    "data": "fas fa-database",
    "paper": "fas fa-file-pdf",
}

function create_model_details_table(details) {
    return table = [
        ["Model", ...details['titles'].map((title) => title.replace(/_/g, " "))],
        ["Fine-Tuned Model", ...details['fine_tuned_models'].map((fine_tuned_model) => `<a href="https://huggingface.co/${fine_tuned_model}" target="_blank">${fine_tuned_model}</a>`)],
        ["Training Approach", ...details['training_approaches'].map((training_approach) => training_approach.replace(/_/g, " "))],
        ["Base Model", ...details['base_models'].map((base_model) => `<a href="https://huggingface.co/${base_model}" target="_blank">${base_model}</a>`)],
        ["Tokenizer", ...details['tokenizers'].map((tokenizer) => `<a href="https://huggingface.co/${tokenizer}" target="_blank">${tokenizer}</a>`)],
        ["Training Config", ...details['training_configs'].map((training_config) => `<a href="${training_config}" target="_blank">View Config</a>`)],
        ["Model Revision", ...details['revisions']]
        
    ];
}

let model_details = {
    'TinyLLama': {
        'titles': ['Zephyr TinyLLama SFT Qlora v0.1'],
        'training_approaches': ['sft'],
        'base_models': ['TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T'],
        'tokenizers': ['Ritvik19/zephyr-tinylamma-sft-qlora'],
        'fine_tuned_models': ['Ritvik19/zephyr-tinylamma-sft-qlora'],
        'training_configs': ['https://github.com/Ritvik19/alignment-handbook/blob/main/recipes/zephyr-tinyllama/sft/config_qlora.yaml'],
        'revisions': ['v0.1']
    }
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
    temperature=0.0,
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
prompt = prompt_template.format(prompt=user_query)
response = generate_response(prompt)
print(response)
`.trim(),

}

let project_contents = {
    "Abstract": [
        {"type": "text", "content": "This project centers on the alignment and fine-tuning of small language models (LLMs), drawing on insights from the Zephyr paper by Hugging Face and utilising the resources from the Alignment-Handbook repository. Our objective is to create detailed training recipes for small LLMs and provide aligned checkpoints for open-source distribution."},
        {"type": "text", "content": "We aim to fine-tune smaller LLMs utilizing a combination of tried-and-true methods and novel approaches. Our project embraces the principles of open-source development, ensuring that all training recipes and aligned checkpoints are accessible to the broader community. This approach promotes collaboration, enabling a continuous cycle of improvement and knowledge sharing among researchers and developers."},
        {"type": "text", "content": "Through this work, we offer a flexible framework for aligning small LLMs, designed for easy integration with various applications. The project's overarching goal is to advance the conversation on AI alignment while setting a benchmark for community-driven AI development."},

    ],
    "TinyLLama": [
        {"type": "table", "columns": ["", ""], "rows": create_model_details_table(model_details['TinyLLama'])},
        {"type": "text", "content": "Prompt Template:"},
        {"type": "code", "content": `<|user|>\\n{prompt}\\n<|assistant>\\n`},
    ],
    "Results": [
        {"type": "text", "content": "This section is under development..."},
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
    ],
}

