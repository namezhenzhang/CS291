# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
import random
from tqdm import tqdm
from torch.utils.data import DataLoader
from datasets import load_dataset
import numpy as np
import json
from find_files import clone_and_extract_files
os.environ["TOKENIZERS_PARALLELISM"] = "false"

instrution = \
'''
Resolve the issue given by problem statement. Generate code patch following the output format.
'''.strip()

output_format = \
'''
diff --git a/path/to/file b/path/to/file
--- a/path/to/file
+++ b/path/to/file
@@ -x,y +x,y @@
- old content
+ new content
'''.strip()

template = \
'''# Instruction
{}

# Output Format
{}

# Repo Name
{}

# Problem Statement
{}

# Related File
## path:
{}
## context:
{}'''

preference_issue_template = \
'''# Repo Name
{}

# Problem Statement
{}'''

preference_file_template = \
'''# Related File
## path:
{}
## context:
{}'''

model_name = "meta-llama/Llama-3.2-1B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map='auto'
    )

tokenizer = AutoTokenizer.from_pretrained(model_name)
end_header_id = tokenizer.encode("<|end_header_id|>")[-1]

ds = load_dataset("princeton-nlp/SWE-bench")
train_dataset = ds['train'].select(range(10))



loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
all_issue_ppl = []
for data in tqdm(train_dataset):
    per_issue_ppl = []
    files = clone_and_extract_files(data['repo'],data['base_commit'])

    for file_path, file_context in tqdm(files):
        if file_context is None:
            continue
        prompt = template.format(instrution, output_format, data['repo'], data['problem_statement'], file_path, file_context)
        ground_truth = data['patch']
        chat = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": ground_truth}
        ]
        input_tokens = tokenizer.apply_chat_template(chat,tokenize=True)
        if len(input_tokens) > 20000:
            print("too long", end=" ")
            continue
        tmp_tokens = input_tokens[::-1]
        index = tmp_tokens.index(end_header_id)
        output_start_index = len(input_tokens) - (index-1)
        
        input_ids = torch.Tensor(input_tokens).to(device=model.device,dtype=int).unsqueeze(0)[:,:-1]
        output_token_label = torch.Tensor(input_tokens).to(device=model.device,dtype=int).unsqueeze(0)[:,1:]
        output_token_label[:,:output_start_index] = -100

        with torch.no_grad():
            model_output = model(input_ids,return_dict=True)

        loss = loss_fn(model_output.logits.reshape(-1,model_output.logits.shape[-1]),output_token_label.reshape(-1))
        ppl = torch.exp(loss)

        result = {
            "file_path": file_path,
            "ppl": ppl.item(),
            "file_context": file_context
        }
        result.update(data)
        per_issue_ppl.append(result)
    all_issue_ppl.append(per_issue_ppl)
preference_data = []
for per_issue_ppl in tqdm(all_issue_ppl):
    l = len(per_issue_ppl)
    for i in range(l-1):
        random_pair = random.randint(i + 1, l - 1)
        per_issue_ppl[i]
        per_issue_ppl[random_pair]
        if per_issue_ppl[i]['ppl'] < per_issue_ppl[random_pair]['ppl']:
            prefered = per_issue_ppl[i]
            rejected = per_issue_ppl[random_pair]
        else:
            prefered = per_issue_ppl[random_pair]
            rejected = per_issue_ppl[i]
        preference_data.append({
            "conversations": [
            {
                "from": "human",
                "value": preference_issue_template.format(prefered['repo'], prefered['problem_statement']),
            }
            ],
            "chosen": {
            "from": "gpt",
            "value": preference_file_template.format(prefered['file_path'], prefered['file_context']),
            },
            "rejected": {
            "from": "gpt",
            "value": preference_file_template.format(rejected['file_path'], rejected['file_context']),
            }
        })


output_file = "preference_data.json"
with open(os.path.join('data',output_file), "w") as f:
    json.dump(preference_data, f, indent=4)
