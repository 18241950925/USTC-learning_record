from modelscope import AutoModelForCausalLM, AutoTokenizer
import json
import jsonlines
from tqdm import tqdm

device = "cuda"  # the device to load the model onto

input_file = '../data/valid.jsonl'

model = AutoModelForCausalLM.from_pretrained(
    "../qwen/Qwen1___5-1___8B-Chat",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("../qwen/Qwen1___5-1___8B-Chat")

response = ''
with jsonlines.open(input_file, 'r') as reader:
    for obj in tqdm(reader, desc="Processing entries"):
        # Extract human and chatgpt answers
        prompt = "对于一个计算机问题："
        question = obj['question'][0]
        human_answer = obj['human_answers'][0]  # Assuming only one human answer per question
        chatgpt_answer = obj['chatgpt_answers'][0]  # Assuming only one GPT answer per question
        prompt = prompt + question + "判断下面的回答是来自于人类还是GPT，若是人类则输出0，反之输出1,只输出0或1"
        prompt = prompt + "回答为：" + human_answer

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(device)

        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=100
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        res = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        response = response + res

        prompt = "对于一个计算机问题："
        prompt = prompt + question + "判断下面的回答是来自于人类还是GPT，若是人类则输出0，反之输出1"
        prompt = prompt + "回答为：" + chatgpt_answer

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(device)

        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=100
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        res = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        response = response + res
print(response)
