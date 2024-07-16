import jsonlines
import csv
from tqdm import tqdm

input_file = '../data/valid.jsonl'  # Replace with your JSON lines file path
output_file = '../data/ceval/default_val.csv'  # Output CSV file path

id_counter = 1  # Initialize id counter starting from 1

# Open input JSON lines file for reading
with jsonlines.open(input_file, 'r') as reader:
    # Open output CSV file for writing
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        # Define CSV writer
        csv_writer = csv.writer(csvfile)
        # Write CSV header
        csv_writer.writerow(['id', 'question', 'A', 'B', 'C', 'D', 'answer', 'explanation'])

        # Iterate through each JSON object in the input file
        for obj in tqdm(reader, desc="Processing entries"):
            # Assuming each object has 'question', 'human_answers', and 'chatgpt_answers' fields

            human_answer = obj['human_answers'][0]  # Assuming only one human answer per question
            chatgpt_answer = obj['chatgpt_answers'][0]  # Assuming only one GPT answer per question
            question = f"针对一个计算机知识的提问所给出的回答：{human_answer}这句话是由____回答的。"
            answer = 'A'

            # Write the formatted row to CSV
            csv_writer.writerow([id_counter, question, 'human', 'chatgpt', '', '', answer, obj['question']])

            # Increment id counter
            id_counter += 1
            question = f"针对一个计算机知识的提问所给出的回答：{chatgpt_answer}这句话是由____回答的。"

            answer = 'B'
            csv_writer.writerow([id_counter, question, 'human', 'chatgpt', '', '', answer, obj['question']])
            id_counter += 1
print(f"CSV file '{output_file}' has been successfully created.")