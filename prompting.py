'''

Run this file on terminal and pass your together.ai api key as an argument. Following is the format for the command to run this file:

python prompting.py YOUR_API_KEY

'''
from openai import OpenAI
import os
from tqdm import tqdm
import json
import os
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix

import argparse

#getting apikey

MODEL = "mistralai/Mistral-7B-Instruct-v0.2"

parser = argparse.ArgumentParser(description="Process OpenAPI key")
parser.add_argument("api_key", type=str, help="Your OpenAPI key")
args = parser.parse_args()
TOGETHER_API_KEY = args.api_key

current_working_directory = os.getcwd()
f = open(current_working_directory + '/SHROOM_test-labeled/test.model-aware.json')

model_aware_data = json.load(f)
parsed_data = []
for i in model_aware_data:
    datum = {}
    for key in i:
        datum[key] = i[key]
    parsed_data.append(datum)
f.close()

task = 'MT' # change task as per requirement

pd.set_option('display.max_colwidth', None)
df_aware = pd.DataFrame(model_aware_data)
df_task = df_aware[df_aware["task"]==task]

src = df_task.get('src').tolist()
hyp = df_task.get('hyp').tolist()
tgt = df_task.get('tgt').tolist()

# Hallucination generation
with open(f'llama_hallus_{task}.txt', 'w') as f:
  for instruction in tqdm(src[:10]):
    client = OpenAI(
      api_key=TOGETHER_API_KEY,
      base_url='https://api.together.xyz/v1',
    )
    hallu_score = 0
    for _ in range(5):
      chat_completion = client.chat.completions.create(
          messages=[
              {
                  "role": "system",
                  "content": "You are an hallucination detection agent - I will give you a source text and a hypothesis. Your job is to reply with either 0 or 1. 0 Indicating hallucination and 1 indicating non-hallucination. I only want the number in your reply without any suffixes or prefixes."
              },
              {
                  "role": "user", 
                  "content": content,
              }
          ],
          model=MODEL
      )
      if '1' not in chat_completion.choices[0].message.content:
        hallu_score += 1
    f.write(hallu_score)

# Prompting classification
print('Classifications:')
def write_to_file(res, file_name):
    with open(file_name, 'a') as f:
        json.dump(res, f)
        f.write('\n')

if task == 'DM':
  iterables = zip(src, hyp, tgt)
elif task == 'PG':
  iterables = zip(src, hyp)
else:
  iterables = zip(tgt, hyp)

for idx, item in enumerate(tqdm(iterables)):

    if task == 'PG':
      content = f"src: {item[0]}, hyp: {item[1]}"
    elif task == 'MT':
      content = f"src: {item[0]}, hyp: {item[1]}"
    else:
      content = f"src: {item[0]}, hyp: {item[1]}, tgt: {item[2]}"

    client = OpenAI(
        api_key=TOGETHER_API_KEY,
        base_url='https://api.together.xyz/v1',
    )
    hallu_score = 0
    for _ in range(5):
      chat_completion = client.chat.completions.create(
          messages=[
              {
                  "role": "system",
                  "content": "You are an hallucination detection agent - I will give you a source text and a hypothesis. Your job is to reply with either 0 or 1. 0 Indicating hallucination and 1 indicating non-hallucination. I only want the number in your reply without any suffixes or prefixes."
              },
              {
                  "role": "user", 
                  "content": content,
              }
          ],
          model=MODEL
      )
      if '1' not in chat_completion.choices[0].message.content:
        hallu_score += 1
    print(hallu_score/5)
    print(chat_completion.choices[0].message.content)
    if idx % 1 == 0:
        write_to_file(hallu_score/5, f'{task}_prompting.json')

print('Saved... Good Night. :)')

# scoring
preds = []
with open(f'{task}_prompting.json', 'r') as f:
    lines = f.readlines()
    for line in lines:
        preds.append(np.round(float(line.strip())))

preds = np.array(preds)

if task == 'PG':
  print('Paraphrase Generation')
elif task == 'DM':
  print('Definition Modelling')
else:
  print('Machine Translation')

print('f1_score =', f1_score(preds == 'Hallucination', df_task['label'] == 'Hallucination'))
print('accuracy_score =', accuracy_score(preds == 'Hallucination', df_task['label'] == 'Hallucination'))
print('precision_score =', precision_score(preds == 'Hallucination', df_task['label'] == 'Hallucination'))
print('recall_score =', recall_score(preds == 'Hallucination', df_task['label'] == 'Hallucination'))
print(confusion_matrix(preds == 'Hallucination', df_task['label'] == 'Hallucination'))