import datasets 
import json
import tqdm

dataset = datasets.load_dataset('hynky/czech_news_dataset_v2', split='train')

# go line by line, and write to a jsonl file the "text" field.
with open('czech_news_dataset_v2.jsonl', 'w') as f:
    for line in tqdm.tqdm(dataset):
        # make a json object with the text field
        json.dump({"text": line['content']}, f)
        # write a newline
        f.write('\n')

