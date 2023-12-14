import json

import openai
from torch.utils.data import Dataset
from tqdm import tqdm
import time

# Replace 'your-api-key' with your actual OpenAI API key
api_key = ''


def paraphrase_string(input_string):
    # Set your OpenAI API key
    openai.api_key = api_key

    # Specify the prompt for paraphrasing
    prompt = f"Paraphrase the following sentence:\n'{input_string}'\nParaphrased sentence:"

    # Use the OpenAI Completions API to generate a paraphrased version
    response = openai.Completion.create(
        engine="text-davinci-002",  # You can try other engines as well
        prompt=prompt,
        temperature=0.7,
        max_tokens=100,
        n=1,
        stop=None,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )

    # Extract the paraphrased sentence from the API response
    paraphrased_sentence = response['choices'][0]['text'].strip()

    print(f"\n{input_string=} <-> {paraphrased_sentence=}")

    return paraphrased_sentence


def load_jsonl(filename):
    with open(filename, "r") as f:
        return [json.loads(l.strip("\n")) for l in f.readlines()]


def save_jsonl(data, filename):
    """data is a list"""
    with open(filename, "w") as f:
        f.write("\n".join([json.dumps(e) for e in data]))


def count_words(input_string):
    # Split the string into words using spaces as separators
    words = input_string.split()

    # Count the number of words
    word_count = len(words)

    return word_count


def get_paraphrased_strings():
    input_file = "../data/highlight_train_release_paraphrased.jsonl"
    json_data = load_jsonl(input_file)
    data_dict = {}

    print("Loading data...")
    for data in tqdm(json_data):
        data_dict[f"{data['qid']}_{data['aug_id']}"] = data

    print("Processing data...")
    for data in tqdm(json_data):
        if count_words(data['query']) > 32:
            print(f"found problematic query: {data['qid']=} {data['aug_id']=} -> {data['query']}")
            if data['aug_id'] == 1:
                data['query'] = paraphrase_string(data_dict[f"{data['qid']}_0"]['query'])
                time.sleep(0.5)

    save_jsonl(json_data, input_file+"1")


if __name__ == "__main__":
    get_paraphrased_strings()
