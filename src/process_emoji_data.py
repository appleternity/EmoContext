import json
from data import load_data
from config import *
import os

def load_emoji_char():
    with open(os.path.join(data_path, "char.json"), 'r', encoding='utf-8') as infile:
        data = json.load(infile)
    return data

def build_emoji_dataset():
    data = load_data("train.txt")
    emoji_info = load_emoji_char()
    emoji_dict = {val["character"]:val["index"] for key, val in emoji_info.items()}

    text_data = data["1"] + " " + data["2"] + " " + data["3"]
    dataset = []
    for i, message in enumerate(text_data):
        print(i, len(data), end="\r")

        # check all emoji
        emoji_list = []
        for emoji, index in emoji_dict.items():
            if emoji in message:
                emoji_list.append(index)
        
        for emoji in emoji_list:
            label = data.loc[i, "label"]
            dataset.append([emoji, int(label), label_mapping_reverse[label]])
    
    with open(os.path.join(data_path, "emoji_dataset.json"), 'w', encoding='utf-8') as outfile:
        json.dump(dataset, outfile, indent=4)

def main():
    build_emoji_dataset()

if __name__ == "__main__":
    main()