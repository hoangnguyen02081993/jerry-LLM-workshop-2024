import os
import torch
from datasets import load_dataset, load_from_disk

def download_and_load_data():
    if not os.path.exists("data/ucirvine/sms_spam"):
        data = load_dataset("ucirvine/sms_spam")
        data.save_to_disk("data/ucirvine/sms_spam")
    
    loaded_data = load_from_disk("data/ucirvine/sms_spam")
    return loaded_data

def save_model(model, path):
    if not os.path.exists("output"):
        os.makedirs("output")
    torch.save(model.state_dict(), f"./output/{path}")

def load_data():
    # Use dataset from Hugging Face
    # data = load_dataset("ucirvine/sms_spam")
    data = download_and_load_data()

    # Use 90% for training and 10% for testing
    dataset = data["train"].train_test_split(test_size=0.1)
    train_data = dataset["train"]
    test_data = dataset["test"]
    mapped_train_data = train_data.map(lambda x: {"text": x["sms"], "label": x["label"]})
    mapped_test_data = test_data.map(lambda x: {"text": x["sms"], "label": x["label"]})
    return mapped_train_data, mapped_test_data