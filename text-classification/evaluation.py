from lib.model import ScamSMSClassifier
import torch
from data_util import download_and_load_data


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

def main():
    config = {
        "vocab_size": 50257,
        "emb_dim": 100,
        "hidden_dim": 200,
        "num_classes": 2,
        "epochs": 10
    }

    device = torch.device("mps" if torch.cuda.is_available() else "cpu")

    scam_sms_classifier = ScamSMSClassifier(config["vocab_size"], config["emb_dim"], config["hidden_dim"], config["num_classes"])
    scam_sms_classifier.set_device(device)
    scam_sms_classifier.load_state_dict(torch.load("output/model.pth"))

    _, test_data = load_data()
    scam_sms_classifier.start_eval(test_data)


if __name__ == '__main__':
    main()