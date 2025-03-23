import torch
from data_util import load_data
from lib.model_provider import ModelProvider
from config import get_config

def main():
    device = torch.device("mps" if torch.mps.is_available() else "cpu")

    config = get_config()
    model_version = config["model_version"]
    dataset_version = config["dataset_version"]


    model_provider = ModelProvider()
    scam_sms_classifier = model_provider.get_model(model_version, config)
    scam_sms_classifier.set_device(device)
    scam_sms_classifier.load_state_dict(torch.load(f"output/model_v{model_version}_v{dataset_version}.pth"))

    _, test_data = load_data()
    scam_sms_classifier.start_eval(test_data)


if __name__ == '__main__':
    main()