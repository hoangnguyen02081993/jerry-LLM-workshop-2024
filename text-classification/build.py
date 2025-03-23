import torch
from data_util import save_model, load_data
from config import get_config
from lib.model_provider import ModelProvider

def main():
    print("MPS Available:", torch.backends.mps.is_available())
    print("MPS Built:", torch.backends.mps.is_built())

    # device = torch.device("mps" if torch.mps.is_available() else "cpu")
    device = "cpu"
    torch.empty(1, device=device)  # Force MPS memory allocation

    config = get_config()
    model_version = config["model_version"]
    dataset_version = config["dataset_version"]

    model_provider = ModelProvider()
    scam_sms_classifier = model_provider.get_model(model_version, config)
    scam_sms_classifier.set_device(device)

    train_data, _ = load_data()
    scam_sms_classifier.start_train(train_data, config["lr"], config["epochs"])
    save_model(scam_sms_classifier, f"model_v{model_version}_v{dataset_version}.pth")


if __name__ == '__main__':
    main()