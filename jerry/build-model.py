import model
import torch
import config

def load_tran_data(path: str):
   with open(path, "r", encoding="utf-8") as file:
    text_data = file.read()
    return text_data


def main():
    torch.manual_seed(123)
    device = torch.device("mps")
    print("Using device: ", device)

    # Load traning data
    data = load_tran_data("./data/example.txt")

    # Create model
    gpt2 = model.GPT2Model(config.GPT_CONFIG_124M)
    gpt2.use_device(device)

    # Encode data
    tokens = gpt2.encode(data)

    # Print tokens
    print(tokens)

    # Train model
    gpt2.start_train(data)


if __name__ == "__main__":
    torch.manual_seed(123)
    main()