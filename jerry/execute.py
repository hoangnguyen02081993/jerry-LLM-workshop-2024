from lib import gpt2
import torch
import tiktoken
import model
import config
from importlib.metadata import version
import lib.common as common
import datetime

print("TensorFlow version:", version("tensorflow"))
print("tqdm version:", version("tqdm"))

def main():
    torch.manual_seed(123)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Using device: ", device)


    gpt2 = model.GPT2Model(config.GPT_CONFIG_124M)
    gpt2.use_device(device)
    gpt2.load_model("./output/model.pth")

    idx = gpt2.encode("Every effort moves you ").to(device)

    print("Time before token generation", datetime.datetime.now())
    token_ids = common.generate_text_simple(
        model=gpt2,
        idx=idx,
        max_new_tokens=10,
        context_size=config.GPT_CONFIG_124M["context_length"]
    )
    print("Time after token generation", datetime.datetime.now())

    print(gpt2.decode(token_ids))


if __name__ == "__main__":
    main()