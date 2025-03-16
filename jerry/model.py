import torch.nn as nn
from lib import gpt2
from torch.utils.data import DataLoader
import torch
from lib.layers import LayerNorm, GELU, MultiHeadAttention, FeedForward, TransformerBlock
import lib.common as common

def generate_and_print_sample(model, tokenizer, device, start_context):  
    context_size = model.pos_emb.weight.shape[0]
    # Generate a sample text
    idx = model.encode(start_context).to(device)
    with torch.no_grad():
        idx = common.generate_text_simple(
            model=model,
            idx=idx,
            max_new_tokens=50,
            context_size=context_size
        )

    # Decode the token indices to text
    text = model.decode(idx)
    print(text)

class GPT2Model(nn.Module):
    def __init__(self, cfg):
        super(GPT2Model, self).__init__()
        self.gpt2 = gpt2.GPT2()
        self.config = cfg

        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

    def use_device(self, device):
        self.to(device)
        self.device = device

    def encode(self, text):
        return self.gpt2.encode(text)

    def decode(self, tokens):
        return self.gpt2.decode(tokens)

    def load_model(self, path):
        self.load_state_dict(torch.load(path))

    def start_train(self, data):
        train_loader = self.create_data_loader(data)
        print("Training...")
        for x, y in train_loader:
            print(x.shape, y.shape)

        with torch.no_grad(): # Disable gradient tracking for efficiency because we are not training, yet
            train_loss = self.calc_loss_loader(train_loader, self, self.device)
            print("Train loss: ", train_loss)
        
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.0004, weight_decay=0.1)
        num_epochs = 10
        eval_freq = 5
        eval_iter = 5
        start_context = "Every effort moves you"
        self.train_model_simple(self, train_loader, None, optimizer, self.device, num_epochs, eval_freq, eval_iter, start_context, self.gpt2.tokenizer)
        torch.save(self.state_dict(), "./output/model.pth")
        print ("Training completed.")

    def create_data_loader(self, data):
        # Get config
        max_length = self.config["context_length"]
        stride = self.config["context_length"]
        batch_size = 2
        shuffle = True
        drop_last = True
        num_workers = 0

        # Create dataset
        dataset = gpt2.GPTDataset(data, self.gpt2.tokenizer, max_length, stride)

        # Create dataloader
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

        return dataloader

    def calc_loss_loader(self, data_loader, model, device, num_batches=None):
        total_loss = 0.
        if len(data_loader) == 0:
            return float("nan")
        elif num_batches is None:
            num_batches = len(data_loader)
        else:
            # Reduce the number of batches to match the total number of batches in the data loader
            # if num_batches exceeds the number of batches in the data loader
            num_batches = min(num_batches, len(data_loader))
        for i, (input_batch, target_batch) in enumerate(data_loader):
            if i < num_batches:
                loss = self.calc_loss_batch(input_batch, target_batch, model, device)
                total_loss += loss.item()
            else:
                break
        return total_loss / num_batches

    def calc_loss_batch(self, input_batch, target_batch, model, device):
        input_batch, target_batch = input_batch.to(device), target_batch.to(device)
        logits = model(input_batch)
        loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
        return loss

    def train_model_simple(self, model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer):
        # Initialize lists to track losses and tokens seen
        train_losses, val_losses, track_tokens_seen = [], [], []
        tokens_seen, global_step = 0, -1

        # Main training loop
        for epoch in range(num_epochs):
            model.train()  # Set model to training mode

            for input_batch, target_batch in train_loader:
                optimizer.zero_grad() # Reset loss gradients from previous batch iteration
                loss = self.calc_loss_batch(input_batch, target_batch, model, device)
                loss.backward() # Calculate loss gradients
                optimizer.step() # Update model weights using loss gradients
                tokens_seen += input_batch.numel()
                global_step += 1

                # # Optional evaluation step
                # if global_step % eval_freq == 0:
                #     train_loss, val_loss = evaluate_model(
                #         model, train_loader, val_loader, device, eval_iter)
                #     train_losses.append(train_loss)
                #     val_losses.append(val_loss)
                #     track_tokens_seen.append(tokens_seen)
                #     print(f"Ep {epoch+1} (Step {global_step:06d}): "
                #           f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

            # Print a sample text after each epoch
            model.eval()
            generate_and_print_sample(
                model, tokenizer, device, start_context
            )
            model.train()
            print (f"Epoch {epoch+1}/{num_epochs} completed.")

        return train_losses, val_losses, track_tokens_seen
