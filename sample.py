from model import GPT, GPTConfig
import tiktoken
import argparse
from torch.nn import functional as F
import torch

parser = argparse.ArgumentParser(description="GPT inference script")

parser.add_argument("--chkpt", type=str, help="Path to the checkpoint to use")
parser.add_argument("--prompt", type=str, help="Prompt to give to the model")
parser.add_argument("--max_toks", type=int, help="Maximum number of tokens to generate", default=50)
parser.add_argument("--n_samples", type=int, help="Number of prompt continuations to generate", default=1)
parser.add_argument("--top_k", type=int, help="The Top-K filter to apply", default=50)
parser.add_argument("--temperature", type=float, help="The temperature to use for inference", default=0.9)

args = parser.parse_args()

# Detect the device to use
device = "cuda" if torch.cuda.is_available() else "cpu"

# GPT-2 tokenizer
enc = tiktoken.get_encoding("gpt2")

# Load the model state dict from the checkpoint
chkpt = torch.load(args.chkpt, map_location=device)

# Instanciate the model
model = GPT(GPTConfig(vocab_size=50304))
new_dict = {
    k.replace("_orig_mod.", ""): v for k, v in chkpt["model"].items()
}  # remove compiler prefix
model.load_state_dict(new_dict)

# Tokenize the prompt
tokens = enc.encode(args.prompt)
tokens = torch.tensor(tokens, dtype=torch.long)
tokens = tokens.unsqueeze(0).repeat(args.n_samples, 1)
xgen = tokens.to(device)

sample_rng = torch.Generator(device=device)
while xgen.size(1) < args.max_toks:
    # Forward the model to get the logits
    with torch.no_grad():
        logits, loss = model(xgen) # (B, T, vocab_size)
        # Take the logits at the last position
        logits = logits[:, -1, :] # (B, vocab_size)
        # Apply the temperature parameter
        logits = logits / args.temperature
        # Get the probabilities
        probs = F.softmax(logits, dim=-1)
        # Do top-k sampling
        topk_probs, topk_indices = torch.topk(probs, args.top_k, dim=-1)
        # Select a token from the top-k probabilities
        # Note: multinomial does not demand the input to sum to 1
        ix = torch.multinomial(topk_probs, 1, generator=sample_rng)  # (B, 1)
        # Gather the corresponding indices
        xcol = torch.gather(topk_indices, -1, ix)  # (B, 1)
        # Append to the sequence
        xgen = torch.cat((xgen, xcol), dim=1)

if args.n_samples > 1:
    for i in range(args.n_samples):
        tokens = xgen[i, :args.max_toks].tolist()
        decoded = enc.decode(tokens)
        print(f"Sample {i}: {decoded}")
else:
    print(enc.decode(xgen[0].tolist()))
