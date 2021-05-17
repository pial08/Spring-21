import torch
import torch.nn as nn
import torch.optim as optim
import spacy
from utils import translate_sentence, bleu, save_checkpoint, load_checkpoint
from torch.utils.tensorboard import SummaryWriter
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator
import models

"""
To install spacy languages do:
python -m spacy download en
python -m spacy download de
"""
spacy_ger = spacy.load("de")
spacy_eng = spacy.load("en")


def tokenize_ger(text):
    return [tok.text for tok in spacy_ger.tokenizer(text)]


def tokenize_eng(text):
    return [tok.text for tok in spacy_eng.tokenizer(text)]


german = Field(tokenize=tokenize_ger, lower=True, init_token="<sos>", eos_token="<eos>")

english = Field(
    tokenize=tokenize_eng, lower=True, init_token="<sos>", eos_token="<eos>"
)

train_data, valid_data, test_data = Multi30k.splits(
    exts=(".de", ".en"), fields=(german, english)
)


def make_src_mask(src, src_pad_idx):
        print("source......", src.transpose(0, 1))
        print("src pad index....", src_pad_idx)
        src_mask = src.transpose(0, 1) == src_pad_idx
        print(src_pad_idx)
        print("src_mask---------------------xxxxxxx-------------------------------")
        print(src_mask)
        print("------------------------------------------------------------")
        # (N, src_len)
        return src_mask.to(device)



def generate_square_subsequent_mask(sz: int) :
        
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask



german.build_vocab(train_data, max_size=10000, min_freq=2)
english.build_vocab(train_data, max_size=10000, min_freq=2)
src_pad_idx = english.vocab.stoi["<pad>"]
#print("-x-", english.vocab.itos)
print("train data", vars(train_data.examples[0]))
print("dict...", train_data[1].__dict__['src'])


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

load_model = False
save_model = True

# Training hyperparameters
num_epochs = 5
learning_rate = 3e-4
batch_size = 8

# Model hyperparameters
src_vocab_size = len(german.vocab)
trg_vocab_size = len(english.vocab)
embedding_size = 512
num_heads = 8
num_encoder_layers = 3
num_decoder_layers = 3
dropout = 0.10
max_len = 100
forward_expansion = 4
src_pad_idx = english.vocab.stoi["<pad>"]

# Tensorboard to get nice loss plot
writer = SummaryWriter("runs/loss_plot")
step = 0

train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=batch_size,
    sort_within_batch=True,
    sort_key=lambda x: len(x.src),
    device=device,
)
"""
model = Transformer(
    embedding_size,
    src_vocab_size,
    trg_vocab_size,
    src_pad_idx,
    num_heads,
    num_encoder_layers,
    num_decoder_layers,
    forward_expansion,
    dropout,
    max_len,
    device,
).to(device)
"""

model = models.make_model(len(english.vocab), len(german.vocab), 2)
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=0.1, patience=10, verbose=True
)

pad_idx = english.vocab.stoi["<pad>"]
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

if load_model:
    load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer)

sentence = "ein pferd geht unter einer brücke neben einem boot."
print("num epoch", num_epochs)


for epoch in range(num_epochs):
    print(f"[Epoch {epoch} / {num_epochs}]")

    if save_model:
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)
    """
    model.eval()
    translated_sentence = translate_sentence(
        model, sentence, german, english, device, max_length=50
    )

    print(f"Translated example sentence: \n {translated_sentence}")
    """
    model.train()
    losses = []

    for batch_idx, batch in enumerate(train_iterator):
        # Get input and targets and get to cuda

        inp_data = batch.src.to(device)
        
        target = batch.trg.to(device)
        print("Printing output data.............", target)
        target_sentence = [english.vocab.itos[idx] for idx in target.transpose(0, 1)[0]]
        target_truncated = [english.vocab.itos[idx] for idx in target[:-1, :].transpose(0, 1)[0]]
        print(f"Target sentence: \n {target_sentence}")
        print("Truncated Output sentence\n", target_truncated)

        # Forward prop
        print("value of src_pad_index= ", src_pad_idx)
        src_mask = make_src_mask(inp_data, src_pad_idx)

        trg_seq_length= len(target_sentence)
        trg_mask = generate_square_subsequent_mask(trg_seq_length).to(device)
        output = model(inp_data, target[:-1, :], src_mask, trg_mask)

        # Output is of shape (trg_len, batch_size, output_dim) but Cross Entropy Loss
        # doesn't take input in that form. For example if we have MNIST we want to have
        # output to be: (N, 10) and targets just (N). Here we can view it in a similar
        # way that we have output_words * batch_size that we want to send in into
        # our cost function, so we need to do some reshaping.
        # Let's also remove the start token while we're at it
        output = output.reshape(-1, output.shape[2])
        target = target[1:].reshape(-1)

        optimizer.zero_grad()

        loss = criterion(output, target)
        losses.append(loss.item())

        # Back prop
        loss.backward()
        # Clip to avoid exploding gradient issues, makes sure grads are
        # within a healthy range
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

        # Gradient descent step
        optimizer.step()

        # plot to tensorboard
        writer.add_scalar("Training loss", loss, global_step=step)
        step += 1
        val = input("Enter your value: ")

    mean_loss = sum(losses) / len(losses)
    scheduler.step(mean_loss)
    

# running on entire test data takes a while
score = bleu(test_data[1:100], model, german, english, device)
print(f"Bleu score {score * 100:.2f}")

