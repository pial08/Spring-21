import math
import spacy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator, bleu_score
from utils import translate_sentence, bleu, save_checkpoint, load_checkpoint, print_debug
#from torch.utils.tensorboard import SummaryWriter


print("hello")
#writer = SummaryWriter("runs/loss_plot")
step = 0


class TransformerModel(nn.Module):

    def __init__(self, 
        n_src_token, 
        n_tgt_token, 
        emsize, 
        nhead, 
        nhid, 
        nlayers,
        src_pad_idx, 
        device, 
        max_len, 
        dropout=0.5
    ):
        super(TransformerModel, self).__init__()

        self.device = device
        self.model_type = 'Transformer'
        #self.src_encoder = nn.Embedding(n_src_token, emsize)
        #self.tgt_encoder = nn.Embedding(n_tgt_token, emsize)
        #self.pos_encoder = PositionalEncoding(emsize, dropout)
        
        self.src_word_embedding = nn.Embedding(n_src_token, emsize)
        self.src_position_embedding = nn.Embedding(max_len, emsize)
        self.tgt_word_embedding = nn.Embedding(n_tgt_token, emsize)
        self.tgt_position_embedding = nn.Embedding(max_len, emsize)

        #self.encoder_layers = nn.TransformerEncoderLayer(emsize, nhead, nhid, dropout)
        #self.decoder_layers = nn.TransformerDecoderLayer(emsize, nhead, nhid, dropout)
        #self.transformer_encoder = nn.TransformerEncoder(self.encoder_layers, nlayers)
        #self.transformer_decoder = nn.TransformerDecoder(self.decoder_layers, nlayers)
        

        self.transformer = nn.Transformer(
            emsize,
            nhead,
            nlayers,
            nlayers,
            4,
            dropout,
        )
        
        
        self.ninp = emsize
        self.fc_out = nn.Linear(emsize, n_tgt_token)
        self.dropout = nn.Dropout(dropout)
        self.src_pad_idx = src_pad_idx

    def make_src_mask(self, src):
        #print("source......", src.transpose(0, 1))
        #print("src pad index....", self.src_pad_idx)
        src_mask = src.transpose(0, 1) == self.src_pad_idx
        #print(src_pad_idx)
        #print("src_mask---------------------xxxxxxx-------------------------------")
        #print(src_mask)
        #print("------------------------------------------------------------")
        # (N, src_len)
        return src_mask.to(self.device)

    def forward(self, src, tgt):
        src_seq_length, N = src.shape
        tgt_seq_length, N = tgt.shape
        

        #need to check
        print_debug(str(src.shape))
        #src = self.src_encoder(src) * math.sqrt(self.ninp)
        #src = self.pos_encoder(src)

        #tgt = self.tgt_encoder(tgt) * math.sqrt(self.ninp)
        #tgt = self.pos_encoder(tgt) 


        src_positions = (
            torch.arange(0, src_seq_length)
            .unsqueeze(1)
            .expand(src_seq_length, N)
            .to(self.device)
        )

        tgt_positions = (
            torch.arange(0, tgt_seq_length)
            .unsqueeze(1)
            .expand(tgt_seq_length, N)
            .to(self.device)
        )

        embed_src = self.dropout(
            (self.src_word_embedding(src) + self.src_position_embedding(src_positions))
        )
        embed_tgt = self.dropout(
            (self.tgt_word_embedding(tgt) + self.tgt_position_embedding(tgt_positions))
        )

        src_mask = self.make_src_mask(src)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(self, sz=tgt_seq_length).to(self.device)
        
        print_debug(str(str(src.shape) + " " + str(src_mask.transpose(0, 1).shape)))

        #encoder_output = self.transformer_encoder(src)
        #decoder_output = self.transformer_decoder(tgt, encoder_output, tgt_mask=tgt_mask)
        
        #output = self.dropout(self.fc_out(decoder_output))
        out = self.transformer(
            embed_src,
            embed_tgt,
            #src_key_padding_mask=src_padding_mask,
            tgt_mask=tgt_mask,
        )
        out = self.fc_out(out)
        
        return out


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)



##################  Generating Data ###########################
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


german.build_vocab(train_data, max_size=10000, min_freq=2)
english.build_vocab(train_data, max_size=10000, min_freq=2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
src_pad_idx = english.vocab.stoi["<pad>"]

n_src_token = len(german.vocab) # the size of vocabulary
n_tgt_token = len(english.vocab)
emsize = 512 # embedding dimension
nhid = 768 # the dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 6 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 8 # the number of heads in the multiheadattention models
dropout = 0.2 # the dropout value
max_len = 100
batch_size = 256
learning_rate = 3e-4
model = TransformerModel(n_src_token, n_tgt_token, emsize, nhead, nhid, nlayers, src_pad_idx, device, max_len, dropout)
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=0.1, patience=10, verbose=True
)

criterion = nn.CrossEntropyLoss(ignore_index=src_pad_idx)

print(model)


train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=batch_size,
    sort_within_batch=True,
    sort_key=lambda x: len(x.src),
    device=device,
)


num_epochs = 10
save_model = True
print("************************************************************************8")
sentence = "ein pferd geht unter einer brücke neben einem boot."
for epoch in range(num_epochs):
    print(f"[Epoch {epoch} / {num_epochs}]")

    if save_model:
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)

    model.eval()
    translated_sentence = translate_sentence(
        model, sentence, german, english, device, max_length=50
    )

    #print(f"Translated example sentence: \n {translated_sentence}")
    model.train()
    losses = []

    for batch_idx, batch in enumerate(train_iterator):
        # Get input and targets and get to cuda

        inp_data = batch.src.to(device)
        
        target = batch.trg.to(device)
        #print("Printing output data.............", target)
        target_sentence = [english.vocab.itos[idx] for idx in target.transpose(0, 1)[0]]
        target_truncated = [english.vocab.itos[idx] for idx in target[:-1, :].transpose(0, 1)[0]]
        #print(f"Target sentence: \n {target_sentence}")
        #print("Truncated Output sentence\n", target_truncated)

        # Forward prop
        output = model(inp_data, target[:-1, :])

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
        #writer.add_scalar("Training loss", loss, global_step=step)
        print("Training loss", loss)
        step += 1
        #val = input("Enter your value: ")

    mean_loss = sum(losses) / len(losses)
    scheduler.step(mean_loss)
    

# running on entire test data takes a while
score = bleu(test_data[1:100], model, german, english, device)
print(f"Bleu score {score * 100:.2f}")

