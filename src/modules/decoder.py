import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from src.modules.custom import Conv2dSamePadding


class AttentionDecoder(pl.LightningModule):
    """
        Use seq to seq model to modify the calculation method of attention weight
    """

    def __init__(self, attn_dec_hidden_size, enc_vec_size, enc_seq_length, target_embedding_size, target_vocab_size,
                 batch_size, LOAD_PATH=r'C:\Users\suman\iiith\attention-ocr-pytorch\models\decoder.pth'):
        super(AttentionDecoder, self).__init__()
        self.enc_vec_size = enc_vec_size  # 512
        self.enc_seq_length = enc_seq_length  # 128
        self.batch_size = batch_size
        self.attn_dec_hidden_size = attn_dec_hidden_size  # 128
        self.target_embedding_size = target_embedding_size  # 10
        self.cell_input_size = attn_dec_hidden_size
        self.target_vocab_size = target_vocab_size

        self.conv_1x1 = Conv2dSamePadding(self.enc_vec_size, self.enc_vec_size, kernel_size=(1, 1), stride=1,
                                          bias=False)
        self.embedding = nn.Embedding(self.target_vocab_size, self.target_embedding_size)

        self.attention_projection = nn.Linear(self.enc_vec_size, self.enc_vec_size)

        self.input_projection = nn.Linear(self.target_embedding_size + self.enc_vec_size, self.attn_dec_hidden_size)
        self.output_projection = nn.Linear(self.attn_dec_hidden_size + self.enc_vec_size, self.target_vocab_size)

        self.VT = nn.Linear(self.enc_vec_size, 1, bias=False)

        self.rnn = nn.LSTM(input_size=self.cell_input_size, hidden_size=self.attn_dec_hidden_size, num_layers=2)

        self.attention_weights_history = []
        self.encoder_output = None

        if LOAD_PATH:
            self.load_state_dict(torch.load(LOAD_PATH))

    def attention(self, state_flat):
        attention_proj_out = self.attention_projection(state_flat)
        attention_proj_out = attention_proj_out.view(-1, 1, 1, self.enc_vec_size)

        conv_inp = self.encoder_output.unsqueeze(0)  # 1 128 4 512
        conv_inp = conv_inp.permute(2, 3, 0, 1)  # 4 512 1 128
        hf = self.conv_1x1(conv_inp)
        hf = hf.permute(0, 3, 2, 1)
        hy = hf + attention_proj_out
        hy = hy.squeeze(2)

        th = torch.tanh(hy)
        e = self.VT(th)

        alpha = torch.softmax(e, dim=1)
        alpha = alpha.permute(0, 2, 1)
        hidden = self.encoder_output.permute(1, 0, 2)
        context = torch.bmm(alpha, hidden)

        return context, alpha

    def forward(self, prev_output, attention_context, state):
        prev_symbol = torch.argmax(prev_output, dim=1)
        prev_emb = self.embedding(prev_symbol)
        inp_proj_inp = torch.cat((prev_emb, attention_context), dim=1)

        cell_inp = self.input_projection(inp_proj_inp)
        cell_inp = cell_inp.unsqueeze(0)
        cell_output, state = self.rnn(cell_inp, state)
        h, c = state
        state_flat = torch.cat((c[0], h[0], c[1], h[1]), dim=1)

        attention_context, attention_weights = self.attention(state_flat)
        self.attention_weights_history.append(attention_weights)
        cell_output = cell_output.permute(1, 0, 2)
        out_proj_inp = torch.cat((cell_output, attention_context), dim=2)

        output = self.output_projection(out_proj_inp)
        output = output.squeeze(1)
        attention_context = attention_context.squeeze(1)

        return output, attention_context, state

    def set_encoder_output(self, encoder_output):
        self.encoder_output = encoder_output
