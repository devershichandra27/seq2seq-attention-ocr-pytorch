import random
import torch
from torch import nn
from src.modules.encoder_wpl import Encoder
from src.modules.decoder_wpl import AttentionDecoder
from src.utils import utils, dataset
from src.utils.metrics import WER, CER


class OCR(nn.Module):
    def __init__(self,
                 img_height: int = 32,
                 img_width: int = 512,
                 enc_hidden_size: int = 256,
                 enc_seq_len: int = 128,
                 attn_dec_hidden_size: int = 128,
                 teaching_force_prob: float = 0.5,
                 dropout_p: float = 0.1,
                 output_pred_path: str = 'output_wpl.txt',
                 num_enc_rnn_layers: int = 2,
                 target_embedding_size: int = 10,
                 batch_size: int = 4
                 ):
        """OCR model

                        Args:
                            enc_hidden_size: size of the lstm hidden state
                            enc_seq_len: the width of the feature map out from cnn
                            batch_size: input batch size
                            img_height: the height of the input image to network
                            img_width: the width of the input image to network
                            teaching_forcing_prob: percentage of samples to apply teach forcing
                            dropout_p: Dropout probability in Decoder Dropout layer


                """
        super(OCR, self).__init__()
        self.batch_size = batch_size
        self.img_height = img_height
        self.img_width = img_width

        self.enc_hidden_size = enc_hidden_size
        self.enc_seq_len = enc_seq_len
        self.num_enc_rnn_layers = num_enc_rnn_layers
        self.enc_output_vec_size = self.enc_hidden_size * self.num_enc_rnn_layers

        self.attn_dec_hidden_size = attn_dec_hidden_size
        self.target_embedding_size = target_embedding_size

        self.teaching_forcing_prob = teaching_force_prob
        self.dropout_p = dropout_p
        self.output_pred_path = output_pred_path
        self.alphabet = utils.get_alphabet()
        self.num_classes = len(self.alphabet) + 3  # 0, SOS_TOKEN and EOS_TOKEN

        self.encoder = Encoder(image_channels=1, enc_hidden_size=self.enc_hidden_size)
        self.decoder = AttentionDecoder(attn_dec_hidden_size=self.attn_dec_hidden_size,
                                        enc_vec_size=self.enc_output_vec_size, enc_seq_length=self.enc_seq_len,
                                        target_embedding_size=self.target_embedding_size,
                                        target_vocab_size=self.num_classes, batch_size=self.batch_size)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.converter = utils.ConvertBetweenStringAndLabel(self.alphabet)
        self.wer = WER()
        self.cer = CER()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def forward(self, cpu_images, cpu_texts, is_training=True, return_attentions=False):
        self.batch_size = cpu_images.shape[0]
        encoder_outputs, state = self.encoder(cpu_images)
        state = utils.modify_state_for_tf_compat(state)
        self.decoder.set_encoder_output(encoder_outputs)
        attention_context = torch.zeros((self.batch_size, self.enc_output_vec_size), device=self.device)
        max_length = self.enc_seq_len

        if cpu_texts is not None:
            target_variable = self.converter.encode(cpu_texts, self.device)
            max_length = target_variable.shape[0]
            decoder_input = utils.get_one_hot(torch.tensor([utils.SOS_TOKEN] * self.batch_size, device=self.device),
                                              self.num_classes)

            if is_training:
                teach_forcing = True if random.random() > self.teaching_forcing_prob else False
            else:
                teach_forcing = False
        else:
            teach_forcing = False
            decoder_input = utils.get_one_hot(torch.tensor([1] * self.batch_size, device=self.device), self.num_classes)
        decoder_outputs = []

        for di in range(1, max_length):
            decoder_output, attention_context, state = self.decoder(decoder_input, attention_context, state)
            decoder_outputs.append(decoder_output)
            if teach_forcing and di != max_length -1:
                decoder_input = utils.get_one_hot(target_variable[di], self.num_classes)
            else:
                _, topi = decoder_output.data.topk(1)
                ni = topi.T[0]
                decoder_input = utils.get_one_hot(ni, self.num_classes)

                #STOP in EOS even in training?
                if not is_training:
                    if ni.item()==utils.EOS_TOKEN:
                        break

        if return_attentions:
            attention_matrix = self.decoder.attention_weights_history # 1DE
        else:
            attention_matrix = None
        return decoder_outputs, attention_matrix


class OCRDataModule(torch.utils.data.DataLoader):
    def __init__(self,
                 train_list: str = None,
                 val_list: str = None,
                 test_list: str =None,
                 img_height: int =32,
                 img_width: int = 512,
                 num_workers: int = 2,
                 batch_size: int = 4, # kept 1 because I'm dealing with batch sizes in train_wpl.py file
                 random_sampler: bool = True):
        """

                Args:
                    train_list: path to train dataset list file
                    val_list: path to validation dataset list file
                    test_list: path to test dataset list file
                    img_height: the height of the input image to network
                    img_width: the width of the input image to network
                    num_workers: number of data loading num_workers
                    batch_size: input batch size
                    random_sampler: whether to sample the dataset with random sampler
        """
        self.batch_size = batch_size
        self.train_list = train_list
        self.val_list = val_list
        self.test_list = test_list
        self.img_height = img_height
        self.img_width = img_width
        self.num_workers = num_workers
        self.random_sampler = random_sampler

        self.train_dataset = None
        self.training_sampler = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage=None):
        if self.train_list:
            self.train_dataset = dataset.TextLineDataset(text_line_file=self.train_list, transform=None)
            if self.random_sampler:
                self.training_sampler = dataset.RandomSequentialSampler(self.train_dataset, self.batch_size)

        if self.val_list:
            self.val_dataset = dataset.TextLineDataset(text_line_file=self.val_list,
                                                       transform=dataset.ResizeNormalize(img_width=self.img_width,
                                                                                         img_height=self.img_height))

        if self.test_list:
            self.test_dataset = dataset.TextLineDataset(text_line_file=self.test_list,
                                                        transform=dataset.ResizeNormalize(img_width=self.img_width,
                                                                   img_height=self.img_height))
    def train_val_dataloader(self):
        self.setup()
        return torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=False, sampler=self.training_sampler, num_workers = int (self.num_workers), collate_fn=dataset.AlignCollate(img_height=self.img_height, img_width=self.img_width)
        )

    def val_dataloader(self):
        self.setup()
        return torch.utils.data.DataLoader(self.val_dataset, shuffle=False, batch_size=1, num_workers=int(self.num_workers))

    def test_dataloader(self):
        self.setup()
        return torch.utils.data.DataLoader(self.test_dataset, shuffle=False, batch_size=1, num_workers=int(self.num_workers))
