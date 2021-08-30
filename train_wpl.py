import torch
from src.aocr_wpl import OCR, OCRDataModule
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split


def train_val_dataset(dataset, val_split=0.25):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    datasets = {'train': Subset(dataset, train_idx), 'val': Subset(dataset, val_idx)}
    return datasets


def main():
    train_list = "./data/dataset/trainlist.txt"
    val_list = train_list  # just for now
    model = OCR()
    optimizer = torch.optim.Adadelta(model.parameters(), lr=0.001)
    ocr_datamodule_obj = OCRDataModule(train_list=train_list, val_list=val_list)
    train_dataloader_obj = ocr_datamodule_obj.train_val_dataloader()
    datasets = train_val_dataset(train_dataloader_obj)
    print(len(datasets['train']), len(datasets['val']))
    num_epochs = 2
    # training Loop
    for epochs in range(num_epochs):
        print("Epoch:", epochs, "of", num_epochs)
        print("Training for Epoch:", epochs)
        for (idx, train_batch) in enumerate(datasets['train'].dataset):
            # for x, y in batch:
            print("Training Batch number:", idx)
            cpu_images, cpu_texts = train_batch
            decoder_outputs, _ = model.forward(cpu_images, cpu_texts, is_training=True, return_attentions=False)
            target_variable = model.converter.encode(cpu_texts, model.device)

            loss = 0.0
            for di, decoder_output in enumerate(decoder_outputs, 1):
                loss += model.criterion(decoder_output, target_variable[di])
            print("Train Loss: ", loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print("Training Finished for epoch:", epochs)
        print("Validation started for epoch:", epochs)
        with torch.no_grad():
            for (idx, val_batch) in enumerate(datasets['val'].dataset):
                print("Validation Batch number:", idx)
                cpu_images, cpu_texts = val_batch
                decoder_outputs, _ = model.forward(cpu_images, cpu_texts, is_training=True, return_attentions=False)
                target_variable = model.converter.encode(cpu_texts, model.device)

                loss = 0.0
                for di, decoder_output in enumerate(decoder_outputs, 1):
                    loss += model.criterion(decoder_output, target_variable[di])
                print("Val Loss:", loss)
                log_dict = {
                    'val_loss':loss,
                    'val_wer': model.wer(decoder_outputs, target_variable),
                    'val_cer':model.cer(decoder_outputs, target_variable)
                }


main()
