import pandas as pd
import numpy as np
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from transformers import BertModel, BertTokenizer
import torch.optim as optim
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import argparse


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data):
       author_set = list(set(data['author'].to_list()))
       author_dict = dict()
       for i, a in enumerate(author_set):
         author_dict[a] = i
        
       self.texts = data['comment'].to_list()
       self.coco = data['cocogen'].to_list()
          

          
       self.y = [author_dict[a] for a in data['author'].to_list()]


    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts[index]
        numeric = self.coco[index]

        y = self.y[index]

        return text, numeric, y


class ClassifierHybrid(torch.nn.Module):
    def __init__(self):
        super(ClassifierHybrid, self).__init__()
        self.lstm = nn.LSTM(input_size=354, hidden_size=400, num_layers=3, batch_first=True, dropout=0.3)
        self.hidden_layer = nn.Linear(1200,300)
        self.classification_layer = nn.Linear(1068, 19)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        self.reg_drop = nn.Dropout(p=0.7)
    
    def forward(self, texts, coco_matrix):
        o1, (o2, o3) = self.lstm(coco_matrix)
        hidden = torch.flatten(o2).unsqueeze(dim=0)
        #print(hidden.shape)
        hidden = self.hidden_layer(hidden)
        
        tokenized_input = self.tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors='pt').input_ids.to(device)
        bert_rep = self.bert(tokenized_input).pooler_output
        hidden = torch.cat((hidden, bert_rep), dim=1)

        result = self.classification_layer(hidden)

        return result


def run(args):
    df= pd.read_pickle('reddit_final.pkl')

    df['cocogen'] = df['cocogen'].apply(lambda v: np.nan_to_num(v, copy=True, nan=0.0, posinf=None, neginf=None))

    device = f'cuda:{args.device}'
    binary_prediction = True
    params = {'batch_size': 1,
            'shuffle': True,
            'num_workers': 0}

    
    if args.cross_domain:
        df_train, df_test = df[df['domain'] == 'books'], df[df['domain'] == 'other']
    else:
        df_train, df_test = train_test_split(df, test_size=0.2)


    train_data = Dataset(df_train)
    train_loader = torch.utils.data.DataLoader(train_data, **params)

    test_data = Dataset(df_test)
    test_loader = torch.utils.data.DataLoader(test_data, **params)

    model = ClassifierHybrid().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    loss_f = torch.nn.CrossEntropyLoss()

    for epoch in range(100):
        print()
        print()
        print()
        print(f'training epoch {epoch}')

        model.train()
        correct_predictions = 0
        predictions = []
        truth_labels = []
        iter = 0
        for texts, numeric, label in train_loader:
            iter += 1
            label = label.long()
            model_output = model(texts, numeric.float().to(device))

            loss = loss_f(model_output.cpu(), label)
            #print('pred', model_output.cpu())
            #print('label', label)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            preds = torch.argmax(model_output, dim=1)
            predictions.extend(preds.tolist())
            truth_labels.extend(label.tolist())
        scheduler.step()

        print('training accuracy ', accuracy_score(predictions, truth_labels))
        print('precision', precision_score(predictions, truth_labels, average='macro'))
        print('recall', recall_score(predictions, truth_labels, average='macro'))
        print('f1', f1_score(predictions, truth_labels, average='macro'))
        print()

        model.eval()
        correct_predictions = 0
        predictions = []
        truth_labels = []
        iter = 0
        for texts, numeric, label in test_loader:
            iter += 1
            label = label.long()
            model_output = model(texts, numeric.float().to(device))

            loss = loss_f(model_output.cpu(), label)

            preds = torch.argmax(model_output, dim=1)
            predictions.extend(preds.tolist())
            truth_labels.extend(label.tolist())

        print('testing accuracy ', accuracy_score(predictions, truth_labels))
        print('testing precision', precision_score(predictions, truth_labels, average='macro'))
        print('testing recall', recall_score(predictions, truth_labels, average='macro'))
        print('testing f1', f1_score(predictions, truth_labels, average='macro'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gen_data', action='store_true')
    parser.add_argument('--device', type=int)
    parser.add_argument('--pos-file', type=str, default='')
    parser.add_argument('--neg-file', type=str, default='')
    args = parser.parse_args()