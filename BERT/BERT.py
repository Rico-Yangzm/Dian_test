import json
import pandas as pd
import re
import numpy as np
import torch

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, BertForSequenceClassification


# get data
def load_data(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return pd.DataFrame(data)


# purify data
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[a~z\s]', '', text) # remove letters and
    return text


def filter_texts(df, min_length=4, max_length = 128):
    # sum up the number of characters
    df['char_count'] = df['text'].apply(lambda x: len([c for c in x if '\u4e00' <= c <= '\u9fff']))     #del too long or short
    filtered_df = df[(min_length <= df['char_count']) & (df['char_count'] <= max_length)].drop(columns=['char_count'])
    return filtered_df  # purified data



class CommentData(Dataset):
    def __init__(self, texts, points, tokenizer, max_len=128):
        self.texts = texts
        self.points = points
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)  #return len(self) when written len(commentdata)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        point = self.points[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,  #what Bert need to train (sign samples)
            max_length=self.max_len,
            padding='max_length',     #less than max_length add info
            truncation=True,          #cut when data is longer than self.max_len
            return_attention_mask=True, #return signs
            return_tensors='pt'       #return pytorch tensor(张量)
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),               #text list 1 dimension tensor
            'attention_mask': encoding['attention_mask'].flatten(),     #sign list 1 dimension tensor
            'labels': torch.tensor(point, dtype=torch.float)            #score list 1 dimension tensor
        }


#load local Bert model
local_model_path = "C:\\Users\\29955\\Downloads\\bert"
tokenizer = AutoTokenizer.from_pretrained(local_model_path)


class BertRegressor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertForSequenceClassification.from_pretrained(
            local_model_path,  # use local model
            num_labels=1       # regression task
        )

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        logits = outputs.logits
        if labels is not None:
            loss = torch.nn.MSELoss()(logits.squeeze(), labels)  #MSEloss to calculate the difference
            return loss, logits
        return (logits,)



df = load_data('comments_and_ratings.jsonl')
df['text'] = df['text'].apply(clean_text)
df = filter_texts(df)                           #initialize data
mean = df['point'].mean()
std = df['point'].std()
df['point'] = (df['point'] - mean) / std        #normalization
train_df, test_df = train_test_split(df, test_size=0.1)
train_dataset = CommentData(train_df['text'].values, train_df['point'].values, tokenizer, max_len=128)
test_dataset = CommentData(test_df['text'].values, test_df['point'].values, tokenizer, max_len=128)         #seperete
model = BertRegressor()         #initialize model


#change to get more accurate model

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-6, weight_decay=0.1)  # learning rate
evaluate_loss = torch.nn.MSELoss()  #loss model
num_train_epochs = 30                # training times
train_size = 24    # train parallel samples
evaluate_size = 24     # evaluate parallel samples
update_frequency = 1     # update each time


train_dataloader = DataLoader(train_dataset, batch_size=train_size, shuffle=True)
eval_dataloader = DataLoader(test_dataset, batch_size=evaluate_size, shuffle=False) #create dataloader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)            # train model using gpu

scaler = torch.amp.GradScaler('cuba')       #initialize GradScaler
best_eval_loss = float('inf')
patience = 5
counter = 0
best_epoch = 0

for epoch in range(num_train_epochs):
    model.train()
    for batch in train_dataloader:
        input_ids, attention_mask, labels = batch['input_ids'], batch['attention_mask'], batch['labels']        #load data
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)  # move to GPU
        optimizer.zero_grad()   #set grad as 0

        with torch.amp.autocast('cuda', enabled=True):  # 启用混合精度   using grad
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)    #training
            loss = outputs[0]       #get loss from outputs

        scaler.scale(loss).backward()  # 反向传播  amplify grad in case it will loss
        scaler.step(optimizer)  # 更新优化器 make sure the grad used by "torch.amp" is available
        scaler.update()  #  update


    # evaluation
    model.eval()
    eval_loss = 0
    threshold = 0.8
    all_preds = []
    all_labels = []     #initialize lists and parameters
    with torch.no_grad():   #baned grad
        for batch in eval_dataloader:
            input_ids, attention_mask, labels = batch['input_ids'], batch['attention_mask'], batch['labels']        #load data
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)  #move to gpu

            with torch.amp.autocast('cuda', enabled=True):
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs[0]
                logits = outputs[1]  # get predictions

            batch_preds = logits.squeeze().cpu().numpy() * std + mean
            batch_labels = labels.cpu().numpy() * std + mean    # move to cpu + denormalization

            all_preds.extend(batch_preds.tolist())
            all_labels.extend(batch_labels.tolist())    # change to list

            eval_loss += loss.item()    #store all losses

    eval_loss /= len(eval_dataloader)
    print(f'\nEpoch: {epoch + 1}')
    print(f'Eval Loss: {eval_loss:.4f}')

    mae = mean_absolute_error(all_labels, all_preds)    #平均值
    mse = mean_squared_error(all_labels, all_preds)     #均方差
    r2 = r2_score(all_labels, all_preds)                #决定系数
    print(f'MAE: {mae:.2f} | MSE: {mse:.2f} | R²: {r2:.2f}')

    correct = sum(abs(np.array(all_labels) - np.array(all_preds)) <= threshold)
    accuracy = correct / len(all_labels)
    print(f'Threshold Accuracy (±{threshold}): {accuracy:.2%}')

    print("\nSample Predictions:")
    for i in range(min(5, len(all_labels))):
        print(f" 真实评分: {all_labels[i]:.1f} | 预测评分: {all_preds[i]:.1f}")

    # choose the best model
    if eval_loss < best_eval_loss:
        best_eval_loss = eval_loss
        counter = 0
        best_epoch = epoch + 1
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        counter += 1
        if counter >= patience:
            print(f'Early stopping at epoch {epoch + 1}')
            print(f'best epoch is {best_epoch}')
            break


def predict_rating(text, model, tokenizer, max_len=128, mean=mean, std=std):
    model.eval()
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    with torch.no_grad():
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        output = model(input_ids, attention_mask)

    predicted_score = (output[0].squeeze().item() * std) + mean  # denormalize
    return predicted_score



# 示例预测
sample_text = "画面精美但剧情拖沓"
predicted_score = predict_rating(sample_text, model, tokenizer, max_len=128, mean=mean, std=std)
print(f"预测评分: {predicted_score:.1f}")