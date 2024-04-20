from transformers import DistilBertTokenizer, DistilBertModel
from sentence_transformers import SentenceTransformer, models
from torch import nn
import torch.nn as nn
import torch.optim as optim
import torch.nn as nn
from sentence_transformers import models
from transformers import AutoTokenizer
from transformers import BertTokenizer, BertModel
import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity
import json
from sentence_transformers import SentenceTransformer, util


train_models=False
'''
Loading dataframes by normalizing the sentence similarity score.
''' 

def set_up_stsb(train_csv,validation_csv,test_csv,batch_size):
    df=pd.read_csv(train_csv)
    df=df.drop("idx",axis=1)
    df_cosine_labels = df.copy() 
    df_cosine_labels["label"] = df_cosine_labels["label"] /df_cosine_labels["label"].abs().max() 

    df=pd.read_csv(validation_csv)
    df=df.drop("idx",axis=1)
    validation_cosine_labels = df.copy() 
    validation_cosine_labels["label"] = validation_cosine_labels["label"] /validation_cosine_labels["label"].abs().max() 

    df=pd.read_csv(test_csv)
    df=df.drop("idx",axis=1)
    test_cosine_labels = df.copy() 
    test_cosine_labels["label"] = test_cosine_labels["label"] /test_cosine_labels["label"].abs().max() 


    train_examples = []
    for i,row in df_cosine_labels.iterrows():
        train_examples.append([[row["sentence1"], row["sentence2"]],row["label"]])

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)

    val_examples = []
    for i,row in validation_cosine_labels.iterrows():
        val_examples.append([[row["sentence1"], row["sentence2"]],row["label"]])

    val_dataloader = DataLoader(val_examples, shuffle=True, batch_size=batch_size)

    test_examples = []
    for i,row in test_cosine_labels.iterrows():
        test_examples.append([[row["sentence1"], row["sentence2"]],row["label"]])

    test_dataloader = DataLoader(test_examples, shuffle=True, batch_size=batch_size)

    return train_dataloader,val_dataloader,test_dataloader

def evaluate_shroom(model,val_dataloader):
    criterion = nn.MSELoss()

    val_loss = 0.0
    with torch.no_grad():
        for val_data in val_dataloader:
            val_sentences, val_labels = val_data
            val_sentence1, val_sentence2 = encode_pair(val_sentences)
            val_labels = val_labels.float()
            val_outputs = model(val_sentence1, val_sentence2)
            val_loss += criterion(val_outputs, val_labels).item()
    val_loss /= len(val_dataloader)
    print(f"Test Loss : {val_loss}")

train_dataloader,val_dataloader,test_dataloader= set_up_stsb("stsb_train.csv","stsb_validation.csv","stsb_test.csv",50)

'''
Initial ShroomFormer:
    • Used pretrained models from sentence-bert
'''
class Shroomformer(nn.Module):
    def __init__(self, sequence_length):
        super(Shroomformer, self).__init__()
        self.sequence_length = sequence_length
        self.word_embedding_model = models.Transformer("bert-base-uncased", max_seq_length=self.sequence_length)
        self.pooling_model = models.Pooling(self.word_embedding_model.get_word_embedding_dimension())
        self.twin1 = nn.Sequential(
            self.word_embedding_model,
            self.pooling_model
        )
        self.twin2 = nn.Sequential(
            self.word_embedding_model,
            self.pooling_model
        )
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
           
    def forward(self, sentence1_encoded, sentence2_encoded):
        output1 = self.twin1(sentence1_encoded)["sentence_embedding"]
        output2 = self.twin2(sentence2_encoded)["sentence_embedding"]
        similarity_score = self.cos(output1, output2)
        return similarity_score


# def encode_pair(pair,padding=True,truncation=True,max_length=128):
#     tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
#     return tokenizer(
#     pair[0], padding=padding,truncation=truncation,max_length=max_length, return_tensors="pt"
#     ), tokenizer(
#         pair[1],padding=padding,truncation=truncation,max_length=max_length, return_tensors="pt"
#     )

if train_models:
    model = Shroomformer(128)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    num_epochs=500

    from tqdm import tqdm
    for epoch in tqdm(range(num_epochs)):
        running_loss = 0.0
        correct = 0
        total = 0

        for i, data in enumerate(train_dataloader, 0):
            if i%100==0:
                print(f"Batch: {i+1}")
            sentences, labels = data
            optimizer.zero_grad()
            sentence1,sentence2=encode_pair(sentences)
            outputs = model(sentence1,sentence2)
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print()
        print(f'Epoch {epoch+1} | Loss: {running_loss / len(train_dataloader)}')



'''
ShroomFormer using a downsampler: 
    • Too many parameters, not good performance.
'''


class ShroomformerV2(nn.Module):
    def __init__(self):
        super(ShroomformerV2, self).__init__()
        self.twin1 = BertModel.from_pretrained("bert-base-uncased")
        self.twin2 = BertModel.from_pretrained("bert-base-uncased")
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    
    def mean_pooling(self,word_embeddings, attention_mask):
        token_embeddings = word_embeddings[0] 
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

  
    def forward(self, sentence1_encoded, sentence2_encoded):
        word_embeddings_1 = self.twin1(**sentence1_encoded)
        sentence_embeddings_1=self.mean_pooling(word_embeddings_1, sentence1_encoded["attention_mask"])
        word_embeddings_2 = self.twin2(**sentence2_encoded)
        sentence_embeddings_2=self.mean_pooling(word_embeddings_2, sentence2_encoded["attention_mask"])
        similarity_score = self.cos(sentence_embeddings_1,sentence_embeddings_2)
        return similarity_score
       


def encode_pair(pair, padding=True, truncation=True, max_length=128):
    tokenizer=DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    encoded_pair_1 = tokenizer(
        pair[0], 
        padding=padding, 
        truncation=truncation, 
        max_length=max_length, 
        return_tensors="pt"
    )
    encoded_pair_2 = tokenizer(
        pair[1],
        padding=padding,
        truncation=truncation,
        max_length=max_length,
        return_tensors="pt"
    )
    return encoded_pair_1, encoded_pair_2
if train_models:
    model = ShroomformerV2()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00001)
    num_epochs=100
    best_val_loss = float('inf')
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total parameters in the model:", total_params)
    for epoch in tqdm(range(num_epochs)):
        running_loss = 0.0
        correct = 0
        total = 0
        model.train()
        for i, data in enumerate(train_dataloader, 0):
            if i%100==0:
                print(f"Batch: {i+1}")
            sentences, labels = data
            optimizer.zero_grad()
            sentence1,sentence2=encode_pair(sentences)
            labels = labels.float()
            outputs=model(sentence1,sentence2)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_data in val_dataloader:
                val_sentences, val_labels = val_data
                val_sentence1, val_sentence2 = encode_pair(val_sentences)
                val_labels = val_labels.float()
                val_outputs = model(val_sentence1, val_sentence2)
                val_loss += criterion(val_outputs, val_labels).item()
        val_loss /= len(val_dataloader)

        print(f'Epoch {epoch+1} | Train Loss: {running_loss / len(train_dataloader)} | Validation Loss: {val_loss}')



class ShroomForwardFormer(nn.Module):
    def __init__(self,max_seq_length):
        super(ShroomForwardFormer, self).__init__()
        self.word_embedding_model = models.Transformer("distilbert-base-uncased", max_seq_length=max_seq_length)
        self.pooling_model = models.Pooling(self.word_embedding_model.get_word_embedding_dimension())
        self.dense_model = models.Dense(
            in_features=self.pooling_model.get_sentence_embedding_dimension(),
            out_features=256,
            activation_function=nn.Tanh(),
        )
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward_once(self, x):
        embedding = self.word_embedding_model(x)
        embedding = self.pooling_model(embedding)
        embedding = self.dense_model(embedding)
        return embedding

    def forward(self, sentence1, sentence2):
        output1 = self.forward_once(sentence1)["sentence_embedding"]
        output2 = self.forward_once(sentence2)["sentence_embedding"]
        similarity_score = self.cos(output1, output2)
        return similarity_score

def encode_pair(pair,padding=True,truncation=True,max_length=256):
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    return tokenizer(
    pair[0], padding=padding,truncation=truncation,max_length=max_length, return_tensors="pt"
    ), tokenizer(
        pair[1],padding=padding,truncation=truncation,max_length=max_length, return_tensors="pt"
    )



model=  ShroomForwardFormer(256)
'''Training loop:'''
if train_models:
    for epoch in tqdm(range(num_epochs)):
        running_loss = 0.0
        correct = 0
        total = 0
        model.train()
        for i, data in enumerate(train_dataloader, 0):
            if i%100==0:
                print(f"Batch: {i+1}")
            sentences, labels = data
            optimizer.zero_grad()
            sentence1,sentence2=encode_pair(sentences)
            labels = labels.float()
            outputs=model(sentence1,sentence2)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

# evaluate_shroom(model,test_dataloader)
# O/P: Test Loss : 0.027083140953133505

''' Training Sentence Bert '''
if train_models:
    model = SentenceTransformer("distilbert-base-nli-mean-tokens")
    train_loss = losses.CosineSimilarityLoss(model)
    sentences1 = []
    sentences2 = []
    scores = []
    for i,row in validation_cosine_labels.iterrows():
        sentences1.append(row["sentence1"])
        sentences2.append(row["sentence2"])
        scores.append(row["label"])

    evaluator = evaluation.EmbeddingSimilarityEvaluator(sentences1, sentences2, scores)
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=100,
        warmup_steps=100,
        evaluator=evaluator,
        evaluation_steps=500,
    )

def sbert_inference(file_path,model):
    with open(file_path, 'r') as file:
        data = json.load(file)

    y_true = []
    y_pred = []
    metrics_by_task = {}

    for example in data:
        task = example.get("task")
        if task not in metrics_by_task:
            metrics_by_task[task] = {"y_true": [], "y_pred": []}

        hyp = [example["hyp"]]
        tgt = [example["tgt"]]
        label = example["label"]
        embeddings1 = model.encode(hyp, convert_to_tensor=True)
        embeddings2 = model.encode(tgt, convert_to_tensor=True)

        similarity = util.cos_sim(embeddings1, embeddings2)
    
        predicted_label = "Not Hallucination" if similarity[0][0] > 0.7 else "Hallucination"
        y_true.append(label)
        y_pred.append(predicted_label)

    
        metrics_by_task[task]["y_true"].append(label)
        metrics_by_task[task]["y_pred"].append(predicted_label)

    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label="Hallucination")
    recall = recall_score(y_true, y_pred, pos_label="Hallucination")
    f1 = f1_score(y_true, y_pred, pos_label="Hallucination")
    
    overall_metrics={ "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1}

    task_metrics = {}
    for task, metrics in metrics_by_task.items():
        y_true = metrics["y_true"]
        y_pred = metrics["y_pred"]
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, pos_label="Hallucination")
        recall = recall_score(y_true, y_pred, pos_label="Hallucination")
        f1 = f1_score(y_true, y_pred, pos_label="Hallucination")
        task_metrics[task] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

    return {
        "overall metrics": overall_metrics,
        "task metrics": task_metrics
    }

def shroom_inference(file_path,model):
    with open(file_path, 'r') as file:
        data = json.load(file)

    y_true = []
    y_pred = []
    metrics_by_task = {}

    for example in data:
        task = example.get("task")
        if task not in metrics_by_task:
            metrics_by_task[task] = {"y_true": [], "y_pred": []}

        hyp = example["hyp"]
        tgt = example["tgt"]
        label = example["label"]

        sentence1, sentence2 = encode_pair([hyp, tgt])
        similarity = model(sentence1,sentence2)
    
        predicted_label = "Not Hallucination" if similarity.item() > 0.7 else "Hallucination"
        y_true.append(label)
        y_pred.append(predicted_label)

    
        metrics_by_task[task]["y_true"].append(label)
        metrics_by_task[task]["y_pred"].append(predicted_label)

    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label="Hallucination")
    recall = recall_score(y_true, y_pred, pos_label="Hallucination")
    f1 = f1_score(y_true, y_pred, pos_label="Hallucination")
    
    overall_metrics={ "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1}

    task_metrics = {}
    for task, metrics in metrics_by_task.items():
        y_true = metrics["y_true"]
        y_pred = metrics["y_pred"]
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, pos_label="Hallucination")
        recall = recall_score(y_true, y_pred, pos_label="Hallucination")
        f1 = f1_score(y_true, y_pred, pos_label="Hallucination")
        task_metrics[task] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

    return {
        "overall metrics": overall_metrics,
        "task metrics": task_metrics
    }

'''Evaluating ShroomForwardFormer'''
print("\n ShroomForwardFormer")
state_dict = torch.load("best_model_v2_lr1.pth")
model=ShroomForwardFormer(256)
model.load_state_dict(state_dict)
print(shroom_inference("trial-v1.json",model))

'''Evaluating SentenceBert'''
print("\n SenntenceTransformer Pretrained")
model = SentenceTransformer("distilbert-base-nli-mean-tokens")
model.load_state_dict(torch.load("model_senbert.pth"))
print(sbert_inference("trial-v1.json",model))

'''Evaluating ShroomFormerV2'''
print("\n ShroomFormerV2")
state_dict = torch.load("model_v2_lr1.pth")
model=  ShroomformerV2()
model.load_state_dict(state_dict)
print(shroom_inference("trial-v1.json",model))
