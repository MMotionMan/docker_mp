from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd
import transformers
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from torch import cuda
import json
from dataclasses import dataclass

device = 'cuda' if cuda.is_available() else 'cpu'
CONFIG_MODEL_PATH = "./mp_volume/model_info.json"
MAX_LEN = 200
VALID_BATCH_SIZE = 4
tokenizer = BertTokenizer.from_pretrained('./rubert-tiny')


class CustomDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.comment_text = dataframe.comment_text
        self.targets = self.data.list
        self.max_len = max_len

    def __len__(self):
        return len(self.comment_text)

    def __getitem__(self, index):
        comment_text = str(self.comment_text[index])
        comment_text = " ".join(comment_text.split())

        inputs = self.tokenizer.encode_plus(
            comment_text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }


class BERTClass(torch.nn.Module):
    def __init__(self, numOfComments):
        super(BERTClass, self).__init__()
        self.l1 = transformers.BertModel.from_pretrained('./rubert-tiny', return_dict=False)
        self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(312, numOfComments)

    def forward(self, ids, mask, token_type_ids):
        _, output_1 = self.l1(ids, attention_mask=mask, token_type_ids=token_type_ids)
        output_2 = self.l2(output_1)
        output = self.l3(output_2)
        return output

    def save_weights(self, filepath):
        torch.save(self.state_dict(), filepath)

    def load_weights(self, filepath):
        self.load_state_dict(torch.load(filepath, map_location=device))


def load_model(model_path, numOfComments):
    try:
        model = BERTClass(numOfComments)
        model.to(device)
        model.load_weights(model_path)
        return model
    except FileNotFoundError:
        print(f"Model file not found at: {model_path}")
        return None


with open(CONFIG_MODEL_PATH, 'r') as f:
    models_info = json.load(f)

max_model_num = max(models_info,
                    key=lambda item: int(item['model_number'])
                    )

models = [None for i in range(max_model_num['model_number'] + 1)]

for info in models_info:
    models[info['model_number']] = load_model(info['model_path'], info['numOfComments'])

# models = [load_model(info['model_path'], info['numOfComments']) for info in model_info]

test_params = {'batch_size': VALID_BATCH_SIZE,
               'shuffle': False,
               'num_workers': 0
               }


def prediction(comment, n):
    if n >= len(models):
        print(f"Model index out of range: {n}")
        return None, "Model index out of range"
    elif models[n] is None:
        print(f"No model loaded for index: {n}")
        return None, "Model not loaded"

    data = pd.DataFrame({'comment_text': [comment]})
    numOfComments = models[n].l3.out_features
    columns = np.arange(numOfComments)

    for i in range(numOfComments):
        data[i] = int(0)

    data['list'] = data[data.columns[1:]].values.tolist()
    df = data[['comment_text', 'list']].copy()
    testing_set = CustomDataset(df, tokenizer, MAX_LEN)

    testing_loader = DataLoader(testing_set, **test_params)

    def validation(epoch):
        models[n].eval()
        fin_targets = []
        fin_outputs = []
        with torch.no_grad():
            for _, data in enumerate(testing_loader, 0):
                ids = data['ids'].to(device, dtype=torch.long)
                mask = data['mask'].to(device, dtype=torch.long)
                token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
                targets = data['targets'].to(device, dtype=torch.float)
                outputs = models[n](ids, mask, token_type_ids)
                fin_targets.extend(targets.cpu().detach().numpy().tolist())
                fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
        return fin_outputs, fin_targets

    outputs, targets = validation(1)
    outputs = np.array(outputs)[0]
    return outputs, None


app = FastAPI()


class ModelInfo(BaseModel):
    model_number: int
    model_path: str
    numOfComments: int


class ModelDisableInfo(BaseModel):
    model_number: int


class Comment(BaseModel):
    text: str
    model_number: int


@app.post("/predict")
async def make_prediction(comment: Comment):
    prediction_result, error_message = prediction(comment.text, comment.model_number)
    if prediction_result is None:
        return {"error": error_message}
    return {"prediction": prediction_result.tolist()}


@app.post("/reload_model")
async def reload_model(model_info: ModelInfo):
    model_index = model_info.model_number

    if model_index > len(models):
        models.extend([None for i in range(model_index - len(models))])

    if models[model_index]:
        return {"status": f"Model {model_index} has been activated"}

    models[model_index] = load_model(model_info.model_path, model_info.numOfComments)

    if models[model_index] is None:
        return {"error": f"Failed to reload model from {model_info.model_path}"}
    else:
        models_info.append({'model_number': model_info.model_number,
                            'model_path': model_info.model_path,
                            'numOfComments': model_info.numOfComments})
        with open(CONFIG_MODEL_PATH, 'w') as f:
            json.dump(models_info, f)
        return {"status": "Model reloaded successfully"}


@app.post("/disable_model")
async def disable_model(model_info: ModelDisableInfo):
    model_index = model_info.model_number
    if 0 <= model_index < len(models):
        models[model_index] = None
        del_idx = None
        for i in range(len(models_info)):
            if models_info[i]['model_number'] == model_index:
                del_idx = i
                break

        if del_idx:
            del models_info[del_idx]
        else:
            return {"status": f"Model {model_index} not activated"}

        with open(CONFIG_MODEL_PATH, 'w') as f:
            json.dump(models_info, f)

        # with open(CONFIG_MODEL_PATH, 'r') as f:
        #     print("загруженный json", json.load(f))

        return {"status": f"Model {model_index} disabled successfully"}

    else:
        return {"error": f"No model with index: {model_index}"}
