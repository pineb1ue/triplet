import os
import numpy as np
import pandas as pd
import json
import argparse
from tqdm import tqdm
import torch
from torchvision.models import resnet50
from data import Dataset, get_transforms

def get_features(dataset, model):
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=16, shuffle=False)
    features = torch.Tensor()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()
    for img, label in dataloader:
        img = img.to(device)
        with torch.no_grad():
            output = model(img)
        features = torch.cat((features, output.cpu()))
    return features

def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def results2csv(data_type, weight):
    model = resnet50(pretrained=False)
    model.load_state_dict(torch.load(os.path.join('output', data_type, weight)))

    _db_path = os.path.join('/home/matsunaga/Documents/datasets/reidification/1300_1400', data_type, 'db')
    _query_path = os.path.join('/home/matsunaga/Documents/datasets/reidification/1300_1400', data_type, 'query')

    db_dataset = Dataset(root_dirs=[_db_path], transforms=get_transforms('test'))
    query_dataset = Dataset(root_dirs=[_query_path], transforms=get_transforms('test'))

    db_features = get_features(dataset=db_dataset, model=model)
    query_features = get_features(dataset=query_dataset, model=model)
    db_features, query_features = db_features.numpy(), query_features.numpy()

    info = {}
    acc1, acc3, acc5 = 0, 0, 0
    for query_data, query_feature in tqdm(zip(query_dataset.data, query_features)):
        query_path = query_data['path']
        query_label = query_path.split('/')[-2]
        eucs = []
        db_paths, db_labels = [], []

        for db_data, db_feature in zip(db_dataset.data, db_features):
            db_path = db_data['path']
            db_paths.append(db_path)
            db_labels.append(db_path.split('/')[-2])
            eucs.append(np.linalg.norm(query_feature - db_feature))

        arg = np.argsort(eucs).tolist()
        df = pd.DataFrame({'label': [db_labels[i] for i in arg], 'path': [db_paths[i] for i in arg]})
        _df = df[~df['label'].duplicated()]

        tops = _df['label'].values.tolist()
        top_paths = _df['path'].values.tolist()
        info[query_path] = top_paths[:10]

        if query_label == tops[0]:
            acc1 += 1
        if query_label in tops[:3]:
            acc3 += 1
        if query_label in tops[:5]:
            acc5 += 1

    print(len(query_dataset.data))
    with open(os.path.join('res/csv', f"{data_type}.csv"), mode='a') as f:
        f.write(f"{weight} {acc1 / len(query_dataset.data)} {acc3 / len(query_dataset.data)} {acc5 / len(query_dataset.data)}\n")

    # with open(f"res/{data_type}.json", 'w') as f:
        # json.dump(info, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='result')
    parser.add_argument('-t', '--data_type', type=str, help='input data type')
    args = parser.parse_args()

    results2csv(data_type=args.data_type, weight='triplet_best_model.pth')