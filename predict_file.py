import os
import torch
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils import load_model, dicom_to_tensor
import os
import pickle


"""
    This script takes in a user-specified path to a directory of DICOMs.
    Predictions will be saved as csv files in the `predictions` directory.
    The view-specific predictions files will be named `predictions_{view}.csv`.
    The ensemble study-level predictions file will be named `ensemble_predictions.csv`.
    The predictions file has 3 columns:
        1. filename - name of the DICOM file
        2. prediction - sigmoided output of the model
"""


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, required=True)
parser.add_argument("--weights_dir", type=str, required=True)
parser.add_argument("--batch_size", type=int, default=4)
args = parser.parse_args()

data_dir = Path(args.data_dir)
weights_dir = Path(args.weights_dir)

NEURON_NAMES = ["no", "mild", "mild~moderate", "moderate", "moderate~severe", "severe"]
models = ['PLAX', 'PSAX', 'Apical', 'PLAX_D', 'PSAX_D', 'Apical_D']
classes_4 = ['no', 'mild', 'moderate', 'severe']


class InferenceDataset(torch.utils.data.Dataset):
    def __init__(self, data_paths: Path):
        self.data_paths = data_paths
        self.files = [f for d in data_paths for f in d.iterdir()]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return self.files[idx].parents[1].name, self.files[idx].name, dicom_to_tensor(self.files[idx])


print('Starting inference...')

for view in models:
    print(f"\nRunning inference for view: {view}")

    if len(list(data_dir.glob(f"*/{view}/*")))>0:
        print(f"Found DICOMs dir for view: {view}")
        weights_path = weights_dir/f"{view}.pt"

        print(f"number of files {view}:\t{len(list(data_dir.glob(f"*/{view}/*")))}")
        print(f"number of studies {view}:\t{len(list(data_dir.glob(f"*/{view}")))}")
            
    else:
        print(f"No DICOMs found for {view}, skipping...")
        studies = [d.name for d in data_dir.iterdir() if d.is_dir()]
        pred_dict = {
            'model': [view]*len(studies),
            'studyid': studies,
            'filename': ['NA']*len(studies)
        }
        for n in NEURON_NAMES:
            pred_dict[n] = [np.nan]*len(studies)

        # save empty predictions to file for ensembling later
        os.makedirs("predictions", exist_ok=True)
        pd.DataFrame(pred_dict).to_csv(f"predictions/predictions_{view}.csv", index=None)
        continue


    ### Create Pytorch dataset from user-inputted directory of DICOM files
    ### Convert each DICOM to tensor as input to model

    data_path = list(data_dir.glob(f"*/{view}"))
    dataset = InferenceDataset(data_path)


    ### Set up dataloader
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)


    with torch.no_grad():
        ### Load model
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = load_model(device, weights_path=weights_path, num_classes=len(NEURON_NAMES))

        studyids = []    
        filenames = []
        predictions = []
        ### Run inference on each batch
        for idx, (batch_studyids, batch_filenames, batch_tensor) in tqdm.tqdm(enumerate(dataloader)):
            batch_tensor = batch_tensor.to(device)
            output = torch.softmax(model(batch_tensor), dim=1)  # output from model
            predictions.append(output.cpu())  # output from model
            filenames.extend(batch_filenames)
            studyids.extend(batch_studyids)

        predictions = torch.cat(predictions, dim=0).T  # num_neurons x num_samples
        ### Create dataframe with columns for filename and predictions
        pred_dict = {"filename": filenames, "studyid": studyids}

    
        for key, value in zip(NEURON_NAMES, predictions):
            pred_dict[key] = value
    

        pred_dict["model"] = [view]*predictions.shape[1]
        # print(pred_dict)
        dataframe = pd.DataFrame(pred_dict)
        ### Save predictions to file
        os.makedirs("predictions", exist_ok=True)
        dataframe.to_csv(f"predictions/predictions_{view}.csv", index=None)


# ensemble results from all views
print("\nEnsembling results from all views...")
dfs = [pd.read_csv(f"./predictions/predictions_{m}.csv", dtype={'study_uid': str}, low_memory=False).drop_duplicates() for m in models]
df = pd.concat(dfs, ignore_index=True)
agg = df.groupby(["studyid", "model"])[NEURON_NAMES].mean()
wide = agg.unstack("model")
wide.columns = [f"{mdl}_{col}" for col, mdl in wide.columns]

# merge peakav prediction
d = pd.read_csv('./predictions/metadata_avvmax.csv')
tmp = d.groupby(['studyid'])['peak_velocity'].mean()
wide = wide.merge(tmp, on='studyid', how='left')
wide.rename(columns={'peak_velocity':'peakav'}, inplace=True)


# load ensemble model
with open('./weights/av_stenosis_peakav.pkl', 'rb') as f:
    ens = pickle.load(f)
prob = ens.predict_proba(wide.to_numpy())
wide[classes_4] = prob
wide["final_pred_class"] = wide[classes_4].idxmax(axis=1)

print("\nSample ensemble predictions:")
print(wide[classes_4 + ["final_pred_class"]].head())

wide.to_csv(f"./predictions/ensemble_predictions.csv", index=True)
print("\nSaved ensemble predictions to ./predictions/ensemble_predictions.csv")
