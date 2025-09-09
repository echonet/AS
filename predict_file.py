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
    This script takes in a user-specified path to a directory of A4C DICOMs. Predictions 
    are saved as a csv file in the directory that the script is run from. 
    The predictions file has 3 columns: 
        1. filename - name of the DICOM file 
        2. prediction - sigmoided output of the model
"""

NEURON_NAMES = ["no", "mild", "mild~moderate", "moderate", "moderate~severe", "severe"]

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, required=True)
parser.add_argument("--weights_dir", type=str, required=True)
parser.add_argument("--batch_size", type=int, default=4)
args = parser.parse_args()

data_dir = Path(args.data_dir)
# PLAX_path = Path(args.PLAX_path) if args.PLAX_path is not None else None
# PSAX_path = Path(args.PSAX_path) if args.PSAX_path is not None else None
# Apical_path = Path(args.Apical_path) if args.Apical_path is not None else None
# PLAX_D_path = Path(args.PLAX_D_path) if args.PLAX_D_path is not None else None
# PSAX_D_path = Path(args.PSAX_D_path) if args.PSAX_D_path is not None else None
# Apical_D_path = Path(args.Apical_D_path) if args.Apical_D_path is not None else None
weights_dir = Path(args.weights_dir)

models = ['PLAX', 'PSAX', 'Apical', 'PLAX_D', 'PSAX_D', 'Apical_D']

all_results = {}

class InferenceDataset(torch.utils.data.Dataset):
    def __init__(self, data_paths: Path):
        self.data_paths = data_paths
        self.files = [f for d in data_paths for f in d.iterdir()]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return self.files[idx].parents[1].name, self.files[idx].name, dicom_to_tensor(self.files[idx])


print('Starting inference...')

if (data_dir/f"AV_doppler").exists() and (len(list((data_dir/"AV_doppler").iterdir())) > 0):
    print(f"number of DICOMs AV_doppler:\t{len(list((data_dir/'AV_doppler').iterdir()))}")

else:
    print(f"No AV_doppler DICOMs found, skipping...")



for view in models:
    print(f"\nRunning inference for view: {view}")

    if len(list(data_dir.glob(f"*/{view}/*")))>0:
        print(f"Found DICOMs dir for view: {view}")
        weights_path = weights_dir/f"{view}.pt"

        print(f"number of files {view}:\t{len(list(data_dir.glob(f"*/{view}/*")))}")
        print(f"number of studies {view}:\t{len(list(data_dir.glob(f"*/{view}")))}")
            
    else:
        print(f"No DICOMs found for {view}, skipping...")
        pred_dict = {}
        pred_dict["model"] = [view]
        pred_dict['studyid'] = ['NA']
        pred_dict["filename"] = ['NA']
        for n in NEURON_NAMES:
            pred_dict[n] = [np.nan]
        for k,v in pred_dict.items():
            if k not in all_results:
                all_results[k] = []
            all_results[k].extend(v)
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

    for k,v in pred_dict.items():
        if k not in all_results:
            all_results[k] = []
        all_results[k].extend(v)
# print(all_results)

df = pd.DataFrame(all_results)

agg = df.groupby(['model', 'StudyID'])[NEURON_NAMES].mean()
wide = agg.unstack('model')

wide.columns = [f"{mdl}_{col}" for col, mdl in wide.columns]
wide.loc[:,'peakav'] = np.nan

print(wide)

with open('./weights/av_stenosis_peakav.pkl', 'rb') as f:
    ens = pickle.load(f)

prob = ens.predict_proba(wide.to_numpy())
print(prob)

# add ensemble predictions to all_results

