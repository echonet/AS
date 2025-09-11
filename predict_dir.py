import os
import torch
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import tqdm
from torch.utils.data import Dataset, DataLoader
from utils import load_model, mask_outside_ultrasound, write_to_avi, read_video, crop_and_scale
import os
import pickle
from typing import Tuple, Union, List
import pydicom


"""
    This script takes in a user-specified path to a directory of DICOMs.
    The dicom files will be converted to avi and saved in a directory named `avis` in the current working directory.
    Predictions will be saved as csv files in the `predictions` directory.
    The view-specific predictions files will be named `predictions_{view}.csv`.
    The ensemble study-level predictions file will be named `ensemble_predictions.csv`.
"""


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, required=True)
parser.add_argument("--weights_dir", type=str, default="./weights")
parser.add_argument("--batch_size", type=int, default=4)
args = parser.parse_args()
batch_size = args.batch_size

data_dir = Path(args.data_dir)
weights_dir = Path(args.weights_dir)

NEURON_NAMES = ["no", "mild", "mild~moderate", "moderate", "moderate~severe", "severe"]
models = ['PLAX', 'PSAX', 'Apical', 'PLAX_D', 'PSAX_D', 'Apical_D']
classes_4 = ['no', 'mild', 'moderate', 'severe']


class EchoDataset(Dataset):
    def __init__(
        self,
        data_path: Union[Path, str],
        manifest_path: Union[Path, str] = None,
        split: str = None,
        labels: List[str] = None,
        verify_existing: bool = True,
        drop_na_labels: bool = True,
        n_frames: int = 16,
        random_start: bool = False,
        sample_rate: Union[int, Tuple[int], float] = 2,
        verbose: bool = True,
        resize_res: Tuple[int] = (112, 112),
        zoom: float = 0
    ):
        self.verbose = verbose
        self.data_path = Path(data_path)
        self.split = split
        self.verify_existing = verify_existing
        self.n_frames = n_frames
        self.random_start = random_start
        self.sample_rate = sample_rate
        self.resize_res = resize_res
        self.zoom = zoom
       
        # Read manifest file
        if manifest_path is not None:
            self.manifest_path = Path(manifest_path)
        else:
            self.manifest_path = self.data_path / "manifest.csv"

        if self.manifest_path.exists():
            self.manifest = pd.read_csv(self.manifest_path, low_memory=False)
        else:
            self.manifest = pd.DataFrame(
                {
                    "filename": os.listdir(self.data_path),
                }
            )
        if self.split is not None:
            self.manifest = self.manifest[self.manifest["split"] == self.split]
        if self.verbose:
            print(
                f"Manifest loaded. \nSplit: {self.split}\nLength: {len(self.manifest):,}"
            )

        # Make sure all files actually exist. This can be disabled for efficiency if
        # you have an especially large dataset
        if self.verify_existing:
            old_len = len(self.manifest)
            existing_files = [f for f in self.manifest['filename'] if os.path.exists(f)]
            self.manifest = self.manifest[
                self.manifest["filename"].isin(existing_files)
            ]
            new_len = len(self.manifest)
            if self.verbose:
                print(
                    f"{old_len - new_len} files in the manifest are missing."
                )
        elif (not self.verify_existing) and self.verbose:
            print(
                f"self.verify_existing is set to False, so it's possible for the manifest to contain filenames which are not present"
            )

        
    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, index):
        output = {}
        row = self.manifest.iloc[index]
        filename = row["filename"]
        output["filename"] = filename

        # self.read_file expected in child classes
        primary_input = self.read_file(self.data_path / filename, row)
        output["primary_input"] = primary_input
        return output
   
    def read_file(self, filepath, row=None):

        if isinstance(self.sample_rate, int):  # Simple single int sample period
            vid, vid_shape, fps = read_video(
                filepath,
                self.n_frames,
                res=self.resize_res,
                zoom=self.zoom,
                sample_period=self.sample_rate,
                random_start=self.random_start,
            )
        elif isinstance(self.sample_rate, float):  # Fixed fps
            target_fps = self.sample_rate
            fps = row["fps"]

            vid, vid_shape, fps = read_video(
                filepath,
                self.n_frames,
                1,
                fps=row["fps"],
                out_fps=target_fps,
                frame_interpolation=self.interpolate_frames,
                random_start=self.random_start,
                res=self.resize_res,
                zoom=self.zoom,
            )
        else:  # Tuple sample period ints to be randomly sampled from (1, 2, 3)
            sample_period = np.random.choice(
                [x for x in self.sample_rate if row["frames"] > x * self.n_frames]
            )
            vid, vid_shape, fps = read_video(
                filepath,
                self.n_frames,
                res=self.resize_res,
                zoom=self.zoom,
                sample_period=sample_period,
                random_start=self.random_start,
            )
        vid = torch.from_numpy(vid)
        vid = torch.movedim(vid / 255, -1, 0).to(torch.float32)
        return vid
    


print('Starting inference...')

for view in models:
    print(f"\nRunning inference for view: {view}")

    if len(list(data_dir.glob(f"*/{view}/*")))>0:
        print(f"Found DICOMs dir for view: {view}")
        weights_path = weights_dir/f"{view}.pt"

        print(f"number of files {view}:\t{len(list(data_dir.glob(f'*/{view}/*')))}")
        print(f"number of studies {view}:\t{len(list(data_dir.glob(f'*/{view}')))}")
            
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


    # Convert DICOMs to AVIs
    data_path = list(data_dir.glob(f"*/{view}"))
    dcm_paths = [str(file) for f in data_path for file in f.glob('*')]
    studyids = [file.parent.parent.name for f in data_path for file in f.glob('*')]
    avi_paths = [str(file).replace(str(data_dir), 'avis') + '.avi' for f in data_path for file in f.glob('*')]

    print('Converting DICOMs to AVIs...')
    for dcm, avi in zip(dcm_paths, avi_paths):
        ds = pydicom.dcmread(str(dcm))
        masked_pixel_array = mask_outside_ultrasound(ds.pixel_array)
        Path(avi).parent.mkdir(parents=True, exist_ok=True)
        write_to_avi(masked_pixel_array, avi, fps=30)

    # Create manifest file for dataloader
    manifest_d = {
        "filename": avi_paths,
        "dcm_path": dcm_paths,
        'studyid': studyids,
        "split": ["test"] * len(dcm_paths)
    }
    os.makedirs('manifest', exist_ok=True)
    manifest_path = f"./manifest/{view}.csv"
    pd.DataFrame(manifest_d).to_csv(f"./manifest/{view}.csv", index=None)

    ### Set up dataset
    test_ds = EchoDataset(
        split="test",
        data_path='.',
        manifest_path=manifest_path
    )
    ### Set up dataloader
    test_dl = DataLoader(
        test_ds, batch_size=batch_size, drop_last=False, shuffle=False,
        )

    with torch.no_grad():
        ### Load model
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = load_model(device, weights_path=weights_path, num_classes=len(NEURON_NAMES))

        studyids = []    
        filenames = []
        predictions = []
        ### Run inference on each batch
        for idx, batch in tqdm.tqdm(enumerate(test_dl)):
            batch_tensor = batch['primary_input'].to(device)
            batch_filenames = batch['filename']
            batch_studyids = [Path(f).parents[1].name for f in batch_filenames]
            raw_output = model(batch_tensor)
            output = torch.softmax(raw_output, dim=1)  # output from model
            predictions.append(output.cpu())  # output from model
            filenames.extend(batch_filenames)
            studyids.extend(batch_studyids)

        predictions = torch.cat(predictions, dim=0).T  # num_neurons x num_samples

        ### Create dataframe with columns for filename and predictions
        pred_dict = {"filename": filenames, "dcm_path": dcm_paths, "studyid": studyids}
        for key, value in zip(NEURON_NAMES, predictions):
            pred_dict[key] = value

        pred_dict["model"] = [view]*predictions.shape[1]

        dataframe = pd.DataFrame(pred_dict)

        ### Save predictions to file
        os.makedirs("predictions", exist_ok=True)
        dataframe.to_csv(f"predictions/predictions_{view}.csv", index=None)
        print(f"Saved {view} model predictions to ./predictions/predictions_{view}.csv")

# ensemble results from all views
print("\nEnsembling results from all views...")
dfs = [pd.read_csv(f"./predictions/predictions_{m}.csv", dtype={'study_uid': str}, low_memory=False).drop_duplicates() for m in models]
df = pd.concat(dfs, ignore_index=True)
agg = df.groupby(["studyid", "model"])[NEURON_NAMES].mean()
wide = agg.unstack("model")
wide.columns = [f"{mdl}_{col}" for col, mdl in wide.columns]

# merge peakav prediction from avvmax model
# peakav prediction file is generated from a separate script and should be placed in ./predictions/metadata_avvmax.csv
# the file should have columns: studyid, peak_velocity
peakav_path = './predictions/metadata_avvmax.csv'
if not os.path.exists(peakav_path):
    print(f"Peak AV velocity prediction file not found at {peakav_path}.")
    print("Ensemble will proceed without peak AV velocity.")
    wide['peakav'] = np.nan
else:
    print(f"Loading peak AV velocity predictions from {peakav_path}.")
    d = pd.read_csv('./predictions/metadata_avvmax.csv')

    # calculate mean peak velocity if multiple AV doppler clips per study
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
