import wandb
from pytorch_lightning import Trainer
from torchvision.models.video import r2plus1d_18
from cvair.data.datasets import EchoDataset, CedarsDataLoader
import torch
from cvair.training.model_wrappers import MultiClassificationModelWrapper

import argparse
import datetime
import random
import string
from pathlib import Path
import re
import os

def predict(
        manifest_path, path_column, batch_size, targets, CLASS_LIST, 
        weight_path, save_path, split='test', resize_res=(112,112), num_workers=4, gpus=[0], 
        **kwargs
):
    test_ds = EchoDataset(
        split=split,
        path_column=path_column,
        manifest_path=manifest_path,
        targets=targets,
        resize_res=tuple(resize_res),
    )

    test_dl = CedarsDataLoader(
        test_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        drop_last=False,
    )

    backbone = r2plus1d_18(num_classes=len(CLASS_LIST))
    model = MultiClassificationModelWrapper(backbone, output_names=CLASS_LIST)

    weights = torch.load(weight_path)
    print(model.load_state_dict(weights))

    trainer = Trainer(accelerator="gpu", devices=gpus)

    results = trainer.predict(model, dataloaders=test_dl)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.collate_and_save_predictions(results, save_path=save_path, merge_on=path_column)
    print(f"\nPredictions saved to {save_path}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest_path", required=True)
    parser.add_argument("--view", required=True, choices=['PLAX', 'PSAX', 'Apical', 'PLAX_D', 'PSAX_D', 'Apical_D'])
    parser.add_argument("--path_column", type=str, default='path_column')
    parser.add_argument("--batch_size", type=str, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--targets", required=True, default=None)
    args = parser.parse_args()

    weight_path =  f'./weights/{args.view}.pt'

    predict(
        manifest_path=args.manifest_path,
        path_column=args.path_column,
        batch_size=int(args.batch_size),
        targets=args.targets,
        CLASS_LIST=["no", "mild", "mild~moderate", "moderate", "moderate~severe", "severe"],
        weight_path=weight_path,
        save_path=f"./predictions/{args.view}.csv"
        )

