# AS
- Estimate AS severity from 6 different TTE views and automated doppler measurements

## how to use
0. Setup
```bash
git clone https://github.com/echonet/AS.git
```
install required packages in requirements.txt

1. prepare data
    - The input data is assumed to be **DICOM files**, organized by `study_id`.  
    - Each study directory contains **six echocardiographic views** (PLAX, PSAX, Apical and their color-Dopplers) and Doppler image for peak AV velocity measurement (`AV_doppler`, CW doppler image is expected).

    Example directory structure:
    ```
    test_dcm/
    ├── study_001/
    │   ├── PLAX/
    │   │   ├── file1.dcm
    │   │   └── file2.dcm
    │   ├── PSAX/
    │   │   ├── file3.dcm
    │   │   └── file4.dcm
    │   ├── Apical/
    │   │   └── file5.dcm
    │   ├── PLAX_D/
    │   │   └── file6.dcm
    │   ├── PSAX_D/
    │   │   └── file7.dcm
    │   ├── Apical_D/
    │   │   └── file8.dcm
    │   └── AV_doppler/
    │       ├── file9.dcm
    │       └── file10.dcm
    ├── study_002/
    │   ├── PLAX/
    │   │   └── file11.dcm
    │   ├── PSAX/
    │   │   ├── file12.dcm
    │   │   └── file13.dcm
    │   ├── Apical/
    │   │   └── file14.dcm
    │   ├── PLAX_D/
    │   │   └── file15.dcm
    │   ├── PSAX_D/
    │   │   └── file16.dcm
    │   ├── Apical_D/
    │   │   └── file17.dcm
    │   └── AV_doppler/
    │       └── file10.dcm
    ```
2. download weights
    - Weights for 6 distinct view models
        - download from [https://github.com/echonet/AS/releases/tag/v0.1.0](https://github.com/echonet/AS/releases/tag/v0.1.0) and save them to `./weights` so that each view (e.g., PLAX, PSAX, Apical, ...) model has its own specific pretrained weight file in the `weights` directory.
    - Weight for Peak AV velocity measurement model
        - download `avvmax_weights.ckpt` from [https://github.com/echonet/measurements/blob/main/weights/Doppler_models/avvmax_weights.ckpt](https://github.com/echonet/measurements/blob/main/weights/Doppler_models/avvmax_weights.ckpt) and put it to `./weights`

3. Based on the view classifier, the following views need to be included:
    - PLAX: "PLAX_Zoom_out", "PLAX_AV_MV", "PLAX", "PLAX_zooomed_AV"
    - PSAX: "PSAX_(level_great_vessels)_zoomed_AV", "PSAX_(level_great_vessels)"
    - Apical: "A3C", "A5C"
    - PLAX_D: "DOPPLER_PLAX_AV_MV", "DOPPLER_PLAX_AV_zoomed"
    - PSAX_D: "DOPPLER_PSAX_level_great_vessels_AV"
    - Apical_D: "DOPPLER_A3C", "DOPPLER_A3C_AV", "DOPPLER_A5C"
    - AV_doppler: "Doppler_A5C_AV_CW", "Doppler_A3C_AV_CW" (based on `full_view_classifier.ckpt`)

4. Run model
    - run Peak AV velocity inference model
    ```
    python inference_Doppler_image_folders.py --data_dir ./test_dcm/
    # the output will be saved to ./predictions/metadata_avvmax.csv
    ```
    - run 6 view distinct models
    ```
    python predict_file.py --data_dir ./test_dcm --weights_dir ./weights/
    # the output will be saved to ./predictions/predictions_{view}.csv
    ```
    - The final prediction will be saved at `./predictions/predictions_ensemble.csv`

