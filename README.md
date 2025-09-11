## EchoNet-AS
**EchoNet-AS** is a deep learning framework for estimating **aortic stenosis (AS) severity** from transthoracic echocardiography (TTE).  
The model integrates predictions from **six echocardiographic views** together with the **peak aortic valve (AV) velocity** measured from continuous-wave (CW) Doppler.  

The final output is the probability of **four severity classes** per study:  
`no`, `mild`, `moderate`, `severe`.

---

## Overview
- Input: DICOM echocardiography videos, organized by `study_id`
- Views required:
  - PLAX
  - PSAX
  - Apical
  - PLAX Doppler (PLAX_D)
  - PSAX Doppler (PSAX_D)
  - Apical Doppler (Apical_D)
  - CW Doppler of AV (`AV_doppler`)
- Our framework is robust to missing data. Even if a study lacks one or more views (e.g., no PSAX Doppler available), the model will still make predictions based on the remaining views.
- Output: Study-level probabilities for **no / mild / moderate / severe AS**
- Our framework uses EchoNet-Measurements

## how to use
0. Setup
    ```bash
    git clone https://github.com/echonet/AS.git
    ```
    and install required packages in requirements.txt

1. prepare data
    - The input data is assumed to be **DICOM files**, organized by `study_id`.  
    - Each study directory contains **six echocardiographic views** (PLAX, PSAX, Apical and their color-Dopplers) and Doppler image for peak AV velocity measurement (`AV_doppler`, CW doppler image).

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

2. Based on the view classifier, the following view names are expected
    - PLAX: "PLAX_Zoom_out", "PLAX_AV_MV", "PLAX", "PLAX_zooomed_AV"
    - PSAX: "PSAX_(level_great_vessels)_zoomed_AV", "PSAX_(level_great_vessels)"
    - Apical: "A3C", "A5C"
    - PLAX_D: "DOPPLER_PLAX_AV_MV", "DOPPLER_PLAX_AV_zoomed"
    - PSAX_D: "DOPPLER_PSAX_level_great_vessels_AV"
    - Apical_D: "DOPPLER_A3C", "DOPPLER_A3C_AV", "DOPPLER_A5C"
    - AV_doppler: "Doppler_A5C_AV_CW", "Doppler_A3C_AV_CW"

3. download weights
    - Weights for 6 distinct view models
        - download from [https://github.com/echonet/AS/releases/tag/v0.1.0](https://github.com/echonet/AS/releases/tag/v0.1.0) and save them to `./weights` so that each view (e.g., PLAX, PSAX, Apical, ...) model has its own specific pretrained weight file in the `weights` directory.
    - Weight for Peak AV velocity measurement model
        - download `avvmax_weights.ckpt` from [https://github.com/echonet/measurements/blob/main/weights/Doppler_models/avvmax_weights.ckpt](https://github.com/echonet/measurements/blob/main/weights/Doppler_models/avvmax_weights.ckpt) and put it to `./weights`

4. Run model
    1. **Peak AV velocity inference model**
    ```
    python inference_Doppler_image_folders.py --data_dir ./test_dcm/
    ```

    - the output will be saved to `./predictions/metadata_avvmax.csv`
    <br><br>

    2. **Run six-view specific models**
    ```
    python predict_dir.py --data_dir ./test_dcm --weights_dir ./weights/
    ```
    
5. Output
    - Predictions will be saved as csv files in the `predictions` directory.
    - Peak AV velocity prediction file will be named `metadata_avvmax.csv`
    - The view-specific predictions files will be named `predictions_{view}.csv`.
    - The final study-level predictions file will be named `ensemble_predictions.csv`.
        - column `final_pred_class` is the final prediction of 'no/mild/moderate/severe'
        - the columns `no`, `mild`, `moderate`, and `severe` are the probabilities.


