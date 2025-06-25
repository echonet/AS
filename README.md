# AS
- Estimate AS severity from 6 different TTE views
    - 

## how to use
0. `git clone https://github.com/echonet/AS.git && cd AS && git clone https://github.com/echonet/cvair.git`
1. create a `MANIFEST_CSV` file that has the following columns
    - file_uid
    - study_uid
    - view
    - MRN: patients id
    - frames
    - fps
    - split: "test"
    - path_column: path to the videoes
    - AS_severity:
        - 0: no, trace or trivial
        - 1: mild
        - 2: mild~moderate
        - 3: moderate
        - 4: moderate~severe
        - 5: severe

2. get weights from [https://github.com/echonet/AS/releases/tag/v0.1.0](https://github.com/echonet/AS/releases/tag/v0.1.0) and save them to `./weights`.
3. Each view (e.g., PLAX, PSAX, Apical) has its own specific pretrained weight file in the weights folder.
4. Based on the view classifier, the following views need to be included:
    - PLAX: "PLAX_Zoom_out", "PLAX_AV_MV", "PLAX", "PLAX_zooomed_AV"
    - PSAX: "PSAX_(level_great_vessels)_zoomed_AV", "PSAX_(level_great_vessels)"
    - Apical: "A3C", "A5C"
    - PLAX_D: "DOPPLER_PLAX_AV_MV", "DOPPLER_PLAX_AV_zoomed"
    - PSAX_D: "DOPPLER_PSAX_level_great_vessels_AV"
    - Apical_D: "DOPPLER_A3C", "DOPPLER_A3C_AV", "DOPPLER_A5C"

## example usage
```
python predict.py \
    --manifest_path PATH_TO_MANIFEST_CSV \
    --view PLAX \
    --path_column path_column \
    --batch_size 64 \
    --targets AS_severity
```

