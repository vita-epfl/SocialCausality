# ETH-UCY Setup
Our ETH-UCY dataset with trajnet++ format can be downloaded from [Google drive](https://drive.google.com/file/d/178Vq7MFXT-Y1DPuGelgU2T9eewLeYeay/view?usp=drive_link).
Extract the data :

```
python create_data_npys.py --raw-dataset-path /path/to/synth_data/ --output-npy-path /path/to/output_npys --split <split: train, test, or val>
```

This script will generate preprocessed data into a numpy file called `{split}_{name of dataset}.npy`.
