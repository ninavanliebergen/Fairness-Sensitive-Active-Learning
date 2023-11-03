# Fairness-Sensitive Active Learning

## Running the code:

### Random Sampling, Uncertainty Sampling, Representative Sampling, ReSGrAL Uncertainty and ReSGrAL Representative:
For running one of the sampling strategies, do:
```
 python prompts_reader.py <n_samples> <dataset_path> <target_value> <sensitive_attr> <subgroups> <dataset_name>
```
For example, when sampling with random sampling on the adult dataset:
```
  python Random.py 1 ../datasets/AdultDatasetGroupEncoded.csv income gender groups Adult
```
The results will be saved in ../SamplingStrategies/Results/

### FAL:
When running FAL, there is the option to run it on a subpart of the dataset and also include the amount of workers:
```
python FAL.py <n_samples> <dataset_path> <target_value> <sensitive_attr> <subgroups> <dataset_name> <fraction_dataset> <amount_of_workers>
```
For example, to run FAL on only 20% of the dataset with 16 workers:
```
python FAL.py 1 ../datasets/AdultDatasetGroupEncoded.csv income gender groups Adult 0.2 16
```
However, when the fraction of the dataset and the amount of workers are not specified, it is runned on the total (100%) of the dataset in serial.

### FairReSGrAL:
When running FairReSGrAL, one needs to specify the threshold:
```
python FAL.py <n_samples> <dataset_path> <target_value> <sensitive_attr> <subgroups> <dataset_name> <threshold> <fraction_dataset> <amount_of_workers>
```
For exaxample, to run FairReSGrAL with a threshold of 0.3, run:
```
python FAL.py 1 ../datasets/AdultDatasetGroupEncoded.csv income gender groups Adult 0.3
```
