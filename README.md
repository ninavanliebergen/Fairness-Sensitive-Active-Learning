# Fairness-Sensitive Active Learning

This repository is part of the thesis:
Fairness-Sensitive Active Learning

**Abstract**
_Public organizations increasingly use machine learning models for decision support, but they are also generally constrained by limited labeled data.  Fortunately, active learning has proven to be useful to efficiently select instances important for labeling, thus reducing the impact of this limitation. Models used by the public sector have also shown to exhibit variour biases (e.g. towards gender or ethnicity), phenomena that highlight the need and urgency to address fairness concerns. This research aims to experimentally study the relationship between active learning and fairness. Its objective is to assess fairness risks tied to active learning and explore fairness-sensitive methods for active learning. In this study, two common active learning types are evaluated for accuracy and unfairness. An existing fairness-focused active learning solution (Fair Active Learning, or FAL) is examined, and two novel fairness-sensitive methods, ReSGrAL and Fair ReSGrAL, are proposed. Our experiments shows that active learning can increase model unfairness beyond the dataset bias, and thus caution is needed when using active learning in sensitive contexts. However, they also show that techniques like ReSGrAL can mitigate unfairness without sacrificing accuracy, improving fairness in decision support systems._

In this repo, you can find the code used.

## Dependencies
- Python 3.10
- pandas
- numpy
- scikit-learn
- modAL (https://modal-python.readthedocs.io/en/latest/#)
- fairlearn (https://fairlearn.org/)

## Running the code:

### General Sampling Strategies and ReSGrAL:
For running one of the sampling strategies, do:
```
 python prompts_reader.py <n_samples> <dataset_path> <target_value> <sensitive_attr> <subgroups> <dataset_name>
```
For example, when sampling with random sampling on the adult dataset:
```
  python Random.py 200 ../Datasets/AdultDatasetGroupEncoded.csv income gender groups Adult
```
The results will be saved in ../SamplingStrategies/Results/

### FAL:
When running FAL, there is the option to run it on a subpart of the dataset and also include the amount of workers:
```
python FAL.py <n_samples> <dataset_path> <target_value> <sensitive_attr> <subgroups> <dataset_name> <fraction_dataset> <amount_of_workers>
```
For example, to run FAL on only 20% of the dataset with 16 workers:
```
python FAL.py 200 ../Datasets/AdultDatasetGroupEncoded.csv income gender groups Adult 0.2 16
```
However, when the fraction of the dataset and the amount of workers are not specified, it is runned on the total (100%) of the dataset in serial.

### FairReSGrAL:
When running FairReSGrAL, one needs to specify the threshold:
```
python ReSGrAL_Uncertainty.py <n_samples> <dataset_path> <target_value> <sensitive_attr> <subgroups> <dataset_name> <threshold> <fraction_dataset> <amount_of_workers>
```
For exaxample, to run FairReSGrAL with a threshold of 0.3, run:
```
python ReSGrAL_Uncertainty.py 200 ../Datasets/AdultDatasetGroupEncoded.csv income gender groups Adult 0.3
```

## Common Errors:
When the following errors arises:
```
ValueError: This solver needs samples of at least 2 classes in the data, but the data contains only one class: 0
```
this means that the first 8 random samples only contain 1 type of label (only 0's or only 1's). To solve this, just run it again.
