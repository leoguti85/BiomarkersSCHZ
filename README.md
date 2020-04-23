## Stable Biomarker Identification For Predicting Schizophrenia in the Human Connectome

Here you can find the scripts to reproduce the results of our paper. Check out the preprint [here](https://www.biorxiv.org/content/10.1101/711135v1)


This code was tested on Linux/Mint 18.2 with python 2.7.

## Usage
The entire dataset can be downloded from [Zenodo](https://doi.org/10.5281/zenodo.3758534). The Funcitonal and Structural connectivity matrices should be vectorized and saved in the folders FC and SC.

### Finding stable biomarkers for schizophrenia prediction
The following script will run recursive feature elimination, support vector machines (RFE-SVM) to select features, compute the stability of selected biomarkers and predicting schizophrenia. 

Example,
```
python mainScript_abs.py -connectivity Structural -resolution 83
```
Will run RFE-SVM on structural connectomes with 83 x 83 parcellations. The results are saved in the *mat/abs_subcortical* folder.

### Accuray versus stability plots
Plotting accuracy and stability for different parameters. Figures will be saved in the *results/* folder as .png files.

```
python plotting_accuracy_stability.py
```
<p align="center">
<img src="results/sc_83.png">
</p>


