# Multi-source-uncertainty

Data and codes related with "Uncertainty-Aware Multi-Source Modeling for Materials Property Prediction: A Case Study in Ferroelectrics".

## Requirements
- R 4.4.1

## Project Structure

This project implements a resampling-based approach for quantifying sample or source specific uncertainties of material properties, with a focus on ferroelectrics. The resulting uncertainties are embedded into a Kriging surrogate model, enabling adaptive weighting of data according to reliability, supporting seperately treating multi-source data.
Since the code for the four properties (corresponding to four folders) is very similar, we recommend referring to the R script in any folder to understand the code structure and how to apply this method to other multi-source materials data.

- Each folder contains the original multi-source data and a R script.
- The R script integrates data preprocessing, feature selection (alternative), bootstrap ensemble, uncertainty quantification and optimization, surrogate model training, model evaluation and figure plotting (alternative).
- One can directly run the R script with original data, required packages are listed in the script.

## Raw Measurements

 The raw measurement files for the P–E and S–E loops of the five validation samples are included in the folder "P-E S-E measurement files" in TXT form, as exported from the Radiant equipment.
