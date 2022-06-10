# Overview

This repo provide the code and data for the paper **[An analysis of 45 large-scale wastewater sites in England to estimate SARS-CoV-2 community prevalence ](https://www.researchsquare.com/article/rs-770963/v1)** by Morvan et al. This paper will appear soon in Nature Communications.

The abstract of the paper is below:

Accurate surveillance of the COVID-19 pandemic can be weakened by under-reporting of cases, particularly due to asymptomatic or pre-symptomatic infections, resulting in bias. Quantification of SARS-CoV-2 RNA in wastewater can be used to infer infection prevalence, but uncertainty in sensitivity and considerable variability has meant that accurate measurement remains elusive. Data from 45 sewage sites in England, covering 31% of the population, shows that SARS-CoV-2 prevalence is estimated to within 1.1% of estimates from representative prevalence surveys (with 95% confidence). Using machine learning and phenomenological models, differences between sampled sites, particularly the wastewater flow rate, influence prevalence estimation and require careful interpretation. SARS-CoV-2 signals in wastewater appear 4-5 days earlier in comparison to clinical testing data but are coincident with prevalence surveys suggesting that wastewater surveillance can be a leading indicator for symptomatic viral infections. Surveillance for viruses in wastewater complements and strengthens clinical surveillance, with significant implications for public health.  

The code is written in python (through jupyter notebooks), so hopefully much of the analysis is self-explanatory. If something is not present, or is not working please get in touch with Kath O'Reilly (kathleen.oreilly@lshtm.ac.uk) who will try to assist.

## Instructions for use.

"01_CIS_WW_analysis" is the file that carried out the regression and XGB modelling.

This files loads in the 'agg_data_inner_11_cis_mar22.csv' file, which consists of WW data for the sub-regions and the corresponding CIS positivity estimates. Important fields;
- 'sars_cov2_gc_l_mean' the WW data in units of gc/l (note this is the mean of all samples from within the sub-region, weighted to account for catchment size)
- 'median_prob' the CIS positivity estimates, this field is the median, the fields 'll' and 'ul' are the lower and upper 95% limits.

The file runs through the code that generates the MAE for each model, and the predictions.

It is not possible to generate the regional estimates as the aggregation from sub-region to region requires interaction with ONS files, which we cannot provide.

Figures for the importance of variables in the XBG model, and partial dependency plots, are provided.

The code for the Pearson's correlation coefificent for all variables are also provided.


