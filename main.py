import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from scripts.data_preprocessing import  load_and_preprocess_data
from scripts.regression_analysis import fit_regression_model, calculate_vif, breusch_pagan_test, durbin_watson_test
from scripts.visualizations import plot_regression_results

# Create output directories if they don't exist
if not os.path.exists("output"):
    os.makedirs("output")

if not os.path.exists("output/plots"):
    os.makedirs("output/plots")


# File path and variables
file_path = 'data/JTRAIN.xls'
selected_vars = ['avgsal', 'totrain', 'grant', 'employ', 'sales', 'union', 'year']

# Data loading and preprocessing
data = load_and_preprocess_data(file_path, selected_vars)

# Summary statistics
summary_stats = data.describe()

# Save summary statistics to a separate file
summary_stats.to_csv("output/summary_statistics.csv")

# Correlation matrix
correlation_matrix = data.corr()
# Save correlation matrix to a separate file
correlation_matrix.to_csv("output/correlation_matrix.csv")

# Correlation between 'avgsal' and 'totrain'
correlation = data['avgsal'].corr(data['totrain'])

# Prepare data for regression
X_train = data[['totrain', 'grant', 'employ', 'sales', 'union', 'year']]
y_train = data['avgsal']

# Fit regression model
model = fit_regression_model(X_train, y_train)

# Calculate VIF
vif = calculate_vif(X_train)

# Save VIF to a separate file
vif.to_csv("output/vif.csv")

# Breusch-Pagan test for heteroscedasticity
bp_test = breusch_pagan_test(model)

# Durbin-Watson test for autocorrelation
dw_stat = durbin_watson_test(model)

# Regression diagnostics plots
fitted_values = model.fittedvalues
residuals = model.resid
standardized_residuals = model.get_influence().resid_studentized_internal

plot_regression_results(fitted_values, residuals, standardized_residuals, model, "output/plots")

# Create a report card
with open("output/report_card.txt", "w") as f:
    f.write("Regression Analysis Report\n")
    f.write("==========================\n\n")
    f.write("Summary Statistics:\n")
    f.write(summary_stats.to_string())
    f.write("\n\nCorrelation Matrix:\n")
    f.write(correlation_matrix.to_string())
    f.write(f"\n\nCorrelation between 'avgsal' and 'totrain': {correlation}\n")
    f.write("\n\nRegression Results:\n")
    f.write(model.summary().as_text())
    f.write("\n\nVariance Inflation Factors (VIF):\n")
    f.write(vif.to_string())
    f.write(f"\n\nBreusch-Pagan test for heteroscedasticity:\nLM Statistic: {bp_test[0]}, p-value: {bp_test[1]}\n")
    f.write(f"\n\nDurbin-Watson test for autocorrelation:\nDurbin-Watson Statistic: {dw_stat}\n")
    f.write("\n\nRegression Diagnostics Plots have been saved to the 'output/plots' directory.\n")

print("\n")
print("Check Report Card in the output folder.")
print("------------------------------------------------\n")
