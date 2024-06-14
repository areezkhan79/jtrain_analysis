import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as stats



def plot_regression_results(fitted_values, residuals, standardized_residuals, results, plot_dir):

    # Residual Plot
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=fitted_values, y=residuals)
    plt.axhline(0, color='red', linestyle='--')
    plt.title("Residuals vs Fitted Values")
    plt.xlabel("Fitted Values")
    plt.ylabel("Residuals")
    plt.grid(True)
    plt.legend(['Residuals'])
    plt.savefig(f'{plot_dir}/residuals_vs_fitted.png')
    plt.close()

    # Q-Q Plot
    plt.figure(figsize=(8, 6))
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.plot(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100), color='red')
    plt.title("Q-Q Plot: Normality of Residuals")
    plt.xlabel("Theoretical Quantiles")
    plt.ylabel("Sample Quantiles")
    plt.grid(True)
    plt.legend(['Q-Q Plot'])
    plt.savefig(f'{plot_dir}/qq_plot_residuals.png')
    plt.close()

    # Scale-Location Plot
    sqrt_abs_std_residuals = np.sqrt(np.abs(standardized_residuals))
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=fitted_values, y=sqrt_abs_std_residuals)
    plt.title("Scale-Location Plot: Homoscedasticity Check")
    plt.xlabel("Fitted Values")
    plt.ylabel("Square Root of Absolute Standardized Residuals")
    plt.grid(True)
    plt.legend(['Homoscedasticity Check'])
    plt.savefig(f'{plot_dir}/scale_location_plot.png')
    plt.close()

    # Cook's Distance Plot
    cook_distance = results.get_influence().cooks_distance[0]
    plt.figure(figsize=(8, 6))
    plt.stem(cook_distance, markerfmt=",")
    plt.title("Cook's Distance Plot: Influential Observations")
    plt.xlabel("Observation Index")
    plt.ylabel("Cook's Distance")
    plt.grid(True)
    plt.legend(["Cook's Distance"])
    plt.savefig(f'{plot_dir}/cooks_distance_plot.png')
    plt.close()

    # Leverage-Residual Squared Plot
    leverage = results.get_influence().hat_matrix_diag
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=leverage, y=standardized_residuals**2)
    plt.title("Leverage-Residual Squared Plot")
    plt.xlabel("Leverage")
    plt.ylabel("Standardized Residuals Squared")
    plt.grid(True)
    plt.legend(['Leverage-Residual Squared'])
    plt.savefig(f'{plot_dir}/leverage_residuals_squared_plot.png')
    plt.close()