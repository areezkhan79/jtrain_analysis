def print_summary_stats(summary_stats):
    print("\nSummary Statistics:")
    print(summary_stats)

def print_correlation_matrix(correlation_matrix):
    print("\nCorrelation Matrix:")
    print(correlation_matrix)

def print_training_earnings_corr(correlation_matrix):
    training_earnings_corr = correlation_matrix.loc['avgsal', 'totrain']
    print(f"\nCorrelation between 'avgsal' (average salary) and 'totrain' (total employees trained): {training_earnings_corr}")
