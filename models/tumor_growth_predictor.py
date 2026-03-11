def predict_growth(tumor_percentages):

    if len(tumor_percentages) < 2:
        return None

    growth_rates = []

    for i in range(1, len(tumor_percentages)):
        growth = tumor_percentages[i] - tumor_percentages[i-1]
        growth_rates.append(growth)

    avg_growth = sum(growth_rates) / len(growth_rates)

    predicted_next = tumor_percentages[-1] + avg_growth

    return avg_growth, predicted_next