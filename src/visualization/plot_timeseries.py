import itertools
import matplotlib.pyplot as plt
from collections import defaultdict 

def plot_multi_label_time_series(features, multi_labels, category_names, max_series_per_class=10):
    # Grouping time series by unique combinations of labels
    class_groups = defaultdict(list)
    for feature, labels in zip(features, multi_labels):
        # Create a unique label for each combination of labels
        label_combination = tuple(category_names[i] for i, label in enumerate(labels) if label)
        class_groups[label_combination].append(feature)

    # Define a list of specified colors
    specified_colors = [
        '#97c139', '#c5de89', '#6a8a22', '#006600', '#658d67',
        '#61c086', '#3f5f44', '#34677d', '#004359', '#77a9b5',
        '#002f6c', '#6c8cc7', '#041e42'
    ]

    # Ensuring we have enough colors for each unique label combination by cycling through the specified colors
    color_cycle = itertools.cycle(specified_colors)

    # Plotting a limited number of series per unique label combination with specified colors
    for label_combination, time_series_group in class_groups.items():
        color = next(color_cycle)  # Get the next color from the cycle
        plt.figure()
        for series in time_series_group[:max_series_per_class]:
            plt.plot(series, color=color)
        plt.title(f"Time Series for Label Combination: {', '.join(label_combination)}")
        plt.xlabel("Time Index")
        plt.ylabel("Value")
        plt.show()

