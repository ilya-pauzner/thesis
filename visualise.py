import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot(data, x, y, filename):
    sns.relplot(data=data, x=x, y=y, hue="algo", style="algo", row="abs_task_type", s=200)
    plt.tight_layout()
    plt.savefig(filename)


df = pd.read_csv('results.csv')
algos = {"Initial", "FirstFitDecreasing", "FirstFitDecreasing + migopt", "PyVPSolver", "PyVPSolver + migopt", "Sercon"}
filtered_df = df[df.algo.isin(algos)]
filtered_df = filtered_df.assign(abs_task_type=filtered_df["task_type"].abs())

plot(filtered_df[filtered_df["task_type"] > 0], "active", "migr", "scores_shrink.png")
plot(filtered_df[filtered_df["task_type"] < 0], "active", "migr", "scores_fullpack.png")
plot(filtered_df[filtered_df["task_type"] > 0], "host_count", "time", "times_shrink.png")
plot(filtered_df[filtered_df["task_type"] < 0], "host_count", "time", "times_fullpack.png")
