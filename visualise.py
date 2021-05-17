import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df = pd.read_csv('results.csv')
algos = {"Initial", "FirstFitDecreasing", "FirstFitDecreasing + migopt", "PyVPSolver", "PyVPSolver + migopt", "Sercon"}
filtered_df = df[df.algo.isin(algos)]
filtered_df = filtered_df.assign(abs_task_type=filtered_df["task_type"].abs())

sns.relplot(data=filtered_df[filtered_df["task_type"] > 0], x="active", y="migr", hue="algo", style="algo",
            row="abs_task_type", s=200)
plt.savefig("scores_shrink.png")
sns.relplot(data=filtered_df[filtered_df["task_type"] < 0], x="active", y="migr", hue="algo", style="algo",
            row="abs_task_type", s=200)
plt.savefig("scores_fullpack.png")
sns.relplot(data=filtered_df[filtered_df["task_type"] > 0], x="host_count", y="time", hue="algo", style="algo",
            row="abs_task_type", s=200)
plt.savefig("times_shrink.png")
sns.relplot(data=filtered_df[filtered_df["task_type"] < 0], x="host_count", y="time", hue="algo", style="algo",
            row="abs_task_type", s=200)
plt.savefig("times_fullpack.png")
