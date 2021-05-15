import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df = pd.read_csv('results.csv')
algos = {"Initial", "FirstFitDecreasing", "FirstFitDecreasing + migopt", "PyVPSolver", "PyVPSolver + migopt", "Sercon"}
filtered_df = df[df.algo.isin(algos)]
filtered_df = filtered_df.assign(abs_task_type=filtered_df["task_type"].abs())
sns.relplot(data=filtered_df, x="active", y="migr", style="algo", col="abs_task_type", row="shrink")
sns.relplot(data=filtered_df, x="host_count", y="time", style="algo", col="abs_task_type", row="shrink")
plt.show()
