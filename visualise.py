import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('results.csv')
algos = {"Initial", "FirstFitDecreasing", "FirstFitDecreasing + migopt", "PyVPSolver", "PyVPSolver + migopt", "Sercon"}
filtered_df = df[df.algo.isin(algos)]
g = sns.FacetGrid(filtered_df, col="task_type")
g.map(sns.scatterplot, "active", "migr", "algo")
plt.legend()
g = sns.FacetGrid(filtered_df, col="task_type")
g.map(sns.scatterplot, "host_count", "time", "algo")
plt.legend()
plt.show()
