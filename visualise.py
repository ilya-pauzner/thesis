import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot(data, x, y, filename):
    g = sns.relplot(data=data, x=x, y=y, hue="algo", style="algo", row="abs_task_type", s=200, facet_kws={"legend_out": True})
    plt.gcf().subplots_adjust(bottom=0.25)
    g._legend.set_bbox_to_anchor((0.5, 0.1))
    g._legend.set_frame_on(True)
    plt.savefig(filename)


df = pd.read_csv('results.csv')
algos = {"Initial", "FirstFitDecreasing", "FirstFitDecreasing + migopt", "PyVPSolver", "PyVPSolver + migopt", "Sercon"}
filtered_df = df[df.algo.isin(algos)]
filtered_df = filtered_df.assign(abs_task_type=filtered_df["task_type"].abs())

plot(filtered_df[filtered_df["task_type"] > 0], "active", "migr", "scores_shrink.png")
plot(filtered_df[filtered_df["task_type"] < 0], "active", "migr", "scores_fullpack.png")
plot(filtered_df[filtered_df["task_type"] > 0], "host_count", "time", "times_shrink.png")
plot(filtered_df[filtered_df["task_type"] < 0], "host_count", "time", "times_fullpack.png")
