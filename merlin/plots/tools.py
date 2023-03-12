import matplotlib.pyplot as plt
import seaborn as sns


def plot_histogram(data, value):
    median = data[value].median()
    fig = plt.figure()
    ax = sns.histplot(data=data, x=value, kde=True, line_kws={"linestyle": "--"})
    ax.lines[0].set_color("#555555")
    plt.axvline(median, linestyle=":", color="tab:red")
    plt.text(
        x=median + ax.get_xlim()[1] * 0.01,
        y=ax.get_ylim()[1] * 0.98,
        s=f"median = {median:.0f}",
        c="tab:red",
        va="top",
        ha="left",
    )
    return fig
