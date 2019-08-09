import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sys import argv

def plot_validation(fn) :
    df = pd.read_csv(fn)
    sns.lineplot(data=df,x="i", y="error", hue="tfm_interpolator", palette="tab10", linewidth=2.5)
    plt.show()

if __name__ == "__main__" :
    plot_validation(argv[1])
