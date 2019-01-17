import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sys import argv

df = pd.read_csv(argv[1])
out_fn = argv[2]
print(df)
#g = sns.FacetGrid("Dice", row="Test", col="Slab", sharey=False)
#g.map(plt.scatter, "Iteration", "Dice", color=".3", fit_reg=False, x_jitter=.1);
#g.add_legend()
sns.relplot(x="Iteration", y="Acc", hue="Test", row="Slab", data=df)
plt.savefig(out_fn)

