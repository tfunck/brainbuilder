import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import numpy as np
from glob import glob

fn_list = glob("receptor_to_mri/best_slab_position_*csv")

dfList = []
for fn in fn_list  :
    print(fn)
    dfList.append( pd.read_csv(fn) )

df=pd.concat(dfList)
df["RegSchedule"] = df["RegSchedule"] + "_" + df["Metric"]
df["Y"] = (df["y"] + df["y_end"]) / 2.
#for i, sch in enumerate(np.unique(df["RegSchedule"])):
#    mapping[sch] = float(i)
#df["x"] = list(map(lambda x: mapping[x], df["RegSchedule"]))
#df["y"].append(df["y_end"])
df.to_csv('temp.csv')
#g = sns.FacetGrid(df, row="Offset", size="Metric", hue="RegSchedule",sharey=False,sharex=True)
print(df)
g = sns.lineplot(data=df,y="Y", x="RegSchedule", style="Offset",   hue="slab", alpha=0.75, palette=sns.color_palette("hls"))
plt.show()
