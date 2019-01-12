import pandas as pd
import numpy as np
import sys

fn = sys.argv[1]
ligand = sys.argv[2]
slab = sys.argv[3]

df = pd.read_csv(fn)

if ligand in df["ligand"] :
    print("Error: could not find ligand ", ligand, "in ", fn)
    exit(1)

order_max =df["order"].loc[ (df["slab"] == int(slab)) ].max() 
order = df["order"].loc[ (df["ligand"] == "flum") & (df["slab"] == int(slab)) ]

print("Slice Order:", " ".join( np.flip(order.astype(str).values, axis=0) ))
print("Slice Location:", " ".join( np.flip(np.array(order_max - order ).astype(str), axis=0  ) ))

