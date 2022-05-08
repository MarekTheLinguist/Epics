import pandas as pd
import re
from sklearn.cluster import KMeans


## load up the data
df = pd.read_csv("/home/marek/Documents/Research/Epics/Exp sem.csv", sep = ";")

## transform each row into a string
## remove non-digit elements of a string
df["MEDIUM \nof communication"] = df["MEDIUM \nof communication"].apply(lambda x: str(x))
df["MEDIUM \nof communication"] = df["MEDIUM \nof communication"].apply(lambda x: re.sub("[^0-9]", "", x))

## sometimes there are more numbers; we can accept only one
def onlyFirst(medium):
    if len(medium) > 1:
        return medium[0]
    else:
        return medium

## apply the function to each row
df["MEDIUM \nof communication"] = df["MEDIUM \nof communication"].apply(lambda x: onlyFirst(x))
print(df["MEDIUM \nof communication"])
## convert into floats

df["MEDIUM \nof communication"] = df["MEDIUM \nof communication"].apply(lambda x: float(x))
