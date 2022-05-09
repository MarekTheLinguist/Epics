import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import numpy as np
from sklearn.cluster import KMeans
from kmodes.kmodes import KModes



## load up the data
df = pd.read_csv("/home/marek/Documents/Research/Epics/Exp sem.csv", sep = ";")
print(df.columns)
## transform each row into a string
## remove non-digit elements of a string
df["MEDIUM \nof communication"] = df["MEDIUM \nof communication"].apply(lambda x: str(x))
df["MEDIUM \nof communication"] = df["MEDIUM \nof communication"].apply(lambda x: re.sub("[^0-9]", "", x))
df["VERTICAL \ntransmission?"] = df["VERTICAL \ntransmission?"].apply(lambda x: re.sub("[^0-9]", "", x))


## sometimes there are more numbers; we can accept only one
def onlyFirst(medium):
    if len(medium) > 1:
        return medium[0]
    else:
        return medium

## apply the function to each row
df["MEDIUM \nof communication"] = df["MEDIUM \nof communication"].apply(lambda x: onlyFirst(x))
df["VERTICAL \ntransmission?"] = df["VERTICAL \ntransmission?"].apply(lambda x: onlyFirst(x))

## convert into floats
df["MEDIUM \nof communication"] = df["MEDIUM \nof communication"].apply(lambda x: float(x))
df["VERTICAL \ntransmission?"] = df["VERTICAL \ntransmission?"].apply(lambda x: float(x))


## initialise the model

kmeans = KMeans(n_clusters = 5, init = "k-means++", random_state = 42)

## selecting features
data_in = df[["MEDIUM \nof communication", "VERTICAL \ntransmission?"]]


## check the optimal number of clusters
cost = []
K = range(1,6)
for num_clusters in list(K):
    kmode = KModes(n_clusters=num_clusters, init = "random", n_init = 5, verbose=1)
    kmode.fit_predict(data_in)
    cost.append(kmode.cost_)

plt.plot(K, cost, 'bx-')
plt.xlabel('No. of clusters')
plt.ylabel('Cost')
plt.title('Elbow Method For Optimal k')
plt.show()

kmeans.fit(data_in)
print(len(kmeans.predict(data_in)))
clusters = kmeans.fit_predict(data_in)
data_in.insert(0, "Cluster", clusters, True)
print(data_in)
filtered_label0 = data_in[clusters == 0]
## Plot the data
## CTD

data_in.to_csv("~/Documents/Research/Epics/cluster.csv", sep = ";", index = False)
