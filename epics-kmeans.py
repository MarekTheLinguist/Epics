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

df["MEDIUM \nof communication"] = df["MEDIUM \nof communication"].apply(lambda x: str(x))
df['FEEDBACK'] = df['FEEDBACK'].apply(lambda x: str(x))
df["REFERENTIAL (end) \nvs \nCOORDINATION (means)"] = df["REFERENTIAL (end) \nvs \nCOORDINATION (means)"].apply(lambda x: str(x))
df["SIGNAL SPACE"] = df["SIGNAL SPACE"].apply(lambda x: str(x))
df["MEANING SPACE"] = df["MEANING SPACE"].apply(lambda x: str(x))
df['COMMUNICATION TYPE'] = df['COMMUNICATION TYPE'].apply(lambda x: str(x))
df["GROUP SIZE"] = df["GROUP SIZE"].apply(lambda x: str(x))
df['PARTICIPANTS of the MAIN study: AGE'] = df['PARTICIPANTS of the MAIN study: AGE'].apply(lambda x: str(x))
df['PARTICIPANTS of the MAIN study: POPULATION'] = df['PARTICIPANTS of the MAIN study: POPULATION'].apply(lambda x: str(x))
df['TURN-TAKING'] = df['TURN-TAKING'].apply(lambda x: str(x))
df['INTERCHANGEABILITY \nof the signaller/receiver roles'] = df['INTERCHANGEABILITY \nof the signaller/receiver roles'].apply(lambda x: str(x))
df['SIMULTANEOUS\ninteraction?'] = df['SIMULTANEOUS\ninteraction?'].apply(lambda x: str(x))

## remove non-digit elements of a string
df['FEEDBACK'] = df['FEEDBACK'].apply(lambda x: re.sub("[^0-9]", "", x))
df["REFERENTIAL (end) \nvs \nCOORDINATION (means)"] = df["REFERENTIAL (end) \nvs \nCOORDINATION (means)"].apply(lambda x: re.sub("[^0-9]", "", x))
df["MEDIUM \nof communication"] = df["MEDIUM \nof communication"].apply(lambda x: re.sub("[^0-9]", "", x))
df["VERTICAL \ntransmission?"] = df["VERTICAL \ntransmission?"].apply(lambda x: re.sub("[^0-9]", "", x))
df["SIGNAL SPACE"] = df["SIGNAL SPACE"].apply(lambda x: re.sub("[^0-9]", "", x))
df["MEANING SPACE"] = df["MEANING SPACE"].apply(lambda x: re.sub("[^0-9]", "", x))
df['COMMUNICATION TYPE'] = df['COMMUNICATION TYPE'].apply(lambda x: re.sub("[^0-9]", "", x))
df["GROUP SIZE"] = df["GROUP SIZE"].apply(lambda x: re.sub("[^0-9]", "", x))
df['PARTICIPANTS of the MAIN study: AGE'] = df['PARTICIPANTS of the MAIN study: AGE'].apply(lambda x: re.sub("[^0-9]", "", x))
df['PARTICIPANTS of the MAIN study: POPULATION'] = df['PARTICIPANTS of the MAIN study: POPULATION'].apply(lambda x: re.sub("[^0-9]", "", x))
df['TURN-TAKING'] = df['TURN-TAKING'].apply(lambda x: re.sub("[^0-9]", "", x))
df['INTERCHANGEABILITY \nof the signaller/receiver roles'] = df['INTERCHANGEABILITY \nof the signaller/receiver roles'].apply(lambda x: re.sub("[^0-9]", "", x))
df['SIMULTANEOUS\ninteraction?'] = df['SIMULTANEOUS\ninteraction?'].apply(lambda x: re.sub("[^0-9]", "", x))
## sometimes there are more numbers; we can accept only one
def onlyFirst(medium):
    if len(medium) > 1:
        return medium[0]
    else:
        return medium

## apply the function to each row
df["MEDIUM \nof communication"] = df["MEDIUM \nof communication"].apply(lambda x: onlyFirst(x))
df["VERTICAL \ntransmission?"] = df["VERTICAL \ntransmission?"].apply(lambda x: onlyFirst(x))
df["REFERENTIAL (end) \nvs \nCOORDINATION (means)"] = df["REFERENTIAL (end) \nvs \nCOORDINATION (means)"].apply(lambda x: onlyFirst(x))
df["SIGNAL SPACE"] = df["SIGNAL SPACE"].apply(lambda x: onlyFirst(x))
df["MEANING SPACE"] = df["MEANING SPACE"].apply(lambda x: onlyFirst(x))
df['FEEDBACK'] = df['FEEDBACK'].apply(lambda x: onlyFirst(x))
df['COMMUNICATION TYPE'] = df['COMMUNICATION TYPE'].apply(lambda x: onlyFirst(x))
df['PARTICIPANTS of the MAIN study: AGE'] = df['PARTICIPANTS of the MAIN study: AGE'].apply(lambda x: onlyFirst(x))
df['GROUP SIZE'] = df['GROUP SIZE'].apply(lambda x: onlyFirst(x))
df['TURN-TAKING'] = df['TURN-TAKING'].apply(lambda x: onlyFirst(x))
df['INTERCHANGEABILITY \nof the signaller/receiver roles'] = df['INTERCHANGEABILITY \nof the signaller/receiver roles'].apply(lambda x: onlyFirst(x))
df['SIMULTANEOUS\ninteraction?'] = df['SIMULTANEOUS\ninteraction?'].apply(lambda x: onlyFirst(x))
## convert into floats
df["MEDIUM \nof communication"] = df["MEDIUM \nof communication"].apply(lambda x: float(x))
df["VERTICAL \ntransmission?"] = df["VERTICAL \ntransmission?"].apply(lambda x: float(x))
df['Cites per year'] = df['Cites per year'].apply(lambda x: float(x))
df["MEANING SPACE"] = df["MEANING SPACE"].apply(lambda x: float(x))
df["SIGNAL SPACE"] = df["SIGNAL SPACE"].apply(lambda x: float(x))
df["REFERENTIAL (end) \nvs \nCOORDINATION (means)"] = df["REFERENTIAL (end) \nvs \nCOORDINATION (means)"].apply(lambda x: float(x))
df['COMMUNICATION TYPE'] = df['COMMUNICATION TYPE'].apply(lambda x: float(x))
df['PARTICIPANTS of the MAIN study: AGE'] = df['PARTICIPANTS of the MAIN study: AGE'].apply(lambda x: float(x))
df['PARTICIPANTS of the MAIN study: POPULATION'] = df['PARTICIPANTS of the MAIN study: POPULATION'].apply(lambda x: float(x))
df['TURN-TAKING'] = df['TURN-TAKING'].apply(lambda x: float(x))
df['INTERCHANGEABILITY \nof the signaller/receiver roles'] = df['INTERCHANGEABILITY \nof the signaller/receiver roles'].apply(lambda x: float(x))
df['GROUP SIZE'] = df['GROUP SIZE'].apply(lambda x: float(x))
df['SIMULTANEOUS\ninteraction?'] = df['SIMULTANEOUS\ninteraction?'].apply(lambda x: float(x))

df.to_csv("~/Documents/Research/Epics/database_no_strings_attached.csv", sep = ";", index = False)
## initialise the model

kmeans = KMeans(n_clusters = 6, init = "k-means++", algorithm = "full", random_state = 42)

## selecting features
print(df.columns[7:16])
print(df.columns[19:21])

data_in = df[['Year', 'VERTICAL \ntransmission?',
       'REFERENTIAL (end) \nvs \nCOORDINATION (means)',
       'MEDIUM \nof communication', 'SIGNAL SPACE', 'MEANING SPACE',
       'FEEDBACK', 'COMMUNICATION TYPE', 'GROUP SIZE',
       'PARTICIPANTS of the MAIN study: AGE', 'TURN-TAKING', 'INTERCHANGEABILITY \nof the signaller/receiver roles']]
print(data_in)
## check the optimal number of clusters
cost = []
K = range(1,10)
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
df.insert(0, "Cluster", clusters, True)

print(data_in)
## Plot the data
## CTD

df.to_csv("~/Documents/Research/Epics/cluster.csv", sep = ";", index = False)


data_in = np.array(data_in)
sns.scatterplot(data_in[clusters == 0, 0], data_in[clusters == 0, 1], color = 'yellow', label = 'Cluster 1',s=50)
sns.scatterplot(data_in[clusters == 1, 0], data_in[clusters == 1, 1], color = 'blue', label = 'Cluster 2',s=50)
sns.scatterplot(data_in[clusters == 2, 0], data_in[clusters == 2, 1], color = 'green', label = 'Cluster 3',s=50)

# labeling
plt.grid(False)
plt.title('Three clusters')
plt.xlabel('Status_type')
plt.ylabel('values')
plt.legend()
plt.show()
