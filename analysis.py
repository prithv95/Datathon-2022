# Importing packages

from operator import delitem
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
from tqdm import tqdm
from sklearn.cluster import KMeans


# ===================================================================

def string2nparray(string_array: str) -> np.ndarray:
    string_array_list = string_array.split()
    array = np.array([float(re.sub(r'[\[\],]', '', element)) for element in string_array_list])
    return array


if not os.path.exists('./data/embeddings.txt'):
    print('Reading data')
    df1 = pd.read_csv('data/cnn_samples-54b19b96f3c0775b116bad527df8c7b5.csv')
    print(df1.head(3))
    print('Converting string array to embeddings.')
    embeddings = []
    for string_array in tqdm(df1['embedding'].values):
        embeddings.append(string2nparray(string_array))
    print('Saving embeddings.txt')
    embeddings = np.array(embeddings)
    np.savetxt('./data/embeddings.txt', embeddings, delimiter=',')
else:
    print('embeddings.txt exists')
    print('reading from embeddings.txt')
    embeddings = np.genfromtxt('./data/embeddings.txt', delimiter=',')

# ===================================================================

# KMeans

print('KMeans elbow method')
inertias = []
K = list(range(5, 51, 5))

for k in tqdm(K):
	# Building and fitting the model
	km = KMeans(n_clusters=k).fit(embeddings)
	km.fit(embeddings)
	inertias.append(km.inertia_)

print('Elbow method completed.')
plt.plot(K, inertias, 'bx-')
plt.xlabel('Values of K')
plt.ylabel('Inertia')
plt.title('The Elbow Method using Inertia')
plt.grid()
plt.show()
