import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
from matplotlib import pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram

course_descr = []
titles = []


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs, labels=titles, leaf_rotation=90)
    # print("linkage_matrix")
    # print(linkage_matrix)


with open("cs_courses/CISC.txt", errors="ignore") as file:
    # Read each line in the file
    title_idx = 0
    for line in file:
        # print(line)
        if line.startswith("CISC"):
            # Print each line
            words = line.split(".")
            titles.append(words[0].strip())
            # print("title")
            # print("\t" + words[1].strip())
            title_idx += 1

        elif line.startswith("Attribute"):
            words = line.split(":")
            print(words[1].strip())
            print(titles[title_idx - 1])
            titles[title_idx - 1] += " --- " + words[1].strip()

        elif (
            line == "\n"
            or line.startswith("Mutually Exclusive")
            or line.startswith("Prerequisite")
        ):
            continue
        else:
            print("\t\tdescription")
            # print(line.strip()[0])

            course_descr.append(line.strip())

print(len(course_descr))
print(len(titles))

new_titles = []
new_desc = []

for i in range(len(titles)):
    if (" --- ") in titles[i]:
        new_titles.append(titles[i])
        new_desc.append(course_descr[i])

vectorizer = CountVectorizer(stop_words="english", ngram_range=(1, 1))


# so when I do them togther i seem to get the same size vector
vectors = vectorizer.fit_transform(course_descr)
# print(vectors[0])
# print(vectors[1].shape)
# print(vectors[2].shape)
# print(vectors)
# print(vectorizer.vocabulary_)
similarity = cosine_similarity(vectors)
print(similarity)

model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)

model = model.fit(similarity)

plt.title("Hierarchical Clustering Dendrogram")
# plot the top three levels of the dendrogram
plot_dendrogram(model, truncate_mode="level", p=13)

# plt.rot90()
# plt.figure(figsize=(50, 50))
# plt.xlabel("Number of points in node (or index of point if no parenthesis).")
# plt.xticks(ticks=[5, 10, 15, 20, 25], labels=titles, rotation=90)
plt.show()
