import matplotlib.pyplot as plt
from random import randint
import numpy as np

img = plt.imread('mauritius_beach.png')

plt.imshow(img)
plt.show()
k = int(input("Enter number of clusters: "))

img_height = img.shape[0]
img_width = img.shape[1]
n_channels = img.shape[2]

centroids = []
for i in range(k):
    centroids.append(img[randint(0, img_height)][randint(0, img_width)])  # centroid k random centroids

centroids = np.array(centroids)
pixel_assignments = np.zeros((img_height, img_width, 1), dtype=int)
distance_matrix = np.zeros((img_height, img_width, k), dtype=float)

for _ in range(10):
    for i in range(img_height):  # how can i vectorize these loops?
        for j in range(img_width):
            for c in range(k):
                distance_matrix[i][j][c] = np.linalg.norm(img[i][j] - centroids[c])  # colour channel distances
            pixel_assignments[i][j] = np.argmin(distance_matrix[i][j])

    # find average of respective centroid pixels
    for i in range(k):  # sum of pixels belonging to i
        centroid_sum = np.zeros(3, float)
        count = 0
        for x in range(img_height):
            for y in range(img_width):
                if pixel_assignments[x][y] == i:
                    centroid_sum = np.add(centroid_sum, img[x][y])
                    count = count + 1
        centroids[i] = centroid_sum / float(count)

# now form a re-segmented image
output_img = np.zeros((img_height, img_width, 3), float)
for i in range(img_height):
    for j in range(img_width):
        output_img[i][j] = centroids[pixel_assignments[i][j]]

plt.imshow(output_img)
plt.show()

print("Hello")
