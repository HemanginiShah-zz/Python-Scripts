#How to simplify your dataset
# #
# From the one and only, Siraj Raval!
#   -> https://youtu.be/K796Ae4gLlY?list=PL2-dafEMk2A7YdKv4XfKpfbTH5z6rEEj3

import numpy as np
np.random.seed(1)

# Step One: Create dataset

mu_vec1 = np.array([0,0,0]) # sample mean
cov_mat1 = np.array([[1,0,0],[0,1,0],[0,0,1]]) #sample covariance

class1_sample = np.random.multivariate_normal(mu_vec1, cov_mat1, 20).T
print(class1_sample)
#3 x 20 matrix, 3 columns with 20 rows each

#recreate independent tensor
mu_vec2 = np.array([1,1,1]) # sample mean
cov_mat2 = np.array([[1,0,0],[0,1,0],[0,0,1]]) #sample covariance
class2_sample = np.random.multivariate_normal(mu_vec2, cov_mat2, 20).T
print(class2_sample)

#Todo: visualize data using matplotlib

#Step three: merge data into single dataset

all_samples = np.concatenate((class1_sample, class2_sample), axis=1)
print(all_samples)

#Step 3: compute the demensional mean vector, it will help compute the covariance matrix
#Mean for each feature
mean_x = np.mean(all_samples[0,:])
mean_y = np.mean(all_samples[1,:])
mean_z = np.mean(all_samples[2,:])

#3d mean vector
mean_vector = np.array([[mean_x], [mean_y], [mean_z]])
print('\n', mean_vector)
