# -*- coding: utf-8 -*-
  

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
np.random.seed(42)
from sklearn.preprocessing import StandardScaler 
import seaborn as sns 
import cv2
import os 
from scipy.stats import stats
from PIL import Image
import matplotlib.image as mpimg

class PCA:

    def __init__(self, dir, img_name):#, n_components):
        
        #self.n = n_components
        self.name = img_name
        self.dir = dir
        self.img = cv2.cvtColor(cv2.imread(os.path.join(dir, img)), cv2.COLOR_BGR2RGB) 
        #self.img = cv2.imread(os.path.join(dir, img))

        # Split the image to three channels 
        self.blue, self.green, self.red = cv2.split(self.img)
        self.colors = np.array([self.red, self.green, self.blue])
        self.N_COMPONENTS = [1, 10, 20, 50, 100, 200, 300, 500, 768]
  
    def standardize(self):
        # Standardize the image (x-mu)/sigma 
        standard = []
        mu_all = []
        stds = []
        for color in self.colors:
            mu = np.mean(color, axis=0)
            std = np.std(color, axis=0)
            x = (color - mu)/ std 
            standard.append(x)
            mu_all.append(mu)
            stds.append(std)
        self.std = stds
        self.mu = mu_all
        return standard 
        
    def compute_covariance(self):
         
        z = self.standardize()
        #z = self.channels
        cov_mat = []
        for i in range(len(z)):
            cov_mat.append(np.cov(z[i], rowvar=False))
            #cov_mat.append(np.cov(z[i], rowvar=False))
        
        return cov_mat 
     
    def fit(self, n_components):

        #if self.n is None:
        #    pass 
        #else:
        self.n = n_components
        
        reconstructions = [[], [], []]
        acc_var = []
        scale = self.standardize()
        cov_mat = self.compute_covariance()

        for i, color in enumerate(self.colors):
            
            eigen_val, eigen_vects = np.linalg.eigh(cov_mat[i])
            
            # Sorting values and vectors in descending order
            new_vals = np.flip(eigen_val)
            new_vects = eigen_vects[:, np.argsort(new_vals)]
            n_comp = new_vects[:,0:self.n]
            new_color = np.dot(scale[i], n_comp)
            project_color = np.dot(new_color,n_comp.T)*self.std[i] + self.mu[i]
            reconstructions[i].append(project_color)
            acc_var.append(sum(new_vals[0:self.n])/sum(new_vals))

        compressed_img = (np.dstack((reconstructions[2][0], reconstructions[1][0], reconstructions[0][0]))).astype(np.uint8)

        return compressed_img, acc_var
            
    # Reconstruct the image with self.N_COMPONENTS components
    def reconstruct(self):
        
        img_list = []
        fig = plt.figure(figsize=(15, 10))
        for i in range(len(self.N_COMPONENTS)):
            compressed_img, _ = self.fit(n_components = self.N_COMPONENTS[i])
            fig.add_subplot(3, 3, i+1)
            plt.imshow(compressed_img)
            plt.title(f'{self.N_COMPONENTS[i]} Principle Components')
            plt.show()
            
    # Plot accumulative variance by every channel
    def plot_variance_by_channels(self):

        var_b = []
        var_g = []
        var_r = []
        var_acc = []
        for i in range(len(self.N_COMPONENTS)):
            
            _, var = self.fit(n_components=self.N_COMPONENTS[i])
            var_b.append(var[0])
            var_g.append(var[1])
            var_r.append(var[2])
            #var_acc.append(sum(var))
        plt.plot(self.N_COMPONENTS, var_b, color='blue')
        plt.plot(self.N_COMPONENTS, var_g, color='green')
        plt.plot(self.N_COMPONENTS, var_r, color='red')
        #plt.plot(self.N_COMPONENTS, var_acc, color='yellow')
        plt.xlabel('Number of Components')
        plt.ylabel('Accumulative Variance')
        plt.legend(('Blue Channel Variance', 'Green channel Variance', 'Red channel Variance')) #, 'Cumulative Variance of channels'))
        plt.show()

if __name__ == "__main__":

    dir = '/home/adweeb/Desktop/Applied ML/hw3_43_rbhagcha_jdpancha_ahande'
    img = 'hw3_1.jpeg'  
    pca = PCA(dir=dir, img_name=img) 
    #pca.actual_image()
    pca.reconstruct()

    """### Accumulative Variance vs Number of Components"""

    pca.plot_variance_by_channels()

"""# References ####
1) https://medium.com/@srv96/principal-component-analysis-from-scratch-932ff97eb27f

2) https://bagheri365.github.io/blog/Principal-Component-Analysis-from-Scratch/

3) https://www.askpython.com/python/examples/principal-component-analysis
"""

