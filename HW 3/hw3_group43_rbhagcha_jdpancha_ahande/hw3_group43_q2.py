# -*- coding: utf-8 -*-
"""Assignment.ipynb
#### Here we are going to implement multinomial naive bayes model as the chosen probabilistic model after studying the dataset as it is more suitable to the text classification model and multiple labels.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from category_encoders import OrdinalEncoder
from sklearn.metrics import accuracy_score

"""## Read text file

#### Read US Name
"""

# File is read using pandas read_csv file where header is not taken into consideration
english = pd.read_csv('us.txt', header=None)
# Column name is set to be Name
english.columns = ["Name"]
# A new column is set name nationality to store the name of country where name is used
english['Nationality'] = 'American'
english.head()

"""#### Read Arabic Names"""

# File is read using pandas read_csv file where header is not taken into consideration
arabic = pd.read_csv('arabic.txt', header=None)
# Column name is set to be Name
arabic.columns = ["Name"]
# A new column is set name nationality to store the name of country where name is used
arabic['Nationality'] = 'Arabic'
arabic.head()

"""#### Read Greek Names"""

# File is read using pandas read_csv file where header is not taken into consideration
greek = pd.read_csv('greek.txt', header=None)
# Column name is set to be Name
greek.columns = ["Name"]
# A new column is set name nationality to store the name of country where name is used
greek['Nationality'] = 'Greek'
greek.head()

"""#### Read Japanese Names"""

# File is read using pandas read_csv file where header is not taken into consideration
japanese = pd.read_csv('japan.txt', header=None)
# Column name is set to be Name
japanese.columns = ["Name"]
# A new column is set name nationality to store the name of country where name is used
japanese['Nationality'] = 'Japanese'
japanese.head()

"""## Combining all four name dataframe into one dataframe"""

# concat is done using pandas inbuilt function concat which concat all the dataframe 
# object into one where index from each dataframe is ignore to include in the 
# concated df
df = pd.concat([english, arabic, greek, japanese], axis=0, ignore_index=True)
print(df.shape)
df.head()

"""## Vectorizer

#### A count vectorizer with ordinal encoding is used with fit and transform for transforming text into vector
"""

# library
from sklearn.feature_extraction.text import CountVectorizer

# Calling object
count_vectorizer_object = CountVectorizer()
# training
count_vectorizer_object.fit(df['Name'])

# transform into vector
vectorized_matrix = count_vectorizer_object.transform(df['Name']).toarray()
# vectorized_matrix

"""#### Ordinal encoder to encode ordinal data/target values"""

# defining a mapper to map the target or Nationality using ordinal encoder
mapper = [{'col': 'Nationality', 'mapping': {'American': int(0), 'Arabic': int(1), 'Greek': int(2), 'Japanese': int(3)}}]

# using user defined mapper
mapper_fun = OrdinalEncoder(mapping = mapper) 

# tranform the target variable
df = mapper_fun.fit_transform(df)

"""## Train test split with 70% training data size"""

train_x, test_x, train_y, test_y = train_test_split(vectorized_matrix, df['Nationality'], test_size = 0.3, shuffle=True)

"""## Multinomial Naive Bayes Class"""

# Here we are defining a Multinomial Naive Bayes Class which fit the dataset and train the model and later used that
# training for prediction.
import numpy
class multinomial_naive_bayes_model:
    def int(self, smooth_para = 10):
        # here smooth_parameter represents the technique to overcome the issue of zero number of variables present in
        # the features, here is smoothing parameter is increase the likelihood probability moves towards uniform
        # distribution, but as mentioned in the lecture we are preferring smoothing parameter = 1 as a base case.
        # Referred: Class Notes
        self.smooth_para = smooth_para

    def train(self, features, target):
        # get features samples and features
        self.features_samples, self.features_num = features.shape

        # assigning target to class variable
        self.target = target

        # get features
        self.features = features
        
        # get unique classes
        self.target_classes = numpy.unique(target)

        # get number of target class length
        self.class_len = self.target_classes.shape[0]

        # calculating prior probability by calculating each unique class in target
        # get prior probability
        self.prior_prob = numpy.zeros((self.class_len))
        temp, self.uni_tar = numpy.unique(self.target, return_counts=True)
        for idx in range(self.target_classes.shape[0]):
            self.prior_prob[idx] = self.uni_tar[idx]/self.features_samples
        
        # initialize two variables with number of zeroes
        self.t1 = numpy.zeros((self.class_len))
        self.t2 = numpy.zeros((self.class_len, self.features_num))

        # iterate over classes target:
        for class_idx in self.target_classes:
            temp2 = numpy.argwhere(self.target.to_numpy()==class_idx).flatten()
            add_col = []
            for idx2 in range(self.features_num):
                add_col.append(numpy.sum(self.features[temp2,idx2]))

            # initializing with 1d array and 2d array
            self.t1[class_idx] = numpy.sum(add_col)
            self.t2[class_idx] = add_col

    # finding prob of a given b
    def prob_ab(self, feat, idx, idx2):
        count_target_idx = self.t2[idx2, idx]
        count_target = self.t1[idx2]

        cal = (count_target_idx+self.smooth_para)/(count_target+ (self.smooth_para*self.features_num))
        cal = numpy.power(cal, feat)
        return cal

    def predict(self, test_feature, smooth_para = 10):
        self.smooth_para = smooth_para
        # get sample and features
        self.test_samples, self.test_features = test_feature.shape

        # get predicted probability
        self.predicted_probability = numpy.zeros((self.test_samples, self.class_len))

        # iterate over for loop
        test_f0 = test_feature.shape[0]
        for idx in range(test_f0):
            jp = numpy.zeros((self.class_len))

            # get likelihood
            for idx2 in range(self.class_len):
                # likelihood
                lst = []
                # print(self.test_feature)
                for j in range(test_feature[idx].shape[0]):
                    lst.append(self.prob_ab(test_feature[idx][j], j, idx2))

                likelihood = numpy.prod(lst)

                # joint likelihood which gives prob of a given b * prob(target)
                jp[idx2] = likelihood * self.prior_prob[idx2]

            # naive bayes numerator and denominator
            # get denominator
            cal1 = numpy.sum(jp)

            # get numerator and return sum
            for idx3 in range(self.class_len):
                cal0 = jp[idx3]
                self.predicted_probability[idx, idx3] = (cal0/cal1)


        # return max value result
        idx_max = numpy.argmax(self.predicted_probability, axis=1)
        print(idx_max)
        return self.target_classes[idx_max]

"""## Train Test model"""

# call class
multinomial_nb_model = multinomial_naive_bayes_model()

# train model with training dataset
multinomial_nb_model.train(train_x, train_y)

# predict after training 
prediction = multinomial_nb_model.predict(test_x)

# get accuracy
print('Accuracy: %.4f '% accuracy_score(prediction, test_y))

"""For smooth parameter selected I received more accuracy while selecting smooth parameter = 10. Compare it with smooth parameter 1, 5 and 10.

###### References

1. https://towardsdatascience.com/laplace-smoothing-in-na%C3%AFve-bayes-algorithm-9c237a8bdece

2. https://machinelearningmastery.com/naive-bayes-classifier-scratch-python/

3. https://www.kaggle.com/code/riyadhrazzaq/multinomial-naive-bayes-from-scratch/notebook

4. https://towardsdatascience.com/name-classification-with-naive-bayes-7c5e1415788a
"""