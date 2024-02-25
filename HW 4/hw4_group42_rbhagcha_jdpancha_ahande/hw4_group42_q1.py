#!/usr/bin/env python
# coding: utf-8

# In[43]:


import pandas
import numpy


# In[44]:


data = [[24, 40000, "Yes"], [53, 52000, "No"], [23, 25000, "No"], [25, 77000, "Yes"],
       [32, 48000, "Yes"], [52, 110000, "Yes"], [22, 38000, "Yes"], [43, 44000, "No"], [52, 27000, "No"], [48, 65000, "Yes"]]


# In[45]:


df = pandas.DataFrame(data, columns=["Age", "Salary ($)","College Degree"])
df.dtypes


# In[46]:


import pandas as pd
import numpy as np

def impurity(method, target):
    if method=='entropy':
        # it will only work if the target is in pandas series
        temp = target.value_counts()/target.shape[0] 
        return (numpy.sum(-numpy.log2(temp+1e-9)*temp))

    # else gini index
    else:
        temp = target.value_counts()/target.shape[0] 
        return (1-numpy.sum(pow(temp,2)))

def iG(mfeature, target, method):
    # summing up the mask
    temp1 = sum(mfeature)
    temp2 = mfeature.shape[0] - temp1

    # calculate information gain if temp1 and temp2 are not 0 else information would be 0
    if temp1!=0 and temp2!=0:
        # call the method
        method_val1 = impurity(method, target)
        method_val2 = impurity(method, target[mfeature])
        method_val3 = impurity(method, target[-mfeature])
        val1 = temp1/(temp1+temp2)*method_val2
        val2 = temp2/(temp1+temp2)*method_val3
        val = method_val1 - val1 - val2
        return val
    else:
        return 0



def split_information_gain(feature, target, method='entropy'):
    # sort the feature data and only for unique data to plot in the tree
#     print(feature)
    f = feature.sort_values().unique()[1:]
#     print(f)
    # list to store the information gain and the feature for splitting
    ig_lst = []
    f_lst = []
    # calculate information gain
    for data in f:
        # develop a mask for calculate information gain using gini index or entropy
        mask_feature = data > feature
        information_gain = iG(mask_feature, target, method)
        f_lst.append(data)
        ig_lst.append(information_gain)
#         print(ig_lst)
    if ig_lst:
        # more the information gain better the solution
        return(max(ig_lst), f_lst[ig_lst.index(max(ig_lst))], True, True)
    else:
        x = (None, None, None, False)
        # as there is no ig generated
        return x


# In[47]:


def perform_best_divide(df, target):    
    # getting best split based on the value
    # applying split information gain get the mask value

    mval = df.drop(target, axis=1).apply(split_information_gain, target=df[target])

#     print(mval)
    # check if the mval is not null
    t1 = sum(mval.loc[3,:])
    if t1!=0:
        mtemp = mval.loc[:, mval.loc[3,:]]
        # get information about splitting using information gain
        # the maximize information gain
        maxi_information_gain, maxi_split_val, maxi_flag, mf = mtemp[max(mtemp)]
        return (max(mtemp), maxi_split_val, maxi_information_gain, maxi_flag)
    else:
        return (None, None, None, None)

def divide(df, flag, feature, val):
    if flag==False:
        return (df[df[feature].isin(val)], df[(df[feature].isin(val))==False])
    else:
        return (df[df[feature]< val], df[(df[feature]< val)==False])

def output(df, fact):
    if not fact:
        return df.mean()
    else:
        return df.value_counts().idxmax()


# In[48]:


def fit(df, fact, min_information_gain = 1e-20, c=0):

    maxi, maxi_split_val, maxi_information_gain, maxi_flag = perform_best_divide(df, "College Degree")
#         print(perform_best_divide(df, "College Degree"))
    
    if maxi_information_gain is not None and maxi_information_gain >= min_information_gain:
        # depth of tree
        print("Information Gain at", maxi_split_val, "=", maxi_information_gain)
        c+=1
        # get left and right
        ltree, rtree = divide(df, maxi_flag, maxi, maxi_split_val)
        if maxi_flag==False:
            s_tree = "in"
        else:
            s_tree = "<="
#         print(maxi, maxi_split_val)
        # Build smaller tree
        q = "{} {} {}".format(maxi, s_tree, maxi_split_val)

        # move left and right
#             print(c)
        la = fit(ltree, fact, min_information_gain, c)
        ra = fit(rtree, fact, min_information_gain, c)

        out = {q: []}

        
        if la!=ra:
            out[q].append(la)
            out[q].append(ra)
        else:
            out = la
        
    else:
        res = output(df['College Degree'], fact)
        return res
    return out, c


# In[49]:


tree, depth = fit(df, True)
print("Depth:", depth)
print("Tree Structure:", tree)


# Reference: https://anderfernandez.com/en/blog/code-decision-tree-python-from-scratch/

# # Part 1.2 

# In[62]:


#Importing all the libraries
import pandas
import numpy
# from sklearn.model_selection import train_test_split


# In[63]:


#We will put Yes = 1 for having a college degree and No = 0 for not having a college degree
table = [[24, 40000, 1], [53, 52000, 0], [23, 25000, 0], [25, 77000, 1], [32, 48000, 1], [52, 110000, 1],
          [22, 38000, 1], [43, 44000, 0], [52, 27000, 0], [48, 65000, 1]]


# In[64]:


#Creating DataFrame

dataTable = pandas.DataFrame(table, columns=['Age', 'Salary ($)', 'College Degree'])
print(dataTable)
print(dataTable.dtypes)


# #### For building the multivariate tree, we have applied the concept of Perceptron Algorithm function which is $Y=f(∑wi*xi + b)$, for our case we will have two inputs which is x_age and x_salary
# 
# #### As we are having two inputs the resulting hyperplane will be a linear separation i.e. a line

# In[65]:



costVal = 0 #Calculating error 
signed_val = -1

#Calculating the Mutual Information

def compute_mutual_information(w1, x1, w2, x2, param):
    #print(n0,p0,n1,p1)
    mutual_information = 0
    if ( (w1+w2) * (w1+x1) != 0 and (w1*param)/((w1+w2) * (w1+x1)) != 0 ):
        mutual_information = mutual_information + numpy.log2((w1*param)/((w1+w2) * (w1+x1))) * (w1/param) 
        
    if ( (w1+x1)*(x1+x2) != 0 and (x1*param)/((x1+x2) * (w1+x1)) != 0 ):
        mutual_information = mutual_information + numpy.log2((x1*param)/((x1+x2) * (w1+x1))) * (x1/param) 
        
    if ( (w1+w2) * (w2+x2) != 0 and (w2*param)/((w1+w2) * (w2+x2)) != 0 ):
        mutual_information = mutual_information + numpy.log2((w2*param)/((w1+w2) * (w2+x2))) * (w2/param) 
        
    if ( (x1+x2) * (w2+x2) != 0 and (x2*param)/((x1+x2) * (w2+x2)) != 0 ):
        mutual_information = mutual_information + numpy.log2((x2*param)/((w2+x2) * (x1+x2))) * (x2/param) 
        
    return(mutual_information * signed_val )


# In[66]:


# Function for calculating the impurity getting the values of alpha and beta at each point
a = 1
b = 1
list_a = []
list_b = []

def compute_impurity(costVal):
    
    p_lth_no_cd = 0 # total folks having age lesser than THRESHOLD and NO College Degreee
    p_lth_cd = 0 # total folks having age lesser than THRESHOLD and having a College Degreee
    p_hth_no_cd = 0 # total folks having age more than THRESHOLD and NO College Degreee
    p_hth_cd = 0 # total folks having age more than THRESHOLD and having a College Degreee
    
    impurity = [] #Impurity list to calculate the defects in the prediction on each interation
    
    for b in range(0, 100, 1):
        b = b/100
        
        for a in range(0,100,1):
            imp = 0
            a = a/100
            for j in range(10):
                #Implementing the given condition (eventually perceptron logic)
                if (numpy.sign( a * dataTable['Age'][j] + b * dataTable['Salary ($)'][j] -1 ) == 1):
                    if (dataTable['College Degree'][j] == 0):
                        p_hth_no_cd = p_hth_no_cd + 1
                    else:
                        p_hth_cd = p_hth_cd + 1
                
                elif (numpy.sign( a * dataTable['Age'][j] + b * dataTable['Salary ($)'][j] -1 ) == -1):
                    if (dataTable['College Degree'][j] == 0):
                        p_lth_no_cd = p_lth_no_cd + 1
                    else:
                        p_lth_cd = p_lth_cd + 1
                
                else:
                    print("ERROR")                     
            
            #Calculating the error for threshold value
            if costVal == 1:
                impurity.append([a,b,compute_MI(p_lth_no_cd,p_lth_cd,p_hth_no_cd,p_hth_cd,10)])
                
            elif costVal == 0:
                if ( p_lth_cd < p_lth_no_cd):
                    imp = imp + p_lth_cd
                else:
                    imp = imp + p_lth_no_cd
                if ( p_hth_cd < p_hth_no_cd):
                    imp = imp + p_hth_cd
                else:
                    imp = imp + p_hth_no_cd 
                impurity.append([a,b,imp])

            else:               
                break
                
            # Setting the values to 0 again for next iteration
            p_lth_no_cd = 0
            p_lth_cd = 0
            p_hth_no_cd = 0 
            p_hth_cd = 0            
            


    return impurity


# In[67]:


impurity = compute_impurity(costVal)

list_MI = [-1,-1,10000]
for k in range(len(impurity)):
    if ( impurity[k][2] < list_MI[2] ):
        list_MI[0] = impurity[k][0]
        list_MI[1] = impurity[k][1]
        list_MI[2] = impurity[k][2]
        

print(impurity[0])
print(impurity[-1])
print("\n")
print(impurity)
print (len(impurity))


# ##### Final Multivariate Decison Tree

# <img src="DecisionTreeMultivariate.jpeg" width="500" height="500">

# # Part 1.3

# Multivariate decision trees have several advantages compared to univariate decision trees, including:
# 
# Multivariate decision trees are able to capture more complex relationships between different variables. This allows them to better model real-world data and make more accurate predictions.
# 
# Multivariate decision trees can handle missing or incomplete data more gracefully than univariate decision trees. This is because they are able to use information from other variables to make predictions, even if some data is missing.
# 
# Multivariate decision trees are more interpretable than univariate decision trees. This is because they use multiple variables to make predictions, so the relationships between the variables are more transparent and easier to understand.
# 
# Multivariate DT can be of lesser depth as compared to univariate DT, and it is not restricted to only orthogonal boundaries.
# 
# However, there are also some disadvantages to using multivariate decision trees, including:
# 
# Multivariate decision trees can be more difficult to train and tune than univariate decision trees. This is because they have more parameters and require more data to model the complex relationships between the variables. Also, because it use regression techniques in further splitting, which is expensive computationally. 
# 
# Multivariate decision trees can be more sensitive to noise and outliers in the data. This is because they use multiple variables to make predictions, so a small amount of noise or outlier data can have a larger impact on the model's performance.
# 
# 

# ##### Pros and Cons Specific to the given problem

# The given data can be considered as 2-dimension (as we have two features i.e. age and salary)and hence the algorithm works efficiently. But there can be some challenges when the data is uneven, unseparable and not clustered. So, for such data it is best to use univariate trees. 

# ##### Reference:
# 1) https://courses.cs.washington.edu/courses/cse446/13sp/slides/decision-trees-boosting-annotated.pdf
# 
# 2) https://github.com/lnies?tab=repositories
# 
# 3) https://machinelearningmastery.com/implement-perceptron-algorithm-scratch-python/
# 
# 4) https://machinelearningmastery.com/implement-decision-tree-algorithm-scratch-python/
# 
# 5) https://www.kdnuggets.com/2020/01/decision-tree-algorithm-explained.html
# 
# 6) https://machinelearningmastery.com/information-gain-and-mutual-information/

# In[ ]:




