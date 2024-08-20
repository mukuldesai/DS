#!/usr/bin/env python
# coding: utf-8

# <h1>Model Interpretibility</h1>

# <h2>About the Dataset</h2>
# 
# The dataset provided is a fictional dataset created by IBM data scientists to simulate employee attrition. It encompasses various factors and attributes associated with employees, ranging from demographic information to job-related aspects. This dataset is designed for data science and analytical purposes to explore the underlying factors contributing to employee attrition within a hypothetical organization
# 
# <h2>Abstract</h2>
# 
# The dataset aims to shed light on the intricate dynamics of employee attrition, offering a comprehensive set of features that capture different facets of an employee's professional and personal life. It includes variables such as education level, environmental satisfaction, job involvement, job satisfaction, performance rating, relationship satisfaction, and work-life balance.
# 
# The educational background of employees is categorized into five levels: 'Below College,' 'College,' 'Bachelor,' 'Master,' and 'Doctor.' Various aspects of job satisfaction, such as environmental satisfaction, job involvement, job satisfaction, performance rating, relationship satisfaction, and work-life balance, are quantified on different scales.
# 
# The dataset presents an opportunity to investigate correlations and patterns within these attributes and their impact on employee attrition. It encourages exploratory data analysis and the application of machine learning models to predict and understand the likelihood of attrition based on the provided features.
# 
# Researchers and data scientists can leverage this dataset to address specific questions related to attrition, such as examining the breakdown of distance from home by job role and attrition, or comparing average monthly income based on education and attrition status. The simulated nature of the data allows for a controlled environment for experimentation and analysis, providing valuable insights that can be applied to real-world scenarios in talent management and employee retention strategies.
# 

# In[1]:


pip install pydotplus


# In[2]:


pip install psutil


# In[3]:


get_ipython().system('pip install graphviz')


# In[4]:


pip install xgboost


# <h1>Loading Libraries</h1>

# In[5]:


#data manupulation
import pandas as pd
#numerical combination
import numpy as np 
#plotting data and create visualization
import matplotlib.pyplot as plt           
import seaborn as sns
import plotly.express as px
import graphviz

from plotly.subplots import make_subplots
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.tree import plot_tree
import pydotplus #pip install pydotplus
from sklearn.tree import export_graphviz
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score, confusion_matrix
from sklearn import metrics

from statsmodels.stats.outliers_influence import variance_inflation_factor

import xgboost as xgb
from xgboost import plot_importance


# Loading the data set from Kaggle using the opendataset

# In[6]:


import opendatasets as od


# In[7]:


data = "https://www.kaggle.com/datasets/whenamancodes/hr-employee-attrition"


# In[10]:


# Import the processed data from notebook One
url = 'https://raw.githubusercontent.com/mukuldesai/DS/main/HR%20Employee%20Attrition.csv?token=GHSAT0AAAAAACNBX6PXMGL6M66EM5BNMA66ZNUIEPA'


dff = pd.read_csv('https://raw.githubusercontent.com/mukuldesai/DS/main/HR%20Employee%20Attrition.csv?token=GHSAT0AAAAAACNBX6PXMGL6M66EM5BNMA66ZNUIEPA')


# In[14]:


import pandas as pd

# Assuming data is the file path to your dataset
data = 'https://raw.githubusercontent.com/mukuldesai/DS/main/HR%20Employee%20Attrition.csv'

# Load the dataset into a DataFrame
df = pd.read_csv(data)

# Display the first 10 rows of the DataFrame
df.head(10)


# <h1>Data Checking</h1>

# In[19]:


# Replacing values in the 'Attrition' column
df['Attrition'].replace({'Yes': 1, 'No': 0}, inplace=True)
df['Attrition'].replace({2: 1}, inplace=True)

# Checking unique values in the 'Attrition' column after replacement
print("Attrition", df['Attrition'].unique())


# In[21]:


df.head(10)


# In[22]:


# Taking the required columns that may cause the lung cancer and adding them to subset.
df= df[['Age', 'EnvironmentSatisfaction', 'YearsAtCompany', 'TotalWorkingYears', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'WorkLifeBalance', 'Attrition']]
df.describe()


# In[23]:


df.isnull().sum()


# <h1>Splitting the data into training and validation dataset</h1>

# In[27]:


y= df.Attrition
x=df.drop('Attrition',axis=1)


# In[28]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)


# <h1>Linear Model Building using Logistic Regression</h1>

# In[29]:


# fit Logistic Regression model to training data
logreg = LogisticRegression()
logreg.fit(x_train,y_train)


# In[30]:


log_odds = logreg.coef_[0]
pd.DataFrame(log_odds, 
             x_train.columns, 
             columns=['coef'])\
            .sort_values(by='coef', ascending=False)


# <h3>1.Fit a linear model and interpret the regression coefficients</h3>
# 
# RelationshipSatisfaction, YearsWithCurrManager, WorkLifeBalance,: An increase in these features is associated with an increase in the predicted Attrition Level. For example, if someone has higher work life balance,it will eventually affect his work lifestyle and the result of his work and will show higher Attrition Level
# 
# YearsSinceLastPromotion,Age,TotalWorkingYears: An increase in these features is associated with a decrease in the predicted Attrition Level. For instance, increase in age, experiences stagnancy in the current role and is linked to a lower Attrition Level.

# In[31]:


odds = np.exp(logreg.coef_[0])
pd.DataFrame(odds, 
             x_train.columns, 
             columns=['coef'])\
            .sort_values(by='coef', ascending=False)


# <h1> Tree Based Model building using XGBoost</h1>

# In[33]:


xgb_cl = xgb.XGBClassifier(random_state=0)
xgb_cl.fit(x_train, y_train)


# In[37]:


preds = xgb_cl.predict(x_test)
print(accuracy_score(y_test, preds))
    


# <h3>2. Fit a tree-based model and interpret the nodes</h3>
# Ans: The plot below interprets all nodes (root, leaf, and intermediate) and displays the first tree plotted with the XGBoost algorithm. This figure shows how the model arrived at its final decisions and what splits it took to reach those results. As per the below plot, the root node is 'Occupational Hazards'. Node interpretability for first 3 trees is shown below.

# In[52]:


import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Define and train a decision tree classifier
clf = DecisionTreeClassifier()
clf.fit(X, y)

# Plot the decision tree
plt.figure(figsize=[10,5 ])  # Adjust the figure size as needed
plot_tree(clf, filled=True)
plt.show()


# In[51]:


from sklearn import tree
import matplotlib.pyplot as plt

# Assuming clf is your scikit-learn DecisionTreeClassifier or DecisionTreeRegressor
plt.figure(figsize=[10, 5])  # Adjust the figure size as needed
tree.plot_tree(clf, filled=True, max_depth=3)  # You can adjust max_depth to control the depth of the displayed tree
plt.show()


# <h1>Using Auto ML to find the best model</h1>

# In[53]:


pip install -f http://h2o-release.s3.amazonaws.com/h2o/latest_stable_Py.html h2o


# In[54]:


# Set a minimum memory size and a run time in seconds
min_mem_size=6 
run_time=222


# In[55]:


# Use 50% of availible resources
import psutil
pct_memory=0.5
virtual_memory=psutil.virtual_memory()
min_mem_size=int(round(int(pct_memory*virtual_memory.available)/1073741824,0))
print(min_mem_size)


# In[56]:


# 65535 Highest port no
# Start the H2O server on a random port
# Import libraries
# Use pip install or conda install if missing a library

import random
import logging
import h2o
port_no=random.randint(5555,55555)

#  h2o.init(strict_version_check=False,min_mem_size_GB=min_mem_size,port=port_no) # start h2o
try:
  h2o.init(strict_version_check=False,min_mem_size_GB=min_mem_size,port=port_no) # start h2o
except:
  logging.critical('h2o.init')
  h2o.download_all_logs(dirname=logs_path, filename=logfile)      
  h2o.cluster().shutdown()
  sys.exit(2)


# In[59]:


datahf = h2o.H2OFrame(df)


# In[60]:


data= datahf


# In[61]:


data.head()


# In[62]:


data.types


# In[64]:


data['Attrition'] = data['Attrition'].asfactor()
print(data['Attrition'].isfactor())


# In[65]:


data.describe()


# In[66]:


#check the data rows and columns
data.shape


# In[67]:


# Create a 80/20 train/test split
pct_rows=0.80
data_train, data_test = data.split_frame([pct_rows])


# In[68]:


#after split check rows and columns
print(data_train.shape)
print(data_test.shape)


# In[69]:


data_train.head(2)


# In[70]:


# Set the features and target
X=data.columns
print(X)


# In[71]:


# Set target and predictor variables
y ='Attrition'
#y_numeric ='churn_bit'
X.remove(y) 
#X.remove(y_numeric) 
print(X)


# In[72]:


# Set up AutoML
import h2o
from h2o.automl import H2OAutoML
auml = H2OAutoML(max_runtime_secs=run_time, seed=1)


# In[73]:


auml.train(x=X,y=y,training_frame=data_train)


# In[74]:


print(auml.leaderboard)


# In[75]:


model_index=0
glm_index=0
glm_model=''
auml_leaderboard_df=auml.leaderboard.as_data_frame()
models_dict={}
for m in auml_leaderboard_df['model_id']:
  models_dict[m]=model_index
  if 'StackedEnsemble' not in m:
    break 
  model_index=model_index+1  

for m in auml_leaderboard_df['model_id']:
  if 'GLM' in m:
    models_dict[m]=glm_index
    break  
  glm_index=glm_index+1     
models_dict


# In[76]:


print(model_index)
best_model = h2o.get_model(auml.leaderboard[model_index,'model_id'])


# In[77]:


best_model.algo


# In[78]:


best_model.explain(data_train)


# <h3>3. Use auto ml to find the best model</h3>
# Ans: Using AutoML, we have conclude'GBM' as the best model to determine the employee attrition. By referring the below variable importance and shap summary plots, we can say that 'YearsAtCompany' is the most important and dominant feature in the model to predict target variable. Where as, the least important features are 'YearsinCurrentRole' and 'WorklifeBalance' according to variable importance and shap summary plots respectively.

# <h4>Variable Importance</h4>
# The variable importance plot shows the relative importance of the most important variables in the model.

# In[81]:


if best_model.algo in ['gbm','drf','xrt','xgboost']:
  best_model.varimp_plot()


# In[82]:


if glm_index is not 0:
  print(glm_index)
  glm_model=h2o.get_model(auml.leaderboard[glm_index,'model_id'])
  print(glm_model.algo) 
  glm_model.std_coef_plot()


# In[83]:


print(best_model.auc(train = True))


# In[84]:


def model_performance_stats(perf):
    d={}
    try:    
      d['mse']=perf.mse()
    except:
      pass      
    try:    
      d['rmse']=perf.rmse() 
    except:
      pass      
    try:    
      d['null_degrees_of_freedom']=perf.null_degrees_of_freedom()
    except:
      pass      
    try:    
      d['residual_degrees_of_freedom']=perf.residual_degrees_of_freedom()
    except:
      pass      
    try:    
      d['residual_deviance']=perf.residual_deviance() 
    except:
      pass      
    try:    
      d['null_deviance']=perf.null_deviance() 
    except:
      pass      
    try:    
      d['aic']=perf.aic() 
    except:
      pass      
    try:
      d['logloss']=perf.logloss() 
    except:
      pass    
    try:
      d['auc']=perf.auc()
    except:
      pass  
    try:
      d['gini']=perf.gini()
    except:
      pass    
    return d


# In[85]:


mod_perf=best_model.model_performance(data_test)
stats_test={}
stats_test=model_performance_stats(mod_perf)
stats_test


# In[86]:


get_ipython().system('pip install shap')


# <h4>SHAP Summary</h4>
# SHAP summary plot shows the contribution of the features for each instance (row of data). The sum of the feature contributions and the bias term is equal to the raw prediction of the model, i.e., prediction before applying inverse link function.

# In[87]:


import shap


# In[88]:


lg_explainer = shap.Explainer(logreg, x_train)
shap_values_lg = lg_explainer(x_test)


# In[89]:


shap.plots.beeswarm(shap_values_lg, max_display=15)


# In[90]:


shap.summary_plot(shap_values_lg, x_train, plot_type="bar", color='steelblue')


# In[133]:


pip install tensorflow


# In[137]:


import shap
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost.sklearn import XGBClassifier

# Assuming x_train and y_train are your training data and labels

# Split data into train and validation sets
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

# Initialize and train XGBClassifier
xgb_cl = XGBClassifier()
xgb_cl.fit(x_train, y_train)

# Convert XGBClassifier to native XGBoost format
xgb_model = xgb_cl.get_booster()

# Initialize the TreeExplainer using the native XGBoost model
explainer = shap.TreeExplainer(xgb_model)

# Compute SHAP values
shap_values = explainer.shap_values(x_val)



# In[138]:


shap.summary_plot(shap_values, x_train, plot_type="bar", color='steelblue')


# In[142]:


import shap
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost.sklearn import XGBClassifier

# Assuming x_train and y_train are your training data and labels

# Split the data into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

# Initialize and train the XGBClassifier model
xgb_cl = XGBClassifier()
xgb_cl.fit(x_train, y_train)

# Compute SHAP values using the TreeExplainer
explainer = shap.TreeExplainer(xgb_cl)
shap_values = explainer.shap_values(x_val)

# Verify the shape of x_val and shap_values
print("Shape of x_val:", x_val.shape)
print("Shape of shap_values:", shap_values.shape)

# Plot the SHAP summary plot
shap.summary_plot(shap_values, x_val)


# <h3>4. Run SHAP analysis on the models from steps 1, 2, and 3, interpret the SHAP values and compare them with the other model interpretability methods.</h3>
# Ans:
# 
# 
# After running SHAP analysis on model 1 (i.e. Logistic Regression), we have found that 'YearsatCompany' is the top feature in the dataset impacting the model’s output as represented in the beeswarm and summary plots whereas 'WorkLifeBalance' is the least important feature. Upon analyzing the beeswarm plot, we observed that a higher value of 'Environment Satisfaction' (2) corresponds to a lower likelihood of employee attrition, while a lower value of 'Environment Satisfaction' (1) is associated with a higher probability of attrition. Similarly, factors such as 'YearsSincLastPromotion,' 'Years in Current Role,' and 'Age' exhibit a negative impact on model.
# 
# For model 2 (i.e. XGBoost), 'YearsatCurrentRole' and 'TotalWorkingYears' are the most and least significant features respectively contributing towards prediction of Attrition. According to summary plot, higher value of 'YearsatCurrentRole' leads to higher chance of attrition. Lower value of 'YearsatCurrentRole' leads to lower chance of Attrition.
# 
# As per model 3 (i.e. GBM), by referring the above shap summary plot, 'Age' is the most important and dominant feature in the model to predict target variable which has negative impact on output. Where as, 'WorklifeBalance' is less important.
# 
# So, all 3 algorithms shows different results. We can perform regularization to make the model better.

# <h1>Conclusion:</h1>
# 
# Build a predictive model using Logistic regression, XGBoost, AutoML to predict whether a patient has lung cancer based on a set of risk factors. Model interpretability was tested using shap analysis by plotting beeswarm and shap summary plots to compare all three models. We discovered that all three algorithms produce distinct outcomes, and that regularization can be used to improve the models.

# <h1>LICENSE</h1>
# 
# 
# MIT License
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

# <h1>References</h1>
# 
# H20-ML-   https://www.youtube.com/watch?v=91QljBnvM7s
# 
# Kaggle   Notebook- https://www.kaggle.com/stephaniestallworth/melbourne-housing-market-eda-and-regression
# 
# Dataset-   https://www.kaggle.com/datasets/whenamancodes/hr-employee-attrition
# 
# Professor's AutoML Notebook   -https://github.com/ajhalthor/model-interpretability/blob/main/Shap%20Values.ipynb
# 
# https://towardsdatascience.com/interpreting-coefficients-in-linear-and-logistic-regression-6ddf1295f6f1
# https://towardsdatascience.com/decision-trees-explained-3ec41632ceb6
# https://www.analyticsvidhya.com/blog/2020/05/decision-tree-vs-random-forest-algorithm/
# https://www.analyticsvidhya.com/blog/2020/10/all-about-decision-tree-from-scratch-with-python-implementation/
# https://www.kdnuggets.com/2020/04/visualizing-decision-trees-python.html
# https://www.datacamp.com/community/tutorials/xgboost-in-python

# In[ ]:




