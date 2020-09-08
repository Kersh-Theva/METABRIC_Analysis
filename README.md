# METABRIC Analysis: Project Overview
Used data visualization, a 3-layer neural network and other classification algorithms to predict breast cancer survival using a METABRIC dataset from Kaggle ([Link](https://www.kaggle.com/kershtheva/starter-breast-cancer-gene-expression-3925d3a5-5)).

Full details on the code can be viewed on my public Kaggle notebook (please click [link](https://www.kaggle.com/kershtheva/starter-breast-cancer-gene-expression-3925d3a5-5)). 

## Resources I used 
**Python Version**: 3.8.3 <br>
**Kaggle Database**: [Kaggle](https://www.kaggle.com/raghadalharbi/breast-cancer-gene-expression-profiles-metabric) <br>
**ML Packages**: [Scikit-Learn](https://scikit-learn.org/stable/), [Keras](https://keras.io/) <br>
**Literature**: [Medium](https://towardsdatascience.com/a-beginners-guide-to-xgboost-87f5d4c30ed7), [Keras](https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/)

## Data Cleaning

The following columns were dropped for the model building because they were either patient identifiers or was not expected to be useful in this first iteration: 
- Patient ID
- Cancer type
- Cancer_type_detailed
- Cohort 

The following columns were converted into binary columns with 1 or 0: 
- ER status (measured by IHC) 
- Type of breast surgery 
- HER2 status
- Inferred menopausal states
- Primary tumor laterality
- PR status
- ER status

The following columns were converted into dummy columns 
- Cellularity
- PAM50 + Claudin-low subtype
- Neoplasm histological grade
- Tumor (other histologic subtype) 
- Integrative cluster
- Oncotree code
- HER2 status (measured by snp6)

Undefined and null values were removed. 

## EDA

<p align="center">
<img src="https://github.com/Kersh-Theva/METABRIC_Analysis/blob/master/DeathCorrelatedFactors_Top30.jpg" alt="Test1" width="350" height="600"> 
</p> <br>

**Figure 1. Features that have the strongest correlation to death by breast cancer.** A list of the features with the strongest correlation to a survival value of "0" in the METABRIC breast cancer dataset. Factor names are listed on the left and the Pearson correlation coefficient to death is listed on the right. Highest correlations are with patient ID, overall survival (in months), age at diagnosis, tumor size and mutation count.

<p align="center">
<img src="https://github.com/Kersh-Theva/METABRIC_Analysis/blob/master/SurvivalvsLymphNodes.png" alt="Test1" width="900" height="300"> 
</p> <br>

**Figure 2. Overall survival compared to the number of positive lymph nodes.** Boxplot comparing the overall survival (in months) to the number of lymph nodes containing cancer cells. After 10 positively examined lymph nodes, there are almost no cases of surviving cancer, with the exception of some patients with 15 positively examined lymph nodes. Those who died almost always had less than 100 months to live from the start of the study to their death. In comparison, patients who survived would survive for approximately 50 months longer, on average. 

<p align="center">
<img src="https://github.com/Kersh-Theva/METABRIC_Analysis/blob/master/TopUniqueMutations.png" alt="Test1" width="300" height="450"> 
</p> <br>

**Figure 3. Top 10 genes with the largest number of unique mutations.** A count of the number of unique mutations for each of the 10 genes with the most unique mutations. P53, a well-known oncogene has 212 unique mutations (e.g. deletions, substitutions, insertions) in the METABRIC dataset. <br>

<p align="center">
<img src="https://github.com/Kersh-Theva/METABRIC_Analysis/blob/master/Treatment%20Correlations.png" alt="Test1" width="675" height="450"> 
</p> <br>

**Figure 4. Heatmap showing cooperative use of each treatment modality.** Heatmap showing the Pearson correlation for the cooperative use of treatments. The two treatments that are most commonly used together are chemotherapy and radiotherapy (Pearson correlation = 0.25). The two treatments that are least correlated with each other are hormone therapy and chemotherapy. 

<p align="center">
<img src="https://github.com/Kersh-Theva/METABRIC_Analysis/blob/master/Treatment%20Survival%20%25.png" alt="Test1" width="675" height="300"> 
</p> <br>

**Figure 5. % survival for each treatment modality.** Percent survival for patients treated with each of the four major treatment modalities. Patients treated with chemotherapy had the lowest % survival. Conversely, patients treated with the other three modalities had similar survival %, and all three had survival greater than 50%. 

<p align="center">
<img src="https://github.com/Kersh-Theva/METABRIC_Analysis/blob/master/Treatment%20Effectiveness.png" alt="Test1" width="750" height="300"> 
</p> <br>

**Figure 6. Comparison of the mean survival (in months) and the number of treatments for each treatment combination.** Bar plots showing the average survival and the number of treatments for each of the four major treatment modalities and their combination. Treatment methods are labelled 1-16 and the corresponding combination is described in the table on the right. The treatments with the best average survival with the lowest standard deviation are treatments 9 and 10 (Breast surgery and breast surgery + radio therapy, respectively). The three most common treatments are treatments 4, 12, and 11 (hormone therapy + radio therapy, breast surgery + hormone therapy + radio therapy, breast surgery + hormone therapy, respectively). 

<p align="center">
<img src="https://www.kaggleusercontent.com/kf/42072527/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..3X7NSShQ7OPCk8fSt9UpTg.AJrYgk4PV2_-HU8KE9E3ci4HErFE8n-LZrOxqkz9zbiXRSShp8v0v4J27s0CRA6xbLopf4Q-2TcoMuQNxYfTmbR1XJRKWBXbleS2qjVLP7GWdY2yI56cj1KNv5XA3yPqWJ6nE66vkEcJC1xuGhPmkYUArP81PHumvicrj-_b9-hrGaXQ987PBhzlpHGwaTmD4kwUWjA_T08dgS530UPjA5zmEQ1trbNqf5wnwzgOANomvbZ3JjK4WIB47R3xoHdOW4JibIuQFnx4sXpKDK2oud4-qf5NyCN13LUCwvtV4zEgysUyyiOVQEW6LO-YXPCJNgg-c21QaAa2jsZIgiRX_hYv35EG55cX8Ly9lI8eILiG4AjluYNz7O_UYcXSBXUn2iM9cZh4yPSkNbtxp57Cay3AvCHSsMVSzBILllXWRWmTuvjgm8zkS27Ffw4nJ2_dXvfBSyE6ctwAyp5Dw4Xtw67zm6vKg703at1ns5nU7hTFHNV9BktIAe4uitDwGHmqm7TyQARXBn6zwLpVpSfnU4OU-TB9LMxRWOAr2Qu9-ppNYnx_NvghxhSh0EJw5c3VGVsCKqRjUUDco-1XBGOty-be68Oo8nkHwePjOW22twVsqs0oCnK5HBkbgSNAD2npEOyrRlm_peRIO_57_9zWETxb9LoooSnUGxIzPJJWsbbcSl06cmWk6Ocud2ByO92X.0nZGpxn5ZA2KA5mcVW9K-g/__results___files/__results___58_1.png" alt="Test1" width="700" height="600"> 
</p> <br> 

**Figure 7. mRNA expression heatmap compared to key cancer biomarkers and survival metrics.** Comparing the mRNA expression levels measured in z scores, correlated with the key cancer biomarkers (PR status, ER status, HER2 status) using the Pearson correlation coefficient. Heatmap colors represent strength and direction of correlation (red = positive correlation, blue = negative correlation). Age at diagnosis and overall survival are correlated with the mRNA expression levels as well. 


<ins> Overall survival (in months) </ins>

Died of cancer: 101.69 &plusmn; 74.9 <br>
Did not die of cancer by end of study: 156 &plusmn; 77.9

## Model Building

998 samples were used for training and testing. 80% of the dataset was used for training, and 20% for testing. 

The following table describes the models that were used to predict cancer survival and their accuracy score on the test set: 

|                | Random Forest | Logistic Regression | Gradient Boosted Trees | Ada Boosted Trees | XGBoost | 3-Layer NN |
|----------------|:-------------:|:-------------------:|:----------------------:|:-----------------:|:-------:|:----------:|
| Accuracy Score |      61%      |         59%         |           67%          |       61.5%       |   65%   |     61%    |

## Strategies for Improvement

- Improving ML models with more data 
- Comparing our models with human predictions to get a measure of model bias
- Improve variance measures for neural network using regularization methods (e.g. L2) 
- Using normalization to transform the training data
