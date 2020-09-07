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
**Figure 1.**

<p align="center">
<img src="https://github.com/Kersh-Theva/METABRIC_Analysis/blob/master/SurvivalvsLymphNodes.png" alt="Test1" width="900" height="300"> 
</p> <br>
**Figure 2.**

<p align="left">
<img src="https://github.com/Kersh-Theva/METABRIC_Analysis/blob/master/TopUniqueMutations.png" alt="Test1" width="900" height="300"> 
</p> <br>
**Figure 3.**

<p align="center">
<img src="https://github.com/Kersh-Theva/METABRIC_Analysis/blob/master/Treatment%20Correlations.png" alt="Test1" width="900" height="300"> 
</p> <br>

<p align="center">
<img src="https://github.com/Kersh-Theva/METABRIC_Analysis/blob/master/Treatment%20Survival%20%25.png" alt="Test1" width="900" height="300"> 
</p> <br>
**Figure 4.**

<p align="center">
<img src="https://github.com/Kersh-Theva/METABRIC_Analysis/blob/master/Treatment%20Effectiveness.png" alt="Test1" width="900" height="300"> 
</p> <br>
**Figure 5.**

<p align="center">
<img src="https://github.com/Kersh-Theva/METABRIC_Analysis/blob/master/MutationHeatmap.png" alt="Test1" width="900" height="300"> 
</p> <br>
**Figure 6.**


<p align="center">
<img src="https://www.kaggleusercontent.com/kf/42072527/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..3X7NSShQ7OPCk8fSt9UpTg.AJrYgk4PV2_-HU8KE9E3ci4HErFE8n-LZrOxqkz9zbiXRSShp8v0v4J27s0CRA6xbLopf4Q-2TcoMuQNxYfTmbR1XJRKWBXbleS2qjVLP7GWdY2yI56cj1KNv5XA3yPqWJ6nE66vkEcJC1xuGhPmkYUArP81PHumvicrj-_b9-hrGaXQ987PBhzlpHGwaTmD4kwUWjA_T08dgS530UPjA5zmEQ1trbNqf5wnwzgOANomvbZ3JjK4WIB47R3xoHdOW4JibIuQFnx4sXpKDK2oud4-qf5NyCN13LUCwvtV4zEgysUyyiOVQEW6LO-YXPCJNgg-c21QaAa2jsZIgiRX_hYv35EG55cX8Ly9lI8eILiG4AjluYNz7O_UYcXSBXUn2iM9cZh4yPSkNbtxp57Cay3AvCHSsMVSzBILllXWRWmTuvjgm8zkS27Ffw4nJ2_dXvfBSyE6ctwAyp5Dw4Xtw67zm6vKg703at1ns5nU7hTFHNV9BktIAe4uitDwGHmqm7TyQARXBn6zwLpVpSfnU4OU-TB9LMxRWOAr2Qu9-ppNYnx_NvghxhSh0EJw5c3VGVsCKqRjUUDco-1XBGOty-be68Oo8nkHwePjOW22twVsqs0oCnK5HBkbgSNAD2npEOyrRlm_peRIO_57_9zWETxb9LoooSnUGxIzPJJWsbbcSl06cmWk6Ocud2ByO92X.0nZGpxn5ZA2KA5mcVW9K-g/__results___files/__results___58_1.png" alt="Test1" width="900" height="300"> 
</p> <br>

Overall survival (in months)

Died of cancer: 101.69 &plusmn; 74.9
Did not die of cancer by end of study: 156 &plusmn; 77.9

## Model Building

|                | Decision Tree | Logistic Regression | Gradient Boosted Trees | Ada Boosted Trees | XGBoost | 3-Layer NN |
|----------------|:-------------:|:-------------------:|:----------------------:|:-----------------:|:-------:|:----------:|
| Accuracy Score |      61%      |         59%         |           67%          |       61.5%       |   65%   |     61%    |

## Strategies for Improvement
