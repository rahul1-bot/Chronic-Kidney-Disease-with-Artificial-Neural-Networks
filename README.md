# _Classification of Chronic Kidney Disease with Artificial Neural Networks_

## _Authors_
* Rahul Sawhney <sup>1</sup>
* Aabha Malik <sup>1</sup>
* Shambhavi Lau <sup>1</sup>
* Leah Khan <sup>1</sup>
* Dharvi Kaur <sup>1</sup>

## _Abstract_
Chronic Kidney Disease or CKD is one of the most prevalent disease which influence humans on a larger scale and proves to be fatal as it remains dormant unless irreversible damages have been made to the kidney of an individual. Progression of CKD is related to variety of great complications, including increased incidence of various disorders, anemia, hyperlipidemia, nerve damage, pregnancy complications and even complete kidney failure. Millions of people die from this disease every year. Diagnosing CKD is a cumbersome task as there are no major symptom that can be used as a benchmark to detect the disease. In cases when diagnosis persists, some results may be interpreted incorrectly. This paper proposes a Multi- Layered Perceptron Classifier that uses deep neural network in order to predict whether a patient has CKD or not. The model is trained on a dataset of about four hundred patients and considers diverse signs and symptoms which includes blood pressure, age, sugar level, red blood cell count, etc. The experimental results display that the proposed model can perform classification with the testing accuracy of 100 %. The aim is to help introduce Deep Learning methods in learning from the dataset attribute reports and detect CKD correctly to a large extent.

## _Keywords_
1) Chronic Kidney Disease
2) Deep Neural Networks
3) Supervised Machine Learning
4) Classification Problem

## _Methodology_
![Capture](https://user-images.githubusercontent.com/65220704/132189326-18d0d357-e964-4557-a453-7d32e76649d6.PNG)
![image](https://user-images.githubusercontent.com/65220704/132188318-03c460d7-57e1-479d-b4ad-d5f200401ddd.png)

The model is trained using the principles of Supervised Machine Learning [9] where for given value of X i.e., input values (x1, x2, x3, …, xn) there is a corresponding Y value i.e., target or output value.
As shown in the figure, The data is fed into the model where X comprises various features such as blood features such as blood pressure, age, sugar, etc. and Y is the target class that consists of binaries values i.e., affected by CKD or not. The model uses batch learning or offline learning mechanism for training, in which the training data is provided to the ANN model in batches, and then validated and tested.
The dataset used in the proposed model is collected from Apollo Hospitals in Managiri and Karaikudi, Tamil Nadu, India. It contains some 25 attributes and a site, which is a branch of Google LLC, and this site is an Online machine learning community experts and data scientists, allowing users to publish and find data sets of various problems called Kaggle.

## _Feature Attributes of the Dataset_
![image](https://user-images.githubusercontent.com/65220704/132188495-5f19f589-ae52-4573-81d0-e64a1d7b8efd.png)

## _Model Architecture_
![image](https://user-images.githubusercontent.com/65220704/132188576-3df6f97b-5138-4a27-8aef-607910a78b28.png)

## _Model Hyper-Parameters_
![__temop](https://user-images.githubusercontent.com/65220704/132188734-5f54d446-56fd-4bea-b265-a4521f1663d3.PNG)

## _Results_
![image](https://user-images.githubusercontent.com/65220704/132188840-53e39cf2-902c-4bab-a935-3942df395ef3.png)
![image](https://user-images.githubusercontent.com/65220704/132188848-c423fa9a-3ada-425d-9b19-6fc17848cd0b.png)

The Artificial Neural Network can outperform the SVC and Naïve Bayes Classifier in the task of diagnosing chronic kidney disease. The ANN was able to achieve a highly acceptable testing score as compared to the Naïve Bayes Classifier is 100% and the Logistic Regression and SVM attained a score of 96% and 82% respectively. The generated confusion matrix illustrates the efficient classification performance of the generated model.
A Deep Neural Network model is highly robust to the fluttering environment which makes it immune to noise. Also, many hidden layers help the model is learn complex data patterns.
ANN model for the task of chronic kidney disease classification. The precision score for the model came out to be 1.0 which implies that the model is highly efficient in determining the percentage of the Positive predicted cases for patients suffering from CKD out of the total positive result. For ‘notckd’ class the Recall value also came out to be the most optimal i.e., 1.0 which illustrates the True Negative rate of the model.
To evaluate the model’s performance keeping in mind the precision and the recall trade-off, the F1 score was evaluated which gives the harmonic mean of the precision and recall values. F1 Score for the proposed model is 100% And the Final testing accuracy received from the same is 100%.

## _Conclusion_
The Chronic Kidney Disease is undoubtedly one of the fatal diseases that are very challenging to be diagnosed with high accuracy and precision. In a nutshell, creating an application for detecting chronical diseases will not only help medical professionals for solving critical problems but will also be beneficial for people to whom approaching a doctor has been problematic. One of the reasons why it is hard to diagnose is that CKD does not depend on a single feature, therefore it is hard to predict. Also, common symptoms of CKD are not a significant contribution in identifying the disease.
The most encouraging aspect of this system is its evident capability to resolve the invariant problems despite all the adversities and difficulties. It is posed that even with a complex dataset and despite large numbers and great structural overlap of features to be distinguished, it works with a great accuracy score. It is highly recommended for the readers to work on neural networks as there is so much undiscovered in the field of machine learning. Therefore, this research paper is just a foot in the door, and its remaining ambiguities can be resolved gradually.

