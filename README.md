# Credit-Card-Default-Analysis
# Hardware & Software used

The project was developed using various hardware and software components. The hardware used for this project included a computer system with sufficient computational capabilities to handle the data analysis and modeling tasks.
The software tools and technologies utilized in this project include:

Programming Language:
Python - Python programming language was used as the primary language for data preprocessing, exploratory data analysis, and model development. Python provides a wide range of libraries and frameworks for data manipulation, statistical analysis, and machine learning.

Jupyter Notebooks: Jupyter Notebooks were used as an interactive computing environment for developing and executing the project code. Jupyter Notebooks allow for combining code, visualizations, and narrative text, making it easier to document and share the project workflow.

Data Analysis Libraries: The project relied on several Python libraries for data analysis, including:
•	NumPy: NumPy was used for efficient numerical computations and data manipulation.
•	Pandas: Pandas was used for data preprocessing, manipulation, and analysis, including loading the dataset, handling missing values, and performing feature engineering tasks.
•	Matplotlib: Matplotlib was used for creating visualizations, such as line plots, scatter plots, and histograms.
•	Seaborn: Seaborn, a data visualization library built on top of Matplotlib, was used for creating more advanced and aesthetically pleasing statistical visualizations.
•	Scikit-learn: Scikit-learn was used for implementing machine learning algorithms, including model training, evaluation, and prediction.

Machine Learning Models: Various machine learning models were employed to predict credit card default. Some of the commonly used models in this project might include:
•	Logistic Regression
•	Decision Trees
•	Random Forests
•	Gradient Boosting Models (e.g., XGBoost, LightGBM)
•	Support Vector Machines
•	Neural Networks (e.g., TensorFlow, Keras)

Evaluation Metrics: To assess the performance of the models, standard evaluation metrics were utilized, including accuracy, precision, recall, F1-score, and area under the ROC curve (AUC-ROC).

Version Control: Git and GitHub were used for version control and collaborative development of the project code. The GitHub repository provided a centralized location for storing and sharing the project code, facilitating collaboration and version tracking.

By leveraging these hardware and software components, I was able to perform in-depth data analysis, develop predictive models, and draw meaningful insights from the Credit Card Dataset.



# Introduction	

The aim of this study is to utilize supervised machine learning algorithms to identify the key drivers that determine the likelihood of credit card default, emphasizing the underlying mathematical aspects of the methods employed. Credit card default occurs when individuals become significantly delinquent in their credit card payments. In Taiwan, card-issuing banks overextended credit and cash cards to unqualified applicants in an attempt to expand their market share. Consequently, many cardholders, regardless of their repayment capacity, excessively utilized their credit cards for consumption, resulting in heavy credit and debt accumulation.

The objective is to develop an automated model capable of identifying the primary factors influencing credit card default and predicting default occurrences based on client information and historical transaction data. This report presents an overview of the supervised machine learning paradigm, accompanied by a comprehensive explanation of the techniques and algorithms employed in constructing the models. Specifically, Logistic Regression, Random Forest, and Support Vector Machines algorithms have been applied.

The analysis was performed using Python as the primary programming language. Several machine learning and statistical frameworks were employed, including scikit-learn, numpy, pandas, imblearn, as well as data visualization libraries such as matplotlib and seaborn.
