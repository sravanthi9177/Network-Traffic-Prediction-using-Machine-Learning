// Project Overview
Project analyzes network traffic data to cluster network patterns and predict bytes sent using machine learning. 
It combines unsupervised learning (K-Means) for identifying traffic clusters and supervised learning (Random Forest Regressor) for traffic prediction.
The goal is to help network administrators understand traffic patterns and predict network load efficiently.
Features
  1. Data Cleaning: Handles missing values and removes outliers.
  2. Feature Engineering: Creates new features like traffic_rate and packets_rate.
  3. Clustering: Groups similar traffic patterns using K-Means.
  4. Prediction: Estimates bytes_sent using Random Forest Regressor.
  5. Visualization:
     => Traffic clusters
     => Actual vs Predicted traffic
     => Feature importance

Technologies Used
1. Python 
2. Pandas & NumPy — Data manipulation
3. Matplotlib & Seaborn — Data visualization
4. Scikit-learn — Machine learning (K-Means & Random Forest)
5. VS Code — Code execution

Dataset
File: network_traffic.csv
Columns:
duration — Duration of the network session
packets — Number of packets transmitted
bytes_sent — Bytes sent
bytes_received — Bytes received
protocol — Protocol type
traffic_type — Type of network traffic

  Workflow
1. Load Data
2. Data Cleaning & Feature Engineering
3. K-Means Clustering
4. Random Forest Regression
   Features: ['duration', 'packets', 'bytes_received', 'traffic_rate', 'packets_rate']
   Target: bytes_sent
   Tuned hyperparameters for better prediction
   Evaluated using:
      R² Score: Indicates how well the model predicts traffic
      Mean Squared Error (MSE): Shows prediction error magnitude
5. Visualization
   Scatter plot of Actual vs Predicted bytes
   Feature importance bar chart
