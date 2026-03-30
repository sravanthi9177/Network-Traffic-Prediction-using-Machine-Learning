import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# STAGE 1 — LOAD DATA
df = pd.read_csv("network_traffic.csv")
print("Dataset Shape:", df.shape)
print(df.head())
# STAGE 2 — CLEANING + FEATURE ENGINEERING

# Remove missing values
df = df.dropna()
# Remove outliers (top 1%)
df = df[df['bytes_sent'] < df['bytes_sent'].quantile(0.99)]
# Feature Engineering (IMPORTANT)
df['traffic_rate'] = df['bytes_received'] / (df['duration'] + 1)
df['packets_rate'] = df['packets'] / (df['duration'] + 1)
print("\nAdded new features: traffic_rate, packets_rate")
# STAGE 3 — K-MEANS CLUSTERING

features = ['duration', 'packets', 'bytes_sent', 'bytes_received']
X = df[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X_scaled)
print("\nCluster Distribution:")
print(df['cluster'].value_counts())
# Visualization
plt.figure(figsize=(6,4))
plt.scatter(df['bytes_sent'], df['bytes_received'], c=df['cluster'], cmap='viridis')
plt.xlabel("Bytes Sent")
plt.ylabel("Bytes Received")
plt.title("Network Traffic Clusters")
plt.show()

# STAGE 4 — RANDOM FOREST 
# Updated features
X_reg = df[['duration', 'packets', 'bytes_received', 'traffic_rate', 'packets_rate']]
y_reg = df['bytes_sent']
# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)
# Tuned Random Forest
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    min_samples_split=3,
    min_samples_leaf=2,
    random_state=42
)
model.fit(X_train, y_train)
# Prediction
y_pred = model.predict(X_test)
# Evaluation
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print("MODEL PERFORMANCE")
print("R² Score:", round(r2, 4))
print("MSE:", round(mse, 2))

# VISUALIZATION — ACTUAL VS PREDICTED
plt.figure(figsize=(6,4))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual Traffic")
plt.ylabel("Predicted Traffic")
plt.title(f"Actual vs Predicted (R² = {round(r2,2)})")
plt.show()
# FEATURE IMPORTANCE

importances = model.feature_importances_
features = X_reg.columns
plt.figure(figsize=(6,4))
plt.barh(features, importances)
plt.xlabel("Importance")
plt.title("Feature Importance")
plt.show()