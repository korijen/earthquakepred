import os 
os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"
import datetime
import time
from tensorflow import keras
import tensorflow as tf
import numpy as np
import pandas as pd
import sklearn
import datetime
import time
import matplotlib.pyplot as plt
import folium
from folium.plugins import HeatMap
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px
from sklearn import svm
from sklearn.svm import SVR
import seaborn as sns
from matplotlib import style
#from mpl_toolkits.basemap import Basemap
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.model_selection import cross_val_score, KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from keras.models import Sequential 
from keras.layers import LSTM, Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve, accuracy_score, f1_score, precision_recall_curve, auc
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.dates as mdates

pd.set_option('display.max_columns', None)
pd.options.display.max_rows = 20


data= pd.read_csv('Earthquakes_v3.csv')
data = data[[ 'Datetime', 'Latitude', 'Longitude', 'Depth', 'Magnitude']]
data = data.dropna()
data['Datetime'] = pd.to_datetime(data['Datetime'], format='%m/%d/%Y %H:%M')


data['Timestamp'] = (data['Datetime'] - pd.Timestamp("1970-01-01")) / pd.Timedelta(seconds=1)
x = data[['Timestamp','Latitude', 'Longitude', 'Depth']]
y = data['Magnitude']


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)



######MAGNITUDE AND DEPTH CORRELATION #######


############################################


"""

data = data.sort_values('Datetime')


data['Time_normalized'] = (data['Datetime'] - data['Datetime'].min()) / (data['Datetime'].max() - data['Datetime'].min())

cmap = plt.get_cmap("coolwarm")  
plt.figure(figsize=(10, 6))
sc = plt.scatter(
    data['Depth'],           
    data['Magnitude'],       
    c=data['Time_normalized'],
    cmap=cmap,                
    edgecolor='k',           
    alpha=0.7                 
)

cbar = plt.colorbar(sc)
cbar.set_label('Time (Earlier to Recent)', rotation=270, labelpad=15)

plt.xlabel('Depth (km)')
plt.ylabel('Magnitude')
plt.title('Magnitude vs. Depth of Earthquakes in Athens\n(Color-coded by Time)')
plt.grid(True)

plt.show()


############################################





"""





#####  LINEAR GROWTH IN NUMBER OF EARTHQUAKES OVER TIME ######

############################################


"""

data['Datetime'] = pd.to_datetime(data['Datetime'])

data['Year'] = data['Datetime'].dt.year


earthquakes_per_year = data.groupby('Year').size().cumsum()


plt.figure(figsize=(8, 6))
plt.plot(earthquakes_per_year.index, earthquakes_per_year.values, color='b')
plt.title("Greece")
plt.xlabel("Year")
plt.ylabel("Cumulative Number of Earthquakes")
plt.grid(True)
plt.xticks(rotation=45)  
plt.tight_layout()       
plt.show()



plt.figure(figsize=(12, 6))
sns.lineplot(x='Year', y='Magnitude', data=data)
plt.title('Magnitude Over Year')
plt.xlabel('Year')
plt.ylabel('Magnitude')
plt.show()

"""
############################################





############################################

"""


data.set_index('Datetime', inplace=True)
monthly_earthquakes = data.resample('M').size()

monthly_magnitude_avg = data['Magnitude'].resample('M').mean()

rolling_earthquake_count = monthly_earthquakes.rolling(window=6).mean()
rolling_magnitude_avg = monthly_magnitude_avg.rolling(window=6).mean()

fig, ax1 = plt.subplots(figsize=(14, 8))

ax1.plot(monthly_earthquakes.index, monthly_earthquakes, label='Monthly Earthquake Count', color='blue', alpha=0.4)
ax1.plot(rolling_earthquake_count.index, rolling_earthquake_count, label='6-Month Rolling Average (Count)', color='blue')
ax1.set_xlabel("Time")
ax1.set_ylabel("Earthquake Count", color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.grid(True)


ax2 = ax1.twinx()
ax2.plot(monthly_magnitude_avg.index, monthly_magnitude_avg, label='Monthly Average Magnitude', color='red', alpha=0.4)
ax2.plot(rolling_magnitude_avg.index, rolling_magnitude_avg, label='6-Month Rolling Average (Magnitude)', color='red')
ax2.set_ylabel("Magnitude", color='red')
ax2.tick_params(axis='y', labelcolor='red')

fig.suptitle("Temporal Patterns in Earthquake Occurrences and Magnitudes")
fig.tight_layout() 

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

plt.show()

"""


############################################


############################################
"""

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

sns.histplot(data['Magnitude'], binwidth=0.5, kde=True, color="lightgreen", edgecolor='black', ax=ax1)
ax1.set_title("Earthquake Magnitude Data")
ax1.set_xlabel("Magnitude Scale")
ax1.set_ylabel("Number of Earthquakes")

ax1.legend()

sns.histplot(data['Depth'], bins=20, kde=True, color="lightblue", edgecolor='black', ax=ax2)
ax2.set_title("Earthquake Depth Data")
ax2.set_xlabel("Depth (km)")
ax2.set_ylabel("Number of Earthquakes")

ax2.legend()

plt.tight_layout()
plt.show()

"""

############################################

"""

fig = px.scatter(
    data, 
    x='Longitude', 
    y='Latitude', 
    color='Magnitude', 
    color_continuous_scale='plasma',  
    labels={'Longitude': 'longitude', 'Latitude': 'latitude', 'Magnitude': 'magnitude'},
    title='Latitude vs Longitude with Magnitude Color Scale'
)


fig.update_layout(
    coloraxis_colorbar=dict(
        title="Magnitude",
        tickvals=[3, 4, 5, 6, 7],  
        lenmode="fraction", 
        len=0.75
    )
)

fig.show()

"""

############################################




############################################


"""

data_for_corr = data[['Magnitude', 'Depth', 'Longitude', 'Latitude', 'Timestamp']]
correlation_matrix = data_for_corr.corr()


plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title("Correlation Matrix Heatmap for Earthquake Data")
plt.show()


"""


############################################


############################################

"""

plt.figure(figsize=(12, 6))

plt.subplot(1, 4, 1)
sns.histplot(data=data, x='Magnitude', bins=10, kde=True)
plt.title('Distribution of Magnitude')

plt.subplot(1, 4, 2)
sns.histplot(data=data, x='Depth', bins=10, kde=True)
plt.title('Distribution of Depth')

plt.subplot(1, 4, 3)
sns.histplot(data=data, x='Latitude', bins=10, kde=True)
plt.title('Distribution of Latitude')

plt.subplot(1, 4, 4)
sns.histplot(data=data, x='Longitude', bins=10, kde=True)
plt.title('Distribution of Longitude')

plt.tight_layout()  
plt.show()

"""


###################   KNN regressor model   #########################


"""


knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(x_train, y_train)


data['mag_pred'] = knn.predict(x)
data['Month'] = data['Datetime'].dt.to_period('M').astype(str)
data_filtered = data[(data['Datetime'] >= '2009-01-01') & (data['Datetime'] < '2010-01-01')]

# Melt data for box plot preparation
data_filtered_melted = data_filtered.melt(id_vars=['Month'], value_vars=['Magnitude', 'mag_pred'], 
                                          var_name='Type', value_name='value')

# Calculate IQR and filter to keep one highest and one lowest outlier per month and type
def filter_extreme_outliers(group):
    Q1 = group['value'].quantile(0.25)
    Q3 = group['value'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    low_outliers = group[group['value'] < lower_bound]
    high_outliers = group[group['value'] > upper_bound]
    extreme_outliers = []
    if not low_outliers.empty:
        extreme_outliers.append(low_outliers.loc[low_outliers['value'].idxmin()])
    if not high_outliers.empty:
        extreme_outliers.append(high_outliers.loc[high_outliers['value'].idxmax()])
    
    return pd.DataFrame(extreme_outliers)


extreme_outliers = data_filtered_melted.groupby(['Month', 'Type']).apply(filter_extreme_outliers).reset_index(drop=True)


plt.figure(figsize=(12, 6))
sns.boxplot(x='Month', y='value', hue='Type', data=data_filtered_melted, showfliers=False)

# Adjust x-coordinates for the outliers to ensure they align with either 'Magnitude' or 'mag_pred'
x_positions = {month: i for i, month in enumerate(data_filtered_melted['Month'].unique())}
x_offsets = {'Magnitude': -0.2, 'mag_pred': 0.2}  

# Plot each outlier at the adjusted x position without adding them to the legend
for _, outlier in extreme_outliers.iterrows():
    x_pos = x_positions[outlier['Month']] + x_offsets[outlier['Type']]
    plt.scatter(x_pos, outlier['value'], color='black', zorder=2)


plt.xticks(range(len(x_positions)), x_positions.keys(), rotation=45)
plt.title('Magnitude Distribution per month (2009) compared with KNN mean prediction')
plt.xlabel('Month')
plt.ylabel('Magnitude')
plt.legend(title='Legend', loc='upper right')
plt.show()





############################################



data['mag_pred'] = knn.predict(x)


data_2009 = data[(data['Datetime'] >= '2009-01-01') & (data['Datetime'] < '2010-01-01')]
# Sample 200 records
data_2009_sampled = data_2009.sample(n=200, random_state=1)
data_2009_sampled = data_2009_sampled.sort_values(by='Datetime')

plt.figure(figsize=(12, 6))
plt.plot(data_2009_sampled['Datetime'], data_2009_sampled['Magnitude'], color='red', label='Mag_Actual')
plt.plot(data_2009_sampled['Datetime'], data_2009_sampled['mag_pred'], color='blue', label='Mag_Prediction')
plt.title("Magnitude Distribution: Actual vs KNN Mean Prediction (Sampled Data)")
plt.xlabel("Time")
plt.ylabel("Actual vs Prediction")
plt.xticks(rotation=45)
plt.legend()
plt.show()


"""




############################################
"""
athens_map = folium.Map(location=[37.9838, 23.7275], zoom_start=8)
heat_data = [[row['Latitude'], row['Longitude'], row['Magnitude'] * 2] for index, row in data.iterrows()]
HeatMap(heat_data, radius=10, max_zoom=10).add_to(athens_map)

athens_map.save("athens_earthquake1_heatmap.html")

"""
############################################





###### LINEAR REGRESSION ######





"""

scaler = MinMaxScaler()
x_normalized = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x_normalized, y, test_size=0.2, random_state=0)


ridge_model = Ridge(alpha=1.0)  
lasso_model = Lasso(alpha=0.005)   
ridge_model.fit(x_train, y_train)
lasso_model.fit(x_train, y_train)


y_pred_ridge = ridge_model.predict(x_test)
ridge_r2 = r2_score(y_test, y_pred_ridge)
ridge_mse = mean_squared_error(y_test, y_pred_ridge)

y_pred_lasso = lasso_model.predict(x_test)
lasso_r2 = r2_score(y_test, y_pred_lasso)
lasso_mse = mean_squared_error(y_test, y_pred_lasso)

print("Ridge Regression -> R²: {:.2f}, MSE: {:.2f}".format(ridge_r2, ridge_mse))
print("Lasso Regression -> R²: {:.2f}, MSE: {:.2f}".format(lasso_r2, lasso_mse))


fig, axs = plt.subplots(1, 2, figsize=(14, 6))


axs[0].scatter(y_test, y_pred_ridge, color='blue', alpha=0.5, label="Predicted Values (Ridge)")
axs[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label="Ideal Fit (y=x)")
axs[0].set_xlabel("True Values (y_test)")
axs[0].set_ylabel("Predicted Values (y_pred)")
axs[0].set_title("Ridge Regression: True vs Predicted Magnitudes")
axs[0].legend()


axs[1].scatter(y_test, y_pred_lasso, color='green', alpha=0.5, label="Predicted Values (Lasso)")
axs[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label="Ideal Fit (y=x)")
axs[1].set_xlabel("True Values (y_test)")
axs[1].set_ylabel("Predicted Values (y_pred)")
axs[1].set_title("Lasso Regression: True vs Predicted Magnitudes")
axs[1].legend()

plt.suptitle("Model Comparison: Ridge vs Lasso Regression")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()




regressor = LinearRegression()
regressor.fit(x_train, y_train)
y_pred = regressor.predict(x_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)


print("R^2: {:.2f}, MSE: {:.2f}".format(r2, mse))


plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='green', s=20, label='Predicted Values')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Ideal Fit (y=x)')
plt.xlabel('True Values ')
plt.ylabel('Predicted Values')
plt.title('Model Comparison: True vs Predicted Earthquake Magnitudes')
plt.legend()
plt.grid(True)

plt.show()



# Sample 500 
sample_indices = np.random.choice(len(y_test), 500, replace=False)
y_test_sample = y_test.iloc[sample_indices]
y_pred_sample = y_pred[sample_indices]


plt.figure(figsize=(10, 6))
plt.plot(y_test_sample.values, label='True Values', color='black', alpha=0.6, linewidth=1)
plt.plot(y_pred_sample, label='Predicted Values', color='blue')
plt.title("Linear Regression Model - Predicted vs True Values (Random 500 Samples)")
plt.xlabel("Sample")
plt.ylabel("Magnitude")
plt.legend()
plt.show()

"""


############## Linear regression KFOLD visualization ##################

"""

scaler = StandardScaler()
X_scaled = scaler.fit_transform(x)
model = LinearRegression()

kf = KFold(n_splits=10, shuffle=True, random_state=0)  # 10-fold cross-validation


mse_scores = []
r2_scores = []


for train_index, test_index in kf.split(X_scaled):
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
   
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
  
    mse_scores.append(mean_squared_error(y_test, y_pred))
    r2_scores.append(r2_score(y_test, y_pred))


average_mse = np.mean(mse_scores)
average_r2 = np.mean(r2_scores)

print("K-Fold Cross-Validation Results for Linear Regression:")
print("Average R²: {:.2f}".format(average_r2))
print("Average MSE: {:.2f}".format(average_mse))


plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(range(1, 11), mse_scores, marker='o', color='b', label='MSE per Fold')
plt.axhline(y=average_mse, color='r', linestyle='--', label=f'Average MSE: {average_mse:.2f}')
plt.xlabel('Fold')
plt.ylabel('Mean Squared Error')
plt.title('Mean Squared Error per Fold')
plt.legend()


plt.subplot(1, 2, 2)
plt.plot(range(1, 11), r2_scores, marker='o', color='g', label='R² per Fold')
plt.axhline(y=average_r2, color='r', linestyle='--', label=f'Average R²: {average_r2:.2f}')
plt.xlabel('Fold')
plt.ylabel('R² Score')
plt.title('R² Score per Fold')
plt.legend()

plt.tight_layout()
plt.show()

"""


########################################################



###### RANDOM FOREST REGRESSOR #######

"""


scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)


rf = RandomForestRegressor(n_estimators=100, random_state=0)
rf.fit(x_train_scaled, y_train)

y_pred = rf.predict(x_test_scaled)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error (MSE): {mse:.2f}')
print(f'Mean Absolute Error (MAE): {mae:.2f}')
print(f'R-squared (R2): {r2:.2f}')

# 1. Plot: Actual vs Predicted Magnitudes
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.6, color='blue')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', color='red', lw=2)
plt.xlabel('Actual Magnitude')
plt.ylabel('Predicted Magnitude')
plt.title('Random Forest: Actual vs Predicted Magnitudes')
plt.show()

# 2. Residual Plot (errors between actual and predicted values)
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
plt.scatter(y_test, residuals, alpha=0.6, color='purple')
plt.axhline(0, color='red', linestyle='--', lw=2)
plt.xlabel('Actual Magnitude')
plt.ylabel('Residuals')
plt.title('Residuals of the Random Forest Model')
plt.show()

# 3. Distribution of Residuals
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True, color='green')
plt.title('Distribution of Residuals (Random Forest Model)')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.show()

# 4. Feature Importance Plot
importances = rf.feature_importances_
feature_names = x.columns
plt.figure(figsize=(10, 6))
sns.barplot(x=importances, y=feature_names, color="skyblue")
plt.title('Feature Importances in Random Forest Model')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()





results = pd.DataFrame({
    'Datetime': data['Datetime'].iloc[y_test.index],
    'Actual Magnitude': y_test,
    'Predicted Magnitude': y_pred
})

# Filter results to only include data for January 2021
start_date = pd.to_datetime("2021-01-01")
end_date = pd.to_datetime("2021-02-01")
results_jan_2021 = results[(results['Datetime'] >= start_date) & (results['Datetime'] < end_date)]




plt.figure(figsize=(12, 6))
plt.scatter(results_jan_2021['Datetime'], results_jan_2021['Actual Magnitude'], label='Actual Magnitude', color='blue', alpha=0.6, s=10)
plt.scatter(results_jan_2021['Datetime'], results_jan_2021['Predicted Magnitude'], label='Predicted Magnitude', color='orange', alpha=0.6, s=10)
plt.ylim(0.5, 4.0)
plt.xlabel('Date')
plt.ylabel('Magnitude')
plt.title('Comparison of Actual and Predicted Magnitudes for January 2021')
plt.xticks(rotation=45)

plt.grid(True)
plt.legend()


plt.tight_layout() 
plt.show()

"""



######################## Support Vector Machine #######################




"""

data = data[(data['Datetime'].dt.year >= 2016) & (data['Datetime'].dt.year <= 2021)]


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


svm_model = SVR(kernel='rbf', C=10, epsilon=0.1)  # You may adjust parameters C and epsilon as needed
svm_model.fit(X_train_scaled, y_train)

y_pred = svm_model.predict(X_test_scaled)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"SVM Model Performance:\nMean Squared Error: {mse:.3f}\nR^2 Score: {r2:.3f}")

# Plotting actual vs predicted magnitudes
plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_pred, color='blue', alpha=0.5, edgecolors='k')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # Ideal prediction line
plt.xlabel("Actual Magnitude")
plt.ylabel("Predicted Magnitude")
plt.title("Actual vs Predicted Magnitude (SVM) - 2016 to 2021")
plt.grid()
plt.show()

# Plotting residuals
residuals = y_test - y_pred
plt.figure(figsize=(10, 5))
plt.hist(residuals, bins=30, color='purple', edgecolor='black', alpha=0.7)
plt.xlabel("Residual")
plt.ylabel("Frequency")
plt.title("Residuals of Predicted Magnitude (SVM) - 2016 to 2021")
plt.grid()
plt.show()


plt.figure(figsize=(8, 6))
plt.boxplot([y_test, y_pred], labels=['Actual Magnitude', 'Predicted Magnitude'], patch_artist=True,
            boxprops=dict(facecolor='skyblue'), medianprops=dict(color='darkblue'))
plt.ylabel("Magnitude")
plt.title("Box Plot of Actual vs Predicted Magnitude (SVM Model)")
plt.grid()
plt.show()

"""

############################################################################################






################### NAIVE BAYES ###################



"""

bins = [0, 3, 5, np.inf]
labels = ['Minor', 'Moderate', 'Severe']
y_binned = pd.cut(y, bins=bins, labels=labels)
x_train, x_test, y_train, y_test = train_test_split(x, y_binned, test_size=0.2, random_state=0)


nb_classifier = GaussianNB()
nb_classifier.fit(x_train, y_train)

y_pred = nb_classifier.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')


conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.title("Confusion Matrix for Naive Bayes Classifier")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()


f1 = f1_score(y_test, y_pred, average=None)  
f1_weighted = f1_score(y_test, y_pred, average='weighted')  

class_report = classification_report(y_test, y_pred, target_names=['Minor', 'Moderate', 'Severe'], output_dict=True)
report_df = pd.DataFrame(class_report).transpose()


plt.figure(figsize=(8, 6))
sns.heatmap(report_df.iloc[:-1, :-1], annot=True, cmap="Blues", fmt=".2f")
plt.title('Classification Report - Naive Bayes Model')
plt.ylabel('Metrics')
plt.xlabel('Classes')
plt.show()





lb = LabelBinarizer()
y_test_bin = lb.fit_transform(y_test)
y_pred_bin = lb.transform(y_pred)


for i in range(len(labels)):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_bin[:, i])
    plt.plot(fpr, tpr, label=f"ROC curve for {labels[i]} (area = {roc_auc_score(y_test_bin[:, i], y_pred_bin[:, i]):.2f})")

plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.show()

"""




################################################ NAIVE BAYES 2 ############################

"""

bins = [0, 3, 5, np.inf]
labels = ['Minor', 'Moderate', 'Severe']
y_binned = pd.cut(y, bins=bins, labels=labels)
x_train, x_test, y_train, y_test = train_test_split(x, y_binned, test_size=0.2, random_state=0)


nb_classifier = GaussianNB()
nb_classifier.fit(x_train, y_train)
y_pred = nb_classifier.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')


conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.title("Confusion Matrix for Naive Bayes Classifier")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()


conf_matrix_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_normalized, annot=True, fmt=".2f", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.title("Normalized Confusion Matrix for Naive Bayes Classifier")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()


f1 = f1_score(y_test, y_pred, average=None)  
f1_weighted = f1_score(y_test, y_pred, average='weighted')  
class_report = classification_report(y_test, y_pred, target_names=labels, output_dict=True)
report_df = pd.DataFrame(class_report).transpose()
plt.figure(figsize=(8, 6))
sns.heatmap(report_df.iloc[:-1, :-1], annot=True, cmap="Blues", fmt=".2f")
plt.title('Classification Report - Naive Bayes Model')
plt.ylabel('Metrics')
plt.xlabel('Classes')
plt.show()

lb = LabelBinarizer()
y_test_bin = lb.fit_transform(y_test)
y_pred_bin = lb.transform(y_pred)
plt.figure(figsize=(10, 6))
for i, label in enumerate(labels):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_bin[:, i])
    plt.plot(fpr, tpr, label=f"ROC curve for {label} (AUC = {roc_auc_score(y_test_bin[:, i], y_pred_bin[:, i]):.2f})")
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for Each Class")
plt.legend(loc="lower right")
plt.show()


plt.figure(figsize=(10, 6))
for i, label in enumerate(labels):
    precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_pred_bin[:, i])
    plt.plot(recall, precision, label=f'Precision-Recall for {label}')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve for Each Class")
plt.legend(loc="best")
plt.show()

y_proba = nb_classifier.predict_proba(x_test)
plt.figure(figsize=(10, 6))
for i, label in enumerate(labels):
    sns.histplot(y_proba[:, i], bins=15, kde=True, label=f"{label} Predicted Probability", alpha=0.6)
plt.xlabel("Predicted Probability")
plt.ylabel("Frequency")
plt.title("Predicted Probability Distribution for Each Class")
plt.legend()
plt.show()


fig, ax = plt.subplots(1, 2, figsize=(14, 6))
sns.countplot(x=y_test, order=labels, ax=ax[0], palette="Blues").set(title='True Labels Distribution')
sns.countplot(x=y_pred, order=labels, ax=ax[1], palette="Greens").set(title='Predicted Labels Distribution')
ax[0].set_xlabel("Classes")
ax[1].set_xlabel("Classes")
plt.suptitle("True vs Predicted Class Distributions")
plt.show()

"""

############################################################################## 



##########################  NEURAL NETWORK ###################################






scaler = MinMaxScaler()
x = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = Sequential([
    Dense(64, input_dim=x_train.shape[1], activation='relu'),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1)  
])


model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])
history = model.fit(x_train, y_train, epochs=20, batch_size=32, validation_data=(x_test, y_test), verbose=1)


y_pred = model.predict(x_test).flatten()


mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R^2): {r2:.2f}")

# Plot training & validation loss values
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss During Training')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.show()

# Plot predicted vs actual magnitudes
plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6, edgecolors='k')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Magnitude')
plt.ylabel('Predicted Magnitude')
plt.title('Actual vs Predicted Magnitude')
plt.grid(True)
plt.show()


residuals = y_test - y_pred


plt.figure(figsize=(10, 5))
plt.hist(residuals, bins=30, color='purple', edgecolor='black', alpha=0.7)
plt.title('Residuals of Predicted Magnitude')
plt.xlabel('Residual')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()




#############################################################


