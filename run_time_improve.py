import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.neural_network import MLPRegressor
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
matplotlib.use('Agg')

# Combining the pre-split train and test datasets from kaggle, I do my own training split
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
combined_data = pd.concat([train, test], axis=0, ignore_index=True)

# keep only experienced marathon runners 
# also filter out experienced marathon runners who DNF'd the race 
#   as we cannot have nan's in the outcome varaible
marathon_veteran = combined_data[combined_data['previous_marathon_count']>0]

# make the outcome varaible - the time improvement relative to the runner's PB
marathon_veteran['time_improvement'] = marathon_veteran['personal_best_minutes'] - marathon_veteran['actual_finish_time_minutes']

# create a visualization of the data I am working with
data_dict = {
    'Feature Name': marathon_veteran.columns,
    'Data Type': [str(t) for t in marathon_veteran.dtypes],
    'Missing Values (N)': marathon_veteran.isnull().sum().values,
    'Missing %': (marathon_veteran.isnull().sum().values / len(marathon_veteran)) * 100,
    'Variance (Std Dev)': [f"{marathon_veteran[col].std():.2f}" if pd.api.types.is_numeric_dtype(marathon_veteran[col]) else 'N/A' for col in marathon_veteran.columns]
}

data_df = pd.DataFrame(data_dict).sort_values(by='Missing %', ascending=False).head(15)
data_df['Missing %'] = data_df['Missing %'].map('{:.1f}%'.format)

fig, ax = plt.subplots(figsize=(10, 4.5))
ax.axis('off')

table = ax.table(
    cellText=data_df.values,
    colLabels=data_df.columns,
    cellLoc='center', loc='center',
    colColours=['#2c3e50'] * 5
)

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.1, 1.5)

# make the first column wider so the variable names fit
first_column_width = 0.35
for i in range(len(data_df) + 1):
    table[(i, 0)].set_width(first_column_width)

for j in range(len(data_df.columns)):
    table[(0, j)].get_text().set_color('white')
    table[(0, j)].get_text().set_weight('bold')

plt.savefig('figures/data_health_table.png', dpi=300, bbox_inches='tight')
plt.close(fig)

# now make a heatmap of the most correlated variables with time improvement
target_corr = marathon_veteran.select_dtypes(include=['number']).corr()['time_improvement']
top_correlated_features = target_corr.abs().sort_values(ascending=False).head(15).index

# heatmap is unreadible with all variables 
filtered_corr_matrix = marathon_veteran[top_correlated_features].corr()

# plot the heatmap
fig, ax = plt.subplots(figsize=(10, 8), layout='constrained')
fig.set_constrained_layout_pads(h_pad=0.4, w_pad=0.0, hspace=0.0, wspace=0.0)

sns.heatmap(
    filtered_corr_matrix, 
    annot=True, 
    fmt=".2f", 
    cmap="RdBu_r", 
    vmin=-1, vmax=1, 
    square=True, 
    linewidths=0.5,
    annot_kws={"size": 9}, 
    ax=ax
)

plt.xticks(rotation=45, ha='right')
plt.suptitle('Correlation Matrix of Top 15 Features Most Linked to PR Improvement', y=0.98, fontsize=12, weight='bold')
plt.savefig('figures/correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close(fig)


# restricting my dataset to those who finished the race 
marathon_veteran = marathon_veteran[marathon_veteran['actual_finish_time_minutes'].notna()].copy()

# make the NaNs in the injury_severity column be 'No Injury', which is logical 
# injury severity is a string categorical - mild, moderate, severe
marathon_veteran['injury_severity'] = marathon_veteran['injury_severity'].fillna('No Injury')

# saving the combined and cleaned dataset down to submit
marathon_veteran.to_csv('data/final_data.csv', index=False)

# drop time improvement  since it is the outcome variable
# also drop actual finsih time, because in combindation with personal best, it is a linear comboniation of the y var
# drop medal_outcome as it conveys too much information about the time
# drop marathon date and runner id to avoid a spurious regression 
    # we already have weather, so don't believe there will be seasonality
# drop weekly mileage km as we already have this in miles
drop_cols = [
    'actual_finish_time_minutes', 'medal_outcome',
    'weekly_mileage_km', 'marathon_date', 'runner_id', 'time_improvement'
]
 
y = marathon_veteran['time_improvement']
X = marathon_veteran.drop(columns=drop_cols)

# seperate out categorical columns so we can do a one-hot encoder for MLPRegressor
categorical_cols = ['gender', 'training_program', 'motivation_level', 'marathon_weather', 'course_difficulty', 'injury_severity', 'mental_preparation_score']
numerical_cols = [col for col in X.columns if col not in categorical_cols]

# data preprocessing 
# must scale all data for elastic net and MLP
# use an imputer for all three models as its necessary for MLPRegressor
# However, not ideal for XGBoost, natively doing it would be better
numerical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore')) # to avoid the dummy varaible trap!
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_pipeline, numerical_cols),
        ('cat', categorical_pipeline, categorical_cols)
    ])

# create my train test split
# using a random state for reproducability
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7905)

# transform the data based on pipeliens and preprocessors from above
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)


####            Model 1: Xtreme Gradient Boost          #####
xgb_model = xgb.XGBRegressor(
    n_estimators=500,       # Number of trees
    max_depth=6,            # Depth of trees 
    learning_rate=0.025,     # Step size shrinkage to prevent overfitting         
    subsample=0.8,          # % of rows used per tree
    colsample_bytree=0.8,   # % of columns used per tree
    random_state=7905,      # use a random state for reproducabilty
    n_jobs=-1              
)

# fit the model
xgb_model.fit(
    X_train_processed, y_train,
    eval_set=[(X_test_processed, y_test)],
    verbose=50 
)

# test the model and save the 
y_pred_xgb = xgb_model.predict(X_test_processed)

xgb_mae = mean_absolute_error(y_test, y_pred_xgb)
xgb_rmse = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
xgb_r2 = r2_score(y_test, y_pred_xgb)

print("\n--- XGBoost Model Performance ---")
print(f"Mean Absolute Error (MAE): {xgb_mae:.3f} minutes")
print(f"Root Mean Squared Error (RMSE): {xgb_rmse:.3f} minutes")
print(f"R-squared (R²) Score: {xgb_r2:.3f}")

# data visualization on the xgboost model
# get the feature names out of the preprocessor pipeline
cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
encoded_cat_features = cat_encoder.get_feature_names_out(categorical_cols).tolist()
all_features = numerical_cols + encoded_cat_features

# get feature importances from XGBoost
xgb_importances = xgb_model.feature_importances_

xgb_importance_df = pd.DataFrame({
    'Feature': all_features,
    'Importance': xgb_importances
}).sort_values(by='Importance', ascending=False)

top10_xgb = xgb_importance_df.sort_values(by='Importance', ascending=False).head(10)['Feature'].values

# plot the top 20 most influential features
plt.figure(figsize=(10, 6))
plt.barh(xgb_importance_df['Feature'].head(20)[::-1], xgb_importance_df['Importance'].head(20)[::-1], color='teal')
plt.xlabel('XGBoost Feature Importance (Gain)')
plt.title('Top 20 Predictors of Marathon Time Improvement in XGBoost')
plt.tight_layout()
plt.savefig('figures/xgb_feature_importance.png', dpi=300, bbox_inches='tight')

# now make some a partial dependence plot to show directioanlity
top_mlp_features = xgb_importance_df['Feature'].head(6)

all_features_list = list(all_features)
feature_indices = [all_features_list.index(f) for f in top_mlp_features]

# generate the partial dependence plot for XGBoost
fig, ax = plt.subplots(figsize=(12, 8))
display = PartialDependenceDisplay.from_estimator(
    xgb_model,               
    X_test_processed,        
    features=feature_indices, 
    feature_names=all_features_list,
    ax=ax
)
plt.suptitle('Directional Impact of Top Features on PR Improvement (XGBoost)', y=0.98, fontsize=16)
plt.savefig('figures/xgb_pdp_directional_impact.png', dpi=300, bbox_inches='tight')
plt.close(fig)

# check the directional impact of the most important features               
correlation_data = []

for top_feature in xgb_importance_df['Feature'].head(20).values:
    if top_feature in marathon_veteran.columns:
        correlation = marathon_veteran[top_feature].corr(marathon_veteran['time_improvement'])
        
        # store the correlation
        correlation_data.append({
            'Feature Name': top_feature,
            'Pearson Correlation (r)': correlation
        })

# sort by the strongest absolute relationship
corr_df = pd.DataFrame(correlation_data)
corr_df['Abs_Corr'] = corr_df['Pearson Correlation (r)'].abs()
corr_df = corr_df.sort_values(by='Abs_Corr', ascending=False).drop(columns=['Abs_Corr'])

# format
corr_df['Pearson Correlation (r)'] = corr_df['Pearson Correlation (r)'].map('{:.2f}'.format)

# build the table
fig, ax = plt.subplots(figsize=(7, 6))
ax.axis('off')  

# create the styled table grid
table = ax.table(
    cellText=corr_df.values,
    colLabels=corr_df.columns,
    cellLoc='center',
    loc='center',
    colColours=['#0f4c81', '#0f4c81'] 
)

# formatting
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.1, 1.4)  

for j in range(len(corr_df.columns)):
    table[(0, j)].get_text().set_color('white')
    table[(0, j)].get_text().set_weight('bold')

# apply alternating row background colors for readability
for i in range(1, len(corr_df) + 1):
    for j in range(len(corr_df.columns)):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#f4f6f8')

#  save the figure down
plt.savefig('figures/xgb_top_features_corr.png', dpi=300, bbox_inches='tight')
plt.close(fig)




####            Model 2: Multi-Layer Perceptron Regressor        #####

# initialize the Neural Network
# hidden_layer_sizes=(64, 32) creates a network with:
    # 1st layer: 64 neurons
    # 2nd layer: 32 neurons
mlp_model = MLPRegressor(
    hidden_layer_sizes=(64, 32), 
    activation='relu',         
    solver='adam',             
    max_iter=200,              # max number of epochs 
    early_stopping=True,       # stops if validation performance stops improving
    random_state=7905,         # still using a random_state for reproducabiity 
    verbose=True               
)

# train the network on your dense processed data
X_train_dense = X_train_processed.toarray() if hasattr(X_train_processed, "toarray") else X_train_processed
X_test_dense = X_test_processed.toarray() if hasattr(X_test_processed, "toarray") else X_test_processed

# fit the mdoel
mlp_model.fit(X_train_dense, y_train)
# predict model
y_pred_mlp = mlp_model.predict(X_test_dense)

mlp_mae = mean_absolute_error(y_test, y_pred_mlp)
mlp_rmse = np.sqrt(mean_squared_error(y_test, y_pred_mlp))
mlp_r2 = r2_score(y_test, y_pred_mlp)

print("\n--- Neural Network (MLP) Model Performance ---")
print(f"Mean Absolute Error (MAE): {mlp_mae:.3f} minutes")
print(f"Root Mean Squared Error (RMSE): {mlp_rmse:.3f} minutes")
print(f"R-squared (R²) Score: {mlp_r2:.3f}")


# data visualizaiton 
# calculating MLP Permutation Importance 
result = permutation_importance(
    mlp_model, X_test_dense, y_test, 
    n_repeats=5,         # Shuffle each column 5 times to get an average
    random_state=7905,
    n_jobs=-1
)

# map the results back to the original feature names
mlp_importance_df = pd.DataFrame({
    'Feature': all_features,
    'Importance_Score': result.importances_mean
}).sort_values(by='Importance_Score', ascending=False)

top10_mlp = mlp_importance_df.sort_values(by='Importance_Score', ascending=False).head(10)['Feature'].values

# plot the top 20 features for the Neural Network
plt.figure(figsize=(10, 6))
plt.barh(mlp_importance_df['Feature'].head(20)[::-1], mlp_importance_df['Importance_Score'].head(20)[::-1], color='indigo')
plt.xlabel('Drop in Model Performance (R² Score) When Shuffled')
plt.title('Top 20 Most Important Features for Marathon Time Improvement in MLP Regressor')
plt.tight_layout()
plt.savefig('figures/MLP_feature_importance.png', dpi=300, bbox_inches='tight')

# now make some partial dependence plots to show directioanlity
top_mlp_features = mlp_importance_df['Feature'].head(6)

# get the indices of these features from your processed dataset
feature_indices = [all_features.index(f) for f in top_mlp_features]

# plot the partial dependence for the Neural Network
fig, ax = plt.subplots(figsize=(12, 8))
display = PartialDependenceDisplay.from_estimator(
    mlp_model, 
    X_test_dense, 
    features=feature_indices, 
    feature_names=all_features,
    ax=ax
)

plt.suptitle('Directional Impact of Top Features on PR Improvement (MLP Regressor)', y=0.98, fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('figures/mlp_directional_impact.png', dpi=300, bbox_inches='tight')

####            Model 3: Elastic Net        #####
# initialize model
elastic_model = ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42)

# train model 
elastic_model.fit(X_train_processed, y_train)

# predict model
y_pred_elastic = elastic_model.predict(X_test_processed)

coef_df = pd.DataFrame({
    'Feature': all_features,
    'Coefficient': elastic_model.coef_
})

elastic_copy = coef_df.copy()
elastic_copy['Abs_Coefficient'] = elastic_copy['Coefficient'].abs()
top10_elastic = elastic_copy.sort_values(by='Abs_Coefficient', ascending=False).head(10)['Feature'].values

en_mae = mean_absolute_error(y_test, y_pred_elastic)
en_rmse = np.sqrt(mean_squared_error(y_test, y_pred_elastic))
en_r2 = r2_score(y_test, y_pred_elastic)

print("--- ElasticNet Model Performance ---")
print(f"Mean Absolute Error (MAE): {en_mae:.2f} minutes")
print(f"Root Mean Squared Error (RMSE): {en_rmse:.2f} minutes")
print(f"R-squared (R²) Score: {en_r2:.2f}")

# now make a table ranking the most important featuers from each model
top_features_table = pd.DataFrame({
    'Rank': range(1, 11),
    'XGBoost Regressor': top10_xgb,
    'Neural Network (MLP)': top10_mlp,
    'Elastic Net Baseline': top10_elastic
})

print("=== TOP 10 FEATURE COMPARISON BY MODEL ===")
print(top_features_table.to_string(index=False))


fig, ax = plt.subplots(figsize=(13, 4.5))
ax.axis('off')  
header_colors = ['#2c3e50', '#16a085', '#2980b9', '#8e44ad'] 
table = ax.table(
    cellText=top_features_table.values,
    colLabels=top_features_table.columns,
    cellLoc='center',
    loc='center',
    colColours=header_colors
)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 2.0)  

# format headers
for j in range(len(top_features_table.columns)):
    table[(0, j)].get_text().set_color('white')
    table[(0, j)].get_text().set_weight('bold')

# alternate row shading for readabiltiy 
for i in range(1, 11):
    for j in range(len(top_features_table.columns)):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#f8f9fa')

# save the model
plt.savefig('figures/top_10_features_by_model.png', dpi=300, bbox_inches='tight')
plt.close(fig) 


# now make a table to save down the results
output_stats = {
    'Model Name': ['XGBoost Regressor', 'MLP Regressor', 'Elastic Net'],
    'MAE (Minutes)': [xgb_mae, mlp_mae, en_mae],
    'RMSE (Minutes)': [xgb_rmse, mlp_rmse, en_rmse],
    'R² Score': [xgb_r2, mlp_r2, en_r2]
}

summary_df = pd.DataFrame(output_stats)

# format the table for display
formatted_df = summary_df.copy()
formatted_df['MAE (Minutes)'] = formatted_df['MAE (Minutes)'].map('{:.2f}'.format)
formatted_df['RMSE (Minutes)'] = formatted_df['RMSE (Minutes)'].map('{:.2f}'.format)
formatted_df['R² Score'] = formatted_df['R² Score'].map('{:.2%}'.format)

print(formatted_df.to_string(index=False))

# save this dataframe of results as a figure
fig, ax = plt.subplots(figsize=(10, 2.5)) 
ax.axis('off')

table = ax.table(
    cellText=formatted_df.values, 
    colLabels=formatted_df.columns, 
    cellLoc='center', 
    loc='center',
    colColours=['#1f77b4', '#1f77b4', '#1f77b4', '#1f77b4'] 
)

table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 1.5) 

for j, col in enumerate(formatted_df.columns):
    table[(0, j)].get_text().set_color('white')
    table[(0, j)].get_text().set_weight('bold')

plt.savefig('figures/final_models_table.png', dpi=300, bbox_inches='tight')