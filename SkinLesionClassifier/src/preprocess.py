# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# %%
# Functions
class SexImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.male_proportion = X['sex'].value_counts(normalize=True)['male']
        self.female_proportion = X['sex'].value_counts(normalize=True)['female']
        return self
    
    def transform(self, X):
        X = X.copy()
        X['sex'] = X['sex'].apply(lambda x: np.random.choice(['male', 'female'], p=[self.male_proportion, self.female_proportion]) if pd.isna(x) else x)
        return X

class AgeApproxImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.mean_age_male = X[X['sex'] == 'male']['age_approx'].mean()
        self.mean_age_female = X[X['sex'] == 'female']['age_approx'].mean()
        return self
    
    def transform(self, X):
        X = X.copy()
        X.loc[(X['sex'] == 'male') & (X['age_approx'].isna()), 'age_approx'] = self.mean_age_male
        X.loc[(X['sex'] == 'female') & (X['age_approx'].isna()), 'age_approx'] = self.mean_age_female
        return X
    
def make_corr_plot(df, figsize=(10,8), save=False):
    corr = df.corr()

    # Create a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Set up the matplotlib figure
    plt.figure(figsize=figsize)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, annot=False, fmt=".2f", cmap='coolwarm', 
            xticklabels=corr.columns, yticklabels=corr.columns, 
            square=True, linewidths=.5)
    
    plt.savefig('plots/correlation_matrix.png')
    plt.close()

# %%
# Load the data
train = pd.read_csv('train-metadata.csv')
train = train.drop(columns=['lesion_id',
 'iddx_2',
 'iddx_3',
 'iddx_4',
 'iddx_5',
 'mel_mitotic_index',
 'mel_thick_mm'])

# Select Numeric and Categorical columns
train_numeric = train.select_dtypes(include=[np.number])
train_numeric = train_numeric.drop(columns=['target', 'tbp_lv_dnn_lesion_confidence', 'tbp_lv_nevi_confidence'])
numeric_cols = train_numeric.columns

train_categorical = train.select_dtypes(include=[object])
train_categorical = train_categorical.drop(columns=['isic_id', 'patient_id', 'tbp_lv_location','attribution', 
                                                    'copyright_license', 'iddx_full', 'iddx_1', 'anatom_site_general', 'image_type'])
categorical_cols = train_categorical.columns

# %%
# Pipeline for numeric columns
numeric_pipeline = Pipeline(steps=[
    ('scaler', StandardScaler()),  # Scaling numeric features
])

# Pipeline for categorical columns
categorical_pipeline = Pipeline(steps=[
    ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))  # One-hot encoding categorical features
])

# Full preprocessing pipeline
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_pipeline, numeric_cols),  # Apply numeric pipeline to numeric columns
    ('cat', categorical_pipeline, categorical_cols)  # Apply categorical pipeline to categorical columns
])

# Main pipeline that handles all preprocessing steps
main_pipeline = Pipeline(steps=[
    ('sex_imputer', SexImputer()),  # Custom transformer for imputing 'sex'
    ('age_imputer', AgeApproxImputer()),  # Custom transformer for imputing 'age_approx'
    ('preprocessor', preprocessor)  # Apply preprocessing pipeline to the remaining columns
])

# Apply the pipeline to the dataset
train_processed = main_pipeline.fit_transform(train)

# Convert the transformed data into a DataFrame
numeric_transformed = pd.DataFrame(train_processed[:, :len(numeric_cols)], columns=numeric_cols)
categorical_transformed = pd.DataFrame(train_processed[:, len(numeric_cols):], 
                                       columns=main_pipeline['preprocessor'].transformers_[1][1]['onehot'].get_feature_names_out())

#Add new features
numeric_transformed['lv_size_ratio'] = numeric_transformed['clin_size_long_diam_mm']/numeric_transformed['tbp_lv_minorAxisMM']
numeric_transformed['hue_contrast'] = np.abs(numeric_transformed['tbp_lv_H'] - numeric_transformed['tbp_lv_Hext'])

features_to_drop = ['tbp_lv_A', 'tbp_lv_Aext', 'tbp_lv_B', 'tbp_lv_Bext', 'tbp_lv_C', 'tbp_lv_Cext',
                    'tbp_lv_Lext', 'tbp_lv_L', 'tbp_lv_deltaLB', 'tbp_lv_x', 'tbp_lv_y', 'tbp_lv_z',
                    'tbp_lv_radial_color_std_max', 'tbp_lv_color_std_mean', 'tbp_lv_symm_2axis', 'tbp_lv_area_perim_ratio',
                    'tbp_lv_areaMM2', 'tbp_lv_perimeterMM', 'clin_size_long_diam_mm', 'tbp_lv_minorAxisMM', 
                    'tbp_lv_H', 'tbp_lv_Hext']

numeric_transformed = numeric_transformed.drop(columns=features_to_drop)

# Get Correlation Matrix
make_corr_plot(numeric_transformed, save=True)

# %%
# Save the encoder
import joblib
joblib.dump(main_pipeline, 'preprocessor.joblib')

# %%
# Combine processed numeric and categorical data
full_dataset = pd.concat([numeric_transformed, categorical_transformed], axis=1)
full_dataset['target'] = train['target']  # Add back the target variable

# Add ISIC IDs if necessary
full_dataset['isic_id'] = train['isic_id']

# Save the final dataset to CSV
full_dataset.to_csv('full_dataset.csv', index=False)


