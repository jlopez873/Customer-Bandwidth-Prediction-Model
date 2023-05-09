## Import libraries/packages
import numpy as np
from numpy.linalg import eig
import pandas as pd
from scipy.stats import zscore
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.eval_measures import rmse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

## Read in data
print('Reading Data From "churn_clean.csv"', end='\n\n')
df = pd.read_csv('churn_clean.csv').reset_index(drop=True)

## View data types
print('Reviewing Data', end='\n\n')
df.info()

## Rename survey columns
df.rename({
    'Item1':'TimelyResponse',
    'Item2':'TimelyFixes',
    'Item3':'TimelyReplacements',
    'Item4':'Reliability',
    'Item5':'Options',
    'Item6':'RespectfulResponse',
    'Item7':'CourteousExchange',
    'Item8':'ActiveListening'
}, axis=1, inplace=True)

## View summary statistics
df.describe()

## Drop less meaningful columns
df = df.drop(['CaseOrder', 'Customer_id', 'Interaction', 'UID', 'City', 'State', 'County', 'Zip', 'Lat', 'Lng', 
              'Area', 'TimeZone', 'Job', 'Marital', 'Gender', 'Churn', 'Email', 'Multiple', 'OnlineSecurity',
              'OnlineBackup', 'DeviceProtection', 'TechSupport', 'PaperlessBilling', 'PaymentMethod', 
              'TimelyResponse', 'TimelyFixes', 'TimelyReplacements', 'Reliability', 'Options', 'RespectfulResponse',
              'CourteousExchange', 'ActiveListening'], axis=1)

## Create copy of dataframe
df1 = df.copy()

## Check for missing values
print('\nMissing Values Found:', sum(df1.isna().sum()), end='\n\n')

## Check for duplicate values
print('Duplicate Values Found:', len(df1) - df1.duplicated().value_counts()[0], end='\n\n')

## Check for outliers
df1.describe()

## Separate object variables
df2 = pd.DataFrame([df1[col] for col in df1.columns if df1[col].dtype != 'object']).transpose()

## Normalize data and exclude outliers
df2 = df2[zscore(df2).abs() < 3]

## Count outliers
print('Outliers Found:', sum(df2.isna().sum()), '\nRemoving Outliers', end='\n')

## Drop outlier values
df2.dropna(inplace=True)
print('Outliers Remaining:', sum(df2.isna().sum()), end='\n\n')

## Measure data loss
lost = ((len(df1) - len(df2))/len(df1))*100
remaining = 100 - lost
print('Lost Data: {}%\nRemaining Data: {}%'.format(round(lost, 2), remaining), end='\n\n')

## Combine dataframes
df = df.loc[df2.index]
df1 = df1.loc[df2.index]

## Reset index values
df = df.reset_index(drop=True)
df1 = df1.reset_index(drop=True)

## Calculate summary statistics for dependent variable
df1.Bandwidth_GB_Year.describe()

## Calculate summary statistics for independent variables
independent_vars = pd.DataFrame(columns=['min', 'max', 'std', 'mean', 'median', 'mode'])
for col in df1.columns:
    if df1[col].dtype != object:
        independent_vars.loc[col] = [
            min(df1[col]),
            max(df1[col]),
            np.std(df1[col]),
            df1[col].mean(),
            df1[col].median(),
            df1[col].mode().values[0]
        ]
independent_vars

## Identify variables with Yes or No values
nominal = ['Techie', 'Port_modem', 'Tablet', 'Phone', 'StreamingTV', 'StreamingMovies']

## Store value distribution in data frame
nominal_df = pd.DataFrame([df[var].value_counts() for var in nominal])

## Get contract and internet service distributions
contract = pd.DataFrame(df.Contract.value_counts())
internet_service = pd.DataFrame(df.InternetService.value_counts())

## Calculate correlation coefficients
cont_disc = ['Income', 'Outage_sec_perweek', 'Tenure', 'MonthlyCharge', 'Population', 'Children', 'Age', 'Contacts', 
             'Yearly_equip_failure']
corr = pd.DataFrame(index=cont_disc, columns=['correlation'])
corr['correlation'] = [df1[n].corr(df1['Bandwidth_GB_Year']) for n in cont_disc]

## Create InternetDSL and InternetFiberOptic columns
print('Encoding Variables', end='\n\n')
dsl = []
fiber = []
for i in df1.InternetService:
    if i == 'DSL':
        dsl.append(1)
        fiber.append(0)
    elif i == 'Fiber Optic':
        dsl.append(0)
        fiber.append(1)
    else:
        dsl.append(0)
        fiber.append(0)

## Assign values ro InternetDSL and InternetFiberOptic columns
df1['InternetDSL'] = dsl
df1['InternetFiberOptic'] = fiber

## Encode InternetService variable
internet_service = {'DSL':'Yes', 'Fiber Optic':'Yes', 'None':'No'}
df1.InternetService.replace(internet_service, inplace=True)

## Initiate label encoder
le = LabelEncoder()

## Encode variables
for col in df1.columns:
    if 'Yes' in df1[col].values:
        df1[col] = le.fit_transform(df1[col])

## Encode contract variable
contract = {'Month-to-month':0, 'One year':1, 'Two Year':2}
df1.Contract.replace(contract, inplace=True)

## Review changes
df1.info()

## Store clean data as CSV
print('\nSaving data to "churn_linear_regression.csv"', end='\n\n')
df1.to_csv('churn_linear_regression.csv')

## Split target variable from explanatory variables
X, y = df1.drop('Bandwidth_GB_Year', axis=1), df1['Bandwidth_GB_Year']

## Add intercept
X = sm.add_constant(X)

## Split training and testing samples
X_train, X_test, y_train, y_test = train_test_split(X, y)

## Create and fit model
mod = sm.OLS(y_train, X_train).fit()
mod.summary()

## Predict using initial model
pred = mod.predict(X_test)

## Calculate RMSE
initial_features = X.columns
initial_rmse = rmse(y_test, pred)

## Select feature with highest correlation to target variable
feature_1 = corr.iloc[np.where(corr == corr.max())[0]].index[0]

## Create list of features already included
selected_features = [feature_1]

## Create dictionary to store rmse and features
rmse_vals = {}

## Create dataframe to store r2 values
results = pd.DataFrame(index=X.columns, columns=['R_Squared', 'RMSE'])

## Define evaluator function to score models
def evaluator(X, y, selected_features):
    ## Create results dataframe
    results = pd.DataFrame(index=X.drop(selected_features, axis=1).columns, columns=['R_Squared', 'RMSE'])
    ## Create feature dataframe
    X_feature = X[selected_features]
    ## Add constant
    X_feature = sm.add_constant(X_feature)
    ## Split train test
    X_train, X_test, y_train, y_test = train_test_split(X_feature, y)
    for col in X.drop(selected_features, axis=1).columns:
        ## Add feature to dataframe
        X_train[col] = X.loc[X_train.index][col]
        X_test[col] = X.loc[X_test.index][col]
        ## Create and fit model
        mod = sm.OLS(y_train, X_train).fit()
        ## Predict target
        pred = mod.predict(X_test)
        ## Calculate and store r2 value and rmse
        results['R_Squared'].loc[col] = np.corrcoef(y_test, pred)[0, 1]**2
        results['RMSE'].loc[col] = rmse(y_test, pred)
        ## Revert X_train, X_test
        X_train = X_train[selected_features]
        X_test = X_test[selected_features]
    ## Sort results by r-squared values
    results.sort_values('R_Squared', ascending=False, inplace=True)
    ## Add highest r-squared value to selected features
    selected_features.append(results.index[0])
    ## Display results
    results.head(3)
    ## Test new model
    mod = sm.OLS(y_train, X_train).fit()
    pred = mod.predict(X_test)
    return rmse(y_test, pred), X_test.columns.values

## Conduct step-forward feature selection
print('Performing Step-Forward Feature Selection', end='\n\n')
for i in range(len(X.columns)-2):
    rmse_vals[i] = evaluator(X.drop('const', axis=1), y, selected_features)

## Review model performance
rmse_df = pd.DataFrame(rmse_vals, index=['RMSE', 'Features']).transpose().sort_values('RMSE')
print('Model Performance Metrics:', end='\n')
print(rmse_df)

## Select features from best-performing model
features = rmse_df['Features'].values[0]
X_new = X[features]
print('\nReduced features:')

def print_features(features):
    for feature in features:
        print(feature)
    print('Features Remaining:', len(features), end='\n\n')

print_features(features)

## Define function to display VIF
def get_vif(X):
    ## Add constant
    X['const'] = 1.0
    ## Create dataframe to store vif values
    vif = pd.DataFrame()
    vif['Variable'] = X.columns
    vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif = vif[vif['Variable']!='const']
    return vif

## Calculate vif to check for multicollinearity
print('Calculating Variance Inflation Factors', end='\n\n')
vif = get_vif(X_new)

## Remove variables with VIF over 5
print('Variables With VIF Over 5:', len(vif['Variable'][vif['VIF'] > 5]), end='\n\n')
if len(vif['Variable'][vif['VIF'] > 5]) > 0:
    print('Removing Variables With VIF Over 5', end='\n\n')
X_new = X_new[vif['Variable'][vif['VIF'] < 5]]

## Repeat multicollinearity calculation
vif = get_vif(X_new)
print('Variance Inflation Factors:')
print(vif, end='\n\n')

## View selected features
features = vif.Variable
print('Reduced features:')
print_features(features)

## Create and fit the final model
final_mod = sm.OLS(y_train, sm.add_constant(X_train[features])).fit()
final_mod.summary()

## Remove insignificant variable and fit reduced model
print('Calculating Independent Variable P-Values', end='\n\n')
print('Independent Variable P-Values:')
print(np.round(final_mod.pvalues, 10), end='\n\n')
print('Removing Independent Variables With P-Value Over 0.05', end='\n\n')
reduced_features = final_mod.pvalues[(final_mod.pvalues.values < 0.05)].drop('const').index
reduced_mod = sm.OLS(y_train, sm.add_constant(X_train[reduced_features])).fit()
reduced_mod.summary()
print('Reduced Model Features Used:')
print_features(reduced_features)

## Predict using reduced model
pred = reduced_mod.predict(sm.add_constant(X_test[reduced_features]))

## Print results
final_rmse = rmse(y_test, pred)
print('Model Metrics:', '\nInitial Model RMSE:', initial_rmse, '\nInitial Features:', len(initial_features))
## Calculate residual standard error
residuals = reduced_mod.resid
resid = y_test - pred.values
rse = np.sqrt(sum(resid**2)/(len(y) - len(reduced_features)))
print('Reduced Model RSE:', rse, '\nReduced Model RMSE:', final_rmse, '\nFinal Features:', len(reduced_features), end='\n\n')