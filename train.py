import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib

def load_and_preprocess_data():
    print("Loading data...")
    df = pd.read_csv("cell2celltrain.csv")
    
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = df.select_dtypes(include=['object']).columns
    
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors='coerce')
    df[cat_cols] = df[cat_cols].apply(lambda x: x.str.strip().str.title() if x.dtype == "object" else x)
    
    # HandsetPrice
    df["HandsetPrice"] = df["HandsetPrice"].apply(pd.to_numeric, errors='coerce')
    df['HandsetPrice'] = df.groupby('IncomeGroup')['HandsetPrice'].transform(lambda x: x.fillna(x.median()))
    
    # CreditRating
    df['CreditRating'] = df['CreditRating'].astype(str).str.extract('(\d+)').astype(float)
    df['CreditRating'] = df['CreditRating'].max() + 1 - df['CreditRating']
    
    df = df.dropna()
    
    # Binary variables
    binary_cols = ['ChildrenInHH','HandsetRefurbished','HandsetWebCapable','TruckOwner','RVOwner',
                   'BuysViaMailOrder','RespondsToMailOffers','OptOutMailings','NonUSTravel','OwnsComputer',
                   'HasCreditCard','NewCellphoneUser','NotNewCellphoneUser','OwnsMotorcycle',
                   'MadeCallToRetentionTeam','Churn']
    
    # Only replace if columns exist
    existing_binary = [col for col in binary_cols if col in df.columns]
    df[existing_binary] = df[existing_binary].replace({'Yes':1, 'No':0, 'yes':1, 'no':0})
    
    df.drop(columns=['NotNewCellphoneUser','CustomerID'], inplace=True, errors='ignore')
    
    if 'Homeownership' in df.columns:
        df['Homeownership'] = df['Homeownership'].map({'Known':1, 'Unknown':0})
    
    bool_cols = df.select_dtypes(include='bool').columns
    if len(bool_cols) > 0:
        df[bool_cols] = df[bool_cols].astype(int)
        
    # Drop highly correlated and VIF features as per notebook
    drop_cols = ['TotalCalls','RetentionCalls','MadeCallToRetentionTeam','RetentionOffersAccepted',
                 'DroppedCalls','CustomerCareCalls','OverageMinutes','Handsets','BuysViaMailOrder',
                 'ReceivedCalls','CallForwardingCalls','ServiceArea', 'MonthlyRevenue','MonthlyMinutes',
                 'OutboundCalls','PeakCallsInOut','OffPeakCallsInOut','UniqueSubs','AgeHH2',
                 'UnansweredCalls','HasCreditCard']
    df.drop(columns=drop_cols, inplace=True, errors='ignore')
    
    # Fill remaining NaNs to prevent errors
    df = df.fillna(0)
    return df

def main():
    df = load_and_preprocess_data()
    
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    # Save the expected feature names and their types for Streamlit
    feature_info = {}
    for col in X.columns:
        if X[col].dtype == 'object':
            feature_info[col] = {'type': 'categorical', 'options': X[col].unique().tolist()}
        else:
            feature_info[col] = {'type': 'numeric', 'min': float(X[col].min()), 'max': float(X[col].max()), 'median': float(X[col].median())}
            
    joblib.dump(feature_info, 'feature_info.pkl')
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    cat_cols = X_train.select_dtypes(include='object').columns
    num_cols = X_train.select_dtypes(exclude='object').columns
    
    print("Building pipeline...")
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_cols),
            ('num', 'passthrough', num_cols)
        ]
    )
    
    # Complete Pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('scaler', RobustScaler()),
        ('classifier', LogisticRegression(class_weight='balanced', max_iter=1000))
    ])
    
    print("Training model...")
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    score = pipeline.score(X_test, y_test)
    print(f"Test Accuracy: {score:.4f}")
    
    # Save Pipeline
    print("Saving pipeline to model.pkl...")
    joblib.dump(pipeline, 'model.pkl')
    print("Done!")

if __name__ == '__main__':
    main()
