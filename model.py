import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFECV

class Model:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = LogisticRegression(max_iter=1000)
        self.stats1 = None
        self.stats2 = None

    def load_stats(self, stats):
        self.stats1 = stats.rename(columns=lambda x: x + "1")
        self.stats2 = stats.rename(columns=lambda x: x + "2")
        print(f'Stats loaded. Shape {self.stats1.shape}')

    def flip_and_augment(self, data):
        original = data.copy()

        flipped = data.copy()
        flipped_columns = {col: col[:-1] + '2' if col.endswith('1') else col[:-1] + '1' if col.endswith('2') else col
                        for col in data.columns if col not in ['RESULT']}
        flipped.rename(columns=flipped_columns, inplace=True)

        flipped['RESULT'] = 1 - data['RESULT']

        augmented = pd.concat([original, flipped], ignore_index=True)
        return augmented
    
    def difference_transformation(self, X):
        feature_cols = [col[:-1] for col in X.columns if col.endswith('1')]
        for feature in feature_cols:
            X[f'{feature}_diff'] = X[f'{feature}1'] - X[f'{feature}2']
        X_diff = X[[col for col in X.columns if col.endswith('_diff')]]
        return X_diff

    def fit(self, training_data, cv=5):
        augmented_data = self.flip_and_augment(training_data)

        X = augmented_data.drop(columns=['RESULT'])
        y = augmented_data['RESULT']

        X_diff = self.difference_transformation(X)

        X_diff_scaled = self.scaler.fit_transform(X_diff)

        cv_scores = cross_val_score(self.model, X_diff_scaled, y, cv=cv, scoring='accuracy')
        mean_cv_accuracy = np.mean(cv_scores)

        self.model.fit(X_diff_scaled, y)

        print(f'Model Fitted. Cross-Validated Accuracy (cv={cv}): {mean_cv_accuracy:.4f}')

        coefficients = self.model.coef_[0]  # For binary classification
        features = X_diff.columns

        # Create a DataFrame of features and their coefficients
        coef_df = pd.DataFrame({
            'feature': features,
            'coefficient': coefficients,
            'abs_coefficient': np.abs(coefficients),
            'odds_ratio': np.exp(coefficients)
        }).sort_values('abs_coefficient', ascending=False)

        print(coef_df)

    def predict(self, matchups, result_type):
        if self.stats1 is None or self.stats2 is None:
            raise ValueError("Team stats not loaded. Use load_team_stats() first.")
        
        if result_type.lower() not in ['winner', 'probability']:
            raise ValueError("Invalid result_type. Please input 'winner' or 'probability'")
        
        combined_data = matchups.merge(self.stats1, how='left', on=['TEAM1']).merge(self.stats2, how='left', on=['TEAM2'])
        combined_data.to_csv('test.csv')

        features = [
            'SEED1', 'SEED2', 'G1', 'W1', 'ADJOE1', 'ADJDE1', 'BARTHAG1', 'EFG_O1',
            'EFG_D1', 'TOR1', 'TORD1', 'ORB1', 'DRB1', 'FTR1', 'FTRD1', '2P_O1',
            '2P_D1', '3P_O1', '3P_D1', 'ADJ_T1', 'WAB1', 'G2', 'W2', 'ADJOE2',
            'ADJDE2', 'BARTHAG2', 'EFG_O2', 'EFG_D2', 'TOR2', 'TORD2', 'ORB2',
            'DRB2', 'FTR2', 'FTRD2', '2P_O2', '2P_D2', '3P_O2', '3P_D2', 'ADJ_T2',
            'WAB2'
        ]

        cleaned_data = combined_data.dropna().reset_index(drop=True)[features]
        cleaned_data_diff = self.difference_transformation(cleaned_data)
        scaled_data = self.scaler.transform(cleaned_data_diff)

        if result_type.lower() == 'probability':
            y_probs = self.model.predict_proba(scaled_data)[:, -1]
            matchups['TEAM1 WIN PROBABILITY'] = y_probs
            return matchups

        elif result_type.lower() == 'winner':
            y_pred = self.model.predict(scaled_data)
            matchups['PREDICTED WINNER'] = np.where(y_pred == 1, matchups['TEAM1'], matchups['TEAM2'])
            return matchups
        
        else:
            raise Exception("Something went wrong")