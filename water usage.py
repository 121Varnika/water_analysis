import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class LightGBMWaterPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = ['M1_IN_Klts', 'M2_IN_Klts', 'M3_IN_Klts', 'M4_IN_Klts', 'M5_IN_Klts']
        self.historical_data = []
        self.monthly_predictions = []
        self.current_month = 1
        self.current_year = 2025
        self.trained_features = None

    def load_excel_data(self, file_path):
        """Load water usage data from Excel file"""
        try:
            df = pd.read_excel("Water level.xlsx")
            df.columns = df.columns.str.strip().str.replace(" ", "_")
            print(f"Data loaded successfully. Shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            return df
        except Exception as e:
            print(f"Error loading Excel file: {e}")
            return None

    def create_sample_data(self):
        """Create realistic sample data for testing"""
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        years = [2022, 2023, 2024]

        np.random.seed(42)  # For reproducible results
        data = []

        for year in years:
            for i, month in enumerate(months):
                # Create seasonal patterns
                seasonal_factor = 1 + 0.4 * np.sin(2 * np.pi * i / 12)
                yearly_trend = 1 + 0.03 * (year - 2022)

                # Base consumption patterns (realistic values)
                base_values = {
                    'M1_IN_Klts': 18000 + np.random.normal(0, 1500),
                    'M2_IN_Klts': 10000 + np.random.normal(0, 1000),
                    'M3_IN_Klts': 20000 + np.random.normal(0, 2000),
                    'M4_IN_Klts': 12000 + np.random.normal(0, 1200),
                    'M5_IN_Klts': 8000 + np.random.normal(0, 800)
                }

                row = {'YEAR': f"{month}-{year % 100:02d}"}
                for col, base_val in base_values.items():
                    row[col] = max(0, int(base_val * seasonal_factor * yearly_trend))

                data.append(row)

        return pd.DataFrame(data)

    def feature_engineering(self, df):
        """Create features for LightGBM model"""
        df = df.copy()

        # Calculate total usage
        df['Total_Usage'] = df[self.feature_columns].sum(axis=1)

        # Parse date information
        df['Month'] = df['YEAR'].str.split('-').str[0]
        df['Year'] = df['YEAR'].str.split('-').str[1].astype(int) + 2000

        # Month mapping
        month_map = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
                     'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
        df['Month_Num'] = df['Month'].map(month_map)

        # Cyclical features for seasonality
        df['Month_Sin'] = np.sin(2 * np.pi * df['Month_Num'] / 12)
        df['Month_Cos'] = np.cos(2 * np.pi * df['Month_Num'] / 12)
        df['Quarter'] = ((df['Month_Num'] - 1) // 3) + 1

        # Statistical features for each meter
        for col in self.feature_columns:
            # Percentage of total usage
            df[f'{col}_pct'] = df[col] / df['Total_Usage']

            # Lag features (previous month)
            df[f'{col}_lag1'] = df[col].shift(1)

            # Rolling statistics
            df[f'{col}_roll3_mean'] = df[col].rolling(window=3, min_periods=1).mean()
            df[f'{col}_roll3_std'] = df[col].rolling(window=3, min_periods=1).std()

            # Year-over-year comparison (if available)
            df[f'{col}_yoy'] = df[col] / df[col].shift(12) - 1

        # Total usage statistics
        df['Total_lag1'] = df['Total_Usage'].shift(1)
        df['Total_roll3_mean'] = df['Total_Usage'].rolling(window=3, min_periods=1).mean()
        df['Total_roll6_mean'] = df['Total_Usage'].rolling(window=6, min_periods=1).mean()
        df['Total_yoy'] = df['Total_Usage'] / df['Total_Usage'].shift(12) - 1

        # Interaction features
        df['M1_M2_ratio'] = df['M1_IN_Klts'] / (df['M2_IN_Klts'] + 1)
        df['M3_M4_ratio'] = df['M3_IN_Klts'] / (df['M4_IN_Klts'] + 1)

        # Time-based features
        df['Is_Summer'] = df['Month_Num'].isin([4, 5, 6]).astype(int)
        df['Is_Winter'] = df['Month_Num'].isin([12, 1, 2]).astype(int)
        df['Is_Monsoon'] = df['Month_Num'].isin([7, 8, 9]).astype(int)

        return df

    def prepare_training_data(self, df):
        """Prepare data for LightGBM training"""
        df_features = self.feature_engineering(df)

        # Select features for training
        feature_cols = (
            self.feature_columns +
            [f'{col}_pct' for col in self.feature_columns] +
            [f'{col}_lag1' for col in self.feature_columns] +
            [f'{col}_roll3_mean' for col in self.feature_columns] +
            [f'{col}_roll3_std' for col in self.feature_columns] +
            [f'{col}_yoy' for col in self.feature_columns] +
            ['Month_Sin', 'Month_Cos', 'Quarter', 'Year'] +
            ['Total_lag1', 'Total_roll3_mean', 'Total_roll6_mean', 'Total_yoy'] +
            ['M1_M2_ratio', 'M3_M4_ratio'] +
            ['Is_Summer', 'Is_Winter', 'Is_Monsoon']
        )

        # Remove features that don't exist
        available_features = [col for col in feature_cols if col in df_features.columns]

        # Handle missing values
        df_clean = df_features[available_features + ['Total_Usage']].fillna(0)

        X = df_clean[available_features]
        y = df_clean['Total_Usage']

        self.trained_features = available_features
        return X, y, df_features

    def train_lightgbm(self, df, test_size=0.2):
        """Train LightGBM model with optimized parameters"""
        X, y, df_features = self.prepare_training_data(df)

        # Split data
        if len(X) > 10:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, shuffle=True
            )
        else:
            X_train, X_test, y_train, y_test = X, X, y, y

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # LightGBM parameters optimized for small datasets
        lgb_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': min(31, 2 ** 5),
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_child_samples': max(5, len(X_train) // 20),
            'min_child_weight': 0.01,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'random_state': 42,
            'verbosity': -1,
            'force_col_wise': True
        }

        # Create LightGBM datasets
        train_data = lgb.Dataset(X_train_scaled, label=y_train)
        valid_data = lgb.Dataset(X_test_scaled, label=y_test, reference=train_data)

        # Train model (silent)
        self.model = lgb.train(
            lgb_params,
            train_data,
            valid_sets=[train_data, valid_data],
            num_boost_round=200,
            callbacks=[
                lgb.early_stopping(stopping_rounds=20),
                lgb.log_evaluation(period=0)
            ]
        )

        # Store historical data for plotting
        self.historical_data = df_features[['Month_Num', 'Year', 'Total_Usage']].copy()

        return self.model

    def predict_next_month(self, current_data):
        """Predict water usage for next month using current month's data"""
        if self.model is None:
            print("Model not trained yet!")
            return None

        # Create a temporary dataframe with current data
        temp_data = {
            'YEAR': f"{'Jan'}-{self.current_year % 100:02d}",  # Placeholder
            **current_data
        }
        temp_df = pd.DataFrame([temp_data])

        # Add required columns for feature engineering
        temp_df = self.feature_engineering(temp_df)

        # Prepare features (fill missing with 0)
        features = []
        for feature_name in self.trained_features:
            if feature_name in temp_df.columns:
                value = temp_df[feature_name].iloc[0]
                features.append(value if pd.notna(value) else 0)
            else:
                features.append(0)

        # Scale features
        features_scaled = self.scaler.transform([features])

        # Make prediction
        prediction = self.model.predict(features_scaled)[0]

        return max(prediction, 0)  # Ensure non-negative

    def predict_full_year(self, start_month=1, start_year=2025):
        """Predict water usage for a full year"""
        if self.model is None:
            print("Model not trained yet!")
            return []

        predictions = []
        current_month = start_month
        current_year = start_year

        # Use last known values as base for predictions
        if len(self.monthly_predictions) > 0:
            last_data = self.monthly_predictions[-1].copy()
        else:
            # Use average from historical data
            if len(self.historical_data) > 0:
                avg_total = self.historical_data['Total_Usage'].mean()
                last_data = {col: avg_total / len(self.feature_columns) for col in self.feature_columns}
            else:
                last_data = {col: 15000 for col in self.feature_columns}  # Default values

        for _ in range(12):
            # Predict next month
            prediction = self.predict_next_month(last_data)

            # Store prediction
            month_name = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                         'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][current_month - 1]

            predictions.append({
                'month': current_month,
                'year': current_year,
                'month_name': month_name,
                'prediction': prediction
            })

            # Update last_data with predicted values (distribute total across meters)
            if prediction > 0:
                # Distribute prediction proportionally based on historical averages
                total_current = sum(last_data.values())
                for col in self.feature_columns:
                    if total_current > 0:
                        last_data[col] = (last_data[col] / total_current) * prediction
                    else:
                        last_data[col] = prediction / len(self.feature_columns)

            # Move to next month
            current_month += 1
            if current_month > 12:
                current_month = 1
                current_year += 1

        return predictions

    def plot_predictions(self):
        """Plot historical data and predictions"""
        if len(self.historical_data) == 0 and len(self.monthly_predictions) == 0:
            print("No data to plot!")
            return

        plt.figure(figsize=(16, 8))

        # Prepare historical data
        hist_dates = []
        hist_values = []
        if len(self.historical_data) > 0:
            for _, row in self.historical_data.iterrows():
                month_name = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                             'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][int(row['Month_Num']) - 1]
                hist_dates.append(f"{month_name} {int(row['Year'])}")
                hist_values.append(row['Total_Usage'])

        # Prepare predicted data
        pred_dates = []
        pred_values = []
        if len(self.monthly_predictions) > 0:
            for pred in self.monthly_predictions:
                pred_dates.append(f"{pred['month_name']} {pred['year']}")
                pred_values.append(pred['prediction'])

        # Combine all dates and values
        all_dates = hist_dates + pred_dates
        all_values = hist_values + pred_values

        # Plot historical data
        if hist_values:
            plt.plot(hist_dates, hist_values, label="Historical Usage", marker='o', color='blue')

        # Plot predicted data
        if pred_values:
            # Plot the connecting line between historical and predicted
            if hist_values and pred_values:
                plt.plot([hist_dates[-1], pred_dates[0]],
                        [hist_values[-1], pred_values[0]],
                        linestyle='--', color='gray', alpha=0.5)

            plt.plot(pred_dates, pred_values, label="Predicted Usage", marker='x', linestyle='--', color='orange')

            # Add value annotations for recent predictions
            recent_points = min(6, len(pred_values))
            for i in range(len(pred_values) - recent_points, len(pred_values)):
                plt.annotate(f'{pred_values[i]:.0f}',
                           (len(hist_dates) + i, pred_values[i]),
                           textcoords="offset points",
                           xytext=(0,10),
                           ha='center',
                           fontsize=10,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))

        # Format the plot
        plt.xticks(rotation=45, ha='right')
        plt.ylabel("Total Water Usage (Klts)")
        plt.title("Water Usage: Historical vs Predicted")
        plt.legend()
        plt.grid(True)

        # Adjust x-axis ticks to avoid overcrowding
        if len(all_dates) > 24:  # If too many points
            step = max(1, len(all_dates) // 12)
            tick_positions = list(range(0, len(all_dates), step))
            tick_labels = all_dates[::step]
            plt.xticks(tick_positions, tick_labels, rotation=45, ha='right')

        plt.tight_layout()
        plt.show()

    def run_prediction_system(self, data_file=None):
        """Interactive prediction system"""
        print("="*60)
        print("LIGHTGBM WATER USAGE PREDICTION SYSTEM")
        print("="*60)

        # Load data
        if data_file:
            df = self.load_excel_data(data_file)
            if df is None:
                print("Using sample data instead")
                df = self.create_sample_data()
        else:
            df = self.create_sample_data()

        # Train model
        print("Training model...")
        self.train_lightgbm(df)
        print("Model trained successfully!")

        print("\n" + "="*60)
        print("Enter monthly water usage data for 2025 predictions")
        print("Commands: 'quit' to exit, 'reset' to restart")
        print("="*60)

        months = ['January', 'February', 'March', 'April', 'May', 'June',
                 'July', 'August', 'September', 'October', 'November', 'December']

        while True:
            if self.current_month <= 12:
                month_name = months[self.current_month - 1]
                print(f"\n--- {month_name} {self.current_year} ---")
            else:
                print("\n--- Year Complete! Showing full predictions ---")
                self.plot_predictions()

                # Ask if user wants to continue to next year
                continue_next = input("\nContinue to next year? (y/n): ").strip().lower()
                if continue_next != 'y':
                    print("Thank you for using the prediction system!")
                    break

                # Reset for next year
                self.current_month = 1
                self.current_year += 1
                month_name = months[0]
                print(f"\n--- {month_name} {self.current_year} ---")

            try:
                current_data = {}
                user_input = ""

                # Input for each meter
                for col in self.feature_columns:
                    while True:
                        user_input = input(f"Enter {col.replace('_', ' ')}: ").strip()

                        if user_input.lower() == 'quit':
                            print("Goodbye!")
                            return
                        elif user_input.lower() == 'reset':
                            self.current_month = 1
                            self.current_year = 2025
                            self.monthly_predictions = []
                            print("System reset!")
                            break

                        try:
                            current_data[col] = float(user_input)
                            break
                        except ValueError:
                            print("Please enter a valid number.")

                if user_input.lower() == 'reset':
                    continue

                # Store current data
                self.monthly_predictions.append({
                    'month': self.current_month,
                    'year': self.current_year,
                    'month_name': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][self.current_month - 1],
                    'prediction': sum(current_data.values()),
                    **current_data
                })

                # Calculate current month total
                current_total = sum(current_data.values())
                print(f"\n✓ {month_name} {self.current_year} total: {current_total:.0f} Klts")

                # Show next month prediction
                if self.current_month < 12:
                    next_month_prediction = self.predict_next_month(current_data)
                    next_month_name = months[self.current_month]
                    print(f"→ Predicted {next_month_name} {self.current_year}: {next_month_prediction:.0f} Klts")

                # Move to next month
                self.current_month += 1

                # Show plot automatically
                print("\n" + "="*40)
                self.plot_predictions()
                print("="*40)

            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")
                continue

# Main execution
if __name__ == "__main__":
    predictor = LightGBMWaterPredictor()

    # Uncomment one of these options:

    # Option 1: Use with your Excel file
    # predictor.run_prediction_system("Water level.xlsx")

    # Option 2: Use with sample data (default)
    predictor.run_prediction_system()

    # Option 3: Train and test separately
    # df = predictor.load_excel_data("Water level.xlsx")
    # if df is not None:
    #     predictor.train_lightgbm(df)
    #     prediction = predictor.predict_next_month({
    #         'M1_IN_Klts': 18000,
    #         'M2_IN_Klts': 10000,
    #         'M3_IN_Klts': 20000,
    #         'M4_IN_Klts': 12000,
    #         'M5_IN_Klts': 8000
    #     })
    #     print(f"Prediction: {prediction}")
    # else:
    #     print("Could not load Excel file, using sample data")
    #     predictor.run_prediction_system()