"""
Hourly Health Data Pipeline with Lagging Indicators
==================================================
Advanced pipeline for processing health data at hourly granularity
with comprehensive lagging features for ML modeling
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from pathlib import Path
import glob
from sklearn.impute import KNNImputer
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Updated type mapping based on discoveries
HEALTH_TYPES = {
    3: 'sleep',
    6: 'weight',
    7: 'heart_rate',
    8: 'body_fat',
    9: 'analyzed_sleep',
    10: 'pressure_detection',
    11: 'stress',
    12: 'exercise_intensity',
    13: 'high_hr_alert',
    14: 'low_hr_alert',
    16: 'blood_oxygen',
    18: 'body_temperature',
    19: 'blood_oxygen_remind',
    
    # Discovered types
    500023: 'continuous_hr',      # Minute-by-minute heart rate
    500024: 'resting_hr',         # Resting heart rate updates
    400021: 'continuous_spo2',    # Continuous SpO2
    500026: 'stress_continuous',  # Continuous stress scores
    200005: 'active_hour',        # Hourly activity indicator
    500010: 'sleep_timing',       # Bed/wake times
    500005: 'sleep_daily',        # Daily sleep summary
}

class HourlyHealthPipeline:
    def __init__(self, data_path, air_quality_path=None):
        self.data_path = Path(data_path)
        self.air_quality_path = Path(air_quality_path) if air_quality_path else None
        self.UTC = pytz.UTC
        self.NPT = pytz.timezone('Asia/Kathmandu')
        
        self.data_by_type = {}
        self.hourly_data = {}
        self.daily_data = {}
        
    def convert_timezone(self, timestamp_ms):
        """Convert UTC timestamp to Nepal time"""
        if timestamp_ms == 0 or pd.isna(timestamp_ms):
            return pd.NaT
        utc_time = datetime.fromtimestamp(timestamp_ms / 1000, tz=self.UTC)
        return utc_time.astimezone(self.NPT)
    
    def load_all_data(self):
        """Load all JSON files and organize by type"""
        print("Loading all JSON files...")
        json_files = glob.glob(str(self.data_path / "*.json"))
        
        all_records = []
        for file in json_files:
            with open(file, 'r') as f:
                data = json.load(f)
                all_records.extend(data)
        
        # Organize by health type
        for record in all_records:
            health_type = HEALTH_TYPES.get(record['type'], f"unknown_{record['type']}")
            if health_type not in self.data_by_type:
                self.data_by_type[health_type] = []
            self.data_by_type[health_type].append(record)
        
        print(f"Loaded {len(all_records)} records")
        for htype, records in self.data_by_type.items():
            print(f"  - {htype}: {len(records)} records")
    
    def process_continuous_heart_rate(self):
        """Process minute-by-minute heart rate into hourly aggregates"""
        print("\nProcessing continuous heart rate data...")
        
        hr_records = []
        
        # Process regular heart rate data (type 7)
        for record in self.data_by_type.get('heart_rate', []):
            for sp in record.get('samplePoints', []):
                hr_records.append({
                    'timestamp': self.convert_timezone(sp['startTime']),
                    'heart_rate': float(sp['value']),
                    'source': 'periodic'
                })
        
        # Process continuous heart rate (type 500023)
        for record in self.data_by_type.get('continuous_hr', []):
            for sp in record.get('samplePoints', []):
                try:
                    value_data = json.loads(sp['value'])
                    hr_records.append({
                        'timestamp': self.convert_timezone(sp['startTime']),
                        'heart_rate': float(value_data.get('bpm', 0)),
                        'source': 'continuous'
                    })
                except:
                    continue
        
        if not hr_records:
            return pd.DataFrame()
        
        df_hr = pd.DataFrame(hr_records)
        df_hr = df_hr[df_hr['heart_rate'] > 0]  # Filter out zeros
        
        # Create hourly aggregates
        df_hr['datetime_hour'] = df_hr['timestamp'].dt.floor('h')
        
        hourly_hr = df_hr.groupby('datetime_hour').agg({
            'heart_rate': ['mean', 'min', 'max', 'std', 'count',
                          lambda x: np.percentile(x, 10),  # 10th percentile (proxy for resting)
                          lambda x: np.percentile(x, 90)]  # 90th percentile (proxy for active)
        }).round(2)
        
        hourly_hr.columns = ['hr_mean', 'hr_min', 'hr_max', 'hr_std', 'hr_count', 'hr_p10', 'hr_p90']
        hourly_hr['hr_range'] = hourly_hr['hr_max'] - hourly_hr['hr_min']
        hourly_hr['hr_cv'] = (hourly_hr['hr_std'] / hourly_hr['hr_mean']).round(3)  # Coefficient of variation
        
        self.hourly_data['heart_rate'] = hourly_hr
        return hourly_hr
    
    def process_hourly_stress(self):
        """Process stress data at hourly level"""
        print("\nProcessing hourly stress data...")
        
        stress_records = []
        
        # Regular stress data
        for record in self.data_by_type.get('stress', []):
            for sp in record.get('samplePoints', []):
                try:
                    stress_data = json.loads(sp['value'])
                    stress_records.append({
                        'timestamp': self.convert_timezone(sp['startTime']),
                        'stress_score': stress_data.get('score', np.nan),
                        'stress_grade': stress_data.get('grade', np.nan)
                    })
                except:
                    continue
        
        # Continuous stress (type 500026)
        for record in self.data_by_type.get('stress_continuous', []):
            for sp in record.get('samplePoints', []):
                try:
                    value_data = json.loads(sp['value'])
                    stress_records.append({
                        'timestamp': self.convert_timezone(sp['startTime']),
                        'stress_score': value_data.get('stressScore', np.nan)
                    })
                except:
                    continue
        
        if not stress_records:
            return pd.DataFrame()
        
        df_stress = pd.DataFrame(stress_records)
        df_stress['datetime_hour'] = df_stress['timestamp'].dt.floor('h')
        
        hourly_stress = df_stress.groupby('datetime_hour').agg({
            'stress_score': ['mean', 'min', 'max', 'std', 'count']
        }).round(2)
        
        hourly_stress.columns = ['stress_mean', 'stress_min', 'stress_max', 'stress_std', 'stress_count']
        
        self.hourly_data['stress'] = hourly_stress
        return hourly_stress
    
    def process_hourly_spo2(self):
        """Process continuous SpO2 data"""
        print("\nProcessing hourly SpO2 data...")
        
        spo2_records = []
        
        # Process both SpO2 sources
        for source in ['blood_oxygen', 'continuous_spo2']:
            for record in self.data_by_type.get(source, []):
                for sp in record.get('samplePoints', []):
                    try:
                        value_data = json.loads(sp['value'])
                        spo2_val = value_data.get('spo2') or value_data.get('avgSaturation', np.nan)
                        if spo2_val and 70 <= spo2_val <= 100:  # Valid SpO2 range
                            spo2_records.append({
                                'timestamp': self.convert_timezone(sp['startTime']),
                                'spo2': float(spo2_val)
                            })
                    except:
                        continue
        
        if not spo2_records:
            return pd.DataFrame()
        
        df_spo2 = pd.DataFrame(spo2_records)
        df_spo2['datetime_hour'] = df_spo2['timestamp'].dt.floor('h')
        
        hourly_spo2 = df_spo2.groupby('datetime_hour').agg({
            'spo2': ['mean', 'min', 'std', 'count']
        }).round(2)
        
        hourly_spo2.columns = ['spo2_mean', 'spo2_min', 'spo2_std', 'spo2_count']
        
        self.hourly_data['spo2'] = hourly_spo2
        return hourly_spo2
    
    def process_sleep_data(self):
        """Process sleep data with hourly breakdown"""
        print("\nProcessing sleep data...")
        
        sleep_events = []
        
        # Analyzed sleep phases
        for record in self.data_by_type.get('analyzed_sleep', []):
            for sp in record.get('samplePoints', []):
                phase_map = {
                    'PROFESSIONAL_SLEEP_SHALLOW': 'light',
                    'PROFESSIONAL_SLEEP_DEEP': 'deep',
                    'PROFESSIONAL_SLEEP_DREAM': 'rem',
                    'PROFESSIONAL_SLEEP_WAKE': 'awake',
                    'PROFESSIONAL_SLEEP_NOON': 'nap'
                }
                
                phase = phase_map.get(sp['key'], sp['key'])
                start_time = self.convert_timezone(sp['startTime'])
                end_time = self.convert_timezone(sp['endTime'])
                
                # Create hourly breakdown of sleep phases
                current_time = start_time.replace(minute=0, second=0, microsecond=0)
                while current_time < end_time:
                    next_hour = current_time + timedelta(hours=1)
                    phase_end = min(next_hour, end_time)
                    duration_min = (phase_end - max(current_time, start_time)).total_seconds() / 60
                    
                    if duration_min > 0:
                        sleep_events.append({
                            'datetime_hour': current_time,
                            'phase': phase,
                            'duration_min': duration_min
                        })
                    
                    current_time = next_hour
        
        if not sleep_events:
            return pd.DataFrame()
        
        df_sleep = pd.DataFrame(sleep_events)
        
        # Pivot to get phase durations per hour
        hourly_sleep = df_sleep.pivot_table(
            index='datetime_hour',
            columns='phase',
            values='duration_min',
            aggfunc='sum',
            fill_value=0
        )
        
        # Add sleep quality metrics
        if not hourly_sleep.empty:
            hourly_sleep['total_sleep_min'] = hourly_sleep.drop(columns=['awake', 'nap'], errors='ignore').sum(axis=1)
            hourly_sleep['sleep_efficiency'] = (
                hourly_sleep['total_sleep_min'] / 
                (hourly_sleep['total_sleep_min'] + hourly_sleep.get('awake', 0))
            ).replace([np.inf, -np.inf], 0) * 100
            
        self.hourly_data['sleep'] = hourly_sleep
        return hourly_sleep
    
    def process_air_quality_hourly(self):
        """Process air quality data at hourly level"""
        print("\nProcessing hourly air quality data...")
        
        if not self.air_quality_path or not self.air_quality_path.exists():
            return pd.DataFrame()
        
        df_air = pd.read_csv(self.air_quality_path)
        df_air['datetime'] = pd.to_datetime(df_air['period.datetimeFrom.local'])
        df_air['datetime_hour'] = df_air['datetime'].dt.floor('h')
        
        # Pivot air quality parameters
        hourly_air = df_air.pivot_table(
            index='datetime_hour',
            columns='parameter_name',
            values='value',
            aggfunc='mean'
        )
        
        # Calculate AQI
        if 'pm25' in hourly_air.columns:
            hourly_air['aqi_pm25'] = hourly_air['pm25'].apply(self.calculate_aqi_pm25)
        
        self.hourly_data['air_quality'] = hourly_air
        return hourly_air
    
    @staticmethod
    def calculate_aqi_pm25(pm25):
        """Calculate AQI from PM2.5"""
        if pd.isna(pm25):
            return np.nan
        
        if pm25 <= 12.0:
            return ((50-0)/(12.0-0.0)) * (pm25-0.0) + 0
        elif pm25 <= 35.4:
            return ((100-51)/(35.4-12.1)) * (pm25-12.1) + 51
        elif pm25 <= 55.4:
            return ((150-101)/(55.4-35.5)) * (pm25-35.5) + 101
        elif pm25 <= 150.4:
            return ((200-151)/(150.4-55.5)) * (pm25-55.5) + 151
        else:
            return 301
    
    def create_lagging_features(self, df, feature_cols, lag_configs):
        """
        Create comprehensive lagging features
        
        lag_configs: list of dicts with keys:
            - hours: number of hours to lag
            - agg: aggregation function ('mean', 'max', 'min', 'std', 'sum')
            - name_suffix: suffix for the feature name
        """
        print("\nCreating lagging features...")
        
        lagged_features = df.copy()
        
        for col in feature_cols:
            if col not in df.columns:
                continue
                
            for config in lag_configs:
                hours = config['hours']
                agg = config['agg']
                suffix = config.get('name_suffix', f'lag_{hours}h_{agg}')
                
                if agg == 'mean':
                    lagged_features[f'{col}_{suffix}'] = df[col].rolling(
                        window=hours, min_periods=1
                    ).mean()
                elif agg == 'max':
                    lagged_features[f'{col}_{suffix}'] = df[col].rolling(
                        window=hours, min_periods=1
                    ).max()
                elif agg == 'min':
                    lagged_features[f'{col}_{suffix}'] = df[col].rolling(
                        window=hours, min_periods=1
                    ).min()
                elif agg == 'std':
                    lagged_features[f'{col}_{suffix}'] = df[col].rolling(
                        window=hours, min_periods=1
                    ).std()
                elif agg == 'sum':
                    lagged_features[f'{col}_{suffix}'] = df[col].rolling(
                        window=hours, min_periods=1
                    ).sum()
                elif agg == 'trend':
                    # Calculate trend (slope) over the window
                    lagged_features[f'{col}_{suffix}'] = df[col].rolling(
                        window=hours, min_periods=2
                    ).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
        
        return lagged_features
    
    def add_temporal_features(self, df):
        """Add temporal features like hour of day, day of week, season"""
        print("\nAdding temporal features...")
        
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Time of day categories
        df['is_night'] = df['hour'].isin(range(22, 24)).astype(int) | df['hour'].isin(range(0, 6)).astype(int)
        df['is_morning'] = df['hour'].isin(range(6, 12)).astype(int)
        df['is_afternoon'] = df['hour'].isin(range(12, 18)).astype(int)
        df['is_evening'] = df['hour'].isin(range(18, 22)).astype(int)
        
        # Season (Nepal specific)
        df['season'] = pd.cut(df['month'], 
                             bins=[0, 2, 5, 9, 11, 12],
                             labels=['winter', 'spring', 'monsoon', 'autumn', 'winter2'])
        df['season'] = df['season'].fillna('winter')
        
        # One-hot encode season
        season_dummies = pd.get_dummies(df['season'], prefix='season')
        df = pd.concat([df, season_dummies], axis=1)
        
        return df
    
    def merge_all_hourly_data(self):
        """Merge all hourly data sources"""
        print("\nMerging all hourly data...")
        
        # Get date range
        all_dates = []
        for df in self.hourly_data.values():
            if not df.empty:
                all_dates.extend(df.index.tolist())
        
        if not all_dates:
            return pd.DataFrame()
        
        # Create hourly index
        date_range = pd.date_range(
            start=min(all_dates).floor('h'),
            end=max(all_dates).ceil('h'),
            freq='h'
        )
        
        merged_df = pd.DataFrame(index=date_range)
        merged_df.index.name = 'datetime_hour'
        
        # Merge all sources
        for name, df in self.hourly_data.items():
            if not df.empty:
                merged_df = merged_df.join(df, how='left')
                print(f"  - Merged {name}: {df.shape[1]} columns")
        
        return merged_df
    
    def create_ml_features(self, hourly_df):
        """Create comprehensive features for ML including lagging indicators"""
        print("\nCreating ML features with lagging indicators...")
        
        # Define lag configurations
        lag_configs = [
            # Short-term (hourly) lags
            {'hours': 1, 'agg': 'mean', 'name_suffix': 'prev_1h'},
            {'hours': 3, 'agg': 'mean', 'name_suffix': 'prev_3h'},
            {'hours': 6, 'agg': 'mean', 'name_suffix': 'prev_6h'},
            {'hours': 12, 'agg': 'mean', 'name_suffix': 'prev_12h'},
            
            # Daily patterns
            {'hours': 24, 'agg': 'mean', 'name_suffix': 'prev_24h_mean'},
            {'hours': 24, 'agg': 'max', 'name_suffix': 'prev_24h_max'},
            {'hours': 24, 'agg': 'min', 'name_suffix': 'prev_24h_min'},
            {'hours': 24, 'agg': 'std', 'name_suffix': 'prev_24h_std'},
            
            # Weekly patterns
            {'hours': 168, 'agg': 'mean', 'name_suffix': 'prev_week_mean'},
            {'hours': 168, 'agg': 'std', 'name_suffix': 'prev_week_std'},
            
            # Monthly patterns (approximate)
            {'hours': 720, 'agg': 'mean', 'name_suffix': 'prev_month_mean'},
            
            # Trends
            {'hours': 24, 'agg': 'trend', 'name_suffix': 'trend_24h'},
            {'hours': 168, 'agg': 'trend', 'name_suffix': 'trend_week'},
        ]
        
        # Features to create lags for
        lag_features = [
            # Heart rate
            'hr_mean', 'hr_std', 'hr_range', 'hr_cv',
            # Stress
            'stress_mean', 'stress_std',
            # Air quality
            'pm25', 'aqi_pm25', 'o3',
            # SpO2
            'spo2_mean', 'spo2_min',
            # Sleep (if any in current hour)
            'total_sleep_min', 'deep', 'light', 'rem'
        ]
        
        # Create lagging features
        ml_df = self.create_lagging_features(hourly_df, lag_features, lag_configs)
        
        # Add temporal features
        ml_df = self.add_temporal_features(ml_df)
        
        # Create interaction features
        print("Creating interaction features...")
        
        # Activity × Air Quality interactions
        if 'hr_cv' in ml_df.columns and 'aqi_pm25' in ml_df.columns:
            ml_df['high_activity_bad_air'] = (
                (ml_df['hr_cv'] > ml_df['hr_cv'].quantile(0.75)) & 
                (ml_df['aqi_pm25'] > 100)
            ).astype(int)
            
            ml_df['activity_air_interaction'] = ml_df['hr_cv'] * ml_df['aqi_pm25']
        
        # Stress × Time interactions
        if 'stress_mean' in ml_df.columns:
            ml_df['evening_stress'] = ml_df['stress_mean'] * ml_df['is_evening']
            ml_df['night_stress'] = ml_df['stress_mean'] * ml_df['is_night']
        
        # Cumulative features
        print("Creating cumulative features...")
        
        # Cumulative stress over past week
        if 'stress_mean' in ml_df.columns:
            ml_df['cumulative_stress_week'] = ml_df['stress_mean'].rolling(
                window=168, min_periods=1
            ).sum()
        
        # Sleep debt (cumulative deficit from 8 hours)
        if 'total_sleep_min' in ml_df.columns:
            ml_df['sleep_deficit_hourly'] = 30 - ml_df['total_sleep_min']  # 30 min per hour ideal
            ml_df['cumulative_sleep_debt'] = ml_df['sleep_deficit_hourly'].rolling(
                window=168, min_periods=1
            ).sum()
        
        return ml_df
    
    def create_sleep_quality_targets(self, ml_df):
        """Create target variables for sleep quality prediction"""
        print("\nCreating sleep quality target variables...")
        
        # For each hour, look ahead to next sleep period
        # This is complex - we need to identify sleep periods and quality
        
        # Simple approach: create targets for specific sleep hours (10 PM - 6 AM)
        sleep_hours = list(range(22, 24)) + list(range(0, 6))
        ml_df['is_sleep_hour'] = ml_df.index.hour.isin(sleep_hours)
        
        # Sleep quality metrics as targets
        if 'total_sleep_min' in ml_df.columns:
            ml_df['sleep_quality_score'] = 0
            
            # During sleep hours, calculate quality
            sleep_mask = ml_df['is_sleep_hour']
            
            # Components of sleep quality
            if 'deep' in ml_df.columns:
                deep_score = (ml_df['deep'] / ml_df['total_sleep_min']).fillna(0) * 40
                ml_df.loc[sleep_mask, 'sleep_quality_score'] += deep_score[sleep_mask]
            
            if 'sleep_efficiency' in ml_df.columns:
                eff_score = ml_df['sleep_efficiency'] * 0.3
                ml_df.loc[sleep_mask, 'sleep_quality_score'] += eff_score[sleep_mask]
            
            # Duration score (max 30 points for 30+ minutes of sleep per hour)
            duration_score = (ml_df['total_sleep_min'].clip(0, 30) / 30) * 30
            ml_df.loc[sleep_mask, 'sleep_quality_score'] += duration_score[sleep_mask]
        
        return ml_df
    
    def generate_analysis_report(self, ml_df):
        """Generate comprehensive analysis report"""
        print("\n" + "="*80)
        print("HOURLY HEALTH DATA ANALYSIS REPORT")
        print("="*80)
        
        print(f"\nData Range: {ml_df.index.min()} to {ml_df.index.max()}")
        print(f"Total Hours: {len(ml_df)}")
        print(f"Total Features: {len(ml_df.columns)}")
        
        # Feature categories
        print("\nFeature Categories:")
        
        base_features = [col for col in ml_df.columns if not any(
            x in col for x in ['lag_', 'prev_', 'trend_', 'cumulative_']
        )]
        lag_features = [col for col in ml_df.columns if any(
            x in col for x in ['lag_', 'prev_', 'trend_']
        )]
        
        print(f"  Base features: {len(base_features)}")
        print(f"  Lagging features: {len(lag_features)}")
        print(f"  Temporal features: {len([col for col in ml_df.columns if col in ['hour', 'day_of_week', 'month', 'is_weekend']])}")
        
        # Data quality
        print("\nData Quality:")
        missing_pct = (ml_df.isnull().sum() / len(ml_df) * 100).round(1)
        high_missing = missing_pct[missing_pct > 50]
        
        if len(high_missing) > 0:
            print("  High missing data features (>50%):")
            for col, pct in high_missing.head(10).items():
                print(f"    - {col}: {pct}%")
        else:
            print("  No features with >50% missing data")
        
        # Seasonal patterns
        if 'season' in ml_df.columns:
            print("\nSeasonal Data Distribution:")
            season_counts = ml_df['season'].value_counts()
            for season, count in season_counts.items():
                hours = count
                days = hours / 24
                print(f"  {season}: {hours:,} hours ({days:.0f} days)")
        
        print("\n" + "="*80)

def main():
    # Initialize pipeline
    pipeline = HourlyHealthPipeline(
        data_path="health/data/",
        air_quality_path="airquality/data/kathmandu_air_quality_2019_2025.csv"
    )
    
    # Load all data
    pipeline.load_all_data()
    
    # Process each data type at hourly level
    pipeline.process_continuous_heart_rate()
    pipeline.process_hourly_stress()
    pipeline.process_hourly_spo2()
    pipeline.process_sleep_data()
    pipeline.process_air_quality_hourly()
    
    # Merge all hourly data
    hourly_df = pipeline.merge_all_hourly_data()
    
    if hourly_df.empty:
        print("No data to process!")
        return None
    
    # Create ML features with lagging indicators
    ml_df = pipeline.create_ml_features(hourly_df)
    
    # Create target variables
    ml_df = pipeline.create_sleep_quality_targets(ml_df)
    
    # Generate report
    pipeline.generate_analysis_report(ml_df)
    
    # Save processed data
    print("\nSaving processed data...")
    ml_df.to_csv("hourly_health_ml_features.csv")
    
    # Save a subset for quick inspection
    # ml_df.head(1000).to_excel("hourly_health_sample.xlsx")
    
    print("\nProcessing complete!")
    print(f"Output files:")
    print(f"  - hourly_health_ml_features.csv (full dataset)")
    print(f"  - hourly_health_sample.xlsx (first 1000 hours)")
    
    return ml_df

if __name__ == "__main__":
    ml_df = main()