
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (classification_report, confusion_matrix, 
                           accuracy_score, precision_score, recall_score, f1_score,
                           silhouette_score, adjusted_rand_score)
from sklearn.tree import plot_tree
import warnings
warnings.filterwarnings('ignore')

class SleepQualityAnalysisPipeline:
    def __init__(self, data_path="hourly_health_ml_features.csv"):
        self.data_path = data_path
        self.models = {}
        self.results = {}
        self.cluster_results = {}
        self.selected_features = []
        
        # Top 45 features based on importance analysis
        self.top_features = [
            'o3_prev_month_mean', 'hr_cv_prev_week_std', 'hr_cv_prev_24h_max',
            'hr_range_prev_week_mean', 'hr_mean_trend_24h', 'stress_std_prev_month_mean',
            'hr_cv_prev_6h', 'aqi_pm25_prev_24h_min', 'hr_mean_prev_24h_min',
            'o3_prev_12h', 'spo2_mean_prev_month_mean', 'hr_mean_prev_6h',
            'o3_prev_24h_min', 'o3_prev_week_mean', 'o3_prev_24h_mean',
            'aqi_pm25_prev_24h_std', 'aqi_pm25_prev_24h_mean', 'aqi_pm25_prev_24h_max',
            'aqi_pm25_prev_12h', 'hr_std_trend_24h', 'pm25_prev_month_mean',
            'aqi_pm25_prev_month_mean', 'aqi_pm25_prev_6h', 'day_of_week',
            'pm25_prev_24h_mean', 'o3_prev_24h_max', 'pm25_prev_12h',
            'cumulative_stress_week', 'spo2_min_prev_week_mean', 'o3_prev_6h',
            'pm25_prev_6h', 'hr_mean_prev_24h_max', 'hr_std_prev_24h_max',
            'hr_cv_trend_24h', 'pm25_prev_24h_max', 'hr_range_prev_24h_std',
            'hr_cv_prev_24h_min', 'hr_range_prev_24h_min', 'o3_prev_3h',
            'hr_mean', 'hr_std', 'stress_mean', 'spo2_mean', 'pm25', 'hour'
        ]
        
    def load_and_prepare_data(self):
        """Load data and create classification targets"""
        print("Loading and preparing data for classification...")
        
        # Load data
        self.df = pd.read_csv(self.data_path, index_col=0, parse_dates=True)
        print(f"Loaded {len(self.df)} hourly records")
        
        # Fix negative PM2.5 values
        pm25_cols = [col for col in self.df.columns if 'pm25' in col.lower()]
        for col in pm25_cols:
            if col in self.df.columns:
                neg_count = (self.df[col] < 0).sum()
                if neg_count > 0:
                    print(f"  Fixing {neg_count} negative values in {col}")
                    self.df.loc[self.df[col] < 0, col] = np.nan
        
        # Create sleep quality score (same as before)
        self._create_sleep_quality_score()
        
        # Create classification targets
        self._create_classification_targets()
        
        return self.df
    
    def _create_sleep_quality_score(self):
        """Create sleep quality score for each night"""
        print("Creating sleep quality scores...")
        
        # Focus on sleep hours
        sleep_hours = list(range(22, 24)) + list(range(0, 6))
        sleep_mask = self.df.index.hour.isin(sleep_hours)
        
        self.df['night_sleep_quality'] = 0
        self.df['date'] = self.df.index.date
        
        # Calculate nightly metrics
        nightly_metrics = self.df[sleep_mask].groupby('date').agg({
            'total_sleep_min': 'sum',
            'deep': 'sum', 
            'rem': 'sum',
            'sleep_efficiency': 'mean'
        })
        
        # Calculate composite sleep quality score (0-100)
        nightly_metrics['sleep_quality'] = (
            np.clip(nightly_metrics['total_sleep_min'] / 450 * 40, 0, 40) +
            np.clip(nightly_metrics['deep'] / 90 * 30, 0, 30) +
            np.clip(nightly_metrics['rem'] / 90 * 15, 0, 15) +
            np.clip(nightly_metrics['sleep_efficiency'] / 100 * 15, 0, 15)
        )
        
        # Map back to hourly data
        for date, quality in nightly_metrics['sleep_quality'].items():
            self.df.loc[self.df['date'] == date, 'night_sleep_quality'] = quality
            
        print(f"Sleep quality score - Mean: {self.df['night_sleep_quality'].mean():.1f}")
    
    def _create_classification_targets(self):
        """Create different classification targets"""
        print("Creating classification targets...")
        
        # Target 1: Sleep Quality Categories (4-class)
        def categorize_sleep_quality(score):
            if score >= 80: return 'Excellent'
            elif score >= 70: return 'Good'
            elif score >= 60: return 'Fair'
            else: return 'Poor'
        
        self.df['sleep_category'] = self.df['night_sleep_quality'].apply(categorize_sleep_quality)
        
        # Target 2: Binary classification (Good vs Poor sleep)
        self.df['sleep_binary'] = (self.df['night_sleep_quality'] >= 70).astype(int)
        self.df['sleep_binary_label'] = self.df['sleep_binary'].map({1: 'Good', 0: 'Poor'})
        
        # Target 3: Sleep efficiency categories
        self.df['efficiency_category'] = pd.cut(
            self.df['sleep_efficiency'].fillna(0), 
            bins=[0, 70, 85, 100], 
            labels=['Low', 'Medium', 'High']
        )
        
        print("Classification targets created:")
        print(f"  Sleep Quality: {self.df['sleep_category'].value_counts().to_dict()}")
        print(f"  Binary Sleep: {self.df['sleep_binary_label'].value_counts().to_dict()}")
        
    def prepare_features(self):
        """Prepare feature matrix for modeling"""
        print(f"\nPreparing feature matrix with top {len(self.top_features)} features...")
        
        # Filter to evening hours for prediction
        evening_mask = self.df.index.hour.isin(range(18, 22))
        evening_df = self.df[evening_mask].copy()
        
        # Select available features
        available_features = [f for f in self.top_features if f in evening_df.columns]
        missing_features = [f for f in self.top_features if f not in evening_df.columns]
        
        if missing_features:
            print(f"  Missing features: {len(missing_features)}")
            print(f"  Using {len(available_features)} available features")
        
        # Create feature matrix
        X = evening_df[available_features]
        
        # Remove rows with missing targets
        valid_mask = (
            ~evening_df['sleep_category'].isna() & 
            (evening_df['night_sleep_quality'] > 0)
        )
        
        X = X[valid_mask]
        y_multi = evening_df.loc[valid_mask, 'sleep_category']
        y_binary = evening_df.loc[valid_mask, 'sleep_binary']
        
        # Handle missing values
        X_filled = X.fillna(X.mean()).fillna(0)
        
        # Remove any remaining infinite values
        X_filled = X_filled.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        print(f"Final dataset: {len(X_filled)} samples × {len(available_features)} features")
        print(f"Class distribution: {y_multi.value_counts().to_dict()}")
        
        self.X = X_filled
        self.y_multi = y_multi
        self.y_binary = y_binary
        self.feature_names = available_features
        
        return X_filled, y_multi, y_binary
    
    def train_classification_models(self):
        """Train multiple classification models"""
        print("\nTraining Classification Models...")
        print("="*50)
        
        # Define models - focusing on assignment requirements
        models = {
            'Decision Tree': DecisionTreeClassifier(
                max_depth=10, min_samples_split=20, random_state=42
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=42
            ),
            'SVM': SVC(
                kernel='rbf', C=1.0, random_state=42, probability=True
            ),
            'K-Nearest Neighbors': KNeighborsClassifier(
                n_neighbors=5, weights='distance'
            ),
            'Logistic Regression': LogisticRegression(
                random_state=42, max_iter=1000
            )
        }
        
        # Scale features for SVM and KNN
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, self.y_multi, test_size=0.2, random_state=42, stratify=self.y_multi
        )
        
        # Cross-validation setup
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        results = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
            
            # Train on full training set
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            results[name] = {
                'cv_accuracy_mean': cv_scores.mean(),
                'cv_accuracy_std': cv_scores.std(),
                'test_accuracy': accuracy,
                'test_precision': precision,
                'test_recall': recall,
                'test_f1': f1,
                'predictions': y_pred,
                'true_labels': y_test,
                'model': model
            }
            
            print(f"  CV Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
            print(f"  Test Accuracy: {accuracy:.3f}")
            print(f"  Test F1-Score: {f1:.3f}")
            
        self.classification_results = pd.DataFrame(results).T
        self.models = {name: results[name]['model'] for name in results}
        self.scaler = scaler
        self.X_test = X_test
        self.y_test = y_test
        
        return self.classification_results
    
    def perform_clustering_analysis(self):
        """Perform K-means clustering analysis"""
        print("\nPerforming Clustering Analysis...")
        print("="*40)
        
        # Use scaled features for clustering
        X_scaled = StandardScaler().fit_transform(self.X)
        
        # Try different numbers of clusters
        cluster_range = range(2, 8)
        silhouette_scores = []
        inertias = []
        
        for n_clusters in cluster_range:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X_scaled)
            
            silhouette_avg = silhouette_score(X_scaled, cluster_labels)
            silhouette_scores.append(silhouette_avg)
            inertias.append(kmeans.inertia_)
            
            print(f"  {n_clusters} clusters - Silhouette Score: {silhouette_avg:.3f}")
        
        # Choose optimal number of clusters
        optimal_clusters = cluster_range[np.argmax(silhouette_scores)]
        print(f"\nOptimal number of clusters: {optimal_clusters}")
        
        # Perform final clustering
        final_kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=10)
        cluster_labels = final_kmeans.fit_predict(X_scaled)
        
        # Analyze clusters
        cluster_df = self.X.copy()
        cluster_df['cluster'] = cluster_labels
        cluster_df['sleep_category'] = self.y_multi.values
        
        # Cluster characteristics
        cluster_stats = []
        for i in range(optimal_clusters):
            cluster_data = cluster_df[cluster_df['cluster'] == i]
            
            stats = {
                'cluster': i,
                'size': len(cluster_data),
                'sleep_quality_dist': cluster_data['sleep_category'].value_counts().to_dict(),
                'dominant_sleep_quality': cluster_data['sleep_category'].mode().iloc[0] if len(cluster_data) > 0 else 'Unknown'
            }
            cluster_stats.append(stats)
        
        self.cluster_results = {
            'optimal_clusters': optimal_clusters,
            'silhouette_scores': silhouette_scores,
            'inertias': inertias,
            'cluster_labels': cluster_labels,
            'cluster_stats': cluster_stats,
            'kmeans_model': final_kmeans
        }
        
        print("\nCluster Analysis:")
        for stats in cluster_stats:
            print(f"  Cluster {stats['cluster']}: {stats['size']} samples, "
                  f"Dominant: {stats['dominant_sleep_quality']}")
        
        return self.cluster_results
    
    def save_models(self):
        """Save all trained models and preprocessing objects"""
        print("\nSaving models and preprocessing objects...")
        
        import joblib
        import os
        
        # Create models directory
        if not os.path.exists('saved_models'):
            os.makedirs('saved_models')
        
        # Save all models
        for name, model in self.models.items():
            filename = f"saved_models/sleep_{name.lower().replace(' ', '_')}_model.pkl"
            joblib.dump(model, filename)
            print(f"  Saved: {filename}")
        
        # Save scaler and feature information
        joblib.dump(self.scaler, 'saved_models/feature_scaler.pkl')
        joblib.dump(self.feature_names, 'saved_models/feature_names.pkl')
        joblib.dump(self.cluster_results['kmeans_model'], 'saved_models/kmeans_model.pkl')
        
        # Save model metadata
        model_info = {
            'best_model': self.classification_results['test_f1'].idxmax(),
            'best_accuracy': self.classification_results['test_accuracy'].max(),
            'best_f1': self.classification_results['test_f1'].max(),
            'feature_count': len(self.feature_names),
            'classes': list(self.y_multi.unique()),
            'optimal_clusters': self.cluster_results['optimal_clusters']
        }
        
        import json
        with open('saved_models/model_info.json', 'w') as f:
            json.dump(model_info, f, indent=2)
        
        print("  Saved: model metadata and preprocessing objects")

    def create_comprehensive_visualizations(self):
        """Create comprehensive visualizations"""
        print("\nCreating visualizations...")
        
        # Create images directory
        import os
        if not os.path.exists('images'):
            os.makedirs('images')
            print("Created 'images/' directory")
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # Create multiple figure sets for comprehensive analysis
        self._create_performance_visualizations()
        self._create_detailed_analysis_visualizations()
        self._create_clustering_visualizations()
        self._create_feature_analysis_visualizations()
    
    def _create_performance_visualizations(self):
        """Create model performance visualizations"""
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Model Performance Comparison
        ax1 = plt.subplot(2, 4, 1)
        metrics = ['test_accuracy', 'test_precision', 'test_recall', 'test_f1']
        self.classification_results[metrics].plot(kind='bar', ax=ax1, colormap='viridis')
        ax1.set_title('Classification Model Performance', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Score')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        
        # 2. Cross-Validation Scores with Error Bars
        ax2 = plt.subplot(2, 4, 2)
        cv_means = self.classification_results['cv_accuracy_mean']
        cv_stds = self.classification_results['cv_accuracy_std']
        
        bars = ax2.bar(range(len(cv_means)), cv_means, yerr=cv_stds, 
                       capsize=5, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
        ax2.set_title('Cross-Validation Accuracy\n(with Standard Deviation)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('CV Accuracy')
        ax2.set_xticks(range(len(cv_means)))
        ax2.set_xticklabels([name.replace(' ', '\n') for name in cv_means.index], fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, mean, std) in enumerate(zip(bars, cv_means, cv_stds)):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.01,
                    f'{mean:.3f}', ha='center', va='bottom', fontsize=8)
        
        # 3. Confusion Matrix for Best Model
        ax3 = plt.subplot(2, 4, 3)
        best_model_name = self.classification_results['test_f1'].idxmax()
        best_result = self.classification_results.loc[best_model_name]
        
        cm = confusion_matrix(best_result['true_labels'], best_result['predictions'])
        sns.heatmap(cm, annot=True, fmt='d', ax=ax3, cmap='Blues', 
                   xticklabels=['Excellent', 'Fair', 'Good', 'Poor'],
                   yticklabels=['Excellent', 'Fair', 'Good', 'Poor'])
        ax3.set_title(f'Confusion Matrix\n{best_model_name}', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Predicted')
        ax3.set_ylabel('Actual')
        
        # 4. Model Accuracy Ranking
        ax4 = plt.subplot(2, 4, 4)
        accuracy_sorted = self.classification_results['test_accuracy'].sort_values(ascending=True)
        colors = ['red' if x < 0.6 else 'orange' if x < 0.7 else 'green' for x in accuracy_sorted]
        
        bars = ax4.barh(range(len(accuracy_sorted)), accuracy_sorted, color=colors, alpha=0.7)
        ax4.set_yticks(range(len(accuracy_sorted)))
        ax4.set_yticklabels(accuracy_sorted.index)
        ax4.set_title('Model Accuracy Ranking', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Test Accuracy')
        ax4.grid(True, alpha=0.3)
        
        # Add accuracy values
        for i, (bar, acc) in enumerate(zip(bars, accuracy_sorted)):
            ax4.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{acc:.3f}', va='center', fontsize=9)
        
        # 5. Class Distribution
        ax5 = plt.subplot(2, 4, 5)
        class_counts = self.y_multi.value_counts()
        colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
        wedges, texts, autotexts = ax5.pie(class_counts.values, labels=class_counts.index, 
                                          autopct='%1.1f%%', colors=colors, startangle=90)
        ax5.set_title('Sleep Quality Distribution', fontsize=12, fontweight='bold')
        
        # 6. Precision-Recall by Class
        ax6 = plt.subplot(2, 4, 6)
        
        # Calculate per-class metrics for best model
        from sklearn.metrics import precision_recall_fscore_support
        precision, recall, f1, support = precision_recall_fscore_support(
            best_result['true_labels'], best_result['predictions'], average=None
        )
        
        classes = ['Excellent', 'Fair', 'Good', 'Poor']
        x = np.arange(len(classes))
        width = 0.35
        
        bars1 = ax6.bar(x - width/2, precision, width, label='Precision', alpha=0.8)
        bars2 = ax6.bar(x + width/2, recall, width, label='Recall', alpha=0.8)
        
        ax6.set_xlabel('Sleep Quality Class')
        ax6.set_ylabel('Score')
        ax6.set_title('Precision & Recall by Class\n(Best Model)', fontsize=12, fontweight='bold')
        ax6.set_xticks(x)
        ax6.set_xticklabels(classes)
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # 7. Feature Importance (Top 15)
        ax7 = plt.subplot(2, 4, 7)
        rf_model = self.models['Random Forest']
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False).head(15)
        
        ax7.barh(range(len(feature_importance)), feature_importance['importance'], 
                color='forestgreen', alpha=0.7)
        ax7.set_yticks(range(len(feature_importance)))
        ax7.set_yticklabels([name.replace('_', '\n') for name in feature_importance['feature']], fontsize=8)
        ax7.set_title('Top 15 Feature Importance\n(Random Forest)', fontsize=12, fontweight='bold')
        ax7.set_xlabel('Importance')
        ax7.invert_yaxis()
        ax7.grid(True, alpha=0.3)
        
        # 8. Learning Curves Simulation
        ax8 = plt.subplot(2, 4, 8)
        
        # Simulate learning curves for best model
        train_sizes = np.linspace(0.1, 1.0, 10)
        train_scores = []
        val_scores = []
        
        best_model = self.models[best_model_name]
        X_scaled = self.scaler.transform(self.X)
        
        for size in train_sizes:
            n_samples = int(size * len(X_scaled))
            if n_samples < 10:
                continue
                
            # Simple train/val split for learning curve
            X_temp = X_scaled[:n_samples]
            y_temp = self.y_multi.iloc[:n_samples]
            
            X_train_temp, X_val_temp, y_train_temp, y_val_temp = train_test_split(
                X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp
            )
            
            temp_model = type(best_model)(**best_model.get_params())
            temp_model.fit(X_train_temp, y_train_temp)
            
            train_score = temp_model.score(X_train_temp, y_train_temp)
            val_score = temp_model.score(X_val_temp, y_val_temp)
            
            train_scores.append(train_score)
            val_scores.append(val_score)
        
        ax8.plot(train_sizes[-len(train_scores):], train_scores, 'o-', label='Training Score', color='blue')
        ax8.plot(train_sizes[-len(val_scores):], val_scores, 'o-', label='Validation Score', color='red')
        ax8.set_xlabel('Training Set Size (fraction)')
        ax8.set_ylabel('Accuracy')
        ax8.set_title(f'Learning Curves\n{best_model_name}', fontsize=12, fontweight='bold')
        ax8.legend()
        ax8.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('images/01_model_performance_overview.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Save individual performance charts
        self._save_individual_performance_charts()

    def _create_detailed_analysis_visualizations(self):
        
        # 1. Classification Model Performance
        ax1 = plt.subplot(3, 3, 1)
        metrics = ['test_accuracy', 'test_precision', 'test_recall', 'test_f1']
        self.classification_results[metrics].plot(kind='bar', ax=ax1)
        ax1.set_title('Classification Model Performance', fontsize=12)
        ax1.set_ylabel('Score')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
        
        # 2. Confusion Matrix for Best Model
        ax2 = plt.subplot(3, 3, 2)
        best_model_name = self.classification_results['test_f1'].idxmax()
        best_result = self.classification_results.loc[best_model_name]
        
        cm = confusion_matrix(best_result['true_labels'], best_result['predictions'])
        sns.heatmap(cm, annot=True, fmt='d', ax=ax2, cmap='Blues')
        ax2.set_title(f'Confusion Matrix - {best_model_name}', fontsize=12)
        ax2.set_xlabel('Predicted')
        ax2.set_ylabel('Actual')
        
        # 3. Sleep Quality Distribution
        ax3 = plt.subplot(3, 3, 3)
        self.y_multi.value_counts().plot(kind='pie', ax=ax3, autopct='%1.1f%%')
        ax3.set_title('Sleep Quality Distribution', fontsize=12)
        ax3.set_ylabel('')
        
        # 4. Clustering - Silhouette Scores
        ax4 = plt.subplot(3, 3, 4)
        cluster_range = range(2, len(self.cluster_results['silhouette_scores']) + 2)
        ax4.plot(cluster_range, self.cluster_results['silhouette_scores'], 'bo-')
        ax4.set_title('Clustering - Silhouette Analysis', fontsize=12)
        ax4.set_xlabel('Number of Clusters')
        ax4.set_ylabel('Silhouette Score')
        ax4.grid(True)
        
        # 5. Clustering - Elbow Method
        ax5 = plt.subplot(3, 3, 5)
        ax5.plot(cluster_range, self.cluster_results['inertias'], 'ro-')
        ax5.set_title('Clustering - Elbow Method', fontsize=12)
        ax5.set_xlabel('Number of Clusters')
        ax5.set_ylabel('Inertia')
        ax5.grid(True)
        
        # 6. PCA Visualization of Clusters
        ax6 = plt.subplot(3, 3, 6)
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(StandardScaler().fit_transform(self.X))
        
        scatter = ax6.scatter(X_pca[:, 0], X_pca[:, 1], 
                            c=self.cluster_results['cluster_labels'], 
                            cmap='viridis', alpha=0.6)
        ax6.set_title('Clusters in PCA Space', fontsize=12)
        ax6.set_xlabel(f'First PC ({pca.explained_variance_ratio_[0]:.1%} variance)')
        ax6.set_ylabel(f'Second PC ({pca.explained_variance_ratio_[1]:.1%} variance)')
        plt.colorbar(scatter, ax=ax6)
        
        # 7. Feature Importance (from Random Forest)
        ax7 = plt.subplot(3, 3, 7)
        rf_model = self.models['Random Forest']
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False).head(15)
        
        ax7.barh(range(len(feature_importance)), feature_importance['importance'])
        ax7.set_yticks(range(len(feature_importance)))
        ax7.set_yticklabels(feature_importance['feature'])
        ax7.set_title('Top 15 Feature Importance (Random Forest)', fontsize=12)
        ax7.set_xlabel('Importance')
        ax7.invert_yaxis()
        
        # 8. Cross-Validation Scores
        ax8 = plt.subplot(3, 3, 8)
        cv_means = self.classification_results['cv_accuracy_mean']
        cv_stds = self.classification_results['cv_accuracy_std']
        
        ax8.bar(range(len(cv_means)), cv_means, yerr=cv_stds, capsize=5)
        ax8.set_title('Cross-Validation Accuracy', fontsize=12)
        ax8.set_ylabel('Accuracy')
        ax8.set_xticks(range(len(cv_means)))
        ax8.set_xticklabels(cv_means.index, rotation=45)
        
        # 9. Sleep Quality vs Clusters Heatmap
        ax9 = plt.subplot(3, 3, 9)
        
        # Create cluster-sleep quality cross-tabulation
        cluster_sleep_crosstab = pd.crosstab(
            self.cluster_results['cluster_labels'], 
            self.y_multi, 
            normalize='index'
        )
        
        sns.heatmap(cluster_sleep_crosstab, annot=True, fmt='.2f', ax=ax9, cmap='YlOrRd')
        ax9.set_title('Sleep Quality Distribution by Cluster', fontsize=12)
        ax9.set_xlabel('Sleep Quality')
        ax9.set_ylabel('Cluster')
        
        plt.tight_layout()
        plt.savefig('sleep_quality_classification_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_comprehensive_report(self):
        """Generate comprehensive analysis report"""
        print("\n" + "="*80)
        print("SLEEP QUALITY CLASSIFICATION & CLUSTERING ANALYSIS - FINAL REPORT")
        print("="*80)
        
        print("\n1. DATASET SUMMARY")
        print("-"*50)
        print(f"Total hourly records: {len(self.df):,}")
        print(f"Evening prediction samples: {len(self.X):,}")
        print(f"Features used: {len(self.feature_names)}")
        print(f"Class distribution: {dict(self.y_multi.value_counts())}")
        
        print("\n2. CLASSIFICATION RESULTS")
        print("-"*50)
        print("Model Performance Summary:")
        print(self.classification_results[['cv_accuracy_mean', 'test_accuracy', 'test_f1']].round(3))
        
        best_model = self.classification_results['test_f1'].idxmax()
        best_metrics = self.classification_results.loc[best_model]
        
        print(f"\nBest Model: {best_model}")
        print(f"  Test Accuracy: {best_metrics['test_accuracy']:.3f}")
        print(f"  Test F1-Score: {best_metrics['test_f1']:.3f}")
        print(f"  Test Precision: {best_metrics['test_precision']:.3f}")
        print(f"  Test Recall: {best_metrics['test_recall']:.3f}")
        
        print("\n3. CLUSTERING ANALYSIS")
        print("-"*50)
        print(f"Optimal clusters: {self.cluster_results['optimal_clusters']}")
        print(f"Best silhouette score: {max(self.cluster_results['silhouette_scores']):.3f}")
        
        print("\nCluster Characteristics:")
        for stats in self.cluster_results['cluster_stats']:
            print(f"  Cluster {stats['cluster']}: {stats['size']} samples")
            print(f"    Dominant sleep quality: {stats['dominant_sleep_quality']}")
            print(f"    Distribution: {stats['sleep_quality_dist']}")
        
        print("\n4. KEY INSIGHTS")
        print("-"*50)
        
        # Model performance insights
        best_accuracy = self.classification_results['test_accuracy'].max()
        if best_accuracy > 0.7:
            print(f"✓ Strong classification performance ({best_accuracy:.1%} accuracy)")
        elif best_accuracy > 0.5:
            print(f"○ Moderate classification performance ({best_accuracy:.1%} accuracy)")
        else:
            print(f"✗ Challenging classification task ({best_accuracy:.1%} accuracy)")
        
        # Feature insights
        rf_model = self.models['Random Forest']
        top_feature = self.feature_names[np.argmax(rf_model.feature_importances_)]
        print(f"• Most important feature: {top_feature}")
        
        # Clustering insights
        n_clusters = self.cluster_results['optimal_clusters']
        print(f"• Data naturally groups into {n_clusters} distinct sleep patterns")
        
        # Class balance
        minority_class_pct = self.y_multi.value_counts().min() / len(self.y_multi)
        if minority_class_pct < 0.1:
            print(f"⚠ Highly imbalanced dataset (smallest class: {minority_class_pct:.1%})")
        
        print("\n5. PRACTICAL APPLICATIONS")
        print("-"*50)
        print("• Sleep quality prediction for health monitoring")
        print("• Personalized sleep improvement recommendations")
        print("• Population health analysis and intervention planning")
        print("• Early detection of sleep disorders")
        
        print("\n6. TECHNICAL QUALITY")
        print("-"*50)
        print("✓ Applied 5 different classification algorithms")
        print("✓ Performed K-means clustering analysis")
        print("✓ Used proper cross-validation and train/test split")
        print("✓ Comprehensive feature selection based on importance")
        print("✓ Multiple evaluation metrics and visualizations")
        
        print("\n" + "="*80)

    def _create_detailed_analysis_visualizations(self):
        """Create detailed analysis visualizations"""
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Feature Correlation Heatmap
        ax1 = plt.subplot(3, 3, 1)
        
        # Select top 20 features for correlation
        rf_model = self.models['Random Forest']
        top_20_features = pd.DataFrame({
            'feature': self.feature_names,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False).head(20)['feature'].tolist()
        
        corr_matrix = self.X[top_20_features].corr()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm', center=0,
                   square=True, linewidths=.5, cbar_kws={"shrink": .8}, ax=ax1)
        ax1.set_title('Feature Correlation Matrix\n(Top 20 Features)', fontsize=12, fontweight='bold')
        
        # 2. Sleep Quality by Hour of Day
        ax2 = plt.subplot(3, 3, 2)
        
        # Create hour-based analysis
        hourly_sleep = pd.DataFrame({
            'hour': self.df.index.hour,
            'sleep_quality': self.df['night_sleep_quality']
        })
        hourly_sleep = hourly_sleep[hourly_sleep['sleep_quality'] > 0]
        
        hourly_avg = hourly_sleep.groupby('hour')['sleep_quality'].agg(['mean', 'std']).reset_index()
        
        ax2.plot(hourly_avg['hour'], hourly_avg['mean'], 'o-', linewidth=2, markersize=6)
        ax2.fill_between(hourly_avg['hour'], 
                        hourly_avg['mean'] - hourly_avg['std'],
                        hourly_avg['mean'] + hourly_avg['std'], 
                        alpha=0.3)
        ax2.set_xlabel('Hour of Day')
        ax2.set_ylabel('Average Sleep Quality Score')
        ax2.set_title('Sleep Quality Patterns\nby Hour of Day', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(range(0, 24, 4))
        
        # 3. Model Comparison Radar Chart
        ax3 = plt.subplot(3, 3, 3, projection='polar')
        
        metrics = ['test_accuracy', 'test_precision', 'test_recall', 'test_f1']
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False)
        angles = np.concatenate((angles, [angles[0]]))
        
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        for i, (model_name, row) in enumerate(self.classification_results.iterrows()):
            values = [row[metric] for metric in metrics]
            values += [values[0]]  # Complete the circle
            
            ax3.plot(angles, values, 'o-', linewidth=2, label=model_name, color=colors[i])
            ax3.fill(angles, values, alpha=0.1, color=colors[i])
        
        ax3.set_xticks(angles[:-1])
        ax3.set_xticklabels(['Accuracy', 'Precision', 'Recall', 'F1-Score'])
        ax3.set_ylim(0, 1)
        ax3.set_title('Model Performance\nRadar Chart', fontsize=12, fontweight='bold', pad=20)
        ax3.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        # 4. Feature Distribution by Sleep Quality
        ax4 = plt.subplot(3, 3, 4)
        
        # Select most important feature for distribution analysis
        most_important_feature = self.feature_names[np.argmax(rf_model.feature_importances_)]
        
        for category in ['Excellent', 'Good', 'Fair', 'Poor']:
            if category in self.y_multi.values:
                mask = self.y_multi == category
                data = self.X.loc[mask, most_important_feature].dropna()
                if len(data) > 0:
                    ax4.hist(data, alpha=0.6, label=category, bins=20)
        
        ax4.set_xlabel(most_important_feature.replace('_', ' ').title())
        ax4.set_ylabel('Frequency')
        ax4.set_title(f'Distribution of Most Important Feature\nby Sleep Quality', fontsize=12, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Classification Report Heatmap
        ax5 = plt.subplot(3, 3, 5)
        
        best_model_name = self.classification_results['test_f1'].idxmax()
        best_result = self.classification_results.loc[best_model_name]
        
        from sklearn.metrics import classification_report
        report = classification_report(best_result['true_labels'], best_result['predictions'], 
                                     output_dict=True, zero_division=0)
        
        # Convert to DataFrame for heatmap
        report_df = pd.DataFrame(report).iloc[:-1, :-3].T  # Remove support and avg rows
        
        sns.heatmap(report_df.astype(float), annot=True, fmt='.3f', cmap='YlOrRd', ax=ax5)
        ax5.set_title(f'Classification Report Heatmap\n{best_model_name}', fontsize=12, fontweight='bold')
        ax5.set_ylabel('Sleep Quality Class')
        ax5.set_xlabel('Metrics')
        
        # 6. Air Quality vs Sleep Quality Scatter
        ax6 = plt.subplot(3, 3, 6)
        
        if 'pm25' in self.X.columns and 'o3_prev_month_mean' in self.X.columns:
            scatter_data = pd.DataFrame({
                'pm25': self.X['pm25'] if 'pm25' in self.X.columns else self.X[self.X.columns[0]],
                'o3': self.X['o3_prev_month_mean'] if 'o3_prev_month_mean' in self.X.columns else self.X[self.X.columns[1]],
                'sleep_quality': self.y_multi
            }).dropna()
            
            colors = {'Excellent': 'green', 'Good': 'blue', 'Fair': 'orange', 'Poor': 'red'}
            
            for category in scatter_data['sleep_quality'].unique():
                data = scatter_data[scatter_data['sleep_quality'] == category]
                ax6.scatter(data['pm25'], data['o3'], 
                           c=colors.get(category, 'gray'), 
                           label=category, alpha=0.6, s=30)
            
            ax6.set_xlabel('PM2.5 Level')
            ax6.set_ylabel('O3 Previous Month Mean')
            ax6.set_title('Air Quality vs Sleep Quality', fontsize=12, fontweight='bold')
            ax6.legend()
            ax6.grid(True, alpha=0.3)
        
        # 7. Model Training Time Comparison (simulated)
        ax7 = plt.subplot(3, 3, 7)
        
        # Simulate training times (replace with actual if measured)
        training_times = {
            'Decision Tree': 0.1,
            'Random Forest': 2.5,
            'SVM': 5.2,
            'K-Nearest Neighbors': 0.05,
            'Logistic Regression': 0.3
        }
        
        models = list(training_times.keys())
        times = list(training_times.values())
        colors = ['skyblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink']
        
        bars = ax7.bar(models, times, color=colors, alpha=0.8)
        ax7.set_ylabel('Training Time (seconds)')
        ax7.set_title('Model Training Time Comparison\n(Simulated)', fontsize=12, fontweight='bold')
        ax7.set_xticklabels(models, rotation=45, ha='right')
        ax7.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, time in zip(bars, times):
            ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{time:.1f}s', ha='center', va='bottom')
        
        # 8. Sleep Quality Trends Over Time
        ax8 = plt.subplot(3, 3, 8)
        
        # Group by month to show trends
        monthly_sleep = self.df.groupby(self.df.index.to_period('M'))['night_sleep_quality'].mean()
        monthly_sleep = monthly_sleep[monthly_sleep > 0]
        
        if len(monthly_sleep) > 1:
            ax8.plot(monthly_sleep.index.to_timestamp(), monthly_sleep.values, 'o-', linewidth=2)
            ax8.set_xlabel('Month')
            ax8.set_ylabel('Average Sleep Quality')
            ax8.set_title('Sleep Quality Trends\nOver Time', fontsize=12, fontweight='bold')
            ax8.grid(True, alpha=0.3)
            
            # Rotate x-axis labels
            for tick in ax8.get_xticklabels():
                tick.set_rotation(45)
        
        # 9. Feature Importance Categories
        ax9 = plt.subplot(3, 3, 9)
        
        # Categorize features
        feature_categories = {
            'Heart Rate': [],
            'Air Quality': [],
            'Stress': [],
            'Blood Oxygen': [],
            'Temporal': [],
            'Other': []
        }
        
        for i, feature in enumerate(self.feature_names):
            importance = rf_model.feature_importances_[i]
            
            if 'hr_' in feature:
                feature_categories['Heart Rate'].append(importance)
            elif any(x in feature for x in ['pm25', 'aqi', 'o3']):
                feature_categories['Air Quality'].append(importance)
            elif 'stress' in feature:
                feature_categories['Stress'].append(importance)
            elif 'spo2' in feature:
                feature_categories['Blood Oxygen'].append(importance)
            elif feature in ['hour', 'day_of_week', 'month']:
                feature_categories['Temporal'].append(importance)
            else:
                feature_categories['Other'].append(importance)
        
        category_totals = {k: sum(v) for k, v in feature_categories.items() if v}
        
        if category_totals:
            categories = list(category_totals.keys())
            values = list(category_totals.values())
            colors = plt.cm.Set3(np.linspace(0, 1, len(categories)))
            
            ax9.pie(values, labels=categories, autopct='%1.1f%%', colors=colors, startangle=90)
            ax9.set_title('Feature Importance\nby Category', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('images/02_detailed_analysis_overview.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Save individual detailed analysis charts
        self._save_individual_detailed_charts()

    def _create_clustering_visualizations(self):
        """Create clustering-specific visualizations"""
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Silhouette Analysis
        ax1 = plt.subplot(2, 4, 1)
        cluster_range = range(2, len(self.cluster_results['silhouette_scores']) + 2)
        ax1.plot(cluster_range, self.cluster_results['silhouette_scores'], 'bo-', linewidth=2, markersize=8)
        ax1.set_title('Silhouette Analysis', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Number of Clusters')
        ax1.set_ylabel('Silhouette Score')
        ax1.grid(True, alpha=0.3)
        
        # Highlight optimal point
        optimal_idx = np.argmax(self.cluster_results['silhouette_scores'])
        ax1.plot(cluster_range[optimal_idx], self.cluster_results['silhouette_scores'][optimal_idx], 
                'ro', markersize=12, label=f'Optimal: {cluster_range[optimal_idx]} clusters')
        ax1.legend()
        
        # 2. Elbow Method
        ax2 = plt.subplot(2, 4, 2)
        ax2.plot(cluster_range, self.cluster_results['inertias'], 'ro-', linewidth=2, markersize=8)
        ax2.set_title('Elbow Method', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Number of Clusters')
        ax2.set_ylabel('Inertia')
        ax2.grid(True, alpha=0.3)
        
        # 3. PCA Visualization of Clusters
        ax3 = plt.subplot(2, 4, 3)
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(StandardScaler().fit_transform(self.X))
        
        scatter = ax3.scatter(X_pca[:, 0], X_pca[:, 1], 
                            c=self.cluster_results['cluster_labels'], 
                            cmap='viridis', alpha=0.6, s=50)
        ax3.set_title('Clusters in PCA Space', fontsize=12, fontweight='bold')
        ax3.set_xlabel(f'First PC ({pca.explained_variance_ratio_[0]:.1%} variance)')
        ax3.set_ylabel(f'Second PC ({pca.explained_variance_ratio_[1]:.1%} variance)')
        plt.colorbar(scatter, ax=ax3)
        ax3.grid(True, alpha=0.3)
        
        # 4. Cluster Sizes
        ax4 = plt.subplot(2, 4, 4)
        cluster_sizes = [stats['size'] for stats in self.cluster_results['cluster_stats']]
        cluster_labels = [f"Cluster {stats['cluster']}" for stats in self.cluster_results['cluster_stats']]
        
        bars = ax4.bar(cluster_labels, cluster_sizes, color=['lightblue', 'lightgreen', 'lightcoral'][:len(cluster_sizes)])
        ax4.set_title('Cluster Sizes', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Number of Samples')
        ax4.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, size in zip(bars, cluster_sizes):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                    str(size), ha='center', va='bottom', fontweight='bold')
        
        # 5. Sleep Quality Distribution by Cluster (Detailed)
        ax5 = plt.subplot(2, 4, 5)
        
        cluster_sleep_data = []
        for stats in self.cluster_results['cluster_stats']:
            for quality, count in stats['sleep_quality_dist'].items():
                cluster_sleep_data.append({
                    'Cluster': f"Cluster {stats['cluster']}",
                    'Sleep Quality': quality,
                    'Count': count,
                    'Percentage': count / stats['size'] * 100
                })
        
        cluster_sleep_df = pd.DataFrame(cluster_sleep_data)
        
        # Create stacked bar chart
        pivot_df = cluster_sleep_df.pivot(index='Cluster', columns='Sleep Quality', values='Percentage').fillna(0)
        
        colors = {'Excellent': 'green', 'Good': 'blue', 'Fair': 'orange', 'Poor': 'red'}
        pivot_df.plot(kind='bar', stacked=True, ax=ax5, 
                     color=[colors.get(col, 'gray') for col in pivot_df.columns])
        ax5.set_title('Sleep Quality Distribution\nby Cluster (%)', fontsize=12, fontweight='bold')
        ax5.set_ylabel('Percentage')
        ax5.set_xlabel('Cluster')
        ax5.legend(title='Sleep Quality', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax5.set_xticklabels(ax5.get_xticklabels(), rotation=0)
        
        # 6. Cluster Centroids Heatmap
        ax6 = plt.subplot(2, 4, 6)
        rf_model = self.models['Random Forest']

        # Get cluster centroids
        kmeans_model = self.cluster_results['kmeans_model']
        centroids = kmeans_model.cluster_centers_
        
        # Select top 20 features for visualization
        top_20_indices = np.argsort(rf_model.feature_importances_)[-20:]
        top_20_names = [self.feature_names[i] for i in top_20_indices]
        
        centroids_df = pd.DataFrame(centroids[:, top_20_indices], 
                                   columns=[name.replace('_', '\n') for name in top_20_names],
                                   index=[f'Cluster {i}' for i in range(len(centroids))])
        
        sns.heatmap(centroids_df, annot=False, cmap='RdYlBu_r', center=0, ax=ax6)
        ax6.set_title('Cluster Centroids\n(Top 20 Features)', fontsize=12, fontweight='bold')
        ax6.set_xlabel('Features')
        ax6.set_ylabel('Clusters')
        
        # 7. 3D PCA Visualization
        ax7 = plt.subplot(2, 4, 7, projection='3d')
        
        pca_3d = PCA(n_components=3, random_state=42)
        X_pca_3d = pca_3d.fit_transform(StandardScaler().fit_transform(self.X))
        
        scatter = ax7.scatter(X_pca_3d[:, 0], X_pca_3d[:, 1], X_pca_3d[:, 2],
                            c=self.cluster_results['cluster_labels'], 
                            cmap='viridis', alpha=0.6, s=30)
        ax7.set_title('3D PCA Clusters', fontsize=12, fontweight='bold')
        ax7.set_xlabel(f'PC1 ({pca_3d.explained_variance_ratio_[0]:.1%})')
        ax7.set_ylabel(f'PC2 ({pca_3d.explained_variance_ratio_[1]:.1%})')
        ax7.set_zlabel(f'PC3 ({pca_3d.explained_variance_ratio_[2]:.1%})')
        
        # 8. Cluster Quality Metrics
        ax8 = plt.subplot(2, 4, 8)
        
        # Calculate additional cluster metrics
        from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
        
        X_scaled = StandardScaler().fit_transform(self.X)
        cluster_labels = self.cluster_results['cluster_labels']
        
        silhouette = silhouette_score(X_scaled, cluster_labels)
        calinski = calinski_harabasz_score(X_scaled, cluster_labels)
        davies_bouldin = davies_bouldin_score(X_scaled, cluster_labels)
        
        metrics = ['Silhouette\nScore', 'Calinski-Harabasz\nIndex', 'Davies-Bouldin\nIndex']
        values = [silhouette, calinski/100, 1/davies_bouldin]  # Normalize for visualization
        colors = ['green' if v > 0.5 else 'orange' if v > 0.2 else 'red' for v in [silhouette, calinski/1000, 1/davies_bouldin]]
        
        bars = ax8.bar(metrics, values, color=colors, alpha=0.7)
        ax8.set_title('Clustering Quality Metrics', fontsize=12, fontweight='bold')
        ax8.set_ylabel('Normalized Score')
        ax8.grid(True, alpha=0.3)
        
        # Add actual values as text
        actual_values = [f'{silhouette:.3f}', f'{calinski:.1f}', f'{davies_bouldin:.3f}']
        for bar, val in zip(bars, actual_values):
            ax8.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    val, ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('images/03_clustering_analysis_overview.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Save individual clustering charts
        self._save_individual_clustering_charts()

    def _create_feature_analysis_visualizations(self):
        """Create feature-specific analysis visualizations"""
        fig = plt.figure(figsize=(20, 12))
        
        # Get Random Forest model for feature importance
        rf_model = self.models['Random Forest']
        
        # 1. Feature Importance Distribution
        ax1 = plt.subplot(2, 4, 1)
        
        importances = rf_model.feature_importances_
        ax1.hist(importances, bins=20, edgecolor='black', alpha=0.7, color='skyblue')
        ax1.axvline(np.mean(importances), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(importances):.4f}')
        ax1.set_xlabel('Feature Importance')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Feature Importance Distribution', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Top vs Bottom Features Comparison
        ax2 = plt.subplot(2, 4, 2)
        
        feature_importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        top_5 = feature_importance_df.head(5)
        bottom_5 = feature_importance_df.tail(5)
        
        y_pos_top = np.arange(len(top_5))
        y_pos_bottom = np.arange(len(bottom_5)) - len(bottom_5) - 1
        
        ax2.barh(y_pos_top, top_5['importance'], color='green', alpha=0.7, label='Top 5')
        ax2.barh(y_pos_bottom, bottom_5['importance'], color='red', alpha=0.7, label='Bottom 5')
        
        # Set labels
        all_labels = list(top_5['feature']) + [''] * 2 + list(bottom_5['feature'])
        all_positions = list(y_pos_top) + [-5.5, -6] + list(y_pos_bottom)
        
        ax2.set_yticks(all_positions)
        ax2.set_yticklabels([name.replace('_', '\n') for name in all_labels], fontsize=8)
        ax2.set_xlabel('Importance')
        ax2.set_title('Top vs Bottom Features', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Feature Categories Pie Chart
        ax3 = plt.subplot(2, 4, 3)
        
        categories = {
            'Heart Rate': 0,
            'Air Quality': 0,
            'Stress': 0,
            'Blood Oxygen': 0,
            'Temporal': 0,
            'Sleep': 0,
            'Other': 0
        }
        
        for feature in self.feature_names:
            if 'hr_' in feature:
                categories['Heart Rate'] += 1
            elif any(x in feature for x in ['pm25', 'aqi', 'o3']):
                categories['Air Quality'] += 1
            elif 'stress' in feature:
                categories['Stress'] += 1
            elif 'spo2' in feature:
                categories['Blood Oxygen'] += 1
            elif feature in ['hour', 'day_of_week', 'month']:
                categories['Temporal'] += 1
            elif any(x in feature for x in ['sleep', 'deep', 'rem', 'light']):
                categories['Sleep'] += 1
            else:
                categories['Other'] += 1
        
        # Remove empty categories
        categories = {k: v for k, v in categories.items() if v > 0}
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(categories)))
        ax3.pie(categories.values(), labels=categories.keys(), autopct='%1.1f%%', 
               colors=colors, startangle=90)
        ax3.set_title('Feature Distribution\nby Category', fontsize=12, fontweight='bold')
        
        # 4. Cumulative Feature Importance
        ax4 = plt.subplot(2, 4, 4)
        
        sorted_importances = np.sort(importances)[::-1]
        cumulative_importance = np.cumsum(sorted_importances)
        
        ax4.plot(range(1, len(cumulative_importance) + 1), cumulative_importance, 'b-', linewidth=2)
        ax4.axhline(y=0.8, color='r', linestyle='--', label='80% threshold')
        ax4.axhline(y=0.9, color='orange', linestyle='--', label='90% threshold')
        
        ax4.set_xlabel('Number of Features')
        ax4.set_ylabel('Cumulative Importance')
        ax4.set_title('Cumulative Feature Importance', fontsize=12, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Find where we reach 80% and 90%
        idx_80 = np.where(cumulative_importance >= 0.8)[0]
        idx_90 = np.where(cumulative_importance >= 0.9)[0]
        
        if len(idx_80) > 0:
            ax4.axvline(x=idx_80[0]+1, color='r', linestyle=':', alpha=0.7)
            ax4.text(idx_80[0]+1, 0.4, f'{idx_80[0]+1} features\nfor 80%', 
                    ha='center', va='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # 5. Heart Rate Features Analysis
        ax5 = plt.subplot(2, 4, 5)
        
        hr_features = [(i, name, imp) for i, (name, imp) in enumerate(zip(self.feature_names, importances)) if 'hr_' in name]
        
        if hr_features:
            hr_names = [name.replace('hr_', '').replace('_', '\n') for _, name, _ in hr_features]
            hr_importances = [imp for _, _, imp in hr_features]
            
            # Sort by importance
            sorted_hr = sorted(zip(hr_names, hr_importances), key=lambda x: x[1], reverse=True)
            hr_names_sorted = [x[0] for x in sorted_hr]
            hr_imp_sorted = [x[1] for x in sorted_hr]
            
            ax5.barh(range(len(hr_names_sorted)), hr_imp_sorted, color='lightcoral', alpha=0.8)
            ax5.set_yticks(range(len(hr_names_sorted)))
            ax5.set_yticklabels(hr_names_sorted, fontsize=8)
            ax5.set_xlabel('Importance')
            ax5.set_title('Heart Rate Features\nImportance', fontsize=12, fontweight='bold')
            ax5.grid(True, alpha=0.3)
        
        # 6. Air Quality Features Analysis
        ax6 = plt.subplot(2, 4, 6)
        
        aq_features = [(i, name, imp) for i, (name, imp) in enumerate(zip(self.feature_names, importances)) 
                      if any(x in name for x in ['pm25', 'aqi', 'o3'])]
        
        if aq_features:
            aq_names = [name.replace('_', '\n') for _, name, _ in aq_features]
            aq_importances = [imp for _, _, imp in aq_features]
            
            # Sort by importance
            sorted_aq = sorted(zip(aq_names, aq_importances), key=lambda x: x[1], reverse=True)
            aq_names_sorted = [x[0] for x in sorted_aq]
            aq_imp_sorted = [x[1] for x in sorted_aq]
            
            ax6.barh(range(len(aq_names_sorted)), aq_imp_sorted, color='lightgreen', alpha=0.8)
            ax6.set_yticks(range(len(aq_names_sorted)))
            ax6.set_yticklabels(aq_names_sorted, fontsize=8)
            ax6.set_xlabel('Importance')
            ax6.set_title('Air Quality Features\nImportance', fontsize=12, fontweight='bold')
            ax6.grid(True, alpha=0.3)
        
        # 7. Feature Selection Impact
        ax7 = plt.subplot(2, 4, 7)
        
        # Simulate performance with different numbers of features
        feature_counts = [5, 10, 15, 20, 25, 30, 35, 40, 45]
        simulated_scores = []
        
        for n_features in feature_counts:
            if n_features <= len(self.feature_names):
                # Use top n features
                top_n_indices = np.argsort(importances)[-n_features:]
                X_subset = self.X.iloc[:, top_n_indices]
                
                # Quick cross-validation
                from sklearn.model_selection import cross_val_score
                temp_model = RandomForestClassifier(n_estimators=50, random_state=42)
                scores = cross_val_score(temp_model, X_subset, self.y_multi, cv=3, scoring='accuracy')
                simulated_scores.append(scores.mean())
            else:
                simulated_scores.append(simulated_scores[-1])  # Use last score
        
        ax7.plot(feature_counts, simulated_scores, 'o-', linewidth=2, markersize=6, color='purple')
        ax7.set_xlabel('Number of Features')
        ax7.set_ylabel('Cross-Validation Accuracy')
        ax7.set_title('Feature Selection Impact\non Model Performance', fontsize=12, fontweight='bold')
        ax7.grid(True, alpha=0.3)
        
        # Highlight current number of features
        current_features = len(self.feature_names)
        if current_features in feature_counts:
            idx = feature_counts.index(current_features)
            ax7.plot(current_features, simulated_scores[idx], 'ro', markersize=10, 
                    label=f'Current: {current_features} features')
            ax7.legend()
        
        # 8. Model Stability Analysis
        ax8 = plt.subplot(2, 4, 8)
        
        # Compare feature importance across different random states
        stability_scores = []
        n_trials = 5
        
        for trial in range(n_trials):
            temp_model = RandomForestClassifier(n_estimators=100, random_state=trial+42)
            temp_model.fit(self.scaler.transform(self.X), self.y_multi)
            stability_scores.append(temp_model.feature_importances_)
        
        # Calculate feature importance variance
        stability_scores = np.array(stability_scores)
        feature_variance = np.var(stability_scores, axis=0)
        
        ax8.scatter(importances, feature_variance, alpha=0.6, s=50)
        ax8.set_xlabel('Mean Feature Importance')
        ax8.set_ylabel('Feature Importance Variance')
        ax8.set_title('Feature Importance Stability\n(Lower variance = more stable)', fontsize=12, fontweight='bold')
        ax8.grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(importances, feature_variance, 1)
        p = np.poly1d(z)
        ax8.plot(importances, p(importances), "r--", alpha=0.8)
        
        plt.tight_layout()
        plt.savefig('images/04_feature_analysis_overview.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Save individual feature analysis charts
        self._save_individual_feature_charts()
    
    def _save_individual_performance_charts(self):
        """Save individual performance charts"""
        print("Saving individual performance charts...")
        
        # 1. Model Performance Comparison Bar Chart
        plt.figure(figsize=(12, 8))
        metrics = ['test_accuracy', 'test_precision', 'test_recall', 'test_f1']
        self.classification_results[metrics].plot(kind='bar', colormap='viridis')
        plt.title('Classification Model Performance Comparison', fontsize=16, fontweight='bold')
        plt.ylabel('Score', fontsize=12)
        plt.xlabel('Models', fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('images/05_model_performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Cross-Validation Scores
        plt.figure(figsize=(10, 6))
        cv_means = self.classification_results['cv_accuracy_mean']
        cv_stds = self.classification_results['cv_accuracy_std']
        
        bars = plt.bar(range(len(cv_means)), cv_means, yerr=cv_stds, 
                       capsize=5, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
        plt.title('Cross-Validation Accuracy with Standard Deviation', fontsize=16, fontweight='bold')
        plt.ylabel('CV Accuracy', fontsize=12)
        plt.xlabel('Models', fontsize=12)
        plt.xticks(range(len(cv_means)), [name.replace(' ', '\n') for name in cv_means.index])
        plt.grid(True, alpha=0.3)
        
        # Add value labels
        for i, (bar, mean, std) in enumerate(zip(bars, cv_means, cv_stds)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.01,
                    f'{mean:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('images/06_cross_validation_scores.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Confusion Matrix for Best Model
        plt.figure(figsize=(8, 6))
        best_model_name = self.classification_results['test_f1'].idxmax()
        best_result = self.classification_results.loc[best_model_name]
        
        cm = confusion_matrix(best_result['true_labels'], best_result['predictions'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Excellent', 'Fair', 'Good', 'Poor'],
                   yticklabels=['Excellent', 'Fair', 'Good', 'Poor'])
        plt.title(f'Confusion Matrix - {best_model_name}', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted', fontsize=12)
        plt.ylabel('Actual', fontsize=12)
        plt.tight_layout()
        plt.savefig('images/07_confusion_matrix_best_model.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Model Accuracy Ranking
        plt.figure(figsize=(10, 6))
        accuracy_sorted = self.classification_results['test_accuracy'].sort_values(ascending=True)
        colors = ['red' if x < 0.6 else 'orange' if x < 0.7 else 'green' for x in accuracy_sorted]
        
        bars = plt.barh(range(len(accuracy_sorted)), accuracy_sorted, color=colors, alpha=0.7)
        plt.yticks(range(len(accuracy_sorted)), accuracy_sorted.index)
        plt.title('Model Accuracy Ranking', fontsize=16, fontweight='bold')
        plt.xlabel('Test Accuracy', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Add accuracy values
        for i, (bar, acc) in enumerate(zip(bars, accuracy_sorted)):
            plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{acc:.3f}', va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('images/08_model_accuracy_ranking.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ Individual performance charts saved")

    def _save_individual_detailed_charts(self):
        """Save individual detailed analysis charts"""
        print("Saving individual detailed analysis charts...")
        
        # 1. Feature Correlation Heatmap
        plt.figure(figsize=(12, 10))
        rf_model = self.models['Random Forest']
        top_20_features = pd.DataFrame({
            'feature': self.feature_names,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False).head(20)['feature'].tolist()
        
        corr_matrix = self.X[top_20_features].corr()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm', center=0,
                   square=True, linewidths=.5, cbar_kws={"shrink": .8})
        plt.title('Feature Correlation Matrix (Top 20 Features)', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('images/09_feature_correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Sleep Quality by Hour of Day
        plt.figure(figsize=(12, 6))
        hourly_sleep = pd.DataFrame({
            'hour': self.df.index.hour,
            'sleep_quality': self.df['night_sleep_quality']
        })
        hourly_sleep = hourly_sleep[hourly_sleep['sleep_quality'] > 0]
        hourly_avg = hourly_sleep.groupby('hour')['sleep_quality'].agg(['mean', 'std']).reset_index()
        
        plt.plot(hourly_avg['hour'], hourly_avg['mean'], 'o-', linewidth=3, markersize=8, color='#2c3e50')
        plt.fill_between(hourly_avg['hour'], 
                        hourly_avg['mean'] - hourly_avg['std'],
                        hourly_avg['mean'] + hourly_avg['std'], 
                        alpha=0.3, color='#3498db')
        plt.xlabel('Hour of Day', fontsize=12)
        plt.ylabel('Average Sleep Quality Score', fontsize=12)
        plt.title('Sleep Quality Patterns by Hour of Day', fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.xticks(range(0, 24, 4))
        plt.tight_layout()
        plt.savefig('images/10_sleep_quality_by_hour.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Classification Report Heatmap
        plt.figure(figsize=(8, 6))
        best_model_name = self.classification_results['test_f1'].idxmax()
        best_result = self.classification_results.loc[best_model_name]
        
        from sklearn.metrics import classification_report
        report = classification_report(best_result['true_labels'], best_result['predictions'], 
                                     output_dict=True, zero_division=0)
        
        report_df = pd.DataFrame(report).iloc[:-1, :-3].T
        sns.heatmap(report_df.astype(float), annot=True, fmt='.3f', cmap='YlOrRd')
        plt.title(f'Classification Report Heatmap - {best_model_name}', fontsize=16, fontweight='bold')
        plt.ylabel('Sleep Quality Class', fontsize=12)
        plt.xlabel('Metrics', fontsize=12)
        plt.tight_layout()
        plt.savefig('images/11_classification_report_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ Individual detailed analysis charts saved")

    def _save_individual_clustering_charts(self):
        """Save individual clustering charts"""
        print("Saving individual clustering charts...")
        
        # 1. Silhouette Analysis
        plt.figure(figsize=(10, 6))
        cluster_range = range(2, len(self.cluster_results['silhouette_scores']) + 2)
        plt.plot(cluster_range, self.cluster_results['silhouette_scores'], 'bo-', linewidth=3, markersize=10)
        plt.title('Silhouette Analysis for Optimal Clusters', fontsize=16, fontweight='bold')
        plt.xlabel('Number of Clusters', fontsize=12)
        plt.ylabel('Silhouette Score', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Highlight optimal point
        optimal_idx = np.argmax(self.cluster_results['silhouette_scores'])
        plt.plot(cluster_range[optimal_idx], self.cluster_results['silhouette_scores'][optimal_idx], 
                'ro', markersize=15, label=f'Optimal: {cluster_range[optimal_idx]} clusters')
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig('images/12_silhouette_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Elbow Method
        plt.figure(figsize=(10, 6))
        plt.plot(cluster_range, self.cluster_results['inertias'], 'ro-', linewidth=3, markersize=10)
        plt.title('Elbow Method for Optimal Clusters', fontsize=16, fontweight='bold')
        plt.xlabel('Number of Clusters', fontsize=12)
        plt.ylabel('Inertia', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('images/13_elbow_method.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. PCA Visualization of Clusters
        plt.figure(figsize=(10, 8))
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(StandardScaler().fit_transform(self.X))
        
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], 
                            c=self.cluster_results['cluster_labels'], 
                            cmap='viridis', alpha=0.6, s=50)
        plt.title('Clusters Visualization in PCA Space', fontsize=16, fontweight='bold')
        plt.xlabel(f'First PC ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
        plt.ylabel(f'Second PC ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
        plt.colorbar(scatter)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('images/14_pca_clusters_visualization.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Sleep Quality Distribution by Cluster
        plt.figure(figsize=(10, 6))
        cluster_sleep_data = []
        for stats in self.cluster_results['cluster_stats']:
            for quality, count in stats['sleep_quality_dist'].items():
                cluster_sleep_data.append({
                    'Cluster': f"Cluster {stats['cluster']}",
                    'Sleep Quality': quality,
                    'Count': count,
                    'Percentage': count / stats['size'] * 100
                })
        
        cluster_sleep_df = pd.DataFrame(cluster_sleep_data)
        pivot_df = cluster_sleep_df.pivot(index='Cluster', columns='Sleep Quality', values='Percentage').fillna(0)
        
        colors = {'Excellent': 'green', 'Good': 'blue', 'Fair': 'orange', 'Poor': 'red'}
        pivot_df.plot(kind='bar', stacked=True, 
                     color=[colors.get(col, 'gray') for col in pivot_df.columns],
                     figsize=(10, 6))
        plt.title('Sleep Quality Distribution by Cluster (%)', fontsize=16, fontweight='bold')
        plt.ylabel('Percentage', fontsize=12)
        plt.xlabel('Cluster', fontsize=12)
        plt.legend(title='Sleep Quality', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig('images/15_cluster_sleep_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ Individual clustering charts saved")

    def _save_individual_feature_charts(self):
        """Save individual feature analysis charts"""
        print("Saving individual feature analysis charts...")
        
        rf_model = self.models['Random Forest']
        importances = rf_model.feature_importances_
        
        # 1. Top 20 Feature Importance
        plt.figure(figsize=(12, 8))
        feature_importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False).head(20)
        
        plt.barh(range(len(feature_importance_df)), feature_importance_df['importance'], color='forestgreen', alpha=0.8)
        plt.yticks(range(len(feature_importance_df)), 
                  [name.replace('_', ' ').title() for name in feature_importance_df['feature']])
        plt.title('Top 20 Most Important Features', fontsize=16, fontweight='bold')
        plt.xlabel('Feature Importance', fontsize=12)
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('images/16_top_20_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Feature Categories Distribution
        plt.figure(figsize=(10, 8))
        categories = {
            'Heart Rate': 0, 'Air Quality': 0, 'Stress': 0, 'Blood Oxygen': 0,
            'Temporal': 0, 'Sleep': 0, 'Other': 0
        }
        
        for feature in self.feature_names:
            if 'hr_' in feature:
                categories['Heart Rate'] += 1
            elif any(x in feature for x in ['pm25', 'aqi', 'o3']):
                categories['Air Quality'] += 1
            elif 'stress' in feature:
                categories['Stress'] += 1
            elif 'spo2' in feature:
                categories['Blood Oxygen'] += 1
            elif feature in ['hour', 'day_of_week', 'month']:
                categories['Temporal'] += 1
            elif any(x in feature for x in ['sleep', 'deep', 'rem', 'light']):
                categories['Sleep'] += 1
            else:
                categories['Other'] += 1
        
        categories = {k: v for k, v in categories.items() if v > 0}
        colors = plt.cm.Set3(np.linspace(0, 1, len(categories)))
        
        plt.pie(categories.values(), labels=categories.keys(), autopct='%1.1f%%', 
               colors=colors, startangle=90)
        plt.title('Feature Distribution by Category', fontsize=16, fontweight='bold')
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig('images/17_feature_categories_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Cumulative Feature Importance
        plt.figure(figsize=(12, 6))
        sorted_importances = np.sort(importances)[::-1]
        cumulative_importance = np.cumsum(sorted_importances)
        
        plt.plot(range(1, len(cumulative_importance) + 1), cumulative_importance, 'b-', linewidth=3)
        plt.axhline(y=0.8, color='r', linestyle='--', linewidth=2, label='80% threshold')
        plt.axhline(y=0.9, color='orange', linestyle='--', linewidth=2, label='90% threshold')
        
        plt.xlabel('Number of Features', fontsize=12)
        plt.ylabel('Cumulative Importance', fontsize=12)
        plt.title('Cumulative Feature Importance Analysis', fontsize=16, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Find where we reach 80% and 90%
        idx_80 = np.where(cumulative_importance >= 0.8)[0]
        if len(idx_80) > 0:
            plt.axvline(x=idx_80[0]+1, color='r', linestyle=':', alpha=0.7)
            plt.text(idx_80[0]+1, 0.4, f'{idx_80[0]+1} features\nfor 80%', 
                    ha='center', va='center', 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('images/18_cumulative_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ Individual feature analysis charts saved")
"""
Sleep Quality Classification & Clustering Pipeline
=================================================
Applies multiple ML techniques to predict and analyze sleep quality patterns
- Classification: Decision Trees, SVM, K-Nearest Neighbors
- Clustering: K-means analysis
- Feature selection based on importance analysis
"""


def main():
    # Initialize pipeline
    pipeline = SleepQualityAnalysisPipeline()
    
    # Run complete analysis
    pipeline.load_and_prepare_data()
    pipeline.prepare_features()
    pipeline.train_classification_models()
    pipeline.perform_clustering_analysis()
    pipeline.create_comprehensive_visualizations()
    pipeline.generate_comprehensive_report()
    
    # Save models and results
    pipeline.save_models()
    
    # Save results
    pipeline.classification_results.to_csv('classification_model_results.csv')
    
    # Save cluster analysis
    cluster_summary = pd.DataFrame(pipeline.cluster_results['cluster_stats'])
    cluster_summary.to_csv('clustering_analysis_results.csv', index=False)
    
    print("\nAnalysis complete! Check generated files and visualizations.")
    print("Files created:")
    print("📊 Overview Images:")
    print("- images/01_model_performance_overview.png")
    print("- images/02_detailed_analysis_overview.png") 
    print("- images/03_clustering_analysis_overview.png")
    print("- images/04_feature_analysis_overview.png")
    print("\n📈 Individual Performance Charts:")
    print("- images/05_model_performance_comparison.png")
    print("- images/06_cross_validation_scores.png")
    print("- images/07_confusion_matrix_best_model.png")
    print("- images/08_model_accuracy_ranking.png")
    print("\n🔍 Detailed Analysis Charts:")
    print("- images/09_feature_correlation_heatmap.png")
    print("- images/10_sleep_quality_by_hour.png")
    print("- images/11_classification_report_heatmap.png")
    print("\n🎯 Clustering Analysis Charts:")
    print("- images/12_silhouette_analysis.png")
    print("- images/13_elbow_method.png")
    print("- images/14_pca_clusters_visualization.png")
    print("- images/15_cluster_sleep_distribution.png")
    print("\n⭐ Feature Analysis Charts:")
    print("- images/16_top_20_feature_importance.png")
    print("- images/17_feature_categories_distribution.png")
    print("- images/18_cumulative_feature_importance.png")
    print("\n📋 Data Files:")
    print("- classification_model_results.csv") 
    print("- clustering_analysis_results.csv")
    print("- saved_models/ directory with all trained models")
    print(f"\n🎉 Total: 18 individual charts + 4 overview images + data files!")
    print("Perfect for your assignment report! 📚")
    
    return pipeline

if __name__ == "__main__":
    pipeline = main()