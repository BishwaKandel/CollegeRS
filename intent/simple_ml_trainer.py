import pandas as pd
import numpy as np
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
import joblib

class IntentClassifierTrainer:
    """Simple ML trainer for intent classification using processed data"""
    
    def __init__(self):
        self.vectorizer = None
        self.model = None
        self.label_encoder = LabelEncoder()
        self.class_weights = None
        
    def load_processed_data(self):
        """Load the processed training data"""
        print("📂 Loading processed data...")
        
        # Load datasets
        train_df = pd.read_csv('train_data.csv')
        val_df = pd.read_csv('val_data.csv')
        test_df = pd.read_csv('test_data.csv')
        
        # Load metadata
        with open('label_classes.json', 'r') as f:
            label_classes = json.load(f)
            
        with open('class_weights.json', 'r') as f:
            class_weights = json.load(f)
            # Convert string keys back to integers
            self.class_weights = {int(k): v for k, v in class_weights.items()}
        
        with open('data_stats.json', 'r') as f:
            stats = json.load(f)
        
        print(f"✅ Loaded data:")
        print(f"  Training: {len(train_df)} samples")
        print(f"  Validation: {len(val_df)} samples") 
        print(f"  Test: {len(test_df)} samples")
        print(f"  Classes: {stats['num_classes']}")
        
        return train_df, val_df, test_df, label_classes, stats
    
    def prepare_features(self, train_df, val_df, test_df):
        """Extract TF-IDF features from text"""
        print("\n🔧 Extracting TF-IDF features...")
        
        # Initialize TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),  # Use unigrams and bigrams
            stop_words='english',
            lowercase=True,
            strip_accents='unicode'
        )
        
        # Fit on training data and transform all sets
        X_train = self.vectorizer.fit_transform(train_df['text'])
        X_val = self.vectorizer.transform(val_df['text'])
        X_test = self.vectorizer.transform(test_df['text'])
        
        # Get labels
        y_train = train_df['label_encoded'].values
        y_val = val_df['label_encoded'].values
        y_test = test_df['label_encoded'].values
        
        print(f"✅ Feature extraction complete:")
        print(f"  Feature dimensions: {X_train.shape[1]}")
        print(f"  Training samples: {X_train.shape[0]}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def train_models(self, X_train, y_train, X_val, y_val):
        """Train multiple ML models and compare performance"""
        print("\n🚀 TRAINING MULTIPLE MODELS")
        print("-" * 40)
        
        # Define models to test
        models = {
            'Logistic Regression': LogisticRegression(
                class_weight='balanced',
                max_iter=1000,
                random_state=42
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                class_weight='balanced',
                random_state=42
            ),
            'SVM': SVC(
                class_weight='balanced',
                kernel='linear',
                random_state=42,
                probability=True
            ),
            'Naive Bayes': MultinomialNB(alpha=0.1)
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Validate
            val_pred = model.predict(X_val)
            val_accuracy = accuracy_score(y_val, val_pred)
            val_f1 = f1_score(y_val, val_pred, average='weighted')
            
            results[name] = {
                'model': model,
                'val_accuracy': val_accuracy,
                'val_f1': val_f1
            }
            
            print(f"  Validation Accuracy: {val_accuracy:.4f}")
            print(f"  Validation F1: {val_f1:.4f}")
        
        # Find best model
        best_model_name = max(results.keys(), key=lambda x: results[x]['val_f1'])
        self.model = results[best_model_name]['model']
        
        print(f"\n🏆 Best model: {best_model_name}")
        print(f"  F1 Score: {results[best_model_name]['val_f1']:.4f}")
        
        return results, best_model_name
    
    def evaluate_model(self, X_test, y_test, label_classes):
        """Comprehensive evaluation on test set"""
        print("\n📊 FINAL EVALUATION ON TEST SET")
        print("-" * 40)
        
        # Predictions
        test_pred = self.model.predict(X_test)
        
        # Overall metrics
        test_accuracy = accuracy_score(y_test, test_pred)
        test_f1_weighted = f1_score(y_test, test_pred, average='weighted')
        test_f1_macro = f1_score(y_test, test_pred, average='macro')
        
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test F1 (Weighted): {test_f1_weighted:.4f}")
        print(f"Test F1 (Macro): {test_f1_macro:.4f}")
        
        # Detailed classification report
        print("\nDetailed Classification Report:")
        report = classification_report(
            y_test, test_pred, 
            target_names=label_classes,
            digits=4
        )
        print(report)
        
        # Per-class performance analysis
        report_dict = classification_report(
            y_test, test_pred,
            target_names=label_classes,
            output_dict=True
        )
        
        return {
            'accuracy': test_accuracy,
            'f1_weighted': test_f1_weighted,
            'f1_macro': test_f1_macro,
            'classification_report': report_dict
        }
    
    def save_model(self, model_name='intent_classifier'):
        """Save the trained model and vectorizer"""
        print("\n💾 Saving model...")
        
        # Save model and vectorizer
        joblib.dump(self.model, f'{model_name}_model.pkl')
        joblib.dump(self.vectorizer, f'{model_name}_vectorizer.pkl')
        
        print(f"✅ Model saved as:")
        print(f"  - {model_name}_model.pkl")
        print(f"  - {model_name}_vectorizer.pkl")
    
    def predict_intent(self, text, label_classes):
        """Predict intent for new text"""
        if self.model is None or self.vectorizer is None:
            raise ValueError("Model not trained yet!")
        
        # Preprocess and vectorize
        text_vector = self.vectorizer.transform([text])
        
        # Predict
        prediction = self.model.predict(text_vector)[0]
        confidence = self.model.predict_proba(text_vector)[0]
        
        predicted_intent = label_classes[prediction]
        confidence_score = confidence[prediction]
        
        return predicted_intent, confidence_score

def main():
    """Main training pipeline"""
    print("🚀 INTENT CLASSIFICATION - ML TRAINING PIPELINE")
    print("=" * 60)
    
    # Initialize trainer
    trainer = IntentClassifierTrainer()
    
    try:
        # Load processed data
        train_df, val_df, test_df, label_classes, stats = trainer.load_processed_data()
        
        # Prepare features
        X_train, X_val, X_test, y_train, y_val, y_test = trainer.prepare_features(
            train_df, val_df, test_df
        )
        
        # Train models
        results, best_model_name = trainer.train_models(X_train, y_train, X_val, y_val)
        
        # Final evaluation
        final_results = trainer.evaluate_model(X_test, y_test, label_classes)
        
        # Save model
        trainer.save_model('best_intent_classifier')
        
        # Test predictions
        print("\n🔮 TESTING PREDICTIONS")
        print("-" * 25)
        
        test_texts = [
            "What courses do you offer?",
            "How much is the fee?",
            "Tell me about hostel facilities",
            "When can I visit the campus?",
            "How to apply for scholarship?"
        ]
        
        for text in test_texts:
            intent, confidence = trainer.predict_intent(text, label_classes)
            print(f"Text: '{text}'")
            print(f"  → Intent: {intent} (confidence: {confidence:.3f})")
            print()
        
        print("🎉 TRAINING COMPLETE!")
        print(f"Final Test F1 Score: {final_results['f1_weighted']:.4f}")
        
    except FileNotFoundError as e:
        print("❌ Error: Required files not found. Please run prepare_ml_data.py first.")
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main()
