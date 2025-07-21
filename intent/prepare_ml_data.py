import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import re
import random
from collections import Counter

def load_intent_data(file_path='intent.json'):
    """Load intent data from JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def clean_text(text):
    """Clean and normalize text data"""
    # Convert to lowercase
    text = text.lower().strip()
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep punctuation that adds meaning
    text = re.sub(r'[^\w\s\?\.\!\,\-]', '', text)
    
    return text

def augment_text(text, augmentation_type='simple'):
    """Apply various text augmentation techniques"""
    augmentations = []
    
    if augmentation_type == 'simple':
        # Basic augmentations
        augmentations = [
            text,  # Original
            f"can you {text}",
            f"please {text}",
            f"i want to {text}",
            f"help me {text}",
            f"{text}?",
            f"{text}.",
            text.replace('?', ''),
            text.replace('.', ''),
        ]
    
    elif augmentation_type == 'advanced':
        # More sophisticated augmentations
        variations = [
            text,
            f"could you {text}",
            f"i need to {text}",
            f"how can i {text}",
            f"tell me about {text}",
            f"what about {text}",
            f"information on {text}",
            f"details about {text}",
            f"{text} please",
            f"{text}?",
            text.upper(),
            text.capitalize(),
        ]
        
        # Add contractions and informal versions
        contractions = {
            'what is': "what's",
            'how is': "how's", 
            'where is': "where's",
            'can not': "can't",
            'do not': "don't",
            'will not': "won't"
        }
        
        for original, contracted in contractions.items():
            if original in text:
                variations.append(text.replace(original, contracted))
        
        augmentations = variations
    
    # Remove duplicates and empty strings
    augmentations = list(set([aug.strip() for aug in augmentations if aug.strip()]))
    
    return augmentations

def analyze_data_quality(df):
    """Analyze the quality and balance of the dataset"""
    print("📊 DATASET ANALYSIS")
    print("=" * 50)
    
    # Basic statistics
    total_samples = len(df)
    unique_intents = df['intent'].nunique()
    
    print(f"Total samples: {total_samples}")
    print(f"Unique intents: {unique_intents}")
    print(f"Average samples per intent: {total_samples/unique_intents:.1f}")
    
    # Class distribution
    class_counts = df['intent'].value_counts()
    print(f"\nClass distribution:")
    print(f"Min samples per class: {class_counts.min()}")
    print(f"Max samples per class: {class_counts.max()}")
    print(f"Std deviation: {class_counts.std():.2f}")
    
    # Find imbalanced classes
    mean_samples = class_counts.mean()
    small_classes = class_counts[class_counts < mean_samples * 0.5]
    large_classes = class_counts[class_counts > mean_samples * 2]
    
    if len(small_classes) > 0:
        print(f"\n⚠️ Small classes (< 50% of average):")
        for intent, count in small_classes.items():
            print(f"  {intent}: {count} samples")
    
    if len(large_classes) > 0:
        print(f"\n📈 Large classes (> 200% of average):")
        for intent, count in large_classes.items():
            print(f"  {intent}: {count} samples")
    
    # Text length analysis
    text_lengths = df['text'].str.len()
    print(f"\nText length statistics:")
    print(f"  Mean: {text_lengths.mean():.1f} characters")
    print(f"  Min: {text_lengths.min()}, Max: {text_lengths.max()}")
    print(f"  Standard deviation: {text_lengths.std():.1f}")
    
    return class_counts

def balance_dataset(df, target_samples_per_class=50, max_augmentation=3):
    """Balance the dataset using augmentation and downsampling"""
    print(f"\n🔄 BALANCING DATASET")
    print(f"Target: {target_samples_per_class} samples per class")
    print("-" * 40)
    
    balanced_data = []
    class_counts = df['intent'].value_counts()
    
    for intent in df['intent'].unique():
        intent_data = df[df['intent'] == intent]['text'].tolist()
        current_count = len(intent_data)
        
        if current_count < target_samples_per_class:
            # Augment small classes
            needed = target_samples_per_class - current_count
            print(f"Augmenting '{intent}': {current_count} → {target_samples_per_class}")
            
            # Add original samples
            for text in intent_data:
                balanced_data.append({'text': clean_text(text), 'intent': intent, 'augmented': False})
            
            # Generate augmented samples
            augmented_count = 0
            while augmented_count < needed:
                for original_text in intent_data:
                    if augmented_count >= needed:
                        break
                    
                    # Apply augmentation
                    augmented_texts = augment_text(original_text, 'advanced')
                    
                    for aug_text in augmented_texts[1:max_augmentation+1]:  # Skip original
                        if augmented_count >= needed:
                            break
                        
                        # Avoid duplicates
                        if aug_text not in [item['text'] for item in balanced_data if item['intent'] == intent]:
                            balanced_data.append({
                                'text': clean_text(aug_text), 
                                'intent': intent, 
                                'augmented': True
                            })
                            augmented_count += 1
        
        elif current_count > target_samples_per_class:
            # Downsample large classes
            print(f"Downsampling '{intent}': {current_count} → {target_samples_per_class}")
            sampled_texts = random.sample(intent_data, target_samples_per_class)
            for text in sampled_texts:
                balanced_data.append({'text': clean_text(text), 'intent': intent, 'augmented': False})
        
        else:
            # Keep as is
            for text in intent_data:
                balanced_data.append({'text': clean_text(text), 'intent': intent, 'augmented': False})
    
    balanced_df = pd.DataFrame(balanced_data)
    
    print(f"\n✅ Balancing complete!")
    print(f"Original dataset: {len(df)} samples")
    print(f"Balanced dataset: {len(balanced_df)} samples")
    print(f"Augmented samples: {len(balanced_df[balanced_df['augmented'] == True])}")
    
    return balanced_df

def prepare_ml_features(df):
    """Prepare features for ML training"""
    print(f"\n🔧 PREPARING ML FEATURES")
    print("-" * 30)
    
    # Encode labels
    label_encoder = LabelEncoder()
    df['label_encoded'] = label_encoder.fit_transform(df['intent'])
    
    # Calculate class weights for imbalanced learning
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(df['label_encoded']),
        y=df['label_encoded']
    )
    
    class_weight_dict = dict(zip(np.unique(df['label_encoded']), class_weights))
    
    print(f"Number of classes: {len(label_encoder.classes_)}")
    print(f"Label encoding created")
    print(f"Class weights calculated")
    
    return df, label_encoder, class_weight_dict

def create_train_test_split(df, test_size=0.2, val_size=0.1):
    """Create train, validation, and test splits"""
    print(f"\n📂 CREATING DATA SPLITS")
    print("-" * 25)
    
    # First split: train+val vs test
    train_val_df, test_df = train_test_split(
        df, 
        test_size=test_size, 
        stratify=df['intent'],
        random_state=42
    )
    
    # Second split: train vs val
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_size/(1-test_size),  # Adjust for already split data
        stratify=train_val_df['intent'],
        random_state=42
    )
    
    print(f"Training set: {len(train_df)} samples ({len(train_df)/len(df)*100:.1f}%)")
    print(f"Validation set: {len(val_df)} samples ({len(val_df)/len(df)*100:.1f}%)")
    print(f"Test set: {len(test_df)} samples ({len(test_df)/len(df)*100:.1f}%)")
    
    return train_df, val_df, test_df

def save_processed_data(train_df, val_df, test_df, label_encoder, class_weights, output_dir='./'):
    """Save all processed data and metadata"""
    print(f"\n💾 SAVING PROCESSED DATA")
    print("-" * 25)
    
    # Save datasets
    train_df.to_csv(f'{output_dir}train_data.csv', index=False)
    val_df.to_csv(f'{output_dir}val_data.csv', index=False)
    test_df.to_csv(f'{output_dir}test_data.csv', index=False)
    
    # Save label encoder classes
    with open(f'{output_dir}label_classes.json', 'w') as f:
        json.dump(label_encoder.classes_.tolist(), f, indent=2)
    
    # Save class weights
    with open(f'{output_dir}class_weights.json', 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        weights_serializable = {str(k): float(v) for k, v in class_weights.items()}
        json.dump(weights_serializable, f, indent=2)
    
    # Save data statistics
    stats = {
        'total_samples': len(train_df) + len(val_df) + len(test_df),
        'num_classes': len(label_encoder.classes_),
        'train_samples': len(train_df),
        'val_samples': len(val_df),
        'test_samples': len(test_df),
        'class_distribution': train_df['intent'].value_counts().to_dict()
    }
    
    with open(f'{output_dir}data_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"✅ Saved:")
    print(f"  - train_data.csv ({len(train_df)} samples)")
    print(f"  - val_data.csv ({len(val_df)} samples)")
    print(f"  - test_data.csv ({len(test_df)} samples)")
    print(f"  - label_classes.json")
    print(f"  - class_weights.json")
    print(f"  - data_stats.json")

def main():
    """Main function to prepare ML data"""
    print("🚀 INTENT DATA PREPARATION FOR ML TRAINING")
    print("=" * 60)
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    try:
        # Load data
        print("📂 Loading intent data...")
        intent_data = load_intent_data('intent.json')
        
        # Convert to DataFrame
        texts = []
        intents = []
        
        for intent_obj in intent_data['intents']:
            intent_name = intent_obj['intent']
            for example in intent_obj['examples']:
                texts.append(example)
                intents.append(intent_name)
        
        df = pd.DataFrame({'text': texts, 'intent': intents})
        print(f"✅ Loaded {len(df)} samples from {df['intent'].nunique()} intents")
        
        # Analyze data quality
        class_counts = analyze_data_quality(df)
        
        # Balance dataset
        balanced_df = balance_dataset(df, target_samples_per_class=50)
        
        # Prepare ML features
        processed_df, label_encoder, class_weights = prepare_ml_features(balanced_df)
        
        # Create train/val/test splits
        train_df, val_df, test_df = create_train_test_split(processed_df)
        
        # Save processed data
        save_processed_data(train_df, val_df, test_df, label_encoder, class_weights)
        
        print(f"\n🎉 DATA PREPARATION COMPLETE!")
        print("=" * 40)
        print("Your data is now ready for ML training!")
        print("Next steps:")
        print("1. Use train_data.csv for training")
        print("2. Use val_data.csv for hyperparameter tuning")
        print("3. Use test_data.csv for final evaluation")
        print("4. Use class_weights.json for handling class imbalance")
        
    except FileNotFoundError:
        print("❌ Error: intent.json file not found!")
        print("Please make sure the file exists in the current directory.")
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main()
