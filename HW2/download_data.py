# AI NOTICE
# All the code below is AI generated

import pandas as pd
from datasets import load_dataset

def download_and_save_all_emotions():
    print("Downloading the dair-ai/emotion dataset...")
    # Load the standard train, validation, and test splits
    dataset = load_dataset("dair-ai/emotion")
    
    # The official integer-to-emotion mapping for this dataset
    emotion_map = {
        0: "sadness",
        1: "joy",
        2: "love",
        3: "anger",
        4: "fear",
        5: "surprise"
    }

    def process_split(split_name):
        df = pd.DataFrame(dataset[split_name])
        
        # Keep the original integer 'label'
        # Add a readable 'emotion_name' column so you know what the numbers mean
        df['emotion_name'] = df['label'].map(emotion_map)
        
        # Create one-hot encodings for ALL 6 classes without filtering anything
        for label_id, emotion in emotion_map.items():
            df[emotion] = (df['label'] == label_id).astype(int)
            
        # Drop the intermediate integer label to keep your text and one-hots clean
        return df.drop(columns=['label'])

    # Process both splits
    train_df = process_split('train')
    test_df = process_split('test')

    # Save them to your custom file names
    train_df.to_csv('tweet.train', index=False)
    test_df.to_csv('tweet.test', index=False)

    print(f"\n✅ Success! Saved {len(train_df)} rows to tweet.train")
    print(f"✅ Success! Saved {len(test_df)} rows to tweet.test")
    print("\nColumns available: text, emotion_name, sadness, joy, love, anger, fear, surprise")

if __name__ == "__main__":
    download_and_save_all_emotions()