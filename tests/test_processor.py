
from app.services.data_processor import DataProcessor
import pandas as pd
import os
import pickle

def create_sample_data():
    """Create a sample DataFrame for testing"""
    sample_texts = [
        "Sample telegram message 1",
        "Sample telegram message 2",
        "Sample telegram message 3"
    ]
    return pd.DataFrame({'text': sample_texts})

def main():
    try:
        # Initialize the data processor
        print("Initializing data processor...")
        processor = DataProcessor()
        
        # First, ensure the data directory exists
        os.makedirs("app/data/raw", exist_ok=True)
        
        # Check if we have the pickle file first
        pickle_path = "app/data/raw/final_telegram_df"
        csv_path = "app/data/raw/final_telegram_data.csv"
        
        if os.path.exists(pickle_path):
            print("Loading pickle file...")
            with open(pickle_path, 'rb') as file:
                telegram_df = pickle.load(file)
                # Save as CSV
                telegram_df.to_csv(csv_path, index=False)
        elif not os.path.exists(csv_path):
            print("Creating sample data CSV...")
            # Create sample DataFrame
            df = create_sample_data()
            df.to_csv(csv_path, index=False)
        
        # Load and process data
        print("Loading and processing data...")
        embeddings, df = processor.load_and_process_data()
        
        # Print DataFrame information
        print("\nDataFrame Info:")
        print(df.info())
        print("\nFirst few rows:")
        print(df.head())
        
        # Print basic information
        print("\nData Statistics:")
        stats = processor.get_stats()
        for key, value in stats.items():
            print(f"{key}: {value}")
        
        # Test batch generation
        print("\nTesting batch generation...")
        batch_emb, batch_labels = processor.get_batch(batch_size=5)
        print(f"Batch shape: {batch_emb.shape}")
        print(f"Labels shape: {batch_labels.shape}")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
