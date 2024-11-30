import torch 
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from typing import Tuple, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self, data_dir: str = "app/data"):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.processed_dir.mkdir(exist_ok=True, parents=True)
        
        self.telegram_df = None
        self.embeddings = None
        self.tensor_dims = None

    def load_and_process_data(self) -> Tuple[torch.Tensor, pd.DataFrame]:
        """
        Load and process the telegram data and embeddings
        """
        try:
            logger.info("Loading telegram data...")
            # Read the CSV file
            self.telegram_df = pd.read_csv(self.raw_dir / "final_telegram_data.csv")
            
            logger.info(f"DataFrame columns: {self.telegram_df.columns}")
            logger.info(f"DataFrame shape: {self.telegram_df.shape}")
            logger.info(f"First few rows: \n{self.telegram_df.head()}")
            
            # Normalize and process the data
            self.telegram_df = self._normalize_telegram_data(self.telegram_df)
            
            logger.info("Processing embeddings...")
            try:
                with open(self.raw_dir / "sbert_embeddings.pkl", "rb") as f:
                    embeddings = pickle.load(f)
                
                # Convert embeddings to tensor
                self.embeddings = torch.tensor([emb for emb in embeddings], dtype=torch.float32)
                self.tensor_dims = self.embeddings.size(1)
                
                logger.info(f"Embeddings shape: {self.embeddings.shape}")
                
            except FileNotFoundError:
                logger.warning("SBERT embeddings file not found. Using dummy embeddings for testing.")
                # Create dummy embeddings for testing
                self.embeddings = torch.randn(len(self.telegram_df), 384)  # SBERT usually uses 384 dimensions
                self.tensor_dims = 384
            
            # Save processed data
            self._save_processed_data()
            
            return self.embeddings, self.telegram_df
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def _normalize_telegram_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize the telegram data"""
        # Create a copy to avoid modifying the original
        df = df.copy()
        
        # Add a simple sentiment score based on text length (placeholder)
        # You can replace this with actual sentiment analysis later
        df['score'] = df['text'].str.len().apply(lambda x: np.tanh(x/1000))
        
        # Create normalized score
        df['normalized_score'] = (df['score'] - df['score'].mean()) / df['score'].std()
        
        # Convert to binary labels based on normalized scores
        df['label'] = df['normalized_score'].apply(lambda x: 1 if x > 0 else 0)
        
        logger.info(f"Data processed. Positive labels: {df['label'].sum()}/{len(df)}")
        
        return df

    def _save_processed_data(self):
        """Save processed data for faster loading next time"""
        processed_data = {
            'embeddings': self.embeddings,
            'telegram_df': self.telegram_df,
            'tensor_dims': self.tensor_dims
        }
        torch.save(processed_data, self.processed_dir / "processed_data.pt")

    def get_batch(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a random batch of data for training"""
        indices = torch.randperm(len(self.telegram_df))[:batch_size]
        batch_embeddings = self.embeddings[indices]
        batch_labels = torch.tensor(self.telegram_df['label'].iloc[indices].values)
        return batch_embeddings, batch_labels

    def get_stats(self) -> Dict:
        """Get basic statistics about the data"""
        return {
            'total_posts': len(self.telegram_df),
            'positive_labels': int(self.telegram_df['label'].sum()),
            'embedding_dims': self.tensor_dims,
            'score_mean': float(self.telegram_df['normalized_score'].mean()),
            'score_std': float(self.telegram_df['normalized_score'].std())
        }