from app.services.data_processor import DataProcessor
from app.services.telegram_scraper import TelegramScraper

from datetime import datetime
import asyncio
from typing import Optional, List, Dict
import torch
from sentence_transformers import SentenceTransformer
from pathlib import Path
import pickle
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IntegratedTelegramProcessor:
    def __init__(
        self, 
        data_dir: str = "app/data",
        batch_size: int = 100,
        save_interval: int = 1000
    ):
        # Initialize components
        self.scraper = TelegramScraper()
        self.data_processor = DataProcessor(data_dir)
        
        # Configuration
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.raw_dir.mkdir(exist_ok=True, parents=True)
        self.batch_size = batch_size
        self.save_interval = save_interval
        
        # Initialize SBERT
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Message buffer
        self.message_buffer: List[Dict] = []
        self.embedding_buffer: List[torch.Tensor] = []
        
        # Statistics
        self.processed_count = 0

    async def start(self):
        """Start the scraper and processor"""
        try:
            await self.scraper.start()
            logger.info("Scraper initialized successfully")
        except Exception as e:
            logger.error(f"Failed to start scraper: {e}")
            raise

    async def process_channel(self, channel_username: str, limit: Optional[int] = None):
        """Process messages from a channel"""
        try:
            async for message in self.scraper.client.iter_messages(channel_username, limit=limit):
                # Extract message data
                message_data = await self._extract_message_data(message)
                
                # Add to buffer
                self.message_buffer.append(message_data)
                
                # Process batch if buffer is full
                if len(self.message_buffer) >= self.batch_size:
                    await self._process_batch()
                
                # Save to disk periodically
                if self.processed_count % self.save_interval == 0:
                    self._save_to_disk()
                    
                self.processed_count += 1
                
            # Process remaining messages
            if self.message_buffer:
                await self._process_batch()
                self._save_to_disk()
                
        except Exception as e:
            logger.error(f"Error processing channel {channel_username}: {e}")
            raise
        finally:
            await self.close()

    async def _extract_message_data(self, message) -> Dict:
        """Extract relevant data from telegram message"""
        return {
            'message_id': message.id,
            'text': message.text,
            'date': message.date,
            'views': getattr(message, 'views', 0),
            'forwards': getattr(message, 'forwards', 0),
            'media_type': self.scraper._get_media_type(message),
            'media_path': await self.scraper._download_media(message) if message.media else None
        }

    async def _process_batch(self):
        """Process a batch of messages"""
        try:
            # Generate embeddings
            texts = [msg['text'] for msg in self.message_buffer]
            embeddings = self.embedding_model.encode(texts)
            
            # Convert to tensor
            embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)
            
            # Store embeddings
            self.embedding_buffer.extend(embeddings)
            
            # Create DataFrame for batch
            df_batch = pd.DataFrame(self.message_buffer)
            
            # Process through DataProcessor
            processed_df = self.data_processor._normalize_telegram_data(df_batch)
            
            # Store processed data
            self._store_processed_batch(processed_df, embeddings_tensor)
            
            # Clear buffers
            self.message_buffer = []
            
            logger.info(f"Processed batch of {len(df_batch)} messages")
            
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            raise

    def _store_processed_batch(self, df: pd.DataFrame, embeddings: torch.Tensor):
        """Store processed batch data"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save DataFrame
        df.to_pickle(self.raw_dir / f"batch_{timestamp}_df.pkl")
        
        # Save embeddings
        torch.save(embeddings, self.raw_dir / f"batch_{timestamp}_embeddings.pt")

    def _save_to_disk(self):
        """Save accumulated data to disk"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save full DataFrame
        if hasattr(self.data_processor, 'telegram_df'):
            with open(self.raw_dir / f"final_telegram_df_{timestamp}", 'wb') as f:
                pickle.dump(self.data_processor.telegram_df, f)
        
        # Save all embeddings
        if self.embedding_buffer:
            with open(self.raw_dir / f"sbert_embeddings_{timestamp}.pkl", 'wb') as f:
                pickle.dump(self.embedding_buffer, f)
        
        logger.info(f"Saved data to disk at {timestamp}")

    async def close(self):
        """Close all connections and save final data"""
        await self.scraper.close()
        self._save_to_disk()
        logger.info("Processor closed successfully")

# Usage example
async def main():
    # Initialize processor
    processor = IntegratedTelegramProcessor(
        data_dir="app/data",
        batch_size=100,
        save_interval=1000
    )
    
    try:
        # Start processor
        await processor.start()
        
        # Process channels
        channels = [
            'channel1',
            'channel2'
        ]
        
        for channel in channels:
            logger.info(f"Processing channel: {channel}")
            await processor.process_channel(channel, limit=5000)
            
    except Exception as e:
        logger.error(f"Processing failed: {e}")
    finally:
        await processor.close()

if __name__ == "__main__":
    asyncio.run(main())