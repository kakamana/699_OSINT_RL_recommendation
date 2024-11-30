import pytest
from app.services.data_processor import DataProcessor

def test_data_processor():
    processor = DataProcessor()
    embeddings, df = processor.load_and_process_data()
    
    # Basic tests
    assert embeddings is not None
    assert df is not None
    assert len(df) == len(embeddings)
    
    # Test batch generation
    batch_size = 32
    batch_embeddings, batch_labels = processor.get_batch(batch_size)
    assert batch_embeddings.shape[0] == batch_size
    assert batch_labels.shape[0] == batch_size
    
    # Test stats
    stats = processor.get_stats()
    assert 'total_posts' in stats
    assert 'positive_labels' in stats
    assert 'embedding_dims' in stats