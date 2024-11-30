# Description: Main FastAPI application file that defines the API endpoints and initializes the recommender system.
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
from app.models.reinforcement_model import get_device
from app.services.data_processor import DataProcessor
from app.services.recommender import Recommender
from typing import List, Dict
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="OSINT Recommendation API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
data_processor = DataProcessor()
recommender = None
device = None

@app.on_event("startup")
async def startup_event():
    global recommender, device
    try:
        # Get the best available device
        device = get_device()
        logger.info(f"Using device: {device}")
        
        # Load data and initialize recommender on startup
        logger.info("Loading data and initializing recommender...")
        embeddings, telegram_df = data_processor.load_and_process_data()
        
        # Initialize recommender with device
        recommender = Recommender(embeddings, telegram_df, device)
        logger.info("Recommender initialized successfully")
        
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        raise

@app.get("/health")
async def health_check():
    device_info = {
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "mps_available": hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    }
    
    return {
        "status": "healthy",
        "device_info": device_info,
        "model_loaded": recommender is not None
    }

@app.post("/train")
async def train_model(num_episodes: int = 1000):
    if not recommender:
        raise HTTPException(status_code=500, detail="Recommender not initialized")
    
    try:
        logger.info(f"Starting training for {num_episodes} episodes")
        training_stats = await recommender.train(num_episodes)
        logger.info("Training completed")
        return {"message": "Training completed", "stats": training_stats}
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/recommend/{num_posts}")
async def get_recommendations(num_posts: int = 10):
    if not recommender:
        raise HTTPException(status_code=500, detail="Recommender not initialized")
    
    try:
        logger.info(f"Getting {num_posts} recommendations")
        recommendations = await recommender.get_recommendations(num_posts)
        return {"recommendations": recommendations}
    except Exception as e:
        logger.error(f"Error getting recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model/stats")
async def get_model_stats():
    if not recommender:
        raise HTTPException(status_code=500, detail="Recommender not initialized")
    
    try:
        stats = recommender.get_stats()
        stats['device_info'] = {
            "device": str(device),
            "cuda_available": torch.cuda
        }
        return stats
    except Exception as e:
        logger.error(f"Error getting model stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))