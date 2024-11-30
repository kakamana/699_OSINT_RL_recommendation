# OSINT Recommendation System Using Reinforcement Learning

## Project Overview
This project implements an intelligent recommendation system for OSINT (Open Source Intelligence) data using reinforcement learning. The system processes Telegram data and provides personalized content recommendations based on learned patterns and user interactions.

## Features
- FastAPI-based REST API for easy integration
- Reinforcement learning-based recommendation engine
- Support for GPU acceleration (CUDA and Apple M-series)
- Real-time training and recommendation endpoints
- Automated data processing pipeline
- Health monitoring and model statistics

## Technical Architecture

### Components
1. **Data Processing Pipeline** (`DataProcessor`)
   - Handles raw Telegram data processing
   - Manages embeddings generation and storage
   - Implements data normalization and preprocessing

2. **Recommendation Engine** (`Recommender`)
   - Uses reinforcement learning for content ranking
   - Implements Q-learning with neural networks
   - Supports batch training and real-time recommendations

3. **REST API** (`FastAPI Application`)
   - Provides endpoints for training and recommendations
   - Includes health monitoring
   - Supports CORS for web integration

## Setup and Installation

### Prerequisites
- Python 3.8+
- PyTorch
- FastAPI
- uvicorn
- Other dependencies (listed in requirements.txt)

### Installation Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/kakamana/699_OSINT_RL_recommendation.git
   cd OSINT_RL_RecommendationSystem
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application
1. Start the FastAPI server:
   ```bash
   python run.py
   ```
   Or using uvicorn directly:
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8040 --reload
   ```

2. Access the API:
   - API documentation: http://localhost:8040/docs
   - Health check: http://localhost:8040/health

## API Endpoints

### Health Check
```http
GET /health
```
Returns system health status and device information.

### Training
```http
POST /train?num_episodes=1000
```
Trains the recommendation model with specified number of episodes.

### Get Recommendations
```http
GET /recommend/{num_posts}
```
Returns recommended posts based on trained model.

### Model Statistics
```http
GET /model/stats
```
Returns current model statistics and training information.

## Project Structure
```
project/
├── app/
│   ├── models/
│   │   └── reinforcement_model.py
│   ├── services/
│   │   ├── data_processor.py
│   │   └── recommender.py
│   └── main.py
├── data/
│   ├── raw/
│   └── processed/
├── requirements.txt
└── run.py
```

## Future Scope

### Potential Enhancements
1. **Model Improvements**
   - Implementation of more sophisticated RL algorithms (A2C, PPO)
   - Integration of transformer-based architectures
   - Multi-modal recommendation support

2. **System Features**
   - User feedback integration
   - A/B testing framework
   - Real-time model updates
   - Automated model retraining
   - Performance monitoring dashboard

3. **Scalability**
   - Distributed training support
   - Database integration for large-scale deployment
   - Containerization and orchestration
   - Load balancing and high availability

### Research Directions
1. Exploration of hybrid recommendation approaches
2. Integration of contextual bandits
3. Implementation of explainable AI techniques
4. Privacy-preserving recommendation methods

## Performance Considerations

### Hardware Requirements
- Minimum: 4GB RAM, CPU with 2+ cores
- Recommended: 8GB+ RAM, GPU support (NVIDIA CUDA or Apple M-series)
- Storage: 1GB+ for model and data storage

### Optimization Tips
1. Enable GPU acceleration when available
2. Adjust batch sizes based on available memory
3. Monitor model convergence during training
4. Use appropriate epsilon decay rates for exploration

## Troubleshooting

### Common Issues
1. **CUDA Out of Memory**
   - Reduce batch size
   - Enable gradient checkpointing

2. **Slow Training**
   - Check device utilization
   - Adjust network architecture
   - Enable GPU support if available

3. **API Connection Issues**
   - Verify CORS settings
   - Check port availability
   - Ensure proper network permissions

### Debug Mode
Enable debug logging by setting:
```python
logging.basicConfig(level=logging.DEBUG)
```

## Contributing
Contributions are welcome! Please feel free to submit pull requests, create issues, or suggest improvements.

## License
This is open license and can be used. We have done this project for our Master of Applied Data Science Capstone Project

## Acknowledgments
- Mention any libraries, papers, or resources that inspired the project
- Credit contributors and maintainers
    @NickBermudez: For putting major effort for creating reinforcement model
    @davidsoliven: For providing dataset related to telegram crisis management
- Reference any related research or implementations