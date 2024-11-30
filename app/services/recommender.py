# Description: Service class to train and get recommendations using the reinforcement learning model
import torch
import numpy as np
from typing import List, Dict
from app.models.reinforcement_model import RankingAgent, get_device
import asyncio
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

class Recommender:
    def __init__(self, embeddings: torch.Tensor, telegram_df, device=None):
        self.device = device if device else get_device()
        # Move embeddings to device
        self.embeddings = embeddings.to(self.device)
        self.telegram_df = telegram_df
        self.input_dim = embeddings.shape[1]
        
        logger.info(f"Initializing Recommender on device: {self.device}")
        
        # Initialize agent
        self.agent = RankingAgent(
            input_dim=self.input_dim,
            device=self.device,
            learning_rate=0.001,
            gamma=0.99,
            epsilon=1.0,
            batch_size=64
        )
        
        self.training_stats = {
            'episodes': 0,
            'avg_reward': 0,
            'avg_loss': 0
        }

    async def train(self, num_episodes: int = 1000) -> Dict:
        """Train the recommendation model"""
        self.agent.train_mode()
        logger.info(f"Starting training for {num_episodes} episodes")
        
        episode_rewards = []
        episode_losses = []
        
        try:
            for episode in tqdm(range(num_episodes)):
                state = self.embeddings[np.random.randint(len(self.embeddings))]
                episode_reward = 0
                episode_loss = 0
                
                for step in range(50):  # max steps per episode
                    action = self.agent.choose_action(state)
                    reward = self._calculate_reward(state, action)
                    next_state = self._get_next_state(action)
                    
                    self.agent.store_transition(
                        state.cpu().numpy(),
                        action,
                        reward,
                        next_state.cpu().numpy(),
                        False
                    )
                    
                    loss = self.agent.update()
                    episode_reward += reward
                    episode_loss += loss
                    
                    state = next_state
                    
                    if step % 10 == 0:
                        await asyncio.sleep(0)
                
                episode_rewards.append(episode_reward)
                episode_losses.append(episode_loss / 50)
                
                if episode % 100 == 0:
                    logger.info(f"Episode {episode}/{num_episodes}, Avg Reward: {np.mean(episode_rewards[-100:]):.2f}")
            
            self.training_stats = {
                'episodes': num_episodes,
                'avg_reward': float(np.mean(episode_rewards)),
                'avg_loss': float(np.mean(episode_losses))
            }
            
            logger.info("Training completed successfully")
            return self.training_stats
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise

    async def get_recommendations(self, num_posts: int = 10) -> List[Dict]:
        """Get recommendations using the trained model"""
        self.agent.eval_mode()
        
        try:
            with torch.no_grad():
                # Move data to device and get recommendations
                embeddings_device = self.embeddings.to(self.device)
                all_q_values = self.agent.q_network(embeddings_device)
                recommend_scores = all_q_values[:, 1]
                
                # Get top k recommendations
                top_k_indices = torch.argsort(recommend_scores, descending=True)[:num_posts]
                
                recommendations = []
                for idx in top_k_indices:
                    post = self.telegram_df.iloc[idx.item()]
                    recommendations.append({
                        'text': post['text'],
                        'score': float(recommend_scores[idx].cpu()),
                        'original_score': float(post[1])
                    })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting recommendations: {str(e)}")
            raise

    def get_stats(self) -> Dict:
        """Get current model statistics"""
        return {
            **self.training_stats,
            'device': str(self.device),
            'epsilon': self.agent.epsilon
        }

    def _calculate_reward(self, state: torch.Tensor, action: int) -> float:
        """Calculate reward for state-action pair"""
        post_idx = self._find_closest_embedding(state)
        actual_score = self.telegram_df.iloc[post_idx]['normalized_score']
        
        if action == 1:
            return float(actual_score)
        else:
            return float(-0.1 * actual_score)

    def _get_next_state(self, action: int) -> torch.Tensor:
        """Get next state based on action"""
        idx = np.random.randint(len(self.embeddings))
        return self.embeddings[idx].clone()

    def _find_closest_embedding(self, state: torch.Tensor) -> int:
        """Find the index of the closest embedding to the given state"""
        state_cpu = state.cpu()
        distances = torch.norm(self.embeddings.cpu() - state_cpu, dim=1)
        return torch.argmin(distances).item()
