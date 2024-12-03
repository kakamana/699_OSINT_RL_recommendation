from telethon import TelegramClient, events, sync
from telethon.errors import SessionPasswordNeededError, PasswordHashInvalidError
from telethon.tl.types import Channel, User
import asyncio
import sqlite3
from datetime import datetime
import logging
import os
from dotenv import load_dotenv
import re

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Securely get credentials
API_ID = os.getenv('TELEGRAM_API_ID')
API_HASH = os.getenv('TELEGRAM_API_HASH')
PHONE = os.getenv('TELEGRAM_PHONE')
TWO_FA_PASSWORD = os.getenv('TELEGRAM_2FA_PASSWORD') 

# Validate credentials
if not all([API_ID, API_HASH, PHONE]):
    raise ValueError("Missing required environment variables. Please check your .env file.")

def setup_database():
    try:
        conn = sqlite3.connect('telegram_data.db')
        c = conn.cursor()
        
        c.execute('''
            CREATE TABLE IF NOT EXISTS messages
            (id INTEGER PRIMARY KEY,
             channel_id INTEGER,
             channel_name TEXT,
             message_id INTEGER,
             message_text TEXT,
             date TIMESTAMP,
             views INTEGER,
             forwards INTEGER,
             media_type TEXT,
             media_path TEXT,
             created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)
        ''')
        
        conn.commit()
        logger.info("Database setup completed successfully")
        return conn
    except sqlite3.Error as e:
        logger.error(f"Database setup failed: {str(e)}")
        raise

class TelegramScraper:
    def __init__(self):
        self._validate_phone_number(PHONE)
        self.client = TelegramClient('session_name', API_ID, API_HASH)
        self.db_conn = setup_database()

    @staticmethod
    def _validate_phone_number(phone):
        """Validate phone number format"""
        pattern = r'^\+\d{1,15}$'
        
        if not phone:
            raise ValueError("Phone number is required")
            
        if not re.match(pattern, phone):
            raise ValueError(
                "Invalid phone number format. "
                "Must start with + followed by country code and number. "
                "Example: +14155552671"
            )
        
        return phone.strip()

    async def start(self):
        try:
            if not self.client.is_connected():
                await self.client.connect()

            if not await self.client.is_user_authorized():
                logger.info(f"Sending authentication code to {PHONE}")
                try:
                    await self.client.send_code_request(PHONE)
                except Exception as e:
                    if "phone_number_invalid" in str(e):
                        raise ValueError(
                            "Invalid phone number format. "
                            "Please check your phone number in .env file"
                        )
                    raise
                
                try:
                    code = input('Enter the code you received: ')
                    
                    try:
                        await self.client.sign_in(PHONE, code)
                    except SessionPasswordNeededError:
                        # 2FA is enabled
                        if not TWO_FA_PASSWORD:
                            password = input('Two-step verification is enabled. Please enter your password: ')
                        else:
                            password = TWO_FA_PASSWORD
                        
                        try:
                            await self.client.sign_in(password=password)
                        except PasswordHashInvalidError:
                            raise ValueError("Invalid 2FA password provided")
                    
                except Exception as e:
                    if "phone_code_invalid" in str(e):
                        raise ValueError("Invalid code entered")
                    raise
                
                logger.info("Successfully authenticated with Telegram")
            
            logger.info("Client started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start client: {str(e)}")
            raise

    async def scrape_channel(self, channel_username, limit=None):
        try:
            channel = await self.client.get_entity(channel_username)
            logger.info(f"Started scraping channel: {channel_username}")
            
            async for message in self.client.iter_messages(channel, limit=limit):
                message_data = {
                    'channel_id': channel.id,
                    'channel_name': channel_username,
                    'message_id': message.id,
                    'message_text': message.text,
                    'date': message.date,
                    'views': getattr(message, 'views', 0),
                    'forwards': getattr(message, 'forwards', 0),
                    'media_type': self._get_media_type(message),
                    'media_path': await self._download_media(message) if message.media else None
                }
                
                self._store_message(message_data)
                logger.info(f"Scraped message {message.id} from {channel_username}")
                
        except Exception as e:
            logger.error(f"Error scraping channel {channel_username}: {str(e)}")
            raise

    def _get_media_type(self, message):
        if message.photo:
            return 'photo'
        elif message.video:
            return 'video'
        elif message.document:
            return 'document'
        elif message.audio:
            return 'audio'
        return None

    async def _download_media(self, message):
        if message.media:
            try:
                # Create downloads directory if it doesn't exist
                os.makedirs('downloads', exist_ok=True)
                path = await message.download_media(file="downloads/")
                logger.info(f"Media downloaded successfully to {path}")
                return path
            except Exception as e:
                logger.error(f"Error downloading media: {str(e)}")
                return None
        return None

    def _store_message(self, message_data):
        try:
            cursor = self.db_conn.cursor()
            cursor.execute('''
                INSERT INTO messages 
                (channel_id, channel_name, message_id, message_text, date, 
                 views, forwards, media_type, media_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                message_data['channel_id'],
                message_data['channel_name'],
                message_data['message_id'],
                message_data['message_text'],
                message_data['date'],
                message_data['views'],
                message_data['forwards'],
                message_data['media_type'],
                message_data['media_path']
            ))
            self.db_conn.commit()
            logger.debug(f"Stored message {message_data['message_id']} in database")
        except sqlite3.Error as e:
            logger.error(f"Database error: {str(e)}")
            raise

    async def close(self):
        await self.client.disconnect()
        self.db_conn.close()
        logger.info("Scraper shutdown completed")

async def main():
    # List of channels to scrape
    channels = [
        'https://t.me/intelslava',
        '@BellumActaNews'
        # Add more channels as needed
    ]
    
    scraper = TelegramScraper()
    
    try:
        await scraper.start()
        
        for channel in channels:
            try:
                await scraper.scrape_channel(channel, limit=500)  # Adjust limit as needed
                logger.info(f"Completed scraping channel: {channel}")
            except Exception as e:
                logger.error(f"Failed to scrape channel {channel}: {str(e)}")
                continue
                
    except Exception as e:
        logger.error(f"Main execution failed: {str(e)}")
    finally:
        await scraper.close()

if __name__ == "__main__":
    # Create .env file if it doesn't exist
    if not os.path.exists('.env'):
        with open('.env', 'w') as f:
            f.write('''TELEGRAM_API_ID=
TELEGRAM_API_HASH=
TELEGRAM_PHONE=''')
        logger.warning("Created .env file. Please fill in your credentials.")
        exit(1)
        
    asyncio.run(main())