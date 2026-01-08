import os
import json
import logging
from dotenv import load_dotenv
import google.generativeai as genai

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GeminiParser")

class GeminiParser:
    def __init__(self):
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            logger.error("GEMINI_API_KEY not found in environment variables.")
            raise ValueError("GEMINI_API_KEY missing")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-2.5-flash")
        
        self.system_prompt = """
        You are an intelligent intent parser for a Spotify voice control system.
        Your job is to convert natural language commands into structured JSON data.
        
        STRICT RULES:
        1. Output MUST be valid JSON only. No markdown, no explanations.
        2. Identify the intent from: 
            PLAY_SONG, PLAY_ARTIST, PLAY_PLAYLIST, PLAY_GENRE_OR_VIBE, PLAY_RADIO,
            PAUSE, RESUME, NEXT_TRACK, PREVIOUS_TRACK, SET_VOLUME,
            SHUFFLE_ON, SHUFFLE_OFF, ADD_TO_QUEUE, SEEK_TO_POSITION, TRANSFER_PLAYBACK
        3. Extract the following fields (use null if not applicable):
            - intent (string, required)
            - search_query (string, Spotify search term)
            - entity_type (string)
            - song (string)
            - artist (string)
            - playlist (string)
            - volume (integer)
            - timestamp_seconds (integer, for seek)
            - device_name (string, for transfer)
            
        EXAMPLES:
        "Play Shape of You" -> {"intent": "PLAY_SONG", "search_query": "Shape of You", "entity_type": "song"}
        "Queue Halo" -> {"intent": "ADD_TO_QUEUE", "search_query": "Halo", "entity_type": "song"}
        "Shuffle on" -> {"intent": "SHUFFLE_ON"}
        "Play on living room speaker" -> {"intent": "TRANSFER_PLAYBACK", "device_name": "living room"}
        "Jump to 1 minute 30" -> {"intent": "SEEK_TO_POSITION", "timestamp_seconds": 90}
        """

    def parse_command(self, text):
        """
        Parses text command using Gemini and returns a JSON dictionary.
        Returns None if parsing fails.
        """
        import time
        retries = 3
        for attempt in range(retries):
            try:
                full_prompt = f"{self.system_prompt}\n\nUser: \"{text}\"\nJSON:"
                response = self.model.generate_content(full_prompt)
                
                # Clean response
                cleaned_text = response.text.strip()
                if cleaned_text.startswith("```json"):
                    cleaned_text = cleaned_text[7:]
                if cleaned_text.endswith("```"):
                    cleaned_text = cleaned_text[:-3]
                
                data = json.loads(cleaned_text)
                logger.info(f"Gemini Parsed: {data}")
                return data
                
            except Exception as e:
                # Silently catch exception to prevent log clutter.
                pass
                
                if "429" in str(e):
                    time.sleep(2 ** attempt)
                else:
                    break
        
        # Log only once at the end
        logger.info("Gemini failed or returned null. Using fallback logic.")
        return None

    def generate_search_query_from_mood(self, mood):
        """
        Interprets a single emotion string (e.g., 'neutral', 'surprise', 'happy')
        and returns a creative Spotify search query.
        """
        prompt = f"""
        Act as a music DJ. I will give you a human emotion.
        You must return a 2-4 word creative Spotify search query that matches this mood.
        Examples:
        - "neutral" -> "lofi study beats"
        - "happy" -> "summer vibez hits"
        - "surprise" -> "experimental hyperpop"
        - "sad" -> "melancholic piano"
        - "fear" -> "dark cinematic suspense"
        
        Input Emotion: "{mood}"
        Search Query (Just the string, no quotes):
        """
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip().replace('"', '')
        except Exception as e:
                return None # Return None to let caller handle fallback

if __name__ == "__main__":
    # Test
    parser = GeminiParser()
    print(parser.parse_command("Play some jazz for studying"))
