import os
import time
from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyOAuth

# Load environment variables
load_dotenv()

class SpotifyController:
    def __init__(self):
        """
        Initialize the Spotify Controller with necessary scopes.
        """
        client_id = os.getenv("CLIENT_ID")
        client_secret = os.getenv("CLIENT_SECRET")
        redirect_uri = os.getenv("REDIRECT_URI")

        if not all([client_id, client_secret, redirect_uri]):
            raise ValueError("Missing Spotify credentials in .env file.")

        scope = "user-modify-playback-state user-read-playback-state user-read-currently-playing"
        
        try:
            self.sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
                client_id=client_id,
                client_secret=client_secret,
                redirect_uri=redirect_uri,
                scope=scope
            ))
            print("Spotify Controller Initialized.")
        except Exception as e:
            print(f"Error initializing Spotify client: {e}")
            raise

    def test_connection(self):
        """
        Simple test to verify we can talk to Spotify API.
        Returns the current user's display name.
        """
        try:
            user = self.sp.current_user()
            return user['display_name']
        except Exception as e:
            print(f"Connection test failed: {e}")
            return None

    def play(self):
        """Resume playback."""
        try:
            self.sp.start_playback()
            print("Playback started/resumed.")
        except spotipy.exceptions.SpotifyException as e:
            print(f"Could not play: {e}")

    def pause(self):
        """Pause playback."""
        try:
            self.sp.pause_playback()
            print("Playback paused.")
        except spotipy.exceptions.SpotifyException as e:
            print(f"Could not pause: {e}")

    def set_volume(self, volume_percent):
        """Set volume for the active device."""
        try:
            volume_percent = max(0, min(100, int(volume_percent)))
            
            devices = self.sp.devices()
            active_device = next((d for d in devices['devices'] if d['is_active']), None)
            
            device_id = active_device['id'] if active_device else None
            
            if device_id:
                self.sp.volume(volume_percent, device_id=device_id)
                print(f"Volume set to {volume_percent}% on {active_device['name']}")
            else:
                self.sp.volume(volume_percent)
                print(f"Volume set to {volume_percent}% (No active device found, trying default)")
                
        except spotipy.exceptions.SpotifyException as e:
            pass

    def play_pause_track(self):
        try:
            playback = self.sp.current_playback()
            if playback and playback.get('is_playing'):
                self.sp.pause_playback()
                print("Playing")
                return "Playing"
        except Exception as e:
            print(f"Play/Pause Error: {e}")
            return "Error"

    def next_track(self):
        try:
            self.sp.next_track()
            print("Next Track")
            return "Next"
        except Exception:
            return "Error"

    def previous_track(self):
        try:
            self.sp.previous_track()
            print("Previous Track")
            return "Previous"
        except Exception:
            return "Error"

    def start_playlist(self, playlist_uri):
        """Start playing a specific playlist."""
        try:
            self.sp.start_playback(context_uri=playlist_uri)
            print(f"Started playlist: {playlist_uri}")
        except spotipy.exceptions.SpotifyException as e:
            print(f"Could not start playlist: {e}")

    def search_and_play(self, query, search_type='playlist', randomize=False):
        """
        Search for a playlist or track.
        If randomize=True, picks a random item from top 5 results.
        """
        try:
            print(f"Searching for {search_type}: {query}")
            results = self.sp.search(q=query, type=search_type, limit=10)
            
            items = []
            if search_type == 'playlist':
                if results and 'playlists' in results and results['playlists']['items']:
                    items = results['playlists']['items']
            elif search_type == 'track':
                if results and 'tracks' in results and results['tracks']['items']:
                    items = results['tracks']['items']

            if not items:
                print("No results found.")
                return None
            
            if randomize:
                import random
                target = random.choice(items[:5]) # Top 5
            else:
                target = items[0]
            
            device_id = None
            try:
                active_device = self.get_active_device()
                if active_device: device_id = active_device['id']
                else: 
                    devs = self.sp.devices()
                    if devs['devices']: device_id = devs['devices'][0]['id']
            except: pass

            context_uri = target['uri'] if search_type == 'playlist' else None
            uris = [target['uri']] if search_type == 'track' else None
            
            if device_id:
                if search_type == 'playlist': self.sp.start_playback(device_id=device_id, context_uri=context_uri)
                else: self.sp.start_playback(device_id=device_id, uris=uris)
            else:
                if search_type == 'playlist': self.sp.start_playback(context_uri=context_uri)
                else: self.sp.start_playback(uris=uris)

            print(f"Playing: {target['name']}")
            return target['name']

        except Exception as e:
            print(f"Search Error: {e}")
            return None

    def get_active_device(self):
        try:
            devices = self.sp.devices()
            return next((d for d in devices['devices'] if d['is_active']), None)
        except: return None
        
    def set_volume(self, volume):
        try:
            device_id = None
            active = self.get_active_device()
            if active: device_id = active['id']
            
            if device_id:
                self.sp.volume(volume, device_id=device_id)
                print(f"Volume set to {volume}% on {active['name']}")
            else:
                self.sp.volume(volume)
                print(f"Volume set to {volume}% (Default Device)")
        except Exception as e:
            print(f"Volume Error: {e}")

    def play_pause_track(self):
        try:
            playback = self.sp.current_playback()
            if playback and playback.get('is_playing'):
                self.sp.pause_playback()
                print("Pausing playback")
            else:
                devices = self.sp.devices()
                active_device = next((d for d in devices['devices'] if d['is_active']), None)
                
                if active_device:
                    self.sp.start_playback()
                    print(f"Resuming playback on {active_device['name']}")
                elif devices['devices']:
                    # Force first available
                    first_device = devices['devices'][0]
                    self.sp.start_playback(device_id=first_device['id'])
                    print(f"Resuming playback on {first_device['name']} (Auto-Selected)")
                else:
                    print("No Spotify devices found.")
        except Exception as e:
            print(f"Play/Pause Error: {e}")

if __name__ == "__main__":
    print("--- Spotify Controller Phase 0 Test ---")
    try:
        controller = SpotifyController()
        user_name = controller.test_connection()
        
        if user_name:
            print(f"Connected as: {user_name}")
            print("\nInstructions:")
            print("'p' = Pause")
            print("'r' = Resume/Play")
            print("'v' = Set Volume")
            print("'q' = Quit")
            
            while True:
                cmd = input("\nEnter command: ").strip().lower()
                
                if cmd == 'p':
                    controller.pause()
                elif cmd == 'r':
                    controller.play()
                elif cmd == 'v':
                    try:
                        vol = input("Enter volume (0-100): ")
                        controller.set_volume(vol)
                    except ValueError:
                        print("Invalid number.")
                elif cmd == 'q':
                    print("Exiting...")
                    break
                else:
                    print("Unknown command.")
        else:
            print("Failed to connect to Spotify.")

    except Exception as e:
        print(f"Fatal error: {e}")
