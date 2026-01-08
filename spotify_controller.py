import os
import time
import logging
import spotipy
from dotenv import load_dotenv
from spotipy.oauth2 import SpotifyOAuth

# Suppress Spotipy Errors
logging.getLogger("spotipy").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)

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
        Simple test to verify connection to Spotify API.
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
            # Throws 403 or 404 if no active device
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
            # Ensure volume is 0-100
            volume_percent = max(0, min(100, int(volume_percent)))
            
            # Get active device to be sure
            devices = self.sp.devices()
            active_device = next((d for d in devices['devices'] if d['is_active']), None)
            
            device_id = active_device['id'] if active_device else None
            
            if device_id:
                self.sp.volume(volume_percent, device_id=device_id)
                print(f"Volume set to {volume_percent}% on {active_device['name']}")
            else:
                # Fallback to default (will fail if no active device)
                self.sp.volume(volume_percent)
                print(f"Volume set to {volume_percent}% (No active device found, trying default)")
                
        except spotipy.exceptions.SpotifyException as e:
            # print(f"Could not set volume: {e}") 
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
            # Handle "No active device found"
            active = self.get_active_device()
            if not active:
                print("No active device found for Next Track. Attempting transfer...")
                devices = self.sp.devices()
                if devices['devices']:
                    first_dev = devices['devices'][0]['id']
                    self.sp.transfer_playback(device_id=first_dev, force_play=True)
                    time.sleep(1) # Wait for transfer
            
            self.sp.next_track()
            print("Next Track")
            return "Next"
        except Exception as e:
            print(f"Next Track Error: {e}")
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
        Hybrid Search:
        1. Fetch User's Playlists.
        2. Fetch Global Search Results.
        3. Merge and Fuzzy Match.
        """
        try:
            # Handle "No active device found" - Ensure query is valid
            if not query or len(query.strip()) < 2:
                print("Invalid search query. Defaulting to 'popular playlist'.")
                query = "popular playlist"

            original_query = query
            candidates = []
            
            # 1. USER LIBRARY SEARCH (Personalized)
            if search_type == 'playlist':
                try:
                    # Fetch first 50 user playlists (includes Daily Mixes if saved/followed)
                    user_playlists = self.sp.current_user_playlists(limit=50)
                    if user_playlists and user_playlists['items']:
                        candidates.extend(user_playlists['items'])
                        # print(f"Loaded {len(user_playlists['items'])} user playlists.")
                except Exception as e:
                    print(f"User Playlist Fetch Warning: {e}")

            print(f"Searching Global for {search_type}: {query} (Market: Dynamic)")
            try:
                results = self.sp.search(q=query, type=search_type, limit=20)
                if search_type == 'playlist':
                    if results and results.get('playlists') and results.get('playlists').get('items'):
                        candidates.extend(results['playlists']['items'])
                elif search_type == 'track':
                    if results and results.get('tracks') and results.get('tracks').get('items'):
                        candidates.extend(results['tracks']['items'])
            except Exception as e:
                print(f"Global Search Error: {e}")

            if not candidates:
                print(f"Spotify: No results found anywhere for '{original_query}'.")
                return None
            
            # Fuzzy Matching and Selection
            # Filter for relevance using case-insensitive substring match
            
            matches = []
            query_lower = original_query.lower()
            
            # De-duplicate based on URI
            seen_uris = set()
            unique_candidates = []
            for item in candidates:
                if item and item['uri'] not in seen_uris:
                    unique_candidates.append(item)
                    seen_uris.add(item['uri'])
            
            # Filter matches
            for item in unique_candidates:
                if not item: continue
                name = item['name'].lower()
                # Check 1: Exact Match (High Priority)
                if name == query_lower:
                    matches.insert(0, item) # Push to front
                # Check 2: Substring Match (Medium Priority)
                elif query_lower in name:
                    matches.append(item)
            
            # If no matches found via string, rely on raw search order
            # But if there are ANY matches found, prefer them.
            
            using_fallback = False
            if matches:
                final_pool = matches
            else:
                using_fallback = True
                final_pool = unique_candidates
            
            if not final_pool:
                print("No valid items after filtering.")
                return None
            
            # Selection
            target = None
            if randomize:
                import random
                target = random.choice(final_pool[:5]) # Pick from top 5 valid
            else:
                target = final_pool[0]
            
            if using_fallback:
                print(f"Search Warning: Could not find playlist matching '{original_query}'. Playing available content: '{target['name']}'")
            
            # PLAYBACK
            # Ensure an active device or force transfer
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
            
            # Context-Based Smart Radio
            # If playing a song, generate the radio list FIRST, then play as a single context.
            if search_type == 'track':
                radio_uris = self._generate_smart_radio_uris(target)
                if radio_uris:
                    # Combine target + radio
                    uris.extend(radio_uris)
                    print(f"Queue Context Prepared: {len(uris)} tracks.")
            
            try:
                if device_id:
                    if search_type == 'playlist': self.sp.start_playback(device_id=device_id, context_uri=context_uri)
                    else: self.sp.start_playback(device_id=device_id, uris=uris)
                else:
                    if search_type == 'playlist': self.sp.start_playback(context_uri=context_uri)
                    else: self.sp.start_playback(uris=uris)
                
                # VERBOSE LOGGING
                try:
                    artist_name = "Unknown Artist"
                    if search_type == 'track' and target.get('artists'):
                        artist_name = target['artists'][0]['name']
                    elif search_type == 'playlist':
                        # Guard against missing owner
                        owner = target.get('owner')
                        if owner:
                            artist_name = owner.get('display_name', 'Unknown')
                        else:
                            artist_name = "Unknown Owner"
                    
                    print(f"\nSPOTIFY PLAYBACK STARTED")
                    print(f"Title:  {target.get('name', 'Unknown')}")
                    print(f"Artist: {artist_name}")
                    print(f"URI:    {target.get('uri', 'Unknown')}")
                    print(f"Match:  {'Exact/Fuzzy' if matches else 'Raw Search'}")
                    print(f"--------------------------------\n")
                except Exception as log_err:
                    print(f"Log Error: {log_err}")

            except Exception as play_err:
                print(f"Playback failed: {play_err}")
                return None
            
            return target['name']

        except Exception as e:
            print(f"Search Error: {e}")
            return None

    def _generate_smart_radio_uris(self, track_obj):
        """
        Generates and returns a list of URIs using the 5-4-3-2 Strategy:
        - 5 Tracks: Artist's Top Tracks (Familiarity)
        - 4 Tracks: Tracks from the same Album (Context)
        - 3 Tracks: Songs from the same Genre (Vibe)
        - 2 Tracks: Popular songs from the same Release Year (Era)
        
        Returns list[str] of URIs. Does NOT queue them directly.
        """
        try:
            # Metadata
            track_name = track_obj.get('name', 'Unknown')
            track_uri = track_obj['uri']
            
            # Artist
            artist = track_obj['artists'][0] if track_obj.get('artists') else None
            artist_id = artist['id'] if artist else None
            artist_name = artist['name'] if artist else None
            
            # Album
            album = track_obj.get('album')
            album_id = album['id'] if album else None
            release_date = album.get('release_date', '')
            release_year = release_date.split('-')[0] if release_date else None
            
            print(f"Generating Smart Radio for: {track_name} ({artist_name})")
            
            # Ensure a valid country code for top_tracks.
            # Use user's country code dynamically. If that fails, 'US' is the safest fallback.
            user_country = 'US'
            try:
                user = self.sp.me()
                if user and 'country' in user: user_country = user['country']
            except: pass
            
            queue_candidates = []
            seen_uris = {track_uri} # Avoid queuing the current song
            
            # ARTIST: 5 Songs (Top Tracks)
            if artist_id:
                try:
                    top = self.sp.artist_top_tracks(artist_id, country=user_country)
                    if top and 'tracks' in top:
                        filtered = [t for t in top['tracks'] if t['uri'] not in seen_uris]
                        queue_candidates.extend(filtered[:5])
                        for t in filtered[:5]: seen_uris.add(t['uri'])
                except Exception as e: print(f"Radio Step 1 (Artist) Failed: {e}")

            # ALBUM: 4 Songs
            if album_id:
                try:
                    # album_tracks returns SimpleTrack objects
                    alb_tracks = self.sp.album_tracks(album_id, limit=20)
                    if alb_tracks and 'items' in alb_tracks:
                        filtered = [t for t in alb_tracks['items'] if t['uri'] not in seen_uris]
                        queue_candidates.extend(filtered[:4])
                        for t in filtered[:4]: seen_uris.add(t['uri'])
                except Exception as e: print(f"Radio Step 2 (Album) Failed: {e}")

            # GENRE: 3 Songs (Search)
            if artist_id:
                try:
                    full_artist = self.sp.artist(artist_id)
                    genres = full_artist.get('genres', [])
                    if genres:
                        main_genre = genres[0]
                        q_genre = f'genre:"{main_genre}"'
                        genre_res = self.sp.search(q=q_genre, type='track', limit=10)
                        if genre_res and 'tracks' in genre_res:
                            filtered = [t for t in genre_res['tracks']['items'] 
                                    if t['uri'] not in seen_uris]
                            import random
                            random.shuffle(filtered)
                            queue_candidates.extend(filtered[:3])
                            for t in filtered[:3]: seen_uris.add(t['uri'])
                except Exception as e: print(f"Radio Step 3 (Genre) Failed: {e}")

            # YEAR: 2 Popular Songs (Search)
            if release_year:
                try:
                    q_year = f'year:{release_year}'
                    year_res = self.sp.search(q=q_year, type='track', limit=10)
                    if year_res and 'tracks' in year_res:
                        filtered = [t for t in year_res['tracks']['items'] 
                                if t['uri'] not in seen_uris]
                        queue_candidates.extend(filtered[:2])
                        for t in filtered[:2]: seen_uris.add(t['uri'])
                except Exception as e: print(f"Radio Step 4 (Year) Failed: {e}")

            if queue_candidates:
                print(f"Smart Radio: Prepared {len(queue_candidates)} tracks (5-4-3-2 Strategy).")
                return [t['uri'] for t in queue_candidates]
            else:
                print("Smart Radio: Could not generate any tracks.")
                return []

        except Exception as e:
            print(f"Smart Radio Error: {e}. Cleanly continuing.")
            return []

    def get_active_device(self):
        try:
            devices = self.sp.devices()
            return next((d for d in devices['devices'] if d['is_active']), None)
        except: return None
        
    def set_volume(self, volume):
        try:
            # Ensure to target active device
            device_id = None
            active = self.get_active_device()
            if active: device_id = active['id']
            
            if device_id:
                self.sp.volume(volume, device_id=device_id)
                print(f"Volume set to {volume}% on {active['name']}")
            else:
                self.sp.volume(volume) # Default
                print(f"Volume set to {volume}% (Default Device)")
        except Exception as e:
            # print(f"Volume Error: {e}")
            pass

    def play_pause_track(self):
        try:
            playback = self.sp.current_playback()
            if playback and playback.get('is_playing'):
                self.sp.pause_playback()
                print("Pausing playback")
            else:
                # Resume logic with robustness
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

    def toggle_shuffle(self, state):
        """Turn shuffle on (True) or off (False)."""
        try:
            self.sp.shuffle(state)
            print(f"Shuffle {'ENABLED' if state else 'DISABLED'}")
        except Exception as e:
            print(f"Shuffle Error: {e}")

    def transfer_playback(self, device_name):
        """Transfer playback to a device by fuzzy name match."""
        try:
            name_norm = device_name.lower()
            devices = self.sp.devices()
            target_id = None
            
            for d in devices['devices']:
                if name_norm in d['name'].lower():
                    target_id = d['id']
                    print(f"Transferring to: {d['name']}")
                    break
            
            if target_id:
                self.sp.transfer_playback(device_id=target_id, force_play=True)
            else:
                print(f"Device '{device_name}' not found.")
        except Exception as e:
            print(f"Transfer Error: {e}")

    def add_to_queue(self, uri):
        """Add a specific URI to the queue."""
        try:
            self.sp.add_to_queue(uri)
            print(f"Added to queue: {uri}")
        except Exception as e:
            print(f"Queue Error: {e}")

    def seek_track(self, position_ms):
        """Jump to a specific position (ms)."""
        try:
            self.sp.seek_track(position_ms)
            print(f"Seeked to {position_ms}ms")
        except Exception as e:
            print(f"Seek Error: {e}")

    def search_and_queue(self, query):
        """Find a track and add it to queue."""
        print(f"Search & Queue: '{query}'")
        try:
            results = self.sp.search(q=query, limit=1, type='track', market='TW')
            if results['tracks']['items']:
                track = results['tracks']['items'][0]
                uri = track['uri']
                print(f"Queueing: {track['name']} ({uri})")
                self.add_to_queue(uri)
                return track['name']
            else:
                print(f"Queue Error: No track found for '{query}'")
                return None
        except Exception as e:
            print(f"Queue Search Error: {e}")
            return None

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
