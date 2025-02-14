import os
from typing import List, Dict, Optional

from typing_extensions import TypedDict

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

import spotipy
from spotipy.oauth2 import SpotifyOAuth


# 1. Schema Definition (TypedDict)
class TrackMetadata(TypedDict):
    id: str  # Spotify track ID
    name: str
    artist: str
    genre: Optional[str]  # Genre might be None
    bpm: Optional[int]
    key: Optional[str]
    # ... other relevant metadata


class UserQuery(TypedDict):
    query: str
    playlist_name: Optional[str]  # User might not provide a name initially


class GraphState(TypedDict):  # Overall graph state
    access_token: Optional[str]
    user_query: Optional[UserQuery]
    track_metadata: Optional[List[TrackMetadata]]
    selected_tracks: Optional[List[str]]  # List of track IDs
    new_playlist_id: Optional[str]


# 2. Node Functions

def authenticate(state: GraphState) -> GraphState:
    """Authenticates with Spotify API and gets access token."""
    scope = "playlist-modify-public playlist-read-private playlist-modify-private playlist-modify-public"  # Add more scopes as needed
    sp_oauth = SpotifyOAuth(client_id=os.environ.get("SPOTIFY_CLIENT_ID"),
                            client_secret=os.environ.get("SPOTIFY_CLIENT_SECRET"),
                            redirect_uri="http://localhost:8888/callback",  # Or your redirect URI
                            scope=scope)

    # In a real app, you would handle the OAuth flow (redirect to Spotify, get code, exchange for token)
    # For this example, we'll assume the token is already available (e.g., from a previous run)
    # You can store the token in a file or database for persistence

    token_info = sp_oauth.get_cached_token() # Get the token from cache

    if token_info:
        access_token = token_info['access_token']
        print("Token from Cache")
    else:
        auth_url = sp_oauth.get_authorize_url()
        print("Please authorize your app by visiting this URL: ", auth_url)
        response = input("Enter the authorization code from the URL: ")
        token_info = sp_oauth.get_access_token(response)
        access_token = token_info['access_token']
        print("Token from User")

    state["access_token"] = access_token
    return state


# def get_playlist_data(state: GraphState) -> GraphState:
    """Retrieves playlist and track metadata."""
    access_token = state["access_token"]
    sp = spotipy.Spotify(auth=access_token)

    user_query = state["user_query"]
    if not user_query or not user_query.get("query"):
        raise ValueError("User query is missing.")

    query = user_query["query"]
    playlist_name_or_id = query.split("from")[-1].strip()  # Extract playlist name/ID from the query

    try:
      playlist = sp.playlist(playlist_name_or_id) # if the query is a playlist ID
    except:
      playlists = sp.current_user_playlists() # if the query is a playlist name
      playlist_id = None
      for pl in playlists['items']:
          if pl['name'] == playlist_name_or_id:
              playlist_id = pl['id']
              break
      if playlist_id:
        playlist = sp.playlist(playlist_id)
      else:
        raise ValueError(f"Playlist with name or ID '{playlist_name_or_id}' not found.")

    track_metadata: List[TrackMetadata] = []
    for item in playlist['tracks']['items']:
      track = item['track']
      if track:
        track_metadata.append({
            "id": track['id'],
            "name": track['name'],
            "artist": track['artists'][0]['name'],  # Assuming one artist for simplicity
            "genre": None,  # Spotify API doesn't directly provide genre for tracks
            "bpm": None,  # You might need to use a separate API for BPM
            "key": None,  # You might need to use a separate API for key
        })

    state["track_metadata"] = track_metadata
    return state

from langchain.agents import Tool, AgentExecutor, ZeroShotAgent, initialize_agent, AgentType

def get_playlist_data(state: GraphState) -> GraphState:
    """Retrieves playlist and track metadata using LLM for playlist identification."""
    access_token = state["access_token"]
    sp = spotipy.Spotify(auth=access_token)
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)

    user_query = state["user_query"]
    if not user_query or not user_query.get("query"):
        raise ValueError("User query is missing.")

    query = user_query["query"]

    # 1. LLM Tool Call for Playlist Identification
    playlist_tool = Tool(
        name="GetPlaylistID",
        func=lambda playlist_name_or_id: _get_playlist_id(sp, playlist_name_or_id),  # Helper function (see below)
        description="""
        Use this tool to get the playlist ID. Input should be a string which is the name of the playlist, or the ID of the playlist.
        """,
    )

    tools = [playlist_tool]
    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

    try:
        playlist_id = agent.run(query)  # LLM decides which tool to call
    except ValueError as e: # if there is no playlist by that name
        raise ValueError(f"Playlist not found: {e}")
    except Exception as e:
        raise ValueError(f"Error finding playlist: {e}")

    playlist = sp.playlist(playlist_id)

    track_metadata: List[TrackMetadata] = []
    for item in playlist['tracks']['items']:
      track = item['track']
      if track:
        track_metadata.append({
            "id": track['id'],
            "name": track['name'],
            "artist": track['artists'][0]['name'],  # Assuming one artist for simplicity
            "genre": None,  # Spotify API doesn't directly provide genre for tracks
            "bpm": None,  # will need to use a  separate API endpoint for BPM
            "key": None,  # for key as well {refer web API specs}
        })

    state["track_metadata"] = track_metadata
    return state

def _get_playlist_id(sp: spotipy.Spotify, playlist_name_or_id: str) -> str:
    """Helper function to get playlist ID from name or ID."""
    try:
        playlist = sp.playlist(playlist_name_or_id)  # Try if it's an ID
        return playlist['id']
    except:
        playlists = sp.current_user_playlists()
        for pl in playlists['items']:
            if pl['name'] == playlist_name_or_id:
                return pl['id']
        raise ValueError(f"Playlist with name or ID '{playlist_name_or_id}' not found.")



def call_llm(state: GraphState) -> GraphState:
    """Calls the LLM to process the query and select tracks, using few-shot prompting."""

    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)  # Adjust temperature as needed

    user_query = state["user_query"]
    track_metadata = state["track_metadata"]

    # Few-shot prompt examples (improve and expand these!)
    few_shot_examples = [
        {"query": "Suggest a few house music songs from popular artists.", "response": ["track_id_1", "track_id_2"]},  # Replace with actual track IDs
        {"query": "Give me a playlist of Deep house songs in this playlist.", "response": ["track_id_3", "track_id_4"]},
        {"query": "Filter and pick the songs that have high danceability.", "response": ["track_id_5", "track_id_6"]},
        {"query": "Find tracks with a BPM between 120 and 128.", "response": ["track_id_7", "track_id_8"]},
        {"query": "I want some Tech House tracks with a driving bassline.", "response": ["track_id_9", "track_id_10"]},
        {"query": "Suggest some melodic house tracks with a focus on piano melodies.", "response": ["track_id_11", "track_id_12"]}
    ]

    prompt = f"""
    User Query: {user_query['query']}

    Track Metadata:
    {track_metadata}

    Few-Shot Examples:
    """
    for example in few_shot_examples:
        prompt += f"Query: {example['query']}\nResponse: {example['response']}\n\n"

    # Add the current user query to the prompt
    prompt += f"Query: {user_query['query']}\nResponse:"  # Let the LLM fill in the response

    llm_response: BaseMessage = llm.invoke(prompt)
    print(llm_response.content) # print the llm response for debugging

    # Parse LLM response to get selected track IDs (very important!)
    selected_tracks = _extract_track_ids(llm_response.content)  # Implement this parsing function!

    state["selected_tracks"] = selected_tracks
    return state


def create_playlist(state: GraphState) -> GraphState:
    """Creates a new Spotify playlist."""
    access_token = state["access_token"]
    sp = spotipy.Spotify(auth=access_token)

    user_query = state["user_query"]
    playlist_name = user_query.get("playlist_name") # get the playlist name if the user has provided it

    if not playlist_name: # if the user hasn't provided the playlist name
      playlist_name = "EDM_Playlist_" + str(len(sp.current_user_playlists()['items']) + 1) # create a default playlist name

    new_playlist = sp.user_playlist_create(sp.me()['id'], playlist_name, public=False, collaborative=False, description="A playlist created by the EDM DJ Chatbot")
    playlist_id = new_playlist['id']
    track_ids = state['selected_tracks']

    sp.playlist_add_items(playlist_id, track_ids)

    state["new_playlist_id"] = playlist_id
    return state


# 3. Helper Function (Parsing Track IDs from LLM Response)
def _extract_track_ids(llm_response: str) -> List[str]:
    """Parses track IDs from the LLM's response.  This is crucial and will depend on how you prompt the LLM."""
    # Example (very basic - adapt as needed):
    lines = llm_response.strip().split('\n')
    track_ids = [line.strip() for line in lines if line.strip()]  # Assuming each line is a track ID
    return track_ids


# 4. LangGraph Implementation
graph = StateGraph(GraphState)

graph.add_node("authenticate", authenticate)
graph.add_node("get_playlist_data", get_playlist_data)
graph.add_node("call_llm", call_llm)
graph.add_node("create_playlist", create_playlist)

graph.add_edge("authenticate", "get_playlist_data")
graph.add_edge("get_playlist_data", "call_llm")
graph.add_edge("call_llm", "create_playlist")
graph.set_entry_point("authenticate")

# 5. Example Usage
if __name__ == "__main__":
    user_query: UserQuery = {"query": "Suggest me some good House music songs from my playlist Deep House Vibes with a bpm of  range 115-130", "playlist_name": "My_New_Playlist"}  # Get user input
    initial_state: GraphState = {"access_token": None, "user_query": user_query, "track_metadata": None, "selected_tracks": None, "new_playlist_id": None}
    final_state = graph.run(initial_state)

    print("Created playlist:", final_state["new_playlist_id"])