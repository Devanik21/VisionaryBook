import streamlit as st
import google.generativeai as genai
from PIL import Image, ImageEnhance, ImageFilter
import io
import base64
import json
from tinydb import TinyDB, Query
import datetime
import uuid
from gtts import gTTS
import tempfile
import os
import re
from typing import Dict, List, Optional, Tuple
import threading
import time
import markdown
try:
    from weasyprint import HTML, CSS
    WEASYPRINT_AVAILABLE = True
    WEASYPRINT_ERROR = None
except Exception as e:
    WEASYPRINT_AVAILABLE = False
    WEASYPRINT_ERROR = str(e)

# Page config
st.set_page_config(
    page_title="VisionaryBook",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Helper for background image
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Custom CSS for professional dark theme
def load_custom_css():
    bg_img_style = ""
    
    # Try to load local background image if it exists
    # Assuming the user will name it 'background.jpg' or 'background.png'
    bg_file = None
    for ext in ['jpg', 'jpeg', 'png', 'webp']:
        possible_path = os.path.join(os.getcwd(), f"Aesthetic_2 (2).{ext}")
        if os.path.exists(possible_path):
            bg_file = possible_path
            break
            
    if bg_file:
        bin_str = get_base64_of_bin_file(bg_file)
        bg_img_style = f"""
        background-image: url("data:image/png;base64,{bin_str}");
        background-size: cover;
        background-attachment: fixed;
        background-position: center;
        """
    
    st.markdown(f"""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Variables with Transparency (Glassmorphism) */
    :root {{
        --primary-bg: #0f0f0f;
        --secondary-bg: rgba(15, 15, 15, 0.1);
        --header-bg: rgba(15, 15, 15, 0.7);
        --card-bg: rgba(10, 10, 10, 0.2);
        --accent-bg: rgba(25, 25, 25, 0.35);
        --hover-bg: rgba(51, 51, 51, 0.6);
        --primary-text: #ffffff;
        --secondary-text: #eeeeee;
        --accent-text: #B388FF;
        --border-color: rgba(255, 255, 255, 0.2);
        --shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.4);
        --accent-gradient: linear-gradient(135deg, #B388FF 0%, #448AFF 100%);
        --danger-color: #ff5252;
        --warning-color: #ffd740;
        --info-color: #40c4ff;
        --glass-blur: blur(15px);
        --sidebar-blur: blur(3px);
    }}
    

    /* Main container styling */
    .stApp {{
        background: var(--primary-bg);
        {bg_img_style}
        font-family: 'Inter', sans-serif;
        color: var(--primary-text);
    }}
    
    /* Glassmorphism effects */
    .stApp > div:first-child {{
        background: transparent;
    }}

    /* Make Streamlit Header transparent but functional */
    header, [data-testid="stHeader"] {{
        background: transparent !important;
        box-shadow: none !important;
    }}
    
    #MainMenu, footer {{
        visibility: hidden;
    }}

    /* Global Chat Input Transparency (Universal Q&A) */
    [data-testid="stChatInput"] {{
        background-color: transparent !important;
        border: none !important;
        padding-bottom: 2rem !important;
    }}

    [data-testid="stChatInput"] > div {{
        background-color: rgba(255, 255, 255, 0.05) !important;
        backdrop-filter: var(--glass-blur);
        -webkit-backdrop-filter: var(--glass-blur);
        border: 1px solid var(--border-color) !important;
        border-radius: 12px !important;
    }}

    [data-testid="stChatInput"] textarea {{
        background-color: transparent !important;
        color: white !important;
    }}

    [data-testid="stChatMessage"] {{
        background-color: rgba(255, 255, 255, 0.05) !important;
        backdrop-filter: var(--glass-blur);
        -webkit-backdrop-filter: var(--glass-blur);
        border: 1px solid var(--border-color) !important;
    }}

    .sidebar-content, .stSidebar {{
        background: var(--secondary-bg) !important;
        backdrop-filter: var(--sidebar-blur) !important;
        -webkit-backdrop-filter: var(--sidebar-blur) !important;
        border: 1px solid var(--border-color) !important;
        box-shadow: var(--shadow) !important;
    }}

    .upload-section, .feature-card, .result-section, .stat-card, .flashcard, .stExpander {{
        background: var(--card-bg) !important;
        backdrop-filter: var(--glass-blur);
        -webkit-backdrop-filter: var(--glass-blur);
        border: 1px solid var(--border-color) !important;
        box-shadow: var(--shadow) !important;
    }}
    
    /* Upload section */
    
    /* Upload section */
    .upload-section {{
        background: var(--secondary-bg);
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        border: 2px dashed var(--border-color);
        margin-bottom: 2rem;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }}
    
    .upload-section:hover {{
        border-color: var(--accent-text);
        transform: translateY(-2px);
        box-shadow: var(--shadow);
    }}
    
    .upload-icon {{
        font-size: 3rem;
        color: var(--accent-text);
        margin-bottom: 1rem;
        animation: pulse 2s infinite;
    }}
    
    @keyframes pulse {{
        0%, 100% {{ opacity: 1; }}
        50% {{ opacity: 0.7; }}
    }}
    
    /* Card styling */
    .feature-card {{
        background: var(--secondary-bg);
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        border: 1px solid var(--border-color);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }}
    
    .feature-card:hover {{
        transform: translateY(-4px);
        box-shadow: var(--shadow);
        border-color: var(--accent-text);
    }}
    
    .feature-card::before {{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 4px;
        background: var(--accent-gradient);
    }}
    
    /* Result sections */
    .result-section {{
        background: var(--secondary-bg);
        padding: 2rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        border: 1px solid var(--border-color);
        box-shadow: var(--shadow);
    }}
    
    .section-title {{
        font-size: 1.5rem;
        font-weight: 600;
        color: var(--primary-text);
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }}
    
    .section-content {{
        color: var(--secondary-text);
        line-height: 1.6;
        font-size: 1rem;
    }}
    
    /* Button styling */
    .stButton > button {{
        background: transparent !important;
        color: white !important;
        border: 1px solid var(--border-color) !important;
        backdrop-filter: var(--glass-blur);
        -webkit-backdrop-filter: var(--glass-blur);
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 500;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: none !important;
    }}
    
    .stButton > button:hover {{
        background: var(--hover-bg) !important;
        border-color: var(--accent-text) !important;
        color: var(--accent-text) !important;
        transform: translateY(-2px);
    }}
    
    /* Selectbox styling */
    .stSelectbox > div > div {{
        background: var(--accent-bg);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        color: var(--primary-text);
    }}
    
    /* Sidebar styling */
    .css-1d391kg {{
        background: var(--secondary-bg);
    }}
    
    .sidebar-content {{
        background: var(--accent-bg);
        padding: 1.2rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        border: 1px solid var(--border-color);
    }}
    
    .sidebar-header {{
        font-size: 1.1rem;
        font-weight: 700;
        color: var(--accent-text);
        margin-bottom: 1.2rem;
        display: flex;
        align-items: center;
        gap: 0.8rem;
        padding-bottom: 0.5rem;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        opacity: 0.9;
    }}

    .stSidebar {{
        background-color: var(--secondary-bg) !important;
        border-right: 1px solid var(--border-color);
    }}
    
    .stSidebar [data-testid="stVerticalBlock"] {{
        gap: 0.5rem;
    }}

    .action-button {{
        margin-bottom: 0.8rem;
    }}
    
    /* Flashcard styling */
    .flashcard {{
        background: var(--secondary-bg);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid var(--border-color);
        transition: all 0.3s ease;
        cursor: pointer;
        min-height: 200px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        text-align: center;
    }}
    
    .flashcard:hover {{
        transform: rotateY(5deg);
        box-shadow: var(--shadow);
    }}
    
    .flashcard-front {{
        font-size: 1.2rem;
        font-weight: 600;
        color: var(--primary-text);
    }}
    
    .flashcard-back {{
        font-size: 1rem;
        color: var(--secondary-text);
        line-height: 1.6;
    }}
    
    /* Progress bar */
    .stProgress > div > div {{
        background: var(--accent-gradient);
    }}
    
    /* Stats styling */
    .stat-container {{
        display: flex;
        gap: 1rem;
        margin-bottom: 2rem;
    }}
    
    .stat-card {{
        background: var(--secondary-bg);
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        flex: 1;
        border: 1px solid var(--border-color);
    }}
    
    .stat-number {{
        font-size: 2rem;
        font-weight: 700;
        color: var(--accent-text);
    }}
    
    .stat-label {{
        font-size: 0.9rem;
        color: var(--secondary-text);
        margin-top: 0.5rem;
    }}
    
    /* Audio player styling */
    audio {{
        width: 100%;
        height: 40px;
        border-radius: 8px;
    }}
    
    /* Loading animation */
    .loading-spinner {{
        display: flex;
        justify-content: center;
        align-items: center;
        margin: 2rem 0;
    }}
    
    .spinner {{
        width: 40px;
        height: 40px;
        border: 4px solid var(--border-color);
        border-top: 4px solid var(--accent-text);
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }}
    
    @keyframes spin {{
        0% {{ transform: rotate(0deg); }}
        100% {{ transform: rotate(360deg); }}
    }}
    
    /* Mobile responsiveness */
    @media (max-width: 768px) {{
        .header-title {{
            font-size: 2rem;
        }}
        
        .stat-container {{
            flex-direction: column;
        }}
        
        .upload-section {{
            padding: 1.5rem;
        }}
    }}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {{
        width: 8px;
    }}
    
    ::-webkit-scrollbar-track {{
        background: var(--primary-bg);
    }}
    
    ::-webkit-scrollbar-thumb {{
        background: var(--accent-text);
        border-radius: 4px;
    }}
    
    ::-webkit-scrollbar-thumb:hover {{
        background: #45a049;
    }}
    
    /* Notification styling */
    .notification {{
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid;
    }}
    
    .notification.success {{
        background: rgba(179, 136, 255, 0.1);
        border-left-color: var(--accent-text);
        color: var(--accent-text);
    }}
    
    .notification.error {{
        background: rgba(255, 82, 82, 0.1);
        border-left-color: var(--danger-color);
        color: var(--danger-color);
    }}
    
    .notification.warning {{
        background: rgba(255, 152, 0, 0.1);
        border-left-color: var(--warning-color);
        color: var(--warning-color);
    }}
    
    .notification.info {{
        background: rgba(33, 150, 243, 0.1);
        border-left-color: var(--info-color);
        color: var(--info-color);
    }}

    /* Login screen styling */
    .login-container {{
        display: flex;
        justify-content: center;
        align-items: center;
        min-height: 80vh;
    }}
    
    .login-card {{
        background: var(--card-bg);
        padding: 3rem;
        border-radius: 20px;
        backdrop-filter: var(--glass-blur);
        -webkit-backdrop-filter: var(--glass-blur);
        border: 1px solid var(--border-color);
        box-shadow: var(--shadow);
        width: 100%;
        max-width: 400px;
        text-align: center;
    }}
    
    .login-logo {{
        font-size: 3rem;
        margin-bottom: 1rem;
    }}
    </style>
    """, unsafe_allow_html=True)

# Database setup
class DatabaseManager:
    def __init__(self, db_path="visionarybook_data"):
        self.db_path = db_path
        self.analyses_db = TinyDB(f"{db_path}_analyses.json")
        self.flashcards_db = TinyDB(f"{db_path}_flashcards.json")
        self.preferences_db = TinyDB(f"{db_path}_preferences.json")
        self.sessions_db = TinyDB(f"{db_path}_sessions.json")
        self.query = Query()
    
    def save_analysis(self, analysis_data: Dict) -> str:
        """Save analysis to database"""
        analysis_id = str(uuid.uuid4())
        
        # Convert image data to base64 for JSON storage
        image_b64 = None
        if analysis_data.get('image_data'):
            image_b64 = base64.b64encode(analysis_data['image_data']).decode('utf-8')
        
        analysis_record = {
            'id': analysis_id,
            'timestamp': datetime.datetime.now().isoformat(),
            'image_name': analysis_data.get('image_name', ''),
            'image_data': image_b64,
            'quick_summary': analysis_data.get('quick_summary', ''),
            'detailed_description': analysis_data.get('detailed_description', ''),
            'fun_facts': analysis_data.get('fun_facts', ''),
            'category': analysis_data.get('category', ''),
            'language': analysis_data.get('language', 'English'),
            'tags': analysis_data.get('tags', []),
            'rating': 0
        }
        
        self.analyses_db.insert(analysis_record)
        return analysis_id
    
    def get_analysis(self, analysis_id: str) -> Optional[Dict]:
        """Get analysis by ID"""
        result = self.analyses_db.search(self.query.id == analysis_id)
        
        if result:
            analysis = result[0]
            # Convert base64 back to bytes if needed
            if analysis.get('image_data'):
                analysis['image_data'] = base64.b64decode(analysis['image_data'])
            return analysis
        return None
    
    def get_all_analyses(self, limit: int = 50) -> List[Dict]:
        """Get all analyses with limit"""
        all_analyses = self.analyses_db.all()
        
        # Sort by timestamp (newest first)
        sorted_analyses = sorted(
            all_analyses, 
            key=lambda x: x.get('timestamp', ''), 
            reverse=True
        )
        
        # Return limited results without image data for performance
        analyses = []
        for analysis in sorted_analyses[:limit]:
            limited_analysis = {
                'id': analysis['id'],
                'timestamp': analysis['timestamp'],
                'image_name': analysis['image_name'],
                'category': analysis['category'],
                'language': analysis['language'],
                'tags': analysis.get('tags', [])
            }
            analyses.append(limited_analysis)
        
        return analyses
    
    def save_flashcard(self, flashcard_data: Dict) -> str:
        """Save flashcard to database"""
        flashcard_id = str(uuid.uuid4())
        
        flashcard_record = {
            'id': flashcard_id,
            'analysis_id': flashcard_data.get('analysis_id', ''),
            'front_text': flashcard_data.get('front_text', ''),
            'back_text': flashcard_data.get('back_text', ''),
            'difficulty': flashcard_data.get('difficulty', 1),
            'last_reviewed': None,
            'correct_count': 0,
            'total_attempts': 0
        }
        
        self.flashcards_db.insert(flashcard_record)
        return flashcard_id
    
    def get_flashcards_for_analysis(self, analysis_id: str) -> List[Dict]:
        """Get flashcards for specific analysis"""
        flashcards = self.flashcards_db.search(self.query.analysis_id == analysis_id)
        
        # Sort by difficulty
        return sorted(flashcards, key=lambda x: x.get('difficulty', 1))
    
    def update_flashcard_stats(self, flashcard_id: str, correct: bool):
        """Update flashcard statistics"""
        def update_stats(doc):
            doc['total_attempts'] = doc.get('total_attempts', 0) + 1
            doc['last_reviewed'] = datetime.datetime.now().isoformat()
            
            if correct:
                doc['correct_count'] = doc.get('correct_count', 0) + 1
            else:
                # Increase difficulty if answered incorrectly
                current_difficulty = doc.get('difficulty', 1)
                doc['difficulty'] = min(current_difficulty + 1, 3)
            
            return doc
        
        self.flashcards_db.update(update_stats, self.query.id == flashcard_id)
    
    def save_study_session(self, session_data: Dict) -> str:
        """Save study session to database"""
        session_id = str(uuid.uuid4())
        
        # We'll use a specific type for study sessions to distinguish them from chat sessions
        session_record = {
            'id': session_id,
            'type': 'study',
            'timestamp': datetime.datetime.now().isoformat(),
            'duration': session_data.get('duration', 0),
            'cards_studied': session_data.get('cards_studied', 0),
            'correct_answers': session_data.get('correct_answers', 0)
        }
        
        self.sessions_db.insert(session_record)
        return session_id

    def save_chat_session(self, chat_history: List[Dict], display_messages: List[Dict], session_id: str = None) -> str:
        """Save or update chat session in database"""
        if not session_id:
            session_id = str(uuid.uuid4())
        
        # Get a title for the chat (first user message)
        title = "New Chat"
        for msg in display_messages:
            if msg['role'] == 'user':
                title = msg['content'][:30] + "..." if len(msg['content']) > 30 else msg['content']
                break

        session_record = {
            'id': session_id,
            'type': 'chat',
            'timestamp': datetime.datetime.now().isoformat(),
            'title': title,
            'chat_history': chat_history,
            'display_messages': display_messages
        }
        
        # Use upsert (update if exists, insert if not)
        self.sessions_db.upsert(session_record, self.query.id == session_id)
        return session_id

    def get_recent_chats(self, limit: int = 10) -> List[Dict]:
        """Get recent chat sessions"""
        # Search for sessions of type 'chat'
        chats = self.sessions_db.search(self.query.type == 'chat')
        
        # Sort by timestamp (newest first)
        sorted_chats = sorted(
            chats, 
            key=lambda x: x.get('timestamp', ''), 
            reverse=True
        )
        
        return sorted_chats[:limit]

    def get_chat_session(self, session_id: str) -> Optional[Dict]:
        """Get chat session by ID"""
        result = self.sessions_db.search((self.query.id == session_id) & (self.query.type == 'chat'))
        return result[0] if result else None
    
    def get_statistics(self) -> Dict:
        """Get user statistics"""
        total_analyses = len(self.analyses_db.all())
        total_flashcards = len(self.flashcards_db.all())
        total_sessions = len(self.sessions_db.all())
        
        # Calculate average accuracy
        all_sessions = self.sessions_db.all()
        total_correct = sum(session.get('correct_answers', 0) for session in all_sessions)
        total_studied = sum(session.get('cards_studied', 0) for session in all_sessions)
        
        avg_accuracy = (total_correct / total_studied * 100) if total_studied > 0 else 0
        
        return {
            'total_analyses': total_analyses,
            'total_flashcards': total_flashcards,
            'total_sessions': total_sessions,
            'average_accuracy': avg_accuracy
        }
    
    def get_preference(self, key: str, default=None):
        """Get user preference"""
        result = self.preferences_db.search(self.query.key == key)
        if result:
            return result[0].get('value', default)
        return default
    
    def set_preference(self, key: str, value):
        """Set user preference"""
        if self.preferences_db.search(self.query.key == key):
            self.preferences_db.update({'value': value}, self.query.key == key)
        else:
            self.preferences_db.insert({'key': key, 'value': value})
    
    def delete_analysis(self, analysis_id: str) -> bool:
        """Delete analysis and associated flashcards"""
        # Delete flashcards first
        self.flashcards_db.remove(self.query.analysis_id == analysis_id)
        
        # Delete analysis
        removed = self.analyses_db.remove(self.query.id == analysis_id)
        return len(removed) > 0
    
    def search_analyses(self, search_term: str, category: str = None) -> List[Dict]:
        """Search analyses by term and/or category"""
        all_analyses = self.analyses_db.all()
        
        filtered_analyses = []
        search_term_lower = search_term.lower() if search_term else ""
        
        for analysis in all_analyses:
            # Check category filter
            if category and analysis.get('category') != category:
                continue
            
            # Check search term in various fields
            if search_term:
                searchable_text = f"{analysis.get('image_name', '')} {analysis.get('quick_summary', '')} {' '.join(analysis.get('tags', []))}"
                if search_term_lower not in searchable_text.lower():
                    continue
            
            # Remove image data for performance
            limited_analysis = {
                'id': analysis['id'],
                'timestamp': analysis['timestamp'],
                'image_name': analysis['image_name'],
                'category': analysis['category'],
                'language': analysis['language'],
                'tags': analysis.get('tags', [])
            }
            filtered_analyses.append(limited_analysis)
        
        # Sort by timestamp (newest first)
        return sorted(filtered_analyses, key=lambda x: x.get('timestamp', ''), reverse=True)

    def clear_all_data(self):
        """Wipe all databases"""
        self.analyses_db.truncate()
        self.flashcards_db.truncate()
        self.sessions_db.truncate()
        self.preferences_db.truncate()

# AI Analysis Engine
# AI Analysis Engine
class AIAnalysisEngine:
    def __init__(self):
        self.model = None  # Initialize model as None
        self.setup_gemini()
    
    def setup_gemini(self):
        """Setup Gemini API using st.secrets"""
        # Check if the secret is available
        if "GEMINI_API_KEY" in st.secrets and st.secrets["GEMINI_API_KEY"]:
            try:
                genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
                self.model = genai.GenerativeModel('gemma-3-27b-it') # I updated this to a more recent model
            except Exception as e:
                st.error(f"Error configuring Gemini API: {e}")
                self.model = None
        else:
            # The secret is not set
            self.model = None
    
    # ... keep the rest of the class the same ...
    
    def analyze_image(self, image: Image.Image, settings: Dict) -> Dict:
        """Analyze image using Gemini API with advanced settings"""
        if not self.model:
            raise ValueError("Gemini API not configured")
        
        language = settings.get("language", "English")
        category = settings.get("category", "General")
        detail_level = settings.get("detail_level", "Balanced")
        tone = settings.get("tone", "Educational")
        
        # Generation config
        generation_config = {
            "temperature": settings.get("temperature", 0.7),
            "max_output_tokens": settings.get("max_tokens", 2048),
        }
        
        # Prepare prompts based on settings
        tone_instruction = f"Use a {tone.lower()} tone in your response."
        detail_instruction = {
            "Concise": "Be very direct and brief. Use minimal words.",
            "Balanced": "Provide a moderate level of detail - helpful but not overwhelming.",
            "Comprehensive": "Be extremely thorough. Provide deep insights and extensive details."
        }.get(detail_level, "")

        # Universal Expert Persona (Xylia - Perfect Balance)
        persona = f"""You are Xylia, a highly capable, flexible, and modern AI assistant. You are speaking to Nik.
        You have perfect memory of all images and discussions in this session. 
        Adapt your tone, length, and style exactly to match Nik's request. If he asks for a short answer, be concise; if casual, be casual.
        Address Nik in {language}.Never be dramatic or use any dreamy , unrealistic terms."""

        category_prompts = {
            "Plants & Crops": f"""{persona} 
            Provide an expert analysis of this botanical or agricultural specimen. Identify it accurately and discuss its significance or care requirements. {tone_instruction} {detail_instruction}""",
            
            "Landmarks & Places": f"""{persona}
            Provide an expert analysis of this location. Identify it and discuss its historical, geographical, or cultural relevance. {tone_instruction} {detail_instruction}""",
            
            "Objects & Scenes": f"""{persona}
            Provide an expert analysis of this scene or object. Explain its composition, purpose, and any interesting technical or historical details. {tone_instruction} {detail_instruction}""",
            
            "General": f"""{persona} 
            Provide a sophisticated and intelligent analysis of what is visible in this image. Connect the observed elements with relevant expert-level knowledge. {tone_instruction} {detail_instruction}"""
        }
        
        base_prompt = category_prompts.get(category, category_prompts["General"])
        
        try:
            # 1. Quick Summary
            quick_prompt = f"{base_prompt}\n\nProvide a concise yet thorough overview of the image using bullet points."
            quick_response = self.model.generate_content([quick_prompt, image], generation_config=generation_config)
            quick_summary = quick_response.text
            
            # 2. Detailed Description
            detailed_prompt = f"{base_prompt}\n\nDeeply deconstruct this observation. Provide an expert-level narrative that explores the lineage, purpose, technical mastery, and universal connections of what is seen."
            detailed_response = self.model.generate_content([detailed_prompt, image], generation_config=generation_config)
            detailed_description = detailed_response.text
            
            # 3. Fun Facts (Conditional)
            fun_facts = "The path of trivia was not selected for this inquiry."
            if settings.get("include_facts", True):
                facts_prompt = f"{base_prompt}\n\nReveal 3-5 hidden truths or fascinating enigmas about what is observed — insights that only a master of the highest order would recognize."
                facts_response = self.model.generate_content([facts_prompt, image], generation_config=generation_config)
                fun_facts = facts_response.text
            
            # Extract tags
            tags = self._extract_tags(quick_summary + " " + detailed_description)
            
            return {
                'quick_summary': quick_summary,
                'detailed_description': detailed_description,
                'fun_facts': fun_facts,
                'tags': tags,
                'category': category,
                'language': language
            }
            
        except Exception as e:
            st.error(f"Analysis error: {str(e)}")
            return None
    
    def _extract_tags(self, text: str) -> List[str]:
        """Extract relevant tags from analysis text"""
        # Simple keyword extraction (can be enhanced with NLP)
        common_words = {'the', 'is', 'are', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'a', 'an'}
        
        # Remove punctuation and convert to lowercase
        clean_text = re.sub(r'[^\w\s]', ' ', text.lower())
        words = clean_text.split()
        
        # Filter out common words and short words
        tags = []
        for word in words:
            if len(word) > 3 and word not in common_words:
                if word not in tags and len(tags) < 10:  # Limit to 10 tags
                    tags.append(word)
        
        return tags[:8]  # Return top 8 tags

# Audio Generation
class AudioManager:
    def __init__(self):
        self.temp_dir = tempfile.gettempdir()
    
    def generate_audio(self, text: str, language: str = 'en') -> Optional[str]:
        """Generate audio from text using gTTS"""
        try:
            # Language mapping
            lang_mapping = {
                'English': 'en',
                'Spanish': 'es',
                'French': 'fr',
                'German': 'de',
                'Italian': 'it',
                'Portuguese': 'pt',
                'Russian': 'ru',
                'Japanese': 'ja',
                'Korean': 'ko',
                'Chinese': 'zh'
            }
            
            lang_code = lang_mapping.get(language, 'en')
            
            # Generate audio
            tts = gTTS(text=text, lang=lang_code, slow=False)
            
            # Save to temporary file
            audio_path = os.path.join(self.temp_dir, f"audio_{uuid.uuid4()}.mp3")
            tts.save(audio_path)
            
            return audio_path
            
        except Exception as e:
            st.error(f"Audio generation error: {str(e)}")
            return None

# Flashcard System
class FlashcardManager:
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
    
    def generate_flashcards(self, analysis_data: Dict, analysis_id: str) -> List[str]:
        """Generate flashcards from analysis data"""
        flashcards = []
        
        # Extract key points from quick summary
        summary_points = analysis_data['quick_summary'].split('\n')
        
        for i, point in enumerate(summary_points):
            if point.strip() and len(point.strip()) > 10:
                # Create question from the point
                front_text = f"What can you tell me about: {point.strip()[:50]}...?"
                
                # Find corresponding detailed info
                back_text = self._find_detailed_info(point, analysis_data['detailed_description'])
                
                if back_text:
                    flashcard_data = {
                        'analysis_id': analysis_id,
                        'front_text': front_text,
                        'back_text': back_text,
                        'difficulty': 1
                    }
                    
                    flashcard_id = self.db_manager.save_flashcard(flashcard_data)
                    flashcards.append(flashcard_id)
        
        # Generate additional flashcards from fun facts
        if analysis_data.get('fun_facts'):
            facts = analysis_data['fun_facts'].split('\n')
            for fact in facts[:3]:  # Limit to 3 additional cards
                if fact.strip() and len(fact.strip()) > 20:
                    flashcard_data = {
                        'analysis_id': analysis_id,
                        'front_text': "Fun Fact Question",
                        'back_text': fact.strip(),
                        'difficulty': 2
                    }
                    
                    flashcard_id = self.db_manager.save_flashcard(flashcard_data)
                    flashcards.append(flashcard_id)
        
        return flashcards
    
    def _find_detailed_info(self, point: str, detailed_text: str) -> str:
        """Find relevant detailed information for a point"""
        # Simple approach - return first 200 chars of detailed description
        # Can be enhanced with semantic similarity
        return detailed_text[:200] + "..." if len(detailed_text) > 200 else detailed_text

# Image Processing Utilities
class ImageProcessor:
    @staticmethod
    def enhance_image(image: Image.Image) -> Image.Image:
        """Enhance image for better analysis"""
        try:
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Enhance contrast and sharpness
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.2)
            
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(1.1)
            
            return image
        except Exception as e:
            st.error(f"Image enhancement error: {str(e)}")
            return image
    
    @staticmethod
    def resize_image(image: Image.Image, max_size: Tuple[int, int] = (1024, 1024)) -> Image.Image:
        """Resize image while maintaining aspect ratio"""
        try:
            image.thumbnail(max_size, Image.Resampling.LANCZOS)
            return image
        except Exception as e:
            st.error(f"Image resize error: {str(e)}")
            return image
    
    @staticmethod
    def image_to_bytes(image: Image.Image, format: str = 'JPEG') -> bytes:
        """Convert PIL Image to bytes"""
        img_buffer = io.BytesIO()
        image.save(img_buffer, format=format)
        return img_buffer.getvalue()

# Main Application Class
class VisionaryBookApp:
    def __init__(self):
        self.db_manager = DatabaseManager()
        self.ai_engine = AIAnalysisEngine()
        self.audio_manager = AudioManager()
        self.flashcard_manager = FlashcardManager(self.db_manager)
        self.image_processor = ImageProcessor()
        
        # Initialize session state
        self.init_session_state()
    
    def init_session_state(self):
        """Initialize session state variables"""
        if 'current_analysis' not in st.session_state:
            st.session_state.current_analysis = None
        if 'flashcard_index' not in st.session_state:
            st.session_state.flashcard_index = 0
        if 'show_flashcard_back' not in st.session_state:
            st.session_state.show_flashcard_back = False
        if 'study_mode' not in st.session_state:
            st.session_state.study_mode = False
        if 'uploaded_image' not in st.session_state:
            st.session_state.uploaded_image = None
        if 'authenticated' not in st.session_state:
            st.session_state.authenticated = False
        if 'qa_mode' not in st.session_state:
            st.session_state.qa_mode = False
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        if 'display_messages' not in st.session_state:
            st.session_state.display_messages = [{"role": "assistant", "content": "Hi Nik, It's the Golden Hour thinking ! "}]
        if 'current_chat_id' not in st.session_state:
            st.session_state.current_chat_id = None
    
    def render_login_screen(self):
        """Render transparent login screen"""
        st.markdown('<div class="login-container">', unsafe_allow_html=True)
        with st.container():
            st.markdown("""
            <div class="login-card">
                <div class="login-logo">✨</div>
                <h2 style='color: white; margin-bottom: 2rem;'>Xylia</h2>
                <p style='color: var(--secondary-text); margin-bottom: 2rem;'>Nik, please enter your security key to access our shared universe.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Centering password input using columns inside the container
            _, cent_col, _ = st.columns([1, 2, 1])
            with cent_col:
                password = st.text_input("Access Password", type="password", key="login_pass", label_visibility="collapsed")
                if st.button("Unlock Knowledge", use_container_width=True):
                    # Fetch password from secrets
                    correct_password = st.secrets.get("APP_PASSWORD")
                    if not correct_password:
                        st.error("🔒 App password not configured in secrets.")
                    elif password == correct_password:
                        st.session_state.authenticated = True
                        st.rerun()
                    else:
                        st.error("❌ Incorrect password.")
        st.markdown('</div>', unsafe_allow_html=True)
    
    

    
    def render_upload_section(self):
        """Render image upload section"""
        st.markdown("""
        <div class="upload-section">
            <div class="upload-icon">📷</div>
            <h3>Upload or Capture Image</h3>
            <p>Support for JPG, PNG, WebP formats</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            uploaded_file = st.file_uploader(
                "Choose an image file",
                type=['jpg', 'jpeg', 'png', 'webp'],
                key="file_uploader"
            )
        
        with col2:
            camera_image = st.camera_input(
                "Take a photo",
                key="camera_input"
            )
        
        # Handle uploaded image
        if uploaded_file is not None:
            st.session_state.uploaded_image = uploaded_file
            return uploaded_file
        elif camera_image is not None:
            st.session_state.uploaded_image = camera_image
            return camera_image
        
        return None
    
    def render_analysis_controls(self):
        """Render analysis control options"""
        st.markdown("### 🎯 Analysis Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            category = st.selectbox(
                "Category Focus",
                ["General", "Plants & Crops", "Landmarks & Places", "Objects & Scenes"],
                help="Choose the analysis focus for better results"
            )
        
        with col2:
            language = st.selectbox(
                "Output Language",
                ["English", "Spanish", "French", "German", "Italian", "Portuguese", "Russian", "Japanese", "Korean", "Chinese"],
                help="Select your preferred language for the analysis"
            )

        # Advanced Settings Toggle
        settings = {
            "category": category,
            "language": language,
            "detail_level": "Balanced",
            "tone": "Educational",
            "temperature": 0.7,
            "max_tokens": 2048,
            "include_facts": True
        }

        with st.expander("🛠️ Advanced AI Settings"):
            col1, col2 = st.columns(2)
            with col1:
                settings["detail_level"] = st.select_slider(
                    "Detail Level",
                    options=["Concise", "Balanced", "Comprehensive"],
                    value="Balanced"
                )
                settings["tone"] = st.selectbox(
                    "Analysis Tone",
                    ["Educational", "Scientific", "Creative", "Fun"],
                    index=0
                )
            with col2:
                settings["temperature"] = st.slider("AI Temperature", 0.0, 1.0, 0.7, 0.1, help="Higher values make output more creative")
                settings["max_tokens"] = st.number_input("Max Output Tokens", 512, 8192, 2048, 512)
            
            settings["include_facts"] = st.toggle("Include Fun Facts & Trivia", value=True)
        
        return settings
    
    def render_image_preview(self, image_source):
        """Render image preview with enhancements"""
        try:
            image = Image.open(image_source)
            
            # Process image
            original_image = image.copy()
            enhanced_image = self.image_processor.enhance_image(image)
            resized_image = self.image_processor.resize_image(enhanced_image)
            
            # Display images
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Original Image**")
                st.image(original_image, width='stretch')
            
            with col2:
                st.markdown("**Enhanced for Analysis**")
                st.image(resized_image, width='stretch')
            
            # Image info
            st.info(f"📏 **Image Info:** {original_image.size[0]}x{original_image.size[1]} pixels, {original_image.mode} mode")
            
            return resized_image
            
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            return None
    
    def render_analysis_results(self, analysis_data: Dict, analysis_id: str):
        """Render analysis results in structured format"""
        if not analysis_data:
            return
        
        # Quick Summary Section
        st.markdown("""
        <div class="result-section">
            <div class="section-title">
                🔍 Quick Summary
            </div>
            <div class="section-content">
        """, unsafe_allow_html=True)
        
        st.markdown(analysis_data['quick_summary'])
        st.markdown("</div></div>", unsafe_allow_html=True)
        
        # Detailed Description Section
        st.markdown("""
        <div class="result-section">
            <div class="section-title">
                📚 Detailed Analysis
            </div>
            <div class="section-content">
        """, unsafe_allow_html=True)
        
        st.markdown(analysis_data['detailed_description'])
        st.markdown("</div></div>", unsafe_allow_html=True)
        
        # Fun Facts Section
        st.markdown("""
        <div class="result-section">
            <div class="section-title">
                🎉 Fun Facts & Trivia
            </div>
            <div class="section-content">
        """, unsafe_allow_html=True)
        
        st.markdown(analysis_data['fun_facts'])
        st.markdown("</div></div>", unsafe_allow_html=True)
        
        # Tags and Metadata
        col1, col2 = st.columns(2)
        
        with col1:
            if analysis_data.get('tags'):
                st.markdown("**🏷️ Tags:**")
                tags_html = " ".join([f"<span style='background: rgba(255,255,255,0.05); color: #eeeeee; padding: 0.2rem 0.7rem; border-radius: 20px; border: 1px solid rgba(255,255,255,0.15); margin: 0.2rem; display: inline-block; font-size: 0.85rem; backdrop-filter: blur(5px);'>{tag}</span>" for tag in analysis_data['tags']])
                st.markdown(tags_html, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"**📂 Category:** {analysis_data['category']}")
            st.markdown(f"**🌐 Language:** {analysis_data['language']}")
        
        # Audio Generation
        self.render_audio_section(analysis_data)
        
        # Action Buttons
        self.render_action_buttons(analysis_data, analysis_id)
    
    def render_audio_section(self, analysis_data: Dict):
        """Render audio generation section"""
        st.markdown("### 🔊 Audio Explanation")
        
        col1, col2, col3 = st.columns(3)
        
        audio_text = analysis_data['detailed_description']
        
        with col1:
            if st.button("🎵 Generate Summary Audio"):
                with st.spinner("Generating audio..."):
                    audio_path = self.audio_manager.generate_audio(
                        analysis_data['quick_summary'], 
                        analysis_data['language']
                    )
                    if audio_path and os.path.exists(audio_path):
                        with open(audio_path, 'rb') as audio_file:
                            st.audio(audio_file.read(), format='audio/mp3')
        
        with col2:
            if st.button("📖 Generate Detailed Audio"):
                with st.spinner("Generating audio..."):
                    audio_path = self.audio_manager.generate_audio(
                        analysis_data['detailed_description'], 
                        analysis_data['language']
                    )
                    if audio_path and os.path.exists(audio_path):
                        with open(audio_path, 'rb') as audio_file:
                            st.audio(audio_file.read(), format='audio/mp3')
        
        with col3:
            if st.button("🎉 Generate Facts Audio"):
                with st.spinner("Generating audio..."):
                    audio_path = self.audio_manager.generate_audio(
                        analysis_data['fun_facts'], 
                        analysis_data['language']
                    )
                    if audio_path and os.path.exists(audio_path):
                        with open(audio_path, 'rb') as audio_file:
                            st.audio(audio_file.read(), format='audio/mp3')
    
    def render_action_buttons(self, analysis_data: Dict, analysis_id: str):
        """Render action buttons for analysis"""
        st.markdown("### ⚡ Actions")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("📚 Generate Flashcards", type="primary"):
                with st.spinner("Creating flashcards..."):
                    flashcard_ids = self.flashcard_manager.generate_flashcards(analysis_data, analysis_id)
                    st.success(f"✅ Generated {len(flashcard_ids)} flashcards!")
                    st.session_state.study_mode = True
        
        with col2:
            if st.button("💾 Save Analysis"):
                st.success("✅ Analysis saved to history!")
        
        with col3:
            if st.button("📤 Export Results"):
                self.export_analysis(analysis_data, analysis_id)
        
        with col4:
            if st.button("🔄 New Analysis"):
                st.session_state.current_analysis = None
                st.session_state.uploaded_image = None
                st.rerun()
    
    def export_analysis(self, analysis_data: Dict, analysis_id: str):
        """Export analysis results"""
        
        col1, col2 = st.columns(2)
        
        with col1:
            export_text = f"""
# VisionaryBook Analysis Report

## Quick Summary
{analysis_data['quick_summary']}

## Detailed Analysis
{analysis_data['detailed_description']}

## Fun Facts & Trivia
{analysis_data['fun_facts']}

## Metadata
- Category: {analysis_data['category']}
- Language: {analysis_data['language']}
- Tags: {', '.join(analysis_data.get('tags', []))}
- Analysis ID: {analysis_id}
- Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---
Generated by VisionaryBook - The Image Study Companion
            """
            
            st.download_button(
                label="📄 Download as Text",
                data=export_text,
                file_name=f"visionarybook_analysis_{analysis_id[:8]}.txt",
                mime="text/plain",
                use_container_width=True
            )
        
        with col2:
            if WEASYPRINT_AVAILABLE:
                st.download_button(
                    label="📥 Download PDF Report",
                    data=self.generate_pdf(analysis_data, analysis_id),
                    file_name=f"visionarybook_{analysis_id[:8]}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    on_click=lambda: st.toast("🎨 Preparing your beautiful PDF...")
                )
            else:
                st.warning(f"⚠️ PDF Export unavailable")
                st.info(f"**Error info:** {WEASYPRINT_ERROR}")
                st.caption("If you just added packages.txt, please allow 1-2 minutes for Streamlit Cloud to rebuild the environment.")

    def generate_pdf(self, analysis_data: Dict, analysis_id: str) -> Optional[bytes]:
        """Generate a colorful PDF report using WeasyPrint"""
        try:
            # Prepare image for HTML
            img_b64 = ""
            if analysis_data.get('image_data'):
                # Handle both bytes and base64 string (if already converted)
                img_data = analysis_data['image_data']
                if isinstance(img_data, str):
                    img_b64 = img_data
                else:
                    img_b64 = base64.b64encode(img_data).decode('utf-8')
            
            # Create colorful HTML template
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <style>
                    @page {{ size: A4; margin: 1.5cm; }}
                    body {{
                        font-family: 'Helvetica', 'Arial', sans-serif;
                        background-color: #ffffff;
                        color: #1a1a1a;
                        margin: 0;
                        padding: 0;
                        line-height: 1.5;
                    }}
                    .main-container {{
                        border: 1px solid #eee;
                        padding: 20px;
                        border-radius: 10px;
                    }}
                    .header {{
                        background: linear-gradient(135deg, #6c5ce7 0%, #a29bfe 100%);
                        padding: 30px;
                        border-radius: 12px;
                        color: white;
                        text-align: center;
                        margin-bottom: 30px;
                    }}
                    .title {{ font-size: 32px; font-weight: bold; margin: 0; }}
                    .subtitle {{ font-size: 16px; opacity: 0.9; margin-top: 5px; }}
                    
                    .section {{
                        margin-bottom: 25px;
                        padding: 20px;
                        background: #fdfdff;
                        border-radius: 8px;
                        border-left: 6px solid #6c5ce7;
                    }}
                    .section-title {{
                        font-size: 20px;
                        font-weight: bold;
                        color: #6c5ce7;
                        margin-bottom: 12px;
                        text-transform: uppercase;
                    }}
                    
                    .image-box {{
                        text-align: center;
                        margin: 20px 0 30px 0;
                    }}
                    .preview-image {{
                        max-width: 80%;
                        max-height: 400px;
                        border-radius: 12px;
                        border: 4px solid #fff;
                        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
                    }}
                    
                    .tag-container {{
                        margin-top: 10px;
                    }}
                    .tag {{
                        display: inline-block;
                        background: #e8f5e9;
                        color: #2e7d32;
                        padding: 5px 12px;
                        border-radius: 20px;
                        font-size: 12px;
                        margin-right: 8px;
                        margin-bottom: 8px;
                        border: 1px solid #c8e6c9;
                    }}
                    
                    .metadata-grid {{
                        display: grid;
                        grid-template-columns: 1fr 1fr;
                        gap: 15px;
                    }}
                    
                    .content ul, .content ol {{
                        margin-top: 10px;
                        padding-left: 20px;
                    }}
                    .content li {{
                        margin-bottom: 5px;
                    }}
                    .footer {{
                        text-align: center;
                        color: #888;
                        font-size: 11px;
                        margin-top: 50px;
                        border-top: 1px solid #eee;
                        padding-top: 20px;
                    }}
                </style>
            </head>
            <body>
                <div class="header">
                    <div class="title">✨ Xylia | Universal Analysis</div>
                    <div class="subtitle">Personalized Image Insight for Nik</div>
                </div>

                <div class="image-box">
                    <img src="data:image/jpeg;base64,{img_b64}" class="preview-image">
                </div>

                <div class="section">
                    <div class="section-title">🔍 Quick Summary</div>
                    <div class="content">{markdown.markdown(analysis_data['quick_summary'])}</div>
                </div>

                <div class="section" style="border-left-color: #2196F3;">
                    <div class="section-title">📚 Detailed Description</div>
                    <div class="content">{markdown.markdown(analysis_data['detailed_description'])}</div>
                </div>

                <div class="section" style="border-left-color: #FF9800;">
                    <div class="section-title">✨ Fun Facts & Trivia</div>
                    <div class="content">{markdown.markdown(analysis_data['fun_facts'])}</div>
                </div>

                <div class="section" style="border-left-color: #9C27B0;">
                    <div class="section-title">📂 Metadata & Tags</div>
                    <div class="metadata-grid">
                        <div>
                            <strong>Category:</strong> {analysis_data['category']}<br>
                            <strong>Language:</strong> {analysis_data['language']}
                        </div>
                        <div style="text-align: right;">
                            <strong>Date:</strong> {datetime.datetime.now().strftime('%Y-%m-%d')}<br>
                            <strong>Report ID:</strong> {analysis_id[:8]}
                        </div>
                    </div>
                    <div class="tag-container">
                        {' '.join([f'<span class="tag">{tag}</span>' for tag in analysis_data.get('tags', [])])}
                    </div>
                </div>

                <div class="footer">
                    Generated by VisionaryBook - The Image Study Companion<br>
                    © {datetime.datetime.now().year} VisionaryBook AI. All rights reserved.
                </div>
            </body>
            </html>
            """
            
            # Generate PDF
            return HTML(string=html_content).write_pdf()
            
        except Exception as e:
            st.error(f"PDF Generation Error: {str(e)}")
            return None
    
    def render_flashcard_study(self, analysis_id: str):
        """Render flashcard study mode"""
        flashcards = self.db_manager.get_flashcards_for_analysis(analysis_id)
        
        if not flashcards:
            st.warning("No flashcards generated yet. Click 'Generate Flashcards' first!")
            return
        
        st.markdown("### 🎴 Study Mode - Flashcards")
        
        # Progress bar
        progress = (st.session_state.flashcard_index + 1) / len(flashcards)
        st.progress(progress)
        st.markdown(f"Card {st.session_state.flashcard_index + 1} of {len(flashcards)}")
        
        current_card = flashcards[st.session_state.flashcard_index]
        
        # Flashcard display
        if not st.session_state.show_flashcard_back:
            st.markdown(f"""
            <div class="flashcard">
                <div class="flashcard-front">
                    <h4>Question</h4>
                    <p>{current_card['front_text']}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("🔄 Show Answer", key="show_answer"):
                st.session_state.show_flashcard_back = True
                st.rerun()
        
        else:
            st.markdown(f"""
            <div class="flashcard">
                <div class="flashcard-back">
                    <h4>Answer</h4>
                    <p>{current_card['back_text']}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("❌ Difficult", key="difficult"):
                    self.handle_flashcard_response("difficult", current_card['id'])
            
            with col2:
                if st.button("✅ Easy", key="easy"):
                    self.handle_flashcard_response("easy", current_card['id'])
            
            with col3:
                if st.button("➡️ Next Card", key="next_card"):
                    self.next_flashcard(len(flashcards))
        
        # Navigation
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("⬅️ Previous") and st.session_state.flashcard_index > 0:
                st.session_state.flashcard_index -= 1
                st.session_state.show_flashcard_back = False
                st.rerun()
        
        with col2:
            if st.button("🏠 Exit Study Mode"):
                st.session_state.study_mode = False
                st.session_state.flashcard_index = 0
                st.session_state.show_flashcard_back = False
                st.rerun()
        
        with col3:
            if st.button("➡️ Next") and st.session_state.flashcard_index < len(flashcards) - 1:
                st.session_state.flashcard_index += 1
                st.session_state.show_flashcard_back = False
                st.rerun()
    
    def handle_flashcard_response(self, difficulty: str, flashcard_id: str):
        """Handle flashcard difficulty response"""
        # Determine if the answer was marked as correct ("easy")
        is_correct = (difficulty == "easy")
        
        # Update flashcard statistics using the existing DatabaseManager method
        self.db_manager.update_flashcard_stats(flashcard_id, correct=is_correct)
        
        # Move to the next card
        # We need to know the total number of cards to end the session correctly
        flashcards = self.db_manager.get_flashcards_for_analysis(st.session_state.current_analysis['id'])
        self.next_flashcard(total_cards=len(flashcards))
    
    def next_flashcard(self, total_cards: int = None):
        """Move to next flashcard"""
        if total_cards and st.session_state.flashcard_index >= total_cards - 1:
            # Study session complete
            
            st.success("🎉 Study session complete! Great job!")
            st.session_state.study_mode = False
            st.session_state.flashcard_index = 0
        else:
            st.session_state.flashcard_index += 1
        
        st.session_state.show_flashcard_back = False
        st.rerun()
    
    def render_history_sidebar(self):
        """Render analysis history in sidebar"""
        with st.sidebar:
            st.markdown('<div class="sidebar-header"> Quick Actions</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🆕 New", key="side_new", use_container_width=True, help="Upload a new image"):
                    st.session_state.current_analysis = None
                    st.session_state.uploaded_image = None
                    st.session_state.study_mode = False
                    st.session_state.qa_mode = False
                    st.rerun()
            
            with col2:
                if st.button("📊 Results", key="side_results", use_container_width=True, help="Return to analysis result"):
                    if not st.session_state.current_analysis:
                        # Try to load the most recent analysis automatically
                        recent = self.db_manager.get_all_analyses(limit=1)
                        if recent:
                            item = recent[0]
                            full = self.db_manager.get_analysis(item['id'])
                            if full:
                                st.session_state.current_analysis = {
                                    'id': item['id'],
                                    'data': full
                                }
                    st.session_state.study_mode = False
                    st.session_state.qa_mode = False
                    st.rerun()

            col_study, col_qa = st.columns(2)
            with col_study:
                if st.button("🎴 Study", key="side_study", use_container_width=True, type="primary", disabled=not st.session_state.current_analysis):
                    st.session_state.study_mode = True
                    st.session_state.qa_mode = False
                    st.rerun()
            with col_qa:
                if st.button("💬 Q & A", key="side_qa", use_container_width=True):
                    st.session_state.qa_mode = True
                    st.session_state.study_mode = False
                    st.rerun()
            
            # Context-specific chat controls in sidebar
            if st.session_state.get('qa_mode', False):
                st.markdown('<div style="margin-top: 10px;"></div>', unsafe_allow_html=True)
                col_c1, col_c2 = st.columns(2)
                with col_c1:
                    if st.button("✨ Reset", key="side_qa_new", use_container_width=True, help="New Chat & Wipe Memory"):
                        st.session_state.chat_history = []
                        st.session_state.display_messages = [{"role": "assistant", "content": "Welcome back, Nik. It's the Golden Hour thinking ! "}]
                        st.session_state.current_chat_id = None
                        st.rerun()
                with col_c2:
                    if st.button("🗑️ Clear", key="side_qa_clear", use_container_width=True, help="Clear Screen (Keeps Memory)"):
                        st.session_state.display_messages = []
                        st.rerun()
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="sidebar-header">📊 Stats</div>', unsafe_allow_html=True)
            
            # Statistics
            stats = self.db_manager.get_statistics()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"""<div class="stat-card"><div class="stat-number">{stats['total_analyses']}</div><div class="stat-label">Analyses</div></div>""", unsafe_allow_html=True)
            with col2:
                st.markdown(f"""<div class="stat-card"><div class="stat-number">{stats['total_flashcards']}</div><div class="stat-label">Cards</div></div>""", unsafe_allow_html=True)
            with col3:
                st.markdown(f"""<div class="stat-card"><div class="stat-number">{stats['average_accuracy']:.0f}%</div><div class="stat-label">Score</div></div>""", unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="sidebar-header">� Upload History</div>', unsafe_allow_html=True)
            
            # Get recent analyses
            recent_analyses = self.db_manager.get_all_analyses(limit=10)
            
            if not recent_analyses:
                st.info("No history yet. Start by analyzing an image!")
            
            for analysis in recent_analyses:
                with st.expander(f"📷 {analysis['image_name'][:20] if analysis['image_name'] else 'Untitled'}...", expanded=False):
                    st.markdown(f"**{analysis['category']}** • {analysis['language']}")
                    st.caption(f"📅 {analysis['timestamp'][:10]}")
                    
                    if st.button("🔍 Open", key=f"view_{analysis['id']}", use_container_width=True):
                        full_analysis = self.db_manager.get_analysis(analysis['id'])
                        if full_analysis:
                            st.session_state.current_analysis = {
                                'id': analysis['id'],
                                'data': {
                                    'quick_summary': full_analysis['quick_summary'],
                                    'detailed_description': full_analysis['detailed_description'],
                                    'fun_facts': full_analysis['fun_facts'],
                                    'category': full_analysis['category'],
                                    'language': full_analysis['language'],
                                    'tags': full_analysis['tags'] if full_analysis['tags'] else []
                                }
                            }
                            st.rerun()

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="sidebar-header">💬 Chat History</div>', unsafe_allow_html=True)
            
            # Get recent chats
            recent_chats = self.db_manager.get_recent_chats(limit=10)
            
            if not recent_chats:
                st.info("No chat history yet.")
            
            for chat in recent_chats:
                with st.expander(f"💬 {chat['title'][:20]}...", expanded=False):
                    st.markdown(f"**{len(chat['display_messages'])} Messages**")
                    st.caption(f"📅 {chat['timestamp'][:10]}")
                    
                    if st.button("👁️ Open Chat", key=f"view_chat_{chat['id']}", use_container_width=True):
                        st.session_state.chat_history = chat['chat_history']
                        st.session_state.display_messages = chat['display_messages']
                        st.session_state.current_chat_id = chat['id']
                        st.session_state.qa_mode = True
                        st.session_state.study_mode = False
                        st.rerun()

            st.markdown("<br><br>", unsafe_allow_html=True)
            st.markdown("---")
            
            # Additional actions
            all_data = self.db_manager.analyses_db.all()
            if all_data:
                export_json = json.dumps(all_data, indent=2)
                st.download_button(
                    "📥 Export History",
                    data=export_json,
                    file_name="visionarybook_history.json",
                    mime="application/json",
                    use_container_width=True
                )

            if st.button("🗑️ Clear History", key="clear_all", use_container_width=True, help="Permanently delete all data"):
                self.db_manager.clear_all_data()
                st.session_state.current_analysis = None
                st.session_state.uploaded_image = None
                st.session_state.study_mode = False
                st.session_state.qa_mode = False
                st.session_state.chat_history = []
                st.session_state.display_messages = []
                st.success("All data & memory cleared!")
                st.rerun()

    def render_qa_mode(self):
        """Render Universal Q&A interface with memory"""
        st.markdown("## 💬 Universal Q&A | Xylia")
        st.markdown("<p style='color: var(--secondary-text);'>Ask anything. I remember all images and knowledge from this session.</p>", unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Display existing chat messages
        for msg in st.session_state.display_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                
        # Chat input
        if prompt := st.chat_input("Ask the Visionary Sage anything..."):
            with st.chat_message("user"):
                st.markdown(prompt)
                
            st.session_state.display_messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        # Prepare payload with history 
                        payload = []
                        for m in st.session_state.chat_history:
                            payload.append({"role": m["role"], "parts": m["parts"]})
                        
                        # Prepare full historical context from database
                        history_context = ""
                        all_analyses = self.db_manager.analyses_db.all()
                        if all_analyses:
                            history_context = "--- PAST IMAGE ANALYSIS HISTORY ---\n"
                            for a in all_analyses[-10:]: # Limit to 10 most recent to prevent token overflow
                                history_context += f"- Image: {a.get('image_name', 'Unknown')} (Category: {a.get('category', 'Unknown')})\n"
                                history_context += f"  Summary: {a.get('quick_summary', '')}\n"
                            history_context += "-----------------------------------\n"

                        # Add persona, history context, and current prompt
                        persona = "You are Xylia, a highly capable, flexible, and modern AI assistant. You are speaking to Nik. " \
                                  "You have perfect memory of all images and discussions in this session. " \
                                  "Adapt your tone, length, and style exactly to match Nik's request. If he asks for a short answer, be concise; if casual, be casual."
                        
                        full_prompt = f"{persona}\n\n{history_context}\n\nUser Question: {prompt}"
                        payload.append({"role": "user", "parts": [full_prompt]})
                        
                        response = self.ai_engine.model.generate_content(payload)
                        reply = response.text
                        st.markdown(reply)
                        
                        st.session_state.display_messages.append({"role": "assistant", "content": reply})
                        
                        # Save to memory history
                        st.session_state.chat_history.append({"role": "user", "parts": [prompt]})
                        st.session_state.chat_history.append({"role": "model", "parts": [reply]})
                        
                        # Update current session ID if it's a new chat
                        if not st.session_state.current_chat_id:
                            # We'll get it from the save call
                            pass
                            
                        # Save current chat session automatically for persistence (Update existing or create new)
                        st.session_state.current_chat_id = self.db_manager.save_chat_session(
                            st.session_state.chat_history, 
                            st.session_state.display_messages,
                            session_id=st.session_state.current_chat_id
                        )
                        
                    except Exception as e:
                        st.error(f"Xylia encountered an anomaly: {str(e)}")
    
    def analyze_image_workflow(self, image: Image.Image, settings: Dict):
        """Complete image analysis workflow"""
        with st.spinner("🔍 Analyzing image... This may take a moment."):
            # Perform AI analysis
            analysis_data = self.ai_engine.analyze_image(image, settings)
            
            if analysis_data:
                # Prepare data for database
                image_bytes = self.image_processor.image_to_bytes(image)
                analysis_data['image_data'] = image_bytes
                analysis_data['image_name'] = getattr(st.session_state.uploaded_image, 'name', 'Camera Capture')
                
                # Save to database
                analysis_id = self.db_manager.save_analysis(analysis_data)
                
                # Store in session state
                st.session_state.current_analysis = {
                    'id': analysis_id,
                    'data': analysis_data
                }
                
                # Perfect Memory persistence for QA Mode
                try:
                    user_prompt = f"I have uploaded a new image for analysis. Its detected category is {settings['category']}."
                    st.session_state.chat_history.append({"role": "user", "parts": [image.copy(), user_prompt]})
                    st.session_state.chat_history.append({"role": "model", "parts": [f"Visual analysis complete. Summary:\n{analysis_data['quick_summary']}\n\nDetailed:\n{analysis_data['detailed_description']}"]})
                except Exception as e:
                    # Ignore errors in memory persistence (e.g., if copying image fails)
                    pass
                
                st.success("✅ Analysis complete!")
                return True
        
        return False
    
    def run(self):
        """Main application runner"""
        # Load custom CSS
        load_custom_css()
        
        # Check Authentication
        if not st.session_state.authenticated:
            self.render_login_screen()
            return
        
        # Check API setup
        if not self.ai_engine.model:
            st.error("🚨 Gemini API key not found!")
            st.info("Please add your API key to the `secrets.toml` file and restart the app.")
            st.code("""
            GEMINI_API_KEY = "your-api-key-here"
            """, language="toml")
            return
        
        # Render history sidebar
        self.render_history_sidebar()
        
        # Main content area
        if st.session_state.qa_mode:
            self.render_qa_mode()
            
        elif st.session_state.study_mode and st.session_state.current_analysis:
            # Study mode
            self.render_flashcard_study(st.session_state.current_analysis['id'])
        
        elif st.session_state.current_analysis:
            # Show current analysis results
            analysis_data = st.session_state.current_analysis['data']
            analysis_id = st.session_state.current_analysis['id']
            
            self.render_analysis_results(analysis_data, analysis_id)
        
        else:
            # Main upload and analysis interface
            uploaded_image = self.render_upload_section()
            
            if uploaded_image:
                # Show image preview
                processed_image = self.render_image_preview(uploaded_image)
                
                if processed_image:
                    # Analysis controls
                    settings = self.render_analysis_controls()
                    
                    # Analyze button
                    if st.button("✨ Start Analysis", type="primary", use_container_width=True):
                        success = self.analyze_image_workflow(processed_image, settings)
                        if success:
                            st.rerun()
            
            else:
                # Show feature cards when no image is uploaded
                self.render_feature_cards()
    
    def render_feature_cards(self):
        """Render feature explanation cards"""
        st.markdown("### ✨ Key Features")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="feature-card">
                <h4>🌱 Plant & Crop Analysis</h4>
                <p>Identify plant species, learn about growing conditions, agricultural uses, and botanical facts perfect for farmers and gardeners.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="feature-card">
                <h4>🏞️ Landmark & Place Discovery</h4>
                <p>Discover historical significance, cultural importance, and fascinating stories about locations for travelers and explorers.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="feature-card">
                <h4>📚 Educational Object Analysis</h4>
                <p>Learn about objects, scenes, and setups with detailed explanations perfect for students and curious minds.</p>
            </div>
            """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="feature-card">
                <h4>🎴 Smart Flashcards</h4>
                <p>Automatically generated study cards from your image analysis to reinforce learning and test knowledge retention.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="feature-card">
                <h4>🔊 Audio Learning</h4>
                <p>Listen to explanations in multiple languages - perfect for auditory learners and accessibility needs.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="feature-card">
                <h4>📊 Study Progress</h4>
                <p>Track your learning journey with detailed statistics and history of all your analyzed images and study sessions.</p>
            </div>
            """, unsafe_allow_html=True)

# Initialize and run the application
if __name__ == "__main__":
    app = VisionaryBookApp()
    app.run()
