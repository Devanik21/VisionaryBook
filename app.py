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

# Page config
st.set_page_config(
    page_title="VisionaryBook",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for professional dark theme
def load_custom_css():
    st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Variables */
    :root {
        --primary-bg: #0f0f0f;
        --secondary-bg: #1a1a1a;
        --accent-bg: #262626;
        --hover-bg: #333333;
        --primary-text: #ffffff;
        --secondary-text: #b3b3b3;
        --accent-text: #4CAF50;
        --border-color: #333333;
        --shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        --accent-gradient: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
        --danger-color: #f44336;
        --warning-color: #ff9800;
        --info-color: #2196f3;
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Main container styling */
    .stApp {
        background: var(--primary-bg);
        font-family: 'Inter', sans-serif;
        color: var(--primary-text);
    }
    
    /* Header styling */
    .main-header {
        background: var(--secondary-bg);
        padding: 1.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        border: 1px solid var(--border-color);
        box-shadow: var(--shadow);
    }
    
    .header-title {
        font-size: 2.5rem;
        font-weight: 700;
        background: var(--accent-gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    
    .header-subtitle {
        font-size: 1.1rem;
        color: var(--secondary-text);
        margin-top: 0.5rem;
        font-weight: 400;
    }
    
    /* Upload section */
    .upload-section {
        background: var(--secondary-bg);
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        border: 2px dashed var(--border-color);
        margin-bottom: 2rem;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .upload-section:hover {
        border-color: var(--accent-text);
        transform: translateY(-2px);
        box-shadow: var(--shadow);
    }
    
    .upload-icon {
        font-size: 3rem;
        color: var(--accent-text);
        margin-bottom: 1rem;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    
    /* Card styling */
    .feature-card {
        background: var(--secondary-bg);
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        border: 1px solid var(--border-color);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .feature-card:hover {
        transform: translateY(-4px);
        box-shadow: var(--shadow);
        border-color: var(--accent-text);
    }
    
    .feature-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 4px;
        background: var(--accent-gradient);
    }
    
    /* Result sections */
    .result-section {
        background: var(--secondary-bg);
        padding: 2rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        border: 1px solid var(--border-color);
        box-shadow: var(--shadow);
    }
    
    .section-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: var(--primary-text);
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .section-content {
        color: var(--secondary-text);
        line-height: 1.6;
        font-size: 1rem;
    }
    
    /* Button styling */
    .stButton > button {
        background: var(--accent-gradient);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 2px 10px rgba(76, 175, 80, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 20px rgba(76, 175, 80, 0.4);
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div {
        background: var(--accent-bg);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        color: var(--primary-text);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: var(--secondary-bg);
    }
    
    .sidebar-content {
        background: var(--accent-bg);
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    
    /* Flashcard styling */
    .flashcard {
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
    }
    
    .flashcard:hover {
        transform: rotateY(5deg);
        box-shadow: var(--shadow);
    }
    
    .flashcard-front {
        font-size: 1.2rem;
        font-weight: 600;
        color: var(--primary-text);
    }
    
    .flashcard-back {
        font-size: 1rem;
        color: var(--secondary-text);
        line-height: 1.6;
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background: var(--accent-gradient);
    }
    
    /* Stats styling */
    .stat-container {
        display: flex;
        gap: 1rem;
        margin-bottom: 2rem;
    }
    
    .stat-card {
        background: var(--secondary-bg);
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        flex: 1;
        border: 1px solid var(--border-color);
    }
    
    .stat-number {
        font-size: 2rem;
        font-weight: 700;
        color: var(--accent-text);
    }
    
    .stat-label {
        font-size: 0.9rem;
        color: var(--secondary-text);
        margin-top: 0.5rem;
    }
    
    /* Audio player styling */
    audio {
        width: 100%;
        height: 40px;
        border-radius: 8px;
    }
    
    /* Loading animation */
    .loading-spinner {
        display: flex;
        justify-content: center;
        align-items: center;
        margin: 2rem 0;
    }
    
    .spinner {
        width: 40px;
        height: 40px;
        border: 4px solid var(--border-color);
        border-top: 4px solid var(--accent-text);
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Mobile responsiveness */
    @media (max-width: 768px) {
        .header-title {
            font-size: 2rem;
        }
        
        .stat-container {
            flex-direction: column;
        }
        
        .upload-section {
            padding: 1.5rem;
        }
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--primary-bg);
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--accent-text);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #45a049;
    }
    
    /* Notification styling */
    .notification {
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid;
    }
    
    .notification.success {
        background: rgba(76, 175, 80, 0.1);
        border-left-color: var(--accent-text);
        color: var(--accent-text);
    }
    
    .notification.error {
        background: rgba(244, 67, 54, 0.1);
        border-left-color: var(--danger-color);
        color: var(--danger-color);
    }
    
    .notification.warning {
        background: rgba(255, 152, 0, 0.1);
        border-left-color: var(--warning-color);
        color: var(--warning-color);
    }
    
    .notification.info {
        background: rgba(33, 150, 243, 0.1);
        border-left-color: var(--info-color);
        color: var(--info-color);
    }
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
        
        session_record = {
            'id': session_id,
            'timestamp': datetime.datetime.now().isoformat(),
            'duration': session_data.get('duration', 0),
            'cards_studied': session_data.get('cards_studied', 0),
            'correct_answers': session_data.get('correct_answers', 0)
        }
        
        self.sessions_db.insert(session_record)
        return session_id
    
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

# AI Analysis Engine
class AIAnalysisEngine:
    def __init__(self):
        self.setup_gemini()
    
    def setup_gemini(self):
        """Setup Gemini API"""
        if 'gemini_api_key' not in st.session_state:
            st.session_state.gemini_api_key = None
        
        if st.session_state.gemini_api_key:
            genai.configure(api_key=st.session_state.gemini_api_key)
            self.model = genai.GenerativeModel('gemma-3-27b-it')
        else:
            self.model = None
    
    def analyze_image(self, image: Image.Image, language: str = "English", 
                     category: str = "General") -> Dict:
        """Analyze image using Gemini API"""
        if not self.model:
            raise ValueError("Gemini API not configured")
        
        # Prepare prompts based on category
        category_prompts = {
            "Plants & Crops": f"""You are a botanical study guide. Analyze this image in {language}. 
            Focus on plant identification, scientific names, growing conditions, uses, and agricultural significance.
            Provide practical information for farmers and gardeners.""",
            
            "Landmarks & Places": f"""You are a travel and history study guide. Analyze this image in {language}.
            Focus on geographical features, historical significance, cultural importance, and interesting facts about the location.
            Provide information useful for tourists and travelers.""",
            
            "Objects & Scenes": f"""You are a comprehensive study guide. Analyze this image in {language}.
            Identify all objects, their purposes, materials, historical context, and practical applications.
            Provide educational insights for students and curious learners.""",
            
            "General": f"""You are a comprehensive study guide. Analyze this image in {language}.
            Identify and explain everything visible, providing educational value and interesting facts."""
        }
        
        base_prompt = category_prompts.get(category, category_prompts["General"])
        
        # Quick summary prompt
        quick_prompt = f"""{base_prompt}
        
        Format your response as a concise bullet-point summary of the main elements in the image.
        Focus on identification and basic facts. Keep each point under 20 words."""
        
        # Detailed description prompt
        detailed_prompt = f"""{base_prompt}
        
        Provide a comprehensive educational analysis covering:
        1. Detailed identification and description
        2. Scientific or technical information
        3. Historical context and significance
        4. Practical applications and uses
        5. Interesting connections and relationships
        
        Format as clear, study-friendly paragraphs with educational value."""
        
        # Fun facts prompt
        facts_prompt = f"""{base_prompt}
        
        Share fascinating, lesser-known facts and trivia about what's shown in the image.
        Include surprising connections, interesting statistics, cultural significance, or remarkable properties.
        Make it engaging and memorable for learners."""
        
        try:
            # Generate quick summary
            quick_response = self.model.generate_content([quick_prompt, image])
            quick_summary = quick_response.text
            
            # Generate detailed description
            detailed_response = self.model.generate_content([detailed_prompt, image])
            detailed_description = detailed_response.text
            
            # Generate fun facts
            facts_response = self.model.generate_content([facts_prompt, image])
            fun_facts = facts_response.text
            
            # Extract tags for categorization
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
    
    def render_header(self):
        """Render application header"""
        st.markdown("""
        <div class="main-header">
            <div class="header-title">
                🌍 VisionaryBook
            </div>
            <div class="header-subtitle">
                Transform every image into a comprehensive study resource
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    def render_api_setup(self):
        """Render Gemini API setup"""
        if not st.session_state.gemini_api_key:
            st.warning("⚠️ Please configure your Gemini API key to start analyzing images.")
            
            with st.expander("🔑 API Configuration", expanded=True):
                api_key = st.text_input(
                    "Enter your Gemini API Key:",
                    type="password",
                    help="Get your API key from Google AI Studio"
                )
                
                if st.button("Save API Key", type="primary"):
                    if api_key:
                        st.session_state.gemini_api_key = api_key
                        self.ai_engine.setup_gemini()
                        st.success("✅ API key saved successfully!")
                        st.rerun()
                    else:
                        st.error("Please enter a valid API key.")
            
            st.info("💡 **How to get your API key:**\n1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)\n2. Create a new API key\n3. Copy and paste it above")
            return False
        
        return True
    
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
        
        return category, language
    
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
                st.image(original_image, use_container_width=True)
            
            with col2:
                st.markdown("**Enhanced for Analysis**")
                st.image(resized_image, use_container_width=True)
            
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
                tags_html = " ".join([f"<span style='background: #4CAF50; color: white; padding: 0.2rem 0.5rem; border-radius: 12px; margin: 0.2rem; display: inline-block;'>{tag}</span>" for tag in analysis_data['tags']])
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
            mime="text/plain"
        )
    
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
        
        conn.commit()
        conn.close()
        
        # Move to next card
        self.next_flashcard()
    
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
            st.markdown("## 📚 Study History")
            
            # Statistics
            stats = self.db_manager.get_statistics()
            
            st.markdown(f"""
            <div class="stat-container">
                <div class="stat-card">
                    <div class="stat-number">{stats['total_analyses']}</div>
                    <div class="stat-label">Analyses</div>
                </div>
            </div>
            <div class="stat-container">
                <div class="stat-card">
                    <div class="stat-number">{stats['total_flashcards']}</div>
                    <div class="stat-label">Flashcards</div>
                </div>
            </div>
            <div class="stat-container">
                <div class="stat-card">
                    <div class="stat-number">{stats['average_accuracy']:.1f}%</div>
                    <div class="stat-label">Accuracy</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("### Recent Analyses")
            
            # Get recent analyses
            recent_analyses = self.db_manager.get_all_analyses(limit=10)
            
            for analysis in recent_analyses:
                with st.expander(f"📷 {analysis['image_name'] or 'Untitled'}", expanded=False):
                    st.markdown(f"**Category:** {analysis['category']}")
                    st.markdown(f"**Language:** {analysis['language']}")
                    st.markdown(f"**Date:** {analysis['timestamp'][:16]}")
                    
                    if st.button("🔍 View Details", key=f"view_{analysis['id']}"):
                        # Load this analysis
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
                                    'tags': json.loads(full_analysis['tags']) if full_analysis['tags'] else []
                                }
                            }
                            st.rerun()
    
    def analyze_image_workflow(self, image: Image.Image, category: str, language: str):
        """Complete image analysis workflow"""
        with st.spinner("🔍 Analyzing image... This may take a moment."):
            # Perform AI analysis
            analysis_data = self.ai_engine.analyze_image(image, language, category)
            
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
                
                st.success("✅ Analysis complete!")
                return True
        
        return False
    
    def run(self):
        """Main application runner"""
        # Load custom CSS
        load_custom_css()
        
        # Render header
        self.render_header()
        
        # Check API setup
        if not self.render_api_setup():
            return
        
        # Render history sidebar
        self.render_history_sidebar()
        
        # Main content area
        if st.session_state.study_mode and st.session_state.current_analysis:
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
                    category, language = self.render_analysis_controls()
                    
                    # Analyze button
                    if st.button("🚀 Start Analysis", type="primary", use_container_width=True):
                        success = self.analyze_image_workflow(processed_image, category, language)
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
