import streamlit as st
import pandas as pd
import numpy as np
import random
from PIL import Image, ImageDraw, ImageFont
import requests
from io import BytesIO

# Set page config
st.set_page_config(
    page_title="Advanced Personality Match System", 
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state dengan benar
if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    st.session_state.current_page = "home"
    st.session_state.test_history = []
    st.session_state.ideal_test_page = 0
    st.session_state.ideal_answers = []
    st.session_state.comp_test_done = False
    st.session_state.ideal_user_data = {}
    st.session_state.comp_results = {}

# Enhanced CSS dengan sidebar styling
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #E3F2FD 0%, #FCE4EC 100%);
        background-attachment: fixed;
    }
    
    .main {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        padding: 25px;
        margin: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    [data-testid="stSidebar"] .sidebar-content {
        color: white;
    }
    
    .sidebar-title {
        font-size: 24px;
        font-weight: bold;
        color: white;
        text-align: center;
        margin-bottom: 30px;
        padding: 15px;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .nav-section {
        margin: 25px 0;
        padding: 20px;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .section-title {
        font-size: 16px;
        font-weight: bold;
        color: #FFD700;
        margin-bottom: 15px;
        text-transform: uppercase;
        letter-spacing: 1px;
        text-align: center;
    }
    
    .nav-button {
        width: 100%;
        padding: 12px 15px;
        margin: 8px 0;
        background: rgba(255, 255, 255, 0.15);
        border: none;
        border-radius: 10px;
        color: white;
        text-align: left;
        font-size: 14px;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .nav-button:hover {
        background: rgba(255, 255, 255, 0.25);
        transform: translateX(5px);
    }
    
    .icon-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 10px;
        margin: 15px 0;
    }
    
    .icon-button {
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 10px;
        padding: 15px 10px;
        color: white;
        text-align: center;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .icon-button:hover {
        background: rgba(255, 255, 255, 0.2);
        transform: scale(1.05);
    }
    
    .icon-button .emoji {
        font-size: 24px;
        display: block;
        margin-bottom: 5px;
    }
    
    .icon-button .label {
        font-size: 12px;
        opacity: 0.9;
    }
    
    .user-info {
        text-align: center;
        padding: 15px;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        margin-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ==================== ALGORITHM & COMPATIBILITY SYSTEMS ====================

# Personality compatibility matrix
COMPATIBILITY_MATRIX = {
    "green_flag": {
        "green_flag": 95, "yellow_flag": 75, "red_flag": 40, "black_flag": 20
    },
    "yellow_flag": {
        "green_flag": 75, "yellow_flag": 85, "red_flag": 60, "black_flag": 30
    },
    "red_flag": {
        "green_flag": 40, "yellow_flag": 60, "red_flag": 90, "black_flag": 70
    },
    "black_flag": {
        "green_flag": 20, "yellow_flag": 30, "red_flag": 70, "black_flag": 85
    }
}

# Relationship type based on compatibility score
RELATIONSHIP_TYPES = {
    90: {"type": "Soulmate 💑", "desc": "Kalian sangat cocok! Hubungan yang harmonis dan saling melengkapi."},
    80: {"type": "Bestie 👯‍♀️", "desc": "Teman terbaik! Kalian mengerti satu sama lain dengan sangat baik."},
    70: {"type": "Partner 💞", "desc": "Pasangan yang solid dengan chemistry yang baik."},
    60: {"type": "Penggemar 🤩", "desc": "Saling mengagumi, tapi butuh penyesuaian untuk hubungan serius."},
    50: {"type": "Teman 🫂", "desc": "Cocok sebagai teman, hubungan romantis mungkin kurang ideal."},
    40: {"type": "Kenalan 🤝", "desc": "Bisa akrab sebagai kenalan, tapi ada banyak perbedaan."},
    30: {"type": "Orang Asing 🚶‍♂️", "desc": "Sangat berbeda, lebih baik tetap sebagai orang asing."}
}

# ML Training Function
@st.cache_resource
def train_ml_models():
    # Simulate training data for personality prediction
    np.random.seed(42)
    n_samples = 1000
    
    # Features: [communication, trust, independence, passion, mystery]
    X = np.random.rand(n_samples, 5) * 10
    
    # Labels based on feature combinations
    y = []
    for i in range(n_samples):
        comm, trust, ind, passion, mystery = X[i]
        
        if trust > 7 and comm > 6 and ind > 5:
            y.append("green_flag")
        elif passion > 6 and mystery < 5:
            y.append("yellow_flag") 
        elif passion > 7 and mystery > 4:
            y.append("red_flag")
        else:
            y.append("black_flag")
    
    # Train SVM Model
    from sklearn.svm import SVC
    svm_model = SVC(kernel='rbf', probability=True)
    svm_model.fit(X, y)
    
    return svm_model

# ML Prediction function
def predict_personality_ml(answers):
    # Convert answers to feature vector
    feature_map = {"A": [8, 9, 7, 3, 2], "B": [7, 6, 5, 5, 3], 
                  "C": [4, 5, 6, 8, 4], "D": [3, 4, 8, 2, 9]}
    
    features = np.mean([feature_map[ans] for ans in answers], axis=0)
    
    # Get ML prediction
    svm_model = train_ml_models()
    svm_pred = svm_model.predict([features])[0]
    svm_prob = np.max(svm_model.predict_proba([features]))
    
    return svm_pred, svm_prob

# ==================== DATASET LENGKAP ====================

PERSONALITY_DATA = {
    "green_flag": {
        "name": "🟢 Green Flag - The Healthy Partner",
        "traits": ["respectful", "communicative", "trusting", "supportive", "independent"],
        "description": "Kamu cocok dengan tipe partner yang sehat dan matang! Mereka menghargai privasi, percaya pada pasangan, dan membangun hubungan yang saling mendukung.",
        "celebrities": ["Lee Min Ho","Yang Mi", "Zhao Liying", "Yang Yang", "Song Wei Long", "Dilraba Dilmurat", "Lin Yi", "Park Seo Joon", "IU", "Lee Jong Suk", "Bae Suzy", "Wang Yibo", "Xiao Zhan", "Dylan Wang", "Cha Eun Woo", "Park Bo Young", "Park Hyung Sik", "Kim Go Eun", "Shin Min Ah"],
        "fictional_chars": ["Han Ji Pyeong (Start-Up)", "Kim Shin (Goblin)", "Yoon Se Ri (Crash Landing)", "Kaelus (The Remarried Empress)", "Yuri (See You in My 19th Life)", "Haru (A Business Proposal)", "Tohru Honda (Fruits Basket)", "Tanjiro Kamado (Demon Slayer)", "Zhou Zishu (Word of Honor)", "Lan Wangji (Mo Dao Zu Shi)"],
        "flags": ["Respectful", "Good Communication", "Trusting", "Supportive"],
        "emoji": "💚"
    },
    "yellow_flag": {
        "name": "🟡 Yellow Flag - The Protective One", 
        "traits": ["slightly_jealous", "protective", "affectionate", "sometimes_insecure"],
        "description": "Kamu butuh seseorang yang punya sisi protektif! Tipe ini manis dan perhatian, tapi butuh komunikasi yang baik untuk menjaga keseimbangan!",
        "celebrities": ["Song Kang", "Kim Seon Ho", "Han So Hee", "Angela Baby", "Huang Xiaoming"],
        "fictional_chars": ["Seo Gang Jae (Nevertheless)", "Park Jae Eon (Nevertheless)", "Goo Seung Hyun (Nevertheless)", "Seo Junghyuk (Why Raelina Ended Up at The Duke's Mansion)", "Nine (I Stole The Male Lead's First Night)", "Kyle Leonheimer (I Failed to Oust The Villain)", "Satoru Gojo (Jujutsu Kaisen)", "Kyo Sohma (Fruits Basket)", "Levi Ackerman (Attack On Titan)", "Xie Lian (Heaven Official's Blessing)", "Shen Qiao (Thousand Autumns)", "Damon Salvatore (The Vampire Diaries)", "Spike (Buffy The Vampire Slayer)"],
        "flags": ["Slightly Possessive", "Over-Protective", "Sometimes Insecure"],
        "emoji": "💛"
    },
    "red_flag": {
        "name": "🔴 Red Flag - The Passionate Lover",
        "traits": ["intense", "emotional", "expressive", "spontaneous"],
        "description": "Kamu tertarik dengan hubungan yang penuh passion! Tipe ini membawa warna dalam hidupmu, tapi butuh kesabaran ekstra.",
        "celebrities": ["Johnny Depp, Amber Heard, Armie Hammer, Park Bom, Kim Sae Ron", "Fan Bingbing"],
        "fictional_chars": ["Park Joon Young (The World of Married)", "Kang Ma Roo (Just Between Lovers)", "Sasuke Uchiha (Naruto)", "Natsuo (Domestic Girlfriend)", "Matthias (Cry, or Better Yet, Beg)", "Bastian (Bastian)"],
        "flags": ["Emotional Intensity", "Spontaneous", "Expressive"],
        "emoji": "❤️"
    },
    "black_flag": {
        "name": "⚫ Black Flag - The Mysterious One",
        "traits": ["mysterious", "complex", "deep_thinker", "reserved"],
        "description": "Kamu terpikat dengan aura misterius! Tipe ini punya kedalaman, tapi butuh waktu dan pengertian lebih untuk memahaminya.",
        "celebrities": ["Bill Cosby", "R. Kelly", "Harvey Weinstein", "Jung Joon Young", "Kim Hyun Joong", "Kris Wu", "Zheng Shuang", "Zhao Wei", "Wu Yifan"],
        "fictional_chars": ["Lucifer (Demon's King)", "Iaros (Wannabe U)", "Raniero (My Tyrant Husband is Obsessed with The Wrong Person)", "Leon Winston (Try Begging)", "Oh Joo Wan (Cheese in Trap)", "Yang Se Jong (My Perfect Stranger)", "The Emperor (The Remarried Empress)", "Daniel (The Abandoned Wife)", "Duke Grisham (The Villainess Turns The Hourglass)", "Makima (Chainsaw Man)", "Light Yagami (Death Note)"],
        "flags": ["Mysterious", "Complex Personality", "Reserved"],
        "emoji": "🖤"
    }
}

# Foto-foto untuk setiap personality type
PERSONALITY_IMAGES = {
    "green_flag": [
        "https://i.pinimg.com/1200x/80/b8/59/80b859a0b1e821e98ebd3e6895b6497c.jpg",
        "https://i.pinimg.com/1200x/92/81/ca/9281cae87da4b38db36a0ec3e8486b08.jpg",
        "https://i.pinimg.com/736x/8e/8e/55/8e8e55131fa78d8e3e052667d1b70aaa.jpg",
    ],
    "yellow_flag": [
        "https://i.pinimg.com/736x/83/7b/1f/837b1fdb5a24ab3cf5f2a6d502ea8a00.jpg",
        "https://i.pinimg.com/1200x/98/e8/c8/98e8c8c93dbac439e8fa0a7f3ca1d461.jpg", 
        "https://i.pinimg.com/736x/57/71/17/577117441943fbbcbc255dbb9ec9c934.jpg",
    ],
    "red_flag": [
        "https://i.pinimg.com/736x/c9/0c/19/c90c198f9f61725310d056370acb0f51.jpg", 
        "https://i.pinimg.com/736x/3f/0c/85/3f0c85f4d159c85069c896c73e3acfcd.jpg", 
        "https://i.pinimg.com/736x/1f/d4/a2/1fd4a2dae604919d9c86318be5edd65d.jpg",
    ],
    "black_flag": [
        "https://i.pinimg.com/736x/59/4a/60/594a60ff94ee38f4ec9ce7e0ac62c401.jpg",
        "https://i.pinimg.com/1200x/ba/e4/cc/bae4cc7f123482b95f56697a80520393.jpg", 
        "https://i.pinimg.com/1200x/ae/c5/c5/aec5c574b94192a37610021ee56c509d.jpg",
    ]
}

# ==================== MODERN SIDEBAR NAVIGATION ====================

def main():
    # Modern Sidebar dengan Icon Grid
    with st.sidebar:
        # Header dengan gradient
        st.markdown("""
        <div class="sidebar-title">
            🎭 Personality Match<br>
            <small style="font-size: 14px; opacity: 0.8;">AI Relationship Analysis</small>
        </div>
        """, unsafe_allow_html=True)
        
        # Main Navigation Section
        st.markdown('<div class="nav-section">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🧭 Main Navigation</div>', unsafe_allow_html=True)
        
        if st.button("🏠 Dashboard Home", key="nav_home", use_container_width=True):
            st.session_state.current_page = "home"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Personality Tests Section
        st.markdown('<div class="nav-section">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">💖 Personality Tests</div>', unsafe_allow_html=True)

        # Custom styled buttons dengan icon dan label
        st.markdown("""
        <style>
        .custom-nav-btn {
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 10px;
            padding: 15px 10px;
            color: white;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
            margin: 5px 0;
            width: 100%;
        }
        .custom-nav-btn:hover {
            background: rgba(255, 255, 255, 0.2);
            transform: translateY(-2px);
        }
        .custom-nav-btn .icon {
            font-size: 24px;
            display: block;
            margin-bottom: 5px;
        }
        .custom-nav-btn .label {
            font-size: 12px;
            font-weight: 500;
        }
        </style>
        """, unsafe_allow_html=True)

        # Create custom buttons
        col1, col2 = st.columns(2)

        with col1:
            if st.button("💖", key="nav_ideal_custom", help="Tes Tipe Ideal dengan AI", use_container_width=True):
                st.session_state.ideal_test_page = 0
                st.session_state.ideal_answers = []
                st.session_state.ideal_user_data = {}
                st.session_state.current_page = "ideal_test"
                st.rerun()
            st.markdown('<div style="text-align: center; color: white; font-size: 12px; margin-top: -10px; margin-bottom: 15px;">Tipe Ideal</div>', unsafe_allow_html=True)
            
            if st.button("💑", key="nav_comp_custom", help="Tes Kecocokan Pasangan", use_container_width=True):
                st.session_state.comp_test_done = False
                st.session_state.current_page = "compatibility_test"
                st.rerun()
            st.markdown('<div style="text-align: center; color: white; font-size: 12px; margin-top: -10px; margin-bottom: 15px;">Tes Kecocokan</div>', unsafe_allow_html=True)

        with col2:
            if st.button("🤝", key="nav_match_custom", help="Personality Match", use_container_width=True):
                st.session_state.current_page = "personality_match"
                st.rerun()
            st.markdown('<div style="text-align: center; color: white; font-size: 12px; margin-top: -10px; margin-bottom: 15px;">Personality Match</div>', unsafe_allow_html=True)
            
            if st.button("📊", key="nav_history_custom", help="Riwayat & Analisis", use_container_width=True):
                st.session_state.current_page = "history"
                st.rerun()
            st.markdown('<div style="text-align: center; color: white; font-size: 12px; margin-top: -10px; margin-bottom: 15px;">Riwayat Tes</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)
        
        # Quick Actions Section
        st.markdown('<div class="nav-section">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">⚡ Quick Actions</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Reset", key="quick_reset", use_container_width=True):
                # Reset SEMUA state
                st.session_state.test_history = []
                st.session_state.ideal_test_page = 0
                st.session_state.ideal_answers = []
                st.session_state.comp_test_done = False
                st.session_state.ideal_user_data = {}
                st.session_state.comp_results = {}
                st.success("Semua data telah direset!")
                st.rerun()
                
        with col2:
            if st.button("🎯 Start", key="quick_start", use_container_width=True):
                # Reset state tes sebelum mulai baru
                st.session_state.ideal_test_page = 0
                st.session_state.ideal_answers = []
                st.session_state.ideal_user_data = {}
                st.session_state.current_page = "ideal_test"
                st.rerun()
        
        if st.button("📱 Save Progress", key="quick_save", use_container_width=True):
            st.info("Progress tersimpan secara otomatis!")
            
        st.markdown('</div>', unsafe_allow_html=True)
        
        # User Info Section
        st.markdown('<div class="user-info">', unsafe_allow_html=True)
        if 'user_data' in st.session_state:
            user_data = st.session_state.user_data
            st.write(f"**👤 {user_data.get('nama', 'User')}**")
            test_count = len(st.session_state.get('test_history', []))
            st.write(f"**📊 {test_count} Tests**")
            st.progress(min(test_count/10, 1.0))
        else:
            st.write("**👤 Guest User**")
            st.write("Start test to begin!")
            if st.button("🎮 Get Started", key="start_guest", use_container_width=True):
                # Reset state sebelum mulai
                st.session_state.ideal_test_page = 0
                st.session_state.ideal_answers = []
                st.session_state.ideal_user_data = {}
                st.session_state.current_page = "ideal_test"
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # Page Routing
    current_page = st.session_state.current_page
    
    if current_page == "home":
        show_home()
    elif current_page == "ideal_test":
        show_ideal_type_test()
    elif current_page == "compatibility_test":
        show_compatibility_test()
    elif current_page == "personality_match":
        show_personality_match()
    elif current_page == "history":
        show_history_analysis()

# ==================== HOME PAGE ====================

def show_home():
    st.markdown('<div class="main">', unsafe_allow_html=True)
    
    st.title("🎭 Advanced Personality Match System")
    st.markdown("### **Find Your Perfect Match with AI-Powered Analysis** 🚀")
    
    # Feature Cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="text-align: center; padding: 20px; background: #f0f8ff; border-radius: 15px; margin: 10px; border: 1px solid #e0f0ff;">
            <h3>💖 AI Personality Test</h3>
            <p>Temukan tipe idealmu dengan algoritma Machine Learning</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 20px; background: #fff0f5; border-radius: 15px; margin: 10px; border: 1px solid #ffe6f2;">
            <h3>💑 Smart Compatibility</h3>
            <p>Analisis kecocokan dengan 7 tipe hubungan berbeda</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="text-align: center; padding: 20px; background: #f0fff0; border-radius: 15px; margin: 10px; border: 1px solid #e6ffe6;">
            <h3>📊 Progress Tracking</h3>
            <p>Pantau perkembangan dan dapatkan insights</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Quick Start Section
    st.subheader("🚀 Mulai Perjalanan Personality-mu!")
    
    start_col1, start_col2, start_col3 = st.columns(3)
    with start_col1:
        if st.button("💖 Tes Tipe Ideal", use_container_width=True):
            # Reset state sebelum mulai
            st.session_state.ideal_test_page = 0
            st.session_state.ideal_answers = []
            st.session_state.ideal_user_data = {}
            st.session_state.current_page = "ideal_test"
            st.rerun()
    with start_col2:
        if st.button("💑 Tes Kecocokan", use_container_width=True):
            # Reset state sebelum mulai
            st.session_state.comp_test_done = False
            st.session_state.current_page = "compatibility_test"
            st.rerun()
    with start_col3:
        if st.button("🤝 Personality Match", use_container_width=True):
            st.session_state.current_page = "personality_match"
            st.rerun()
    
    # Stats Preview
    if 'test_history' in st.session_state and st.session_state.test_history:
        st.markdown("---")
        st.subheader("📈 Progress Kamu")
        test_count = len(st.session_state.test_history)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Tests", test_count)
        with col2:
            st.metric("Personality Types", "4")
        with col3:
            st.metric("Relationship Types", "7")
    
    st.markdown('</div>', unsafe_allow_html=True)

# ==================== IDEAL TYPE TEST ====================

def show_ideal_type_test():
    st.markdown('<div class="main">', unsafe_allow_html=True)
    
    st.title("💖 Tes Tipe Ideal dengan AI")
    st.write("Temukan tipe pasangan idealmu dengan **algoritma Machine Learning**!")
    
    if 'ideal_test_page' not in st.session_state:
        st.session_state.ideal_test_page = 0
        st.session_state.ideal_answers = []
    
    IDEAL_QUESTIONS = [
        {
            "question": "Dalam hubungan, yang paling kamu hargai...",
            "options": {
                "A": "Kebebasan dan saling percaya",
                "B": "Perhatian dan komunikasi yang intens", 
                "C": "Passion dan spontanitas",
                "D": "Kedamaian dan stabilitas emosi"
            }
        },
        {
            "question": "Ketika punya masalah dengan pasangan, kamu lebih memilih...",
            "options": {
                "A": "Diskusi langsung dan cari solusi", 
                "B": "Diam sebentar lalu bicara dari hati",
                "C": "Ekspresikan perasaan dengan jujur",
                "D": "Berikan space lalu selesaikan pelan-pelan"
            }
        },
        {
            "question": "Bagaimana idealnya pasangan memperlakukan teman-temanmu?",
            "options": {
                "A": "Ramah tapi tidak terlalu ikut campur",
                "B": "Akrab dan sering hangout bersama",
                "C": "Sedikit protektif dan selektif",
                "D": "Menghormati tapi menjaga batasan"
            }
        },
        {
            "question": "Dalam hal karier dan passion, kamu mengharapkan pasangan...",
            "options": {
                "A": "Supportive dan memahami prioritasmu",
                "B": "Selalu ada untukmu kapanpun dibutuhkan", 
                "C": "Mendorongmu untuk lebih ambisius",
                "D": "Memberimu ruang untuk berkembang sendiri"
            }
        },
        {
            "question": "Ketika ada konflik, sikap seperti apa yang kamu inginkan dari pasangan?",
            "options": {
                "A": "Tenang dan mencari solusi bersama",
                "B": "Emosional tapi jujur tentang perasaan",
                "C": "Intens dan penuh semangat",
                "D": "Bijaksana dan penuh pertimbangan"
            }
        },
        {
            "question": "Seberapa penting romance dalam hubungan bagimu?",
            "options": {
                "A": "Penting, tapi keseimbangan lebih utama",
                "B": "Sangat penting, butuh perhatian terus",
                "C": "Extremely penting, hidup butuh passion!",
                "D": "Tidak terlalu, kedalaman lebih berarti"
            }
        },
        {
            "question": "Bagaimana pandanganmu tentang kepercayaan dalam hubungan?",
            "options": {
                "A": "Modal utama, harus dibangun sejak awal",
                "B": "Penting, tapi butuh bukti dan waktu",
                "C": "Harus diberikan sepenuhnya tanpa ragu",
                "D": "Kompleks, butuh pemahaman mendalam"
            }
        },
        {
            "question": "Idealnya, bagaimana pasangan menghabiskan waktu luang?",
            "options": {
                "A": "Quality time yang meaningful",
                "B": "Intens dan selalu bersama",
                "C": "Spontan dan penuh adventure",
                "D": "Tenang dan penuh refleksi"
            }
        }
    ]
    
    if st.session_state.ideal_test_page == 0:
        st.subheader("🎯 Informasi Diri")
        
        col1, col2 = st.columns(2)
        with col1:
            nama = st.text_input("Nama kamu:")
            umur = st.number_input("Umur:", min_value=15, max_value=60, value=20)
        with col2:
            gender = st.radio("Gender:", ["Laki-laki", "Perempuan", "Lainnya"])
            preference = st.radio("Mencari tipe:", ["Cewek", "Cowok", "Keduanya"])
        
        if st.button("Mulai Tes AI!"):
            if nama.strip():
                st.session_state.ideal_user_data = {"nama": nama, "umur": umur, "gender": gender, "preference": preference}
                st.session_state.ideal_test_page = 1
                st.rerun()
            else:
                st.warning("⚠️ Harap isi nama terlebih dahulu!")
    
    elif 1 <= st.session_state.ideal_test_page <= len(IDEAL_QUESTIONS):
        current_q = IDEAL_QUESTIONS[st.session_state.ideal_test_page - 1]
        
        st.subheader(f"Pertanyaan {st.session_state.ideal_test_page} dari {len(IDEAL_QUESTIONS)}")
        st.write(f"**{current_q['question']}**")
        
        answer = st.radio("Pilih jawaban:", list(current_q['options'].values()), key=f"ideal_q{st.session_state.ideal_test_page}")
        
        if st.button("Lanjut ➡️" if st.session_state.ideal_test_page < len(IDEAL_QUESTIONS) else "🔍 Analisis dengan AI!"):
            for key, value in current_q['options'].items():
                if value == answer:
                    st.session_state.ideal_answers.append(key)
                    break
            
            if st.session_state.ideal_test_page < len(IDEAL_QUESTIONS):
                st.session_state.ideal_test_page += 1
            else:
                st.session_state.ideal_test_page = 99
            st.rerun()
    
    elif st.session_state.ideal_test_page == 99:
        personality, confidence = predict_personality_ml(st.session_state.ideal_answers)
        result_data = PERSONALITY_DATA[personality]
        user_data = st.session_state.ideal_user_data
        
        st.balloons()
        st.title("🎊 Hasil Tes Tipe Ideal!")
        
        st.header(f"Hai, {user_data['nama']}! 🎯")
        st.subheader(f"Tipe Idealmu: {result_data['name']}")
        st.write(f"**🤖 Confidence Level AI: {confidence*100:.1f}%**")
        
        # Progress bar
        progress_values = {"green_flag": 0.25, "yellow_flag": 0.5, "red_flag": 0.75, "black_flag": 1.0}
        st.progress(progress_values[personality])
        
        # Display personality images
        st.subheader("🖼️ Visual Personality Kamu:")
        try:
            personality_images = get_all_personality_images(personality)
            if personality_images:
                cols = st.columns(len(personality_images))
                for idx, img in enumerate(personality_images):
                    with cols[idx]:
                        st.image(img, caption=f"Contoh {idx+1}", use_container_width=True)
            else:
                st.warning("Gambar tidak dapat dimuat. Cek koneksi internet.")
        except Exception as e:
            st.warning(f"Tidak dapat menampilkan gambar: {str(e)}")
        
        # Display result
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🚩 Karakteristik:")
            for flag in result_data['flags']:
                st.write(f"- {flag}")
            
            st.subheader("💫 Sifat-Sifat:")
            for trait in result_data['traits']:
                st.write(f"- {trait.replace('_', ' ').title()}")
                
        with col2:
            st.subheader("🎬 Contoh Karakter Fiksi:")
            for char in result_data['fictional_chars'][:5]:
                st.write(f"- {char}")
            
            st.subheader("⭐ Contoh Selebritas:")
            for celeb in result_data['celebrities'][:5]:
                st.write(f"- {celeb}")
        
        # Detailed analysis
        st.subheader("📊 Analisis Mendalam:")
        
        analysis_col1, analysis_col2 = st.columns(2)
        with analysis_col1:
            st.write("**💪 Kekuatan Hubungan:**")
            if personality == "green_flag":
                st.write("- Hubungan sehat dan seimbang")
                st.write("- Komunikasi terbuka dan jujur")
                st.write("- Saling mendukung dan menghargai")
            elif personality == "yellow_flag":
                st.write("- Perhatian dan protektif")
                st.write("- Hubungan yang intens secara emosional")
                st.write("- Komitmen yang kuat")
            elif personality == "red_flag":
                st.write("- Passion dan excitement tinggi")
                st.write("- Spontanitas dan adventure")
                st.write("- Ekspresi emosi yang bebas")
            else:
                st.write("- Kedalaman emosi dan pemikiran")
                st.write("- Misteri yang menarik")
                st.write("- Hubungan yang meaningful")
        
        with analysis_col2:
            st.write("**💡 Tips & Saran:**")
            if personality == "green_flag":
                st.write("- Pertahankan komunikasi yang sehat")
                st.write("- Beri ruang untuk pertumbuhan individu")
                st.write("- Jaga keseimbangan antara bersama dan mandiri")
            elif personality == "yellow_flag":
                st.write("- Komunikasikan perasaan dengan jelas")
                st.write("- Jangan terlalu posesif")
                st.write("- Bangun kepercayaan secara bertahap")
            elif personality == "red_flag":
                st.write("- Kelola emosi dengan bijak")
                st.write("- Temukan keseimbangan dalam intensitas")
                st.write("- Komunikasikan kebutuhan dengan jelas")
            else:
                st.write("- Buka diri secara bertahap")
                st.write("- Beri waktu untuk proses memahami")
                st.write("- Komunikasikan batasan dengan jelas")
        
        # Save to history
        if 'test_history' not in st.session_state:
            st.session_state.test_history = []
        
        st.session_state.test_history.append({
            "test_type": "Tipe Ideal AI",
            "result": result_data['name'],
            "confidence": f"{confidence*100:.1f}%",
            "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
        })
        
        # Navigation
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Tes Lagi", use_container_width=True):
                st.session_state.ideal_test_page = 0
                st.session_state.ideal_answers = []
                st.rerun()
        with col2:
            if st.button("💑 Tes Kecocokan", use_container_width=True):
                # Reset state tes kecocokan sebelum pindah
                st.session_state.comp_test_done = False
                st.session_state.current_page = "compatibility_test"
                st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

# ==================== COMPATIBILITY TEST ====================

def show_compatibility_test():
    st.markdown('<div class="main">', unsafe_allow_html=True)
    
    st.title("💑 Tes Kecocokan Pasangan")
    st.write("Cek chemistry dan kecocokanmu dengan pasangan atau crush-mu!")
    
    if 'comp_test_done' not in st.session_state:
        st.session_state.comp_test_done = False
    
    if not st.session_state.comp_test_done:
        st.subheader("🎯 Profil Diri & Pasangan")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Profil Kamu:**")
            my_name = st.text_input("Nama kamu:", key="my_name_input")
            my_personality = st.selectbox("Tipe Personality kamu:", 
                ["🟢 Green Flag", "🟡 Yellow Flag", "🔴 Red Flag", "⚫ Black Flag"], key="my_personality_select")
            
            # Upload foto diri
            st.write("**🖼️ Foto Kamu (Opsional):**")
            my_photo = st.file_uploader("Upload foto kamu", type=['jpg', 'png', 'jpeg'], key="my_photo_upload")
            if my_photo:
                st.image(my_photo, caption="Foto Kamu", width=150)
        
        with col2:
            st.write("**Profil Pasangan/Crush:**")
            partner_name = st.text_input("Nama pasangan/crush:", key="partner_name_input")
            partner_personality = st.selectbox("Tipe Personality pasangan:", 
                ["🟢 Green Flag", "🟡 Yellow Flag", "🔴 Red Flag", "⚫ Black Flag"], key="partner_personality_select")
            
            # Upload foto pasangan
            st.write("**🖼️ Foto Pasangan (Opsional):**")
            partner_photo = st.file_uploader("Upload foto pasangan", type=['jpg', 'png', 'jpeg'], key="partner_photo_upload")
            if partner_photo:
                st.image(partner_photo, caption="Foto Pasangan", width=150)
        
        # Additional questions for better analysis
        st.subheader("💞 Dynamics Hubungan")
        q1 = st.slider("Seberapa sering kalian berkomunikasi?", 1, 10, 5, key="comm_slider")
        q2 = st.slider("Seberapa baik kalian menyelesaikan konflik?", 1, 10, 5, key="conflict_slider")
        q3 = st.slider("Seberapa banyak kesamaan interest/hobi?", 1, 10, 5, key="interest_slider")
        
        if st.button("💞 Tes Kecocokan!", key="compatibility_test_button"):
            # Validasi lebih detail
            if not my_name or not my_name.strip():
                st.error("⚠️ Harap isi nama kamu terlebih dahulu!")
            elif not partner_name or not partner_name.strip():
                st.error("⚠️ Harap isi nama pasangan/crush terlebih dahulu!")
            else:
                my_type = my_personality.split(" ")[1].lower() + "_flag"
                partner_type = partner_personality.split(" ")[1].lower() + "_flag"
                
                base_compatibility = COMPATIBILITY_MATRIX[my_type][partner_type]
                
                # Adjust based on dynamics
                dynamics_bonus = (q1 + q2 + q3) / 30  # Max 1.0 bonus
                final_compatibility = base_compatibility + (dynamics_bonus * 20) + random.randint(-5, 5)
                final_compatibility = max(10, min(100, final_compatibility))
                
                st.session_state.comp_results = {
                    "my_name": my_name.strip(),
                    "partner_name": partner_name.strip(),
                    "my_type": my_type,
                    "partner_type": partner_type,
                    "compatibility": final_compatibility,
                    "my_photo": my_photo,
                    "partner_photo": partner_photo,
                    "dynamics": {"communication": q1, "conflict": q2, "interests": q3}
                }
                st.session_state.comp_test_done = True
                st.rerun()
    
    else:
        results = st.session_state.comp_results
        
        st.balloons()
        st.title("💘 Hasil Tes Kecocokan!")
        
        # Display photos if available
        col1, col2 = st.columns(2)
        with col1:
            st.subheader(f"{results['my_name']}")
            my_data = PERSONALITY_DATA[results['my_type']]
            st.write(f"**Tipe:** {my_data['name']}")
            if results['my_photo']:
                st.image(results['my_photo'], caption=f"{results['my_name']}", width=200)
        
        with col2:
            st.subheader(f"{results['partner_name']}")
            partner_data = PERSONALITY_DATA[results['partner_type']]
            st.write(f"**Tipe:** {partner_data['name']}")
            if results['partner_photo']:
                st.image(results['partner_photo'], caption=f"{results['partner_name']}", width=200)
        
        # Compatibility score
        st.subheader(f"💞 Tingkat Kecocokan: {results['compatibility']}%")
        st.progress(results['compatibility'] / 100)
        
        # Relationship type
        rel_type = "Partner 💞"
        rel_desc = "Pasangan yang solid dengan chemistry yang baik."
        
        for threshold, rel_info in RELATIONSHIP_TYPES.items():
            if results['compatibility'] >= threshold:
                rel_type = rel_info["type"]
                rel_desc = rel_info["desc"]
                break
        
        st.subheader(f"🤝 Tipe Hubungan Terbaik: {rel_type}")
        st.write(rel_desc)
        
        # Detailed Analysis
        st.subheader("📊 Analisis Mendalam:")
        
        analysis_col1, analysis_col2 = st.columns(2)
        
        with analysis_col1:
            st.write("**💪 Kekuatan Hubungan:**")
            
            # Dynamic strengths based on compatibility
            if results['compatibility'] >= 80:
                st.write("- Chemistry yang sangat kuat")
                st.write("- Saling memahami dengan baik")
                st.write("- Potensi hubungan jangka panjang")
            elif results['compatibility'] >= 60:
                st.write("- Chemistry yang baik")
                st.write("- Bisa saling melengkapi")
                st.write("- Potensi berkembang dengan komunikasi")
            else:
                st.write("- Perbedaan yang menarik")
                st.write("- Kesempatan untuk belajar")
                st.write("- Butuh usaha lebih untuk memahami")
            
            # Personality-specific strengths
            combo = f"{results['my_type']}_{results['partner_type']}"
            if combo in ["green_flag_green_flag", "yellow_flag_yellow_flag", "red_flag_red_flag", "black_flag_black_flag"]:
                st.write("- Kesamaan nilai dan pandangan")
                st.write("- Pemahaman yang natural")
            elif combo in ["green_flag_yellow_flag", "yellow_flag_green_flag"]:
                st.write("- Keseimbangan antara kebebasan dan perhatian")
                st.write("- Bisa saling melengkapi dengan baik")
        
        with analysis_col2:
            st.write("**💡 Area Perbaikan:**")
            
            if results['compatibility'] >= 80:
                st.write("- Pertahankan komunikasi yang sehat")
                st.write("- Jaga keseimbangan dalam hubungan")
            elif results['compatibility'] >= 60:
                st.write("- Tingkatkan komunikasi terbuka")
                st.write("- Pelajari bahasa cinta masing-masing")
                st.write("- Beri waktu untuk penyesuaian")
            else:
                st.write("- Butuh komunikasi ekstra")
                st.write("- Perlahan bangun kepercayaan")
                st.write("- Fokus pada kesamaan yang ada")
            
            # Dynamics analysis
            st.write("**📈 Dynamics Score:**")
            dynamics = results['dynamics']
            comm_score = "🟢 Baik" if dynamics['communication'] >= 7 else "🟡 Cukup" if dynamics['communication'] >= 5 else "🔴 Perlu ditingkatkan"
            conflict_score = "🟢 Baik" if dynamics['conflict'] >= 7 else "🟡 Cukup" if dynamics['conflict'] >= 5 else "🔴 Perlu ditingkatkan"
            interest_score = "🟢 Baik" if dynamics['interests'] >= 7 else "🟡 Cukup" if dynamics['interests'] >= 5 else "🔴 Perlu ditingkatkan"
            
            st.write(f"- Komunikasi: {comm_score}")
            st.write(f"- Penyelesaian Konflik: {conflict_score}")
            st.write(f"- Kesamaan Minat: {interest_score}")
        
        # Recommendations
        st.subheader("🎯 Rekomendasi & Tips:")
        
        rec_col1, rec_col2 = st.columns(2)
        with rec_col1:
            st.write("**❤️ Untuk Memperkuat Hubungan:**")
            if results['compatibility'] >= 80:
                st.write("- Coba aktivitas baru bersama")
                st.write("- Tetap jaga komunikasi yang jujur")
                st.write("- Rayakan pencapaian kecil bersama")
            else:
                st.write("- Fokus pada quality time berkualitas")
                st.write("- Pelajari love language masing-masing")
                st.write("- Bangun kepercayaan secara bertahap")
        
        with rec_col2:
            st.write("**🌟 Aktivitas Rekomendasi:**")
            if "green" in results['my_type'] or "green" in results['partner_type']:
                st.write("- Diskusi meaningful topics")
                st.write("- Aktivitas outdoor yang menenangkan")
            if "red" in results['my_type'] or "red" in results['partner_type']:
                st.write("- Adventure dates")
                st.write("- Aktivitas spontan dan exciting")
            if "yellow" in results['my_type'] or "yellow" in results['partner_type']:
                st.write("- Romantic dinners")
                st.write("- Aktivitas yang melibatkan quality time")
            if "black" in results['my_type'] or "black" in results['partner_type']:
                st.write("- Deep conversations")
                st.write("- Aktivitas yang stimulatif mental")
        
        # Save to history
        if 'test_history' not in st.session_state:
            st.session_state.test_history = []
        
        st.session_state.test_history.append({
            "test_type": "Tes Kecocokan",
            "result": f"{results['compatibility']}% dengan {results['partner_name']}",
            "relationship": rel_type,
            "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
        })
        
        # Navigation
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Tes dengan Pasangan Lain", key="comp_retest", use_container_width=True):
                st.session_state.comp_test_done = False
                st.rerun()
        with col2:
            if st.button("🤝 Personality Match", key="comp_to_match", use_container_width=True):
                st.session_state.current_page = "personality_match"
                st.session_state.comp_test_done = False
                st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

# ==================== PERSONALITY MATCH TEST ====================

def show_personality_match():
    st.markdown('<div class="main">', unsafe_allow_html=True)
    
    st.title("🤝 Tes Personality Match")
    st.write("Temukan tipe hubungan terbaik untuk kepribadianmu dengan berbagai tipe orang!")
    
    q1 = st.slider("Seberapa extrovert kamu?", 1, 10, 5)
    q2 = st.slider("Seberapa emotional kamu?", 1, 10, 5)
    q3 = st.slider("Seberapa independent kamu?", 1, 10, 5)
    q4 = st.slider("Seberapa adventurous kamu?", 1, 10, 5)
    
    if st.button("🔍 Analisis Personality Match"):
        if q1 > 6 and q2 < 5 and q3 > 5:
            p_type = "green_flag"
        elif q1 > 5 and q2 > 5 and q3 < 6:
            p_type = "yellow_flag"
        elif q2 > 7 and q4 > 6:
            p_type = "red_flag"
        else:
            p_type = "black_flag"
        
        user_data = PERSONALITY_DATA[p_type]
        
        st.balloons()
        st.subheader(f"🎯 Personality Kamu: {user_data['name']}")
        st.write(user_data['description'])
        
        matches = []
        for other_type in PERSONALITY_DATA.keys():
            if other_type != p_type:
                score = COMPATIBILITY_MATRIX[p_type][other_type]
                rel_info = None
                for threshold, info in RELATIONSHIP_TYPES.items():
                    if score >= threshold:
                        rel_info = info
                        break
                if rel_info:
                    matches.append({
                        "type": other_type,
                        "score": score,
                        "rel_type": rel_info["type"],
                        "desc": rel_info["desc"],
                        "data": PERSONALITY_DATA[other_type]
                    })
        
        matches.sort(key=lambda x: x["score"], reverse=True)
        
        st.subheader("💫 Rekomendasi Hubungan Terbaik:")
        
        for i, match in enumerate(matches):
            with st.expander(f"{i+1}. {match['data']['name']} - {match['score']}% ({match['rel_type']})", expanded=i<2):
                st.write(f"**Deskripsi:** {match['desc']}")
                st.progress(match['score'] / 100)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**💪 Kelebihan:**")
                    st.write("⭐ Chemistry alami")
                    st.write("⭐ Saling melengkapi")
                    st.write("⭐ Growth bersama")
                    
                with col2:
                    st.write("**💡 Tips:**")
                    st.write("🎯 Komunikasi terbuka")
                    st.write("🎯 Saling menghargai")
                    st.write("🎯 Quality time")
        
        if 'test_history' not in st.session_state:
            st.session_state.test_history = []
        
        st.session_state.test_history.append({
            "test_type": "Personality Match",
            "result": f"{user_data['name']} - {len(matches)} rekomendasi",
            "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
        })
    
    st.markdown('</div>', unsafe_allow_html=True)

# ==================== HISTORY & ANALYSIS ====================

def show_history_analysis():
    st.markdown('<div class="main">', unsafe_allow_html=True)
    
    st.title("📊 Riwayat & Analisis Tes")
    
    if 'test_history' not in st.session_state or not st.session_state.test_history:
        st.info("📝 Belum ada riwayat tes. Silakan ikuti tes terlebih dahulu!")
    else:
        st.subheader("📈 Riwayat Tes Kamu:")
        
        for i, test in enumerate(reversed(st.session_state.test_history)):
            with st.expander(f"{test['test_type']} - {test['timestamp']}", expanded=i==0):
                st.write(f"**Hasil:** {test['result']}")
                if 'confidence' in test:
                    st.write(f"**Confidence AI:** {test['confidence']}")
                if 'relationship' in test:
                    st.write(f"**Tipe Hubungan:** {test['relationship']}")
        
        st.subheader("📊 Analisis Trend:")
        
        test_types = [test['test_type'] for test in st.session_state.test_history]
        type_count = pd.Series(test_types).value_counts()
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Jumlah Tes per Tipe:**")
            for test_type, count in type_count.items():
                st.write(f"- {test_type}: {count} kali")
        
        with col2:
            st.write("**Rekomendasi:**")
            st.write("🎯 Lanjutkan eksplorasi personality-mu")
            st.write("💡 Coba tes yang berbeda untuk insight lengkap")
        
        if st.button("🗑️ Hapus Semua Riwayat", key="clear_all_history"):
            # Reset SEMUA state, bukan hanya history
            st.session_state.test_history = []
            st.session_state.ideal_test_page = 0
            st.session_state.ideal_answers = []
            st.session_state.comp_test_done = False
            st.session_state.ideal_user_data = {}
            st.session_state.comp_results = {}
            st.success("Semua data dan riwayat telah dihapus!")
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

# ==================== HELPER FUNCTION ====================

def get_all_personality_images(personality_type):
    images = []
    try:
        if personality_type in PERSONALITY_IMAGES and PERSONALITY_IMAGES[personality_type]:
            urls_to_use = PERSONALITY_IMAGES[personality_type]
        else:
            # Fallback URLs yang lebih reliable
            fallback_urls = {
                "green_flag": [
                    "https://i.pinimg.com/1200x/f5/ca/4f/f5ca4f319aaf7937b8dc36c69aa22a8a.jpg",
                    "https://i.pinimg.com/1200x/c4/49/95/c449955b4ec09d1c661340d89ec68dd7.jpg"
                ],
                "yellow_flag": [
                    "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=300", 
                    "https://images.unsplash.com/photo-1488426862026-3ee34a7d66df?w=300"
                ],
                "red_flag": [
                    "https://images.unsplash.com/photo-1544005313-94ddf0286df2?w=300",
                    "https://images.unsplash.com/photo-1488161628813-04466f872be2?w=300"
                ],
                "black_flag": [
                    "https://images.unsplash.com/photo-1531746020798-e6953c6e8e04?w=300",
                    "https://images.unsplash.com/photo-1517841905240-472988babdf9?w=300"
                ],
            }
            urls_to_use = fallback_urls.get(personality_type, [])
        
        for image_url in urls_to_use:
            try:
                response = requests.get(image_url, timeout=10)
                if response.status_code == 200:
                    img = Image.open(BytesIO(response.content))
                    img = img.resize((200, 200))  # Ukuran lebih konsisten
                    images.append(img)
            except Exception as e:
                continue
                
    except Exception as e:
        st.warning(f"Error loading images: {str(e)}")
    
    return images

# ==================== INITIALIZE & RUN ====================

if __name__ == "__main__":
    main()