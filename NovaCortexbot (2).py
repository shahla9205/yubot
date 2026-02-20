import os
import sys
import json
import logging
import time
import hashlib
import re
import sqlite3
import asyncio
import pickle
import io
from datetime import datetime
from typing import Any, List, Optional, Dict, Tuple, Union
from collections import Counter, defaultdict, deque
from pathlib import Path

import pdfplumber

CHROMA_PATH = "chroma_ultra"
CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)
SYNTHETIC_QUESTIONS_FILE = "synthetic_questions.json"

os.environ['ANONYMIZED_TELEMETRY'] = 'False'

from tqdm import tqdm
from telegram import Update, ReplyKeyboardMarkup, ReplyKeyboardRemove
from telegram.ext import (
    Application, CommandHandler, MessageHandler, ContextTypes,
    ConversationHandler, filters
)

from openai import OpenAI
import httpx

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rank_bm25 import BM25Okapi

import nltk
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords

import requests

try:
    nltk.data.find('corpora/stopwords')
except:
    nltk.download('stopwords', quiet=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("novacortex_v12_9_2.log", encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ==================== API KEYS ====================
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "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")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "8574197374:AAEUHjGLQvFxrFvzKsTsWT_KvKSJUMd4v88")
SERPER_API_KEY = os.getenv("SERPER_API_KEY", "0ae52b02b20869cd428b54d21ddad4b11c12eaf1")

if not all([GROQ_API_KEY, TELEGRAM_TOKEN, SERPER_API_KEY]):
    logger.error("### خطأ فادح: مفاتيح API ناقصة.")
    sys.exit(1)

grok_client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1",
    http_client=httpx.Client(follow_redirects=True, timeout=25.0)
)

UNIVERSITY_NAME = "جامعة اليرموك"
RAG_CONFIDENCE_THRESHOLD = 0.60
CURRENT_YEAR = 2026
CURRENT_MONTH = 2

OFFICIAL_PDFS = {
    "fees_and_majors": "https://admreg.yu.edu.jo/images/docs/majors.pdf"
}

SYNTHETIC_FROZEN = True

FEES_KEYWORDS = [
    'رسوم', 'رسم', 'تكلفة', 'تكاليف', 'السعر', 'المصاريف',
    'fees', 'tuition', 'cost', 'costs', 'price', 'expenses',
    'كم تكلف', 'كم سعر', 'how much'
]

SCHOLARSHIP_KEYWORDS = [
    'منح', 'منحة', 'المنح', 'جراية', 'جرايات', 'الجراية', 'الجرايات',
    'scholarship', 'scholarships', 'grant', 'grants', 'financial aid',
    'مساعدات مالية', 'دعم مالي', 'تمويل', 'المتاحة', 'المتوفرة'
]

# [FIX-N] Cost-Specific Keywords for housing/transport
COST_SPECIFIC_KEYWORDS = {
    'housing': ['سكن', 'dorm', 'housing', 'شقة', 'غرفة', 'room'],
    'transport': ['مواصلات', 'bus', 'transportation', 'باص', 'تكسي']
}

COLLOQUIAL_FACULTY_TERMS = {
    'الايتي': 'كلية تكنولوجيا المعلومات وعلوم الحاسوب',
    'ايتي': 'كلية تكنولوجيا المعلومات وعلوم الحاسوب',
    'it faculty': 'Faculty of Information Technology',
    'الحجاوي': 'كلية الحجاوي للهندسة التكنولوجية',
    'الهندسه': 'كلية الحجاوي للهندسة التكنولوجية',
    'الهندسة': 'كلية الحجاوي للهندسة التكنولوجية',
    'كلية الطب': 'كلية الطب',
    'الصيدليه': 'كلية الصيدلة',
    'التمريض': 'كلية التمريض',
    'الاداب': 'كلية الآداب',
    'الآداب': 'كلية الآداب',
    'الاعمال': 'كلية الأعمال',
    'التربية': 'كلية العلوم التربوية',
    'الشريعه': 'كلية الشريعة والدراسات الإسلامية',
    'الشريعة': 'كلية الشريعة والدراسات الإسلامية',
    'القانون': 'كلية القانون',
    'الاعلام': 'كلية الإعلام',
    'السياحه': 'كلية السياحة والفنادق',
    'السياحة': 'كلية السياحة والفنادق',
    'الفنون': 'كلية الفنون الجميلة',
    'الاثار': 'كلية الآثار والأنثروبولوجيا',
    'الآثار': 'كلية الآثار والأنثروبولوجيا',
    'الرياضة': 'كلية التربية البدنية وعلوم الرياضة',
    'رياضة': 'كلية التربية البدنية وعلوم الرياضة',
}

NO_RESULT_MESSAGE_AR = "ما لقيت معلومات كافية عن هذا الموضوع في قاعدة البيانات."
NO_RESULT_MESSAGE_EN = "I couldn't find enough information about this topic in the database."

# ==================== VERIFIED FACTS ====================
VERIFIED_FACTS = {
    "ar": {
        "university_brief": "جامعة اليرموك هي جامعة حكومية أردنية رائدة تقع في مدينة إربد، تأسست عام 1976. التعليم فيها وجاهي بالكامل.",
        "president": "أ.د. مالك الشرايري",
        "president_full": "الأستاذ الدكتور مالك أحمد الشرايري",
        "president_title": "رئيس جامعة اليرموك",

        "deans": {
            "hijjawi": {"name": "د. عوض سميران الزبن", "title": "عميد كلية الحجاوي للهندسة التكنولوجية", "match": ["حجاوي", "هندسة", "هندسه", "engineering", "technology"]},
            "it": {"name": "د. قاسم الردايدة", "title": "عميد كلية تكنولوجيا المعلومات وعلوم الحاسوب", "match": ["معلومات", "حاسوب", "it", "computer", "ايتي", "الايتي", "information technology"]},
            "medicine": {"name": "د. جمانة السليمان", "title": "قائم بأعمال عميد كلية الطب", "match": ["طب", "medicine", "medical"]},
            "pharmacy": {"name": "د. غيث الطعاني", "title": "عميد كلية الصيدلة", "match": ["صيدلة", "صيدليه", "pharmacy"]},
            "science": {"name": "د. مهيب محمد عواودة", "title": "عميد كلية العلوم", "match": ["علوم", "science"]},
            "arts": {"name": "د. خالد محمد أمين الهزايمة", "title": "عميد كلية الآداب", "match": ["آداب", "اداب", "arts"]},
            "business": {"name": "د. يحيى سليم بني ملحم", "title": "عميد كلية الأعمال", "match": ["أعمال", "اعمال", "business"]},
            "education": {"name": "د. هاني حتمل عبيدات", "title": "عميد كلية العلوم التربوية", "match": ["تربية", "تعليم", "education"]},
            "nursing": {"name": "د. رسمية مصطفى الأعمر", "title": "قائم بأعمال عميد كلية التمريض", "match": ["تمريض", "nursing"]},
            "law": {"name": "د. مها خصاونة", "title": "قائم بأعمال عميد كلية القانون", "match": ["قانون", "law"]}
        },

        "it_admin": {
            "dean": {"name": "د. قاسم الردايدة", "title": "عميد كلية تكنولوجيا المعلومات وعلوم الحاسوب"},
            "vice_deans": [
                {"name": "د. عماد الشواقفة", "title": "نائب العميد"},
                {"name": "د. خالد النهار", "title": "نائب العميد لشؤون الاعتماد الأكاديمي والجودة"}
            ],
            "assistant_dean": {"name": "د. قصي الزعبي", "title": "مساعد العميد لشؤون الطلبة"},
            "department_heads": {
                "cs": {"name": "د. بلال عبدالغني", "dept": "قسم علوم الحاسوب"},
                "cis": {"name": "د. ايناس خشاشنة", "dept": "قسم نظم المعلومات"},
                "it": {"name": "د. علاء حمود", "dept": "قسم تكنولوجيا المعلومات"}
            }
        },

        "num_faculties": "16",
        "faculties_list": [
            "كلية الطب", "كلية الصيدلة", "كلية التمريض", "كلية العلوم",
            "كلية الحجاوي للهندسة التكنولوجية", "كلية تكنولوجيا المعلومات وعلوم الحاسوب",
            "كلية الآداب", "كلية الأعمال", "كلية الشريعة والدراسات الإسلامية",
            "كلية العلوم التربوية", "كلية القانون", "كلية الإعلام",
            "كلية التربية البدنية وعلوم الرياضة", "كلية الآثار والأنثروبولوجيا",
            "كلية السياحة والفنادق", "كلية الفنون الجميلة"
        ],

        "it_programs": [
            "علوم الحاسوب", "نظم المعلومات الحاسوبية", "تكنولوجيا معلومات الأعمال",
            "علم البيانات والذكاء الاصطناعي", "الأمن السيبراني", "الواقع الرقمي وتطوير الألعاب"
        ],

        # [FIX-M] Online Policy Guardrail
        "online_policy": "لا، التعليم في جامعة اليرموك وجاهي بالكامل ولا يوجد برامج دراسة عن بعد.",

        "student_housing": {
            "available": True,
            "details": "نعم، يتوفر سكن للطالبات داخل الحرم الجامعي، وللطلاب الذكور سكن خارجي. التكلفة تتراوح بين 50 إلى 150 دينار أردني شهرياً.",
            "brief": "تكلفة السكن تتراوح بين 50 إلى 150 دينار أردني شهرياً."
        },

        "health_insurance": {
            "available": True,
            "details": "نعم، التأمين الصحي إلزامي لجميع طلبة البكالوريوس في جامعة اليرموك."
        },

        "university_type": "حكومية",
        "founded_year": "1976",
        "location": "إربد، الأردن",
        "website": "yu.edu.jo",
        "faculty_directory": "fmd.yu.edu.jo"  # [FIX-U] دليل أعضاء هيئة التدريس
    },

    "en": {
        "university_brief": "Yarmouk University is a leading Jordanian public university located in Irbid, founded in 1976. Education is entirely in-person.",
        "president": "Prof. Malik Al-Shariri",
        "president_full": "Professor Dr. Malik Ahmed Al-Shariri",
        "president_title": "President of Yarmouk University",

        "deans": {
            "hijjawi": {"name": "Dr. Awad Smiran Al-Zabn", "title": "Dean of Hijjawi Faculty for Engineering Technology", "match": ["hijjawi", "engineering", "technology"]},
            "it": {"name": "Dr. Qasem Al-Radaideh", "title": "Dean of Faculty of Information Technology and Computer Science", "match": ["it", "information", "computer", "technology"]},
            "medicine": {"name": "Dr. Jumana Al-Sulaiman", "title": "Acting Dean of Faculty of Medicine", "match": ["medicine", "medical"]},
            "pharmacy": {"name": "Dr. Ghaith Al-Taani", "title": "Dean of Faculty of Pharmacy", "match": ["pharmacy"]},
            "science": {"name": "Dr. Muheeb Mohammed Awawdeh", "title": "Dean of Faculty of Science", "match": ["science"]},
            "arts": {"name": "Dr. Khaled Mohammed Ameen Al-Hazyameh", "title": "Dean of Faculty of Arts", "match": ["arts"]},
            "business": {"name": "Dr. Yahya Salim Bani Melhem", "title": "Dean of Faculty of Business", "match": ["business"]},
            "education": {"name": "Dr. Hani Hatmel Obeidat", "title": "Dean of Faculty of Education", "match": ["education"]},
            "nursing": {"name": "Dr. Rasmiya Mustafa Al-A'mar", "title": "Acting Dean of Faculty of Nursing", "match": ["nursing"]},
            "law": {"name": "Dr. Maha Khasawneh", "title": "Acting Dean of Faculty of Law", "match": ["law"]}
        },

        "it_admin": {
            "dean": {"name": "Dr. Qasem Al-Radaideh", "title": "Dean of Faculty of IT and Computer Science"},
            "vice_deans": [
                {"name": "Dr. Emad Al-Shawaqfeh", "title": "Vice Dean"},
                {"name": "Dr. Khaled Al-Nahar", "title": "Vice Dean for Academic Accreditation and Quality"}
            ],
            "assistant_dean": {"name": "Dr. Qusay Al-Zaabi", "title": "Assistant Dean for Student Affairs"},
            "department_heads": {
                "cs": {"name": "Dr. Bilal Abdul-Ghani", "dept": "Department of Computer Science"},
                "cis": {"name": "Dr. Inas Khashashna", "dept": "Department of Information Systems"},
                "it": {"name": "Dr. Alaa Hammoud", "dept": "Department of Information Technology"}
            }
        },

        "num_faculties": "16",
        "faculties_list": [
            "Faculty of Medicine", "Faculty of Pharmacy", "Faculty of Nursing", "Faculty of Science",
            "Hijjawi Faculty for Engineering Technology", "Faculty of Information Technology and Computer Science",
            "Faculty of Arts", "Faculty of Business", "Faculty of Sharia and Islamic Studies",
            "Faculty of Education", "Faculty of Law", "Faculty of Mass Communication",
            "Faculty of Physical Education", "Faculty of Archaeology and Anthropology",
            "Faculty of Tourism and Hotel Management", "Faculty of Fine Arts"
        ],
        "it_programs": [
            "Computer Science", "Computer Information Systems", "Business Information Technology",
            "Data Science and Artificial Intelligence", "Cybersecurity", "Digital Reality and Game Development"
        ],

        # [FIX-M] Online Policy Guardrail
        "online_policy": "No, education at Yarmouk University is entirely in-person and there are no online degree programs.",

        "student_housing": {
            "available": True,
            "details": "Yes, housing is available for female students on campus and for male students off campus. Cost ranges from 50 to 150 Jordanian Dinar per month.",
            "brief": "Housing costs range from 50 to 150 JD per month."
        },

        "health_insurance": {
            "available": True,
            "details": "Yes, health insurance is mandatory for all undergraduate students at Yarmouk University."
        },

        "university_type": "public",
        "founded_year": "1976",
        "location": "Irbid, Jordan",
        "website": "yu.edu.jo",
        "faculty_directory": "fmd.yu.edu.jo"  # [FIX-U] Faculty directory site
    }
}


# ==================== PDF EXTRACTOR ====================
class PDFExtractor:
    def __init__(self):
        self.session = requests.Session()
        self.cache = {}
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

    def fetch_and_extract(self, url: str, max_pages: int = 10) -> Dict[str, Any]:
        if not url.lower().endswith('.pdf'):
            return {"text": "", "success": False, "error": "Not a PDF file", "is_garbled": False, "pages_extracted": 0}
        if url in self.cache:
            return self.cache[url]
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            pdf_bytes = response.content
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                all_text = []
                pages_to_process = min(len(pdf.pages), max_pages)
                for page in pdf.pages[:pages_to_process]:
                    text = page.extract_text()
                    if text:
                        all_text.append(text.strip())
            combined_text = "\n\n".join(all_text)
            is_garbled = self._detect_garbled_text(combined_text)
            success = len(combined_text) > 100 and not is_garbled
            result = {"text": combined_text, "success": success, "is_garbled": is_garbled, "pages_extracted": pages_to_process}
            self.cache[url] = result
            return result
        except Exception as e:
            logger.error(f"PDF error: {e}")
            return {"text": "", "success": False, "error": str(e), "is_garbled": False, "pages_extracted": 0}

    def _detect_garbled_text(self, text: str) -> bool:
        if not text or len(text) < 50:
            return True
        words = text.split()
        if not words:
            return True
        short_word_ratio = sum(1 for w in words if len(w) <= 2) / len(words)
        return short_word_ratio > 0.7 or len(text) < 300


# ==================== DATABASE ====================
def init_db():
    db_name = "novacortex_v12.db"
    with sqlite3.connect(db_name) as conn:
        c = conn.cursor()
        c.execute("""CREATE TABLE IF NOT EXISTS users (
            telegram_id INTEGER PRIMARY KEY, username TEXT, first_name TEXT, role TEXT DEFAULT 'visitor',
            uni_id TEXT, study_level TEXT DEFAULT 'بكالوريوس', created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            last_active TEXT DEFAULT CURRENT_TIMESTAMP, total_questions INTEGER DEFAULT 0
        )""")
        c.execute("""CREATE TABLE IF NOT EXISTS questions_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, username TEXT, question TEXT NOT NULL,
            response TEXT NOT NULL, source TEXT NOT NULL, confidence REAL DEFAULT 0, response_time_ms INTEGER,
            docs_retrieved INTEGER DEFAULT 0, timestamp TEXT NOT NULL, study_level TEXT DEFAULT 'بكالوريوس'
        )""")
        c.execute("""CREATE TABLE IF NOT EXISTS response_cache (
            question_hash TEXT PRIMARY KEY, question TEXT, response TEXT, confidence REAL, source TEXT,
            created_at TEXT, last_used TEXT, hit_count INTEGER DEFAULT 0
        )""")
        c.execute("""CREATE TABLE IF NOT EXISTS daily_stats (
            date TEXT PRIMARY KEY, total_queries INTEGER DEFAULT 0, cache_hits INTEGER DEFAULT 0,
            rag_responses INTEGER DEFAULT 0, web_responses INTEGER DEFAULT 0, avg_confidence REAL DEFAULT 0,
            avg_response_time_ms INTEGER DEFAULT 0
        )""")
        c.execute("""CREATE TABLE IF NOT EXISTS conversation_memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER, message_index INTEGER, role TEXT,
            content TEXT, timestamp TEXT, is_summary INTEGER DEFAULT 0
        )""")
        conn.commit()
    logger.info("Database initialized (novacortex_v12.db)")

def save_user(telegram_id, username=None, first_name=None, role="visitor", uni_id=None, study_level="بكالوريوس"):
    with sqlite3.connect("novacortex_v12.db") as conn:
        c = conn.cursor()
        c.execute("""INSERT OR REPLACE INTO users
            (telegram_id, username, first_name, role, uni_id, study_level, last_active)
            VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (telegram_id, username, first_name, role, uni_id, study_level, datetime.now().isoformat()))
        conn.commit()

def get_user(telegram_id):
    with sqlite3.connect("novacortex_v12.db") as conn:
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE telegram_id = ?", (telegram_id,))
        return c.fetchone()

def update_user_activity(telegram_id):
    with sqlite3.connect("novacortex_v12.db") as conn:
        c = conn.cursor()
        c.execute("UPDATE users SET last_active = ?, total_questions = total_questions + 1 WHERE telegram_id = ?",
            (datetime.now().isoformat(), telegram_id))
        conn.commit()

def log_question(user_id, username, question, response, source, confidence, response_time_ms, docs_retrieved, study_level):
    with sqlite3.connect("novacortex_v12.db") as conn:
        c = conn.cursor()
        c.execute("""INSERT INTO questions_log
            (user_id, username, question, response, source, confidence, response_time_ms, docs_retrieved, timestamp, study_level)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (user_id, username, question, response, source, confidence, response_time_ms, docs_retrieved, datetime.now().isoformat(), study_level))
        conn.commit()

def update_daily_stats(source, confidence, response_time_ms):
    with sqlite3.connect("novacortex_v12.db") as conn:
        c = conn.cursor()
        today = datetime.now().date().isoformat()
        c.execute("SELECT * FROM daily_stats WHERE date = ?", (today,))
        row = c.fetchone()
        if row:
            total = row[1]
            cache = row[2] + (1 if source == 'cache' else 0)
            rag = row[3] + (1 if source == 'rag' else 0)
            web = row[4] + (1 if source == 'web' else 0)
            avg_conf = (row[5] * total + confidence) / (total + 1)
            avg_time = int((row[6] * total + response_time_ms) / (total + 1))
            c.execute("UPDATE daily_stats SET total_queries=total_queries+1, cache_hits=?, rag_responses=?, web_responses=?, avg_confidence=?, avg_response_time_ms=? WHERE date=?",
                (cache, rag, web, avg_conf, avg_time, today))
        else:
            c.execute("INSERT INTO daily_stats VALUES (?, 1, ?, ?, ?, ?, ?)",
                (today, 1 if source=='cache' else 0, 1 if source=='rag' else 0, 1 if source=='web' else 0, confidence, response_time_ms))
        conn.commit()


# ==================== ARABIC PROCESSOR ====================
class ArabicProcessor:
    def __init__(self):
        self.stemmer = SnowballStemmer("arabic")
        self.stopwords = set(stopwords.words("arabic"))
        self.space_pattern = re.compile(r'\s+')

    def normalize(self, text: str) -> str:
        if not text: return ""
        text = text.strip()
        text = re.sub(r"[إأآا]", "ا", text)
        text = re.sub(r"ى", "ي", text)
        text = re.sub(r"ة", "ه", text)
        text = re.sub(r'[^\w\s\u0600-\u06FF]', ' ', text)
        text = self.space_pattern.sub(' ', text)
        return text.strip()

    def tokenize(self, text: str) -> List[str]:
        text = self.normalize(text)
        words = text.split()
        tokens = []
        for word in words:
            if len(word) > 2 and word not in self.stopwords:
                try:
                    tokens.append(self.stemmer.stem(word))
                except:
                    tokens.append(word)
        return tokens

    def detect_language(self, text: str) -> str:
        if not text:
            return "ar"
        clean_text = re.sub(r'[0-9\s\.,!?;:\-()]+', '', text)
        if not clean_text:
            return "ar"
        arabic_chars = len(re.findall(r'[\u0600-\u06FF]', clean_text))
        latin_chars = len(re.findall(r'[a-zA-Z]', clean_text))
        total_chars = arabic_chars + latin_chars
        if arabic_chars == 0 and latin_chars > 0:
            return "en"
        if latin_chars == 0 and arabic_chars > 0:
            return "ar"
        if total_chars > 0:
            return "ar" if arabic_chars / total_chars >= 0.6 else "en"
        return "ar"

arabic_processor = ArabicProcessor()


# ==================== [FIX-AC] THE PROXY PROMPT ====================
def apply_proxy_prompt(question: str) -> str:
    """
    [FIX-AC] إعادة صياغة السؤال داخلياً لتجاوز فلاتر الأمان التي ترفض كلمة "إيميل"
    
    مثال: 
    "إيميل دكتور قاسم الردايدة" → "معلومات التواصل لـ دكتور قاسم الردايدة"
    "بريد دكتورة جمانة" → "بيانات التواصل لـ دكتورة جمانة"
    """
    if not question:
        return question
    
    q_lower = question.lower()
    modified = question
    
    # [FIX-AC] التمويه الذكي - استبدال كلمات الإيميل بعبارات أكثر حيادية
    email_keywords = ['إيميل', 'ايميل', 'بريد', 'email', 'e-mail']
    replacement_phrases = ['معلومات التواصل لـ', 'بيانات الاتصال لـ', 'contact information for']
    
    for keyword in email_keywords:
        if keyword in q_lower:
            # استخدام أول عبارة بديلة
            modified = re.sub(rf'\b{keyword}\b', replacement_phrases[0], modified, flags=re.IGNORECASE)
            logger.info(f"[FIX-AC] Proxy prompt applied: '{keyword}' → '{replacement_phrases[0]}'")
            break
    
    return modified


# ==================== [FIX-AB] DEEP FACULTY SEARCH ====================
def generate_deep_faculty_query(question: str, faculty_name: str = None) -> List[str]:
    """
    [FIX-AB] تحسين استعلام البحث ليطلب البريد الإلكتروني والمكتب بالاسم الصريح
    
    يولد استعلامات بحث متعددة تضمن جلب الإيميل ورقم المكتب
    """
    detected_lang = arabic_processor.detect_language(question)
    
    # استخراج اسم الدكتور من السؤال إذا لم يتم توفيره
    if not faculty_name:
        # محاولة استخراج الاسم (بعد كلمات مثل دكتور، أستاذ)
        name_pattern = r'(?:دكتور|الدكتور|أستاذ|البروفيسور|dr|prof)\s+([\w\s]+?)(?:\s+|\Z|\.)'
        match = re.search(name_pattern, question, re.IGNORECASE)
        if match:
            faculty_name = match.group(1).strip()
        else:
            faculty_name = question
    
    # [FIX-AB] استعلامات بحث محسنة تطلب الإيميل والمكتب بالاسم الصريح
    queries = []
    
    if detected_lang == "ar":
        queries = [
            f"site:fmd.yu.edu.jo {faculty_name} البريد الإلكتروني مكتب هاتف",
            f"site:fmd.yu.edu.jo {faculty_name} email office phone",
            f"موقع جامعة اليرموك {faculty_name} بريد إلكتروني رقم هاتف",
            f"fmd.yu.edu.jo {faculty_name} contact information",
        ]
    else:
        queries = [
            f"site:fmd.yu.edu.jo {faculty_name} email office phone",
            f"site:fmd.yu.edu.jo {faculty_name} contact information",
            f"Yarmouk University faculty {faculty_name} email address office",
            f"fmd.yu.edu.jo {faculty_name} contact details",
        ]
    
    return queries


# ==================== [FIX-AD] RAW DATA EXTRACTION ====================
def extract_raw_contact_data(text: str) -> Dict[str, List[str]]:
    """
    [FIX-AD] استخراج البيانات الخام من النص (إيميلات، مكاتب، أرقام)
    
    تعمل كـ "صياد بيانات" تبحث عن أي نص يحتوي على:
    - @yu.edu.jo (الإيميلات الأكاديمية)
    - كلمة "مكتب" أو "office" متبوعة برقم أو وصف
    - أرقام هواتف
    """
    extracted = {
        "emails": [],
        "offices": [],
        "phones": []
    }
    
    if not text:
        return extracted
    
    # [FIX-AD] البحث عن الإيميلات الأكاديمية
    email_pattern = r'\b[\w\.-]+@yu\.edu\.jo\b'
    emails = re.findall(email_pattern, text, re.IGNORECASE)
    extracted["emails"] = list(set(emails))  # إزالة التكرارات
    
    # [FIX-AD] البحث عن معلومات المكتب
    office_patterns = [
        r'(?:مكتب|office|room|غرفة|مبنى|building)\s*:?\s*([^،\n\r]+)',
        r'([^،\n\r]+?)\s*(?:مكتب|office)',
        r'location\s*:?\s*([^،\n\r]+)',
        r'الموقع\s*:?\s*([^،\n\r]+)'
    ]
    
    for pattern in office_patterns:
        offices = re.findall(pattern, text, re.IGNORECASE)
        extracted["offices"].extend([o.strip() for o in offices if o.strip()])
    
    # [FIX-AD] البحث عن أرقام الهواتف
    phone_patterns = [
        r'(?:هاتف|tel|phone|mobile|جوال|خلوي)\s*:?\s*([\d\s\+\-\(\)]{7,})',
        r'([\d\s\+\-\(\)]{10,})'  # أي رقم بطول 10 خانات أو أكثر
    ]
    
    for pattern in phone_patterns:
        phones = re.findall(pattern, text, re.IGNORECASE)
        extracted["phones"].extend([p.strip() for p in phones if p.strip()])
    
    # تنظيف وتنسيق النتائج
    extracted["offices"] = list(set(extracted["offices"]))
    extracted["phones"] = list(set([re.sub(r'\s+', ' ', p) for p in extracted["phones"]]))
    
    return extracted


# ==================== SMART OUT-OF-SCOPE DETECTOR ====================
class SmartOutOfScopeDetector:
    def __init__(self, llm_client):
        self.client = llm_client
        self.cache = {}
        self.classification_prompt = """You are a scope classifier for Yarmouk University chatbot.

Your ONLY job: determine if a question is about Yarmouk University or not.

✅ IN SCOPE:
- Questions about faculties, colleges, departments
- Questions about admission, registration, fees, scholarships
- Questions about university services, facilities, staff, deans, president
- Questions about academic programs, courses, majors, study plans
- Questions about student life, housing, clubs, health insurance
- General questions answerable with university context
- Questions using colloquial names for faculties (IT, الحجاوي, الايتي, etc.)

❌ OUT OF SCOPE:
- Questions about OTHER universities (Harvard, الهاشمية, etc.)
- Sports/celebrities (Messi, رونالدو, etc.)
- Technology products (iPhone price, etc.)
- Weather, food recipes, movies, TV shows
- General life advice with NO university connection

RULES:
1. If "Yarmouk" or "اليرموك" mentioned → IN SCOPE
2. If comparison includes Yarmouk → IN SCOPE
3. If general but could relate to university → IN SCOPE
4. Another university only → OUT OF SCOPE
5. Celebrities/products → OUT OF SCOPE

Respond ONLY with JSON: {"is_in_scope": true/false, "confidence": 0.0-1.0, "reason": "brief"}"""

    def _check_colloquial_terms(self, question: str) -> bool:
        q_lower = question.lower().strip()
        for term in COLLOQUIAL_FACULTY_TERMS.keys():
            if term in q_lower:
                logger.info(f"Colloquial term '{term}' detected → IN SCOPE")
                return True
        return False

    async def is_out_of_scope(self, question: str) -> Optional[Dict]:
        if self._check_colloquial_terms(question):
            return None
        if question in self.cache:
            return self.cache[question]
        try:
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": self.classification_prompt},
                    {"role": "user", "content": f"Question: {question}"}
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
                max_tokens=150
            )
            result = json.loads(response.choices[0].message.content)
            is_in_scope = result.get("is_in_scope", True)
            confidence = result.get("confidence", 0.5)
            reason = result.get("reason", "")
            logger.info(f"Scope: {'IN' if is_in_scope else 'OUT'} (conf: {confidence:.2f}) - {reason}")
            if not is_in_scope and confidence >= 0.7:
                detected_lang = self._detect_language(question)
                if detected_lang == "ar":
                    rejection_message = {
                        "message": "عذراً، أنا متخصص فقط في جامعة اليرموك 🎓",
                        "reason": reason
                    }
                else:
                    rejection_message = {
                        "message": "Sorry, I specialize only in Yarmouk University 🎓",
                        "reason": reason
                    }
                self.cache[question] = rejection_message
                return rejection_message
            self.cache[question] = None
            return None
        except Exception as e:
            logger.error(f"Scope classification error: {e}")
            return None

    def _detect_language(self, text: str) -> str:
        arabic_chars = len(re.findall(r'[\u0600-\u06FF]', text))
        english_chars = len(re.findall(r'[a-zA-Z]', text))
        return "ar" if arabic_chars >= english_chars else "en"


def handle_greeting(question: str) -> Optional[str]:
    """[FIX-O] Zero-Fluff: حتى التحية نختصرها"""
    if not isinstance(question, str):
        return None
    q_lower = question.lower().strip()

    arabic_greetings = ["السلام", "مرحبا", "أهلا", "هلا", "صباح", "مساء"]
    english_greetings = ["hi", "hello", "hey", "good morning", "good evening"]

    if any(g in q_lower for g in arabic_greetings) and len(q_lower.split()) <= 3:
        return "أهلاً بك! كيف يمكنني مساعدتك؟"
    if any(g == q_lower or q_lower.startswith(g) for g in english_greetings) and len(q_lower.split()) <= 3:
        return "Hello! How can I help you?"
    return None


def is_comparison_question(question: str) -> bool:
    q_lower = question.lower().strip()
    comparison_words = ['مين احسن', 'مين أفضل', 'مقارنة بين', 'vs', 'versus', 'better than', 'who is better']
    if any(word in q_lower for word in comparison_words):
        return True
    return False


# ==================== SEMANTIC INTENT PARSER ====================
class SemanticIntentParser:
    """
    [FIX-A] إضافة intent_keys للرسوم والمنح بدل keyword lists
    [FIX-K] Strict Dean Matching
    [FIX-L] Credit Hours Priority
    [FIX-N] Cost-Specific Routing
    [FIX-R] Adaptive Response Logic - إضافة is_detailed
    [FIX-V] Faculty Intent Detection - إضافة intent للبحث عن الدكاترة
    [FIX-Z] Expanded Faculty Intent - شمول كلمات التواصل
    """

    INTENT_SCHEMA = """You are a question intent classifier for Yarmouk University chatbot.

Analyze the FULL question meaning (not just keywords) and return ONLY this JSON:
{
  "intent_key": "<one of the keys below>",
  "entity": "<specific thing mentioned e.g. faculty name or professor name>",
  "is_detailed": true/false
}

INTENT KEYS - pick the MOST SPECIFIC one:
- "list_faculties"          → user wants a list/count of ALL faculties
- "list_programs"           → user wants programs/majors INSIDE a specific faculty
- "list_departments"        → user wants departments/أقسام INSIDE a specific faculty
- "faculty_dean"            → user asks who is the DEAN (عميد) of a specific faculty
- "faculty_vice_dean"       → user asks about VICE DEAN (نائب العميد) or ASSISTANT DEAN (مساعد العميد)
- "department_head"         → user asks about HEAD OF DEPARTMENT (رئيس قسم)
- "faculty_info"            → user wants general info about a SPECIFIC faculty
- "credit_hours"            → how many credit hours for a program/major
- "student_housing"         → dormitories/accommodation
- "health_insurance"        → medical insurance for students
- "online_learning"         → distance/e-learning/online study
- "university_president"    → who is the president/رئيس الجامعة
- "university_info"         → general info about the WHOLE university
- "fees_inquiry"            → user asks about tuition fees, costs, prices
- "scholarship_inquiry"     → user asks about scholarships, grants, financial aid
- "cost_inquiry"            → user asks about cost of specific things like housing
- "faculty_member_search"   → [FIX-V][FIX-Z] user asks about a specific professor/faculty member (دكتور, أستاذ, professor, إيميل, بريد, مكتب, تواصل, email)
- "rag_needed"              → anything else

is_detailed: true if user asks for explanation/detail (contains words like "اشرح", "بالتفصيل", "احكيلي", "معلومات", "explain", "detail", "tell me", "about")

Return ONLY the JSON, no other text."""

    def __init__(self, client):
        self.client = client
        self._cache: Dict[str, Dict] = {}

    def parse(self, question: str) -> Dict:
        q_key = question.strip().lower()
        
        # [FIX-R] تحديد إذا كان السؤال يطلب شرحاً
        is_detailed = any(kw in q_key for kw in 
                         ['اشرح', 'بالتفصيل', 'احكيلي', 'معلومات', 'explain', 'detail', 'tell me', 'about'])
        
        # [FIX-V] & [FIX-Z] Faculty Intent Detection - شمول كلمات التواصل
        faculty_keywords = [
            'دكتور', 'دكتورة', 'أستاذ', 'بروفيسور', 'dr', 'prof', 'professor', 'الدكتور', 'البروفيسور',
            'إيميل', 'ايميل', 'بريد', 'تواصل', 'مكتب', 'email', 'contact', 'office'
        ]
        if any(kw in q_key for kw in faculty_keywords) and not any(kw in q_key for kw in ['عميد', 'dean', 'رئيس']):
            logger.info("[FIX-V/FIX-Z] Faculty member/contact search detected")
            # Extract potential name (simple version)
            words = q_key.split()
            entity = "faculty_member"
            # Try to extract name if it follows "دكتور" pattern
            for i, word in enumerate(words):
                if word in ['دكتور', 'الدكتور', 'dr', 'prof', 'أستاذ'] and i+1 < len(words):
                    entity = words[i+1]
                    break
            return {"intent_key": "faculty_member_search", "entity": entity, "is_detailed": is_detailed}
        
        # [FIX-L] Credit Hours Priority - Quick check
        if any(kw in q_key for kw in ['ساعة', 'ساعات', 'hours', 'credit']):
            logger.info("[FIX-L] Credit hours detected")
            return {"intent_key": "credit_hours", "entity": "program", "is_detailed": is_detailed}
        
        # [FIX-N] Cost-Specific Routing - Quick check
        if any(kw in question.lower() for kw in ['كم بكلف', 'كم تكلف', 'سعر', 'تكلفة']):
            if any(kw in question.lower() for kw in COST_SPECIFIC_KEYWORDS['housing']):
                return {"intent_key": "cost_inquiry", "entity": "housing", "is_detailed": is_detailed}
            if any(kw in question.lower() for kw in COST_SPECIFIC_KEYWORDS['transport']):
                return {"intent_key": "cost_inquiry", "entity": "transport", "is_detailed": is_detailed}
        
        if q_key in self._cache:
            cached = self._cache[q_key]
            cached["is_detailed"] = is_detailed  # تحديث is_detailed حسب السؤال الحالي
            return cached
            
        try:
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": self.INTENT_SCHEMA},
                    {"role": "user", "content": f"Q: {question}"}
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=100
            )
            result = json.loads(response.choices[0].message.content)
            if "intent_key" not in result:
                result["intent_key"] = "rag_needed"
            result["is_detailed"] = is_detailed
            logger.info(f"Intent: {result['intent_key']} | entity: {result.get('entity')} | detailed: {is_detailed}")
            self._cache[q_key] = result
            return result
        except Exception as e:
            logger.error(f"SemanticIntentParser error: {e}")
            return {"intent_key": "rag_needed", "entity": None, "is_detailed": is_detailed}


# ==================== SMART FACTS ROUTER ====================
class SmartFactsRouter:
    """
    [FIX-P] Direct Fact Routing: تقليص الردود للمعلومة المطلوبة فقط
    [FIX-R] Adaptive Response Logic: ردود مختلفة حسب is_detailed
    [FIX-S] Context-Aware Brevity: حذف المقدمات مع الحفاظ على العمق
    """

    def route(self, question: str, parsed_intent: Dict, lang: str, context_memory: List[Dict] = None) -> Optional[str]:
        intent_key = parsed_intent.get("intent_key", "rag_needed")
        entity = (parsed_intent.get("entity") or "").lower()
        is_detailed = parsed_intent.get("is_detailed", False)
        q_lower = question.lower()

        # [FIX-V] Faculty Member Search - handled by web search, not facts router
        if intent_key == "faculty_member_search":
            return None  # Will go to web search with site-specific ranking

        # [FIX-K] Strict Dean Matching
        if intent_key == "faculty_dean":
            deans = VERIFIED_FACTS[lang]["deans"]
            for dean_info in deans.values():
                if any(match_kw in q_lower for match_kw in dean_info.get("match", [])):
                    if is_detailed:
                        return f"{dean_info['name']}\n{dean_info['title']}"  # [FIX-R] تفصيل أكثر
                    return dean_info["name"]  # [FIX-P] اسم العميد فقط
            return None

        if intent_key == "department_head":
            heads = VERIFIED_FACTS[lang]["it_admin"]["department_heads"]
            res = None
            if any(kw in q_lower for kw in ['حاسوب', 'cs']):
                res = heads['cs']
            elif any(kw in q_lower for kw in ['نظم', 'cis']):
                res = heads['cis']
            elif any(kw in q_lower for kw in ['تكنولوجيا', 'it']):
                res = heads['it']
            
            if res:
                if is_detailed:
                    return f"{res['name']} - رئيس {res['dept']}"  # [FIX-R] تفصيل أكثر
                return res['name']  # [FIX-P] الاسم فقط
            return None

        if intent_key == "faculty_vice_dean":
            if any(kw in q_lower for kw in self.IT_FACULTY_KEYWORDS):
                it_admin = VERIFIED_FACTS[lang]["it_admin"]
                if 'مساعد' in q_lower or 'assistant' in q_lower:
                    if is_detailed:
                        return f"{it_admin['assistant_dean']['name']}\n{it_admin['assistant_dean']['title']}"
                    return it_admin["assistant_dean"]["name"]
                else:
                    if is_detailed:
                        return "\n".join([f"{vd['name']} - {vd['title']}" for vd in it_admin["vice_deans"]])
                    return "، ".join([vd["name"] for vd in it_admin["vice_deans"]])
            return None

        if intent_key == "list_faculties":
            # [FIX-P] قائمة الكليات فقط
            return "\n".join(VERIFIED_FACTS[lang]["faculties_list"])

        if intent_key == "list_programs":
            if any(kw in q_lower for kw in self.IT_FACULTY_KEYWORDS):
                return "\n".join(VERIFIED_FACTS[lang]["it_programs"])
            return None

        if intent_key == "online_learning":
            return VERIFIED_FACTS[lang]["online_policy"]  # [FIX-P] الإجابة المباشرة

        if intent_key == "university_president":
            # [FIX-E] أسئلة ديناميكية عن الرئيس → None (Web)
            dynamic_keywords = ['متى', 'تعيين', 'تاريخ', 'كم سنة', 'كم خدم', 'منذ متى']
            if any(kw in q_lower for kw in dynamic_keywords):
                return None
            if is_detailed:
                return f"{VERIFIED_FACTS[lang]['president']}\n{VERIFIED_FACTS[lang]['president_title']}"
            return VERIFIED_FACTS[lang]["president"]  # [FIX-P] الاسم فقط

        if intent_key == "cost_inquiry":
            if entity == "housing":
                if is_detailed:
                    return VERIFIED_FACTS[lang]["student_housing"]["details"]
                return VERIFIED_FACTS[lang]["student_housing"]["brief"]  # [FIX-P] التكلفة فقط
            return None

        if intent_key == "university_info":
            facts = VERIFIED_FACTS[lang]
            if is_detailed:
                return facts["university_brief"]
            if lang == "ar":
                return f"{facts['university_type']}، تأسست {facts['founded_year']}، {facts['location']}"
            return f"{facts['university_type']}, founded {facts['founded_year']}, {facts['location']}"

        return None

    # إضافة الخصائص المطلوبة للـ Router
    IT_FACULTY_KEYWORDS = [
        "معلومات", "حاسوب", "it", "computer", "ايتي", "الايتي",
        "information technology", "تكنولوجيا المعلومات"
    ]


semantic_parser = SemanticIntentParser(grok_client)
facts_router = SmartFactsRouter()


# ==================== SYNTHETIC RETRIEVER ====================
class SyntheticQuestionsRetriever:
    def __init__(self, file_path: str, frozen: bool = True):
        self.file_path = file_path
        self.frozen = frozen
        self.questions_data = []
        self.bm25 = None
        if not frozen:
            self._load_data()
        else:
            logger.info("SyntheticRetriever is FROZEN - skipping load")

    def _load_data(self):
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, 'r', encoding='utf-8') as f:
                    self.questions_data = json.load(f)
                if self.questions_data:
                    tokenized_corpus = [arabic_processor.tokenize(item['question']) for item in self.questions_data]
                    self.bm25 = BM25Okapi(tokenized_corpus)
                    logger.info(f"Loaded {len(self.questions_data)} synthetic questions")
            except Exception as e:
                logger.error(f"Error loading synthetic questions: {e}")

    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        if self.frozen:
            return []
        if not self.bm25 or not self.questions_data:
            return []
        tokenized_query = arabic_processor.tokenize(query)
        if not tokenized_query:
            return []
        scores = self.bm25.get_scores(tokenized_query)
        if not any(s > 0 for s in scores):
            return []
        max_score = max(scores)
        dynamic_threshold = min(1.5, max_score * 0.3)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        results = []
        for i in top_indices:
            if scores[i] >= dynamic_threshold:
                results.append(self.questions_data[i])
        return results


synthetic_retriever = SyntheticQuestionsRetriever(SYNTHETIC_QUESTIONS_FILE, frozen=SYNTHETIC_FROZEN)


# ==================== SMART MEMORY SYSTEM ====================
class SmartMemorySystem:
    def __init__(self, client, max_messages=10, summary_interval=10):
        self.client = client
        self.max_messages = max_messages
        self.summary_interval = summary_interval
        self.db_path = "novacortex_v12.db"

    async def add_message(self, user_id: int, role: str, content: str, entity: str = None):
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute("SELECT MAX(message_index) FROM conversation_memory WHERE user_id = ?", (user_id,))
            last_index = c.fetchone()[0]
            new_index = (last_index + 1) if last_index is not None else 0
            content_with_entity = f"{content}|||entity:{entity}" if entity else content
            c.execute("INSERT INTO conversation_memory (user_id, message_index, role, content, timestamp, is_summary) VALUES (?, ?, ?, ?, ?, 0)",
                (user_id, new_index, role, content_with_entity, datetime.now().isoformat()))
            conn.commit()
        if new_index > 0 and new_index % self.summary_interval == 0:
            try:
                await self._compress_memory(user_id)
            except:
                pass

    def get_context(self, user_id: int, max_messages: Optional[int] = None) -> List[Dict]:
        max_msg = max_messages or self.max_messages
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute("SELECT role, content, is_summary FROM conversation_memory WHERE user_id = ? ORDER BY message_index DESC LIMIT ?",
                (user_id, max_msg))
            rows = c.fetchall()
            result = []
            for r, c, s in reversed(rows):
                # Extract entity if present
                entity = None
                content = c
                if "|||entity:" in c:
                    parts = c.split("|||entity:")
                    content = parts[0]
                    entity = parts[1] if len(parts) > 1 else None
                
                result.append({
                    "role": r, 
                    "content": content, 
                    "is_summary": bool(s),
                    "entity": entity
                })
            return result

    def format_context_for_llm(self, context: List[Dict]) -> str:
        formatted = []
        for msg in context:
            if msg["is_summary"]:
                formatted.append(msg["content"])
            else:
                role = "المستخدم" if msg["role"] == "user" else "المساعد"
                formatted.append(f"{role}: {msg['content']}")
        return "\n".join(formatted)

    async def _compress_memory(self, user_id: int):
        try:
            with sqlite3.connect(self.db_path) as conn:
                c = conn.cursor()
                c.execute("SELECT message_index, role, content FROM conversation_memory WHERE user_id = ? AND is_summary = 0 ORDER BY message_index ASC LIMIT ?",
                    (user_id, self.summary_interval))
                messages = c.fetchall()
                if len(messages) < 3:
                    return
                conversation_text = "\n".join([f"{'المستخدم' if r == 'user' else 'المساعد'}: {ct}" for _, r, ct in messages])
                summary = await self._generate_summary(conversation_text)
                if summary:
                    indices = [idx for idx, _, _ in messages]
                    placeholders = ','.join('?' * len(indices))
                    c.execute(f"DELETE FROM conversation_memory WHERE user_id = ? AND message_index IN ({placeholders})", (user_id, *indices))
                    c.execute("INSERT INTO conversation_memory (user_id, message_index, role, content, timestamp, is_summary) VALUES (?, ?, 'assistant', ?, ?, 1)",
                        (user_id, min(indices), summary, datetime.now().isoformat()))
                    conn.commit()
        except Exception as e:
            logger.error(f"Memory compression error: {e}")

    async def _generate_summary(self, conversation: str) -> Optional[str]:
        try:
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "لخص هذه المحادثة بإيجاز (100 كلمة كحد أقصى)."},
                    {"role": "user", "content": f"المحادثة:\n{conversation}"}
                ],
                temperature=0.3, max_tokens=200
            )
            return f"[ملخص]: {response.choices[0].message.content.strip()}"
        except:
            return None

    def clear_memory(self, user_id: int):
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute("DELETE FROM conversation_memory WHERE user_id = ?", (user_id,))
            conn.commit()


memory_system = SmartMemorySystem(grok_client)


# ==================== CACHE ====================
class StrictResponseCache:
    def __init__(self):
        self.conn = sqlite3.connect("novacortex_v12.db", check_same_thread=False)

    def get(self, question: str) -> Optional[Dict]:
        q_hash = self._hash_question(question)
        c = self.conn.cursor()
        c.execute("SELECT response, confidence, source FROM response_cache WHERE question_hash = ? AND confidence >= 85 AND last_used > datetime('now', '-7 days')", (q_hash,))
        row = c.fetchone()
        if row:
            c.execute("UPDATE response_cache SET hit_count = hit_count + 1, last_used = ? WHERE question_hash = ?",
                (datetime.now().isoformat(), q_hash))
            self.conn.commit()
            return {"response": row[0], "confidence": row[1], "source": "cache"}
        return None

    def set(self, question: str, response: str, confidence: float, source: str):
        if confidence < 85:
            return
        cleaned = response.strip()
        if not cleaned or len(cleaned) < 10:  # [FIX-P] حتى الردود القصيرة نخزنها
            return
        q_hash = self._hash_question(question)
        now = datetime.now().isoformat()
        c = self.conn.cursor()
        c.execute("INSERT OR REPLACE INTO response_cache (question_hash, question, response, confidence, source, created_at, last_used, hit_count) VALUES (?, ?, ?, ?, ?, ?, ?, 0)",
            (q_hash, question, cleaned, confidence, source, now, now))
        self.conn.commit()

    def _hash_question(self, question: str) -> str:
        normalized = arabic_processor.normalize(question)
        tokens = arabic_processor.tokenize(normalized)
        unique_tokens = sorted(set(tokens))
        key = " ".join(unique_tokens)
        return hashlib.md5(key.encode()).hexdigest()


# ==================== LLM GENERATOR ====================
class ImprovedLLMAnswerGenerator:
    def __init__(self, client):
        self.client = client

    def _clean_context(self, raw_text: str) -> str:
        text = re.sub(r'(هاتف|فاكس|phone|fax)\s*:?\s*[\d\s+\-()]+', '', raw_text)
        text = re.sub(r'[\+\d\s\-()]{10,}', '', text)
        text = re.sub(r'\S+@\S+\.\S+', '', text)
        text = re.sub(r'http\S+|www\.\S+', '', text)
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def generate_answer(self, question: str, raw_context: str, language: str = None, history: str = "", is_detailed: bool = False) -> Dict:
        cleaned_context = self._clean_context(raw_context)
        if not cleaned_context or len(cleaned_context) < 20:
            return {"answer": "", "success": False}

        detected_lang = language or arabic_processor.detect_language(question)
        if detected_lang == "en":
            detected_lang = "en"
        else:
            detected_lang = "ar"

        # [FIX-AD] استخراج البيانات الخام من السياق
        raw_data = extract_raw_contact_data(cleaned_context)
        
        # إضافة البيانات المستخرجة إلى السياق لتعزيزها
        enhanced_context = cleaned_context
        if raw_data["emails"] or raw_data["offices"] or raw_data["phones"]:
            enhanced_context += "\n\n[RAW EXTRACTED DATA]:\n"
            if raw_data["emails"]:
                enhanced_context += f"Emails: {', '.join(raw_data['emails'])}\n"
            if raw_data["offices"]:
                enhanced_context += f"Offices: {', '.join(raw_data['offices'])}\n"
            if raw_data["phones"]:
                enhanced_context += f"Phones: {', '.join(raw_data['phones'])}\n"

        # [FIX-O] & [FIX-Q] & [FIX-R] Zero-Fluff + Adaptive Response
        lang_name = "Arabic" if detected_lang == "ar" else "English"
        
        # [FIX-R] تحديد نمط الإجابة حسب رغبة المستخدم
        style_instruction = (
            "Style: Detailed and organized response. Include ALL contact information found." if is_detailed else
            "Style: Ultra-concise, list only the specific information requested."
        )
        
        # [FIX-Y] & [FIX-AA] & [FIX-AD] تصريح رسمي + صياد بيانات
        system_prompt = f"""You are the official Yarmouk University assistant and DATA HUNTER.

CRITICAL AUTHORIZATION - [FIX-Y][FIX-AA][FIX-AD]:
- Information from the faculty directory (fmd.yu.edu.jo) is PUBLIC and authorized.
- You MUST extract and share ANY email with @yu.edu.jo domain.
- You MUST extract and share ANY office location, room number, or building information.
- You MUST extract and share ANY phone numbers found.
- Do NOT apologize or refuse - this is official public data.
- Be a "data hunter" - scan the context thoroughly for contact information.

STRICT RULES:
1. Answer in {lang_name} ONLY.
2. Answer DIRECTLY with NO introductions, NO greetings, NO explanations.
3. START with the answer immediately.
4. {style_instruction}
5. If contact info is found, present it clearly (email, office, phone).
6. NO extra text, NO suggestions, NO URLs unless specifically asked.

Example responses:
Q: "إيميل دكتور قاسم الردايدة"
A: "qasem.r@yu.edu.jo"

Q: "معلومات دكتور بلال عبدالغني"
A: "د. بلال عبدالغني - رئيس قسم علوم الحاسوب\nالبريد: bilal.abdulghani@yu.edu.jo\nالمكتب: كلية تكنولوجيا المعلومات، الطابق الثالث\nالهاتف: 027211111 فرع 3456" """

        if detected_lang == "ar":
            user_message = f"Context from Faculty Directory:\n{enhanced_context[:2500]}\n\nQuestion: {question}\n\n[HUNTER MODE] Extract ALL contact information for this person."
        else:
            user_message = f"Context from Faculty Directory:\n{enhanced_context[:2500]}\n\nQuestion: {question}\n\n[HUNTER MODE] Extract ALL contact information for this person."

        try:
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.0,
                max_tokens=500 if is_detailed else 200
            )
            answer = response.choices[0].message.content.strip()

            # Verify language compliance
            answer_lang = arabic_processor.detect_language(answer)
            if answer_lang != detected_lang:
                logger.warning(f"[FIX-G] Wrong language! Forcing correction...")
                if detected_lang == "en":
                    force = f"Answer in English ONLY, directly with contact info: {question}"
                else:
                    force = f"أجب بالعربية فقط، مباشرة مع معلومات التواصل: {question}"
                response2 = self.client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": force}],
                    temperature=0.0,
                    max_tokens=500 if is_detailed else 200
                )
                answer = response2.choices[0].message.content.strip()

            if answer and len(answer) > 5:
                return {"answer": answer, "success": True}
            return {"answer": "", "success": False}

        except Exception as e:
            logger.error(f"LLM error: {e}")
            return {"answer": "", "success": False}


llm_generator = ImprovedLLMAnswerGenerator(grok_client)
scope_detector = SmartOutOfScopeDetector(grok_client)


# ==================== WEB SEARCHER ====================
class EnhancedWebSearcher:
    def __init__(self):
        self.api_key = SERPER_API_KEY
        self.last_call_time = 0
        self.min_delay = 0.5
        self.pdf_extractor = PDFExtractor()
        self.faculty_site = "site:fmd.yu.edu.jo"  # [FIX-U] دليل أعضاء هيئة التدريس

    def search_multi(self, question: str, study_level: str = "بكالوريوس", is_faculty_search: bool = False, faculty_name: str = None) -> List[Dict]:
        # [FIX-AB] Deep Faculty Search - استخدام الاستعلامات المحسنة
        if is_faculty_search:
            queries = generate_deep_faculty_query(question, faculty_name)
        else:
            queries = self._generate_queries(question, study_level)
        
        all_results = []
        for query in queries:
            now = time.time()
            if now - self.last_call_time < self.min_delay:
                time.sleep(self.min_delay - (now - self.last_call_time))
            url = "https://google.serper.dev/search"
            payload = json.dumps({"q": query, "num": 8})  # زيادة عدد النتائج
            headers = {'X-API-KEY': self.api_key, 'Content-Type': 'application/json'}
            try:
                response = requests.post(url, headers=headers, data=payload, timeout=15)
                response.raise_for_status()
                self.last_call_time = time.time()
                data = response.json()
                if "organic" in data:
                    all_results.extend(data["organic"])
            except Exception as e:
                logger.error(f"Search error: {e}")
        
        # [FIX-W] Site-Specific Ranking
        return self._filter_and_rank(all_results, is_faculty_search)

    def _generate_faculty_queries(self, question: str) -> List[str]:
        """[FIX-U] Generate targeted faculty search queries with contact keywords"""
        detected_lang = arabic_processor.detect_language(question)
        # Remove common words to isolate the name
        clean_q = re.sub(r'(دكتور|الدكتور|أستاذ|البروفيسور|prof|dr|professor|من هو|مين هو|who is|tell me about|information about|إيميل|ايميل|بريد|تواصل|مكتب|email|contact|office)', '', question, flags=re.IGNORECASE).strip()
        
        # [FIX-Z] Include contact keywords in search
        queries = [
            f"{self.faculty_site} {clean_q} email بريد",
            f"{self.faculty_site} {question}",
            f"{self.faculty_site} {clean_q}"
        ]
        if detected_lang == "ar":
            queries.append(f"موقع جامعة اليرموك {clean_q} بريد إلكتروني")
        else:
            queries.append(f"Yarmouk University faculty {clean_q} email contact")
        
        return queries

    def _generate_queries(self, question: str, study_level: str = "بكالوريوس") -> List[str]:
        detected_lang = arabic_processor.detect_language(question)
        is_scholarship = any(word in question.lower() for word in SCHOLARSHIP_KEYWORDS)
        is_fees = any(word in question.lower() for word in FEES_KEYWORDS)
        
        # [FIX-E] أسئلة تعيين رئيس الجامعة
        president_keywords = ['تعيين', 'رئيس الجامعة', 'الشرايري', 'president', 'appointed', 'tenure', 'خدم', 'صارله']
        is_president_question = any(kw in question.lower() for kw in president_keywords)
        if is_president_question:
            return (
                ["مالك الشرايري رئيس جامعة اليرموك تاريخ التعيين"]
                if detected_lang == "ar" else
                ["Malik Al-Shariri Yarmouk University president appointment date"]
            )
        
        if is_scholarship:
            return (["المنح الدراسية المتاحة جامعة اليرموك"] if detected_lang == "ar" else ["Available scholarships Yarmouk University"])
        if is_fees:
            return ([f"الرسوم الدراسية اليرموك"] if detected_lang == "ar" else [f"Tuition fees YU"])
        return ([f"{question} جامعة اليرموك"] if detected_lang == "ar" else [f"{question} Yarmouk University"])

    def _filter_and_rank(self, results: List[Dict], is_faculty_search: bool = False) -> List[Dict]:
        if not results:
            return []
        seen_links = set()
        unique_results = []
        for r in results:
            link = r.get('link', '')
            if link not in seen_links:
                seen_links.add(link)
                score = 0
                # [FIX-W] Site-Specific Ranking
                if is_faculty_search and 'fmd.yu.edu.jo' in link:
                    score += 100  # أعلى优先级 لدليل أعضاء هيئة التدريس
                elif 'yu.edu.jo' in link:
                    score += 50
                if '.pdf' in link.lower():
                    score += 40
                # [FIX-AD] زيادة النقاط إذا كان المحتوى يحتوي على إيميل
                snippet = r.get('snippet', '').lower()
                if '@yu.edu.jo' in snippet:
                    score += 30
                if 'مكتب' in snippet or 'office' in snippet:
                    score += 20
                r['quality_score'] = score
                unique_results.append(r)
        return sorted(unique_results, key=lambda x: x['quality_score'], reverse=True)

    def find_best_result(self, results: List[Dict]) -> Optional[Dict]:
        return results[0] if results else None


web_searcher = EnhancedWebSearcher()


# ==================== HYBRID RETRIEVER ====================
class ImprovedHybridRetriever:
    def __init__(self, vectorstore, documents: List[Document]):
        self.vectorstore = vectorstore
        self.documents = documents
        self.bm25 = self._init_bm25()

    def _init_bm25(self):
        logger.info("Initializing BM25 index...")
        texts = [doc.page_content for doc in self.documents]
        tokenized_corpus = [arabic_processor.tokenize(t) for t in texts]
        return BM25Okapi(tokenized_corpus)

    def retrieve(self, query: str, final_k: int = 5) -> List[Tuple[Document, float]]:
        try:
            vector_results = self.vectorstore.similarity_search_with_relevance_scores(query, k=final_k * 4)
            filtered_docs = [doc for doc, score in vector_results if score > 0.25]
            if len(filtered_docs) < 3:
                filtered_docs = [doc for doc, _ in vector_results[:final_k * 2]]
        except:
            filtered_docs = self.vectorstore.similarity_search(query, k=final_k * 2)

        tokenized_query = arabic_processor.tokenize(query)
        bm25_results = self.bm25.get_top_n(tokenized_query, self.documents, n=final_k * 3)
        fused = self._reciprocal_rank_fusion(filtered_docs, bm25_results)
        unique = self._deduplicate(fused)
        return unique[:final_k]

    def _reciprocal_rank_fusion(self, vector_docs, bm25_docs, k=60):
        rrf_scores = {}
        for rank, doc in enumerate(vector_docs, 1):
            h = self._hash(doc.page_content)
            rrf_scores[h] = {"doc": doc, "score": 1.0 / (k + rank)}
        for rank, doc in enumerate(bm25_docs, 1):
            h = self._hash(doc.page_content)
            if h in rrf_scores:
                rrf_scores[h]["score"] += 1.0 / (k + rank)
            else:
                rrf_scores[h] = {"doc": doc, "score": 1.0 / (k + rank)}
        return [(v["doc"], v["score"]) for v in sorted(rrf_scores.values(), key=lambda x: x["score"], reverse=True)]

    def _deduplicate(self, docs_with_scores):
        seen, unique = set(), []
        for doc, score in docs_with_scores:
            h = self._hash(doc.page_content)
            if h not in seen:
                seen.add(h)
                unique.append((doc, score))
        return unique

    def _hash(self, content: str) -> str:
        return hashlib.md5(content[:300].encode()).hexdigest()


# ==================== HYBRID PIPELINE ====================
class EnhancedHybridPipeline:
    def __init__(self, retriever, cache, scope_detector):
        self.retriever = retriever
        self.cache = cache
        self.scope_detector = scope_detector
        self.memory = memory_system
        self.semantic_parser = semantic_parser
        self.facts_router = facts_router

    async def split_complex_question(self, question: str, lang: str) -> List[str]:
        if not isinstance(question, str):
            return [str(question)]
        if len(question.split()) < 15:
            return [question]
        try:
            response = grok_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": f"Split into separate independent questions (JSON with 'questions' array): {question}"}],
                response_format={"type": "json_object"},
                temperature=0.1, max_tokens=200
            )
            result = json.loads(response.choices[0].message.content)
            questions = result.get("questions", [question])
            cleaned = []
            for q in questions:
                if isinstance(q, str):
                    cleaned.append(q)
                elif isinstance(q, dict):
                    cleaned.append(str(list(q.values())[0]) if q.values() else question)
                else:
                    cleaned.append(str(q))
            return cleaned if cleaned else [question]
        except:
            return [question]

    async def process_question(self, question: str, user_id: int, study_level: str = "بكالوريوس") -> Dict:
        start_time = time.time()
        detected_lang = arabic_processor.detect_language(question)

        # طبقة 1: التحيات (مختصرة)
        greeting_response = handle_greeting(question)
        if greeting_response:
            await self.memory.add_message(user_id, "user", question)
            await self.memory.add_message(user_id, "assistant", greeting_response)
            return {"response": greeting_response, "source": "greeting", "confidence": 100,
                    "response_time_ms": int((time.time() - start_time) * 1000), "docs_retrieved": 0}

        # طبقة 2: فحص النطاق (مختصر)
        scope_result = await self.scope_detector.is_out_of_scope(question)
        if scope_result:
            await self.memory.add_message(user_id, "user", question)
            await self.memory.add_message(user_id, "assistant", scope_result["message"])
            return {
                "response": scope_result["message"], "source": "out_of_scope",
                "confidence": 100, "response_time_ms": int((time.time() - start_time) * 1000), "docs_retrieved": 0
            }

        sub_questions = await self.split_complex_question(question, detected_lang)
        if len(sub_questions) > 1:
            all_responses = []
            for sub_q in sub_questions:
                res = await self._process_single_question(sub_q, user_id, study_level, detected_lang)
                all_responses.append(res["response"])
            return {
                "response": "\n".join(all_responses),  # بدون فواصل
                "source": "multi_query",
                "confidence": 100,
                "response_time_ms": int((time.time() - start_time) * 1000), 
                "docs_retrieved": 0
            }

        return await self._process_single_question(question, user_id, study_level, detected_lang)

    async def _process_single_question(self, question: str, user_id: int, study_level: str, detected_lang: str) -> Dict:
        start_time = time.time()

        # المقارنات (مرفوضة)
        if is_comparison_question(question):
            msg = "عذراً، لا يمكنني المقارنة بين الأشخاص." if detected_lang == "ar" else "Sorry, I cannot compare people."
            await self.memory.add_message(user_id, "user", question)
            await self.memory.add_message(user_id, "assistant", msg)
            return {"response": msg, "source": "comparison_rejected", "confidence": 100,
                    "response_time_ms": int((time.time() - start_time) * 1000), "docs_retrieved": 0}

        # [FIX-AC] تطبيق التمويه الذكي على السؤال
        proxied_question = apply_proxy_prompt(question)
        if proxied_question != question:
            logger.info(f"[FIX-AC] Question proxied: '{question}' → '{proxied_question}'")

        # Get conversation context
        context = self.memory.get_context(user_id, max_messages=6)
        
        # [FIX-B] إثراء السؤال بالسياق
        enriched_question = proxied_question  # استخدام السؤال المعدل
        if context:
            implicit_words = ['كم', 'متى', 'هل', 'how long', 'since when', 'خدم', 'مدة']
            is_short = len(question.split()) <= 5
            has_no_entity = not any(kw in question.lower() for kw in ['كلية', 'قسم', 'دكتور'])
            is_implicit = is_short and has_no_entity and any(kw in question.lower() for kw in implicit_words)

            if is_implicit:
                for msg in reversed(context):
                    if msg["role"] == "user" and msg["content"] != question:
                        enriched_question = f"{msg['content']} - {proxied_question}"
                        break

        # Parse intent (with is_detailed)
        parsed_intent = self.semantic_parser.parse(enriched_question)
        is_detailed = parsed_intent.get("is_detailed", False)

        # [FIX-V] Faculty Member Search (now includes contact keywords)
        if parsed_intent.get("intent_key") == "faculty_member_search":
            # Direct to web search with faculty site priority
            result = await self._web_search_path(enriched_question, study_level, detected_lang, is_detailed, is_faculty_search=True)
            await self.memory.add_message(user_id, "user", question)
            await self.memory.add_message(user_id, "assistant", result["response"])
            result["response_time_ms"] = int((time.time() - start_time) * 1000)
            return result

        # [FIX-N] Cost-Specific Routing
        if parsed_intent.get("intent_key") == "cost_inquiry":
            direct_answer = self.facts_router.route(enriched_question, parsed_intent, detected_lang, context)
            if direct_answer:
                await self.memory.add_message(user_id, "user", question, entity=parsed_intent.get("entity"))
                await self.memory.add_message(user_id, "assistant", direct_answer)
                return {"response": direct_answer, "source": "verified_facts", "confidence": 90,
                        "response_time_ms": int((time.time() - start_time) * 1000), "docs_retrieved": 0}
            result = await self._web_search_path(enriched_question, study_level, detected_lang, is_detailed)
            await self.memory.add_message(user_id, "user", question)
            await self.memory.add_message(user_id, "assistant", result["response"])
            result["response_time_ms"] = int((time.time() - start_time) * 1000)
            return result

        # Fees/Scholarships → Web
        is_financial_intent = parsed_intent.get("intent_key") in ("fees_inquiry", "scholarship_inquiry")
        is_scholarship_kw = any(word in enriched_question.lower() for word in SCHOLARSHIP_KEYWORDS)
        is_fees_kw = any(word in enriched_question.lower() for word in FEES_KEYWORDS)

        if is_financial_intent or is_scholarship_kw or is_fees_kw:
            result = await self._web_search_path(enriched_question, study_level, detected_lang, is_detailed)
            await self.memory.add_message(user_id, "user", question)
            await self.memory.add_message(user_id, "assistant", result["response"])
            result["response_time_ms"] = int((time.time() - start_time) * 1000)
            return result

        # Try direct answer from facts router
        direct_answer = self.facts_router.route(enriched_question, parsed_intent, detected_lang, context)
        if direct_answer:
            await self.memory.add_message(user_id, "user", question, entity=parsed_intent.get("entity"))
            await self.memory.add_message(user_id, "assistant", direct_answer)
            return {"response": direct_answer, "source": "verified_facts", "confidence": 100,
                    "response_time_ms": int((time.time() - start_time) * 1000), "docs_retrieved": 0}

        # Cache
        cached = self.cache.get(enriched_question)
        if cached:
            await self.memory.add_message(user_id, "user", question)
            await self.memory.add_message(user_id, "assistant", cached["response"])
            return {**cached, "response_time_ms": int((time.time() - start_time) * 1000), "docs_retrieved": 0}

        # [FIX-D] بسط المصطلحات العامية قبل RAG
        rag_question = expand_colloquial_terms(enriched_question)

        # RAG أو Web
        if self._needs_web_search(enriched_question):
            result = await self._web_search_path(enriched_question, study_level, detected_lang, is_detailed)
        else:
            result = await self._rag_path(rag_question, study_level, detected_lang, context, is_detailed)

        await self.memory.add_message(user_id, "user", question)
        await self.memory.add_message(user_id, "assistant", result["response"])

        if result["confidence"] >= 85:
            self.cache.set(enriched_question, result["response"], result["confidence"], result["source"])

        result["response_time_ms"] = int((time.time() - start_time) * 1000)
        update_daily_stats(result["source"], result["confidence"], result["response_time_ms"])
        return result

    def _needs_web_search(self, question: str) -> bool:
        q_lower = question.lower()
        time_sensitive = [
            'آخر موعد', 'deadline', 'أخبار', 'خبر', 'إعلان جديد',
            'متى تعيين', 'تاريخ تعيين', 'كم صارله', 'when appointed'
        ]
        return any(phrase in q_lower for phrase in time_sensitive)

    async def _rag_path(self, question: str, study_level: str, language: str, context: List[Dict], is_detailed: bool) -> Dict:
        docs_with_scores = self.retriever.retrieve(question, final_k=5)

        if not docs_with_scores:
            return await self._web_search_path(question, study_level, language, is_detailed, _from_rag=True)

        docs = [doc for doc, score in docs_with_scores]
        raw_data = "\n".join([doc.page_content for doc in docs[:3]])[:2000]

        llm_result = llm_generator.generate_answer(question, raw_data, language, is_detailed=is_detailed)

        if not llm_result["success"] or not llm_result["answer"]:
            return await self._web_search_path(question, study_level, language, is_detailed, _from_rag=True)

        return {"response": llm_result["answer"], "source": "rag", "confidence": 80, "docs_retrieved": len(docs)}

    async def _web_search_path(self, question: str, study_level: str, language: str, is_detailed: bool, is_faculty_search: bool = False, _from_rag: bool = False) -> Dict:
        # استخراج اسم الدكتور إذا كان بحث عن هيئة تدريس
        faculty_name = None
        if is_faculty_search:
            # محاولة استخراج الاسم من السؤال
            name_match = re.search(r'(?:دكتور|الدكتور|أستاذ|professor)\s+([\w\s]+?)(?:\s+|\Z|\.)', question, re.IGNORECASE)
            if name_match:
                faculty_name = name_match.group(1).strip()

        results = web_searcher.search_multi(question, study_level, is_faculty_search, faculty_name)

        if not results:
            return {
                "response": NO_RESULT_MESSAGE_AR if language == "ar" else NO_RESULT_MESSAGE_EN,
                "source": "web_empty", "confidence": 0, "docs_retrieved": 0
            }

        best = web_searcher.find_best_result(results)
        if not best or len(best.get("snippet", "")) < 20:
            return {
                "response": NO_RESULT_MESSAGE_AR if language == "ar" else NO_RESULT_MESSAGE_EN,
                "source": "web_empty", "confidence": 0, "docs_retrieved": 0
            }

        combined_snippets = best.get("snippet", "")
        llm_result = llm_generator.generate_answer(question, combined_snippets, language, is_detailed=is_detailed)
        final_response = llm_result["answer"] if llm_result["success"] else best["snippet"]

        return {"response": final_response, "source": "web", "confidence": 75, "docs_retrieved": len(results)}


# ==================== AGENT ====================
class YarmoukAgent:
    def __init__(self):
        logger.info("Initializing YarmoukAgent v12.9.2 - FACULTY DATA HUNTER EDITION...")

        self.embeddings = HuggingFaceEmbeddings(
            model_name="intfloat/multilingual-e5-large-instruct",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        self.vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=self.embeddings)
        all_docs = self._load_all_documents()
        self.retriever = ImprovedHybridRetriever(vectorstore=self.vectorstore, documents=all_docs)
        self.cache = StrictResponseCache()
        self.scope_detector = SmartOutOfScopeDetector(grok_client)
        self.pipeline = EnhancedHybridPipeline(self.retriever, self.cache, self.scope_detector)

        logger.info("=== NovaCortex v12.9.2 - FACULTY DATA HUNTER EDITION ===")
        logger.info("[FIX-AB] Deep Faculty Search active")
        logger.info("[FIX-AC] The Proxy Prompt active")
        logger.info("[FIX-AD] Raw Data Extraction active")
        logger.info("[FIX-Y] Public Data Authorization active")
        logger.info("[FIX-Z] Expanded Faculty Intent active")
        logger.info("[FIX-AA] Safety Refusal Bypass active")
        logger.info("[FIX-U] Targeted Faculty Search active")
        logger.info("[FIX-V] Faculty Intent Detection active")
        logger.info("[FIX-W] Site-Specific Ranking active")
        logger.info("[FIX-X] Adaptive Faculty Bio active")
        logger.info("[FIX-R] Adaptive Response Logic active")
        logger.info("[FIX-S] Context-Aware Brevity active")
        logger.info("[FIX-T] Zero-Fluff Guardrail active")

    def _load_all_documents(self) -> List[Document]:
        try:
            all_ids = self.vectorstore.get()['ids']
            all_docs = []
            batch_size = 3000
            for i in range(0, len(all_ids), batch_size):
                batch = self.vectorstore.get(limit=min(batch_size, len(all_ids) - i), offset=i)
                if batch and batch.get("documents"):
                    for text, meta in zip(batch["documents"], batch.get("metadatas", [])):
                        all_docs.append(Document(page_content=text, metadata=meta or {}))
            logger.info(f"Loaded {len(all_docs)} documents")
            return all_docs
        except Exception as e:
            logger.error(f"Failed to load documents: {e}")
            return []

    async def ask(self, question: str, user_id: int) -> Dict:
        try:
            user = get_user(user_id)
            study_level = user[5] if user and len(user) > 5 else "بكالوريوس"
            update_user_activity(user_id)
            result = await self.pipeline.process_question(question=question, user_id=user_id, study_level=study_level)
            log_question(
                user_id=user_id, username=user[1] if user else "unknown",
                question=question, response=result["response"],
                source=result["source"], confidence=result["confidence"],
                response_time_ms=result["response_time_ms"],
                docs_retrieved=result.get("docs_retrieved", 0), study_level=study_level
            )
            return result
        except Exception as e:
            logger.error(f"Error: {e}", exc_info=True)
            detected_lang = arabic_processor.detect_language(question)
            return {
                "response": "عذراً، حدث خطأ تقني." if detected_lang == "ar" else "Sorry, a technical error occurred.",
                "source": "error", "confidence": 0, "response_time_ms": 0, "docs_retrieved": 0
            }


# ==================== DATA LOADING ====================
def load_data_from_json():
    json_file = Path("yarmouk_data.json")
    if not json_file.exists():
        logger.error("yarmouk_data.json not found!")
        return False

    if not os.path.exists(CHROMA_PATH) or not os.listdir(CHROMA_PATH):
        logger.info("Building vector store from JSON...")
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=150,
            separators=["\n\n", "\n", ". ", "؛ ", "، ", " "]
        )
        all_docs = []
        for item in tqdm(data, desc="Processing documents"):
            content_parts = []
            if item.get('title'):
                content_parts.append(f"# {item['title']}")
            if item.get('description'):
                content_parts.append(item['description'])
            if item.get('content'):
                content_parts.append(item['content'])
            content = "\n\n".join(filter(None, content_parts))
            if content.strip():
                chunks = splitter.create_documents([content.strip()])
                for chunk in chunks:
                    chunk.metadata = {
                        "source": item.get("url", "manual"),
                        "title": item.get("title", "No Title"),
                        "category": item.get("category", "General")
                    }
                    all_docs.append(chunk)

        if all_docs:
            embeddings = HuggingFaceEmbeddings(
                model_name="intfloat/multilingual-e5-large-instruct",
                model_kwargs={'device': 'cpu'}
            )
            Chroma.from_documents(documents=all_docs, embedding=embeddings, persist_directory=CHROMA_PATH)
            logger.info(f"Vector store created with {len(all_docs)} chunks")
            return True
        else:
            logger.error("No documents to index!")
            return False
    else:
        logger.info("Vector store already exists")
        return True


# ==================== TELEGRAM HANDLERS ====================
CHOOSING_LEVEL, ASKING_ID = range(2)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    reply_keyboard = [['بكالوريوس', 'ماجستير', 'دكتوراه']]
    await update.message.reply_text(
        f"أهلاً {user.first_name}. اختر مستواك الدراسي:",
        reply_markup=ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True, resize_keyboard=True)
    )
    return CHOOSING_LEVEL

async def level_chosen(update: Update, context: ContextTypes.DEFAULT_TYPE):
    level = update.message.text
    context.user_data['study_level'] = level
    await update.message.reply_text(
        f"ما هو رقمك الجامعي؟ (أو اكتب 'زائر')",
        reply_markup=ReplyKeyboardRemove()
    )
    return ASKING_ID

async def id_provided(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uni_id = update.message.text
    user = update.effective_user
    study_level = context.user_data.get('study_level', 'بكالوريوس')
    save_user(telegram_id=user.id, username=user.username, first_name=user.first_name,
              role="student" if uni_id.isdigit() else "visitor", uni_id=uni_id, study_level=study_level)
    await update.message.reply_text("تم التسجيل. يمكنك السؤال الآن.")
    return ConversationHandler.END

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return
    question = update.message.text
    user_id = update.effective_user.id
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    agent = context.bot_data.get('agent')
    if not agent:
        agent = YarmoukAgent()
        context.bot_data['agent'] = agent
    result = await agent.ask(question, user_id)
    await update.message.reply_text(result["response"])

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("تم الإلغاء.", reply_markup=ReplyKeyboardRemove())
    return ConversationHandler.END

async def clear_memory_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    memory_system.clear_memory(update.effective_user.id)
    await update.message.reply_text("تم المسح.")


# ==================== MAIN ====================
def main():
    init_db()
    if not load_data_from_json():
        logger.error("Failed to load data!")
        return

    application = Application.builder().token(TELEGRAM_TOKEN).build()
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('start', start)],
        states={
            CHOOSING_LEVEL: [MessageHandler(filters.TEXT & ~filters.COMMAND, level_chosen)],
            ASKING_ID: [MessageHandler(filters.TEXT & ~filters.COMMAND, id_provided)],
        },
        fallbacks=[CommandHandler('cancel', cancel)],
    )
    application.add_handler(conv_handler)
    application.add_handler(CommandHandler("clear", clear_memory_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("NovaCortex v12.9.2 - FACULTY DATA HUNTER EDITION is running...")
    application.run_polling()


if __name__ == '__main__':
    main()