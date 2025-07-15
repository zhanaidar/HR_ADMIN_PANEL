"""
Конфигурация HR Admin Panel v2.0
Простая и понятная архитектура
"""

import os
import json
from pathlib import Path
from datetime import datetime

# Импорты пользователей и ролей
from users import find_user, get_demo_credentials
from roles import (
    get_user_role_name, 
    can_user_create_profession,
    can_user_approve_profession, 
    can_user_return_to_hr,
    can_user_view_questions,
    can_user_edit_tags,
    ROLE_NAMES,
    PROFESSION_STATUSES
)

# Базовые настройки
DEBUG = True
APP_HOST = "localhost"
APP_PORT = 8002

# API ключи
import os
from dotenv import load_dotenv

# Загружаем переменные окружения
load_dotenv()

# API ключи из .env файла
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError("⚠️ OPENAI_API_KEY не найден в .env файле!")

# Пути
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"
UPLOADS_DIR = BASE_DIR / "uploads"

# Создаем папки если их нет
for directory in [DATA_DIR, TEMPLATES_DIR, STATIC_DIR, UPLOADS_DIR]:
    directory.mkdir(exist_ok=True)

# Настройки файлов
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = {'.pdf', '.docx', '.doc', '.txt'}

# Настройки ИИ
OPENAI_MODEL = "gpt-4"
AI_TEMPERATURE = 0.2
AI_MAX_TOKENS = 2000

# Организация
ORGANIZATION = {
    "name": "Halyk Bank",
    "logo": "/static/images/logo.png",
    "colors": {
        "primary": "#1DB584",
        "secondary": "#FFD700",
        "accent": "#2C3E50"
    }
}

# Справочники по умолчанию (будут создаваться автоматически)
DEFAULT_DEPARTMENTS = [
    {
        "id": 1,
        "name": "IT Department",
        "description": "Информационные технологии",
        "head_email": "AlmasNy@halykbank.kz"
    },
    {
        "id": 2,
        "name": "HR Department", 
        "description": "Управление персоналом",
        "head_email": "DzhamilyaBa@halykbank.kz"
    },
    {
        "id": 3,
        "name": "Risk Department",
        "description": "Управление рисками", 
        "head_email": "head.risk@halykbank.kz"
    },
    {
        "id": 4,
        "name": "Analytics Department",
        "description": "Бизнес-аналитика",
        "head_email": "head.analytics@halykbank.kz"
    }
]

DEFAULT_PROFESSIONS = [
    "Software Developer",
    "Data Analyst", 
    "HR Specialist",
    "Risk Manager",
    "Business Analyst"
]

DEFAULT_SPECIALIZATIONS = [
    "Frontend Development",
    "Backend Development",
    "Full Stack Development",
    "Data Science",
    "Machine Learning",
    "DevOps",
    "Mobile Development",
    "Business Intelligence",
    "Talent Acquisition",
    "Risk Assessment"
]

# Функции теперь импортируются из users.py и roles.py
# find_user, get_user_role_name, can_user_* и другие

# Инициализация данных
def initialize_data():
    """Инициализация базовых данных при первом запуске"""
    
    # Создаем departments.json
    departments_file = DATA_DIR / "departments.json"
    if not departments_file.exists():
        with open(departments_file, 'w', encoding='utf-8') as f:
            json.dump({"departments": DEFAULT_DEPARTMENTS}, f, ensure_ascii=False, indent=2)
        print(f"✅ Создан {departments_file}")
    
    # Создаем profession_records.json (основной файл)
    records_file = DATA_DIR / "profession_records.json"
    if not records_file.exists():
        with open(records_file, 'w', encoding='utf-8') as f:
            json.dump({"profession_records": []}, f, ensure_ascii=False, indent=2)
        print(f"✅ Создан {records_file}")
    
    # Создаем справочники (пустые, будут заполняться автоматически)
    reference_files = {
        "professions.json": {"professions": DEFAULT_PROFESSIONS},
        "specializations.json": {"specializations": DEFAULT_SPECIALIZATIONS},
        "bank_titles.json": {"bank_titles": []},
        "tags.json": {"tags": []}
    }
    
    for filename, default_data in reference_files.items():
        file_path = DATA_DIR / filename
        if not file_path.exists():
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(default_data, f, ensure_ascii=False, indent=2)
            print(f"✅ Создан {file_path}")

# Статусы профессий (импортируются из roles.py)

# Инициализируем данные при импорте
initialize_data()

print("✅ Конфигурация HR Admin Panel v2.0 загружена")
print(f"📁 Данные: {DATA_DIR}")
print(f"🤖 OpenAI: {'✅' if OPENAI_API_KEY else '❌'}")
print(f"🌐 Порт: {APP_PORT}")
print(f"👥 Ролей: {len(ROLE_NAMES)}")
print(f"📊 Статусов: {len(PROFESSION_STATUSES)}")