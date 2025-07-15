"""
ИИ Агенты HR Admin Panel v2.0
Модульная система специализированных ИИ помощников
"""

from .hr_assistant import HRAssistant
from .tags_generator import TagsGenerator
from .head_approval import HeadApproval
from .questions_generator import QuestionsGenerator

# Версия модуля ИИ агентов
__version__ = "2.0.0"

# Экспортируемые классы
__all__ = [
    "HRAssistant",
    "TagsGenerator", 
    "HeadApproval",
    "QuestionsGenerator"
]

print("🤖 ИИ Агенты загружены v{__version__}")