"""
Tags Generator - ИИ генератор тегов для профессий
Создает максимум 10 умных тегов с весами 10-100%
"""

import json
import logging
import re
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime

# ИИ
from openai import AsyncOpenAI
import httpx

logger = logging.getLogger(__name__)


class TagsGenerator:
    """ИИ генератор тегов для профессий"""
    
    def __init__(self, openai_api_key: str, data_dir: Path):
        self.openai_api_key = openai_api_key
        self.data_dir = data_dir
        self.openai_client = None
        self.profession_data = {}
        
        self._initialize_openai()
        self._load_profession_data()
    
    def _initialize_openai(self):
        """Инициализация OpenAI клиента"""
        try:
            if self.openai_api_key:
                self.openai_client = AsyncOpenAI(api_key=self.openai_api_key)
                logger.info("✅ Tags Generator: OpenAI инициализирован")
            else:
                logger.warning("⚠️ Tags Generator: OpenAI API ключ не найден")
        except Exception as e:
            logger.error(f"❌ Tags Generator: Ошибка инициализации OpenAI: {e}")
    
    def _load_profession_data(self):
        """Загрузка данных о профессиях для анализа"""
        try:
            records_file = self.data_dir / "profession_records.json"
            if records_file.exists():
                with open(records_file, 'r', encoding='utf-8') as f:
                    self.profession_data = json.load(f)
            else:
                self.profession_data = {"profession_records": []}
            
            logger.info(f"✅ Tags Generator: Загружено {len(self.profession_data.get('profession_records', []))} записей")
            
        except Exception as e:
            logger.error(f"❌ Tags Generator: Ошибка загрузки данных: {e}")
            self.profession_data = {"profession_records": []}
    
    # === ГЕНЕРАЦИЯ ТЕГОВ ===
    
    async def generate_tags(self, profession_data: Dict[str, Any]) -> Dict[str, Any]:
        """Генерация тегов для профессии"""
        try:
            # Анализируем похожие профессии
            similar_analysis = self._analyze_similar_professions(profession_data)
            
            # Генерируем теги через ИИ
            if self.openai_client:
                tags = await self._ai_generate_tags(profession_data, similar_analysis)
            else:
                tags = self._manual_generate_tags(profession_data, similar_analysis)
            
            # Валидируем теги
            validated_tags = self._validate_tags(tags)
            
            # Создаем версию тегов
            tags_version = self._create_tags_version(validated_tags, "system", "ИИ генерация тегов")
            
            return {
                "success": True,
                "tags": validated_tags,
                "tags_version": tags_version,
                "similar_analysis": similar_analysis,
                "total_tags": len(validated_tags),
                "ai_confidence": tags.get("confidence", 0.8)
            }
            
        except Exception as e:
            logger.error(f"❌ Tags Generator: Ошибка генерации тегов: {e}")
            return {
                "success": False,
                "error": str(e),
                "tags": {}
            }
    
    def _analyze_similar_professions(self, profession_data: Dict[str, Any]) -> Dict[str, Any]:
        """Анализ похожих профессий в базе"""
        try:
            existing_records = self.profession_data.get("profession_records", [])
            similar_records = []
            
            current_real_name = profession_data.get('real_name', '').lower()
            current_specialization = profession_data.get('specialization', '').lower()
            
            for record in existing_records:
                # Рассматриваем только утвержденные записи
                if record.get("status") not in ["approved_by_head", "questions_generated", "active"]:
                    continue
                
                similarity = self._calculate_profession_similarity(
                    current_real_name, current_specialization,
                    record.get('real_name', '').lower(), record.get('specialization', '').lower()
                )
                
                if similarity >= 0.3:  # Минимум 30% схожести
                    similar_records.append({
                        **record,
                        'similarity_score': similarity
                    })
            
            # Сортируем по убыванию похожести
            similar_records.sort(key=lambda x: x['similarity_score'], reverse=True)
            
            return {
                "similar_records": similar_records[:3],  # Топ-3
                "found_similar": len(similar_records) > 0,
                "analysis_confidence": len(similar_records) * 0.2
            }
            
        except Exception as e:
            logger.error(f"❌ Tags Generator: Ошибка анализа похожих профессий: {e}")
            return {"similar_records": [], "found_similar": False, "analysis_confidence": 0.0}
    
    def _calculate_profession_similarity(self, name1: str, spec1: str, name2: str, spec2: str) -> float:
        """Расчет похожести профессий"""
        try:
            score = 0.0
            
            # Похожесть названий профессий (вес 70%)
            if name1 and name2:
                name_words1 = set(name1.split())
                name_words2 = set(name2.split())
                
                if name_words1 and name_words2:
                    intersection = name_words1.intersection(name_words2)
                    union = name_words1.union(name_words2)
                    name_similarity = len(intersection) / len(union)
                    score += name_similarity * 0.7
            
            # Похожесть специализаций (вес 30%)
            if spec1 and spec2:
                spec_words1 = set(spec1.split())
                spec_words2 = set(spec2.split())
                
                if spec_words1 and spec_words2:
                    intersection = spec_words1.intersection(spec_words2)
                    union = spec_words1.union(spec_words2)
                    spec_similarity = len(intersection) / len(union)
                    score += spec_similarity * 0.3
            
            return min(1.0, score)
            
        except Exception as e:
            logger.error(f"❌ Tags Generator: Ошибка расчета похожести: {e}")
            return 0.0
    
    async def _ai_generate_tags(self, profession_data: Dict[str, Any], similar_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """ИИ генерация тегов"""
        try:
            prompt = self._create_smart_tags_prompt(profession_data, similar_analysis)
            
            response = await self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "Ты эксперт по профессиям и навыкам в банковской сфере. Генерируешь точные теги с весами для проверки кандидатов."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=1200
            )
            
            # Парсим ответ ИИ
            ai_response = response.choices[0].message.content
            tags = self._parse_ai_tags_response(ai_response)
            
            return {
                "tags": tags,
                "source": "ai_generated",
                "confidence": 0.85
            }
            
        except Exception as e:
            logger.error(f"❌ Tags Generator: Ошибка ИИ генерации: {e}")
            return {"tags": {}, "source": "ai_error", "confidence": 0.0}
    
    def _create_smart_tags_prompt(self, profession_data: Dict[str, Any], similar_analysis: Dict[str, Any]) -> str:
        """Создание умного промпта для генерации тегов"""
        
        profession_name = profession_data.get('real_name', '')
        specialization = profession_data.get('specialization', '')
        department = profession_data.get('department', '')
        
        # Информация о похожих профессиях
        similar_info = ""
        if similar_analysis.get("found_similar"):
            similar_records = similar_analysis.get("similar_records", [])
            similar_info = f"ПОХОЖИЕ ПРОФЕССИИ В БАНКЕ:\n"
            for record in similar_records[:2]:
                if record.get("tags"):
                    top_tags = sorted(record["tags"].items(), key=lambda x: x[1], reverse=True)[:5]
                    similar_info += f"• {record['bank_title']}: {', '.join([f'{tag}({weight}%)' for tag, weight in top_tags])}\n"
        
        return f"""
        Сгенерируй МАКСИМУМ 10 самых важных тегов для профессии "{profession_name}" специализации "{specialization}" в банке Halyk Bank.
        
        ПРОФЕССИЯ: {profession_name}
        СПЕЦИАЛИЗАЦИЯ: {specialization}
        ДЕПАРТАМЕНТ: {department}
        
        {similar_info}
        
        ЗАДАЧА: Дай ТОЛЬКО самые критичные навыки, которые реально нужны для работы.
        
        ПРИНЦИПЫ:
        1. МАКСИМУМ 10 тегов (лучше меньше, но точнее)
        2. Только РЕАЛЬНО необходимые навыки
        3. НЕ включай очевидные вещи (типа "Email", "Microsoft Office")
        4. Фокус на специфические технические и профессиональные навыки
        5. Веса от 50% до 95% по важности
        
        ВЕСА:
        - 90-95%: Без этого навыка работать невозможно
        - 80-89%: Очень важный навык для эффективной работы  
        - 70-79%: Важный навык для качественной работы
        - 60-69%: Полезный навык для расширения возможностей
        - 50-59%: Дополнительный навык для развития
        
        ПРИМЕРЫ ДЛЯ РАЗНЫХ ПРОФЕССИЙ:
        
        Data Scientist:
        - Python: 90% (критично)
        - SQL: 85% (очень важно) 
        - Machine Learning: 88% (основа работы)
        - Statistics: 82% (необходимо)
        - Pandas: 80% (ежедневно)
        
        Software Developer:
        - Programming Language: 95% (основа)
        - Git: 90% (обязательно)
        - Algorithms: 85% (важно)
        - Testing: 75% (нужно)
        - Code Review: 70% (практика)
        
        ФОРМАТ ОТВЕТА (ТОЛЬКО JSON):
        {{
            "Python": 90,
            "SQL": 85,
            "Machine Learning": 88,
            "Statistics": 82,
            "Data Visualization": 75,
            "Git": 70,
            "Communication": 65,
            "Problem Solving": 80
        }}
        
        КРИТИЧНО ВАЖНО:
        - НЕ больше 10 тегов!
        - НЕ добавляй объяснений
        - ТОЛЬКО JSON формат
        - Только РЕАЛЬНО нужные навыки
        
        Отвечай СТРОГО в JSON формате!
        """
    
    def _parse_ai_tags_response(self, response: str) -> Dict[str, int]:
        """Парсинг ответа ИИ для извлечения тегов"""
        try:
            # Ищем JSON в ответе
            json_match = re.search(r'\{[^}]*\}', response, re.DOTALL)
            if json_match:
                tags_data = json.loads(json_match.group())
                return {k: int(v) for k, v in tags_data.items() if isinstance(v, (int, float))}
            
            # Если JSON не найден, парсим текст
            tags = {}
            lines = response.split('\n')
            
            for line in lines:
                # Ищем паттерны: "Tag": 85 или Tag: 85%
                match = re.search(r'["\']?([^"\':\n]+)["\']?\s*:\s*(\d+)', line)
                if match:
                    tag_name = match.group(1).strip()
                    weight = int(match.group(2))
                    if 50 <= weight <= 100:  # Валидный диапазон
                        tags[tag_name] = weight
            
            return tags
            
        except Exception as e:
            logger.error(f"❌ Tags Generator: Ошибка парсинга ответа ИИ: {e}")
            return {}
    
    def _manual_generate_tags(self, profession_data: Dict[str, Any], similar_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Ручная генерация тегов (fallback)"""
        tags = {}
        
        # Базовые теги по профессии
        real_name = profession_data.get('real_name', '').lower()
        
        if any(word in real_name for word in ['developer', 'разработчик', 'programmer']):
            tags.update({
                "Programming": 90,
                "Git": 85,
                "Problem Solving": 80,
                "Code Review": 75,
                "Testing": 70
            })
        elif any(word in real_name for word in ['analyst', 'аналитик', 'data']):
            tags.update({
                "Data Analysis": 90,
                "SQL": 85,
                "Excel": 80,
                "Reporting": 75,
                "Statistics": 70
            })
        elif any(word in real_name for word in ['manager', 'менеджер', 'lead']):
            tags.update({
                "Management": 85,
                "Leadership": 80,
                "Communication": 75,
                "Project Management": 70,
                "Team Building": 65
            })
        else:
            # Универсальные навыки
            tags.update({
                "Communication": 75,
                "Problem Solving": 80,
                "Critical Thinking": 70,
                "Teamwork": 65,
                "Time Management": 60
            })
        
        # Используем данные из похожих профессий
        if similar_analysis.get("found_similar"):
            similar_records = similar_analysis.get("similar_records", [])
            for record in similar_records[:1]:  # Берем только самую похожую
                existing_tags = record.get("tags", {})
                for tag, weight in existing_tags.items():
                    if tag not in tags:
                        tags[tag] = max(50, weight - 5)  # Немного снижаем вес
        
        # Ограничиваем до 10 тегов
        if len(tags) > 10:
            sorted_tags = sorted(tags.items(), key=lambda x: x[1], reverse=True)
            tags = dict(sorted_tags[:10])
        
        return {
            "tags": tags,
            "source": "manual_generated",
            "confidence": 0.6
        }
    
    def _validate_tags(self, tags_data: Dict[str, Any]) -> Dict[str, int]:
        """Валидация тегов"""
        tags = tags_data.get("tags", {})
        validated = {}
        
        for tag, weight in tags.items():
            # Проверяем диапазон 50-100 (более строгий чем у других ИИ)
            if isinstance(weight, (int, float)):
                validated_weight = max(50, min(100, int(weight)))
                validated[tag] = validated_weight
            else:
                validated[tag] = 70  # Дефолтное значение
        
        # Ограничиваем до 10 тегов
        if len(validated) > 10:
            sorted_tags = sorted(validated.items(), key=lambda x: x[1], reverse=True)
            validated = dict(sorted_tags[:10])
        
        return validated
    
    def _create_tags_version(self, tags: Dict[str, int], created_by: str, action: str) -> Dict[str, Any]:
        """Создание версии тегов"""
        return {
            "version": 1,
            "created_by": created_by,
            "timestamp": datetime.now().isoformat() + "Z",
            "action": action,
            "tags": tags.copy(),
            "total_tags": len(tags),
            "ai_confidence": 0.85 if created_by == "system" else None
        }
    
    # === АНАЛИЗ ТЕГОВ ===
    
    def analyze_generated_tags(self, tags: Dict[str, int], profession_data: Dict[str, Any]) -> Dict[str, Any]:
        """Анализ сгенерированных тегов"""
        try:
            analysis = {
                "total_tags": len(tags),
                "weight_distribution": self._analyze_weight_distribution(tags),
                "categories": self._categorize_tags(tags),
                "completeness_score": self._calculate_completeness(tags, profession_data),
                "recommendations": self._generate_tag_recommendations(tags, profession_data)
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Tags Generator: Ошибка анализа тегов: {e}")
            return {}
    
    def _analyze_weight_distribution(self, tags: Dict[str, int]) -> Dict[str, Any]:
        """Анализ распределения весов тегов"""
        if not tags:
            return {}
        
        weights = list(tags.values())
        
        return {
            "min_weight": min(weights),
            "max_weight": max(weights),
            "avg_weight": sum(weights) / len(weights),
            "high_priority": len([w for w in weights if w >= 80]),
            "medium_priority": len([w for w in weights if 65 <= w < 80]),
            "low_priority": len([w for w in weights if w < 65])
        }
    
    def _categorize_tags(self, tags: Dict[str, int]) -> Dict[str, List[str]]:
        """Категоризация тегов"""
        categories = {
            "technical": [],
            "tools": [],
            "soft_skills": [],
            "methodologies": [],
            "other": []
        }
        
        technical_keywords = ["python", "sql", "java", "javascript", "programming", "coding", "development"]
        tools_keywords = ["git", "docker", "aws", "tableau", "excel", "jira"]
        soft_keywords = ["communication", "leadership", "teamwork", "management", "problem solving"]
        methodology_keywords = ["agile", "scrum", "devops", "testing", "review"]
        
        for tag in tags.keys():
            tag_lower = tag.lower()
            
            if any(keyword in tag_lower for keyword in technical_keywords):
                categories["technical"].append(tag)
            elif any(keyword in tag_lower for keyword in tools_keywords):
                categories["tools"].append(tag)
            elif any(keyword in tag_lower for keyword in soft_keywords):
                categories["soft_skills"].append(tag)
            elif any(keyword in tag_lower for keyword in methodology_keywords):
                categories["methodologies"].append(tag)
            else:
                categories["other"].append(tag)
        
        return categories
    
    def _calculate_completeness(self, tags: Dict[str, int], profession_data: Dict[str, Any]) -> float:
        """Расчет полноты набора тегов"""
        # Простая эвристика: считаем полноту на основе количества тегов и их весов
        if not tags:
            return 0.0
        
        # Базовая оценка по количеству (оптимально 7-10 тегов)
        count_score = min(1.0, len(tags) / 8)
        
        # Оценка по распределению весов
        weights = list(tags.values())
        high_weight_count = len([w for w in weights if w >= 80])
        weight_score = min(1.0, high_weight_count / 3)  # Оптимально 3+ высокоприоритетных тега
        
        return (count_score + weight_score) / 2
    
    def _generate_tag_recommendations(self, tags: Dict[str, int], profession_data: Dict[str, Any]) -> List[str]:
        """Генерация рекомендаций по тегам"""
        recommendations = []
        
        if len(tags) < 5:
            recommendations.append("Рекомендуется добавить больше тегов (минимум 5-7)")
        elif len(tags) > 10:
            recommendations.append("Слишком много тегов, рекомендуется оставить самые важные")
        
        high_weight_tags = [tag for tag, weight in tags.items() if weight >= 90]
        if len(high_weight_tags) > 5:
            recommendations.append("Слишком много критично важных тегов (90%+)")
        
        if not any(weight >= 80 for weight in tags.values()):
            recommendations.append("Добавьте хотя бы один высокоприоритетный тег (80%+)")
        
        return recommendations

# Экспорт класса
__all__ = ['TagsGenerator']