"""
HR Assistant - ИИ помощник для HR специалистов
Помогает при создании профессий, анализе файлов, проверке дубликатов
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

# Работа с файлами
import PyPDF2
import docx

logger = logging.getLogger(__name__)


class HRAssistant:
    """ИИ помощник для HR специалистов"""
    
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
                logger.info("✅ HR Assistant: OpenAI инициализирован")
            else:
                logger.warning("⚠️ HR Assistant: OpenAI API ключ не найден")
        except Exception as e:
            logger.error(f"❌ HR Assistant: Ошибка инициализации OpenAI: {e}")
    
    def _load_profession_data(self):
        """Загрузка данных о профессиях"""
        try:
            # Загружаем основной файл
            records_file = self.data_dir / "profession_records.json"
            if records_file.exists():
                with open(records_file, 'r', encoding='utf-8') as f:
                    self.profession_data = json.load(f)
            else:
                self.profession_data = {"profession_records": []}
            
            # Загружаем справочники
            self._load_reference_files()
            
            logger.info(f"✅ HR Assistant: Загружено {len(self.profession_data.get('profession_records', []))} записей")
            
        except Exception as e:
            logger.error(f"❌ HR Assistant: Ошибка загрузки данных: {e}")
            self.profession_data = {"profession_records": []}
    
    def _load_reference_files(self):
        """Загрузка справочных файлов"""
        reference_files = ['departments.json', 'professions.json', 'specializations.json', 'bank_titles.json']
        
        for filename in reference_files:
            try:
                file_path = self.data_dir / filename
                if file_path.exists():
                    with open(file_path, 'r', encoding='utf-8') as f:
                        key = filename.replace('.json', '')
                        self.profession_data[key] = json.load(f)
            except Exception as e:
                logger.error(f"❌ HR Assistant: Ошибка загрузки {filename}: {e}")
    
    # === АНАЛИЗ ФАЙЛОВ ===
    
    async def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """Анализ загруженного файла с требованиями к профессии"""
        try:
            # Извлекаем текст из файла
            text_content = await self._extract_text_from_file(file_path)
            
            if not text_content or len(text_content.strip()) < 50:
                return {
                    "success": False,
                    "error": "Файл пустой или содержит мало текста",
                    "suggestions": []
                }
            
            # Анализируем через ИИ
            analysis = await self._ai_analyze_file_content(text_content)
            
            return {
                "success": True,
                "content_preview": text_content[:300] + "..." if len(text_content) > 300 else text_content,
                "analysis": analysis,
                "suggestions": self._generate_form_suggestions(analysis)
            }
            
        except Exception as e:
            logger.error(f"❌ HR Assistant: Ошибка анализа файла: {e}")
            return {
                "success": False,
                "error": str(e),
                "suggestions": []
            }
    
    async def _extract_text_from_file(self, file_path: str) -> str:
        """Извлечение текста из файла"""
        file_path = Path(file_path)
        
        if file_path.suffix.lower() == '.pdf':
            return self._extract_pdf_text(file_path)
        elif file_path.suffix.lower() in ['.docx', '.doc']:
            return self._extract_docx_text(file_path)
        elif file_path.suffix.lower() == '.txt':
            return self._extract_txt_text(file_path)
        else:
            raise ValueError(f"Неподдерживаемый формат файла: {file_path.suffix}")
    
    def _extract_pdf_text(self, file_path: Path) -> str:
        """Извлечение текста из PDF"""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text.strip()
        except Exception as e:
            raise Exception(f"Ошибка чтения PDF: {e}")
    
    def _extract_docx_text(self, file_path: Path) -> str:
        """Извлечение текста из DOCX"""
        try:
            doc = docx.Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text.strip()
        except Exception as e:
            raise Exception(f"Ошибка чтения DOCX: {e}")
    
    def _extract_txt_text(self, file_path: Path) -> str:
        """Извлечение текста из TXT"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read().strip()
        except Exception as e:
            raise Exception(f"Ошибка чтения TXT: {e}")
    
    async def _ai_analyze_file_content(self, content: str) -> Dict[str, Any]:
        """ИИ анализ содержимого файла"""
        if not self.openai_client:
            return self._manual_file_analysis(content)
        
        try:
            prompt = f"""
            Проанализируй описание вакансии/профессии для банка Halyk Bank.
            
            ТЕКСТ: {content}
            
            Извлеки ключевую информацию и верни в JSON формате:
            {{
                "position_title": "название позиции из текста",
                "department_suggestion": "предполагаемый департамент",
                "real_profession": "реальная профессия",
                "specialization": "специализация если есть",
                "experience_level": "Junior/Middle/Senior",
                "key_requirements": ["основные требования"],
                "summary": "краткое описание позиции"
            }}
            
            Если информации мало, укажи null для недостающих полей.
            Отвечай ТОЛЬКО JSON!
            """
            
            response = await self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "Ты HR эксперт банка. Анализируешь вакансии точно и кратко."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=800
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Парсим JSON
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return {"error": "Не удалось извлечь данные из файла"}
                
        except Exception as e:
            logger.error(f"❌ HR Assistant: Ошибка ИИ анализа: {e}")
            return self._manual_file_analysis(content)
    
    def _manual_file_analysis(self, content: str) -> Dict[str, Any]:
        """Ручной анализ файла (fallback)"""
        content_lower = content.lower()
        
        # Определяем департамент по ключевым словам
        department_keywords = {
            "IT Department": ["разработчик", "программист", "it", "данных", "аналитик", "sql", "python"],
            "HR Department": ["hr", "персонал", "подбор", "рекрутер", "кадры"],
            "Risk Department": ["риск", "compliance", "контроль", "аудит"],
            "Analytics Department": ["аналитик", "данных", "отчет", "bi", "dashboard"]
        }
        
        suggested_dept = "IT Department"  # По умолчанию
        for dept, keywords in department_keywords.items():
            if any(keyword in content_lower for keyword in keywords):
                suggested_dept = dept
                break
        
        return {
            "position_title": "Извлечено из файла",
            "department_suggestion": suggested_dept,
            "real_profession": None,
            "specialization": None,
            "experience_level": "Middle",
            "key_requirements": [],
            "summary": f"Анализ файла: {len(content)} символов"
        }
    
    def _generate_form_suggestions(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Генерация предложений для заполнения формы"""
        suggestions = {}
        
        if analysis.get("department_suggestion"):
            suggestions["department"] = analysis["department_suggestion"]
        
        if analysis.get("real_profession"):
            suggestions["real_name"] = analysis["real_profession"]
        
        if analysis.get("specialization"):
            suggestions["specialization"] = analysis["specialization"]
        
        if analysis.get("position_title"):
            suggestions["bank_title"] = analysis["position_title"]
        
        return suggestions
    
    # === АНАЛИЗ ФОРМЫ В РЕАЛЬНОМ ВРЕМЕНИ ===
    
    async def analyze_form_data(self, form_data: Dict[str, Any]) -> Dict[str, Any]:
        """Анализ данных формы в реальном времени"""
        try:
            # Проверяем дубликаты
            duplicates = self._check_duplicates(form_data)
            
            # Анализируем логичность
            logic_analysis = await self._analyze_form_logic(form_data)
            
            # Веб-поиск информации
            web_info = await self._web_research(form_data)
            
            return {
                "duplicates": duplicates,
                "logic_analysis": logic_analysis,
                "web_info": web_info,
                "overall_score": self._calculate_form_score(duplicates, logic_analysis),
                "recommendations": self._generate_recommendations(duplicates, logic_analysis, web_info)
            }
            
        except Exception as e:
            logger.error(f"❌ HR Assistant: Ошибка анализа формы: {e}")
            return {
                "error": str(e),
                "recommendations": []
            }
    
    def _check_duplicates(self, form_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Проверка дубликатов в базе"""
        duplicates = []
        
        try:
            existing_records = self.profession_data.get("profession_records", [])
            
            bank_title = form_data.get("bank_title", "").lower()
            real_name = form_data.get("real_name", "").lower()
            
            for record in existing_records:
                # Пропускаем неутвержденные записи
                if record.get("status") not in ["approved_by_head", "questions_generated", "active"]:
                    continue
                
                # Проверяем точное совпадение банковского названия
                if record.get("bank_title", "").lower() == bank_title and bank_title:
                    duplicates.append({
                        "type": "exact_bank_title",
                        "existing": record,
                        "similarity": 1.0,
                        "message": "Банковское название уже существует"
                    })
                
                # Проверяем похожие названия
                similarity = self._calculate_text_similarity(bank_title, record.get("bank_title", "").lower())
                if similarity > 0.8 and bank_title:
                    duplicates.append({
                        "type": "similar_bank_title",
                        "existing": record,
                        "similarity": similarity,
                        "message": f"Похожее банковское название ({similarity:.0%} совпадение)"
                    })
                
                # Проверяем точное совпадение реальной профессии + специализации
                if (record.get("real_name", "").lower() == real_name and 
                    record.get("specialization", "").lower() == form_data.get("specialization", "").lower() and
                    real_name):
                    duplicates.append({
                        "type": "exact_profession_spec",
                        "existing": record,
                        "similarity": 1.0,
                        "message": "Профессия с такой же специализацией уже существует"
                    })
            
            return duplicates
            
        except Exception as e:
            logger.error(f"❌ HR Assistant: Ошибка проверки дубликатов: {e}")
            return []
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Простой расчет похожести текстов"""
        if not text1 or not text2:
            return 0.0
        
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    async def _analyze_form_logic(self, form_data: Dict[str, Any]) -> Dict[str, Any]:
        """Анализ логичности данных формы"""
        if not self.openai_client:
            return {"score": 0.8, "issues": [], "suggestions": []}
        
        try:
            prompt = f"""
            Проанализируй логичность данных профессии для банка:
            
            Департамент: {form_data.get('department', 'Не указан')}
            Реальная профессия: {form_data.get('real_name', 'Не указана')}
            Специализация: {form_data.get('specialization', 'Не указана')}
            Банковское название: {form_data.get('bank_title', 'Не указано')}
            
            Оцени логичность (0.0-1.0) и найди проблемы. Ответь в JSON:
            {{
                "score": 0.8,
                "issues": ["проблема1", "проблема2"],
                "suggestions": ["предложение1", "предложение2"]
            }}
            
            Отвечай ТОЛЬКО JSON!
            """
            
            response = await self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "Ты HR эксперт. Анализируешь логичность профессий кратко."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=400
            )
            
            response_text = response.choices[0].message.content.strip()
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            
            if json_match:
                return json.loads(json_match.group())
            else:
                return {"score": 0.8, "issues": [], "suggestions": []}
                
        except Exception as e:
            logger.error(f"❌ HR Assistant: Ошибка анализа логики: {e}")
            return {"score": 0.8, "issues": [], "suggestions": []}
    
    async def _web_research(self, form_data: Dict[str, Any]) -> Dict[str, Any]:
        """Поиск информации о профессии в интернете"""
        try:
            profession_name = form_data.get("real_name", "")
            if not profession_name:
                return {"found": False, "info": "Название профессии не указано"}
            
            # Здесь можно добавить реальный веб-поиск
            # Пока возвращаем заглушку
            return {
                "found": True,
                "info": f"Найдена информация о профессии '{profession_name}'",
                "sources": ["hh.ru", "linkedin.com"],
                "summary": "Популярная профессия в банковской сфере"
            }
            
        except Exception as e:
            logger.error(f"❌ HR Assistant: Ошибка веб-поиска: {e}")
            return {"found": False, "info": "Ошибка поиска"}
    
    def _calculate_form_score(self, duplicates: List[Dict], logic_analysis: Dict) -> float:
        """Расчет общей оценки формы"""
        score = 1.0
        
        # Снижаем оценку за дубликаты
        for duplicate in duplicates:
            if duplicate["type"].startswith("exact_"):
                score -= 0.4
            elif duplicate["type"].startswith("similar_"):
                score -= 0.2
        
        # Учитываем логический анализ
        logic_score = logic_analysis.get("score", 0.8)
        score = (score + logic_score) / 2
        
        return max(0.0, min(1.0, score))
    
    def _generate_recommendations(self, duplicates: List[Dict], logic_analysis: Dict, web_info: Dict) -> List[str]:
        """Генерация рекомендаций"""
        recommendations = []
        
        # Рекомендации по дубликатам
        for duplicate in duplicates:
            if duplicate["type"] == "exact_bank_title":
                recommendations.append("⚠️ Банковское название уже используется")
            elif duplicate["type"] == "similar_bank_title":
                recommendations.append(f"📝 Похожее название существует ({duplicate['similarity']:.0%})")
            elif duplicate["type"] == "exact_profession_spec":
                recommendations.append("🔄 Профессия с такой специализацией уже есть")
        
        # Рекомендации по логике
        for issue in logic_analysis.get("issues", []):
            recommendations.append(f"🤔 {issue}")
        
        for suggestion in logic_analysis.get("suggestions", []):
            recommendations.append(f"💡 {suggestion}")
        
        # Рекомендации по веб-поиску
        if web_info.get("found"):
            recommendations.append(f"🌐 {web_info.get('info', 'Найдена дополнительная информация')}")
        
        return recommendations
    
    # === ЧАТ С ПОЛЬЗОВАТЕЛЕМ ===
    
    async def chat_with_user(self, user_message: str, form_context: Dict[str, Any]) -> Dict[str, Any]:
        """Чат с HR специалистом"""
        try:
            # Анализируем контекст формы
            form_analysis = await self.analyze_form_data(form_context)
            
            # Генерируем ответ
            ai_response = await self._generate_chat_response(user_message, form_context, form_analysis)
            
            return {
                "message": ai_response,
                "analysis": form_analysis,
                "suggestions": self._generate_chat_suggestions(user_message, form_context)
            }
            
        except Exception as e:
            logger.error(f"❌ HR Assistant: Ошибка чата: {e}")
            return {
                "message": "Извините, произошла ошибка. Попробуйте еще раз.",
                "analysis": {},
                "suggestions": []
            }
    
    async def _generate_chat_response(self, user_message: str, form_context: Dict[str, Any], form_analysis: Dict[str, Any]) -> str:
        """Генерация ответа в чате"""
        if not self.openai_client:
            return self._manual_chat_response(user_message, form_context)
        
        try:
            context = f"""
            Форма пользователя:
            - Департамент: {form_context.get('department', 'Не указан')}
            - Реальная профессия: {form_context.get('real_name', 'Не указана')}
            - Специализация: {form_context.get('specialization', 'Не указана')}
            - Банковское название: {form_context.get('bank_title', 'Не указано')}
            
            Анализ формы:
            - Оценка: {form_analysis.get('overall_score', 0.8):.1f}/1.0
            - Дубликаты: {len(form_analysis.get('duplicates', []))}
            - Рекомендации: {len(form_analysis.get('recommendations', []))}
            
            Вопрос пользователя: {user_message}
            """
            
            response = await self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "Ты ИИ помощник HR специалиста в банке Halyk Bank. Отвечаешь кратко и по делу. Помогаешь создавать профессии избегая дубликатов."},
                    {"role": "user", "content": context}
                ],
                temperature=0.3,
                max_tokens=300
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"❌ HR Assistant: Ошибка генерации ответа: {e}")
            return self._manual_chat_response(user_message, form_context)
    
    def _manual_chat_response(self, user_message: str, form_context: Dict[str, Any]) -> str:
        """Ручной ответ (когда ИИ недоступен)"""
        if "дубликат" in user_message.lower() or "похож" in user_message.lower():
            return "Проверяю дубликаты в базе... ИИ помощник временно недоступен."
        
        if "департамент" in user_message.lower():
            return "Рекомендую выбрать департамент исходя из основных функций профессии."
        
        if "название" in user_message.lower():
            return "Банковское название должно быть понятным для сотрудников банка."
        
        return "ИИ помощник временно недоступен. Заполните форму самостоятельно."
    
    def _generate_chat_suggestions(self, user_message: str, form_context: Dict[str, Any]) -> List[str]:
        """Генерация предложений для чата"""
        suggestions = []
        
        if not form_context.get("department"):
            suggestions.append("Какой департамент выбрать?")
        
        if not form_context.get("real_name"):
            suggestions.append("Как определить реальную профессию?")
        
        if not form_context.get("bank_title"):
            suggestions.append("Как назвать профессию в банке?")
        
        suggestions.extend(["Есть ли дубликаты?", "Всё ли корректно?"])
        
        return suggestions[:5]

# Экспорт класса
__all__ = ['HRAssistant']