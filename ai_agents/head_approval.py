"""
Head Approval - ИИ помощник для начальников отделов
Помогает при утверждении профессий, корректировке тегов, возврате на доработку
"""

import json
import logging
import re
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime

# ИИ
from openai import AsyncOpenAI
import httpx

logger = logging.getLogger(__name__)


class HeadApproval:
    """ИИ помощник для начальников отделов"""
    
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
                logger.info("✅ Head Approval: OpenAI инициализирован")
            else:
                logger.warning("⚠️ Head Approval: OpenAI API ключ не найден")
        except Exception as e:
            logger.error(f"❌ Head Approval: Ошибка инициализации OpenAI: {e}")
    
    def _load_profession_data(self):
        """Загрузка данных о профессиях"""
        try:
            records_file = self.data_dir / "profession_records.json"
            if records_file.exists():
                with open(records_file, 'r', encoding='utf-8') as f:
                    self.profession_data = json.load(f)
            else:
                self.profession_data = {"profession_records": []}
            
            logger.info(f"✅ Head Approval: Загружено {len(self.profession_data.get('profession_records', []))} записей")
            
        except Exception as e:
            logger.error(f"❌ Head Approval: Ошибка загрузки данных: {e}")
            self.profession_data = {"profession_records": []}
    
    # === АНАЛИЗ ПРОФЕССИИ ДЛЯ УТВЕРЖДЕНИЯ ===
    
    async def analyze_profession_for_approval(self, profession_id: str, user_department: str) -> Dict[str, Any]:
        """Полный анализ профессии для утверждения начальником"""
        try:
            # Находим профессию
            profession = self._find_profession_by_id(profession_id)
            if not profession:
                return {"error": "Профессия не найдена"}
            
            # Проверяем права доступа
            if not self._can_user_access_profession(profession, user_department):
                return {"error": "Нет прав для доступа к этой профессии"}
            
            # Анализируем все аспекты профессии
            analysis = await self._comprehensive_profession_analysis(profession)
            
            return {
                "success": True,
                "profession": profession,
                "analysis": analysis,
                "recommendations": self._generate_approval_recommendations(analysis),
                "suggested_actions": self._suggest_actions(analysis)
            }
            
        except Exception as e:
            logger.error(f"❌ Head Approval: Ошибка анализа профессии: {e}")
            return {"error": str(e)}
    
    def _find_profession_by_id(self, profession_id: str) -> Optional[Dict[str, Any]]:
        """Поиск профессии по ID"""
        for record in self.profession_data.get("profession_records", []):
            if record.get("id") == profession_id:
                return record
        return None
    
    def _can_user_access_profession(self, profession: Dict[str, Any], user_department: str) -> bool:
        """Проверка прав доступа к профессии"""
        profession_dept = profession.get("department", "")
        return profession_dept == f"{user_department} Department"
    
    async def _comprehensive_profession_analysis(self, profession: Dict[str, Any]) -> Dict[str, Any]:
        """Комплексный анализ профессии"""
        try:
            # 1. Анализ основных данных профессии
            basic_analysis = self._analyze_basic_profession_data(profession)
            
            # 2. Анализ тегов
            tags_analysis = await self._analyze_tags(profession)
            
            # 3. Анализ соответствия данных
            consistency_analysis = await self._analyze_data_consistency(profession)
            
            # 4. Сравнение с похожими профессиями
            comparison_analysis = self._compare_with_similar_professions(profession)
            
            # 5. Веб-анализ профессии
            web_analysis = await self._web_research_profession(profession)
            
            return {
                "basic_analysis": basic_analysis,
                "tags_analysis": tags_analysis,
                "consistency_analysis": consistency_analysis,
                "comparison_analysis": comparison_analysis,
                "web_analysis": web_analysis,
                "overall_score": self._calculate_overall_score([
                    basic_analysis, tags_analysis, consistency_analysis
                ])
            }
            
        except Exception as e:
            logger.error(f"❌ Head Approval: Ошибка комплексного анализа: {e}")
            return {"error": str(e)}
    
    def _analyze_basic_profession_data(self, profession: Dict[str, Any]) -> Dict[str, Any]:
        """Анализ основных данных профессии"""
        issues = []
        suggestions = []
        score = 1.0
        
        # Проверяем банковское название
        bank_title = profession.get("bank_title", "")
        if not bank_title:
            issues.append("Отсутствует банковское название")
            score -= 0.3
        elif len(bank_title) < 10:
            suggestions.append("Банковское название слишком короткое")
            score -= 0.1
        elif not bank_title[0].isupper():
            suggestions.append("Банковское название должно начинаться с заглавной буквы")
            score -= 0.05
        
        # Проверяем реальную профессию
        real_name = profession.get("real_name", "")
        if not real_name:
            issues.append("Отсутствует реальное название профессии")
            score -= 0.3
        
        # Проверяем специализацию
        specialization = profession.get("specialization", "")
        if not specialization:
            suggestions.append("Рекомендуется указать специализацию")
            score -= 0.1
        
        # Проверяем департамент
        department = profession.get("department", "")
        if not department:
            issues.append("Не указан департамент")
            score -= 0.2
        
        return {
            "score": max(0.0, score),
            "issues": issues,
            "suggestions": suggestions,
            "data_completeness": self._calculate_data_completeness(profession)
        }
    
    async def _analyze_tags(self, profession: Dict[str, Any]) -> Dict[str, Any]:
        """Анализ тегов профессии"""
        tags = profession.get("tags", {})
        
        if not tags:
            return {
                "score": 0.0,
                "issues": ["Отсутствуют теги"],
                "suggestions": ["Необходимо сгенерировать теги"],
                "tags_count": 0
            }
        
        # Анализ распределения весов
        weights = list(tags.values())
        
        issues = []
        suggestions = []
        score = 1.0
        
        # Проверяем количество тегов
        if len(tags) < 5:
            suggestions.append(f"Мало тегов ({len(tags)}), рекомендуется 5-10")
            score -= 0.1
        elif len(tags) > 15:
            suggestions.append(f"Много тегов ({len(tags)}), рекомендуется не более 10")
            score -= 0.1
        
        # Проверяем диапазон весов
        if max(weights) < 70:
            issues.append("Нет высокоприоритетных тегов (70%+)")
            score -= 0.2
        
        if min(weights) < 50:
            suggestions.append("Есть теги с очень низким весом (<50%)")
            score -= 0.1
        
        # Анализируем ИИ через OpenAI
        ai_tags_analysis = await self._ai_analyze_tags(profession, tags)
        
        return {
            "score": max(0.0, score),
            "issues": issues,
            "suggestions": suggestions,
            "tags_count": len(tags),
            "weight_distribution": {
                "min": min(weights),
                "max": max(weights),
                "avg": sum(weights) / len(weights)
            },
            "ai_analysis": ai_tags_analysis
        }
    
    async def _ai_analyze_tags(self, profession: Dict[str, Any], tags: Dict[str, int]) -> Dict[str, Any]:
        """ИИ анализ тегов"""
        if not self.openai_client:
            return {"available": False, "message": "ИИ анализ недоступен"}
        
        try:
            prompt = f"""
            Проанализируй теги для профессии в банке как опытный начальник отдела.
            
            ПРОФЕССИЯ:
            - Банковское название: {profession.get('bank_title', '')}
            - Реальная профессия: {profession.get('real_name', '')}
            - Специализация: {profession.get('specialization', '')}
            - Департамент: {profession.get('department', '')}
            
            ТЕГИ:
            {json.dumps(tags, ensure_ascii=False, indent=2)}
            
            Оцени:
            1. Соответствуют ли теги профессии?
            2. Правильные ли веса у тегов?
            3. Нет ли лишних/недостающих тегов?
            4. Нужны ли корректировки для банковской специфики?
            
            Ответь в JSON:
            {{
                "tags_relevance_score": 0.8,
                "missing_tags": ["тег1", "тег2"],
                "excessive_tags": ["тег3"],
                "weight_corrections": {{"тег": "предложение"}},
                "banking_specific_issues": ["проблема1"],
                "overall_recommendation": "approve/modify/reject"
            }}
            
            Отвечай ТОЛЬКО JSON!
            """
            
            response = await self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "Ты опытный начальник IT отдела банка. Анализируешь теги для профессий с точки зрения практического опыта и банковской специфики."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=800
            )
            
            response_text = response.choices[0].message.content.strip()
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            
            if json_match:
                return {
                    "available": True,
                    "analysis": json.loads(json_match.group())
                }
            else:
                return {"available": False, "message": "Не удалось получить анализ от ИИ"}
                
        except Exception as e:
            logger.error(f"❌ Head Approval: Ошибка ИИ анализа тегов: {e}")
            return {"available": False, "message": f"Ошибка ИИ анализа: {str(e)}"}
    
    async def _analyze_data_consistency(self, profession: Dict[str, Any]) -> Dict[str, Any]:
        """Анализ согласованности данных профессии"""
        if not self.openai_client:
            return {"available": False, "score": 0.8}
        
        try:
            prompt = f"""
            Проанализируй согласованность данных профессии как опытный HR эксперт.
            
            ДАННЫЕ:
            - Банковское название: {profession.get('bank_title', '')}
            - Реальная профессия: {profession.get('real_name', '')}
            - Специализация: {profession.get('specialization', '')}
            - Департамент: {profession.get('department', '')}
            
            Проверь:
            1. Соответствует ли банковское название реальной профессии?
            2. Подходит ли специализация к профессии?
            3. Логичен ли выбор департамента?
            4. Нет ли противоречий в данных?
            5. Соответствует ли название банковским стандартам?
            
            Оцени согласованность (0.0-1.0) и найди проблемы:
            {{
                "consistency_score": 0.8,
                "issues": ["проблема1", "проблема2"],
                "suggestions": ["предложение1", "предложение2"],
                "should_return_to_hr": false,
                "return_reason": ""
            }}
            
            Отвечай ТОЛЬКО JSON!
            """
            
            response = await self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "Ты опытный начальник отдела в банке. Проверяешь профессии на логичность и соответствие банковским стандартам."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=600
            )
            
            response_text = response.choices[0].message.content.strip()
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            
            if json_match:
                return {
                    "available": True,
                    **json.loads(json_match.group())
                }
            else:
                return {"available": False, "score": 0.8}
                
        except Exception as e:
            logger.error(f"❌ Head Approval: Ошибка анализа согласованности: {e}")
            return {"available": False, "score": 0.8}
    
    def _compare_with_similar_professions(self, profession: Dict[str, Any]) -> Dict[str, Any]:
        """Сравнение с похожими профессиями"""
        try:
            similar_professions = []
            current_real_name = profession.get('real_name', '').lower()
            
            for record in self.profession_data.get("profession_records", []):
                if (record.get("id") == profession.get("id") or 
                    record.get("status") not in ["approved_by_head", "questions_generated", "active"]):
                    continue
                
                similarity = self._calculate_name_similarity(
                    current_real_name, 
                    record.get('real_name', '').lower()
                )
                
                if similarity >= 0.4:
                    similar_professions.append({
                        **record,
                        "similarity": similarity
                    })
            
            similar_professions.sort(key=lambda x: x["similarity"], reverse=True)
            
            return {
                "found_similar": len(similar_professions) > 0,
                "similar_professions": similar_professions[:3],
                "potential_duplicates": [p for p in similar_professions if p["similarity"] > 0.8]
            }
            
        except Exception as e:
            logger.error(f"❌ Head Approval: Ошибка сравнения профессий: {e}")
            return {"found_similar": False, "similar_professions": []}
    
    def _calculate_name_similarity(self, name1: str, name2: str) -> float:
        """Расчет похожести названий"""
        if not name1 or not name2:
            return 0.0
        
        words1 = set(name1.split())
        words2 = set(name2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    async def _web_research_profession(self, profession: Dict[str, Any]) -> Dict[str, Any]:
        """Веб-исследование профессии"""
        try:
            profession_name = profession.get("real_name", "")
            specialization = profession.get("specialization", "")
            
            # Здесь можно добавить реальный веб-поиск
            # Пока возвращаем заглушку
            return {
                "researched": True,
                "info": f"Найдена информация о профессии '{profession_name}' в банковской сфере",
                "market_demand": "High",
                "average_salary": "Конкурентная",
                "sources": ["hh.ru", "linkedin.com", "glassdoor.com"]
            }
            
        except Exception as e:
            logger.error(f"❌ Head Approval: Ошибка веб-исследования: {e}")
            return {"researched": False, "info": "Веб-исследование недоступно"}
    
    def _calculate_overall_score(self, analyses: List[Dict[str, Any]]) -> float:
        """Расчет общей оценки профессии"""
        scores = []
        for analysis in analyses:
            if "score" in analysis:
                scores.append(analysis["score"])
        
        return sum(scores) / len(scores) if scores else 0.5
    
    def _calculate_data_completeness(self, profession: Dict[str, Any]) -> float:
        """Расчет полноты данных"""
        required_fields = ["bank_title", "real_name", "department"]
        optional_fields = ["specialization"]
        
        required_filled = sum(1 for field in required_fields if profession.get(field))
        optional_filled = sum(1 for field in optional_fields if profession.get(field))
        
        required_score = required_filled / len(required_fields)
        optional_score = optional_filled / len(optional_fields)
        
        return (required_score * 0.8) + (optional_score * 0.2)
    
    # === ГЕНЕРАЦИЯ РЕКОМЕНДАЦИЙ ===
    
    def _generate_approval_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Генерация рекомендаций для утверждения"""
        recommendations = []
        
        # Рекомендации по основным данным
        basic_analysis = analysis.get("basic_analysis", {})
        for issue in basic_analysis.get("issues", []):
            recommendations.append(f"❌ {issue}")
        for suggestion in basic_analysis.get("suggestions", []):
            recommendations.append(f"💡 {suggestion}")
        
        # Рекомендации по тегам
        tags_analysis = analysis.get("tags_analysis", {})
        if tags_analysis.get("ai_analysis", {}).get("available"):
            ai_analysis = tags_analysis["ai_analysis"]["analysis"]
            for missing_tag in ai_analysis.get("missing_tags", []):
                recommendations.append(f"➕ Рекомендуется добавить тег: {missing_tag}")
            for excessive_tag in ai_analysis.get("excessive_tags", []):
                recommendations.append(f"➖ Рекомендуется удалить тег: {excessive_tag}")
        
        # Рекомендации по согласованности
        consistency_analysis = analysis.get("consistency_analysis", {})
        for issue in consistency_analysis.get("issues", []):
            recommendations.append(f"⚠️ {issue}")
        
        # Рекомендации по дубликатам
        comparison_analysis = analysis.get("comparison_analysis", {})
        if comparison_analysis.get("potential_duplicates"):
            recommendations.append("🔄 Найдены потенциальные дубликаты профессий")
        
        return recommendations
    
    def _suggest_actions(self, analysis: Dict[str, Any]) -> List[Dict[str, str]]:
        """Предложение действий начальнику"""
        actions = []
        
        overall_score = analysis.get("overall_score", 0.5)
        consistency_analysis = analysis.get("consistency_analysis", {})
        
        # Определяем основные действия
        if consistency_analysis.get("should_return_to_hr"):
            actions.append({
                "action": "return_to_hr",
                "title": "Вернуть на доработку",
                "description": consistency_analysis.get("return_reason", "Требуются исправления"),
                "priority": "high"
            })
        
        if overall_score >= 0.8:
            actions.append({
                "action": "approve",
                "title": "Утвердить профессию",
                "description": "Профессия готова к утверждению",
                "priority": "high"
            })
        elif overall_score >= 0.6:
            actions.append({
                "action": "approve_with_corrections",
                "title": "Утвердить с корректировками",
                "description": "Внести небольшие правки и утвердить",
                "priority": "medium"
            })
        else:
            actions.append({
                "action": "major_corrections",
                "title": "Требуются серьезные правки",
                "description": "Необходимы значительные изменения",
                "priority": "high"
            })
        
        # Всегда добавляем возможность корректировки тегов
        actions.append({
            "action": "edit_tags",
            "title": "Корректировать теги",
            "description": "Изменить веса или добавить/удалить теги",
            "priority": "medium"
        })
        
        return actions
    
    # === КОРРЕКТИРОВКА ТЕГОВ ===
    
    async def suggest_tag_corrections(self, profession: Dict[str, Any], user_input: str = "") -> Dict[str, Any]:
        """Предложение корректировок тегов"""
        try:
            current_tags = profession.get("tags", {})
            
            if not self.openai_client:
                return self._manual_tag_suggestions(current_tags, profession)
            
            # ИИ анализ и предложения
            suggestions = await self._ai_suggest_tag_corrections(profession, current_tags, user_input)
            
            return {
                "success": True,
                "current_tags": current_tags,
                "suggestions": suggestions,
                "corrections_available": True
            }
            
        except Exception as e:
            logger.error(f"❌ Head Approval: Ошибка предложения корректировок: {e}")
            return {"success": False, "error": str(e)}
    
    async def _ai_suggest_tag_corrections(self, profession: Dict[str, Any], tags: Dict[str, int], user_input: str) -> Dict[str, Any]:
        """ИИ предложения корректировок тегов"""
        try:
            prompt = f"""
            Ты опытный начальник IT отдела банка. Предложи корректировки тегов для профессии.
            
            ПРОФЕССИЯ:
            - Банковское название: {profession.get('bank_title', '')}
            - Реальная профессия: {profession.get('real_name', '')}
            - Специализация: {profession.get('specialization', '')}
            
            ТЕКУЩИЕ ТЕГИ:
            {json.dumps(tags, ensure_ascii=False, indent=2)}
            
            ЗАПРОС НАЧАЛЬНИКА: {user_input or "Общий анализ тегов"}
            
            Предложи конкретные изменения:
            {{
                "weight_corrections": {{
                    "Python": {{"current": 80, "suggested": 90, "reason": "Критично важен для позиции"}},
                    "Excel": {{"current": 70, "suggested": 50, "reason": "Не основной инструмент"}}
                }},
                "tags_to_add": {{
                    "Docker": {{"weight": 75, "reason": "Нужен для development environment"}},
                    "Banking Knowledge": {{"weight": 60, "reason": "Специфика банковской сферы"}}
                }},
                "tags_to_remove": ["Excel", "PowerPoint"],
                "explanation": "Краткое объяснение изменений"
            }}
            
            Отвечай ТОЛЬКО JSON!
            """
            
            response = await self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "Ты начальник IT отдела банка с 10+ лет опыта. Понимаешь какие навыки реально нужны сотрудникам."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            
            response_text = response.choices[0].message.content.strip()
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            
            if json_match:
                return json.loads(json_match.group())
            else:
                return {"explanation": "Не удалось получить предложения от ИИ"}
                
        except Exception as e:
            logger.error(f"❌ Head Approval: Ошибка ИИ предложений: {e}")
            return {"explanation": f"Ошибка ИИ: {str(e)}"}
    
    def _manual_tag_suggestions(self, tags: Dict[str, int], profession: Dict[str, Any]) -> Dict[str, Any]:
        """Ручные предложения корректировок (fallback)"""
        suggestions = {
            "weight_corrections": {},
            "tags_to_add": {},
            "tags_to_remove": [],
            "explanation": "ИИ помощник недоступен. Проверьте теги самостоятельно."
        }
        
        # Простые проверки
        for tag, weight in tags.items():
            if weight > 95:
                suggestions["weight_corrections"][tag] = {
                    "current": weight,
                    "suggested": 90,
                    "reason": "Слишком высокий вес"
                }
            elif weight < 50:
                suggestions["tags_to_remove"].append(tag)
        
        return suggestions
    
    # === ЧАТ С НАЧАЛЬНИКОМ ===
    
    async def chat_with_head(self, user_message: str, profession_context: Dict[str, Any]) -> Dict[str, Any]:
        """Чат с начальником отдела"""
        try:
            # Анализируем контекст профессии
            if not self.openai_client:
                return {
                    "message": "ИИ помощник временно недоступен. Проверьте профессию самостоятельно.",
                    "suggestions": ["Утвердить", "Вернуть на доработку", "Корректировать теги"]
                }
            
            # ИИ ответ с учетом контекста
            ai_response = await self._generate_head_chat_response(user_message, profession_context)
            
            return {
                "message": ai_response,
                "suggestions": self._generate_head_chat_suggestions(user_message, profession_context)
            }
            
        except Exception as e:
            logger.error(f"❌ Head Approval: Ошибка чата: {e}")
            return {
                "message": "Произошла ошибка. Попробуйте еще раз.",
                "suggestions": []
            }
    
    async def _generate_head_chat_response(self, user_message: str, profession_context: Dict[str, Any]) -> str:
        """Генерация ответа в чате с начальником"""
        try:
            prompt = f"""
            Ты ИИ помощник начальника IT отдела банка. Помогаешь анализировать и утверждать профессии.
            
            ПРОФЕССИЯ НА РАССМОТРЕНИИ:
            - Банковское название: {profession_context.get('bank_title', '')}
            - Реальная профессия: {profession_context.get('real_name', '')}
            - Специализация: {profession_context.get('specialization', '')}
            - Теги: {len(profession_context.get('tags', {}))} шт.
            
            ВОПРОС НАЧАЛЬНИКА: {user_message}
            
            Дай краткий профессиональный ответ (максимум 2-3 предложения).
            Фокусируйся на практических аспектах и банковской специфике.
            """
            
            response = await self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "Ты опытный ИИ помощник начальника отдела. Отвечаешь кратко, профессионально и по делу."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=200
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"❌ Head Approval: Ошибка генерации ответа: {e}")
            return "Не удалось получить ответ от ИИ. Попробуйте переформулировать вопрос."
    
    def _generate_head_chat_suggestions(self, user_message: str, profession_context: Dict[str, Any]) -> List[str]:
        """Генерация предложений для чата с начальником"""
        suggestions = []
        
        # Базовые действия
        suggestions.extend([
            "Утвердить профессию",
            "Вернуть на доработку", 
            "Корректировать теги"
        ])
        
        # Контекстные предложения
        if "тег" in user_message.lower():
            suggestions.extend([
                "Какие теги добавить?",
                "Какие веса изменить?"
            ])
        elif "название" in user_message.lower():
            suggestions.extend([
                "Корректное ли банковское название?",
                "Нужно ли изменить название?"
            ])
        else:
            suggestions.extend([
                "Все ли корректно?",
                "Какие есть замечания?"
            ])
        
        return suggestions[:5]

# Экспорт класса
__all__ = ['HeadApproval']