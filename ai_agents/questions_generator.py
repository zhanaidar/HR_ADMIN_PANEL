"""
Questions Generator - ИИ генератор вопросов для тестирования
Создает 30-50 умных вопросов на каждый тег в 3 уровнях сложности
"""

import json
import logging
import re
import uuid
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime

# ИИ
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


class QuestionsGenerator:
    """ИИ генератор вопросов для тестирования навыков"""
    
    def __init__(self, openai_api_key: str, data_dir: Path):
        self.openai_api_key = openai_api_key
        self.data_dir = data_dir
        self.openai_client = None
        
        self._initialize_openai()
    
    def _initialize_openai(self):
        """Инициализация OpenAI клиента"""
        try:
            if self.openai_api_key:
                self.openai_client = AsyncOpenAI(api_key=self.openai_api_key)
                logger.info("✅ Questions Generator: OpenAI инициализирован")
            else:
                logger.warning("⚠️ Questions Generator: OpenAI API ключ не найден")
        except Exception as e:
            logger.error(f"❌ Questions Generator: Ошибка инициализации OpenAI: {e}")
    
    # === ГЕНЕРАЦИЯ ВОПРОСОВ ДЛЯ ПРОФЕССИИ ===
    
    async def generate_questions_for_profession(self, profession: Dict[str, Any]) -> Dict[str, Any]:
        """Генерация вопросов для всех тегов профессии"""
        try:
            tags = profession.get("tags", {})
            if not tags:
                return {
                    "success": False,
                    "error": "У профессии нет тегов для генерации вопросов"
                }
            
            profession_context = {
                "bank_title": profession.get("bank_title", ""),
                "real_name": profession.get("real_name", ""),
                "specialization": profession.get("specialization", ""),
                "department": profession.get("department", "")
            }
            
            all_questions = []
            generation_stats = {
                "total_tags": len(tags),
                "successful_tags": 0,
                "failed_tags": 0,
                "total_questions": 0,
                "questions_by_difficulty": {"easy": 0, "medium": 0, "hard": 0}
            }
            
            # Генерируем вопросы для каждого тега
            for tag, weight in tags.items():
                logger.info(f"🎯 Генерация вопросов для тега: {tag} ({weight}%)")
                
                tag_questions = await self._generate_questions_for_tag(
                    tag, weight, profession_context
                )
                
                if tag_questions.get("success"):
                    all_questions.extend(tag_questions["questions"])
                    generation_stats["successful_tags"] += 1
                    generation_stats["total_questions"] += len(tag_questions["questions"])
                    
                    # Подсчет по сложности
                    for question in tag_questions["questions"]:
                        difficulty = question.get("difficulty", "medium")
                        generation_stats["questions_by_difficulty"][difficulty] += 1
                else:
                    generation_stats["failed_tags"] += 1
                    logger.error(f"❌ Не удалось сгенерировать вопросы для тега: {tag}")
            
            return {
                "success": True,
                "questions": all_questions,
                "stats": generation_stats,
                "generated_at": datetime.now().isoformat() + "Z"
            }
            
        except Exception as e:
            logger.error(f"❌ Questions Generator: Ошибка генерации вопросов: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _generate_questions_for_tag(self, tag: str, weight: int, profession_context: Dict[str, Any]) -> Dict[str, Any]:
        """Генерация 30-50 вопросов для одного тега в 3 уровнях сложности"""
        try:
            if not self.openai_client:
                return await self._manual_generate_questions_for_tag(tag, weight, profession_context)
            
            # Определяем количество вопросов по уровням на основе веса тега
            questions_distribution = self._calculate_questions_distribution(weight)
            
            all_questions = []
            
            # Генерируем вопросы для каждого уровня сложности
            for difficulty, count in questions_distribution.items():
                difficulty_questions = await self._generate_difficulty_level_questions(
                    tag, difficulty, count, profession_context
                )
                
                if difficulty_questions:
                    all_questions.extend(difficulty_questions)
            
            # Добавляем метаданные к вопросам
            for question in all_questions:
                question["tag"] = tag
                question["tag_weight"] = weight
                question["profession_context"] = profession_context["real_name"]
                question["id"] = str(uuid.uuid4())
                question["generated_at"] = datetime.now().isoformat() + "Z"
            
            return {
                "success": True,
                "tag": tag,
                "questions": all_questions,
                "total_questions": len(all_questions),
                "distribution": questions_distribution
            }
            
        except Exception as e:
            logger.error(f"❌ Questions Generator: Ошибка генерации для тега {tag}: {e}")
            return {
                "success": False,
                "tag": tag,
                "error": str(e)
            }
    
    def _calculate_questions_distribution(self, weight: int) -> Dict[str, int]:
        """Расчет распределения вопросов по сложности на основе веса тега"""
        # Базовое распределение: 30-50 вопросов
        if weight >= 85:
            # Критично важный тег - максимум вопросов
            return {"easy": 5, "medium": 7, "hard": 5}  # 50 вопросов
        elif weight >= 70:
            # Важный тег - много вопросов
            return {"easy": 4, "medium": 5, "hard": 4}  # 40 вопросов
        elif weight >= 55:
            # Средний тег - стандартное количество
            return {"easy": 3, "medium": 4, "hard": 3}  # 32 вопроса
        else:
            # Низкий вес - минимум вопросов
            return {"easy": 3, "medium": 4, "hard": 2}   # 25 вопросов
    
    async def _generate_difficulty_level_questions(self, tag: str, difficulty: str, count: int, profession_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Генерация вопросов определенного уровня сложности"""
        try:
            prompt = self._create_questions_prompt(tag, difficulty, count, profession_context)
            
            response = await self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": self._get_system_prompt(difficulty)},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.4,  # Немного творчества для разнообразия вопросов
                max_tokens=4000
            )
            
            response_text = response.choices[0].message.content.strip()
            questions = self._parse_questions_response(response_text, difficulty)
            
            logger.info(f"✅ Сгенерировано {len(questions)} вопросов уровня {difficulty} для тега {tag}")
            return questions
            
        except Exception as e:
            logger.error(f"❌ Ошибка генерации вопросов {difficulty} для {tag}: {e}")
            return []
    
    def _create_questions_prompt(self, tag: str, difficulty: str, count: int, profession_context: Dict[str, Any]) -> str:
        """Создание простого и эффективного промпта для генерации вопросов"""
        
        # Извлекаем данные профессии
        profession = profession_context.get('real_name', '')
        specialization = profession_context.get('specialization', '')
        
        # Получаем релевантный пример
        example = self._get_example_for_context(tag, profession, specialization, difficulty)
        
        return f"""
    Создай {count} уникальных вопросов уровня {difficulty} по тегу "{tag}" для профессии "{profession}" (специализация: {specialization}).

    ПРИМЕР СТРУКТУРЫ:
    {example['question']}
    A) {example['options'][0]} ✓
    B) {example['options'][1]}
    C) {example['options'][2]}
    D) {example['options'][3]}

    ПРАВИЛА:
    1. Все вопросы должны быть РАЗНЫЕ (не повторяться)
    2. Дистракторы ТОЛЬКО из области "{tag}" или смежных технологий
    3. НЕ используй универсальные ответы из других областей
    4. Дистракторы НЕ должны повторяться между вопросами
    5. Все варианты ответов должны выглядеть правдоподобно (нельзя угадать методом исключения)

    ФОРМАТ JSON:
    [
        {{
            "question": "Вопрос о {tag} для {profession}?",
            "options": ["Правильный ответ", "Дистрактор 1", "Дистрактор 2", "Дистрактор 3"],
            "correct_answer": "Правильный ответ",
            "explanation": "Краткое объяснение",
            "difficulty": "{difficulty}",
            "category": "Категория вопроса"
        }}
    ]
    """

    def _get_example_for_context(self, tag: str, profession: str, specialization: str, difficulty: str) -> dict:
        """Генерирует конкретный пример для контекста"""
        
        examples = {
            # Python примеры
            ("Python", "Data Scientist", "Machine Learning", "easy"): {
                "question": "Какая библиотека Python используется для работы с массивами и матрицами?",
                "options": ["NumPy", "requests", "datetime", "os"]
            },
            ("Python", "Data Scientist", "Machine Learning", "medium"): {
                "question": "Какой метод sklearn используется для разделения данных на train/test?",
                "options": ["train_test_split()", "split_data()", "divide_dataset()", "separate_data()"]
            },
            ("Python", "Data Scientist", "Computer Vision", "medium"): {
                "question": "Какая библиотека используется для обработки изображений в Python?",
                "options": ["OpenCV", "requests", "json", "datetime"]
            },
            ("Python", "Software Developer", "Backend Development", "easy"): {
                "question": "Какой фреймворк используется для создания REST API в Python?",
                "options": ["FastAPI", "pandas", "numpy", "matplotlib"]
            },
            
            # Machine Learning примеры
            ("Machine Learning", "Data Scientist", "Machine Learning", "easy"): {
                "question": "Какой алгоритм лучше всего подходит для задач классификации с линейно разделимыми данными?",
                "options": ["Логистическая регрессия", "K-means", "PCA", "DBSCAN"]
            },
            ("Machine Learning", "Data Scientist", "Machine Learning", "medium"): {
                "question": "Какая метрика используется для оценки качества модели классификации?",
                "options": ["F1-score", "MSE", "MAE", "R²"]
            },
            ("Machine Learning", "Data Scientist", "Computer Vision", "medium"): {
                "question": "Какая архитектура нейросети лучше для обработки изображений?",
                "options": ["CNN", "RNN", "LSTM", "GRU"]
            },
            
            # SQL примеры
            ("SQL", "Data Scientist", "Machine Learning", "easy"): {
                "question": "Какая команда SQL используется для выборки данных из таблицы?",
                "options": ["SELECT", "GET", "FETCH", "RETRIEVE"]
            },
            ("SQL", "Data Analyst", "Business Intelligence", "medium"): {
                "question": "Какая команда SQL используется для объединения таблиц по ключу?",
                "options": ["JOIN", "MERGE", "COMBINE", "UNITE"]
            },
            
            # Deep Learning примеры
            ("Deep Learning", "Data Scientist", "Computer Vision", "medium"): {
                "question": "Какая функция активации чаще всего используется в скрытых слоях нейросетей?",
                "options": ["ReLU", "Sigmoid", "Tanh", "Linear"]
            },
            ("Deep Learning", "Data Scientist", "Machine Learning", "hard"): {
                "question": "Какой метод используется для предотвращения переобучения в нейросетях?",
                "options": ["Dropout", "Normalization", "Standardization", "Aggregation"]
            },
            
            # Computer Vision примеры
            ("Computer Vision", "Data Scientist", "Computer Vision", "easy"): {
                "question": "Какой фильтр используется для размытия изображения?",
                "options": ["Гауссов фильтр", "Медианный фильтр", "Собел фильтр", "Лапласиан"]
            },
            ("Computer Vision", "Data Scientist", "Computer Vision", "medium"): {
                "question": "Какой алгоритм используется для детекции границ на изображении?",
                "options": ["Canny", "SIFT", "SURF", "ORB"]
            },
            
            # Data Analysis примеры
            ("Data Analysis", "Data Scientist", "Machine Learning", "easy"): {
                "question": "Какая мера центральной тенденции устойчива к выбросам?",
                "options": ["Медиана", "Среднее арифметическое", "Размах", "Дисперсия"]
            },
            ("Data Analysis", "Data Analyst", "Business Intelligence", "medium"): {
                "question": "Какой коэффициент измеряет линейную связь между переменными?",
                "options": ["Корреляция Пирсона", "Дисперсия", "Среднее отклонение", "Коэффициент вариации"]
            },
            
            # Statistics примеры
            ("Statistics", "Data Scientist", "Machine Learning", "medium"): {
                "question": "Какой тест используется для проверки нормальности распределения?",
                "options": ["Тест Шапиро-Уилка", "t-тест", "Тест Фишера", "Тест Левена"]
            },
            
            # TensorFlow примеры
            ("TensorFlow", "Data Scientist", "Deep Learning", "medium"): {
                "question": "Какой класс в TensorFlow используется для создания полносвязного слоя?",
                "options": ["Dense", "Conv2D", "LSTM", "Embedding"]
            },
            
            # OpenCV примеры
            ("OpenCV", "Data Scientist", "Computer Vision", "easy"): {
                "question": "Какая функция OpenCV используется для чтения изображения?",
                "options": ["cv2.imread()", "cv2.load()", "cv2.open()", "cv2.get()"]
            },
            
            # Git примеры (для любых IT профессий)
            ("Git", "Software Developer", "Backend Development", "easy"): {
                "question": "Какая команда Git используется для создания коммита?",
                "options": ["git commit", "git save", "git push", "git add"]
            },
            
            # Docker примеры
            ("Docker", "Software Developer", "DevOps", "medium"): {
                "question": "Какая команда Docker используется для запуска контейнера?",
                "options": ["docker run", "docker start", "docker launch", "docker execute"]
            }
        }
        
        # Ищем точное совпадение
        key = (tag, profession, specialization, difficulty)
        if key in examples:
            return examples[key]
        
        # Ищем без учета сложности
        for (t, p, s, d), example in examples.items():
            if t == tag and p == profession and s == specialization:
                return example
        
        # Ищем без специализации
        for (t, p, s, d), example in examples.items():
            if t == tag and p == profession:
                return example
        
        # Ищем только по тегу
        for (t, p, s, d), example in examples.items():
            if t == tag:
                return example
        
        # Дефолтный пример
        return {
            "question": f"Вопрос о {tag} для {profession}?",
            "options": ["Правильный ответ", "Дистрактор 1", "Дистрактор 2", "Дистрактор 3"]
        }
    
    def _get_system_prompt(self, difficulty: str) -> str:
        """Получение системного промпта в зависимости от сложности"""
        base_prompt = "Ты эксперт по техническим интервью в банке. Создаешь качественные вопросы для проверки навыков кандидатов."
        
        if difficulty == "easy":
            return base_prompt + " Фокусируйся на базовых концепциях и определениях."
        elif difficulty == "medium":
            return base_prompt + " Создавай практические вопросы о применении технологий в реальной работе."
        else:  # hard
            return base_prompt + " Генерируй сложные вопросы для экспертов, включая оптимизацию и архитектуру."
    
    def _parse_questions_response(self, response: str, difficulty: str) -> List[Dict[str, Any]]:
        """Парсинг ответа ИИ с вопросами"""
        try:
            # Ищем JSON массив в ответе
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                questions_data = json.loads(json_match.group())
                
                # Валидируем и дополняем каждый вопрос
                validated_questions = []
                for question in questions_data:
                    if self._validate_question(question):
                        question["difficulty"] = difficulty
                        validated_questions.append(question)
                
                return validated_questions
            else:
                logger.error("❌ Не найден JSON массив в ответе ИИ")
                return []
                
        except json.JSONDecodeError as e:
            logger.error(f"❌ Ошибка парсинга JSON: {e}")
            return []
        except Exception as e:
            logger.error(f"❌ Ошибка парсинга вопросов: {e}")
            return []
    
    def _validate_question(self, question: Dict[str, Any]) -> bool:
        """Валидация структуры вопроса"""
        required_fields = ["question", "options", "correct_answer", "explanation"]
        
        # Проверяем наличие всех обязательных полей
        for field in required_fields:
            if field not in question:
                logger.warning(f"⚠️ Отсутствует поле {field} в вопросе")
                return False
        
        # Проверяем что есть 4 варианта ответа
        if not isinstance(question["options"], list) or len(question["options"]) != 4:
            logger.warning("⚠️ Неправильное количество вариантов ответа")
            return False
        
        # Проверяем что правильный ответ есть среди вариантов
        if question["correct_answer"] not in question["options"]:
            logger.warning("⚠️ Правильный ответ не найден среди вариантов")
            return False
        
        # Проверяем минимальную длину вопроса
        if len(question["question"].strip()) < 10:
            logger.warning("⚠️ Слишком короткий вопрос")
            return False
        
        return True
    
    # === РУЧНАЯ ГЕНЕРАЦИЯ (FALLBACK) ===
    
    async def _manual_generate_questions_for_tag(self, tag: str, weight: int, profession_context: Dict[str, Any]) -> Dict[str, Any]:
        """Ручная генерация вопросов когда ИИ недоступен"""
        try:
            # Базовые шаблоны вопросов
            manual_questions = self._create_manual_questions_templates(tag, weight)
            
            return {
                "success": True,
                "tag": tag,
                "questions": manual_questions,
                "total_questions": len(manual_questions),
                "source": "manual_fallback"
            }
            
        except Exception as e:
            logger.error(f"❌ Ошибка ручной генерации для {tag}: {e}")
            return {"success": False, "tag": tag, "error": str(e)}
    
    def _create_manual_questions_templates(self, tag: str, weight: int) -> List[Dict[str, Any]]:
        """Создание базовых шаблонов вопросов"""
        questions = []
        
        # Базовые вопросы для любого тега
        base_questions = [
            {
                "question": f"Что такое {tag} и для чего используется?",
                "options": [
                    "Язык программирования",
                    "Технология разработки", 
                    "Инструмент анализа",
                    "Зависит от контекста"
                ],
                "correct_answer": "Зависит от контекста",
                "explanation": f"{tag} может быть различным в зависимости от области применения",
                "difficulty": "easy",
                "category": "basic_concepts"
            },
            {
                "question": f"Какой уровень знания {tag} требуется для эффективной работы?",
                "options": [
                    "Базовый уровень",
                    "Средний уровень",
                    "Продвинутый уровень", 
                    f"Зависит от важности {tag}"
                ],
                "correct_answer": f"Зависит от важности {tag}",
                "explanation": f"Уровень знания {tag} определяется его важностью для конкретной позиции",
                "difficulty": "medium",
                "category": "skill_level"
            }
        ]
        
        # Добавляем метаданные
        for question in base_questions:
            question["tag"] = tag
            question["tag_weight"] = weight
            question["id"] = str(uuid.uuid4())
            question["generated_at"] = datetime.now().isoformat() + "Z"
            question["source"] = "manual_template"
        
        return base_questions
    
    # === АНАЛИЗ СГЕНЕРИРОВАННЫХ ВОПРОСОВ ===
    
    def analyze_generated_questions(self, questions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Анализ качества сгенерированных вопросов"""
        try:
            if not questions:
                return {"total_questions": 0, "analysis": "Нет вопросов для анализа"}
            
            analysis = {
                "total_questions": len(questions),
                "questions_by_difficulty": self._analyze_difficulty_distribution(questions),
                "questions_by_tag": self._analyze_tag_distribution(questions),
                "questions_by_category": self._analyze_category_distribution(questions),
                "quality_metrics": self._analyze_quality_metrics(questions),
                "recommendations": self._generate_questions_recommendations(questions)
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Ошибка анализа вопросов: {e}")
            return {"error": str(e)}
    
    def _analyze_difficulty_distribution(self, questions: List[Dict[str, Any]]) -> Dict[str, int]:
        """Анализ распределения по сложности"""
        distribution = {"easy": 0, "medium": 0, "hard": 0}
        
        for question in questions:
            difficulty = question.get("difficulty", "medium")
            if difficulty in distribution:
                distribution[difficulty] += 1
        
        return distribution
    
    def _analyze_tag_distribution(self, questions: List[Dict[str, Any]]) -> Dict[str, int]:
        """Анализ распределения по тегам"""
        distribution = {}
        
        for question in questions:
            tag = question.get("tag", "unknown")
            distribution[tag] = distribution.get(tag, 0) + 1
        
        return distribution
    
    def _analyze_category_distribution(self, questions: List[Dict[str, Any]]) -> Dict[str, int]:
        """Анализ распределения по категориям"""
        distribution = {}
        
        for question in questions:
            category = question.get("category", "general")
            distribution[category] = distribution.get(category, 0) + 1
        
        return distribution
    
    def _analyze_quality_metrics(self, questions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Анализ метрик качества"""
        metrics = {
            "avg_question_length": 0,
            "avg_explanation_length": 0,
            "questions_with_categories": 0,
            "unique_questions": 0
        }
        
        question_texts = set()
        total_question_length = 0
        total_explanation_length = 0
        
        for question in questions:
            # Длина вопросов
            question_text = question.get("question", "")
            total_question_length += len(question_text)
            
            # Уникальность
            question_texts.add(question_text.lower())
            
            # Длина объяснений
            explanation = question.get("explanation", "")
            total_explanation_length += len(explanation)
            
            # Наличие категорий
            if question.get("category"):
                metrics["questions_with_categories"] += 1
        
        if questions:
            metrics["avg_question_length"] = total_question_length / len(questions)
            metrics["avg_explanation_length"] = total_explanation_length / len(questions)
            metrics["unique_questions"] = len(question_texts)
            metrics["uniqueness_percentage"] = (len(question_texts) / len(questions)) * 100
        
        return metrics
    
    def _generate_questions_recommendations(self, questions: List[Dict[str, Any]]) -> List[str]:
        """Генерация рекомендаций по улучшению вопросов"""
        recommendations = []
        
        # Анализируем распределение по сложности
        difficulty_dist = self._analyze_difficulty_distribution(questions)
        total = sum(difficulty_dist.values())
        
        if total > 0:
            easy_pct = (difficulty_dist["easy"] / total) * 100
            medium_pct = (difficulty_dist["medium"] / total) * 100
            hard_pct = (difficulty_dist["hard"] / total) * 100
            
            if easy_pct < 20:
                recommendations.append("Рекомендуется добавить больше простых вопросов")
            elif easy_pct > 50:
                recommendations.append("Слишком много простых вопросов")
            
            if hard_pct < 15:
                recommendations.append("Добавьте больше сложных вопросов для экспертной оценки")
            elif hard_pct > 40:
                recommendations.append("Слишком много сложных вопросов")
        
        # Анализируем качество
        quality_metrics = self._analyze_quality_metrics(questions)
        
        if quality_metrics.get("uniqueness_percentage", 100) < 90:
            recommendations.append("Обнаружены дублирующиеся вопросы")
        
        if quality_metrics.get("avg_question_length", 0) < 20:
            recommendations.append("Вопросы слишком короткие, добавьте детализации")
        
        if quality_metrics.get("questions_with_categories", 0) < len(questions) * 0.8:
            recommendations.append("Добавьте категории для большего количества вопросов")
        
        return recommendations

# Экспорт класса
__all__ = ['QuestionsGenerator']