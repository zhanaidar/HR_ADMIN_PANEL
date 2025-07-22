"""
HR Admin Panel v2.0 - Основной сервер
Простая архитектура с умными ИИ агентами
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

# FastAPI
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, Form, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
# from fastapi.middleware.sessions import SessionMiddleware
from starlette.middleware.sessions import SessionMiddleware
import uvicorn

# Планировщик задач
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

# Конфигурация и роли
from config import *
from users import find_user, update_last_login, get_demo_credentials
from roles import (
    get_user_role_name, 
    can_user_create_profession,
    can_user_approve_profession,
    can_user_return_to_hr,
    can_user_view_questions,
    can_user_access_profession,
    PROFESSION_STATUSES
)

# ИИ агенты
from ai_agents import HRAssistant, TagsGenerator, HeadApproval, QuestionsGenerator

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Создаем приложение
app = FastAPI(
    title="HR Admin Panel v2.0",
    description="Умная система создания профессий с ИИ помощниками",
    version="2.0.0"
)

# Middleware для сессий
app.add_middleware(SessionMiddleware, secret_key="hr-admin-panel-v2-secret-key-2025")

# Статические файлы и шаблоны
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Инициализируем ИИ агентов
hr_assistant = HRAssistant(OPENAI_API_KEY, DATA_DIR)
tags_generator = TagsGenerator(OPENAI_API_KEY, DATA_DIR)
head_approval = HeadApproval(OPENAI_API_KEY, DATA_DIR)
questions_generator = QuestionsGenerator(OPENAI_API_KEY, DATA_DIR)

# Планировщик задач
scheduler = AsyncIOScheduler()

# Хранилище активных WebSocket соединений
active_connections: Dict[str, WebSocket] = {}

# === СОБЫТИЯ ПРИЛОЖЕНИЯ ===

@app.on_event("startup")
async def startup_event():
    """Инициализация при запуске сервера"""
    logger.info("🚀 Запуск HR Admin Panel v2.0")
    
    # Обновляем справочники
    await update_reference_files()
    
    # Запускаем планировщик
    scheduler.start()
    
    # Планируем генерацию вопросов каждый день в 00:00
    scheduler.add_job(
        func=daily_questions_generation,
        trigger=CronTrigger(hour=0, minute=0),
        id='daily_questions_generation',
        replace_existing=True
    )
    
    logger.info("✅ Система готова к работе")

@app.on_event("shutdown")
async def shutdown_event():
    """Завершение работы"""
    scheduler.shutdown()
    logger.info("💤 HR Admin Panel остановлен")

# === ОСНОВНЫЕ МАРШРУТЫ ===

@app.get("/")
async def root():
    """Главная страница - редирект"""
    return RedirectResponse(url="/login")

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    """Страница логина"""
    return templates.TemplateResponse("login.html", {
        "request": request,
        "demo_accounts": get_demo_credentials(),
        "organization": ORGANIZATION
    })

@app.post("/login")
async def login(request: Request, email: str = Form(...), password: str = Form(...)):
    """Обработка логина"""
    user = find_user(email, password)
    if user:
        # Обновляем время последнего входа
        update_last_login(user["id"], datetime.now().isoformat())
        
        # Сохраняем в сессию
        request.session["user"] = user
        
        logger.info(f"✅ Пользователь вошел: {user['name']} ({user['email']})")
        
        # Перенаправляем в зависимости от роли
        if can_user_create_profession(user["role"]):
            return RedirectResponse(url="/create-profession", status_code=303)
        elif can_user_approve_profession(user["role"]):
            return RedirectResponse(url="/pending-approvals", status_code=303)
        elif can_user_view_questions(user["role"]):
            return RedirectResponse(url="/questions", status_code=303)
        else:
            return RedirectResponse(url="/dashboard", status_code=303)
    else:
        logger.warning(f"❌ Неудачная попытка входа: {email}")
        return templates.TemplateResponse("login.html", {
            "request": request,
            "error": "Неверный email или пароль",
            "demo_accounts": get_demo_credentials(),
            "organization": ORGANIZATION
        })

@app.get("/logout")
async def logout(request: Request):
    """Выход из системы"""
    user = request.session.get("user")
    if user:
        logger.info(f"👋 Пользователь вышел: {user['name']}")
    request.session.clear()
    return RedirectResponse(url="/login")

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Дашборд пользователя"""
    user = request.session.get("user")
    if not user:
        return RedirectResponse(url="/login")
    
    # Получаем статистику для пользователя
    stats = await get_user_statistics(user)
    
    # Проверяем есть ли профессии на подтверждение (для начальников)
    pending_notifications = []
    if can_user_approve_profession(user["role"]):
        pending_professions = get_pending_professions_for_user(user)
        if pending_professions:
            pending_notifications.append({
                "type": "pending_approvals",
                "count": len(pending_professions),
                "message": f"У вас {len(pending_professions)} профессий ожидают подтверждения",
                "action_url": "/pending-approvals"
            })
    
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "user": user,
        "user_role_name": get_user_role_name(user["role"]),
        "stats": stats,
        "notifications": pending_notifications,
        "can_create": can_user_create_profession(user["role"]),
        "can_approve": can_user_approve_profession(user["role"]),
        "can_view_questions": can_user_view_questions(user["role"])
    })

# === СОЗДАНИЕ ПРОФЕССИЙ ===

@app.get("/create-profession", response_class=HTMLResponse)
async def create_profession_page(request: Request):
    """Страница создания профессии"""
    user = request.session.get("user")
    if not user:
        return RedirectResponse(url="/login")
    
    # Проверяем права на создание
    if not can_user_create_profession(user["role"]):
        return templates.TemplateResponse("access_denied.html", {
            "request": request,
            "user": user,
            "user_role_name": get_user_role_name(user["role"]),
            "message": "У вас нет прав для создания профессий",
            "allowed_roles": ["super_admin", "hr_head_admin"]
        })
    
    # Загружаем справочники
    reference_data = load_reference_data()
    
    return templates.TemplateResponse("create_profession.html", {
        "request": request,
        "user": user,
        "user_role_name": get_user_role_name(user["role"]),
        "departments": reference_data.get("departments", []),
        "professions": reference_data.get("professions", []),
        "specializations": reference_data.get("specializations", [])
    })

@app.post("/api/analyze-file")
async def analyze_file(request: Request, file: UploadFile = File(...)):
    """Анализ загруженного файла"""
    user = request.session.get("user")
    if not user:
        return JSONResponse({"error": "Не авторизован"}, status_code=401)
    
    try:
        # Проверяем размер файла
        if file.size > MAX_FILE_SIZE:
            return JSONResponse({"error": "Файл слишком большой (максимум 10MB)"}, status_code=400)
        
        # Проверяем расширение
        file_extension = Path(file.filename).suffix.lower()
        if file_extension not in ALLOWED_EXTENSIONS:
            return JSONResponse({"error": f"Неподдерживаемый формат файла: {file_extension}"}, status_code=400)
        
        # Сохраняем файл
        file_path = UPLOADS_DIR / f"{user['id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
        
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Анализируем через HR Assistant
        analysis_result = await hr_assistant.analyze_file(str(file_path))
        
        # Удаляем файл после анализа
        file_path.unlink(missing_ok=True)
        
        logger.info(f"📄 Файл проанализирован: {file.filename} пользователем {user['name']}")
        return JSONResponse(analysis_result)
        
    except Exception as e:
        logger.error(f"❌ Ошибка анализа файла: {e}")
        return JSONResponse({"error": f"Ошибка анализа файла: {str(e)}"}, status_code=500)

@app.post("/api/analyze-form")
async def analyze_form(request: Request, form_data: dict):
    """Анализ данных формы в реальном времени"""
    user = request.session.get("user")
    if not user:
        return JSONResponse({"error": "Не авторизован"}, status_code=401)
    
    try:
        # Анализируем через HR Assistant
        analysis_result = await hr_assistant.analyze_form_data(form_data)
        
        return JSONResponse({
            "success": True,
            "analysis": analysis_result
        })
        
    except Exception as e:
        logger.error(f"❌ Ошибка анализа формы: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/api/create-profession")
async def create_profession(request: Request, profession_data: dict):
    """Создание профессии с полным workflow"""
    user = request.session.get("user")
    if not user:
        return JSONResponse({"error": "Не авторизован"}, status_code=401)
    
    # Проверяем права
    if not can_user_create_profession(user["role"]):
        return JSONResponse({"error": "У вас нет прав для создания профессий"}, status_code=403)
    
    try:
        # 1. Создаем профессию со статусом "created_by_hr"
        profession_id = await save_profession(profession_data, user, "created_by_hr")
        
        # 2. Генерируем теги через Tags Generator
        tags_result = await tags_generator.generate_tags(profession_data)
        
        if tags_result.get("success"):
            # 3. Сохраняем теги и меняем статус на "tags_generated"
            await update_profession_with_tags(profession_id, tags_result, user)
            
            # 4. Отправляем уведомление начальнику отдела
            await notify_department_head(profession_data, profession_id)
            
            logger.info(f"✅ Профессия создана: {profession_data.get('bank_title')} (ID: {profession_id}) пользователем {user['name']}")
            
            return JSONResponse({
                "success": True,
                "profession_id": profession_id,
                "tags_count": len(tags_result["tags"]),
                "message": "Профессия создана! Теги сгенерированы ИИ и отправлены на подтверждение начальнику отдела."
            })
        else:
            logger.warning(f"⚠️ Профессия создана, но теги не сгенерированы: {tags_result.get('error')}")
            return JSONResponse({
                "success": True,
                "profession_id": profession_id,
                "warning": "Профессия создана, но теги не сгенерированы. Требуется ручная настройка."
            })
        
    except Exception as e:
        logger.error(f"❌ Ошибка создания профессии: {e}")
        return JSONResponse({"error": f"Ошибка создания: {str(e)}"}, status_code=500)

# === УТВЕРЖДЕНИЕ ПРОФЕССИЙ ===

@app.get("/pending-approvals", response_class=HTMLResponse)
async def pending_approvals_page(request: Request):
    """Страница подтверждения профессий для начальников"""
    user = request.session.get("user")
    if not user:
        return RedirectResponse(url="/login")
    
    # Проверяем права
    if not can_user_approve_profession(user["role"]):
        return templates.TemplateResponse("access_denied.html", {
            "request": request,
            "user": user,
            "user_role_name": get_user_role_name(user["role"]),
            "message": "У вас нет прав для подтверждения профессий",
            "allowed_roles": ["head_admin", "super_admin"]
        })
    
    # Получаем профессии ожидающие подтверждения
    pending_professions = get_pending_professions_for_user(user)
    
    return templates.TemplateResponse("pending_approvals.html", {
        "request": request,
        "user": user,
        "user_role_name": get_user_role_name(user["role"]),
        "professions": pending_professions,
        "total_pending": len(pending_professions)
    })

@app.get("/api/profession/{profession_id}")
async def get_profession_details(profession_id: str, request: Request):
    """Получение деталей профессии"""
    user = request.session.get("user")
    if not user:
        return JSONResponse({"error": "Не авторизован"}, status_code=401)
    
    try:
        profession = get_profession_by_id(profession_id)
        if not profession:
            return JSONResponse({"error": "Профессия не найдена"}, status_code=404)
        
        # Проверяем права доступа
        if not can_user_access_profession(user, profession):
            return JSONResponse({"error": "Нет прав для просмотра этой профессии"}, status_code=403)
        
        # Если это начальник, добавляем анализ для утверждения
        if can_user_approve_profession(user["role"]):
            analysis = await head_approval.analyze_profession_for_approval(
                profession_id, user["department"]
            )
            
            return JSONResponse({
                "success": True,
                "profession": profession,
                "analysis": analysis.get("analysis", {}),
                "recommendations": analysis.get("recommendations", []),
                "suggested_actions": analysis.get("suggested_actions", [])
            })
        else:
            return JSONResponse({
                "success": True,
                "profession": profession
            })
        
    except Exception as e:
        logger.error(f"❌ Ошибка получения профессии {profession_id}: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/api/approve-profession/{profession_id}")
async def approve_profession(profession_id: str, request: Request, approval_data: dict):
    """Подтверждение профессии начальником"""
    user = request.session.get("user")
    if not user:
        return JSONResponse({"error": "Не авторизован"}, status_code=401)
    
    try:
        profession = get_profession_by_id(profession_id)
        if not profession:
            return JSONResponse({"error": "Профессия не найдена"}, status_code=404)
        
        # Проверяем права
        if not can_user_approve_profession(user["role"]):
            return JSONResponse({"error": "У вас нет прав для подтверждения профессий"}, status_code=403)
        
        # Проверяем что это профессия из отдела пользователя
        if not can_user_access_profession(user, profession):
            return JSONResponse({"error": "Нет прав для подтверждения этой профессии"}, status_code=403)
        
        action = approval_data.get("action")
        
        if action == "approve":
            # Утверждаем профессию
            corrected_tags = approval_data.get("tags", profession.get("tags", {}))
            comment = approval_data.get("comment", "")
            
            await approve_profession_by_head(profession_id, corrected_tags, user, comment)
            
            logger.info(f"✅ Профессия утверждена: {profession_id} пользователем {user['name']}")
            
            return JSONResponse({
                "success": True,
                "message": "Профессия утверждена! Вопросы будут сгенерированы автоматически."
            })
            
        elif action == "return_to_hr":
            # Возвращаем на доработку
            return_reason = approval_data.get("return_reason", "")
            return_comment = approval_data.get("return_comment", "")
            
            await return_profession_to_hr(profession_id, return_reason, return_comment, user)
            
            logger.info(f"↩️ Профессия возвращена на доработку: {profession_id} пользователем {user['name']}")
            
            return JSONResponse({
                "success": True,
                "message": "Профессия возвращена HR на доработку. Уведомление отправлено."
            })
        else:
            return JSONResponse({"error": "Неизвестное действие"}, status_code=400)
        
    except Exception as e:
        logger.error(f"❌ Ошибка подтверждения профессии {profession_id}: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

# === ПРОСМОТР ВОПРОСОВ ===

@app.get("/questions", response_class=HTMLResponse)
async def questions_page(request: Request):
    """Страница вопросов - только для супер админа"""
    user = request.session.get("user")
    if not user:
        return RedirectResponse(url="/login")
    
    # Только супер админ может видеть вопросы
    if not can_user_view_questions(user["role"]):
        return templates.TemplateResponse("access_denied.html", {
            "request": request,
            "user": user,
            "user_role_name": get_user_role_name(user["role"]),
            "message": "У вас нет прав для просмотра вопросов",
            "allowed_roles": ["super_admin"]
        })
    
    # Получаем все вопросы
    questions_data = get_all_questions()
    
    return templates.TemplateResponse("questions.html", {
        "request": request,
        "user": user,
        "user_role_name": get_user_role_name(user["role"]),
        "questions": questions_data["questions"],
        "stats": questions_data["stats"],
        "total_questions": len(questions_data["questions"])
    })

@app.post("/api/generate-questions/{profession_id}")
async def generate_questions_manual(profession_id: str, request: Request):
    """Ручная генерация вопросов (для тестирования)"""
    user = request.session.get("user")
    if not user:
        return JSONResponse({"error": "Не авторизован"}, status_code=401)
    
    # Только супер админ может генерировать вопросы
    if not can_user_view_questions(user["role"]):
        return JSONResponse({"error": "У вас нет прав для генерации вопросов"}, status_code=403)
    
    try:
        profession = get_profession_by_id(profession_id)
        if not profession:
            return JSONResponse({"error": "Профессия не найдена"}, status_code=404)
        
        if profession.get("status") != "approved_by_head":
            return JSONResponse({"error": "Профессия еще не утверждена"}, status_code=400)
        
        # Генерируем вопросы
        questions_result = await questions_generator.generate_questions_for_profession(profession)
        
        if questions_result.get("success"):
            # Сохраняем вопросы
            await save_profession_questions(profession_id, questions_result["questions"])
            
            # Обновляем статус
            await update_profession_status(profession_id, "questions_generated")
            
            logger.info(f"❓ Вопросы сгенерированы для {profession_id}: {questions_result['stats']['total_questions']} вопросов")
            
            return JSONResponse({
                "success": True,
                "questions_count": questions_result["stats"]["total_questions"],
                "stats": questions_result["stats"],
                "message": "Вопросы успешно сгенерированы!"
            })
        else:
            return JSONResponse({"error": f"Ошибка генерации вопросов: {questions_result.get('error')}"}, status_code=500)
        
    except Exception as e:
        logger.error(f"❌ Ошибка генерации вопросов для {profession_id}: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)



# === УПРАВЛЕНИЕ ГЕНЕРАЦИЕЙ ВОПРОСОВ ===

@app.get("/questions-management", response_class=HTMLResponse)
async def questions_management_page(request: Request):
    """Страница управления генерацией вопросов - только для супер админа"""
    user = request.session.get("user")
    if not user:
        return RedirectResponse(url="/login")
    
    # Только супер админ может управлять вопросами
    if user["role"] != "super_admin":
        return templates.TemplateResponse("access_denied.html", {
            "request": request,
            "user": user,
            "user_role_name": get_user_role_name(user["role"]),
            "message": "Управление генерацией вопросов доступно только супер администратору",
            "allowed_roles": ["super_admin"]
        })
    
    return templates.TemplateResponse("questions_management.html", {
        "request": request,
        "user": user,
        "user_role_name": get_user_role_name(user["role"])
    })

@app.get("/api/questions-overview")
async def get_questions_overview(request: Request):
    """Получение обзора всех профессий с вопросами"""
    user = request.session.get("user")
    if not user:
        return JSONResponse({"error": "Не авторизован"}, status_code=401)
    
    if user["role"] != "super_admin":
        return JSONResponse({"error": "Доступ запрещен"}, status_code=403)
    
    try:
        records_file = DATA_DIR / "profession_records.json"
        
        if not records_file.exists():
            return JSONResponse({
                "success": True,
                "ready": [],
                "pending": []
            })
        
        with open(records_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        ready_professions = []
        pending_professions = []
        
        for record in data.get("profession_records", []):
            profession_key = f"{record.get('real_name', '')} - {record.get('specialization', 'Общая')}"
            
            profession_info = {
                "profession_key": profession_key,
                "profession": record.get('real_name', ''),
                "specialization": record.get('specialization', ''),
                "profession_id": record.get('id', ''),
                "updated_at": record.get('questions_generated_at', record.get('updated_at'))
            }
            
            if record.get("status") == "questions_generated" and record.get("questions"):
                # Подсчитываем вопросы по сложности
                questions = record.get("questions", [])
                breakdown = {"easy": 0, "medium": 0, "hard": 0}
                
                for question in questions:
                    difficulty = question.get("difficulty", "medium")
                    if difficulty in breakdown:
                        breakdown[difficulty] += 1
                
                profession_info.update({
                    "questions_count": len(questions),
                    "breakdown": breakdown
                })
                
                ready_professions.append(profession_info)
                
            elif record.get("status") == "approved_by_head":
                # Считаем ожидаемое количество вопросов на основе тегов
                tags = record.get("tags", {})
                tags_count = len(tags)
                expected_questions = calculate_expected_questions_count(tags)
                
                # Получаем топ-3 тега
                top_tags = []
                if tags:
                    sorted_tags = sorted(tags.items(), key=lambda x: x[1], reverse=True)
                    top_tags = [tag for tag, weight in sorted_tags[:3]]
                
                profession_info.update({
                    "expected_questions": f"~{expected_questions}",
                    "tags_count": tags_count,
                    "top_tags": top_tags,
                    "status": "pending"
                })
                
                pending_professions.append(profession_info)
        
        return JSONResponse({
            "success": True,
            "ready": ready_professions,
            "pending": pending_professions
        })
        
    except Exception as e:
        logger.error(f"❌ Ошибка получения обзора вопросов: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

@app.delete("/api/questions/{profession_key}")
async def delete_questions_by_key(profession_key: str, request: Request):
    """Удаление всех вопросов для профессии-специализации"""
    user = request.session.get("user")
    if not user:
        return JSONResponse({"error": "Не авторизован"}, status_code=401)
    
    if user["role"] != "super_admin":
        return JSONResponse({"error": "Доступ запрещен"}, status_code=403)
    
    try:
        # Декодируем ключ профессии
        profession_key = profession_key.replace("%20", " ")
        profession_name, specialization = parse_profession_key(profession_key)
        
        records_file = DATA_DIR / "profession_records.json"
        
        with open(records_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        found = False
        for record in data["profession_records"]:
            if (record.get("real_name") == profession_name and 
                record.get("specialization", "Общая") == specialization):
                
                # Очищаем вопросы и возвращаем статус
                record["questions"] = []
                record["status"] = "approved_by_head"
                record.pop("questions_generated_at", None)
                
                # Добавляем в историю
                record["workflow_history"].append({
                    "status": "questions_cleared",
                    "timestamp": datetime.now().isoformat() + "Z",
                    "user": user["email"],
                    "action": f"Вопросы удалены супер админом"
                })
                
                found = True
                break
        
        if not found:
            return JSONResponse({"error": "Профессия не найдена"}, status_code=404)
        
        # Сохраняем изменения
        with open(records_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"🗑️ Вопросы удалены для {profession_key} пользователем {user['name']}")
        
        return JSONResponse({
            "success": True,
            "message": f"Все вопросы для '{profession_key}' успешно удалены"
        })
        
    except Exception as e:
        logger.error(f"❌ Ошибка удаления вопросов: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/api/generate-single/{profession_key}")
async def generate_questions_for_single_profession(profession_key: str, request: Request):
    """Генерация вопросов для одной профессии-специализации"""
    user = request.session.get("user")
    if not user:
        return JSONResponse({"error": "Не авторизован"}, status_code=401)
    
    if user["role"] != "super_admin":
        return JSONResponse({"error": "Доступ запрещен"}, status_code=403)
    
    try:
        # Декодируем ключ профессии
        profession_key = profession_key.replace("%20", " ")
        profession_name, specialization = parse_profession_key(profession_key)
        
        # Находим профессию
        records_file = DATA_DIR / "profession_records.json"
        
        with open(records_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        target_profession = None
        for record in data["profession_records"]:
            if (record.get("real_name") == profession_name and 
                record.get("specialization", "Общая") == specialization and
                record.get("status") == "approved_by_head"):
                target_profession = record
                break
        
        if not target_profession:
            return JSONResponse({"error": "Профессия не найдена или не готова к генерации"}, status_code=404)
        
        # Обновляем статус на "генерируется"
        target_profession["status"] = "generating"
        target_profession["generation_started_at"] = datetime.now().isoformat() + "Z"
        target_profession["workflow_history"].append({
            "status": "generation_started",
            "timestamp": datetime.now().isoformat() + "Z",
            "user": user["email"],
            "action": f"Запущена генерация вопросов супер админом"
        })
        
        # Сохраняем промежуточный статус
        with open(records_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        # Запускаем генерацию в фоне
        import asyncio
        asyncio.create_task(generate_questions_background(target_profession, user))
        
        logger.info(f"🤖 Запущена генерация вопросов для {profession_key} пользователем {user['name']}")
        
        return JSONResponse({
            "success": True,
            "message": f"Генерация вопросов для '{profession_key}' запущена",
            "profession_id": target_profession["id"]
        })
        
    except Exception as e:
        logger.error(f"❌ Ошибка запуска генерации: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

# === ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ===

def calculate_expected_questions_count(tags: Dict[str, int]) -> int:
    """Расчет ожидаемого количества вопросов на основе тегов"""
    if not tags:
        return 0
    
    total_questions = 0
    for tag, weight in tags.items():
        if weight >= 85:
            total_questions += 50  # Критично важный тег
        elif weight >= 70:
            total_questions += 40  # Важный тег
        elif weight >= 55:
            total_questions += 32  # Средний тег
        else:
            total_questions += 25  # Низкий тег
    
    return total_questions

def parse_profession_key(profession_key: str) -> tuple:
    """Парсинг ключа профессии 'Название - Специализация'"""
    if " - " in profession_key:
        parts = profession_key.split(" - ", 1)
        return parts[0], parts[1]
    else:
        return profession_key, "Общая"

async def generate_questions_background(profession: Dict[str, Any], user: Dict[str, Any]):
    """Фоновая генерация вопросов для профессии"""
    try:
        # Генерируем вопросы
        questions_result = await questions_generator.generate_questions_for_profession(profession)
        
        # Обновляем профессию в файле
        records_file = DATA_DIR / "profession_records.json"
        
        with open(records_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Находим и обновляем профессию
        for record in data["profession_records"]:
            if record["id"] == profession["id"]:
                if questions_result.get("success"):
                    record["questions"] = questions_result["questions"]
                    record["status"] = "questions_generated"
                    record["questions_generated_at"] = datetime.now().isoformat() + "Z"
                    record["workflow_history"].append({
                        "status": "questions_generated",
                        "timestamp": datetime.now().isoformat() + "Z",
                        "user": "system",
                        "action": f"ИИ сгенерировал {questions_result['stats']['total_questions']} вопросов"
                    })
                    
                    logger.info(f"✅ Фоновая генерация завершена для {record['real_name']}: {questions_result['stats']['total_questions']} вопросов")
                else:
                    record["status"] = "approved_by_head"  # Возвращаем исходный статус
                    record["workflow_history"].append({
                        "status": "generation_failed",
                        "timestamp": datetime.now().isoformat() + "Z",
                        "user": "system",
                        "action": f"Ошибка генерации: {questions_result.get('error', 'Неизвестная ошибка')}"
                    })
                    
                    logger.error(f"❌ Фоновая генерация не удалась для {record['real_name']}: {questions_result.get('error')}")
                
                record.pop("generation_started_at", None)
                break
        
        # Сохраняем результат
        with open(records_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
    except Exception as e:
        logger.error(f"❌ Ошибка фоновой генерации: {e}")
        
        # В случае ошибки возвращаем статус обратно
        try:
            records_file = DATA_DIR / "profession_records.json"
            with open(records_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for record in data["profession_records"]:
                if record["id"] == profession["id"]:
                    record["status"] = "approved_by_head"
                    record.pop("generation_started_at", None)
                    break
            
            with open(records_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except:
            pass


# === WEBSOCKET ДЛЯ ЧАТА ===

@app.websocket("/ws/chat/{user_id}")
async def websocket_chat(websocket: WebSocket, user_id: str):
    """WebSocket для чата с ИИ помощниками"""
    await websocket.accept()
    active_connections[user_id] = websocket
    
    try:
        await websocket.send_text(json.dumps({
            "type": "system",
            "message": "🤖 ИИ Помощник подключен! Задавайте вопросы."
        }))
        
        while True:
            # Получаем сообщение от пользователя
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            user_message = message_data.get("message", "")
            form_context = message_data.get("form_context", {})
            chat_type = message_data.get("chat_type", "hr_assistant")  # hr_assistant, head_approval
            
            # Выбираем нужного ИИ агента
            if chat_type == "head_approval":
                ai_response = await head_approval.chat_with_head(user_message, form_context)
            else:
                ai_response = await hr_assistant.chat_with_user(user_message, form_context)
            
            # Отправляем ответ пользователю
            await websocket.send_text(json.dumps({
                "type": "ai_response",
                "message": ai_response.get("message", ""),
                "analysis": ai_response.get("analysis", {}),
                "suggestions": ai_response.get("suggestions", [])
            }))
            
    except WebSocketDisconnect:
        logger.info(f"🔌 WebSocket отключен: {user_id}")
        if user_id in active_connections:
            del active_connections[user_id]
    except Exception as e:
        logger.error(f"❌ Ошибка WebSocket: {e}")

# === ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ===

async def save_profession(profession_data: Dict[str, Any], user: Dict[str, Any], status: str) -> str:
    """Сохранение профессии"""
    try:
        # Загружаем существующие записи
        records_file = DATA_DIR / "profession_records.json"
        
        if records_file.exists():
            with open(records_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            data = {"profession_records": []}
        
        # Создаем новую запись
        profession_id = f"prof_{len(data['profession_records']) + 1:04d}"
        
        # Определяем начальника отдела
        department_head = get_department_head_email(profession_data.get("department", ""))
        
        new_profession = {
            "id": profession_id,
            "bank_title": profession_data.get("bank_title", ""),
            "real_name": profession_data.get("real_name", ""),
            "specialization": profession_data.get("specialization", ""),
            "department": profession_data.get("department", ""),
            "department_head": department_head,
            "created_by": user["email"],
            "created_at": datetime.now().isoformat() + "Z",
            "status": status,
            "tags": {},
            "tags_versions": [],
            "questions": [],
            "workflow_history": [
                {
                    "status": status,
                    "timestamp": datetime.now().isoformat() + "Z",
                    "user": user["email"],
                    "action": "Профессия создана HR директором"
                }
            ]
        }
        
        data["profession_records"].append(new_profession)
        
        # Сохраняем
        with open(records_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        return profession_id
        
    except Exception as e:
        logger.error(f"❌ Ошибка сохранения профессии: {e}")
        raise

async def update_profession_with_tags(profession_id: str, tags_result: Dict[str, Any], user: Dict[str, Any]):
    """Обновление профессии с тегами"""
    try:
        records_file = DATA_DIR / "profession_records.json"
        
        with open(records_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Находим профессию
        for profession in data["profession_records"]:
            if profession["id"] == profession_id:
                profession["tags"] = tags_result["tags"]
                profession["tags_versions"] = [tags_result["tags_version"]]
                profession["status"] = "tags_generated"
                profession["tags_generated_at"] = datetime.now().isoformat() + "Z"
                profession["workflow_history"].append({
                    "status": "tags_generated",
                    "timestamp": datetime.now().isoformat() + "Z",
                    "user": "system",
                    "action": f"ИИ сгенерировал {len(tags_result['tags'])} тегов"
                })
                break
        
        # Сохраняем
        with open(records_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
    except Exception as e:
        logger.error(f"❌ Ошибка обновления тегов: {e}")
        raise

def load_reference_data() -> Dict[str, Any]:
    """Загрузка справочных данных"""
    try:
        data = {}
        
        reference_files = ['departments.json', 'professions.json', 'specializations.json', 'bank_titles.json']
        
        for filename in reference_files:
            file_path = DATA_DIR / filename
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    key = filename.replace('.json', '')
                    data[key] = json.load(f).get(key, [])
        
        return data
        
    except Exception as e:
        logger.error(f"❌ Ошибка загрузки справочных данных: {e}")
        return {}

async def update_reference_files():
    """Автообновление справочников из утвержденных профессий"""
    try:
        records_file = DATA_DIR / "profession_records.json"
        
        if not records_file.exists():
            logger.info("📁 Файл profession_records.json не найден, пропускаем обновление справочников")
            return
        
        with open(records_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Извлекаем данные только из утвержденных профессий
        approved_records = [
            record for record in data.get("profession_records", [])
            if record.get("status") in ["approved_by_head", "questions_generated", "active"]
        ]
        
        if not approved_records:
            logger.info("📊 Нет утвержденных профессий для обновления справочников")
            return
        
        # Собираем уникальные значения
        departments = set()
        professions = set()
        specializations = set()
        bank_titles = set()
        all_tags = set()
        
        for record in approved_records:
            if record.get("department"):
                departments.add(record["department"])
            if record.get("real_name"):
                professions.add(record["real_name"])
            if record.get("specialization"):
                specializations.add(record["specialization"])
            if record.get("bank_title"):
                bank_titles.add(record["bank_title"])
            
            # Собираем теги
            for tag in record.get("tags", {}):
                all_tags.add(tag)
        
        # Сохраняем справочники
        reference_data = {
            "professions.json": {"professions": sorted(list(professions))},
            "specializations.json": {"specializations": sorted(list(specializations))},
            "bank_titles.json": {"bank_titles": sorted(list(bank_titles))},
            "tags.json": {"tags": sorted(list(all_tags))}
        }
        
        for filename, content in reference_data.items():
            file_path = DATA_DIR / filename
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(content, f, ensure_ascii=False, indent=2)
        
        logger.info(f"📊 Справочники обновлены: {len(professions)} профессий, {len(bank_titles)} названий, {len(all_tags)} тегов")
        
    except Exception as e:
        logger.error(f"❌ Ошибка обновления справочников: {e}")

def get_department_head_email(department: str) -> str:
    """Получение email начальника отдела"""
    try:
        departments_file = DATA_DIR / "departments.json"
        
        if departments_file.exists():
            with open(departments_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            for dept in data.get("departments", []):
                if dept["name"] == department:
                    return dept.get("head_email", "unknown@halykbank.kz")
        
        return "unknown@halykbank.kz"
        
    except Exception as e:
        logger.error(f"❌ Ошибка получения начальника отдела: {e}")
        return "unknown@halykbank.kz"

async def notify_department_head(profession_data: Dict[str, Any], profession_id: str):
    """Уведомление начальника отдела"""
    try:
        department_head = get_department_head_email(profession_data.get("department", ""))
        
        logger.info(f"📧 УВЕДОМЛЕНИЕ НАЧАЛЬНИКУ ОТДЕЛА: {department_head}")
        logger.info(f"🏦 Новая профессия требует подтверждения:")
        logger.info(f"   • ID: {profession_id}")
        logger.info(f"   • Банковское название: {profession_data.get('bank_title')}")
        logger.info(f"   • Реальная профессия: {profession_data.get('real_name')}")
        logger.info(f"   • Департамент: {profession_data.get('department')}")
        logger.info(f"🔗 Ссылка: http://localhost:{APP_PORT}/pending-approvals")
        
        # В реальной системе здесь будет отправка email
        return True
        
    except Exception as e:
        logger.error(f"❌ Ошибка уведомления начальника: {e}")
        return False

def get_pending_professions_for_user(user: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Получение профессий ожидающих подтверждения пользователем"""
    try:
        records_file = DATA_DIR / "profession_records.json"
        
        if not records_file.exists():
            return []
        
        with open(records_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        pending = []
        for record in data.get("profession_records", []):
            if record.get("status") == "tags_generated":
                # Супер админ видит все
                if user["role"] == "super_admin":
                    pending.append(record)
                # Начальник отдела видит только свои
                elif (user["role"] == "head_admin" and 
                      record.get("department") == f"{user['department']} Department"):
                    pending.append(record)
        
        return pending
        
    except Exception as e:
        logger.error(f"❌ Ошибка получения ожидающих профессий: {e}")
        return []

def get_profession_by_id(profession_id: str) -> Optional[Dict[str, Any]]:
    """Получение профессии по ID"""
    try:
        records_file = DATA_DIR / "profession_records.json"
        
        if not records_file.exists():
            return None
        
        with open(records_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for record in data.get("profession_records", []):
            if record["id"] == profession_id:
                return record
        
        return None
        
    except Exception as e:
        logger.error(f"❌ Ошибка получения профессии {profession_id}: {e}")
        return None

async def approve_profession_by_head(profession_id: str, corrected_tags: Dict[str, int], user: Dict[str, Any], comment: str):
    """Утверждение профессии начальником отдела"""
    try:
        records_file = DATA_DIR / "profession_records.json"
        
        with open(records_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for record in data["profession_records"]:
            if record["id"] == profession_id:
                # Создаем новую версию тегов
                new_version = {
                    "version": len(record.get("tags_versions", [])) + 1,
                    "created_by": user["email"],
                    "timestamp": datetime.now().isoformat() + "Z",
                    "action": "Корректировка и утверждение начальником отдела",
                    "tags": corrected_tags.copy(),
                    "total_tags": len(corrected_tags),
                    "comment": comment,
                    "changes": calculate_tags_changes(record.get("tags", {}), corrected_tags)
                }
                
                # Обновляем профессию
                record["tags"] = corrected_tags
                record["tags_versions"].append(new_version)
                record["status"] = "approved_by_head"
                record["approved_at"] = datetime.now().isoformat() + "Z"
                record["approved_by"] = user["email"]
                record["approval_comment"] = comment
                record["workflow_history"].append({
                    "status": "approved_by_head",
                    "timestamp": datetime.now().isoformat() + "Z",
                    "user": user["email"],
                    "action": f"Профессия утверждена с {len(corrected_tags)} тегами"
                })
                break
        
        with open(records_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        # Обновляем справочники
        await update_reference_files()
        
    except Exception as e:
        logger.error(f"❌ Ошибка утверждения профессии: {e}")
        raise

def calculate_tags_changes(old_tags: Dict[str, int], new_tags: Dict[str, int]) -> Dict[str, Any]:
    """Расчет изменений в тегах"""
    changes = {
        "added": [],
        "removed": [],
        "modified": {},
        "unchanged": []
    }
    
    # Найдем добавленные теги
    for tag in new_tags:
        if tag not in old_tags:
            changes["added"].append(tag)
    
    # Найдем удаленные теги
    for tag in old_tags:
        if tag not in new_tags:
            changes["removed"].append(tag)
    
    # Найдем измененные и неизмененные теги
    for tag in old_tags:
        if tag in new_tags:
            if old_tags[tag] != new_tags[tag]:
                changes["modified"][tag] = {
                    "from": old_tags[tag],
                    "to": new_tags[tag]
                }
            else:
                changes["unchanged"].append(tag)
    
    return changes

async def return_profession_to_hr(profession_id: str, return_reason: str, return_comment: str, user: Dict[str, Any]):
    """Возврат профессии на доработку HR"""
    try:
        records_file = DATA_DIR / "profession_records.json"
        
        with open(records_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for record in data["profession_records"]:
            if record["id"] == profession_id:
                record["status"] = "returned_to_hr"
                record["returned_at"] = datetime.now().isoformat() + "Z"
                record["returned_by"] = user["email"]
                record["return_reason"] = return_reason
                record["return_comment"] = return_comment
                record["workflow_history"].append({
                    "status": "returned_to_hr",
                    "timestamp": datetime.now().isoformat() + "Z",
                    "user": user["email"],
                    "action": f"Возвращена на доработку: {return_reason}"
                })
                break
        
        with open(records_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        # Уведомляем HR
        await notify_hr_about_return(profession_id, return_reason, return_comment)
        
    except Exception as e:
        logger.error(f"❌ Ошибка возврата профессии: {e}")
        raise

async def notify_hr_about_return(profession_id: str, return_reason: str, return_comment: str):
    """Уведомление HR о возврате профессии"""
    logger.info(f"📧 УВЕДОМЛЕНИЕ HR О ВОЗВРАТЕ: {profession_id}")
    logger.info(f"   • Причина: {return_reason}")
    logger.info(f"   • Комментарий: {return_comment}")
    # В реальной системе здесь будет отправка email

async def save_profession_questions(profession_id: str, questions: List[Dict[str, Any]]):
    """Сохранение вопросов для профессии"""
    try:
        records_file = DATA_DIR / "profession_records.json"
        
        with open(records_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for record in data["profession_records"]:
            if record["id"] == profession_id:
                record["questions"] = questions
                record["questions_generated_at"] = datetime.now().isoformat() + "Z"
                record["workflow_history"].append({
                    "status": "questions_generated",
                    "timestamp": datetime.now().isoformat() + "Z",
                    "user": "system",
                    "action": f"ИИ сгенерировал {len(questions)} вопросов"
                })
                break
        
        with open(records_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
    except Exception as e:
        logger.error(f"❌ Ошибка сохранения вопросов: {e}")
        raise

async def update_profession_status(profession_id: str, status: str):
    """Обновление статуса профессии"""
    try:
        records_file = DATA_DIR / "profession_records.json"
        
        with open(records_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for record in data["profession_records"]:
            if record["id"] == profession_id:
                record["status"] = status
                record["status_updated_at"] = datetime.now().isoformat() + "Z"
                break
        
        with open(records_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
    except Exception as e:
        logger.error(f"❌ Ошибка обновления статуса: {e}")
        raise

def get_all_questions() -> Dict[str, Any]:
    """Получение всех вопросов"""
    try:
        records_file = DATA_DIR / "profession_records.json"
        
        if not records_file.exists():
            return {"questions": [], "stats": {}}
        
        with open(records_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        all_questions = []
        stats = {"total_questions": 0, "questions_by_difficulty": {"easy": 0, "medium": 0, "hard": 0}}
        
        for record in data.get("profession_records", []):
            if record.get("questions"):
                for question in record["questions"]:
                    question["profession_id"] = record["id"]
                    question["profession_title"] = record["bank_title"]
                    all_questions.append(question)
                    
                    # Статистика
                    stats["total_questions"] += 1
                    difficulty = question.get("difficulty", "medium")
                    if difficulty in stats["questions_by_difficulty"]:
                        stats["questions_by_difficulty"][difficulty] += 1
        
        return {"questions": all_questions, "stats": stats}
        
    except Exception as e:
        logger.error(f"❌ Ошибка получения всех вопросов: {e}")
        return {"questions": [], "stats": {}}

async def get_user_statistics(user: Dict[str, Any]) -> Dict[str, Any]:
    """Получение статистики для пользователя"""
    try:
        records_file = DATA_DIR / "profession_records.json"
        
        if not records_file.exists():
            return {}
        
        with open(records_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        stats = {
            "total_professions": 0,
            "created_by_user": 0,
            "pending_approval": 0,
            "approved": 0,
            "questions_generated": 0
        }
        
        for record in data.get("profession_records", []):
            stats["total_professions"] += 1
            
            if record.get("created_by") == user["email"]:
                stats["created_by_user"] += 1
            
            status = record.get("status", "")
            if status == "tags_generated":
                stats["pending_approval"] += 1
            elif status == "approved_by_head":
                stats["approved"] += 1
            elif status == "questions_generated":
                stats["questions_generated"] += 1
        
        return stats
        
    except Exception as e:
        logger.error(f"❌ Ошибка получения статистики: {e}")
        return {}

# === ПЛАНИРОВЩИК ЗАДАЧ ===

async def daily_questions_generation():
    """Ежедневная генерация вопросов в 00:00"""
    try:
        logger.info("🌙 Запуск ежедневной генерации вопросов")
        
        records_file = DATA_DIR / "profession_records.json"
        
        if not records_file.exists():
            logger.info("📁 Нет файла profession_records.json")
            return
        
        with open(records_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Находим профессии со статусом "approved_by_head"
        approved_professions = [
            record for record in data.get("profession_records", [])
            if record.get("status") == "approved_by_head"
        ]
        
        if not approved_professions:
            logger.info("📊 Нет утвержденных профессий для генерации вопросов")
            return
        
        generated_count = 0
        
        for profession in approved_professions:
            try:
                # Генерируем вопросы
                questions_result = await questions_generator.generate_questions_for_profession(profession)
                
                if questions_result.get("success"):
                    # Сохраняем вопросы
                    await save_profession_questions(profession["id"], questions_result["questions"])
                    
                    # Обновляем статус
                    await update_profession_status(profession["id"], "questions_generated")
                    
                    generated_count += 1
                    logger.info(f"✅ Вопросы сгенерированы для {profession['id']}: {questions_result['stats']['total_questions']} вопросов")
                else:
                    logger.error(f"❌ Не удалось сгенерировать вопросы для {profession['id']}")
                    
            except Exception as e:
                logger.error(f"❌ Ошибка генерации вопросов для {profession['id']}: {e}")
        
        logger.info(f"🌙 Ежедневная генерация завершена: {generated_count} профессий обработано")
        
    except Exception as e:
        logger.error(f"❌ Ошибка ежедневной генерации вопросов: {e}")
        
        
        
# === УПРАВЛЕНИЕ ТЕСТ-СЕССИЯМИ КАНДИДАТОВ ===
# Добавьте эти endpoints в ваш main.py

import uuid
import random
from collections import Counter

@app.get("/create-candidate-test", response_class=HTMLResponse)
async def create_candidate_test_page(request: Request):
    """Страница создания теста для кандидата"""
    user = request.session.get("user")
    if not user:
        return RedirectResponse(url="/login")
    
    # Проверяем права (могут создавать тесты те, кто может просматривать вопросы или создавать профессии)
    if not (can_user_view_questions(user["role"]) or can_user_create_profession(user["role"])):
        return templates.TemplateResponse("access_denied.html", {
            "request": request,
            "user": user,
            "user_role_name": get_user_role_name(user["role"]),
            "message": "У вас нет прав для создания тестов",
            "allowed_roles": ["super_admin", "hr_head_admin"]
        })
    
    # Получаем доступные профессии с вопросами
    available_professions = get_professions_with_questions()
    
    return templates.TemplateResponse("create_candidate_test.html", {
        "request": request,
        "user": user,
        "user_role_name": get_user_role_name(user["role"]),
        "professions": available_professions,
        "total_professions": len(available_professions)
    })

@app.get("/manage-test-sessions", response_class=HTMLResponse)
async def manage_test_sessions_page(request: Request):
    """Страница управления тест-сессиями - только для супер админа"""
    user = request.session.get("user")
    if not user:
        return RedirectResponse(url="/login")
    
    # Только супер админ может управлять тест-сессиями
    if not can_user_view_questions(user["role"]):
        return templates.TemplateResponse("access_denied.html", {
            "request": request,
            "user": user,
            "user_role_name": get_user_role_name(user["role"]),
            "message": "Управление тест-сессиями доступно только супер администратору",
            "allowed_roles": ["super_admin"]
        })
    
    # Получаем все тест-сессии
    all_test_sessions = get_all_test_sessions()
    
    return templates.TemplateResponse("manage_test_sessions.html", {
        "request": request,
        "user": user,
        "user_role_name": get_user_role_name(user["role"]),
        "test_sessions": all_test_sessions["test_sessions"],
        "stats": all_test_sessions["stats"],
        "total_sessions": len(all_test_sessions["test_sessions"])
    })

@app.get("/api/professions-with-questions")
async def get_professions_with_questions(request: Request):
    """Получение профессий с готовыми вопросами для создания тестов"""
    user = request.session.get("user")
    if not user:
        return JSONResponse({"error": "Не авторизован"}, status_code=401)
    
    try:
        professions = get_professions_with_questions()
        return JSONResponse({
            "success": True,
            "professions": professions,
            "total": len(professions)
        })
        
    except Exception as e:
        logger.error(f"❌ Ошибка получения профессий с вопросами: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/api/profession-questions-preview/{profession_id}")
async def get_profession_questions_preview(profession_id: str, request: Request):
    """Получение предварительного просмотра вопросов профессии по уровням"""
    user = request.session.get("user")
    if not user:
        return JSONResponse({"error": "Не авторизован"}, status_code=401)
    
    try:
        profession = get_profession_by_id(profession_id)
        if not profession:
            return JSONResponse({"error": "Профессия не найдена"}, status_code=404)
        
        questions = profession.get("questions", [])
        if not questions:
            return JSONResponse({"error": "У этой профессии нет вопросов"}, status_code=404)
        
        # Группируем вопросы по сложности
        questions_by_difficulty = {"easy": [], "medium": [], "hard": []}
        tags_stats = {}
        
        for question in questions:
            difficulty = question.get("difficulty", "medium")
            tag = question.get("tag", "General")
            
            if difficulty in questions_by_difficulty:
                questions_by_difficulty[difficulty].append(question)
            
            if tag not in tags_stats:
                tags_stats[tag] = 0
            tags_stats[tag] += 1
        
        # Статистика по уровням
        levels_stats = {
            "junior": {"available": len(questions_by_difficulty["easy"]), "difficulty": "easy"},
            "middle": {"available": len(questions_by_difficulty["medium"]), "difficulty": "medium"},
            "senior": {"available": len(questions_by_difficulty["hard"]), "difficulty": "hard"}
        }
        
        return JSONResponse({
            "success": True,
            "profession": {
                "id": profession["id"],
                "name": profession["real_name"],
                "specialization": profession.get("specialization", "Общая"),
                "bank_title": profession["bank_title"]
            },
            "tags": profession.get("tags", {}),
            "levels_stats": levels_stats,
            "tags_stats": tags_stats,
            "total_questions": len(questions)
        })
        
    except Exception as e:
        logger.error(f"❌ Ошибка получения предпросмотра вопросов: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/api/create-candidate-test")
async def create_candidate_test(request: Request, test_data: dict):
    """Создание нового теста для кандидата"""
    user = request.session.get("user")
    if not user:
        return JSONResponse({"error": "Не авторизован"}, status_code=401)
    
    # Проверяем права
    if not (can_user_view_questions(user["role"]) or can_user_create_profession(user["role"])):
        return JSONResponse({"error": "У вас нет прав для создания тестов"}, status_code=403)
    
    try:
        # Валидируем данные
        candidate_name = test_data.get("candidate_name", "").strip()
        candidate_iin = test_data.get("candidate_iin", "").strip()
        candidate_phone = test_data.get("candidate_phone", "").strip()
        candidate_email = test_data.get("candidate_email", "").strip()
        profession_id = test_data.get("profession_id")
        level = test_data.get("level")  # junior/middle/senior
        
        if not candidate_name:
            return JSONResponse({"error": "ФИО кандидата обязательно для заполнения"}, status_code=400)
        
        if not profession_id or not level:
            return JSONResponse({"error": "Выберите профессию и уровень"}, status_code=400)
        
        if level not in ["junior", "middle", "senior"]:
            return JSONResponse({"error": "Некорректный уровень"}, status_code=400)
        
        # Получаем профессию
        profession = get_profession_by_id(profession_id)
        if not profession or not profession.get("questions"):
            return JSONResponse({"error": "Профессия не найдена или у неё нет вопросов"}, status_code=404)
        
        # Создаем тест-сессию
        test_session = await create_test_session(test_data, profession, user)
        
        logger.info(f"👤 Тест-сессия создана для {candidate_name} (ID: {test_session['test_session_id']}) пользователем {user['name']}")
        
        return JSONResponse({
            "success": True,
            "test_session_id": test_session["test_session_id"],
            "test_url": test_session["test_url"],
            "candidate_name": candidate_name,
            "questions_count": len(test_session["questions"]),
            "level": level,
            "message": f"Тест для {candidate_name} успешно создан!"
        })
        
    except Exception as e:
        logger.error(f"❌ Ошибка создания тест-сессии: {e}")
        return JSONResponse({"error": f"Ошибка создания теста: {str(e)}"}, status_code=500)

@app.get("/api/test-sessions-overview")
async def get_test_sessions_overview(request: Request):
    """Получение обзора всех тест-сессий"""
    user = request.session.get("user")
    if not user:
        return JSONResponse({"error": "Не авторизован"}, status_code=401)
    
    if not can_user_view_questions(user["role"]):
        return JSONResponse({"error": "Доступ запрещен"}, status_code=403)
    
    try:
        all_test_sessions = get_all_test_sessions()
        return JSONResponse({
            "success": True,
            "test_sessions": all_test_sessions["test_sessions"],
            "stats": all_test_sessions["stats"]
        })
        
    except Exception as e:
        logger.error(f"❌ Ошибка получения обзора тест-сессий: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/api/test-session/{session_id}")
async def get_test_session_details(session_id: str, request: Request):
    """Получение деталей тест-сессии"""
    user = request.session.get("user")
    if not user:
        return JSONResponse({"error": "Не авторизован"}, status_code=401)
    
    try:
        test_session = get_test_session_by_id(session_id)
        if not test_session:
            return JSONResponse({"error": "Тест-сессия не найдена"}, status_code=404)
        
        return JSONResponse({
            "success": True,
            "test_session": test_session
        })
        
    except Exception as e:
        logger.error(f"❌ Ошибка получения тест-сессии {session_id}: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

@app.delete("/api/test-session/{session_id}")
async def delete_test_session(session_id: str, request: Request):
    """Удаление тест-сессии - только для супер админа"""
    user = request.session.get("user")
    if not user:
        return JSONResponse({"error": "Не авторизован"}, status_code=401)
    
    # Только супер админ может удалять тест-сессии
    if user["role"] != "super_admin":
        return JSONResponse({"error": "Доступ запрещен"}, status_code=403)
    
    try:
        success = await delete_test_session_by_id(session_id)
        
        if success:
            logger.info(f"🗑️ Тест-сессия удалена: {session_id} пользователем {user['name']}")
            return JSONResponse({
                "success": True,
                "message": "Тест-сессия успешно удалена"
            })
        else:
            return JSONResponse({"error": "Тест-сессия не найдена"}, status_code=404)
        
    except Exception as e:
        logger.error(f"❌ Ошибка удаления тест-сессии {session_id}: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)
    
    
@app.post("/api/submit-test")
async def submit_test_results(request: Request):
    """Отправка результатов теста кандидатом"""
    try:
        data = await request.json()
        session_id = data.get('session_id')
        answers = data.get('answers', [])
        time_spent = data.get('time_spent', 0)
        completed_at = data.get('completed_at')
        security_stats = data.get('security_stats', {})
        
        if not session_id:
            return {"status": "error", "message": "session_id обязателен"}
        
        # Загружаем данные тест-сессий
        sessions_file = DATA_DIR / "test_sessions.json"
        
        if not sessions_file.exists():
            return {"status": "error", "message": "Файл тест-сессий не найден"}
        
        with open(sessions_file, 'r', encoding='utf-8') as f:
            sessions_data = json.load(f)
        
        # Находим нужную сессию
        session_found = False
        for session in sessions_data.get("test_sessions", []):
            if session["test_session_id"] == session_id:
                # Рассчитываем результаты
                results = calculate_test_results(session["questions"], answers)
                
                # Генерируем рекомендации через ИИ
                recommendations = await generate_candidate_recommendations(session, results)
                results["recommendations"] = recommendations
                
                # Обновляем данные сессии
                session["answers"] = answers
                session["time_spent"] = time_spent
                session["completed_at"] = completed_at
                session["status"] = "completed"
                session["started_at"] = session.get("started_at") or completed_at
                session["results"] = results
                session["security_stats"] = security_stats
                
                session_found = True
                break
        
        if not session_found:
            return {"status": "error", "message": "Тест-сессия не найдена"}
        
        # Сохраняем обновленные данные
        with open(sessions_file, 'w', encoding='utf-8') as f:
            json.dump(sessions_data, f, ensure_ascii=False, indent=2)
        
        return {
            "status": "success", 
            "message": "Результаты сохранены",
            "results": results
        }
        
    except Exception as e:
        print(f"Ошибка сохранения результатов: {e}")
        return {"status": "error", "message": str(e)}


def calculate_test_results(questions: list, answers: list) -> dict:
    """Рассчитывает результаты тестирования с оценкой и анализом по категориям"""
    if not questions or not answers:
        return {
            "correct_answers": 0,
            "total_questions": len(questions) if questions else 0,
            "answered_questions": 0,
            "percentage": 0.0,
            "grade": "F",
            "grade_text": "Неудовлетворительно",
            "category_breakdown": {}
        }
    
    correct_answers = 0
    answered_questions = 0
    category_stats = {}
    
    for i, question in enumerate(questions):
        category = question.get("category", "General")
        difficulty = question.get("difficulty", "medium")
        
        # Инициализируем статистику по категории
        if category not in category_stats:
            category_stats[category] = {
                "total": 0,
                "correct": 0,
                "answered": 0,
                "questions": []
            }
        
        category_stats[category]["total"] += 1
        
        if i < len(answers) and answers[i] is not None:
            answered_questions += 1
            category_stats[category]["answered"] += 1
            
            # Получаем правильный ответ (полный текст)
            correct_answer_text = question.get("correct_answer", "")
            options = question.get("options", [])
            
            # Находим индекс правильного ответа в массиве options
            correct_index = -1
            for idx, option in enumerate(options):
                if option == correct_answer_text:
                    correct_index = idx
                    break
            
            # Проверяем ответ пользователя
            is_correct = correct_index != -1 and answers[i] == correct_index
            if is_correct:
                correct_answers += 1
                category_stats[category]["correct"] += 1
            
            # Сохраняем информацию о вопросе для рекомендаций
            category_stats[category]["questions"].append({
                "question": question.get("question", ""),
                "difficulty": difficulty,
                "tag": question.get("tag", ""),
                "is_correct": is_correct,
                "user_answer": options[answers[i]] if 0 <= answers[i] < len(options) else "Нет ответа",
                "correct_answer": correct_answer_text
            })
    
    # Рассчитываем процент и оценку
    percentage = (correct_answers / answered_questions * 100) if answered_questions > 0 else 0.0
    grade_info = calculate_grade(percentage)
    
    # Формируем разбивку по категориям
    category_breakdown = {}
    for category, stats in category_stats.items():
        if stats["answered"] > 0:
            cat_percentage = (stats["correct"] / stats["answered"]) * 100
            category_breakdown[category] = {
                "correct": stats["correct"],
                "total": stats["answered"],
                "percentage": round(cat_percentage, 1),
                "grade": calculate_grade(cat_percentage)["grade"]
            }
    
    return {
        "correct_answers": correct_answers,
        "total_questions": len(questions),
        "answered_questions": answered_questions,
        "percentage": round(percentage, 1),
        "grade": grade_info["grade"],
        "grade_text": grade_info["text"],
        "grade_color": grade_info["color"],
        "category_breakdown": category_breakdown,
        "detailed_stats": category_stats  # Для генерации рекомендаций
    }


def calculate_grade(percentage: float) -> dict:
    """Рассчитывает оценку на основе процента правильных ответов"""
    if percentage >= 85:
        return {"grade": "A", "text": "Отлично", "color": "#1DB584"}
    elif percentage >= 70:
        return {"grade": "B", "text": "Хорошо", "color": "#059669"}
    elif percentage >= 50:
        return {"grade": "C", "text": "Удовлетворительно", "color": "#F59E0B"}
    elif percentage >= 30:
        return {"grade": "D", "text": "Слабо", "color": "#F97316"}
    else:
        return {"grade": "F", "text": "Неудовлетворительно", "color": "#EF4444"}


async def generate_candidate_recommendations(test_session: dict, results: dict) -> str:
    """Генерирует персональные рекомендации для кандидата через ИИ"""
    try:
        # Подготавливаем данные для ИИ
        candidate = test_session.get("candidate", {})
        profession = test_session.get("profession", {})
        level = test_session.get("level", "")
        
        # Анализируем слабые места
        weak_categories = []
        strong_categories = []
        
        for category, breakdown in results.get("category_breakdown", {}).items():
            if breakdown["percentage"] < 70:
                weak_categories.append(f"{category} ({breakdown['percentage']}%)")
            elif breakdown["percentage"] >= 85:
                strong_categories.append(f"{category} ({breakdown['percentage']}%)")
        
        # Формируем промпт для ИИ
        prompt = f"""
Создай персональные рекомендации для кандидата на основе результатов тестирования.

ДАННЫЕ КАНДИДАТА:
- ФИО: {candidate.get('full_name', 'Не указано')}
- Должность: {profession.get('name', 'Не указано')} ({profession.get('specialization', '')})
- Уровень: {level.capitalize()}

РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ:
- Общий результат: {results.get('percentage', 0)}% (Оценка: {results.get('grade', 'F')})
- Правильных ответов: {results.get('correct_answers', 0)}/{results.get('answered_questions', 0)}

СИЛЬНЫЕ СТОРОНЫ: {', '.join(strong_categories) if strong_categories else 'Не выявлены'}
СЛАБЫЕ ОБЛАСТИ: {', '.join(weak_categories) if weak_categories else 'Не выявлены'}

ЗАДАЧА:
Создай мотивирующие и конструктивные рекомендации на русском языке (максимум 200 слов).

СТРУКТУРА ОТВЕТА:
1. Краткая оценка результата (1-2 предложения)
2. Что получается хорошо (если есть сильные стороны)
3. Что стоит улучшить (конкретные области)
4. 2-3 практических совета для развития

ТОН: Поддерживающий, профессиональный, мотивирующий
"""

        # Вызываем ИИ (используем OpenAI API как в других агентах)
        import openai
        openai.api_key = OPENAI_API_KEY
        
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Ты опытный HR-специалист, который дает конструктивную обратную связь кандидатам после тестирования."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.7
        )
        
        recommendations = response.choices[0].message.content.strip()
        logger.info(f"✅ Рекомендации сгенерированы для {candidate.get('full_name', 'кандидата')}")
        
        return recommendations
        
    except Exception as e:
        logger.error(f"❌ Ошибка генерации рекомендаций: {e}")
        
        # Fallback рекомендации на основе оценки
        grade = results.get("grade", "F")
        percentage = results.get("percentage", 0)
        
        fallback_recommendations = {
            "A": f"Превосходный результат! Вы показали отличные знания в области {profession.get('name', 'выбранной специальности')}. Продолжайте развиваться в этом направлении и делитесь знаниями с коллегами.",
            
            "B": f"Хороший результат! У вас есть solid понимание основ {profession.get('name', 'специальности')}. Рекомендуем углубить знания в слабых областях и продолжать практическое применение навыков.",
            
            "C": f"Удовлетворительный результат. Базовые знания присутствуют, но есть пространство для роста. Сосредоточьтесь на изучении основных концепций и регулярной практике.",
            
            "D": f"Результат показывает необходимость дополнительной подготовки. Рекомендуем пройти базовые курсы по {profession.get('name', 'специальности')} и получить практический опыт.",
            
            "F": f"Результат указывает на значительные пробелы в знаниях. Рекомендуем начать с изучения основ {profession.get('name', 'специальности')} через курсы и практические задания."
        }
        
        return fallback_recommendations.get(grade, "Продолжайте развивать свои навыки и знания в выбранной области.")
    
    
@app.get("/api/test-session-answers/{session_id}")
async def get_test_session_answers(session_id: str, request: Request):
    """Получение детальных ответов кандидата"""
    user = request.session.get("user")
    if not user:
        return JSONResponse({"error": "Не авторизован"}, status_code=401)
    
    if not can_user_view_questions(user["role"]):
        return JSONResponse({"error": "Доступ запрещен"}, status_code=403)
    
    try:
        test_session = get_test_session_by_id(session_id)
        if not test_session:
            return JSONResponse({"error": "Тест-сессия не найдена"}, status_code=404)
        
        if test_session.get("status") != "completed":
            return JSONResponse({"error": "Тест еще не завершен"}, status_code=400)
        
        questions = test_session.get("questions", [])
        answers = test_session.get("answers", [])
        
        # Формируем детальную информацию об ответах
        answers_details = []
        
        for i, question in enumerate(questions):
            options = question.get("options", [])
            correct_answer_text = question.get("correct_answer", "")
            
            # Ответ пользователя
            user_answer_index = answers[i] if i < len(answers) and answers[i] is not None else None
            user_answer_text = options[user_answer_index] if user_answer_index is not None and 0 <= user_answer_index < len(options) else None
            
            # Правильный ответ
            correct_index = -1
            for idx, option in enumerate(options):
                if option == correct_answer_text:
                    correct_index = idx
                    break
            
            # Проверяем правильность
            is_correct = user_answer_index is not None and user_answer_index == correct_index
            
            answers_details.append({
                "question": question.get("question", ""),
                "options": options,
                "user_answer": user_answer_text,
                "correct_answer": correct_answer_text,
                "is_correct": is_correct,
                "difficulty": question.get("difficulty", "medium"),
                "category": question.get("category", "General")
            })
        
        return JSONResponse({
            "success": True,
            "test_session": test_session,
            "answers_details": answers_details
        })
        
    except Exception as e:
        logger.error(f"❌ Ошибка получения ответов {session_id}: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

# === СТРАНИЦА ПРОХОЖДЕНИЯ ТЕСТА ===

@app.get("/take-test/{session_id}", response_class=HTMLResponse)
async def take_test_page(session_id: str, request: Request):
    """Страница прохождения теста кандидатом"""
    try:
        test_session = get_test_session_by_id(session_id)
        if not test_session:
            return templates.TemplateResponse("test_not_found.html", {
                "request": request,
                "session_id": session_id
            })
        
        # Проверяем статус тест-сессии
        if test_session.get("status") == "completed":
            return templates.TemplateResponse("test_completed.html", {
                "request": request,
                "test_session": test_session
            })
        
        return templates.TemplateResponse("take_test.html", {
            "request": request,
            "test_session": test_session,
            "candidate": test_session.get("candidate", {}),
            "profession": test_session.get("profession", {}),
            "questions": test_session.get("questions", []),
            "total_questions": len(test_session.get("questions", []))
        })
        
    except Exception as e:
        logger.error(f"❌ Ошибка загрузки теста {session_id}: {e}")
        return templates.TemplateResponse("test_error.html", {
            "request": request,
            "error": str(e)
        })

# === ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ===

def get_professions_with_questions() -> List[Dict[str, Any]]:
    """Получение профессий с готовыми вопросами"""
    try:
        records_file = DATA_DIR / "profession_records.json"
        
        if not records_file.exists():
            return []
        
        with open(records_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        professions_with_questions = []
        
        for record in data.get("profession_records", []):
            if (record.get("status") == "questions_generated" and 
                record.get("questions") and len(record["questions"]) > 0):
                
                questions = record["questions"]
                questions_by_difficulty = {"easy": 0, "medium": 0, "hard": 0}
                
                for question in questions:
                    difficulty = question.get("difficulty", "medium")
                    if difficulty in questions_by_difficulty:
                        questions_by_difficulty[difficulty] += 1
                
                professions_with_questions.append({
                    "id": record["id"],
                    "name": record["real_name"],
                    "specialization": record.get("specialization", "Общая"),
                    "bank_title": record["bank_title"],
                    "department": record.get("department", ""),
                    "questions_count": len(questions),
                    "questions_by_difficulty": questions_by_difficulty,
                    "tags": record.get("tags", {}),
                    "updated_at": record.get("questions_generated_at")
                })
        
        return professions_with_questions
        
    except Exception as e:
        logger.error(f"❌ Ошибка получения профессий с вопросами: {e}")
        return []

async def create_test_session(test_data: Dict[str, Any], profession: Dict[str, Any], user: Dict[str, Any]) -> Dict[str, Any]:
    """Создание новой тест-сессии для кандидата"""
    try:
        # Загружаем существующие тест-сессии
        sessions_file = DATA_DIR / "test_sessions.json"
        
        if sessions_file.exists():
            with open(sessions_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            data = {"test_sessions": []}
        
        # Создаем уникальный ID сессии
        session_id = str(uuid.uuid4())
        
        # Отбираем вопросы для теста по уровню и тегам
        selected_questions = select_questions_by_level_and_tags(
            profession, 
            test_data["level"]
        )
        
        # Создаем тест-сессию
        test_session = {
            "test_session_id": session_id,
            "candidate": {
                "full_name": test_data.get("candidate_name", ""),
                "iin": test_data.get("candidate_iin", ""),
                "phone": test_data.get("candidate_phone", ""),
                "email": test_data.get("candidate_email", "")
            },
            "profession": {
                "id": profession["id"],
                "name": profession["real_name"],
                "specialization": profession.get("specialization", "Общая"),
                "bank_title": profession["bank_title"]
            },
            "level": test_data["level"],
            "questions": selected_questions,
            "questions_count": len(selected_questions),
            "test_url": f"http://localhost:8002/take-test/{session_id}",
            "created_by": user["email"],
            "created_at": datetime.now().isoformat() + "Z",
            "status": "pending",  # pending/in_progress/completed
            "started_at": None,
            "completed_at": None,
            "results": None,
            "answers": []
        }
        
        data["test_sessions"].append(test_session)
        
        # Сохраняем
        with open(sessions_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        return test_session
        
    except Exception as e:
        logger.error(f"❌ Ошибка создания тест-сессии: {e}")
        raise

def select_questions_by_level_and_tags(profession: Dict[str, Any], level: str, total_questions: int = 15) -> List[Dict[str, Any]]:
    """Отбор вопросов по уровню и весам тегов"""
    try:
        print(f"🔍 ОТЛАДКА: Выбор вопросов для {profession.get('real_name')} уровня {level}")
        
        # 1. Определяем сложность по уровню
        difficulty_map = {
            "junior": "easy",
            "middle": "medium",
            "senior": "hard"
        }
        target_difficulty = difficulty_map.get(level, "medium")
        
        # 2. Фильтруем вопросы по сложности
        all_questions = profession.get("questions", [])
        questions_by_difficulty = [q for q in all_questions 
                                 if q.get("difficulty") == target_difficulty]
        
        print(f"🔍 ОТЛАДКА: Найдено {len(questions_by_difficulty)} вопросов сложности {target_difficulty}")
        
        if len(questions_by_difficulty) < total_questions:
            logger.warning(f"⚠️ Недостаточно вопросов уровня {target_difficulty}: {len(questions_by_difficulty)} из {total_questions}")
            # Берем все доступные вопросы этой сложности
            selected_questions = questions_by_difficulty.copy()
        else:
            # 3. Рассчитываем распределение по тегам на основе весов
            tags_weights = profession.get("tags", {})
            if not tags_weights:
                # Если нет весов, берем случайные вопросы
                selected_questions = random.sample(questions_by_difficulty, total_questions)
                print(f"🔍 ОТЛАДКА: Выбрано случайно (без тегов)")
            else:
                selected_questions = distribute_questions_by_tags(
                    questions_by_difficulty, 
                    tags_weights, 
                    total_questions
                )
        
        # 4. Перемешиваем вопросы
        random.shuffle(selected_questions)
        
        # 5. Убираем лишнее (если вдруг получилось больше)
        final_questions = selected_questions[:total_questions]
        
        # ОТЛАДКА: Проверяем дубликаты
        print(f"🔍 ОТЛАДКА: Выбрано {len(final_questions)} вопросов")
        question_ids = [q.get('id', 'no-id') for q in final_questions]
        unique_ids = set(question_ids)
        print(f"🔍 ОТЛАДКА: Уникальных ID: {len(unique_ids)}")
        
        if len(question_ids) != len(unique_ids):
            print(f"🚨 ДУБЛИКАТЫ НАЙДЕНЫ!")
            duplicates = [qid for qid in question_ids if question_ids.count(qid) > 1]
            print(f"🚨 Дублирующиеся ID: {set(duplicates)}")
        
        return final_questions
        
    except Exception as e:
        logger.error(f"❌ Ошибка отбора вопросов: {e}")
        return []

def distribute_questions_by_tags(questions: List[Dict[str, Any]], tags_weights: Dict[str, int], total_questions: int) -> List[Dict[str, Any]]:
    """Умное распределение вопросов по тегам с минимизацией искажений пропорций"""
    try:
        print(f"🧠 УМНОЕ РАСПРЕДЕЛЕНИЕ: {len(questions)} вопросов на {total_questions} мест")
        
        # 1. Группируем вопросы по тегам
        questions_by_tag = {}
        for question in questions:
            tag = question.get("tag", "General")
            if tag not in questions_by_tag:
                questions_by_tag[tag] = []
            questions_by_tag[tag].append(question)
        
        # 2. Рассчитываем идеальные пропорции (float)
        total_weight = sum(tags_weights.values())
        ideal_counts = {}
        for tag, weight in tags_weights.items():
            if tag in questions_by_tag:
                ideal_count = (weight / total_weight) * total_questions
                ideal_counts[tag] = ideal_count
                print(f"📊 {tag}: идеально {ideal_count:.2f}")
        
        # 3. Округляем до целых чисел
        tag_counts = {tag: round(ideal) for tag, ideal in ideal_counts.items()}
        print(f"🔄 После округления: {tag_counts} (сумма: {sum(tag_counts.values())})")
        
        # 4. УМНАЯ КОРРЕКЦИЯ по отклонениям от пропорций
        while sum(tag_counts.values()) != total_questions:
            current_sum = sum(tag_counts.values())
            
            # Считаем отклонения (фактическое - идеальное)  
            differences = {tag: tag_counts[tag] - ideal_counts[tag] 
                          for tag in tag_counts}
            
            if current_sum > total_questions:
                # Нужно убрать: ищем тег с наибольшим перебором
                tag_to_reduce = max(differences.keys(), key=lambda x: differences[x])
                if tag_counts[tag_to_reduce] > 0:  # Не уходим в минус
                    tag_counts[tag_to_reduce] -= 1
                    print(f"➖ Убрали 1 вопрос у '{tag_to_reduce}' (перебор: {differences[tag_to_reduce]:.2f})")
                else:
                    break
            else:
                # Нужно добавить: ищем тег с наибольшим недобором
                tag_to_increase = min(differences.keys(), key=lambda x: differences[x])
                max_available = len(questions_by_tag.get(tag_to_increase, []))
                if tag_counts[tag_to_increase] < max_available:  # Не превышаем доступное количество
                    tag_counts[tag_to_increase] += 1
                    print(f"➕ Добавили 1 вопрос к '{tag_to_increase}' (недобор: {differences[tag_to_increase]:.2f})")
                else:
                    break
        
        print(f"✅ Финальное распределение: {tag_counts} (сумма: {sum(tag_counts.values())})")
        
        # 5. Выбираем уникальные вопросы согласно плану
        selected_questions = []
        used_question_ids = set()
        
        for tag, count in tag_counts.items():
            if count > 0 and tag in questions_by_tag:
                available_questions = [
                    q for q in questions_by_tag[tag] 
                    if q.get("id") not in used_question_ids
                ]
                
                if available_questions:
                    actual_count = min(count, len(available_questions))
                    tag_questions = random.sample(available_questions, actual_count)
                    selected_questions.extend(tag_questions)
                    
                    # Помечаем как использованные
                    for q in tag_questions:
                        used_question_ids.add(q.get("id"))
                    
                    print(f"🏷️ '{tag}': выбрано {actual_count} из {len(available_questions)} доступных")
        
        # Финальная проверка на дубликаты
        question_ids = [q.get("id") for q in selected_questions]
        unique_ids = set(question_ids)
        
        if len(question_ids) != len(unique_ids):
            print(f"🚨 ОШИБКА: найдены дубликаты! {len(question_ids)} вопросов, {len(unique_ids)} уникальных")
        else:
            print(f"✅ ВСЕ УНИКАЛЬНО: {len(selected_questions)} вопросов выбрано")
        
        return selected_questions
        
    except Exception as e:
        logger.error(f"❌ Ошибка умного распределения: {e}")
        return random.sample(questions, min(total_questions, len(questions)))

def get_all_test_sessions() -> Dict[str, Any]:
    """Получение всех тест-сессий"""
    try:
        sessions_file = DATA_DIR / "test_sessions.json"
        
        if not sessions_file.exists():
            return {"test_sessions": [], "stats": {}}
        
        with open(sessions_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        test_sessions = data.get("test_sessions", [])
        
        # Статистика
        stats = {
            "total_sessions": len(test_sessions),
            "pending_sessions": len([s for s in test_sessions if s.get("status") == "pending"]),
            "in_progress_sessions": len([s for s in test_sessions if s.get("status") == "in_progress"]),
            "completed_sessions": len([s for s in test_sessions if s.get("status") == "completed"]),
            "by_level": Counter(s.get("level") for s in test_sessions),
            "by_profession": Counter(s.get("profession", {}).get("name") for s in test_sessions)
        }
        
        return {"test_sessions": test_sessions, "stats": stats}
        
    except Exception as e:
        logger.error(f"❌ Ошибка получения всех тест-сессий: {e}")
        return {"test_sessions": [], "stats": {}}

def get_test_session_by_id(session_id: str) -> Optional[Dict[str, Any]]:
    """Получение тест-сессии по ID"""
    try:
        sessions_file = DATA_DIR / "test_sessions.json"
        
        if not sessions_file.exists():
            return None
        
        with open(sessions_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for session in data.get("test_sessions", []):
            if session["test_session_id"] == session_id:
                return session
        
        return None
        
    except Exception as e:
        logger.error(f"❌ Ошибка получения тест-сессии {session_id}: {e}")
        return None

async def delete_test_session_by_id(session_id: str) -> bool:
    """Удаление тест-сессии по ID"""
    try:
        sessions_file = DATA_DIR / "test_sessions.json"
        
        if not sessions_file.exists():
            return False
        
        with open(sessions_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Ищем и удаляем тест-сессию
        sessions = data.get("test_sessions", [])
        initial_count = len(sessions)
        
        data["test_sessions"] = [session for session in sessions if session["test_session_id"] != session_id]
        
        if len(data["test_sessions"]) == initial_count:
            return False  # Сессия не найдена
        
        # Сохраняем изменения
        with open(sessions_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Ошибка удаления тест-сессии {session_id}: {e}")
        return False

# === ОБНОВЛЕНИЕ ФУНКЦИИ СТАТИСТИКИ ===

async def get_user_statistics(user: Dict[str, Any]) -> Dict[str, Any]:
    """Получение статистики для пользователя (обновленная версия с тест-сессиями)"""
    try:
        records_file = DATA_DIR / "profession_records.json"
        sessions_file = DATA_DIR / "test_sessions.json"
        
        stats = {
            "total_professions": 0,
            "created_by_user": 0,
            "pending_approval": 0,
            "approved": 0,
            "questions_generated": 0,
            "test_sessions_created": 0  # Новая статистика для тест-сессий
        }
        
        # Статистика профессий
        if records_file.exists():
            with open(records_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for record in data.get("profession_records", []):
                stats["total_professions"] += 1
                
                if record.get("created_by") == user["email"]:
                    stats["created_by_user"] += 1
                
                status = record.get("status", "")
                if status == "tags_generated":
                    stats["pending_approval"] += 1
                elif status == "approved_by_head":
                    stats["approved"] += 1
                elif status == "questions_generated":
                    stats["questions_generated"] += 1
        
        # Статистика тест-сессий
        if sessions_file.exists():
            with open(sessions_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Считаем все тест-сессии или созданные пользователем (в зависимости от роли)
            if user["role"] == "super_admin":
                stats["test_sessions_created"] = len(data.get("test_sessions", []))
            else:
                stats["test_sessions_created"] = len([s for s in data.get("test_sessions", []) 
                                                   if s.get("created_by") == user["email"]])
        
        return stats
        
    except Exception as e:
        logger.error(f"❌ Ошибка получения статистики: {e}")
        return {}

# === ЗАПУСК СЕРВЕРА ===

if __name__ == "__main__":
    print("🚀 Запуск HR Admin Panel v2.0")
    print(f"🌐 URL: http://localhost:{APP_PORT}")
    print(f"📁 Данные: {DATA_DIR}")
    print(f"🤖 ИИ агенты: 4 инициализированы")
    print(f"⏰ Планировщик: генерация вопросов в 00:00")
    
    uvicorn.run(
        "main:app",
        host=APP_HOST,
        port=APP_PORT,
        reload=DEBUG,
        log_level="info"
    )