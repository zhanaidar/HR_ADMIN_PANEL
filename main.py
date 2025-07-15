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