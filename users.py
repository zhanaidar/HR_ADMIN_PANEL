"""
База пользователей HR Admin Panel v2.0
Централизованное управление пользователями
"""

# База пользователей системы
USERS_DB = [
    {
        "id": 1,
        "email": "JanaydarK@halykbank.kz",
        "name": "Жанайдар К.",
        "role": "super_admin",
        "password": "123456",
        "department": "IT",
        "phone": "+7 777 123 4567",
        "position": "CTO",
        "created_at": "2024-01-01T00:00:00Z",
        "last_login": None,
        "is_active": True
    },
    {
        "id": 2,
        "email": "DzhamilyaBa@halykbank.kz", 
        "name": "Джамиля Б.",
        "role": "hr_head_admin",
        "password": "123456",
        "department": "HR",
        "phone": "+7 777 234 5678",
        "position": "HR Director",
        "created_at": "2024-01-01T00:00:00Z",
        "last_login": None,
        "is_active": True
    },
    {
        "id": 3,
        "email": "FiryzaH@halykbank.kz",
        "name": "Фируза Х.",
        "role": "hr_admin", 
        "password": "123456",
        "department": "HR",
        "phone": "+7 777 345 6789",
        "position": "HR Specialist",
        "created_at": "2024-01-01T00:00:00Z",
        "last_login": None,
        "is_active": True
    },
    {
        "id": 4,
        "email": "AlmasNy@halykbank.kz",
        "name": "Алмас Н.",
        "role": "head_admin",
        "password": "123456", 
        "department": "IT",
        "phone": "+7 777 456 7890",
        "position": "IT Department Head",
        "created_at": "2024-01-01T00:00:00Z",
        "last_login": None,
        "is_active": True
    }
]

# === ФУНКЦИИ РАБОТЫ С ПОЛЬЗОВАТЕЛЯМИ ===

def find_user(email: str, password: str) -> dict:
    """Поиск пользователя по email и паролю"""
    for user in USERS_DB:
        if (user["email"].lower() == email.lower() and 
            user["password"] == password and 
            user["is_active"]):
            return user
    return None

def find_user_by_email(email: str) -> dict:
    """Поиск пользователя только по email"""
    for user in USERS_DB:
        if user["email"].lower() == email.lower():
            return user
    return None

def find_user_by_id(user_id: int) -> dict:
    """Поиск пользователя по ID"""
    for user in USERS_DB:
        if user["id"] == user_id:
            return user
    return None

def get_users_by_role(role: str) -> list:
    """Получить всех пользователей с определенной ролью"""
    return [user for user in USERS_DB if user["role"] == role and user["is_active"]]

def get_users_by_department(department: str) -> list:
    """Получить всех пользователей из определенного департамента"""
    return [user for user in USERS_DB if user["department"] == department and user["is_active"]]

def get_department_head(department: str) -> dict:
    """Получить начальника департамента"""
    # Ищем head_admin в этом департаменте
    heads = [user for user in USERS_DB 
             if user["role"] == "head_admin" 
             and user["department"] == department 
             and user["is_active"]]
    
    return heads[0] if heads else None

def get_hr_directors() -> list:
    """Получить всех HR директоров"""
    return get_users_by_role("hr_head_admin")

def get_super_admins() -> list:
    """Получить всех супер админов"""
    return get_users_by_role("super_admin")

def update_last_login(user_id: int, login_time: str) -> bool:
    """Обновить время последнего входа"""
    try:
        for user in USERS_DB:
            if user["id"] == user_id:
                user["last_login"] = login_time
                return True
        return False
    except Exception:
        return False

def deactivate_user(user_id: int) -> bool:
    """Деактивировать пользователя"""
    try:
        for user in USERS_DB:
            if user["id"] == user_id:
                user["is_active"] = False
                return True
        return False
    except Exception:
        return False

def activate_user(user_id: int) -> bool:
    """Активировать пользователя"""
    try:
        for user in USERS_DB:
            if user["id"] == user_id:
                user["is_active"] = True
                return True
        return False
    except Exception:
        return False

def get_active_users_count() -> int:
    """Получить количество активных пользователей"""
    return len([user for user in USERS_DB if user["is_active"]])

def get_users_statistics() -> dict:
    """Получить статистику пользователей"""
    stats = {
        "total_users": len(USERS_DB),
        "active_users": get_active_users_count(),
        "by_role": {},
        "by_department": {}
    }
    
    # Статистика по ролям
    for user in USERS_DB:
        if user["is_active"]:
            role = user["role"]
            dept = user["department"]
            
            stats["by_role"][role] = stats["by_role"].get(role, 0) + 1
            stats["by_department"][dept] = stats["by_department"].get(dept, 0) + 1
    
    return stats

# === ВАЛИДАЦИЯ ДАННЫХ ===

def validate_email(email: str) -> bool:
    """Проверка корректности email"""
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def validate_password(password: str) -> tuple:
    """Проверка силы пароля"""
    if len(password) < 6:
        return False, "Пароль должен содержать минимум 6 символов"
    
    # В реальной системе добавить более строгие требования
    return True, "Пароль корректен"

def is_email_available(email: str, exclude_user_id: int = None) -> bool:
    """Проверка доступности email"""
    for user in USERS_DB:
        if (user["email"].lower() == email.lower() and 
            user["id"] != exclude_user_id):
            return False
    return True

# === ДЕМО ДАННЫЕ ===

def get_demo_credentials() -> list:
    """Получить список демо аккаунтов для страницы логина"""
    demo_accounts = []
    
    for user in USERS_DB:
        if user["is_active"]:
            demo_accounts.append({
                "email": user["email"],
                "password": user["password"],
                "name": user["name"],
                "role": user["role"],
                "position": user["position"]
            })
    
    return demo_accounts

def get_role_color(role: str) -> str:
    """Получить цвет для роли (для UI)"""
    colors = {
        "super_admin": "text-red-700",
        "hr_head_admin": "text-blue-700", 
        "hr_admin": "text-green-700",
        "head_admin": "text-purple-700"
    }
    return colors.get(role, "text-gray-700")

# === ИНИЦИАЛИЗАЦИЯ ===

print("✅ База пользователей загружена")
print(f"👤 Всего пользователей: {len(USERS_DB)}")
print(f"✅ Активных пользователей: {get_active_users_count()}")
print(f"📊 Статистика: {get_users_statistics()}")