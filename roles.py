"""
Роли и права пользователей HR Admin Panel v2.0
Централизованное управление доступом
"""

# Импортируем функции для работы с пользователями
from users import find_user, find_user_by_email, get_department_head

# Названия ролей
ROLE_NAMES = {
    'super_admin': 'Супер Администратор',
    'hr_head_admin': 'HR Директор',
    'hr_admin': 'HR Специалист', 
    'head_admin': 'Начальник отдела'
}

# Права ролей
ROLE_PERMISSIONS = {
    'super_admin': {
        # Профессии
        'canCreateProfessions': True,
        'canEditAllProfessions': True,
        'canDeleteProfessions': True,
        'canApproveProfessions': True,
        'canReturnToHR': True,
        
        # Теги
        'canEditAllTags': True,
        'canViewAllTags': True,
        
        # Вопросы (только супер админ!)
        'canViewQuestions': True,
        'canManageQuestions': True,
        'canGenerateQuestions': True,
        
        # Администрирование
        'canManageUsers': True,
        'canViewStats': True,
        'canViewAllDepartments': True
    },
    
    'hr_head_admin': {
        # Профессии
        'canCreateProfessions': True,  # Главное право HR директора
        'canEditOwnProfessions': True,
        'canDeleteProfessions': False,
        'canApproveProfessions': False,
        'canReturnToHR': False,
        
        # Теги
        'canEditHRTags': True,
        'canViewAllTags': True,
        
        # Вопросы
        'canViewQuestions': False,  # НЕ может видеть вопросы
        'canManageQuestions': False,
        'canGenerateQuestions': False,
        
        # Администрирование
        'canManageUsers': False,
        'canViewStats': True,
        'canViewAllDepartments': True
    },
    
    'hr_admin': {
        # Профессии
        'canCreateProfessions': False,  # НЕ может создавать
        'canEditOwnProfessions': False,
        'canDeleteProfessions': False,
        'canApproveProfessions': False,
        'canReturnToHR': False,
        
        # Теги
        'canEditAllTags': False,
        'canViewAllTags': True,  # Может просматривать
        
        # Вопросы
        'canViewQuestions': False,
        'canManageQuestions': False,
        'canGenerateQuestions': False,
        
        # Администрирование
        'canManageUsers': False,
        'canViewStats': True,  # Может видеть статистику
        'canViewAllDepartments': True
    },
    
    'head_admin': {
        # Профессии
        'canCreateProfessions': False,  # НЕ может создавать
        'canEditOwnProfessions': False,
        'canDeleteProfessions': False,
        'canApproveProfessions': True,  # Главное право начальника
        'canReturnToHR': True,          # Может возвращать на доработку
        
        # Теги
        'canEditOwnTags': True,         # Может редактировать теги своего отдела
        'canViewOwnTags': True,
        
        # Вопросы
        'canViewQuestions': False,      # НЕ может видеть вопросы
        'canManageQuestions': False,
        'canGenerateQuestions': False,
        
        # Администрирование
        'canManageUsers': False,
        'canViewStats': True,
        'canViewOwnDepartment': True    # Видит только свой отдел
    }
}

# Статусы профессий и кто может их менять
PROFESSION_STATUSES = {
    'created_by_hr': {
        'name': 'Создана HR',
        'description': 'Профессия создана HR директором',
        'can_edit': ['super_admin', 'hr_head_admin'],
        'next_status': 'tags_generated'
    },
    'tags_generated': {
        'name': 'Теги сгенерированы ИИ',
        'description': 'ИИ сгенерировал теги, ожидает проверки начальника',
        'can_edit': ['super_admin'],
        'next_status': ['approved_by_head', 'returned_to_hr']
    },
    'returned_to_hr': {
        'name': 'Возвращена на доработку',
        'description': 'Начальник вернул профессию HR на исправление',
        'can_edit': ['super_admin', 'hr_head_admin'],
        'next_status': 'tags_generated'
    },
    'approved_by_head': {
        'name': 'Утверждена начальником',
        'description': 'Начальник отдела утвердил профессию и теги',
        'can_edit': ['super_admin'],
        'next_status': 'questions_generated'
    },
    'questions_generated': {
        'name': 'Вопросы сгенерированы',
        'description': 'ИИ сгенерировал вопросы для тестирования',
        'can_edit': ['super_admin'],
        'next_status': 'active'
    },
    'active': {
        'name': 'Активна',
        'description': 'Профессия готова к использованию',
        'can_edit': ['super_admin'],
        'next_status': None
    }
}

# === ФУНКЦИИ ПРОВЕРКИ ПРАВ ===

# Функции find_user уже импортированы из users.py

def get_user_role_name(user_role: str) -> str:
    """Получение названия роли пользователя"""
    return ROLE_NAMES.get(user_role, user_role)

def has_permission(user_role: str, permission: str) -> bool:
    """Универсальная проверка прав"""
    return ROLE_PERMISSIONS.get(user_role, {}).get(permission, False)

# Конкретные проверки прав
def can_user_create_profession(user_role: str) -> bool:
    """Может ли пользователь создавать профессии"""
    return has_permission(user_role, 'canCreateProfessions')

def can_user_approve_profession(user_role: str) -> bool:
    """Может ли пользователь утверждать профессии"""
    return has_permission(user_role, 'canApproveProfessions')

def can_user_return_to_hr(user_role: str) -> bool:
    """Может ли пользователь возвращать на доработку"""
    return has_permission(user_role, 'canReturnToHR')

def can_user_view_questions(user_role: str) -> bool:
    """Может ли пользователь просматривать вопросы"""
    return has_permission(user_role, 'canViewQuestions')

def can_user_edit_tags(user_role: str, profession_department: str, user_department: str) -> bool:
    """Может ли пользователь редактировать теги"""
    
    # Супер админ может всё
    if has_permission(user_role, 'canEditAllTags'):
        return True
    
    # HR директор может редактировать HR теги
    if has_permission(user_role, 'canEditHRTags') and profession_department == 'HR':
        return True
    
    # Начальник отдела может редактировать теги своего отдела
    if has_permission(user_role, 'canEditOwnTags') and profession_department == user_department:
        return True
    
    return False

def can_user_access_profession(user: dict, profession: dict) -> bool:
    """Может ли пользователь получить доступ к профессии"""
    user_role = user['role']
    
    # Супер админ может всё
    if user_role == 'super_admin':
        return True
    
    # HR директор может видеть всё
    if user_role == 'hr_head_admin':
        return True
    
    # Начальник отдела может видеть профессии своего отдела
    if user_role == 'head_admin':
        profession_dept = profession.get('department', '')
        user_dept = user.get('department', '')
        return profession_dept == f"{user_dept} Department"
    
    # HR админ может видеть всё (только для чтения)
    if user_role == 'hr_admin':
        return True
    
    # Создатель может видеть свои профессии
    if profession.get('created_by') == user['email']:
        return True
    
    return False

def can_user_change_status(user_role: str, current_status: str, new_status: str) -> bool:
    """Может ли пользователь изменить статус профессии"""
    
    # Проверяем есть ли права на редактирование текущего статуса
    status_info = PROFESSION_STATUSES.get(current_status, {})
    allowed_editors = status_info.get('can_edit', [])
    
    if user_role not in allowed_editors:
        return False
    
    # Проверяем валидность перехода
    allowed_next = status_info.get('next_status')
    if isinstance(allowed_next, list):
        return new_status in allowed_next
    elif isinstance(allowed_next, str):
        return new_status == allowed_next
    
    return False

# === ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ===

def get_user_permissions(user_role: str) -> dict:
    """Получить все права пользователя"""
    return ROLE_PERMISSIONS.get(user_role, {})

def get_allowed_statuses_for_user(user_role: str) -> list:
    """Получить список статусов которые может редактировать пользователь"""
    allowed = []
    for status, info in PROFESSION_STATUSES.items():
        if user_role in info.get('can_edit', []):
            allowed.append(status)
    return allowed

def get_role_description(user_role: str) -> str:
    """Получить описание роли"""
    descriptions = {
        'super_admin': 'Полный доступ ко всем функциям системы',
        'hr_head_admin': 'Создание профессий, управление HR процессами', 
        'hr_admin': 'Просмотр профессий и статистики',
        'head_admin': 'Утверждение профессий и тегов своего отдела'
    }
    return descriptions.get(user_role, 'Стандартный пользователь')

print("✅ Система ролей и прав загружена")
print(f"👥 Ролей: {len(ROLE_NAMES)}")
print(f"📊 Статусов: {len(PROFESSION_STATUSES)}")