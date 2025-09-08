"""
Модуль прокторинга для HR Tech Eval System
Содержит системы аудио и видео прокторинга с ИИ
"""

from .audio_proctoring import AudioProctorSystem, get_audio_proctor

__all__ = ['AudioProctorSystem', 'get_audio_proctor']