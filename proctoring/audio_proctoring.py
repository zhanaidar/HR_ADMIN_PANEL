"""
AudioProctorSystem - Система аудио прокторинга с ИИ и умным кешированием
Модель Whisper загружается ОДИН раз и остается в памяти
ИСПРАВЛЕНА ОБРАБОТКА WEBM БЕЗ FFMPEG
"""

import os
import json
import asyncio
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import tempfile
import base64
import io
import subprocess
import shutil

# Аудио обработка
import librosa
import soundfile as sf
import torch
from scipy.spatial.distance import cosine
from sklearn.preprocessing import StandardScaler

# Whisper для транскрипции
import whisper

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudioProctorSystem:
    """
    Система аудио прокторинга с калибровкой голоса кандидата
    Whisper модель загружается ОДИН раз и кешируется
    УЛУЧШЕНА ОБРАБОТКА WEBM ФАЙЛОВ
    """
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.audio_logs_dir = data_dir / "audio_logs"
        self.audio_logs_dir.mkdir(exist_ok=True)
        
        # Кешированные модели (загружаются один раз!)
        self.whisper_model = None
        self._whisper_loaded = False
        
        # Состояние системы
        self.is_calibrated = False
        self.candidate_voice_profile = None
        self.calibration_samples = []
        
        # Параметры
        self.config = {
            'sample_rate': 16000,
            'calibration_duration': 10,  # секунд
            'min_calibration_samples': 5,
            'similarity_threshold': 0.65,  # порог для определения "свой/чужой"
            'whisper_model': 'base',  # tiny=быстро, base=качественно
            'chunk_duration': 2.0,  # секунд на чанк
            'mfcc_features': 13,  # количество MFCC коэффициентов
        }
        
        # История анализа
        self.analysis_history = []
        
        # Проверяем доступность внешних инструментов
        self._check_external_tools()

            
    def _check_external_tools(self):
        """Проверка доступности внешних инструментов для конвертации"""
        # Сначала ищем в PATH
        self.ffmpeg_available = shutil.which('ffmpeg') is not None
        
        # Если не найден в PATH, пробуем прямой путь
        if not self.ffmpeg_available:
            ffmpeg_path = r"C:\ffmpeg\ffmpeg-2025-09-04-git-2611874a50-full_build\ffmpeg-2025-09-04-git-2611874a50-full_build\bin\ffmpeg.exe"
            self.ffmpeg_available = os.path.exists(ffmpeg_path)
            if self.ffmpeg_available:
                self.ffmpeg_executable = ffmpeg_path
        
        self.ffprobe_available = shutil.which('ffprobe') is not None
        
        if self.ffmpeg_available:
            logger.info("✅ FFmpeg найден - расширенная поддержка форматов включена")
        else:
            logger.info("⚠️ FFmpeg не найден - используем встроенные методы")
    
    async def initialize(self) -> bool:
        """Инициализация моделей (кеширование)"""
        try:
            logger.info("🤖 Инициализация AudioProctorSystem...")
            
            # Загружаем Whisper ОДИН раз
            if not self._whisper_loaded:
                logger.info(f"📥 Загрузка Whisper модели: {self.config['whisper_model']} (займет ~45 сек)")
                start_time = datetime.now()
                
                self.whisper_model = whisper.load_model(self.config['whisper_model'])
                
                load_time = (datetime.now() - start_time).total_seconds()
                logger.info(f"✅ Whisper модель загружена за {load_time:.1f} секунд")
                self._whisper_loaded = True
            else:
                logger.info("♻️ Whisper модель уже загружена (используем кеш)")
            
            logger.info("✅ AudioProctorSystem готов к работе")
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка инициализации: {e}")
            return False
    
    def detect_audio_format(self, audio_data: bytes) -> str:
        """Улучшенное определение формата аудио файла"""
        if len(audio_data) < 12:
            return '.unknown'
        
        # Читаем первые байты для определения формата
        header = audio_data[:12]
        
        # MP4/M4A сигнатуры
        if b'ftyp' in header or b'M4A ' in header or b'mp41' in header or b'mp42' in header:
            return '.mp4'
        
        # OGG сигнатура
        if header.startswith(b'OggS'):
            return '.ogg'
        
        # WebM сигнатуры (расширенная проверка)
        if (b'\x1a\x45\xdf\xa3' in header or  # EBML header
            b'webm' in header[:50] or
            b'matroska' in header[:50]):
            return '.webm'
        
        # WAV сигнатура
        if header.startswith(b'RIFF') and b'WAVE' in header:
            return '.wav'
        
        # FLAC сигнатура
        if header.startswith(b'fLaC'):
            return '.flac'
        
        # MP3 сигнатуры
        if header.startswith(b'ID3') or header.startswith(b'\xff\xfb'):
            return '.mp3'
        
        # По умолчанию считаем WebM (часто браузер не ставит правильные заголовки)
        logger.warning(f"🔍 Неизвестный формат, первые 12 байт: {header.hex()}")
        return '.webm'
    
    def preprocess_audio(self, audio_data: bytes, sample_rate: int = None) -> np.ndarray:
        """УЛУЧШЕННАЯ предобработка аудио данных с расширенной поддержкой WebM"""
        try:
            # Определяем тип файла по содержимому
            file_ext = self.detect_audio_format(audio_data)
            logger.info(f"🔍 Определен формат файла: {file_ext}")
            
            # Создаем временный файл с правильным расширением
            with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as tmp_file:
                tmp_file.write(audio_data)
                tmp_file_path = tmp_file.name
            
            audio = None
            success_method = None
            
            try:
                # МЕТОД 1: Прямая загрузка через librosa (лучше всего для MP4/WAV/FLAC)
                if file_ext in ['.mp4', '.wav', '.flac', '.mp3']:
                    try:
                        audio, sr = librosa.load(
                            tmp_file_path, 
                            sr=sample_rate or self.config['sample_rate'],
                            mono=True
                        )
                        success_method = f"librosa (native {file_ext})"
                        logger.info(f"✅ {success_method}: {len(audio)} сэмплов")
                    except Exception as e:
                        logger.warning(f"Librosa {file_ext} error: {e}")
                
                # МЕТОД 2: SoundFile для OGG и некоторых MP4
                if audio is None and file_ext in ['.ogg', '.mp4', '.wav']:
                    try:
                        audio_sf, sr = sf.read(tmp_file_path)
                        
                        # Конвертируем в моно если нужно
                        if len(audio_sf.shape) > 1:
                            audio_sf = np.mean(audio_sf, axis=1)
                        
                        # Ресемплим если нужно
                        target_sr = sample_rate or self.config['sample_rate']
                        if sr != target_sr:
                            audio = librosa.resample(audio_sf, orig_sr=sr, target_sr=target_sr)
                        else:
                            audio = audio_sf
                        
                        success_method = f"soundfile {file_ext}"
                        logger.info(f"✅ {success_method}: {len(audio)} сэмплов")
                    except Exception as e:
                        logger.warning(f"SoundFile {file_ext} error: {e}")
                
                # МЕТОД 3: PyDub с FFmpeg (если доступен)
                if audio is None and self.ffmpeg_available:
                    try:
                        from pydub import AudioSegment
                        
                        # Определяем формат для PyDub
                        format_map = {
                            '.webm': 'webm',
                            '.ogg': 'ogg', 
                            '.mp4': 'mp4',
                            '.wav': 'wav',
                            '.mp3': 'mp3'
                        }
                        
                        pydub_format = format_map.get(file_ext, 'webm')
                        
                        # Загружаем через PyDub
                        audio_segment = AudioSegment.from_file(tmp_file_path, format=pydub_format)
                        
                        # Конвертируем в нужный формат
                        audio_segment = audio_segment.set_frame_rate(self.config['sample_rate'])
                        audio_segment = audio_segment.set_channels(1)  # Mono
                        
                        # Получаем raw audio данные
                        raw_data = audio_segment.raw_data
                        audio = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0
                        
                        success_method = f"pydub+ffmpeg {file_ext}"
                        logger.info(f"✅ {success_method}: {len(audio)} сэмплов")
                    except Exception as e:
                        logger.warning(f"PyDub {file_ext} error: {e}")
                
                # МЕТОД 4: Прямая конвертация через FFmpeg (для WebM)
                if audio is None and self.ffmpeg_available and file_ext == '.webm':
                    try:
                        audio = self._convert_webm_with_ffmpeg(tmp_file_path)
                        if audio is not None and len(audio) > 0:
                            success_method = "ffmpeg direct conversion"
                            logger.info(f"✅ {success_method}: {len(audio)} сэмплов")
                    except Exception as e:
                        logger.warning(f"FFmpeg direct conversion error: {e}")
                
                # МЕТОД 5: WebM как OGG (простое переименование + библиотеки)
                if audio is None and file_ext == '.webm':
                    try:
                        # Создаем копию как OGG
                        ogg_path = tmp_file_path.replace('.webm', '.ogg')
                        shutil.copy2(tmp_file_path, ogg_path)
                        
                        # Пробуем загрузить как OGG
                        try:
                            audio, sr = librosa.load(ogg_path, sr=self.config['sample_rate'])
                            success_method = "webm as ogg (librosa)"
                            logger.info(f"✅ {success_method}: {len(audio)} сэмплов")
                        except:
                            try:
                                audio_sf, sr = sf.read(ogg_path)
                                if len(audio_sf.shape) > 1:
                                    audio_sf = np.mean(audio_sf, axis=1)
                                if sr != self.config['sample_rate']:
                                    audio = librosa.resample(audio_sf, orig_sr=sr, target_sr=self.config['sample_rate'])
                                else:
                                    audio = audio_sf
                                success_method = "webm as ogg (soundfile)"
                                logger.info(f"✅ {success_method}: {len(audio)} сэмплов")
                            except Exception as e2:
                                logger.warning(f"WebM as OGG failed: {e2}")
                        
                        # Удаляем временный OGG файл
                        try:
                            os.unlink(ogg_path)
                        except:
                            pass
                    except Exception as e:
                        logger.warning(f"WebM→OGG method failed: {e}")
                
                # МЕТОД 6: Попытка чтения как raw PCM (последний шанс)
                if audio is None and file_ext == '.webm':
                    try:
                        audio = self._try_raw_pcm_extraction(audio_data)
                        if audio is not None and len(audio) > 0:
                            success_method = "raw PCM extraction"
                            logger.info(f"✅ {success_method}: {len(audio)} сэмплов")
                    except Exception as e:
                        logger.warning(f"Raw PCM extraction failed: {e}")
            
            finally:
                # Очистка временных файлов
                temp_files = [
                    tmp_file_path,
                    tmp_file_path.replace('.webm', '.ogg'),
                    tmp_file_path.replace('.webm', '.wav'),
                    tmp_file_path + '.wav'
                ]
                
                for temp_file in temp_files:
                    if os.path.exists(temp_file):
                        try:
                            os.unlink(temp_file)
                        except:
                            pass
            
            # Проверяем результат
            if audio is None or len(audio) == 0:
                logger.error(f"🚫 Не удалось обработать аудио файл {file_ext}")
                logger.error("💡 Попробованные методы:")
                logger.error("   1. librosa (native)")
                logger.error("   2. soundfile")
                if self.ffmpeg_available:
                    logger.error("   3. pydub + ffmpeg")
                    logger.error("   4. ffmpeg direct")
                else:
                    logger.error("   3. pydub (FFmpeg недоступен)")
                logger.error("   5. webm as ogg")
                logger.error("   6. raw PCM extraction")
                logger.error("💡 Рекомендации:")
                logger.error("   • Установите FFmpeg для лучшей поддержки WebM")
                logger.error("   • Попробуйте другой браузер (Chrome → Firefox)")
                logger.error("   • Проверьте настройки микрофона")
                return np.array([])
            
            # Нормализация
            audio = librosa.util.normalize(audio)
            
            # Убираем тишину (более мягкая обрезка)
            try:
                audio, _ = librosa.effects.trim(audio, top_db=20)
            except:
                # Если trim не работает, оставляем как есть
                pass
            
            # Проверяем минимальную длину
            min_samples = int(0.3 * self.config['sample_rate'])  # 0.3 секунды
            if len(audio) < min_samples:
                logger.warning(f"Аудио слишком короткое: {len(audio)} сэмплов ({len(audio)/self.config['sample_rate']:.2f}с)")
                return np.array([])
            
            duration = len(audio)/self.config['sample_rate']
            logger.info(f"✅ Аудио обработано ({success_method}): {duration:.2f}с, {len(audio)} сэмплов")
            return audio
            
        except Exception as e:
            logger.error(f"❌ Критическая ошибка предобработки аудио: {e}")
            return np.array([])
        
    def _convert_webm_with_ffmpeg(self, input_path: str) -> Optional[np.ndarray]:
        """Прямая конвертация WebM через FFmpeg в WAV"""
        try:
            output_path = input_path + '_converted.wav'
            
            # Используем сохраненный путь к FFmpeg или команду из PATH
            if hasattr(self, 'ffmpeg_executable'):
                ffmpeg_cmd = self.ffmpeg_executable
            else:
                ffmpeg_cmd = 'ffmpeg'
            
            cmd = [
                ffmpeg_cmd,
                '-i', input_path,
                '-acodec', 'pcm_s16le',
                '-ar', str(self.config['sample_rate']),
                '-ac', '1',
                '-y',
                output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0 and os.path.exists(output_path):
                audio, sr = librosa.load(output_path, sr=self.config['sample_rate'])
                os.unlink(output_path)
                return audio
            else:
                logger.warning(f"FFmpeg conversion failed: {result.stderr}")
                return None
                
        except Exception as e:
            logger.warning(f"FFmpeg conversion error: {e}")
            return None
    
    
    def _try_raw_pcm_extraction(self, audio_data: bytes) -> Optional[np.ndarray]:
        """Попытка извлечь PCM данные напрямую из WebM (экспериментальный метод)"""
        try:
            # Ищем возможные PCM данные в WebM контейнере
            # Это очень упрощенный подход, но может сработать для некоторых WebM файлов
            
            # Пропускаем заголовки WebM (примерно первые 100-500 байт)
            for skip_bytes in [100, 200, 500, 1000]:
                if len(audio_data) <= skip_bytes:
                    continue
                
                try:
                    # Пытаемся интерпретировать данные как 16-bit PCM
                    raw_audio = audio_data[skip_bytes:]
                    
                    # Проверяем что длина четная (для 16-bit данных)
                    if len(raw_audio) % 2 != 0:
                        raw_audio = raw_audio[:-1]
                    
                    # Конвертируем в numpy array
                    audio_int16 = np.frombuffer(raw_audio, dtype=np.int16)
                    
                    # Нормализуем в float32
                    audio_float = audio_int16.astype(np.float32) / 32768.0
                    
                    # Базовая проверка что это похоже на аудио
                    if len(audio_float) > 1000:  # Минимум 1000 сэмплов
                        # Проверяем что данные не являются постоянными (не тишина)
                        if np.std(audio_float) > 0.01:
                            logger.info(f"Raw PCM extraction: найдено {len(audio_float)} сэмплов (пропуск {skip_bytes} байт)")
                            return audio_float
                    
                except Exception:
                    continue
            
            return None
            
        except Exception as e:
            logger.warning(f"Raw PCM extraction failed: {e}")
            return None
    
    async def transcribe_audio(self, audio_data: np.ndarray) -> Dict:
        """Транскрипция аудио через Whisper (БЫСТРО - модель уже в памяти)"""
        try:
            if not self._whisper_loaded or self.whisper_model is None:
                return {"success": False, "error": "Whisper модель не загружена"}
            
            # Whisper ожидает float32
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            # Проверяем длину аудио
            if len(audio_data) < self.config['sample_rate'] * 0.1:  # меньше 0.1 сек
                return {"success": False, "error": "Аудио слишком короткое для транскрипции"}
            
            logger.info(f"🎙️ Транскрибируем аудио: {len(audio_data)/self.config['sample_rate']:.1f} сек")
            
            # Транскрипция (БЫСТРО - модель уже в памяти!)
            start_time = datetime.now()
            result = self.whisper_model.transcribe(
                audio_data,
                language='ru',  # можно автодетект: language=None
                task='transcribe',
                temperature=0.0,  # детерминированный результат
                no_speech_threshold=0.6,
                condition_on_previous_text=False
            )
            
            transcription_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"⚡ Транскрипция завершена за {transcription_time:.1f} сек")
            
            # Проверяем что получили текст
            text = result["text"].strip()
            if not text:
                return {"success": False, "error": "Не удалось распознать речь"}
            
            return {
                "success": True,
                "text": text,
                "language": result.get("language", "unknown"),
                "confidence": 1.0 - result.get("no_speech_prob", 0.5),
                "processing_time": transcription_time,
                "segments": result.get("segments", [])
            }
            
        except Exception as e:
            logger.error(f"❌ Ошибка транскрипции: {e}")
            return {"success": False, "error": str(e)}
    
    def extract_voice_features(self, audio_data: np.ndarray) -> Optional[np.ndarray]:
        """Извлечение голосовых характеристик (MFCC) для speaker identification"""
        try:
            if len(audio_data) == 0:
                return None
            
            # MFCC коэффициенты (основа для speaker identification)
            mfccs = librosa.feature.mfcc(
                y=audio_data,
                sr=self.config['sample_rate'],
                n_mfcc=self.config['mfcc_features'],
                n_fft=2048,
                hop_length=512
            )
            
            # Статистические характеристики по времени
            features = []
            
            # Среднее и стандартное отклонение для каждого MFCC
            features.extend(np.mean(mfccs, axis=1))  # Средние значения
            features.extend(np.std(mfccs, axis=1))   # Стандартные отклонения
            
            # Дополнительные признаки
            # Спектральный центроид (характеристика тембра)
            spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=self.config['sample_rate'])
            features.append(np.mean(spectral_centroids))
            features.append(np.std(spectral_centroids))
            
            # Zero crossing rate (характеристика голоса)
            zcr = librosa.feature.zero_crossing_rate(audio_data)
            features.append(np.mean(zcr))
            features.append(np.std(zcr))
            
            # RMS энергия
            rms = librosa.feature.rms(y=audio_data)
            features.append(np.mean(rms))
            features.append(np.std(rms))
            
            return np.array(features)
            
        except Exception as e:
            logger.error(f"❌ Ошибка извлечения признаков: {e}")
            return None
    
    async def start_calibration(self, session_id: str) -> Dict:
        """Начало калибровки голоса кандидата"""
        try:
            logger.info(f"🎯 Начинаем калибровку голоса для сессии {session_id}")
            
            # Проверяем что Whisper загружен
            if not self._whisper_loaded:
                return {"success": False, "error": "Система не инициализирована"}
            
            # Сбрасываем предыдущие данные
            self.calibration_samples = []
            self.is_calibrated = False
            self.candidate_voice_profile = None
            
            # Создаем папку для сессии
            session_dir = self.audio_logs_dir / session_id
            session_dir.mkdir(exist_ok=True)
            
            return {
                "success": True,
                "message": "🎤 Калибровка начата. Говорите четко и спокойно указанную фразу.",
                "calibration_phrase": "Меня зовут [Ваше имя], я прохожу тестирование в Halyk Bank на должность разработчика. Это моя калибровочная фраза для системы голосового прокторинга. Я говорю естественным голосом.",
                "duration": self.config['calibration_duration'],
                "min_samples": self.config['min_calibration_samples'],
                "session_id": session_id,
                "instructions": [
                    "Говорите своим обычным голосом",
                    "Не шепчите и не кричите",
                    "Повторите фразу 2-3 раза четко",
                    "Избегайте фонового шума"
                ]
            }
            
        except Exception as e:
            logger.error(f"❌ Ошибка начала калибровки: {e}")
            return {"success": False, "error": str(e)}
    
    async def add_calibration_sample(self, session_id: str, audio_data: bytes) -> Dict:
        """Добавление образца для калибровки"""
        try:
            # Предобрабатываем аудио
            audio_array = self.preprocess_audio(audio_data)
            
            if len(audio_array) == 0:
                return {"success": False, "error": "Пустые или слишком короткие аудио данные"}
            
            # Проверяем минимальную длительность
            min_duration = 1.0  # секунда
            actual_duration = len(audio_array) / self.config['sample_rate']
            if actual_duration < min_duration:
                return {
                    "success": False,
                    "error": f"Слишком короткий аудио фрагмент: {actual_duration:.1f}с (нужно минимум {min_duration}с)"
                }
            
            # Извлекаем признаки
            features = self.extract_voice_features(audio_array)
            if features is None:
                return {"success": False, "error": "Не удалось извлечь голосовые признаки"}
            
            # Получаем транскрипцию (БЫСТРО!)
            transcription = await self.transcribe_audio(audio_array)
            
            # Проверяем что транскрипция получилась
            if not transcription.get("success"):
                return {
                    "success": False,
                    "error": f"Ошибка транскрипции: {transcription.get('error', 'Неизвестная ошибка')}"
                }
            
            # Сохраняем образец
            sample = {
                "timestamp": datetime.now().isoformat(),
                "audio_length": actual_duration,
                "features": features.tolist(),
                "transcription": transcription,
                "sample_id": len(self.calibration_samples),
                "feature_quality": "good" if len(features) > 20 else "basic"
            }
            
            self.calibration_samples.append(sample)
            
            # Сохраняем аудио файл
            session_dir = self.audio_logs_dir / session_id
            audio_filename = f"calibration_sample_{len(self.calibration_samples):02d}.wav"
            sf.write(
                session_dir / audio_filename,
                audio_array,
                self.config['sample_rate']
            )
            
            logger.info(f"📥 Добавлен образец калибровки #{len(self.calibration_samples)}: '{transcription.get('text', 'N/A')[:50]}...' ({actual_duration:.1f}с)")
            
            # Прогресс калибровки
            progress = len(self.calibration_samples) / self.config['min_calibration_samples'] * 100
            can_finish = len(self.calibration_samples) >= self.config['min_calibration_samples']
            
            return {
                "success": True,
                "samples_collected": len(self.calibration_samples),
                "min_required": self.config['min_calibration_samples'],
                "progress_percent": min(100, int(progress)),
                "can_finish_calibration": can_finish,
                "transcription": transcription.get('text', ''),
                "audio_length": actual_duration,
                "confidence": transcription.get('confidence', 0.0),
                "processing_time": transcription.get('processing_time', 0.0)
            }
            
        except Exception as e:
            logger.error(f"❌ Ошибка добавления образца: {e}")
            return {"success": False, "error": str(e)}
    
    async def finish_calibration(self, session_id: str) -> Dict:
        """Завершение калибровки и создание голосового профиля"""
        try:
            if len(self.calibration_samples) < self.config['min_calibration_samples']:
                return {
                    "success": False,
                    "error": f"Недостаточно образцов для калибровки. Нужно минимум {self.config['min_calibration_samples']}, есть {len(self.calibration_samples)}"
                }
            
            logger.info(f"🧠 Создаем голосовой профиль из {len(self.calibration_samples)} образцов")
            
            # Извлекаем все признаки
            all_features = []
            good_samples = 0
            
            for sample in self.calibration_samples:
                if sample['features'] and len(sample['features']) > 0:
                    all_features.append(np.array(sample['features']))
                    if sample['transcription'].get('confidence', 0) > 0.7:
                        good_samples += 1
            
            if len(all_features) == 0:
                return {"success": False, "error": "Нет валидных голосовых признаков"}
            
            # Создаем статистический профиль
            features_matrix = np.array(all_features)
            
            # Нормализация признаков
            scaler = StandardScaler()
            normalized_features = scaler.fit_transform(features_matrix)
            
            # Создаем профиль кандидата
            mean_profile = np.mean(normalized_features, axis=0)
            std_profile = np.std(normalized_features, axis=0)
            
            # Вычисляем "эталонное" расстояние (внутри-класс вариация)
            intra_class_distances = []
            for i in range(len(normalized_features)):
                distance = cosine(normalized_features[i], mean_profile)
                if not np.isnan(distance):
                    intra_class_distances.append(distance)
            
            avg_intra_distance = np.mean(intra_class_distances) if intra_class_distances else 0.3
            
            # Сохраняем профиль кандидата
            self.candidate_voice_profile = {
                "mean_features": mean_profile.tolist(),
                "std_features": std_profile.tolist(),
                "scaler_mean": scaler.mean_.tolist(),
                "scaler_scale": scaler.scale_.tolist(),
                "num_samples": len(all_features),
                "good_samples": good_samples,
                "avg_intra_distance": avg_intra_distance,
                "created_at": datetime.now().isoformat(),
                "session_id": session_id,
                "feature_dimension": len(mean_profile),
                "quality_score": good_samples / len(all_features),
                "calibration_config": self.config.copy()
            }
            
            self.is_calibrated = True
            
            # Сохраняем профиль в файл
            session_dir = self.audio_logs_dir / session_id
            profile_file = session_dir / "voice_profile.json"
            with open(profile_file, 'w', encoding='utf-8') as f:
                json.dump(self.candidate_voice_profile, f, ensure_ascii=False, indent=2)
            
            # Сохраняем scaler для будущего использования
            import pickle
            scaler_file = session_dir / "voice_scaler.pkl"
            with open(scaler_file, 'wb') as f:
                pickle.dump(scaler, f)
            
            logger.info("✅ Голосовой профиль кандидата создан успешно")
            
            return {
                "success": True,
                "message": "🎉 Калибровка завершена! Голосовой профиль создан успешно.",
                "profile_stats": {
                    "samples_used": len(all_features),
                    "good_quality_samples": good_samples,
                    "feature_dimension": len(mean_profile),
                    "quality_score": f"{good_samples / len(all_features) * 100:.1f}%",
                    "avg_confidence": avg_intra_distance,
                    "created_at": self.candidate_voice_profile["created_at"]
                },
                "ready_for_testing": True
            }
            
        except Exception as e:
            logger.error(f"❌ Ошибка завершения калибровки: {e}")
            return {"success": False, "error": str(e)}
    
    async def analyze_speech(self, session_id: str, audio_data: bytes) -> Dict:
        """Анализ речи в реальном времени (кто говорит + что говорит)"""
        try:
            if not self.is_calibrated:
                return {"success": False, "error": "Система не откалибрована"}
            
            # Предобрабатываем аудио
            audio_array = self.preprocess_audio(audio_data)
            
            if len(audio_array) == 0:
                return {"success": False, "error": "Пустые аудио данные"}
            
            actual_duration = len(audio_array) / self.config['sample_rate']
            
            # Получаем транскрипцию (БЫСТРО!)
            transcription_result = await self.transcribe_audio(audio_array)
            
            # Извлекаем голосовые признаки
            current_features = self.extract_voice_features(audio_array)
            
            # Определяем говорящего
            speaker_result = self.identify_speaker(current_features)
            
            # Создаем результат анализа
            analysis_result = {
                "timestamp": datetime.now().isoformat(),
                "transcription": transcription_result,
                "speaker_identification": speaker_result,
                "audio_length": actual_duration,
                "session_id": session_id,
                "analysis_id": len(self.analysis_history) + 1
            }
            
            # Сохраняем в историю
            self.analysis_history.append(analysis_result)
            
            # Сохраняем аудио файл
            session_dir = self.audio_logs_dir / session_id
            timestamp_str = datetime.now().strftime('%H%M%S_%f')[:-3]  # миллисекунды
            audio_filename = f"analysis_{timestamp_str}.wav"
            sf.write(
                session_dir / audio_filename,
                audio_array,
                self.config['sample_rate']
            )
            
            # Логируем результат
            is_candidate = speaker_result.get('is_candidate', False)
            confidence = speaker_result.get('confidence', 0) * 100
            text = transcription_result.get('text', 'N/A')[:100]
            processing_time = transcription_result.get('processing_time', 0)
            
            status = "🟢 КАНДИДАТ" if is_candidate else "🔴 НЕ КАНДИДАТ"
            logger.info(f"🎤 {status} ({confidence:.1f}%) [{processing_time:.1f}с] - '{text}'")
            
            return {
                "success": True,
                "result": analysis_result
            }
            
        except Exception as e:
            logger.error(f"❌ Ошибка анализа речи: {e}")
            return {"success": False, "error": str(e)}
    
    def identify_speaker(self, current_features: np.ndarray) -> Dict:
        """Определение говорящего (кандидат или нет) - улучшенная версия"""
        try:
            if not self.is_calibrated or self.candidate_voice_profile is None:
                return {"is_candidate": False, "confidence": 0.0, "error": "Не откалибровано"}
            
            if current_features is None or len(current_features) == 0:
                return {"is_candidate": False, "confidence": 0.0, "error": "Нет признаков"}
            
            # Получаем эталонный профиль
            mean_profile = np.array(self.candidate_voice_profile['mean_features'])
            scaler_mean = np.array(self.candidate_voice_profile['scaler_mean'])
            scaler_scale = np.array(self.candidate_voice_profile['scaler_scale'])
            
            if len(current_features) != len(mean_profile):
                return {"is_candidate": False, "confidence": 0.0, "error": "Несовместимые размерности признаков"}
            
            # Нормализуем текущие признаки тем же scaler'ом
            normalized_current = (current_features - scaler_mean) / scaler_scale
            
            # Вычисляем схожесть (cosine similarity)
            try:
                cos_distance = cosine(normalized_current, mean_profile)
                if np.isnan(cos_distance):
                    cos_distance = 1.0  # максимальное расстояние
                
                # Преобразуем в схожесть (от 0 до 1)
                similarity = 1.0 - cos_distance
                similarity = max(0.0, min(1.0, similarity))
                
                # Дополнительная проверка на основе "внутри-класс" расстояния
                avg_intra_distance = self.candidate_voice_profile.get('avg_intra_distance', 0.3)
                
                # Адаптивный порог на основе качества калибровки
                quality_score = self.candidate_voice_profile.get('quality_score', 0.8)
                adaptive_threshold = self.config['similarity_threshold'] * (0.8 + 0.2 * quality_score)
                
                # Учитываем "внутри-класс" вариацию
                if cos_distance <= avg_intra_distance * 1.5:
                    # Расстояние в пределах нормальной вариации - скорее всего кандидат
                    confidence_boost = 0.1
                else:
                    confidence_boost = 0.0
                
                final_similarity = similarity + confidence_boost
                is_candidate = final_similarity >= adaptive_threshold
                

                return {
                    "is_candidate": bool(is_candidate),
                    "confidence": float(final_similarity),
                    "raw_similarity": float(similarity),
                    "cosine_distance": float(cos_distance),
                    "threshold": float(adaptive_threshold),
                    "intra_class_distance": float(avg_intra_distance),
                    "method": "cosine_similarity_adaptive",
                    "quality_score": float(quality_score)
                }
                
            except Exception as calc_error:
                logger.warning(f"Ошибка вычисления схожести: {calc_error}")
                return {"is_candidate": False, "confidence": 0.0, "error": f"Ошибка вычислений: {calc_error}"}
            
        except Exception as e:
            logger.error(f"❌ Ошибка идентификации говорящего: {e}")
            return {"is_candidate": False, "confidence": 0.0, "error": str(e)}
    
    async def get_session_logs(self, session_id: str) -> Dict:
        """Получение логов сессии"""
        try:
            session_dir = self.audio_logs_dir / session_id
            
            if not session_dir.exists():
                return {"success": False, "error": "Сессия не найдена"}
            
            # Читаем профиль (если есть)
            voice_profile = None
            profile_file = session_dir / "voice_profile.json"
            if profile_file.exists():
                with open(profile_file, 'r', encoding='utf-8') as f:
                    voice_profile = json.load(f)
            
            # Собираем файлы
            audio_files = list(session_dir.glob("*.wav"))
            calibration_files = [f for f in audio_files if "calibration" in f.name]
            analysis_files = [f for f in audio_files if "analysis" in f.name]
            
            return {
                "success": True,
                "session_id": session_id,
                "voice_profile": voice_profile,
                "calibration_files": [f.name for f in calibration_files],
                "analysis_files": [f.name for f in analysis_files],
                "total_audio_files": len(audio_files),
                "analysis_history": self.analysis_history[-20:],  # последние 20
                "is_calibrated": self.is_calibrated,
                "total_samples": len(self.calibration_samples),
                "system_stats": self.get_system_stats()
            }
            
        except Exception as e:
            logger.error(f"❌ Ошибка получения логов: {e}")
            return {"success": False, "error": str(e)}
    
    def get_system_stats(self) -> Dict:
        """Получение статистики системы"""
        return {
            "whisper_model": self.config['whisper_model'],
            "whisper_loaded": self._whisper_loaded,
            "is_calibrated": self.is_calibrated,
            "calibration_samples": len(self.calibration_samples),
            "analysis_history": len(self.analysis_history),
            "config": self.config,
            "voice_profile_quality": self.candidate_voice_profile.get('quality_score', 0.0) if self.candidate_voice_profile else 0.0,
            "ffmpeg_available": self.ffmpeg_available,
            "supported_methods": self._get_supported_methods()
        }
    
    def _get_supported_methods(self) -> List[str]:
        """Получение списка поддерживаемых методов обработки аудио"""
        methods = [
            "librosa (native)",
            "soundfile", 
            "raw PCM extraction"
        ]
        
        if self.ffmpeg_available:
            methods.extend([
                "pydub + ffmpeg",
                "ffmpeg direct conversion"
            ])
        
        methods.append("webm as ogg fallback")
        
        return methods
    
    def reset_system(self):
        """Сброс системы (очистка калибровки)"""
        logger.info("🔄 Сброс системы AudioProctorSystem")
        self.is_calibrated = False
        self.candidate_voice_profile = None
        self.calibration_samples = []
        self.analysis_history = []

# === ГЛОБАЛЬНЫЙ ЭКЗЕМПЛЯР С КЕШИРОВАНИЕМ ===
_global_audio_proctor = None

async def get_audio_proctor(data_dir: Path) -> AudioProctorSystem:
    """Получение глобального экземпляра AudioProctorSystem (с кешированием модели)"""
    global _global_audio_proctor
    
    if _global_audio_proctor is None:
        logger.info("🏗️ Создание глобального экземпляра AudioProctorSystem")
        _global_audio_proctor = AudioProctorSystem(data_dir)
        success = await _global_audio_proctor.initialize()
        
        if not success:
            logger.error("❌ Не удалось инициализировать AudioProctorSystem")
            _global_audio_proctor = None
            raise RuntimeError("Ошибка инициализации аудио системы")
        
        logger.info("✅ Глобальный AudioProctorSystem готов")
    
    return _global_audio_proctor

# === ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ===

def audio_bytes_to_numpy(audio_bytes: bytes, sample_rate: int = 16000) -> np.ndarray:
    """Конвертация bytes в numpy array"""
    try:
        audio_io = io.BytesIO(audio_bytes)
        audio, sr = librosa.load(audio_io, sr=sample_rate)
        return librosa.util.normalize(audio)
    except Exception as e:
        logger.error(f"❌ Ошибка конвертации аудио: {e}")
        return np.array([])

def base64_to_audio(base64_data: str, sample_rate: int = 16000) -> np.ndarray:
    """Конвертация base64 в numpy array"""
    try:
        audio_bytes = base64.b64decode(base64_data)
        return audio_bytes_to_numpy(audio_bytes, sample_rate)
    except Exception as e:
        logger.error(f"❌ Ошибка декодирования base64: {e}")
        return np.array([])

# === БЫСТРЫЙ ТЕСТ СИСТЕМЫ ===
if __name__ == "__main__":
    async def test_system():
        """Быстрый тест системы"""
        from pathlib import Path
        
        print("🚀 Тестирование AudioProctorSystem...")
        
        # Создаем систему
        system = AudioProctorSystem(Path("test_data"))
        
        # Инициализируем (загрузка модели)
        print("📥 Загружаем Whisper модель...")
        if await system.initialize():
            print("✅ Система инициализирована успешно")
            print(f"📊 Статистика: {system.get_system_stats()}")
            
            # Проверяем что модель загружена
            if system._whisper_loaded:
                print("♻️ Повторная инициализация (должна быть мгновенной)...")
                start_time = datetime.now()
                await system.initialize()
                load_time = (datetime.now() - start_time).total_seconds()
                print(f"⚡ Повторная загрузка за {load_time:.2f} сек (кеш работает!)")
            
        else:
            print("❌ Ошибка инициализации")
    
    # Запуск теста
    asyncio.run(test_system())