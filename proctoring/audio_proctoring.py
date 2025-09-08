"""
AudioProctorSystem - Система аудио прокторинга с ИИ и умным кешированием
Модель Whisper загружается ОДИН раз и остается в памяти
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
    
    def preprocess_audio(self, audio_data: bytes, sample_rate: int = None) -> np.ndarray:
        """Предобработка аудио данных с поддержкой MP4/WebM/OGG"""
        try:
            # Определяем тип файла по первым байтам
            file_signature = audio_data[:12]
            
            # Определяем расширение файла
            if b'ftyp' in file_signature or b'mp4' in file_signature:
                file_ext = '.mp4'
            elif b'OggS' in file_signature:
                file_ext = '.ogg'  
            elif b'webm' in file_signature or b'mkv' in file_signature:
                file_ext = '.webm'
            else:
                # По умолчанию пробуем как WebM
                file_ext = '.webm'
            
            logger.info(f"🔍 Определен тип файла: {file_ext}")
            
            # Создаем временный файл с правильным расширением
            with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as tmp_file:
                tmp_file.write(audio_data)
                tmp_file_path = tmp_file.name
            
            audio = None
            
            try:
                # Способ 1: Librosa напрямую (работает с MP4 и OGG)
                audio, sr = librosa.load(
                    tmp_file_path, 
                    sr=sample_rate or self.config['sample_rate'],
                    mono=True
                )
                logger.info(f"✅ Librosa успешно загрузила {file_ext}: {len(audio)} сэмплов")
                
            except Exception as librosa_error:
                logger.warning(f"Librosa {file_ext} error: {librosa_error}")
                
                # Способ 2: SoundFile (хорошо работает с MP4/OGG)
                try:
                    import soundfile as sf
                    audio, sr = sf.read(tmp_file_path)
                    
                    if sr != self.config['sample_rate']:
                        audio = librosa.resample(audio, orig_sr=sr, target_sr=self.config['sample_rate'])
                    
                    logger.info(f"✅ SoundFile успешно прочитал {file_ext}: {len(audio)} сэмплов")
                    
                except Exception as sf_error:
                    logger.warning(f"SoundFile {file_ext} error: {sf_error}")
                    
                    # Способ 3: Pydub (универсальный, но требует ffmpeg для некоторых форматов)
                    try:
                        from pydub import AudioSegment
                        
                        # Пробуем загрузить без указания формата (автодетекция)
                        audio_segment = AudioSegment.from_file(tmp_file_path)
                        
                        # Конвертируем в нужный формат
                        audio_segment = audio_segment.set_frame_rate(self.config['sample_rate'])
                        audio_segment = audio_segment.set_channels(1)  # Mono
                        
                        # Получаем raw audio данные
                        raw_data = audio_segment.raw_data
                        audio = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0
                        
                        logger.info(f"✅ Pydub успешно конвертировал {file_ext}: {len(audio)} сэмплов")
                        
                    except Exception as pydub_error:
                        logger.error(f"Pydub {file_ext} конвертация не удалась: {pydub_error}")
                        
                        # Способ 4: Если это WebM, пробуем как OGG
                        if file_ext == '.webm':
                            try:
                                ogg_path = tmp_file_path.replace('.webm', '.ogg')
                                os.rename(tmp_file_path, ogg_path)
                                
                                audio, sr = librosa.load(ogg_path, sr=self.config['sample_rate'])
                                logger.info(f"✅ WebM обработан как OGG: {len(audio)} сэмплов")
                                tmp_file_path = ogg_path  # Обновляем путь для удаления
                                
                            except Exception as ogg_error:
                                logger.error(f"WebM→OGG конвертация не удалась: {ogg_error}")
            
            finally:
                # Удаляем временные файлы
                for tmp_path in [tmp_file_path, tmp_file_path.replace(file_ext, '.ogg'), tmp_file_path.replace(file_ext, '.wav')]:
                    if os.path.exists(tmp_path):
                        try:
                            os.unlink(tmp_path)
                        except:
                            pass
            
            # Проверяем результат
            if audio is None or len(audio) == 0:
                logger.error(f"🚫 Не удалось обработать аудио файл {file_ext}")
                logger.error("💡 Возможные решения:")
                logger.error("   1. Используйте другой браузер (Chrome, Firefox, Edge)")
                logger.error("   2. Проверьте настройки микрофона")
                logger.error("   3. Установите FFmpeg для расширенной поддержки форматов")
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
            
            logger.info(f"✅ Аудио обработано успешно: {len(audio)/self.config['sample_rate']:.2f}с, {len(audio)} сэмплов")
            return audio
            
        except Exception as e:
            logger.error(f"❌ Критическая ошибка предобработки аудио: {e}")
            return np.array([])
    
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
                    "is_candidate": is_candidate,
                    "confidence": final_similarity,
                    "raw_similarity": similarity,
                    "cosine_distance": cos_distance,
                    "threshold": adaptive_threshold,
                    "intra_class_distance": avg_intra_distance,
                    "method": "cosine_similarity_adaptive",
                    "quality_score": quality_score
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
            "voice_profile_quality": self.candidate_voice_profile.get('quality_score', 0.0) if self.candidate_voice_profile else 0.0
        }
    
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