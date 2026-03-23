#!/usr/bin/env python3
"""
AI Assistant con riconoscimento facciale personale, speech-to-text e genio animato
Versione 12 - Miglioramenti:
- Generazione immagini con DALL-E (comando "disegna" / "genera immagine")
- Log pulito: solo domande e risposte nel file JSON
- Terminale pulito: domande e risposte ben formattate, no log face recognition
- Interfaccia genio: mantiene avviso utente autorizzato
- Riconoscimento facciale personale (anto2025.png)  
- Modalità scientifica avanzata (GPT-4o-mini)
- Controllo vocale completo con conferme vocali
- Sistema wake/sleep vocale
- Anti-accavallamento domande
- Interfaccia grafica animata
"""

import cv2
import numpy as np
import random
import os
import subprocess
import threading
import time
import speech_recognition as sr
from openai import OpenAI
import pygame
import math
from dotenv import load_dotenv
import json
from datetime import datetime
import re

# Carica le variabili d'ambiente dal file .env
load_dotenv()

# Prova a importare face_recognition, fallback se non disponibile
try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
    print("✅ face_recognition disponibile")
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False
    print("⚠️ face_recognition non disponibile - modalità base")

# Sopprime i log verbosi di face_recognition / dlib
import logging
logging.getLogger("face_recognition").setLevel(logging.WARNING)
logging.getLogger("dlib").setLevel(logging.WARNING)


class AIAssistantWithGenie:
    def __init__(self, openai_api_key, reference_image_path="anto2025.png"):
        # Inizializza OpenAI
        self.client = OpenAI(api_key=openai_api_key)
        
        # Inizializza il riconoscitore vocale
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        # Inizializza la webcam
        self.cap = cv2.VideoCapture(0)
        
        # Threading locks per evitare conflitti
        self.microphone_lock = threading.Lock()
        self.speech_lock = threading.Lock()
        
        # SISTEMA DI RICONOSCIMENTO FACCIALE PERSONALE
        self.reference_image_path = reference_image_path
        self.authorized_face_encoding = None
        self.face_recognition_enabled = FACE_RECOGNITION_AVAILABLE
        self.load_reference_face()
        
        # SISTEMA DI LOGGING
        self.log_dir = "log"
        self.current_session = None
        self.setup_logging()
        
        # Directory per le immagini generate
        self.images_dir = "generated_images"
        if not os.path.exists(self.images_dir):
            os.makedirs(self.images_dir)
            print(f"📁 Creata directory immagini: {self.images_dir}")
        
        # Carica il classificatore per il riconoscimento facciale generico
        self.load_face_cascade()
        
        # Liste di domande casuali scientifiche
        self.random_questions = [
            "Quale fenomeno scientifico ti incuriosisce di più?",
            "C'è qualche meccanismo in natura che vorresti capire meglio?",
            "Hai domande su fisica, chimica, biologia o matematica?",
            "Quale tecnologia moderna vorresti che ti spiegassi nel dettaglio?",
            "C'è qualche teoria scientifica che ti affascina?",
            "Vuoi sapere come funziona qualcosa a livello molecolare o atomico?",
            "Quale processo biologico ti interessa approfondire?",
            "Hai curiosità su ricerche scientifiche recenti?",
            "C'è qualche principio fisico che vorresti esplorare?",
            "Quale innovazione tecnologica ti piacerebbe comprendere a fondo?"
        ]
        
        # Parole chiave per richieste di disegno/immagine
        self.draw_keywords = [
            'disegna', 'disegnami', 'genera immagine', "genera un'immagine",
            'crea immagine', "crea un'immagine", 'illustra', 'illustrami',
            'fammi vedere', 'mostrami', 'dipingi', 'dipingimi',
            'draw', 'generate image', 'create image', 'paint',
            'fai un disegno', "fai un'immagine", 'produci immagine'
        ]
        
        # Variabili di stato
        self.face_detected = False
        self.authorized_user_detected = False
        self.last_question_time = 0
        self.question_interval = 10  # secondi tra le domande
        self.is_listening = False
        self.is_speaking = False
        self.is_processing = False
        self.is_paused = False
        self.shutdown = False
        self.cooldown_after_interaction = 15
        self.show_video = True
        self.is_generating_image = False
        
        # Parole chiave per controllo vocale
        self.pause_words = [
            'pausa', 'stop', 'fermati', 'silenzio', 'basta', 'dormi', 
            'sleep', 'pause', 'zitto', 'taci', 'smetti', 'stai zitto'
        ]
        self.wake_words = [
            'risveglia', 'sveglia', 'wake up', 'continua', 'torna', 
            'genio', 'ehi genio', 'ciao genio', 'hey', 'start', 
            'riprendi', 'attivati', 'ci sei'
        ]
        
        # Inizializza pygame
        self.setup_pygame()
        
        print("✅ AI Assistant inizializzato! Premi 'q' per uscire.")
        if self.authorized_face_encoding is not None:
            print("🔐 Riconoscimento facciale personale ATTIVO")
        else:
            print("⚠️ Riconoscimento facciale personale NON DISPONIBILE")
    
    def load_face_cascade(self):
        """Carica il classificatore Haar per rilevamento visi"""
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            if self.face_cascade.empty():
                self.face_cascade = None
            else:
                print("✅ Classificatore facciale generico caricato")
        except AttributeError:
            possible_paths = [
                '/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml',
                '/opt/homebrew/share/opencv4/haarcascades/haarcascade_frontalface_default.xml'
            ]
            
            self.face_cascade = None
            for path in possible_paths:
                if os.path.exists(path):
                    self.face_cascade = cv2.CascadeClassifier(path)
                    if not self.face_cascade.empty():
                        print(f"✅ Classificatore facciale caricato da: {path}")
                        break
            
            if self.face_cascade is None or self.face_cascade.empty():
                print("❌ Impossibile caricare il classificatore facciale")
                self.face_cascade = None
    
    def load_reference_face(self):
        """Carica l'immagine di riferimento dell'utente autorizzato"""
        if not self.face_recognition_enabled:
            self.authorized_face_encoding = None
            return
            
        try:
            if not os.path.exists(self.reference_image_path):
                print(f"❌ Immagine di riferimento non trovata: {self.reference_image_path}")
                print("🔓 Il sistema funzionerà con qualsiasi viso rilevato")
                self.authorized_face_encoding = None
                return
            
            reference_image = face_recognition.load_image_file(self.reference_image_path)
            reference_encodings = face_recognition.face_encodings(reference_image)
            
            if len(reference_encodings) == 0:
                print("❌ Nessun viso rilevato nell'immagine di riferimento!")
                self.authorized_face_encoding = None
                return
            
            if len(reference_encodings) > 1:
                print("⚠️ Più visi rilevati nell'immagine, uso il primo")
            
            self.authorized_face_encoding = reference_encodings[0]
            print("✅ Viso di riferimento caricato con successo!")
            
        except Exception as e:
            print(f"❌ Errore nel caricamento dell'immagine di riferimento: {e}")
            self.authorized_face_encoding = None
    
    def setup_logging(self):
        """Configura il sistema di logging"""
        try:
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
                print(f"📁 Creata directory log: {self.log_dir}")
            
            session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.current_session = {
                "session_id": session_timestamp,
                "start_time": datetime.now().isoformat(),
                "log_file": os.path.join(self.log_dir, f"conversation_{session_timestamp}.json"),
                "conversations": []
            }
            
            self.save_session_log()
            print(f"📝 Log sessione: {self.current_session['log_file']}")
            
        except Exception as e:
            print(f"❌ Errore nel setup logging: {e}")
            self.current_session = None
    
    def log_conversation(self, user_question, ai_response, response_type="text", image_path=None):
        """Registra SOLO domande e risposte nel log (niente face recognition, niente status)"""
        if self.current_session is None:
            return
        
        try:
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "domanda": user_question,
                "risposta": ai_response,
                "tipo": response_type
            }
            
            if image_path:
                log_entry["immagine"] = image_path
            
            self.current_session["conversations"].append(log_entry)
            self.save_session_log()
            
        except Exception as e:
            print(f"❌ Errore nel logging: {e}")
    
    def save_session_log(self):
        """Salva il log della sessione su file"""
        if self.current_session is None:
            return
        
        try:
            self.current_session["last_update"] = datetime.now().isoformat()
            
            with open(self.current_session["log_file"], 'w', encoding='utf-8') as f:
                json.dump(self.current_session, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            print(f"❌ Errore nel salvataggio log: {e}")
    
    def setup_pygame(self):
        """Inizializza pygame per l'animazione"""
        pygame.init()
        self.screen_width, self.screen_height = 400, 300
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Genio AI Assistant - Accesso Sicuro")
        
        self.genie_x = self.screen_width // 2
        self.genie_y = self.screen_height // 2
        self.animation_frame = 0
    
    def create_genie_surface(self, frame):
        """Crea una superficie del genio animato"""
        surface = pygame.Surface((100, 100), pygame.SRCALPHA)
        
        body_size = 40 + int(10 * math.sin(frame * 0.2))
        if self.is_speaking:
            body_size += int(5 * math.sin(frame * 0.5))
        
        color_r = int(100 + 50 * math.sin(frame * 0.1))
        color_g = int(100 + 50 * math.cos(frame * 0.15))
        color_b = int(150 + 50 * math.sin(frame * 0.08))
        
        pygame.draw.ellipse(surface, (color_r, color_g, color_b), 
                          (50 - body_size//2, 50 - body_size//2, body_size, body_size))
        
        # Occhi
        eye_offset = int(3 * math.sin(frame * 0.3)) if self.is_speaking else 0
        pygame.draw.circle(surface, (255, 255, 255), (40, 40 + eye_offset), 8)
        pygame.draw.circle(surface, (255, 255, 255), (60, 40 + eye_offset), 8)
        pygame.draw.circle(surface, (0, 0, 0), (40, 40 + eye_offset), 4)
        pygame.draw.circle(surface, (0, 0, 0), (60, 40 + eye_offset), 4)
        
        # Bocca
        if self.is_speaking:
            mouth_height = int(5 * abs(math.sin(frame * 0.8)))
            pygame.draw.ellipse(surface, (50, 50, 50), (45, 55, 10, mouth_height + 3))
        else:
            pygame.draw.arc(surface, (50, 50, 50), (45, 55, 10, 5), 0, math.pi, 2)
        
        return surface
    
    def update_genie_animation(self):
        """Aggiorna l'animazione del genio"""
        # Colore di sfondo in base allo stato
        if self.is_paused:
            self.screen.fill((40, 20, 60))  # Viola per pausa
        elif not self.authorized_user_detected and self.authorized_face_encoding is not None:
            self.screen.fill((60, 20, 20))  # Rosso per accesso negato
        else:
            self.screen.fill((20, 20, 40))  # Blu normale
        
        # Animazione del genio
        animation_speed = 0.02 if (self.is_paused or not self.authorized_user_detected) else 0.05
        genie_surface = self.create_genie_surface(self.animation_frame)
        
        float_intensity = 5 if (self.is_paused or not self.authorized_user_detected) else 10
        float_offset_x = int(float_intensity * math.sin(self.animation_frame * animation_speed))
        float_offset_y = int(float_intensity * 1.5 * math.cos(self.animation_frame * animation_speed * 0.6))
        
        self.screen.blit(genie_surface, 
                        (self.genie_x - 50 + float_offset_x, 
                         self.genie_y - 50 + float_offset_y))
        
        # Testo di stato
        font = pygame.font.Font(None, 24)
        small_font = pygame.font.Font(None, 18)
        
        if not self.authorized_user_detected and self.authorized_face_encoding is not None:
            # AVVISO ACCESSO NEGATO - solo sulla GUI del genio
            text = font.render("🚫 ACCESSO NEGATO", True, (255, 100, 100))
            self.screen.blit(text, (10, 10))
            auth_text = small_font.render("Solo l'utente autorizzato può accedere", True, (255, 150, 150))
            self.screen.blit(auth_text, (10, 35))
        elif self.is_paused:
            text = font.render("😴 IN PAUSA", True, (255, 150, 255))
            self.screen.blit(text, (10, 10))
            wake_text = small_font.render("Di: 'GENIO' o 'SVEGLIA' per riattivarmi", True, (200, 150, 255))
            self.screen.blit(wake_text, (10, 35))
        elif self.is_generating_image:
            text = font.render("🎨 Sto disegnando...", True, (255, 200, 0))
            self.screen.blit(text, (10, 10))
        elif self.is_processing:
            text = font.render("🤖 Elaborando...", True, (255, 255, 0))
            self.screen.blit(text, (10, 10))
        elif self.face_detected:
            if self.authorized_user_detected:
                # AVVISO UTENTE AUTORIZZATO - solo sulla GUI del genio
                text = font.render("👤✅ Utente autorizzato!", True, (0, 255, 0))
            else:
                text = font.render("👤 Viso rilevato", True, (255, 255, 0))
            self.screen.blit(text, (10, 10))
        else:
            text = font.render("👁️ In attesa...", True, (255, 255, 255))
            self.screen.blit(text, (10, 10))
        
        # Stati aggiuntivi
        if self.authorized_user_detected and not self.is_paused:
            y_offset = 40
            if self.is_listening:
                listen_text = font.render("🎤 In ascolto...", True, (255, 255, 0))
                self.screen.blit(listen_text, (10, y_offset))
                y_offset += 30
            
            if self.is_speaking:
                speak_text = font.render("🗣️ Parlando...", True, (0, 255, 255))
                self.screen.blit(speak_text, (10, y_offset))
        
        # Controlli
        if self.authorized_user_detected and not self.is_paused:
            controls_text = small_font.render("SPAZIO: Parla | Q/ESC: Esci | V: Video | P: Pausa", True, (150, 150, 150))
            self.screen.blit(controls_text, (10, self.screen_height - 40))
        elif not self.authorized_user_detected and self.authorized_face_encoding is not None:
            access_controls = small_font.render("Accesso riservato - Riconoscimento facciale attivo", True, (255, 150, 150))
            self.screen.blit(access_controls, (10, self.screen_height - 40))
        elif self.is_paused:
            pause_controls = small_font.render("Modalità pausa - Ascolta per comandi di risveglio", True, (255, 150, 255))
            self.screen.blit(pause_controls, (10, self.screen_height - 40))
        
        pygame.display.flip()
        self.animation_frame += 1
    
    def detect_faces(self, frame):
        """Rileva i visi nel frame e verifica l'autorizzazione - NESSUN log su terminale/file"""
        faces_detected = False
        authorized_user = False
        
        if self.face_cascade is None:
            cv2.putText(frame, "Face detection disabled", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            return True, True, frame
        
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            faces_detected = len(faces) > 0
            
            if not self.face_recognition_enabled or self.authorized_face_encoding is None:
                authorized_user = faces_detected
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
                    cv2.putText(frame, "NON VERIFICATO", (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                return faces_detected, authorized_user, frame
            
            if faces_detected:
                try:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    face_locations = face_recognition.face_locations(rgb_frame)
                    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                    
                    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                        matches = face_recognition.compare_faces([self.authorized_face_encoding], face_encoding, tolerance=0.6)
                        face_distances = face_recognition.face_distance([self.authorized_face_encoding], face_encoding)
                        
                        if matches[0]:
                            authorized_user = True
                            confidence = (1 - face_distances[0]) * 100
                            
                            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                            cv2.putText(frame, f"AUTORIZZATO ({confidence:.1f}%)", 
                                       (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            # Nessun log su terminale, nessun log su file
                        else:
                            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                            cv2.putText(frame, "NON AUTORIZZATO", 
                                       (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                            # Nessun log su terminale, nessun log su file
                
                except Exception:
                    for (x, y, w, h) in faces:
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)
                
        except Exception:
            authorized_user = False
        
        return faces_detected, authorized_user, frame
    
    def speak_text(self, text):
        """Pronuncia il testo usando il comando 'say' di macOS"""
        if self.shutdown:
            return None
            
        def speak():
            with self.speech_lock:
                if self.shutdown:
                    return
                self.is_speaking = True
                try:
                    subprocess.run(['say', text], check=True)
                except subprocess.CalledProcessError:
                    pass
                except FileNotFoundError:
                    if not self.shutdown:
                        print("⚠️ Comando 'say' non trovato. Sei su macOS?")
                except Exception:
                    pass
                finally:
                    self.is_speaking = False
        
        thread = threading.Thread(target=speak, daemon=True)
        try:
            thread.start()
            return thread
        except RuntimeError:
            return None
    
    def ask_random_question(self):
        """Fa una domanda casuale (solo se utente autorizzato)"""
        if (not self.is_listening and not self.is_speaking and not self.is_processing and 
            not self.is_paused and self.authorized_user_detected):
            
            question = random.choice(self.random_questions)
            print(f"\n🧞‍♂️ Genio: {question}")
            
            self.speak_text(question)
            self.last_question_time = time.time()
            self.is_processing = True
    
    def check_for_control_commands(self, text):
        """Controlla se il testo contiene comandi di controllo"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in self.pause_words):
            if not self.is_paused:
                print("😴 Comando pausa rilevato!")
                self.is_paused = True
                self.speak_text("Va bene, mi metto in pausa. Dimmi genio o sveglia per riattivarmi.")
                return "PAUSA"
        
        if any(word in text_lower for word in self.wake_words):
            if self.is_paused:
                print("🔥 Comando risveglio rilevato!")
                self.is_paused = False
                self.speak_text("Eccomi! Sono tornato attivo. Come posso aiutarti?")
                return "RISVEGLIO"
        
        return None
    
    def listen_for_wake_command(self):
        """Ascolta per comandi di risveglio quando in pausa"""
        if self.shutdown or not self.is_paused:
            return None
            
        if not self.microphone_lock.acquire(blocking=False):
            return None
        
        try:
            with self.microphone as source:
                if self.shutdown or not self.is_paused:
                    return None
                
                self.recognizer.adjust_for_ambient_noise(source, duration=0.3)
                audio = self.recognizer.listen(source, timeout=3, phrase_time_limit=3)
            
            if self.shutdown:
                return None
                
            try:
                wake_text = self.recognizer.recognize_google(audio, language='it-IT')
                return self.check_for_control_commands(wake_text)
            except (sr.UnknownValueError, sr.RequestError):
                return None
                
        except sr.WaitTimeoutError:
            return None
        except Exception:
            return None
        finally:
            self.microphone_lock.release()
    
    def listen_to_user(self):
        """Ascolta l'input dell'utente dal microfono (solo se autorizzato)"""
        if (self.shutdown or self.is_listening or self.is_paused or 
            (not self.authorized_user_detected and self.authorized_face_encoding is not None)):
            return None
            
        if not self.microphone_lock.acquire(blocking=False):
            return None
        
        try:
            print("\n🎤 Sto ascoltando... Parla ora!")
            self.is_listening = True
            
            with self.microphone as source:
                if self.shutdown:
                    return None
                    
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                audio = self.recognizer.listen(source, timeout=10)
            
            if self.shutdown:
                return None
                
            try:
                user_text = self.recognizer.recognize_google(audio, language='it-IT')
                print(f"📝 Ho sentito: '{user_text}'")
                
                # Controlla comandi di controllo
                control_command = self.check_for_control_commands(user_text)
                if control_command:
                    return None
                
                # Conferma vocale
                return self.get_voice_confirmation(user_text)
                    
            except sr.UnknownValueError:
                if not self.shutdown:
                    print("❌ Non sono riuscito a capire quello che hai detto.")
                return None
            except sr.RequestError as e:
                if not self.shutdown:
                    print(f"❌ Errore nel servizio di riconoscimento vocale: {e}")
                return None
                
        except sr.WaitTimeoutError:
            if not self.shutdown:
                print("⏰ Tempo scaduto, nessun audio rilevato.")
            return None
        except Exception as e:
            if not self.shutdown:
                print(f"❌ Errore microfono: {e}")
            return None
        finally:
            self.is_listening = False
            self.microphone_lock.release()
    
    def get_voice_confirmation(self, user_text):
        """Ottiene conferma vocale dall'utente"""
        try:
            self.speak_text("Hai detto: " + user_text + ". Vuoi che lo invii a OpenAI? Rispondi sì o no.")
            
            time.sleep(3)
            print("🎤 Dimmi: SÌ o NO...")
            
            with self.microphone as source:
                if self.shutdown:
                    return None
                
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = self.recognizer.listen(source, timeout=8)
            
            if self.shutdown:
                return None
                
            try:
                confirmation = self.recognizer.recognize_google(audio, language='it-IT').lower()
                print(f"🔊 Conferma: '{confirmation}'")
                
                confirm_words = ['sì', 'si', 'sí', 'yes', 'y', 'vai', 'ok', 'okay', 'invia', 'mandalo', 'perfetto']
                reject_words = ['no', 'n', 'niente', 'annulla', 'stop', 'basta', 'cancella', 'non']
                
                if any(word in confirmation for word in confirm_words):
                    print("✅ Confermato!")
                    self.speak_text("Perfetto! Invio la domanda.")
                    time.sleep(1)
                    return user_text
                
                elif any(word in confirmation for word in reject_words):
                    print("❌ Annullato.")
                    self.speak_text("Va bene, annullo la richiesta.")
                    return None
                
                else:
                    print("🤔 Non ho capito. Annullo per sicurezza.")
                    self.speak_text("Non ho capito. Annullo la richiesta.")
                    return None
                    
            except sr.UnknownValueError:
                self.speak_text("Non ho sentito una risposta. Annullo.")
                return None
            except sr.RequestError:
                return None
                
        except sr.WaitTimeoutError:
            self.speak_text("Tempo scaduto. Annullo la richiesta.")
            return None
        except Exception:
            return None
    
    # =========================================================================
    # NUOVA FUNZIONALITÀ: GENERAZIONE IMMAGINI CON DALL-E
    # =========================================================================
    
    def is_draw_request(self, question):
        """Determina se la domanda è una richiesta di disegno/immagine"""
        question_lower = question.lower()
        return any(keyword in question_lower for keyword in self.draw_keywords)
    
    def extract_image_prompt(self, question):
        """Estrae il prompt per la generazione dell'immagine dalla domanda dell'utente"""
        prompt = question
        
        # Rimuove le parole chiave di comando per ottenere il soggetto
        for keyword in sorted(self.draw_keywords, key=len, reverse=True):
            prompt = re.sub(re.escape(keyword), '', prompt, flags=re.IGNORECASE).strip()
        
        # Rimuove preposizioni iniziali rimaste
        prompt = re.sub(r'^(un |una |uno |il |la |lo |i |le |gli |di |del |della |dello )', '', prompt, flags=re.IGNORECASE).strip()
        
        if not prompt:
            prompt = question  # fallback al testo completo
        
        return prompt
    
    def generate_image(self, prompt):
        """Genera un'immagine usando DALL-E 3"""
        try:
            self.is_generating_image = True
            print(f"🎨 Generazione immagine in corso per: '{prompt}'")
            
            response = self.client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                size="1024x1024",
                quality="standard",
                n=1,
            )
            
            image_url = response.data[0].url
            revised_prompt = response.data[0].revised_prompt
            
            # Salva l'immagine localmente
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_name = re.sub(r'[^\w\s-]', '', prompt[:40]).strip().replace(' ', '_')
            image_filename = f"{timestamp}_{safe_name}.png"
            image_path = os.path.join(self.images_dir, image_filename)
            
            # Scarica l'immagine dall'URL
            import urllib.request
            urllib.request.urlretrieve(image_url, image_path)
            
            print(f"✅ Immagine salvata: {image_path}")
            
            # Mostra l'immagine con OpenCV
            try:
                img = cv2.imread(image_path)
                if img is not None:
                    display_size = 600
                    h, w = img.shape[:2]
                    scale = display_size / max(h, w)
                    new_w, new_h = int(w * scale), int(h * scale)
                    img_resized = cv2.resize(img, (new_w, new_h))
                    
                    cv2.imshow(f'Immagine Generata - {prompt[:30]}', img_resized)
                    print("🖼️  Immagine mostrata in finestra separata")
            except Exception as e:
                print(f"⚠️ Non riesco a mostrare l'immagine: {e}")
            
            return image_path, revised_prompt, image_url
            
        except Exception as e:
            error_msg = f"Errore nella generazione dell'immagine: {str(e)}"
            print(f"❌ {error_msg}")
            return None, error_msg, None
        finally:
            self.is_generating_image = False
    
    # =========================================================================
    # RISPOSTA TESTUALE (invariata nella logica, pulita nei log)
    # =========================================================================
    
    def get_openai_response(self, user_question):
        """Ottiene una risposta da OpenAI con prompt scientifico avanzato"""
        try:
            scientific_prompt = """Sei un assistente AI altamente qualificato con competenze scientifiche avanzate. 
            
            ISTRUZIONI per le tue risposte:
            1. Fornisci risposte DETTAGLIATE e SCIENTIFICAMENTE ACCURATE
            2. Includi dati, numeri, statistiche quando disponibili
            3. Cita principi scientifici, leggi fisiche, formule se rilevanti
            4. Spiega i meccanismi sottostanti dei fenomeni
            5. Usa terminologia scientifica appropriata ma spiegala quando necessario
            6. Includi esempi concreti e analogie per chiarire concetti complessi
            7. Menziona ricerche recenti o sviluppi nel campo quando pertinenti
            8. Se incerto su dettagli specifici, specifica il grado di incertezza
            9. Struttura la risposta in modo logico e progressivo
            10. Rispondi sempre in italiano con linguaggio tecnico ma accessibile
            
            Obiettivo: Essere l'equivalente di un professore universitario esperto che spiega a uno studente curioso."""
            
            question_lower = user_question.lower()
            is_scientific = any(word in question_lower for word in [
                'come', 'perché', 'meccanismo', 'funziona', 'scienza', 'fisica', 'chimica', 
                'biologia', 'matematica', 'tecnologia', 'ricerca', 'studio', 'teoria'
            ])
            
            if is_scientific:
                max_tokens = 600
                temperature = 0.3
                additional_prompt = "\n\nQuesta domanda richiede una spiegazione scientifica approfondita. Fornisci dettagli tecnici, meccanismi, e esempi pratici."
            else:
                max_tokens = 400
                temperature = 0.5
                additional_prompt = "\n\nFornisci una risposta completa e ben strutturata."
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": scientific_prompt},
                    {"role": "user", "content": user_question + additional_prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                presence_penalty=0.1,
                frequency_penalty=0.1
            )
            
            full_response = response.choices[0].message.content.strip()
            return full_response
            
        except Exception as e:
            return f"Mi dispiace, c'è stato un errore nel contattare OpenAI: {str(e)}"
    
    def should_use_web_search(self, question):
        """Determina se una domanda richiede ricerca web per informazioni aggiornate"""
        web_search_indicators = [
            'recente', 'nuovo', '2024', '2025', '2026', 'attuale', 'oggi', 'ora', 
            'scoperta', 'ricerca recente', 'studio recente', 'ultimi',
            'notizie', 'sviluppi', 'aggiornamenti', 'tendenze'
        ]
        
        question_lower = question.lower()
        return any(indicator in question_lower for indicator in web_search_indicators)
    
    def enhanced_response_with_research(self, user_question):
        """Risposta potenziata con eventuale ricerca web"""
        try:
            needs_web = self.should_use_web_search(user_question)
            
            if needs_web:
                web_context = "\n\n[NOTA: Per informazioni più aggiornate su questo argomento, ti consiglio di verificare fonti scientifiche recenti come PubMed, Nature, o Science.]"
            else:
                web_context = ""
            
            main_response = self.get_openai_response(user_question)
            
            if web_context:
                enhanced_response = main_response + web_context
            else:
                enhanced_response = main_response
                
            return enhanced_response
            
        except Exception as e:
            return f"Errore nell'elaborazione avanzata: {str(e)}"
    
    # =========================================================================
    # INTERAZIONE COMPLETA (testo O immagine)
    # =========================================================================
    
    def complete_interaction(self, user_question):
        """Completa un'interazione: distingue tra richiesta di disegno e domanda testuale"""
        try:
            # ==========================================
            # STAMPA DOMANDA SUL TERMINALE
            # ==========================================
            print(f"\n{'='*70}")
            print(f"  🗣️  DOMANDA: {user_question}")
            print(f"{'='*70}")
            
            # Determina se è una richiesta di disegno
            if self.is_draw_request(user_question):
                # ===== MODALITÀ DISEGNO (DALL-E) =====
                image_prompt = self.extract_image_prompt(user_question)
                
                self.speak_text(f"Sto generando un'immagine di {image_prompt}. Attendi qualche secondo.")
                
                image_path, revised_prompt, image_url = self.generate_image(image_prompt)
                
                if image_path:
                    response_text = f"Immagine generata: '{image_prompt}' → {image_path}"
                    
                    # STAMPA RISPOSTA SUL TERMINALE
                    print(f"  🎨 RISPOSTA: Immagine generata con successo!")
                    print(f"  📁 File: {image_path}")
                    if revised_prompt:
                        print(f"  📝 Prompt DALL-E: {revised_prompt}")
                    print(f"{'='*70}\n")
                    
                    # LOG: solo domanda e risposta
                    self.log_conversation(user_question, response_text, 
                                         response_type="image", image_path=image_path)
                    
                    self.speak_text(f"Ecco fatto! Ho creato l'immagine di {image_prompt}. Puoi vederla nella finestra che si è aperta.")
                else:
                    error_text = f"Errore generazione: {revised_prompt}"
                    
                    print(f"  ❌ RISPOSTA: {error_text}")
                    print(f"{'='*70}\n")
                    
                    self.log_conversation(user_question, error_text, response_type="image_error")
                    self.speak_text("Mi dispiace, non sono riuscito a generare l'immagine. Riprova.")
                
            else:
                # ===== MODALITÀ RISPOSTA TESTUALE =====
                response = self.enhanced_response_with_research(user_question)
                
                # STAMPA RISPOSTA SUL TERMINALE
                print(f"  🤖 RISPOSTA: {response}")
                print(f"{'='*70}\n")
                
                # LOG: solo domanda e risposta
                self.log_conversation(user_question, response, response_type="text")
                
                if len(response) > 800:
                    speech_response = response[:800] + "... continua la lettura sul terminale."
                else:
                    speech_response = response
                
                if not self.shutdown:
                    speech_thread = self.speak_text(speech_response)
                    if speech_thread:
                        speech_thread.join()
            
            # Cooldown dopo l'interazione
            print(f"💤 Pausa di {self.cooldown_after_interaction} secondi...")
            time.sleep(self.cooldown_after_interaction)
            
        finally:
            self.is_processing = False
            self.last_question_time = time.time()
    
    # =========================================================================
    # LOOP PRINCIPALE
    # =========================================================================
    
    def run(self):
        """Loop principale dell'applicazione"""
        clock = pygame.time.Clock()
        last_wake_check = 0
        
        try:
            while True:
                current_time = time.time()
                
                # Gestisce gli eventi di pygame
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                            print("👋 Chiusura richiesta dal genio...")
                            return
                        elif event.key == pygame.K_v:
                            self.show_video = not self.show_video
                            if not self.show_video:
                                cv2.destroyWindow('AI Assistant - Webcam (Riconoscimento Attivo)')
                            print(f"📺 Video: {'ON' if self.show_video else 'OFF'}")
                        elif event.key == pygame.K_p:
                            self.is_paused = not self.is_paused
                            status = "in pausa 😴" if self.is_paused else "attivo 🔥"
                            print(f"⏯️ Genio ora è {status}")
                        elif event.key == pygame.K_SPACE and self.face_detected and not self.is_processing and not self.is_paused and self.authorized_user_detected:
                            def manual_interaction():
                                user_question = self.listen_to_user()
                                if user_question and not self.shutdown:
                                    self.complete_interaction(user_question)
                            
                            thread = threading.Thread(target=manual_interaction, daemon=True)
                            try:
                                thread.start()
                            except RuntimeError:
                                pass
                
                # SE IN PAUSA: Ascolta solo per comandi di risveglio
                if self.is_paused:
                    if current_time - last_wake_check > 2:
                        def wake_listener():
                            self.listen_for_wake_command()
                        
                        thread = threading.Thread(target=wake_listener, daemon=True)
                        try:
                            thread.start()
                        except RuntimeError:
                            pass
                        last_wake_check = current_time
                
                # SE ATTIVO: Logica normale
                elif not self.is_paused:
                    ret, frame = self.cap.read()
                    if ret:
                        faces_detected, authorized_user, processed_frame = self.detect_faces(frame)
                        self.face_detected = faces_detected
                        self.authorized_user_detected = authorized_user
                        
                        if self.show_video:
                            cv2.imshow('AI Assistant - Webcam (Riconoscimento Attivo)', processed_frame)
                            key = cv2.waitKey(1) & 0xFF
                            if key == ord('q'):
                                break
                        else:
                            cv2.waitKey(1)
                        
                        # LOGICA ANTI-ACCAVALLAMENTO (solo se utente autorizzato)
                        if (authorized_user and 
                            current_time - self.last_question_time > self.question_interval and
                            not self.is_listening and 
                            not self.is_speaking and 
                            not self.is_processing):
                            
                            self.ask_random_question()
                            
                            def auto_interaction():
                                time.sleep(3)
                                if not self.shutdown and not self.is_listening and self.is_processing and not self.is_paused and self.authorized_user_detected:
                                    user_question = self.listen_to_user()
                                    if user_question and not self.shutdown:
                                        self.complete_interaction(user_question)
                                    else:
                                        self.is_processing = False
                                        self.last_question_time = time.time()
                            
                            thread = threading.Thread(target=auto_interaction, daemon=True)
                            try:
                                thread.start()
                            except RuntimeError:
                                pass
                
                self.update_genie_animation()
                clock.tick(30)
                
        except KeyboardInterrupt:
            print("\n👋 Arrivederci!")
        finally:
            self.shutdown = True
            time.sleep(0.5)
            self.cleanup()
    
    def cleanup(self):
        """Pulisce le risorse e chiude la sessione di logging"""
        print("🧹 Pulizia risorse...")
        self.shutdown = True
        
        if self.current_session:
            self.current_session["end_time"] = datetime.now().isoformat()
            self.current_session["total_conversations"] = len(self.current_session["conversations"])
            self.save_session_log()
        
        if hasattr(self, 'cap') and self.cap:
            self.cap.release()
        
        cv2.destroyAllWindows()
        pygame.quit()
        
        print("✅ Pulizia completata!")
        if self.current_session:
            print(f"📝 Log salvato in: {self.current_session['log_file']}")


def main():
    print("🚀 Avvio AI Assistant con Genio (v12 - Disegno + Log pulito)...")
    
    api_key = os.getenv('OPENAI_API_KEY')
    
    if not api_key:
        print("❌ Chiave API non trovata nel file .env!")
        print("📁 Crea un file .env con:")
        print("   OPENAI_API_KEY=sk-tua-chiave-qui")
        
        api_key = input("🔑 Oppure inserisci la tua OpenAI API key ora: ").strip()
        
        if not api_key:
            print("❌ Chiave API richiesta!")
            return
    
    reference_image = "anto2025.png"
    if os.path.exists(reference_image):
        print(f"🔐 Immagine di riferimento trovata: {reference_image}")
        print("🔒 Modalità sicurezza ATTIVA - solo utente autorizzato")
    else:
        print(f"⚠️ Immagine di riferimento non trovata: {reference_image}")
        print("🔓 Modalità aperta - qualsiasi viso può attivare il sistema")
        response = input("Vuoi continuare? (s/n): ")
        if response.lower() not in ['s', 'si', 'sì', 'y', 'yes']:
            print("👋 Operazione annullata")
            return
    
    try:
        assistant = AIAssistantWithGenie(api_key, reference_image)
        print("\n" + "="*70)
        print("  💡 CONTROLLI:")
        print("  SPAZIO: Parla manualmente | Q/ESC: Esci")
        print("  V: Nascondi/mostra video  | P: Toggle pausa")
        print("  ")
        print("  🎤 COMANDI VOCALI:")
        print("  Conferma: SÌ, OK, VAI   | Rifiuta: NO, ANNULLA")
        print("  Pausa: PAUSA, STOP, DORMI | Sveglia: GENIO, SVEGLIA, HEY")
        print("  ")
        print("  🎨 GENERAZIONE IMMAGINI:")
        print("  Di: 'disegna...', 'genera immagine di...', 'illustra...'")
        print("  Le immagini vengono salvate in 'generated_images/'")
        print("  ")
        print("  📝 LOG: solo domande e risposte in 'log/'")
        print("="*70 + "\n")
        
        assistant.run()
        
    except Exception as e:
        print(f"❌ Errore: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
