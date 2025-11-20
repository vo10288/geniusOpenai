#!/usr/bin/env python3
"""
AI Assistant con riconoscimento facciale, speech-to-text e genio animato
Versione scientifica avanzata con controllo vocale completo + riconoscimento utente specifico
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
import face_recognition
import json
from datetime import datetime

# Carica le variabili d'ambiente dal file .env
load_dotenv()

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
        self.load_reference_face()
        
        # SISTEMA DI LOGGING
        self.log_dir = "log"
        self.current_session = None
        self.setup_logging()
        
        # Carica il classificatore per il riconoscimento facciale generico
        try:
            # Prova con cv2.data (versioni nuove)
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        except AttributeError:
            # Fallback per versioni vecchie o installazioni incomplete
            import os
            possible_paths = [
                '/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml',
                '/opt/homebrew/share/opencv4/haarcascades/haarcascade_frontalface_default.xml',
                os.path.join(os.path.dirname(cv2.__file__), 'data', 'haarcascade_frontalface_default.xml'),
                'haarcascade_frontalface_default.xml'  # se è nella directory corrente
            ]
            
            cascade_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    cascade_path = path
                    break
            
            if cascade_path is None:
                print("⚠️ File haarcascades non trovato. Scarico da GitHub...")
                # Download del file se non trovato
                import urllib.request
                url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
                cascade_path = "haarcascade_frontalface_default.xml"
                try:
                    urllib.request.urlretrieve(url, cascade_path)
                    print(f"✅ File scaricato in: {cascade_path}")
                except Exception as e:
                    print(f"❌ Impossibile scaricare il file: {e}")
                    cascade_path = None
        
        if cascade_path and os.path.exists(cascade_path):
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            if self.face_cascade.empty():
                print("⚠️ Classificatore facciale non caricato correttamente")
                self.face_cascade = None
            else:
                print(f"✅ Classificatore facciale generico caricato da: {cascade_path}")
        else:
            print("❌ Impossibile caricare il classificatore facciale generico")
            self.face_cascade = None
        
        # Variabili di stato
        self.face_detected = False
        self.authorized_user_detected = False  # Nuovo flag per utente autorizzato
        self.last_question_time = 0
        self.question_interval = 10  # secondi tra le domande
        self.is_listening = False
        self.is_speaking = False
        self.is_processing = False  # Flag per elaborazione OpenAI
        self.is_paused = False  # Flag per stato pausa
        self.shutdown = False  # Flag per chiusura pulita
        self.cooldown_after_interaction = 15  # Pausa dopo ogni interazione completa
        self.show_video = True  # Controlla se mostrare la finestra video
        
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
        
        # Inizializza pygame per l'animazione del genio
        pygame.init()
        self.screen_width, self.screen_height = 400, 300
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Genio AI Assistant - Accesso Sicuro")
        
        # Variabili per l'animazione del genio
        self.genie_x = self.screen_width // 2
        self.genie_y = self.screen_height // 2
        self.animation_frame = 0
        
        print("✅ AI Assistant inizializzato! Premi 'q' per uscire.")
        if self.authorized_face_encoding is not None:
            print("🔐 Riconoscimento facciale personale ATTIVO")
        else:
            print("⚠️ Riconoscimento facciale personale NON DISPONIBILE")
    
    def load_reference_face(self):
        """Carica l'immagine di riferimento dell'utente autorizzato"""
        try:
            if not os.path.exists(self.reference_image_path):
                print(f"❌ Immagine di riferimento non trovata: {self.reference_image_path}")
                print("🔓 Il sistema funzionerà con qualsiasi viso rilevato")
                self.authorized_face_encoding = None
                return
            
            print(f"🔍 Caricamento immagine di riferimento: {self.reference_image_path}")
            
            # Carica l'immagine di riferimento
            reference_image = face_recognition.load_image_file(self.reference_image_path)
            reference_encodings = face_recognition.face_encodings(reference_image)
            
            if len(reference_encodings) == 0:
                print("❌ Nessun viso rilevato nell'immagine di riferimento!")
                print("📋 Assicurati che l'immagine contenga chiaramente un viso")
                self.authorized_face_encoding = None
                return
            
            if len(reference_encodings) > 1:
                print("⚠️ Più visi rilevati nell'immagine, uso il primo")
            
            self.authorized_face_encoding = reference_encodings[0]
            print("✅ Viso di riferimento caricato con successo!")
            print("🔐 Solo l'utente autorizzato può attivare il sistema")
            
        except Exception as e:
            print(f"❌ Errore nel caricamento dell'immagine di riferimento: {e}")
            print("💡 Assicurati di aver installato: pip install face-recognition")
            self.authorized_face_encoding = None
    
    def setup_logging(self):
        """Configura il sistema di logging"""
        try:
            # Crea la directory log se non esiste
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
                print(f"📁 Creata directory log: {self.log_dir}")
            
            # Crea nuovo file di sessione con timestamp
            session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.current_session = {
                "session_id": session_timestamp,
                "start_time": datetime.now().isoformat(),
                "log_file": os.path.join(self.log_dir, f"conversation_{session_timestamp}.json"),
                "conversations": []
            }
            
            # Salva header della sessione
            self.save_session_log()
            print(f"📝 Log sessione: {self.current_session['log_file']}")
            
        except Exception as e:
            print(f"❌ Errore nel setup logging: {e}")
            self.current_session = None
    
    def log_interaction(self, interaction_type, data):
        """Registra un'interazione nel log"""
        if self.current_session is None:
            return
        
        try:
            timestamp = datetime.now().isoformat()
            log_entry = {
                "timestamp": timestamp,
                "type": interaction_type,
                "data": data
            }
            
            self.current_session["conversations"].append(log_entry)
            self.save_session_log()
            
            # Stampa anche su console per debug
            print(f"📝 LOG [{interaction_type}]: {data.get('summary', str(data)[:100])}")
            
        except Exception as e:
            print(f"❌ Errore nel logging: {e}")
    
    def save_session_log(self):
        """Salva il log della sessione su file"""
        if self.current_session is None:
            return
        
        try:
            # Aggiorna timestamp di fine sessione
            self.current_session["last_update"] = datetime.now().isoformat()
            
            # Salva in formato JSON leggibile
            with open(self.current_session["log_file"], 'w', encoding='utf-8') as f:
                json.dump(self.current_session, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            print(f"❌ Errore nel salvataggio log: {e}")
        
        # Liste di domande casuali PIU' SCIENTIFICHE
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
    
    def detect_faces(self, frame):
        """Rileva i visi nel frame e verifica l'autorizzazione dell'utente"""
        faces_detected = False
        authorized_user = False
        
        if self.face_cascade is None:
            # Se il classificatore non è disponibile, simula sempre un viso presente
            cv2.putText(frame, "Face detection disabled", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            return True, True, frame  # Simula sempre presenza di utente autorizzato
        
        try:
            # Rileva visi generici con OpenCV
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            faces_detected = len(faces) > 0
            
            # Se non c'è sistema di riconoscimento personale, autorizza qualsiasi viso
            if self.authorized_face_encoding is None:
                authorized_user = faces_detected
                # Disegna rettangoli gialli per visi non verificati
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
                    cv2.putText(frame, "NON VERIFICATO", (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                return faces_detected, authorized_user, frame
            
            # Sistema di riconoscimento personale attivo
            if faces_detected:
                # Converti frame per face_recognition
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_frame)
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                
                for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                    # Confronta con il viso autorizzato
                    matches = face_recognition.compare_faces([self.authorized_face_encoding], face_encoding, tolerance=0.6)
                    face_distances = face_recognition.face_distance([self.authorized_face_encoding], face_encoding)
                    
                    if matches[0]:
                        authorized_user = True
                        confidence = (1 - face_distances[0]) * 100
                        
                        # Rettangolo verde per utente autorizzato
                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                        cv2.putText(frame, f"AUTORIZZATO ({confidence:.1f}%)", 
                                   (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                        # Log del riconoscimento
                        self.log_interaction("face_recognition", {
                            "authorized": True,
                            "confidence": float(confidence),
                            "summary": f"Utente autorizzato riconosciuto (confidenza: {confidence:.1f}%)"
                        })
                    else:
                        # Rettangolo rosso per utente non autorizzato
                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                        cv2.putText(frame, "NON AUTORIZZATO", 
                                   (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        
                        # Log del tentativo non autorizzato
                        self.log_interaction("unauthorized_access", {
                            "authorized": False,
                            "face_distance": float(face_distances[0]),
                            "summary": "Tentativo di accesso da utente non autorizzato"
                        })
                
        except Exception as e:
            print(f"❌ Errore nel riconoscimento facciale: {e}")
            # Fallback al comportamento standard
            faces_detected = len(faces) > 0 if 'faces' in locals() else False
            authorized_user = faces_detected
        
        return faces_detected, authorized_user, frame
        
    def create_genie_surface(self, frame):
        """Crea una superficie del genio animato"""
        # Crea un cerchio base per il genio (semplificato)
        surface = pygame.Surface((100, 100), pygame.SRCALPHA)
        
        # Corpo del genio (variabile con l'animazione)
        body_size = 40 + int(10 * math.sin(frame * 0.2))
        if self.is_speaking:
            body_size += int(5 * math.sin(frame * 0.5))
        
        # Colori che cambiano con l'animazione
        color_r = int(100 + 50 * math.sin(frame * 0.1))
        color_g = int(100 + 50 * math.cos(frame * 0.15))
        color_b = int(150 + 50 * math.sin(frame * 0.08))
        
        # Disegna il corpo del genio
        pygame.draw.ellipse(surface, (color_r, color_g, color_b), 
                          (50 - body_size//2, 50 - body_size//2, body_size, body_size))
        
        # Occhi
        eye_offset = int(3 * math.sin(frame * 0.3)) if self.is_speaking else 0
        pygame.draw.circle(surface, (255, 255, 255), (40, 40 + eye_offset), 8)
        pygame.draw.circle(surface, (255, 255, 255), (60, 40 + eye_offset), 8)
        pygame.draw.circle(surface, (0, 0, 0), (40, 40 + eye_offset), 4)
        pygame.draw.circle(surface, (0, 0, 0), (60, 40 + eye_offset), 4)
        
        # Bocca (si muove quando parla)
        if self.is_speaking:
            mouth_height = int(5 * abs(math.sin(frame * 0.8)))
            pygame.draw.ellipse(surface, (50, 50, 50), (45, 55, 10, mouth_height + 3))
        else:
            pygame.draw.arc(surface, (50, 50, 50), (45, 55, 10, 5), 0, math.pi, 2)
        
        return surface
    
    def update_genie_animation(self):
        """Aggiorna l'animazione del genio"""
        # Colore di sfondo diverso in base allo stato
        if self.is_paused:
            self.screen.fill((40, 20, 60))  # Sfondo viola per pausa
        elif not self.authorized_user_detected and self.authorized_face_encoding is not None:
            self.screen.fill((60, 20, 20))  # Sfondo rosso per accesso negato
        else:
            self.screen.fill((20, 20, 40))  # Sfondo blu normale
        
        # Crea e disegna il genio (più lento se in pausa o accesso negato)
        animation_speed = 0.02 if (self.is_paused or not self.authorized_user_detected) else 0.05
        genie_surface = self.create_genie_surface(self.animation_frame)
        
        # Movimento fluttuante (ridotto se in pausa o accesso negato)
        float_intensity = 5 if (self.is_paused or not self.authorized_user_detected) else 10
        float_offset_x = int(float_intensity * math.sin(self.animation_frame * animation_speed))
        float_offset_y = int(float_intensity * 1.5 * math.cos(self.animation_frame * animation_speed * 0.6))
        
        self.screen.blit(genie_surface, 
                        (self.genie_x - 50 + float_offset_x, 
                         self.genie_y - 50 + float_offset_y))
        
        # Testo di stato migliorato
        font = pygame.font.Font(None, 24)
        small_font = pygame.font.Font(None, 18)
        
        # Stato principale
        if not self.authorized_user_detected and self.authorized_face_encoding is not None:
            text = font.render("🚫 ACCESSO NEGATO", True, (255, 100, 100))
            self.screen.blit(text, (10, 10))
            auth_text = small_font.render("Solo l'utente autorizzato può accedere", True, (255, 150, 150))
            self.screen.blit(auth_text, (10, 35))
        elif self.is_paused:
            text = font.render("😴 IN PAUSA", True, (255, 150, 255))
            self.screen.blit(text, (10, 10))
            wake_text = small_font.render("Di: 'GENIO' o 'SVEGLIA' per riattivarmi", True, (200, 150, 255))
            self.screen.blit(wake_text, (10, 35))
        elif self.is_processing:
            text = font.render("🤖 Elaborando...", True, (255, 255, 0))
        elif self.face_detected:
            if self.authorized_user_detected:
                text = font.render("👤✅ Utente autorizzato!", True, (0, 255, 0))
            else:
                text = font.render("👤 Viso rilevato", True, (255, 255, 0))
        else:
            text = font.render("👁️ In attesa...", True, (255, 255, 255))
        
        if not (self.is_paused or (not self.authorized_user_detected and self.authorized_face_encoding is not None)):
            self.screen.blit(text, (10, 10))
        
        # Stati aggiuntivi (solo se autorizzato e non in pausa)
        if self.authorized_user_detected and not self.is_paused:
            y_offset = 40
            if self.is_listening:
                listen_text = font.render("🎤 In ascolto...", True, (255, 255, 0))
                self.screen.blit(listen_text, (10, y_offset))
                y_offset += 30
            
            if self.is_speaking:
                speak_text = font.render("🗣️ Parlando...", True, (0, 255, 255))
                self.screen.blit(speak_text, (10, y_offset))
                y_offset += 30
        
        # Controlli (aggiornati)
        if self.authorized_user_detected and not self.is_paused:
            controls_text = small_font.render("SPAZIO: Parla | Q/ESC: Esci | V: Video | Dì 'PAUSA' per dormire", True, (150, 150, 150))
            self.screen.blit(controls_text, (10, self.screen_height - 60))
            
            # Cooldown timer se applicabile
            if self.is_processing:
                cooldown_text = small_font.render("Attendi... no nuove domande", True, (255, 100, 100))
                self.screen.blit(cooldown_text, (10, self.screen_height - 40))
        elif not self.authorized_user_detected and self.authorized_face_encoding is not None:
            # Controlli per accesso negato
            access_controls = small_font.render("Accesso riservato - Riconoscimento facciale attivo", True, (255, 150, 150))
            self.screen.blit(access_controls, (10, self.screen_height - 40))
        elif self.is_paused:
            # Controlli per modalità pausa
            pause_controls = small_font.render("Modalità pausa - Ascolta per comandi di risveglio", True, (255, 150, 255))
            self.screen.blit(pause_controls, (10, self.screen_height - 40))
        
        pygame.display.flip()
        self.animation_frame += 1
    
    def speak_text(self, text):
        """Pronuncia il testo usando il comando 'say' di macOS"""
        if self.shutdown:
            return None
            
        def speak():
            with self.speech_lock:  # Evita sovrapposizioni vocali
                if self.shutdown:
                    return
                self.is_speaking = True
                try:
                    # Usa il comando 'say' di macOS
                    subprocess.run(['say', text], check=True)
                except subprocess.CalledProcessError as e:
                    if not self.shutdown:
                        print(f"Errore nel pronunciare: {text} - {e}")
                except FileNotFoundError:
                    if not self.shutdown:
                        print("Comando 'say' non trovato. Sei su macOS?")
                except Exception as e:
                    if not self.shutdown:
                        print(f"Errore inatteso: {e}")
                finally:
                    self.is_speaking = False
        
        # Avvia in un thread separato per non bloccare l'interfaccia
        thread = threading.Thread(target=speak, daemon=True)
        try:
            thread.start()
            return thread
        except RuntimeError:
            # Se non può creare thread (shutdown), ritorna None
            return None
    
    def ask_random_question(self):
        """Fa una domanda casuale (solo se utente autorizzato)"""
        if (not self.is_listening and not self.is_speaking and not self.is_processing and 
            not self.is_paused and self.authorized_user_detected):
            
            question = random.choice(self.random_questions)
            print(f"\n🧞‍♂️ Genio: {question}")
            
            # Log della domanda
            self.log_interaction("question_asked", {
                "question": question,
                "type": "automatic",
                "summary": f"Domanda automatica: {question[:50]}..."
            })
            
            self.speak_text(question)
            self.last_question_time = time.time()
            self.is_processing = True  # Blocca nuove domande
    
    def check_for_control_commands(self, text):
        """Controlla se il testo contiene comandi di controllo (pausa/risveglio)"""
        text_lower = text.lower()
        
        # Controlla comandi di pausa
        if any(word in text_lower for word in self.pause_words):
            if not self.is_paused:
                print("😴 Comando pausa rilevato!")
                self.is_paused = True
                self.speak_text("Va bene, mi metto in pausa. Dimmi genio o sveglia per riattivarmi.")
                return "PAUSA"
        
        # Controlla comandi di risveglio
        if any(word in text_lower for word in self.wake_words):
            if self.is_paused:
                print("🔥 Comando risveglio rilevato!")
                self.is_paused = False
                self.speak_text("Eccomi! Sono tornato attivo. Come posso aiutarti?")
                return "RISVEGLIO"
        
        return None
    
    def listen_for_wake_command(self):
        """Ascolta continuamente per comandi di risveglio quando in pausa"""
        if self.shutdown or not self.is_paused:
            return None
            
        if not self.microphone_lock.acquire(blocking=False):
            return None
        
        try:
            with self.microphone as source:
                if self.shutdown or not self.is_paused:
                    return None
                
                # Ascolto più veloce per i comandi
                self.recognizer.adjust_for_ambient_noise(source, duration=0.3)
                audio = self.recognizer.listen(source, timeout=3, phrase_time_limit=3)
            
            if self.shutdown:
                return None
                
            try:
                wake_text = self.recognizer.recognize_google(audio, language='it-IT')
                print(f"🔊 (In pausa) Sentito: '{wake_text}'")
                
                # Controlla solo comandi di risveglio
                return self.check_for_control_commands(wake_text)
                    
            except sr.UnknownValueError:
                return None
            except sr.RequestError:
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
            
        # Usa il lock per evitare conflitti con il microfono
        if not self.microphone_lock.acquire(blocking=False):
            print("🎤 Microfono occupato, riprova...")
            return None
        
        try:
            print("\n🎤 Sto ascoltando... Parla ora!")
            self.is_listening = True
            
            with self.microphone as source:
                if self.shutdown:
                    return None
                    
                # Riduce il rumore di fondo
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                # Ascolta per massimo 10 secondi
                audio = self.recognizer.listen(source, timeout=10)
            
            if self.shutdown:
                return None
                
            print("🔄 Sto elaborando quello che hai detto...")
            
            # Converte l'audio in testo
            try:
                user_text = self.recognizer.recognize_google(audio, language='it-IT')
                print(f"📝 Ho sentito: '{user_text}'")
                
                # Log dell'input vocale
                self.log_interaction("voice_input", {
                    "text": user_text,
                    "length": len(user_text),
                    "summary": f"Input vocale: {user_text[:50]}..."
                })
                
                # PRIMA controlla i comandi di controllo
                control_command = self.check_for_control_commands(user_text)
                if control_command:
                    return None  # I comandi di controllo non vanno a OpenAI
                
                # Se non è un comando di controllo, procede con la conferma
                return self.get_voice_confirmation(user_text)
                    
            except sr.UnknownValueError:
                if not self.shutdown:
                    print("❌ Non sono riuscito a capire quello che hai detto.")
                    self.log_interaction("voice_error", {
                        "error": "UnknownValueError",
                        "summary": "Audio non riconosciuto"
                    })
                return None
            except sr.RequestError as e:
                if not self.shutdown:
                    print(f"❌ Errore nel servizio di riconoscimento vocale: {e}")
                    self.log_interaction("voice_error", {
                        "error": str(e),
                        "summary": "Errore servizio riconoscimento vocale"
                    })
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
            print("🗣️ Conferma vocale...")
            self.speak_text("Hai detto: " + user_text + ". Vuoi che lo invii a OpenAI? Rispondi sì o no.")
            
            # Aspetta che finisca di parlare
            time.sleep(3)
            
            print("\n🎤 Dimmi: SÌ o NO...")
            
            with self.microphone as source:
                if self.shutdown:
                    return None
                
                # Ascolta la conferma (timeout più breve)
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = self.recognizer.listen(source, timeout=8)
            
            if self.shutdown:
                return None
                
            print("🔄 Elaborando la tua risposta...")
            
            # Riconosce la conferma
            try:
                confirmation = self.recognizer.recognize_google(audio, language='it-IT').lower()
                print(f"🔊 Conferma ricevuta: '{confirmation}'")
                
                # Parole di conferma
                confirm_words = ['sì', 'si', 'sí', 'yes', 'y', 'vai', 'ok', 'okay', 'invia', 'mandalo', 'perfetto']
                # Parole di rifiuto  
                reject_words = ['no', 'n', 'niente', 'annulla', 'stop', 'basta', 'cancella', 'non', 'nein']
                
                # Controlla se contiene parole di conferma
                if any(word in confirmation for word in confirm_words):
                    print("✅ Confermato! Invio a OpenAI...")
                    self.speak_text("Perfetto! Invio la domanda.")
                    time.sleep(1)
                    return user_text
                
                # Controlla se contiene parole di rifiuto
                elif any(word in confirmation for word in reject_words):
                    print("❌ Richiesta annullata.")
                    self.speak_text("Va bene, annullo la richiesta.")
                    return None
                
                else:
                    print("🤔 Non ho capito la risposta. Annullo per sicurezza.")
                    self.speak_text("Non ho capito. Annullo la richiesta.")
                    return None
                    
            except sr.UnknownValueError:
                print("❌ Non ho sentito una risposta chiara. Annullo.")
                self.speak_text("Non ho sentito una risposta. Annullo.")
                return None
            except sr.RequestError as e:
                print(f"❌ Errore nel riconoscimento: {e}")
                return None
                
        except sr.WaitTimeoutError:
            print("⏰ Tempo scaduto per la conferma. Annullo.")
            self.speak_text("Tempo scaduto. Annullo la richiesta.")
            return None
        except Exception as e:
            if not self.shutdown:
                print(f"❌ Errore conferma vocale: {e}")
            return None
    
    def get_openai_response(self, user_question):
        """Ottiene una risposta da OpenAI con prompt scientifico migliorato"""
        try:
            print("🤖 Sto chiedendo a OpenAI (modalità scientifica)...")
            self.is_processing = True  # Segna che stiamo elaborando
            
            # PROMPT SISTEMA MIGLIORATO per risposte scientifiche
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
            
            # Analisi della domanda per determinare il tipo di risposta
            question_lower = user_question.lower()
            is_scientific = any(word in question_lower for word in [
                'come', 'perché', 'meccanismo', 'funziona', 'scienza', 'fisica', 'chimica', 
                'biologia', 'matematica', 'tecnologia', 'ricerca', 'studio', 'teoria'
            ])
            
            # Parametri dinamici basati sul tipo di domanda
            if is_scientific:
                max_tokens = 600  # Risposte più lunghe per argomenti scientifici
                temperature = 0.3  # Più preciso per informazioni scientifiche
                additional_prompt = "\n\nQuesta domanda richiede una spiegazione scientifica approfondita. Fornisci dettagli tecnici, meccanismi, e esempi pratici."
            else:
                max_tokens = 400
                temperature = 0.5
                additional_prompt = "\n\nFornisci una risposta completa e ben strutturata."
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",  # Modello più avanzato per risposte migliori
                messages=[
                    {"role": "system", "content": scientific_prompt},
                    {"role": "user", "content": user_question + additional_prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                presence_penalty=0.1,  # Incoraggia varietà nei contenuti
                frequency_penalty=0.1   # Riduce ripetizioni
            )
            
            full_response = response.choices[0].message.content.strip()
            
            # Aggiungi nota sulla lunghezza se la risposta è stata troncata
            if len(full_response) > 500:
                print(f"📊 Risposta dettagliata ({len(full_response)} caratteri)")
            
            return full_response
            
        except Exception as e:
            return f"Mi dispiace, c'è stato un errore nel contattare OpenAI: {str(e)}"
    
    def should_use_web_search(self, question):
        """Determina se una domanda richiede ricerca web per informazioni aggiornate"""
        web_search_indicators = [
            'recente', 'nuovo', '2024', '2025', 'attuale', 'oggi', 'ora', 
            'scoperta', 'ricerca recente', 'studio recente', 'ultimi',
            'notizie', 'sviluppi', 'aggiornamenti', 'tendenze'
        ]
        
        question_lower = question.lower()
        return any(indicator in question_lower for indicator in web_search_indicators)
    
    def enhanced_response_with_research(self, user_question):
        """Risposta potenziata con eventuale ricerca web"""
        try:
            # Determina se serve ricerca web
            needs_web = self.should_use_web_search(user_question)
            
            if needs_web:
                print("🔍 Rilevata domanda su argomenti recenti - ricerca web consigliata...")
                # Simula ricerca web (potresti integrare una vera API di ricerca)
                web_context = "\n\n[NOTA: Per informazioni più aggiornate su questo argomento, ti consiglio di verificare fonti scientifiche recenti come PubMed, Nature, o Science.]"
            else:
                web_context = ""
            
            # Ottiene risposta principale dal modello migliorato
            main_response = self.get_openai_response(user_question)
            
            # Aggiunge contesto web se necessario
            if web_context:
                enhanced_response = main_response + web_context
            else:
                enhanced_response = main_response
                
            return enhanced_response
            
        except Exception as e:
            return f"Errore nell'elaborazione avanzata: {str(e)}"
    
    def complete_interaction(self, user_question):
        """Completa un'interazione: Ricerca + OpenAI + risposta vocale + cooldown + logging"""
        interaction_start = datetime.now()
        
        try:
            # Ottieni risposta potenziata (con eventuale ricerca)
            response = self.enhanced_response_with_research(user_question)
            print(f"\n🤖 Risposta Scientifica: {response}")
            
            # Log della conversazione completa
            self.log_interaction("conversation", {
                "user_question": user_question,
                "ai_response": response,
                "question_length": len(user_question),
                "response_length": len(response),
                "start_time": interaction_start.isoformat(),
                "end_time": datetime.now().isoformat(),
                "processing_time": (datetime.now() - interaction_start).total_seconds(),
                "summary": f"Q: {user_question[:50]}... A: {response[:50]}..."
            })
            
            # Per risposte molto lunghe, dividi in chunks per l'audio
            if len(response) > 800:
                print("📢 Risposta lunga - dividendo in sezioni per l'audio...")
                # Prendi solo i primi 800 caratteri per l'audio
                speech_response = response[:800] + "... continua la lettura sul terminale."
            else:
                speech_response = response
            
            # Pronuncia la risposta
            if not self.shutdown:
                speech_thread = self.speak_text(speech_response)
                if speech_thread:
                    speech_thread.join()  # Aspetta che finisca di parlare
            
            # Cooldown dopo l'interazione completa
            print(f"💤 Pausa di {self.cooldown_after_interaction} secondi...")
            time.sleep(self.cooldown_after_interaction)
            
        finally:
            self.is_processing = False  # Libera il sistema per nuove domande
            self.last_question_time = time.time()  # Reset del timer
    
    def detect_faces(self, frame):
        """Rileva i visi nel frame"""
        if self.face_cascade is None:
            # Se il classificatore non è disponibile, simula sempre un viso presente
            cv2.putText(frame, "Face detection disabled", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            return True, frame  # Simula sempre presenza di viso
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        # Disegna rettangoli intorno ai visi
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        return len(faces) > 0, frame
    
    def run(self):
        """Loop principale dell'applicazione"""
        clock = pygame.time.Clock()
        last_wake_check = 0  # Per il controllo periodico di risveglio
        
        try:
            while True:
                current_time = time.time()
                
                # Gestisce gli eventi di pygame
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return
                    elif event.type == pygame.KEYDOWN:
                        # Controlli dalla finestra del genio
                        if event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                            print("👋 Chiusura richiesta dal genio...")
                            return
                        elif event.key == pygame.K_v:
                            # Toggle finestra video
                            self.show_video = not self.show_video
                            if not self.show_video:
                                cv2.destroyWindow('AI Assistant - Webcam')
                            print(f"📺 Video: {'ON' if self.show_video else 'OFF'}")
                        elif event.key == pygame.K_p:
                            # Toggle pausa manuale
                            self.is_paused = not self.is_paused
                            status = "in pausa 😴" if self.is_paused else "attivo 🔥"
                            print(f"⏯️ Genio ora è {status}")
                        elif event.key == pygame.K_SPACE and self.face_detected and not self.is_processing and not self.is_paused and self.authorized_user_detected:
                            # Spacebar per iniziare l'ascolto manuale (solo se autorizzato)
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
                    if current_time - last_wake_check > 2:  # Controlla ogni 2 secondi
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
                    # Legge frame dalla webcam
                    ret, frame = self.cap.read()
                    if ret:
                        # Rileva visi e verifica autorizzazione
                        faces_detected, authorized_user, processed_frame = self.detect_faces(frame)
                        self.face_detected = faces_detected
                        self.authorized_user_detected = authorized_user
                        
                        # Mostra il video solo se abilitato
                        if self.show_video:
                            cv2.imshow('AI Assistant - Webcam (Riconoscimento Attivo)', processed_frame)
                            # Controlla se premuto 'q' nella finestra video
                            key = cv2.waitKey(1) & 0xFF
                            if key == ord('q'):
                                break
                        else:
                            # Se il video è nascosto, gestisci comunque waitKey per OpenCV
                            cv2.waitKey(1)
                        
                        # LOGICA ANTI-ACCAVALLAMENTO (solo se utente autorizzato):
                        if (authorized_user and 
                            current_time - self.last_question_time > self.question_interval and
                            not self.is_listening and 
                            not self.is_speaking and 
                            not self.is_processing):
                            
                            self.ask_random_question()
                            
                            # Avvia l'interazione completa in un thread separato
                            def auto_interaction():
                                time.sleep(3)  # Aspetta che finisca di parlare
                                if not self.shutdown and not self.is_listening and self.is_processing and not self.is_paused and self.authorized_user_detected:
                                    user_question = self.listen_to_user()
                                    if user_question and not self.shutdown:
                                        self.complete_interaction(user_question)
                                    else:
                                        # Se non ha ricevuto input, libera il sistema
                                        self.is_processing = False
                                        self.last_question_time = time.time()
                            
                            thread = threading.Thread(target=auto_interaction, daemon=True)
                            try:
                                thread.start()
                            except RuntimeError:
                                pass
                
                # Aggiorna sempre l'animazione del genio
                self.update_genie_animation()
                
                clock.tick(30)  # 30 FPS
                
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
        
        # Log della fine sessione
        if self.current_session:
            self.log_interaction("session_end", {
                "end_time": datetime.now().isoformat(),
                "duration": (datetime.now() - datetime.fromisoformat(self.current_session["start_time"])).total_seconds(),
                "total_interactions": len(self.current_session["conversations"]),
                "summary": "Sessione terminata"
            })
        
        # Chiude la webcam
        if hasattr(self, 'cap') and self.cap:
            self.cap.release()
        
        # Chiude le finestre OpenCV
        cv2.destroyAllWindows()
        
        # Chiude pygame
        pygame.quit()
        
        print("✅ Pulizia completata!")
        if self.current_session:
            print(f"📝 Log salvato in: {self.current_session['log_file']}")

def main():
    print("🚀 Avvio AI Assistant con Genio Sicuro...")
    
    # Legge la chiave API dal file .env
    api_key = os.getenv('OPENAI_API_KEY')
    
    if not api_key:
        print("❌ Chiave API non trovata nel file .env!")
        print("📁 Crea un file .env con:")
        print("   OPENAI_API_KEY=sk-tua-chiave-qui")
        
        # Fallback: chiede manualmente
        api_key = input("🔑 Oppure inserisci la tua OpenAI API key ora: ").strip()
        
        if not api_key:
            print("❌ Chiave API richiesta!")
            return
    
    # Controlla se esiste l'immagine di riferimento
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
        # Crea e avvia l'assistente
        assistant = AIAssistantWithGenie(api_key, reference_image)
        print("💡 Suggerimenti:")
        print("   - Posizionati davanti alla webcam")
        if assistant.authorized_face_encoding is not None:
            print("   🔐 Solo l'utente autorizzato può attivare il sistema")
        print("   - Il genio ti farà domande automaticamente")
        print("   🎮 CONTROLLI (finestra del genio):")
        print("     • SPAZIO: Parla manualmente (solo se autorizzato)")
        print("     • Q o ESC: Esci dal programma")
        print("     • V: Nascondi/mostra finestra video")
        print("     • P: Toggle pausa manuale")
        print("   📺 CONTROLLI (finestra video):")
        print("     • Q: Esci dal programma")
        print("   🎤 ESPERIENZA HANDS-FREE:")
        print("     • Fai la tua domanda vocalmente")
        print("     • Conferma con: SÌ, YES, OK, VAI")
        print("     • Rifiuta con: NO, ANNULLA, STOP")
        print("   😴 COMANDI VOCALI DI CONTROLLO:")
        print("     • Per PAUSA: 'PAUSA', 'STOP', 'FERMATI', 'SILENZIO', 'DORMI'")
        print("     • Per RISVEGLIO: 'GENIO', 'SVEGLIA', 'RISVEGLIA', 'CONTINUA', 'HEY'")
        print("   🧪 MODALITÀ SCIENTIFICA AVANZATA:")
        print("     • GPT-4o-mini per risposte di alta qualità")
        print("     • Prompt specializzato per spiegazioni scientifiche dettagliate")
        print("     • Risposte fino a 600 token per argomenti complessi")
        print("     • Rilevamento automatico domande scientifiche")
        print("     • Divisione automatica risposte lunghe per l'audio")
        print("     • Domande più specifiche e tecniche")
        print("   📝 SISTEMA DI LOGGING:")
        print("     • Tutte le conversazioni vengono registrate")
        print("     • Log salvati in cartella 'log' con timestamp")
        print("     • Tracking completo delle interazioni")
        if assistant.authorized_face_encoding is not None:
            print("   🔒 SICUREZZA AVANZATA:")
            print("     • Riconoscimento facciale personale attivo")
            print("     • Accesso limitato all'utente autorizzato")
            print("     • Log dei tentativi di accesso non autorizzato")
        print("   ⚙️  LOGICA AVANZATA:")
        print("     • Nessun accavallamento di domande")
        print("     • Pausa automatica di 15 secondi dopo ogni risposta")
        print("     • Controllo vocale completo")
        print("     • Modalità pausa intelligente")
        print("     • Indicatori di stato in tempo reale")
        print("     • Rilevamento automatico argomenti che richiedono ricerca web")
        print("\n" + "="*90)
        
        assistant.run()
        
    except Exception as e:
        print(f"❌ Errore: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
