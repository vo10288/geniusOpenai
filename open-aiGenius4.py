#!/usr/bin/env python3
"""
AI Assistant con riconoscimento facciale, speech-to-text e genio animato
Requisiti: macOS (per il comando 'say'), webcam, microfono
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

# Carica le variabili d'ambiente dal file .env
load_dotenv()

class AIAssistantWithGenie:
    def __init__(self, openai_api_key):
        # Inizializza OpenAI - VERSIONE CORRETTA
        self.client = OpenAI(api_key=openai_api_key)
        
        # Inizializza il riconoscitore vocale
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        # Inizializza la webcam
        self.cap = cv2.VideoCapture(0)
        
        # Threading locks per evitare conflitti
        self.microphone_lock = threading.Lock()
        self.speech_lock = threading.Lock()
        
        # Variabili di stato
        self.face_detected = False
        self.last_question_time = 0
        self.question_interval = 10  # secondi tra le domande
        self.is_listening = False
        self.is_speaking = False
        self.shutdown = False  # Flag per chiusura pulita
        # Carica il classificatore per il riconoscimento facciale - VERSION FIX
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
                print(f"✅ Classificatore facciale caricato da: {cascade_path}")
        else:
            print("❌ Impossibile caricare il classificatore facciale")
            self.face_cascade = None
        
        # Liste di domande casuali
        self.random_questions = [
            "Come posso soddisfare la tua curiosità?",
            "Cosa posso fare per te?",
            "Hai qualcosa da chiedermi?",
            "Di cosa vorresti parlare oggi?",
            "C'è qualcosa che ti incuriosisce?",
            "Quale domanda ti frulla per la testa?",
            "Come posso esserti utile?",
            "Cosa desideri sapere?",
            "C'è qualcosa che posso spiegarti?",
            "Quale argomento ti interessa di più?"
        ]
        
        # Inizializza pygame per l'animazione del genio
        pygame.init()
        self.screen_width, self.screen_height = 400, 300
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Genio AI Assistant")
        
        # Variabili per l'animazione del genio
        self.genie_x = self.screen_width // 2
        self.genie_y = self.screen_height // 2
        self.animation_frame = 0
        
        print("✅ AI Assistant inizializzato! Premi 'q' per uscire.")
        
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
        self.screen.fill((20, 20, 40))  # Sfondo blu scuro
        
        # Crea e disegna il genio
        genie_surface = self.create_genie_surface(self.animation_frame)
        
        # Movimento fluttuante
        float_offset_x = int(10 * math.sin(self.animation_frame * 0.05))
        float_offset_y = int(15 * math.cos(self.animation_frame * 0.03))
        
        self.screen.blit(genie_surface, 
                        (self.genie_x - 50 + float_offset_x, 
                         self.genie_y - 50 + float_offset_y))
        
        # Testo di stato
        font = pygame.font.Font(None, 24)
        if self.face_detected:
            text = font.render("Viso rilevato!", True, (0, 255, 0))
        else:
            text = font.render("In attesa...", True, (255, 255, 255))
        
        self.screen.blit(text, (10, 10))
        
        if self.is_listening:
            listen_text = font.render("🎤 In ascolto...", True, (255, 255, 0))
            self.screen.blit(listen_text, (10, 40))
        
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
        """Fa una domanda casuale"""
        if not self.is_listening and not self.is_speaking:
            question = random.choice(self.random_questions)
            print(f"\n🧞‍♂️ Genio: {question}")
            self.speak_text(question)
            self.last_question_time = time.time()
    
    def listen_to_user(self):
        """Ascolta l'input dell'utente dal microfono"""
        if self.shutdown or self.is_listening:
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
                
                # Chiede conferma
                confirm = input("❓ Vuoi inviare questa domanda a OpenAI? (s/n): ")
                
                if confirm.lower() in ['s', 'si', 'sì', 'y', 'yes']:
                    return user_text
                else:
                    print("❌ Richiesta annullata.")
                    return None
                    
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
    
    def get_openai_response(self, user_question):
        """Ottiene una risposta da OpenAI"""
        try:
            print("🤖 Sto chiedendo a OpenAI...")
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Sei un assistente AI amichevole e utile. Rispondi in modo conciso e chiaro in italiano."},
                    {"role": "user", "content": user_question}
                ],
                max_tokens=200,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"Mi dispiace, c'è stato un errore nel contattare OpenAI: {str(e)}"
    
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
        
        try:
            while True:
                # Gestisce gli eventi di pygame
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_SPACE and self.face_detected:
                            # Spacebar per iniziare l'ascolto manuale
                            user_question = self.listen_to_user()
                            if user_question:
                                response = self.get_openai_response(user_question)
                                print(f"\n🤖 OpenAI: {response}")
                                self.speak_text(response)
                
                # Legge frame dalla webcam
                ret, frame = self.cap.read()
                if ret:
                    # Rileva visi
                    faces_detected, processed_frame = self.detect_faces(frame)
                    self.face_detected = faces_detected
                    
                    # Mostra il video
                    cv2.imshow('AI Assistant - Webcam', processed_frame)
                    
                    # Se c'è un viso e è passato abbastanza tempo, fa una domanda
                    current_time = time.time()
                    if (faces_detected and 
                        current_time - self.last_question_time > self.question_interval and
                        not self.is_listening and not self.is_speaking):
                        
                        self.ask_random_question()
                        
                        # Avvia l'ascolto dopo 3 secondi
                        def delayed_listen():
                            time.sleep(3)
                            if not self.shutdown and not self.is_listening and not self.is_speaking:
                                user_question = self.listen_to_user()
                                if user_question and not self.shutdown:
                                    response = self.get_openai_response(user_question)
                                    print(f"\n🤖 OpenAI: {response}")
                                    if not self.shutdown:
                                        self.speak_text(response)
                        
                        thread = threading.Thread(target=delayed_listen, daemon=True)
                        try:
                            thread.start()
                        except RuntimeError:
                            pass  # Ignora errori durante shutdown
                
                # Aggiorna l'animazione del genio
                self.update_genie_animation()
                
                # Controlla se premuto 'q' nella finestra video
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                clock.tick(30)  # 30 FPS
                
        except KeyboardInterrupt:
            print("\n👋 Arrivederci!")
        finally:
            self.shutdown = True  # Segnala a tutti i thread di terminare
            time.sleep(0.5)  # Aspetta un po' per i thread
            self.cleanup()
    
    def cleanup(self):
        """Pulisce le risorse"""
        print("🧹 Pulizia risorse...")
        self.shutdown = True
        
        # Chiude la webcam
        if hasattr(self, 'cap') and self.cap:
            self.cap.release()
        
        # Chiude le finestre OpenCV
        cv2.destroyAllWindows()
        
        # Chiude pygame
        pygame.quit()
        
        print("✅ Pulizia completata!")

def main():
    print("🚀 Avvio AI Assistant con Genio...")
    
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
    
    try:
        # Crea e avvia l'assistente
        assistant = AIAssistantWithGenie(api_key)
        print("💡 Suggerimenti:")
        print("   - Posizionati davanti alla webcam")
        print("   - Il genio ti farà domande automaticamente")
        print("   - Premi SPAZIO nella finestra del genio per parlare manualmente")
        print("   - Premi 'q' nella finestra video per uscire")
        print("\n" + "="*50)
        
        assistant.run()
        
    except Exception as e:
        print(f"❌ Errore: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
