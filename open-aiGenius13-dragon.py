#!/usr/bin/env python3
"""
AI Assistant v13 - Drago Procedurale sul Desktop
=================================================
- Drago procedurale con sistema particellare (fuoco, fumo, scintille)
- Widget desktop che cammina sui bordi dello schermo
- Riconoscimento emozioni dal volto → cambia umore del drago
- Animazioni reattive (fuoco quando parla, dorme quando in pausa)
- Modalità traduttore simultaneo (italiano ↔ inglese)
- Cronologia ricercabile delle conversazioni con GUI
- Generazione immagini con DALL-E 3
- Riconoscimento facciale personale (anto2025.png)
- Modalità scientifica avanzata (GPT-4o-mini)
- Controllo vocale + wake/sleep
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
import tkinter as tk
from tkinter import ttk, scrolledtext

load_dotenv()

try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False

import logging
logging.getLogger("face_recognition").setLevel(logging.WARNING)
logging.getLogger("dlib").setLevel(logging.WARNING)


# =============================================================================
# SISTEMA PARTICELLARE
# =============================================================================

class Particle:
    """Singola particella del sistema"""
    __slots__ = ['x', 'y', 'vx', 'vy', 'life', 'max_life', 'size', 'color_start', 'color_end', 'particle_type']
    
    def __init__(self, x, y, vx, vy, life, size, color_start, color_end, particle_type="fire"):
        self.x, self.y = x, y
        self.vx, self.vy = vx, vy
        self.life = self.max_life = life
        self.size = size
        self.color_start = color_start
        self.color_end = color_end
        self.particle_type = particle_type


class ParticleSystem:
    """Sistema particellare per fuoco, fumo e scintille"""
    
    def __init__(self, max_particles=250):
        self.particles = []
        self.max_particles = max_particles
    
    def emit_fire(self, x, y, count=5, intensity=1.0):
        """Emette particelle di fuoco"""
        for _ in range(count):
            vx = random.uniform(-1.5, 1.5) * intensity
            vy = random.uniform(-4, -1.5) * intensity
            life = random.uniform(15, 40)
            size = random.uniform(2, 6) * intensity
            # Giallo → Arancione → Rosso
            color_start = (255, random.randint(180, 255), 0, 255)
            color_end = (200, 0, 0, 0)
            self._add(Particle(x, y, vx, vy, life, size, color_start, color_end, "fire"))
    
    def emit_smoke(self, x, y, count=2):
        """Emette particelle di fumo"""
        for _ in range(count):
            vx = random.uniform(-0.8, 0.8)
            vy = random.uniform(-2, -0.5)
            life = random.uniform(30, 60)
            size = random.uniform(3, 8)
            gray = random.randint(80, 150)
            color_start = (gray, gray, gray, 120)
            color_end = (gray, gray, gray, 0)
            self._add(Particle(x, y, vx, vy, life, size, color_start, color_end, "smoke"))
    
    def emit_sparkle(self, x, y, count=3):
        """Emette scintille"""
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed
            life = random.uniform(10, 25)
            size = random.uniform(1, 3)
            color_start = (255, 255, random.randint(100, 255), 255)
            color_end = (255, 200, 0, 0)
            self._add(Particle(x, y, vx, vy, life, size, color_start, color_end, "sparkle"))
    
    def emit_zzz(self, x, y, count=1):
        """Emette 'Z' per lo stato dormiente"""
        for _ in range(count):
            vx = random.uniform(-0.3, 0.3)
            vy = random.uniform(-1.5, -0.5)
            life = random.uniform(40, 70)
            size = random.uniform(4, 8)
            color_start = (150, 150, 255, 200)
            color_end = (100, 100, 200, 0)
            self._add(Particle(x, y, vx, vy, life, size, color_start, color_end, "zzz"))
    
    def _add(self, p):
        if len(self.particles) < self.max_particles:
            self.particles.append(p)
    
    def update(self):
        """Aggiorna tutte le particelle"""
        alive = []
        for p in self.particles:
            p.life -= 1
            if p.life > 0:
                p.x += p.vx
                p.y += p.vy
                if p.particle_type == "smoke":
                    p.vx += random.uniform(-0.1, 0.1)
                    p.size += 0.05
                elif p.particle_type == "fire":
                    p.vy -= 0.05  # accelera verso l'alto
                alive.append(p)
        self.particles = alive
    
    def draw(self, surface):
        """Disegna tutte le particelle"""
        for p in self.particles:
            t = 1 - (p.life / p.max_life)  # 0→1 nel tempo
            r = int(p.color_start[0] + (p.color_end[0] - p.color_start[0]) * t)
            g = int(p.color_start[1] + (p.color_end[1] - p.color_start[1]) * t)
            b = int(p.color_start[2] + (p.color_end[2] - p.color_start[2]) * t)
            a = int(p.color_start[3] + (p.color_end[3] - p.color_start[3]) * t)
            
            r, g, b, a = max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b)), max(0, min(255, a))
            size = max(1, int(p.size * (p.life / p.max_life)))
            
            if p.particle_type == "zzz" and p.life > 10:
                font = pygame.font.Font(None, int(size * 4))
                z_text = font.render("Z", True, (r, g, b))
                z_text.set_alpha(a)
                surface.blit(z_text, (int(p.x), int(p.y)))
            else:
                particle_surf = pygame.Surface((size * 2, size * 2), pygame.SRCALPHA)
                pygame.draw.circle(particle_surf, (r, g, b, a), (size, size), size)
                surface.blit(particle_surf, (int(p.x - size), int(p.y - size)))


# =============================================================================
# DRAGO PROCEDURALE
# =============================================================================

class ProceduralDragon:
    """Drago animato generato proceduralmente"""
    
    # Umore → colori
    MOODS = {
        "neutral":  {"body": (40, 120, 40),  "belly": (80, 160, 80),  "eye": (255, 200, 0)},
        "happy":    {"body": (50, 180, 50),  "belly": (100, 220, 100),"eye": (255, 255, 100)},
        "sleeping": {"body": (30, 60, 80),   "belly": (50, 90, 110),  "eye": (100, 100, 150)},
        "speaking": {"body": (180, 60, 20),  "belly": (220, 100, 40), "eye": (255, 100, 0)},
        "angry":    {"body": (150, 20, 20),  "belly": (200, 50, 30),  "eye": (255, 0, 0)},
        "thinking": {"body": (60, 60, 150),  "belly": (90, 90, 200),  "eye": (150, 150, 255)},
    }
    
    def __init__(self):
        self.mood = "neutral"
        self.frame = 0
        self.facing_right = True
        self.wing_angle = 0
        self.tail_wave = 0
        self.mouth_open = 0
        self.breath_intensity = 0
        self.eye_blink_timer = 0
        self.eye_is_blinking = False
    
    def set_mood(self, mood):
        if mood in self.MOODS:
            self.mood = mood
    
    def update(self, is_speaking=False, is_sleeping=False, is_processing=False):
        self.frame += 1
        
        # Ali
        if is_sleeping:
            self.wing_angle = 5 * math.sin(self.frame * 0.02)
        else:
            speed = 0.15 if is_speaking else 0.08
            amplitude = 25 if is_speaking else 15
            self.wing_angle = amplitude * math.sin(self.frame * speed)
        
        # Coda
        self.tail_wave = 10 * math.sin(self.frame * 0.06)
        
        # Bocca
        if is_speaking:
            self.mouth_open = 8 + 5 * abs(math.sin(self.frame * 0.3))
            self.breath_intensity = min(1.0, self.breath_intensity + 0.1)
        else:
            self.mouth_open = max(0, self.mouth_open - 1)
            self.breath_intensity = max(0, self.breath_intensity - 0.05)
        
        # Blink
        self.eye_blink_timer += 1
        if self.eye_blink_timer > random.randint(80, 200):
            self.eye_is_blinking = True
            self.eye_blink_timer = 0
        if self.eye_is_blinking:
            self.eye_blink_timer += 1
            if self.eye_blink_timer > 6:
                self.eye_is_blinking = False
                self.eye_blink_timer = 0
        
        # Mood auto
        if is_sleeping:
            self.set_mood("sleeping")
        elif is_speaking:
            self.set_mood("speaking")
        elif is_processing:
            self.set_mood("thinking")
    
    def draw(self, surface, cx, cy, scale=1.0):
        """Disegna il drago centrato su (cx, cy)"""
        colors = self.MOODS.get(self.mood, self.MOODS["neutral"])
        body_col = colors["body"]
        belly_col = colors["belly"]
        eye_col = colors["eye"]
        
        flip = 1 if self.facing_right else -1
        bob = int(4 * math.sin(self.frame * 0.04))
        s = scale
        
        # --- CODA (dietro il corpo) ---
        tail_points = []
        for i in range(8):
            t = i / 7
            tx = cx - flip * int((30 + i * 12) * s)
            ty = cy + int((10 + i * 5) * s) + bob + int(self.tail_wave * t * math.sin(self.frame * 0.06 + t * 3))
            tail_points.append((tx, ty))
        if len(tail_points) > 1:
            dark_body = tuple(max(0, c - 30) for c in body_col)
            pygame.draw.lines(surface, dark_body, False, tail_points, max(1, int(4 * s)))
            # Punta della coda (triangolo)
            last = tail_points[-1]
            spike_size = int(6 * s)
            spike = [
                last,
                (last[0] - flip * spike_size, last[1] - spike_size),
                (last[0] - flip * spike_size, last[1] + spike_size),
            ]
            pygame.draw.polygon(surface, (180, 50, 20), spike)
        
        # --- ALI ---
        wing_base_x = cx - flip * int(5 * s)
        wing_base_y = cy - int(15 * s) + bob
        wing_tip_offset = self.wing_angle * s
        
        wing_pts = [
            (wing_base_x, wing_base_y),
            (wing_base_x - flip * int(10 * s), wing_base_y - int((35 + wing_tip_offset) * s)),
            (wing_base_x + flip * int(15 * s), wing_base_y - int((20 + wing_tip_offset * 0.6) * s)),
            (wing_base_x + flip * int(5 * s), wing_base_y - int(5 * s)),
        ]
        # Membrana ala
        wing_membrane = tuple(min(255, c + 40) for c in body_col) + (120,)
        wing_surf = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
        pygame.draw.polygon(wing_surf, wing_membrane, wing_pts)
        surface.blit(wing_surf, (0, 0))
        # Bordo ala
        pygame.draw.lines(surface, body_col, False, wing_pts, max(1, int(2 * s)))
        
        # --- CORPO ---
        body_rect = pygame.Rect(
            cx - int(22 * s), cy - int(18 * s) + bob,
            int(44 * s), int(36 * s)
        )
        pygame.draw.ellipse(surface, body_col, body_rect)
        
        # Pancia
        belly_rect = pygame.Rect(
            cx - int(14 * s), cy - int(6 * s) + bob,
            int(28 * s), int(22 * s)
        )
        pygame.draw.ellipse(surface, belly_col, belly_rect)
        
        # Scaglie sulla schiena
        for i in range(5):
            sx = cx + flip * int((-10 + i * 6) * s)
            sy = cy - int(18 * s) + bob - int(3 * s)
            spike_h = int((4 + 2 * math.sin(self.frame * 0.1 + i)) * s)
            pts = [
                (sx, sy),
                (sx - int(3 * s), sy + spike_h),
                (sx + int(3 * s), sy + spike_h),
            ]
            pygame.draw.polygon(surface, (180, 50, 20), pts)
        
        # --- ZAMPE ---
        for leg_offset in [-12, 8]:
            lx = cx + int(leg_offset * s)
            ly = cy + int(16 * s) + bob
            leg_swing = int(3 * math.sin(self.frame * 0.08 + leg_offset))
            pygame.draw.ellipse(surface, tuple(max(0, c - 20) for c in body_col),
                              (lx - int(5 * s), ly + leg_swing, int(10 * s), int(8 * s)))
        
        # --- TESTA ---
        head_x = cx + flip * int(28 * s)
        head_y = cy - int(12 * s) + bob
        head_size = int(20 * s)
        
        # Collo
        pygame.draw.line(surface, body_col,
                        (cx + flip * int(15 * s), cy - int(10 * s) + bob),
                        (head_x - flip * int(5 * s), head_y + int(5 * s)),
                        max(1, int(10 * s)))
        
        # Testa
        pygame.draw.circle(surface, body_col, (head_x, head_y), head_size)
        
        # Muso
        muzzle_x = head_x + flip * int(14 * s)
        muzzle_y = head_y + int(4 * s)
        pygame.draw.ellipse(surface, tuple(min(255, c + 20) for c in body_col),
                          (muzzle_x - int(10 * s), muzzle_y - int(6 * s),
                           int(20 * s), int(12 * s)))
        
        # Narici
        nostril_glow = int(80 * self.breath_intensity)
        nostril_col = (200 + min(55, nostril_glow), 80, 20)
        for ny_off in [-2, 2]:
            pygame.draw.circle(surface, nostril_col,
                             (muzzle_x + flip * int(7 * s), muzzle_y + int(ny_off * s)),
                             max(1, int(2 * s)))
        
        # Occhio
        eye_x = head_x + flip * int(6 * s)
        eye_y = head_y - int(5 * s)
        eye_size = max(1, int(6 * s))
        
        if self.eye_is_blinking or self.mood == "sleeping":
            # Occhio chiuso
            pygame.draw.line(surface, (30, 30, 30),
                           (eye_x - eye_size, eye_y),
                           (eye_x + eye_size, eye_y), 2)
        else:
            pygame.draw.circle(surface, (240, 240, 220), (eye_x, eye_y), eye_size)
            pupil_size = max(1, int(3 * s))
            pygame.draw.circle(surface, eye_col, (eye_x, eye_y), pupil_size)
            pygame.draw.circle(surface, (0, 0, 0), (eye_x, eye_y), max(1, int(2 * s)))
            # Riflesso
            pygame.draw.circle(surface, (255, 255, 255),
                             (eye_x - max(1, int(2 * s)), eye_y - max(1, int(2 * s))),
                             max(1, int(1.5 * s)))
        
        # Corna
        for horn_side in [-1, 1]:
            hx = head_x + int(horn_side * 8 * s)
            hy = head_y - int(14 * s)
            horn = [
                (hx, head_y - int(8 * s)),
                (hx + int(horn_side * 4 * s), hy),
                (hx - int(horn_side * 2 * s), hy + int(4 * s)),
            ]
            pygame.draw.polygon(surface, (180, 150, 80), horn)
        
        # Bocca / Fiamma
        if self.mouth_open > 2:
            mouth_y = muzzle_y + int(3 * s)
            mouth_w = int(self.mouth_open * s)
            pygame.draw.ellipse(surface, (80, 20, 20),
                              (muzzle_x + flip * int(2 * s) - mouth_w // 2,
                               mouth_y - int(3 * s),
                               mouth_w, int(6 * s)))
        
        # Restituisci posizione bocca per le particelle di fuoco
        fire_x = muzzle_x + flip * int(15 * s)
        fire_y = muzzle_y + int(2 * s)
        return fire_x, fire_y


# =============================================================================
# RICONOSCIMENTO EMOZIONI
# =============================================================================

class EmotionDetector:
    """Rileva emozioni base dal volto usando Haar cascades"""
    
    def __init__(self):
        self.smile_cascade = None
        self.eye_cascade = None
        self.current_emotion = "neutral"
        self.emotion_confidence = 0
        self._load_cascades()
    
    def _load_cascades(self):
        try:
            smile_path = cv2.data.haarcascades + 'haarcascade_smile.xml'
            self.smile_cascade = cv2.CascadeClassifier(smile_path)
            if self.smile_cascade.empty():
                self.smile_cascade = None
        except Exception:
            self.smile_cascade = None
        
        try:
            eye_path = cv2.data.haarcascades + 'haarcascade_eye.xml'
            self.eye_cascade = cv2.CascadeClassifier(eye_path)
            if self.eye_cascade.empty():
                self.eye_cascade = None
        except Exception:
            self.eye_cascade = None
    
    def detect(self, frame, face_rect):
        """Rileva emozione nella regione del viso. face_rect = (x,y,w,h)"""
        if face_rect is None:
            self.current_emotion = "neutral"
            return "neutral"
        
        x, y, w, h = face_rect
        face_roi_gray = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
        
        smile_detected = False
        eyes_detected = True
        
        if self.smile_cascade is not None:
            smiles = self.smile_cascade.detectMultiScale(face_roi_gray, 1.7, 22, minSize=(25, 25))
            smile_detected = len(smiles) > 0
        
        if self.eye_cascade is not None:
            eyes = self.eye_cascade.detectMultiScale(face_roi_gray, 1.1, 5, minSize=(20, 20))
            eyes_detected = len(eyes) >= 1
        
        if smile_detected:
            self.current_emotion = "happy"
        elif not eyes_detected:
            self.current_emotion = "sleeping"
        else:
            self.current_emotion = "neutral"
        
        return self.current_emotion


# =============================================================================
# GUI CRONOLOGIA CONVERSAZIONI
# =============================================================================

class ConversationHistoryGUI:
    """Finestra tkinter per visualizzare e cercare la cronologia"""
    
    def __init__(self, log_dir="log"):
        self.log_dir = log_dir
        self.window = None
        self.is_open = False
    
    def open(self):
        """Apre la finestra della cronologia in un thread separato"""
        if self.is_open:
            return
        thread = threading.Thread(target=self._create_window, daemon=True)
        thread.start()
    
    def _create_window(self):
        self.is_open = True
        root = tk.Tk()
        root.title("📜 Cronologia Conversazioni - Dragon AI")
        root.geometry("700x550")
        root.configure(bg="#1a1a2e")
        
        self.window = root
        
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("Custom.TFrame", background="#1a1a2e")
        style.configure("Custom.TLabel", background="#1a1a2e", foreground="#e0e0e0")
        style.configure("Custom.TEntry", fieldbackground="#16213e", foreground="#e0e0e0")
        style.configure("Custom.TButton", background="#0f3460", foreground="#e0e0e0")
        
        # Header
        header = tk.Label(root, text="🐉 Cronologia Conversazioni", 
                         font=("Helvetica", 16, "bold"),
                         bg="#1a1a2e", fg="#e94560")
        header.pack(pady=10)
        
        # Search frame
        search_frame = tk.Frame(root, bg="#1a1a2e")
        search_frame.pack(fill=tk.X, padx=10)
        
        tk.Label(search_frame, text="🔍 Cerca:", bg="#1a1a2e", fg="#e0e0e0",
                font=("Helvetica", 11)).pack(side=tk.LEFT, padx=5)
        
        self.search_var = tk.StringVar()
        self.search_var.trace("w", lambda *args: self._filter())
        
        search_entry = tk.Entry(search_frame, textvariable=self.search_var,
                               bg="#16213e", fg="#e0e0e0", insertbackground="#e0e0e0",
                               font=("Helvetica", 11), relief=tk.FLAT)
        search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, ipady=4)
        
        # Text area
        self.text_area = scrolledtext.ScrolledText(
            root, wrap=tk.WORD, bg="#16213e", fg="#e0e0e0",
            font=("Courier", 10), relief=tk.FLAT, padx=10, pady=10,
            insertbackground="#e0e0e0"
        )
        self.text_area.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Tags per colorazione
        self.text_area.tag_configure("question", foreground="#e94560", font=("Courier", 10, "bold"))
        self.text_area.tag_configure("answer", foreground="#0cffe1")
        self.text_area.tag_configure("timestamp", foreground="#666680")
        self.text_area.tag_configure("separator", foreground="#333355")
        self.text_area.tag_configure("image_tag", foreground="#ffc107")
        
        # Carica conversazioni
        self.all_conversations = self._load_all()
        self._display(self.all_conversations)
        
        # Status bar
        count = sum(len(s.get("conversations", [])) for s in self.all_conversations)
        status = tk.Label(root, text=f"💬 {count} conversazioni in {len(self.all_conversations)} sessioni",
                         bg="#1a1a2e", fg="#888", font=("Helvetica", 9))
        status.pack(pady=5)
        
        def on_close():
            self.is_open = False
            root.destroy()
        
        root.protocol("WM_DELETE_WINDOW", on_close)
        root.mainloop()
    
    def _load_all(self):
        """Carica tutti i file di log"""
        sessions = []
        if not os.path.exists(self.log_dir):
            return sessions
        
        for f in sorted(os.listdir(self.log_dir), reverse=True):
            if f.endswith(".json"):
                try:
                    with open(os.path.join(self.log_dir, f), 'r', encoding='utf-8') as fh:
                        data = json.load(fh)
                        sessions.append(data)
                except Exception:
                    pass
        return sessions
    
    def _display(self, sessions):
        """Mostra le conversazioni nel text area"""
        self.text_area.config(state=tk.NORMAL)
        self.text_area.delete(1.0, tk.END)
        
        for session in sessions:
            sid = session.get("session_id", "?")
            start = session.get("start_time", "?")[:19]
            self.text_area.insert(tk.END, f"\n{'═'*60}\n", "separator")
            self.text_area.insert(tk.END, f"  Sessione: {sid}  |  {start}\n", "timestamp")
            self.text_area.insert(tk.END, f"{'═'*60}\n", "separator")
            
            for conv in session.get("conversations", []):
                ts = conv.get("timestamp", "")[:19]
                q = conv.get("domanda", conv.get("data", {}).get("user_question", ""))
                r = conv.get("risposta", conv.get("data", {}).get("ai_response", ""))
                tipo = conv.get("tipo", "text")
                
                if not q and not r:
                    continue
                
                self.text_area.insert(tk.END, f"\n  [{ts}]\n", "timestamp")
                self.text_area.insert(tk.END, f"  🗣️ {q}\n", "question")
                
                if tipo == "image":
                    img_path = conv.get("immagine", "")
                    self.text_area.insert(tk.END, f"  🎨 {r}\n", "image_tag")
                    if img_path:
                        self.text_area.insert(tk.END, f"  📁 {img_path}\n", "image_tag")
                else:
                    self.text_area.insert(tk.END, f"  🤖 {r}\n", "answer")
        
        self.text_area.config(state=tk.DISABLED)
    
    def _filter(self):
        """Filtra conversazioni in base al testo di ricerca"""
        query = self.search_var.get().lower().strip()
        if not query:
            self._display(self.all_conversations)
            return
        
        filtered = []
        for session in self.all_conversations:
            matched_convs = []
            for conv in session.get("conversations", []):
                q = conv.get("domanda", "").lower()
                r = conv.get("risposta", "").lower()
                if query in q or query in r:
                    matched_convs.append(conv)
            if matched_convs:
                filtered_session = dict(session)
                filtered_session["conversations"] = matched_convs
                filtered.append(filtered_session)
        
        self._display(filtered)


# =============================================================================
# AI ASSISTANT PRINCIPALE CON DRAGO
# =============================================================================

class DragonAIAssistant:
    def __init__(self, openai_api_key, reference_image_path="anto2025.png"):
        self.client = OpenAI(api_key=openai_api_key)
        
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.cap = cv2.VideoCapture(0)
        
        self.microphone_lock = threading.Lock()
        self.speech_lock = threading.Lock()
        
        # Face recognition
        self.reference_image_path = reference_image_path
        self.authorized_face_encoding = None
        self.face_recognition_enabled = FACE_RECOGNITION_AVAILABLE
        self.load_reference_face()
        
        # Logging
        self.log_dir = "log"
        self.current_session = None
        self.setup_logging()
        
        # Directory immagini
        self.images_dir = "generated_images"
        os.makedirs(self.images_dir, exist_ok=True)
        
        # Face cascade
        self.load_face_cascade()
        
        # Emotion detector
        self.emotion_detector = EmotionDetector()
        self.current_emotion = "neutral"
        self.last_face_rect = None
        
        # Conversation history GUI
        self.history_gui = ConversationHistoryGUI(self.log_dir)
        
        # Drago e particelle
        self.dragon = ProceduralDragon()
        self.particles = ParticleSystem(300)
        
        # Modalità traduttore
        self.translator_mode = False
        
        # Domande casuali
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
        
        # Keywords disegno
        self.draw_keywords = [
            'disegna', 'disegnami', 'genera immagine', "genera un'immagine",
            'crea immagine', "crea un'immagine", 'illustra', 'illustrami',
            'fammi vedere', 'mostrami', 'dipingi', 'dipingimi',
            'draw', 'generate image', 'create image', 'paint',
            'fai un disegno', "fai un'immagine", 'produci immagine'
        ]
        
        # Stato
        self.face_detected = False
        self.authorized_user_detected = False
        self.last_question_time = 0
        self.question_interval = 10
        self.is_listening = False
        self.is_speaking = False
        self.is_processing = False
        self.is_paused = False
        self.shutdown = False
        self.cooldown_after_interaction = 15
        self.show_video = True
        self.is_generating_image = False
        
        # Controllo vocale
        self.pause_words = ['pausa', 'stop', 'fermati', 'silenzio', 'basta', 'dormi', 'sleep', 'pause', 'zitto', 'taci']
        self.wake_words = ['risveglia', 'sveglia', 'wake up', 'continua', 'torna', 'genio', 'ehi genio', 'ciao genio', 'hey', 'start', 'riprendi', 'attivati', 'ci sei', 'drago']
        
        # Desktop border walking
        self.screen_info = None
        self.widget_x = 100
        self.widget_y = 100
        self.border_position = 0.0  # 0→1 lungo il perimetro
        self.border_speed = 0.0008
        self.border_direction = 1
        
        self.setup_pygame()
        
        print("✅ Dragon AI Assistant v13 inizializzato!")
        if self.authorized_face_encoding is not None:
            print("🔐 Riconoscimento facciale personale ATTIVO")
    
    def load_face_cascade(self):
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            if self.face_cascade.empty():
                self.face_cascade = None
        except Exception:
            self.face_cascade = None
    
    def load_reference_face(self):
        if not self.face_recognition_enabled:
            self.authorized_face_encoding = None
            return
        try:
            if not os.path.exists(self.reference_image_path):
                self.authorized_face_encoding = None
                return
            reference_image = face_recognition.load_image_file(self.reference_image_path)
            encodings = face_recognition.face_encodings(reference_image)
            self.authorized_face_encoding = encodings[0] if encodings else None
            if self.authorized_face_encoding is not None:
                print("✅ Viso di riferimento caricato!")
        except Exception:
            self.authorized_face_encoding = None
    
    def setup_logging(self):
        try:
            os.makedirs(self.log_dir, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.current_session = {
                "session_id": ts,
                "start_time": datetime.now().isoformat(),
                "log_file": os.path.join(self.log_dir, f"conversation_{ts}.json"),
                "conversations": []
            }
            self.save_session_log()
            print(f"📝 Log: {self.current_session['log_file']}")
        except Exception as e:
            print(f"❌ Errore logging: {e}")
            self.current_session = None
    
    def log_conversation(self, question, response, response_type="text", image_path=None):
        if not self.current_session:
            return
        entry = {
            "timestamp": datetime.now().isoformat(),
            "domanda": question,
            "risposta": response,
            "tipo": response_type
        }
        if image_path:
            entry["immagine"] = image_path
        self.current_session["conversations"].append(entry)
        self.save_session_log()
    
    def save_session_log(self):
        if not self.current_session:
            return
        try:
            self.current_session["last_update"] = datetime.now().isoformat()
            with open(self.current_session["log_file"], 'w', encoding='utf-8') as f:
                json.dump(self.current_session, f, indent=2, ensure_ascii=False)
        except Exception:
            pass
    
    def setup_pygame(self):
        pygame.init()
        
        # Ottieni dimensioni schermo
        info = pygame.display.Info()
        self.desktop_w = info.current_w
        self.desktop_h = info.current_h
        
        # Dimensioni widget drago
        self.widget_w, self.widget_h = 280, 200
        
        # Finestra senza bordi (widget desktop)
        self.screen = pygame.display.set_mode(
            (self.widget_w, self.widget_h),
            pygame.NOFRAME
        )
        pygame.display.set_caption("Dragon AI")
        
        # Posiziona in basso a destra inizialmente
        self.widget_x = self.desktop_w - self.widget_w - 50
        self.widget_y = self.desktop_h - self.widget_h - 80
        
        # Sposta finestra (cross-platform)
        try:
            if os.name == 'posix':
                os.environ['SDL_VIDEO_WINDOW_POS'] = f"{self.widget_x},{self.widget_y}"
        except Exception:
            pass
        
        # macOS: always on top
        try:
            from AppKit import NSApplication, NSFloatingWindowLevel
            app = NSApplication.sharedApplication()
            for w in app.windows():
                w.setLevel_(NSFloatingWindowLevel)
        except Exception:
            pass
    
    def update_widget_position(self):
        """Muove il widget lungo i bordi dello schermo"""
        self.border_position += self.border_speed * self.border_direction
        if self.border_position > 1.0:
            self.border_position = 0.0
        elif self.border_position < 0.0:
            self.border_position = 1.0
        
        # Perimetro: bottom → right → top → left
        margin = 10
        perim = 2 * (self.desktop_w + self.desktop_h)
        pos_px = self.border_position * perim
        
        if pos_px < self.desktop_w:
            # Bottom edge → sinistra a destra
            self.widget_x = int(pos_px) - self.widget_w // 2
            self.widget_y = self.desktop_h - self.widget_h - margin
            self.dragon.facing_right = True
        elif pos_px < self.desktop_w + self.desktop_h:
            # Right edge → basso a alto
            offset = pos_px - self.desktop_w
            self.widget_x = self.desktop_w - self.widget_w - margin
            self.widget_y = self.desktop_h - int(offset) - self.widget_h
            self.dragon.facing_right = False
        elif pos_px < 2 * self.desktop_w + self.desktop_h:
            # Top edge → destra a sinistra
            offset = pos_px - self.desktop_w - self.desktop_h
            self.widget_x = self.desktop_w - int(offset) - self.widget_w
            self.widget_y = margin
            self.dragon.facing_right = False
        else:
            # Left edge → alto a basso
            offset = pos_px - 2 * self.desktop_w - self.desktop_h
            self.widget_x = margin
            self.widget_y = int(offset)
            self.dragon.facing_right = True
        
        # Clamp
        self.widget_x = max(0, min(self.widget_x, self.desktop_w - self.widget_w))
        self.widget_y = max(0, min(self.widget_y, self.desktop_h - self.widget_h))
        
        try:
            os.environ['SDL_VIDEO_WINDOW_POS'] = f"{self.widget_x},{self.widget_y}"
            # Riposiziona la finestra
            self.screen = pygame.display.set_mode(
                (self.widget_w, self.widget_h), pygame.NOFRAME
            )
        except Exception:
            pass
    
    def detect_faces(self, frame):
        """Rileva visi + emozioni - nessun log su terminale/file"""
        faces_detected = False
        authorized_user = False
        self.last_face_rect = None
        
        if self.face_cascade is None:
            return True, True, frame
        
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            faces_detected = len(faces) > 0
            
            if faces_detected:
                # Salva primo viso per emotion detection
                self.last_face_rect = tuple(faces[0])
            
            if not self.face_recognition_enabled or self.authorized_face_encoding is None:
                authorized_user = faces_detected
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
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
                            cv2.putText(frame, f"OK ({confidence:.0f}%)", 
                                       (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        else:
                            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                except Exception:
                    for (x, y, w, h) in faces:
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)
                
        except Exception:
            pass
        
        return faces_detected, authorized_user, frame
    
    def speak_text(self, text):
        if self.shutdown:
            return None
        def speak():
            with self.speech_lock:
                if self.shutdown:
                    return
                self.is_speaking = True
                try:
                    subprocess.run(['say', text], check=True)
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
        if (not self.is_listening and not self.is_speaking and not self.is_processing and 
            not self.is_paused and self.authorized_user_detected):
            question = random.choice(self.random_questions)
            print(f"\n🐉 Drago: {question}")
            self.speak_text(question)
            self.last_question_time = time.time()
            self.is_processing = True
    
    def check_for_control_commands(self, text):
        text_lower = text.lower()
        if any(w in text_lower for w in self.pause_words):
            if not self.is_paused:
                print("😴 Pausa!")
                self.is_paused = True
                self.speak_text("Mi metto a dormire. Dimmi drago o sveglia per riattivarmi.")
                return "PAUSA"
        if any(w in text_lower for w in self.wake_words):
            if self.is_paused:
                print("🔥 Sveglia!")
                self.is_paused = False
                self.speak_text("Eccomi! Sono tornato. Come posso aiutarti?")
                return "RISVEGLIO"
        return None
    
    def listen_for_wake_command(self):
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
                text = self.recognizer.recognize_google(audio, language='it-IT')
                return self.check_for_control_commands(text)
            except (sr.UnknownValueError, sr.RequestError):
                return None
        except Exception:
            return None
        finally:
            self.microphone_lock.release()
    
    def listen_to_user(self):
        if (self.shutdown or self.is_listening or self.is_paused or 
            (not self.authorized_user_detected and self.authorized_face_encoding is not None)):
            return None
        if not self.microphone_lock.acquire(blocking=False):
            return None
        try:
            print("\n🎤 Ascoltando...")
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
                print(f"📝 Sentito: '{user_text}'")
                ctrl = self.check_for_control_commands(user_text)
                if ctrl:
                    return None
                return self.get_voice_confirmation(user_text)
            except sr.UnknownValueError:
                print("❌ Non capito.")
                return None
            except sr.RequestError as e:
                print(f"❌ Errore STT: {e}")
                return None
        except sr.WaitTimeoutError:
            return None
        except Exception:
            return None
        finally:
            self.is_listening = False
            self.microphone_lock.release()
    
    def get_voice_confirmation(self, user_text):
        try:
            self.speak_text(f"Hai detto: {user_text}. Confermi?")
            time.sleep(3)
            print("🎤 SÌ o NO?")
            with self.microphone as source:
                if self.shutdown:
                    return None
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = self.recognizer.listen(source, timeout=8)
            if self.shutdown:
                return None
            try:
                conf = self.recognizer.recognize_google(audio, language='it-IT').lower()
                print(f"🔊 Conferma: '{conf}'")
                if any(w in conf for w in ['sì', 'si', 'yes', 'vai', 'ok', 'okay', 'invia', 'perfetto']):
                    print("✅ OK!")
                    self.speak_text("Perfetto!")
                    time.sleep(1)
                    return user_text
                else:
                    print("❌ Annullato.")
                    self.speak_text("Annullo.")
                    return None
            except Exception:
                self.speak_text("Non ho sentito. Annullo.")
                return None
        except Exception:
            return None
    
    # =========================================================================
    # GENERAZIONE IMMAGINI
    # =========================================================================
    
    def is_draw_request(self, q):
        return any(k in q.lower() for k in self.draw_keywords)
    
    def extract_image_prompt(self, q):
        prompt = q
        for kw in sorted(self.draw_keywords, key=len, reverse=True):
            prompt = re.sub(re.escape(kw), '', prompt, flags=re.IGNORECASE).strip()
        prompt = re.sub(r'^(un |una |uno |il |la |lo |i |le |gli |di |del |della |dello )', '', prompt, flags=re.IGNORECASE).strip()
        return prompt if prompt else q
    
    def generate_image(self, prompt):
        try:
            self.is_generating_image = True
            print(f"🎨 Generazione: '{prompt}'")
            resp = self.client.images.generate(model="dall-e-3", prompt=prompt, size="1024x1024", quality="standard", n=1)
            url = resp.data[0].url
            revised = resp.data[0].revised_prompt
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe = re.sub(r'[^\w\s-]', '', prompt[:40]).strip().replace(' ', '_')
            path = os.path.join(self.images_dir, f"{ts}_{safe}.png")
            import urllib.request
            urllib.request.urlretrieve(url, path)
            print(f"✅ Salvata: {path}")
            try:
                img = cv2.imread(path)
                if img is not None:
                    scale = 600 / max(img.shape[:2])
                    img = cv2.resize(img, (int(img.shape[1]*scale), int(img.shape[0]*scale)))
                    cv2.imshow(f'Immagine - {prompt[:30]}', img)
            except Exception:
                pass
            return path, revised, url
        except Exception as e:
            return None, str(e), None
        finally:
            self.is_generating_image = False
    
    # =========================================================================
    # TRADUTTORE SIMULTANEO
    # =========================================================================
    
    def translate_response(self, text, target_lang="inglese"):
        """Traduce la risposta nella lingua target usando OpenAI"""
        try:
            resp = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": f"Sei un traduttore professionale. Traduci il seguente testo in {target_lang}. Restituisci SOLO la traduzione, nient'altro."},
                    {"role": "user", "content": text}
                ],
                max_tokens=800,
                temperature=0.2
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            return f"[Errore traduzione: {e}]"
    
    # =========================================================================
    # RISPOSTE OPENAI
    # =========================================================================
    
    def get_openai_response(self, question):
        try:
            prompt = """Sei un assistente AI altamente qualificato con competenze scientifiche avanzate. 
            Rispondi in italiano con linguaggio tecnico ma accessibile. Fornisci risposte dettagliate
            e scientificamente accurate con esempi concreti."""
            
            q_lower = question.lower()
            is_sci = any(w in q_lower for w in ['come', 'perché', 'meccanismo', 'funziona', 'fisica', 'chimica', 'biologia', 'matematica'])
            max_tok = 600 if is_sci else 400
            temp = 0.3 if is_sci else 0.5
            
            resp = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": prompt}, {"role": "user", "content": question}],
                max_tokens=max_tok, temperature=temp,
                presence_penalty=0.1, frequency_penalty=0.1
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            return f"Errore OpenAI: {e}"
    
    def complete_interaction(self, user_question):
        try:
            print(f"\n{'='*70}")
            print(f"  🗣️  DOMANDA: {user_question}")
            print(f"{'='*70}")
            
            if self.is_draw_request(user_question):
                prompt = self.extract_image_prompt(user_question)
                self.speak_text(f"Genero un'immagine di {prompt}. Un attimo.")
                path, revised, url = self.generate_image(prompt)
                if path:
                    resp_text = f"Immagine: '{prompt}' → {path}"
                    print(f"  🎨 RISPOSTA: Immagine generata!")
                    print(f"  📁 {path}")
                    print(f"{'='*70}\n")
                    self.log_conversation(user_question, resp_text, "image", path)
                    self.speak_text(f"Fatto! Immagine di {prompt} pronta.")
                else:
                    print(f"  ❌ Errore: {revised}")
                    print(f"{'='*70}\n")
                    self.log_conversation(user_question, f"Errore: {revised}", "image_error")
                    self.speak_text("Non sono riuscito a generare l'immagine.")
            else:
                response = self.get_openai_response(user_question)
                
                # Traduzione se attiva
                translation = None
                if self.translator_mode:
                    translation = self.translate_response(response, "inglese")
                
                print(f"  🤖 RISPOSTA: {response}")
                if translation:
                    print(f"  🌐 TRADUZIONE: {translation}")
                print(f"{'='*70}\n")
                
                log_resp = response
                if translation:
                    log_resp += f"\n\n[EN] {translation}"
                self.log_conversation(user_question, log_resp, "text")
                
                speech = response[:800] + "..." if len(response) > 800 else response
                if not self.shutdown:
                    t = self.speak_text(speech)
                    if t:
                        t.join()
                
                # Se traduttore attivo, leggi anche la traduzione
                if translation and not self.shutdown:
                    time.sleep(0.5)
                    self.speak_text("In English: " + (translation[:500] if len(translation) > 500 else translation))
            
            print(f"💤 Pausa {self.cooldown_after_interaction}s...")
            time.sleep(self.cooldown_after_interaction)
        finally:
            self.is_processing = False
            self.last_question_time = time.time()
    
    # =========================================================================
    # RENDERING DRAGO + PARTICELLE
    # =========================================================================
    
    def render_dragon_widget(self):
        """Renderizza il drago con particelle nella finestra widget"""
        # Sfondo scuro semi-trasparente
        self.screen.fill((10, 10, 25))
        
        # Bordo arrotondato decorativo
        border_col = (40, 40, 80)
        if self.is_paused:
            border_col = (60, 30, 80)
        elif self.is_speaking:
            border_col = (120, 50, 20)
        elif self.authorized_user_detected:
            border_col = (30, 80, 40)
        pygame.draw.rect(self.screen, border_col, (0, 0, self.widget_w, self.widget_h), 2, border_radius=8)
        
        # Centro del drago nel widget
        dcx = self.widget_w // 2
        dcy = self.widget_h // 2 + 10
        
        # Aggiorna drago
        self.dragon.update(
            is_speaking=self.is_speaking,
            is_sleeping=self.is_paused,
            is_processing=self.is_processing
        )
        
        # Applica emozione dal volto
        if not self.is_paused and not self.is_speaking and not self.is_processing:
            if self.current_emotion == "happy":
                self.dragon.set_mood("happy")
            elif self.current_emotion == "sleeping":
                self.dragon.set_mood("sleeping")
            else:
                self.dragon.set_mood("neutral")
        
        # Disegna drago → ritorna posizione bocca per fuoco
        fire_x, fire_y = self.dragon.draw(self.screen, dcx, dcy, scale=1.2)
        
        # Emetti particelle in base allo stato
        if self.is_speaking:
            self.particles.emit_fire(fire_x, fire_y, count=8, intensity=1.5)
            self.particles.emit_sparkle(fire_x, fire_y, count=2)
        elif self.is_processing or self.is_generating_image:
            self.particles.emit_smoke(dcx, dcy - 30, count=2)
            self.particles.emit_sparkle(dcx, dcy - 40, count=1)
        elif self.is_paused:
            if self.dragon.frame % 20 == 0:
                self.particles.emit_zzz(dcx + 30, dcy - 40, count=1)
        elif self.authorized_user_detected:
            if self.dragon.frame % 8 == 0:
                self.particles.emit_fire(fire_x, fire_y, count=1, intensity=0.4)
        
        # Aggiorna e disegna particelle
        self.particles.update()
        self.particles.draw(self.screen)
        
        # Testo stato (piccolo, in basso)
        small = pygame.font.Font(None, 16)
        tiny = pygame.font.Font(None, 14)
        
        if self.is_paused:
            txt = small.render("💤 Dormendo...", True, (180, 130, 220))
            self.screen.blit(txt, (8, 6))
        elif not self.authorized_user_detected and self.authorized_face_encoding is not None:
            txt = small.render("🚫 Accesso negato", True, (255, 80, 80))
            self.screen.blit(txt, (8, 6))
        elif self.is_generating_image:
            txt = small.render("🎨 Sto disegnando...", True, (255, 200, 0))
            self.screen.blit(txt, (8, 6))
        elif self.is_speaking:
            txt = small.render("🔥 Parlando...", True, (255, 150, 50))
            self.screen.blit(txt, (8, 6))
        elif self.is_listening:
            txt = small.render("🎤 Ascoltando...", True, (255, 255, 100))
            self.screen.blit(txt, (8, 6))
        elif self.is_processing:
            txt = small.render("🤔 Elaborando...", True, (100, 150, 255))
            self.screen.blit(txt, (8, 6))
        elif self.authorized_user_detected:
            txt = small.render("👤✅ Autorizzato", True, (80, 220, 80))
            self.screen.blit(txt, (8, 6))
        
        # Indicatori in basso
        indicators = []
        if self.translator_mode:
            indicators.append("🌐 IT↔EN")
        indicators.append("H:storia" if not self.history_gui.is_open else "H:aperta")
        indicators.append("T:trad" if not self.translator_mode else "T:off")
        
        ind_text = " | ".join(indicators)
        ind_surf = tiny.render(ind_text, True, (100, 100, 130))
        self.screen.blit(ind_surf, (8, self.widget_h - 18))
        
        pygame.display.flip()
    
    # =========================================================================
    # LOOP PRINCIPALE
    # =========================================================================
    
    def run(self):
        clock = pygame.time.Clock()
        last_wake_check = 0
        border_move_counter = 0
        
        try:
            while True:
                current_time = time.time()
                
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return
                    elif event.type == pygame.KEYDOWN:
                        if event.key in (pygame.K_q, pygame.K_ESCAPE):
                            print("👋 Chiusura...")
                            return
                        elif event.key == pygame.K_v:
                            self.show_video = not self.show_video
                            if not self.show_video:
                                cv2.destroyWindow('Webcam')
                            print(f"📺 Video: {'ON' if self.show_video else 'OFF'}")
                        elif event.key == pygame.K_p:
                            self.is_paused = not self.is_paused
                            print(f"⏯️ {'Pausa' if self.is_paused else 'Attivo'}")
                        elif event.key == pygame.K_t:
                            self.translator_mode = not self.translator_mode
                            status = "ATTIVO 🌐" if self.translator_mode else "DISATTIVO"
                            print(f"🌐 Traduttore: {status}")
                            self.speak_text(f"Modalità traduttore {'attivata' if self.translator_mode else 'disattivata'}.")
                        elif event.key == pygame.K_h:
                            print("📜 Apro cronologia...")
                            self.history_gui.open()
                        elif event.key == pygame.K_SPACE and self.face_detected and not self.is_processing and not self.is_paused and self.authorized_user_detected:
                            def manual():
                                q = self.listen_to_user()
                                if q and not self.shutdown:
                                    self.complete_interaction(q)
                            threading.Thread(target=manual, daemon=True).start()
                
                # Pausa: ascolta solo wake
                if self.is_paused:
                    if current_time - last_wake_check > 2:
                        threading.Thread(target=self.listen_for_wake_command, daemon=True).start()
                        last_wake_check = current_time
                else:
                    ret, frame = self.cap.read()
                    if ret:
                        faces_detected, authorized_user, processed_frame = self.detect_faces(frame)
                        self.face_detected = faces_detected
                        self.authorized_user_detected = authorized_user
                        
                        # Emotion detection
                        if self.last_face_rect and authorized_user:
                            self.current_emotion = self.emotion_detector.detect(frame, self.last_face_rect)
                        
                        if self.show_video:
                            cv2.imshow('Webcam', processed_frame)
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                break
                        else:
                            cv2.waitKey(1)
                        
                        if (authorized_user and 
                            current_time - self.last_question_time > self.question_interval and
                            not self.is_listening and not self.is_speaking and not self.is_processing):
                            
                            self.ask_random_question()
                            
                            def auto():
                                time.sleep(3)
                                if not self.shutdown and not self.is_listening and self.is_processing and not self.is_paused and self.authorized_user_detected:
                                    q = self.listen_to_user()
                                    if q and not self.shutdown:
                                        self.complete_interaction(q)
                                    else:
                                        self.is_processing = False
                                        self.last_question_time = time.time()
                            threading.Thread(target=auto, daemon=True).start()
                
                # Muovi widget lungo i bordi (ogni N frame)
                border_move_counter += 1
                if border_move_counter >= 3:
                    self.update_widget_position()
                    border_move_counter = 0
                
                self.render_dragon_widget()
                clock.tick(30)
                
        except KeyboardInterrupt:
            print("\n👋 Ciao!")
        finally:
            self.shutdown = True
            time.sleep(0.5)
            self.cleanup()
    
    def cleanup(self):
        print("🧹 Pulizia...")
        self.shutdown = True
        if self.current_session:
            self.current_session["end_time"] = datetime.now().isoformat()
            self.current_session["total_conversations"] = len(self.current_session["conversations"])
            self.save_session_log()
        if hasattr(self, 'cap') and self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        pygame.quit()
        print("✅ Completato!")
        if self.current_session:
            print(f"📝 Log: {self.current_session['log_file']}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("🐉 Avvio Dragon AI Assistant v13...")
    print("="*50)
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ OPENAI_API_KEY non trovata nel .env!")
        api_key = input("🔑 Inserisci API key: ").strip()
        if not api_key:
            print("❌ Chiave richiesta!")
            return
    
    ref_img = "anto2025.png"
    if os.path.exists(ref_img):
        print(f"🔐 Sicurezza ATTIVA ({ref_img})")
    else:
        print(f"⚠️ {ref_img} non trovata - modalità aperta")
        r = input("Continuare? (s/n): ")
        if r.lower() not in ['s', 'si', 'sì', 'y', 'yes']:
            return
    
    try:
        assistant = DragonAIAssistant(api_key, ref_img)
        print("\n" + "="*50)
        print("  🐉 DRAGON AI - CONTROLLI")
        print("  " + "-"*46)
        print("  SPAZIO  Parla           Q/ESC  Esci")
        print("  V       Video on/off    P      Pausa")
        print("  T       Traduttore IT↔EN")
        print("  H       Cronologia conversazioni")
        print("  " + "-"*46)
        print("  🎤 Voce: PAUSA/STOP | DRAGO/SVEGLIA")
        print("  🎨 Di: 'disegna...' per generare immagini")
        print("  🌐 Traduttore: risponde in IT + EN")
        print("="*50 + "\n")
        
        assistant.run()
        
    except Exception as e:
        print(f"❌ Errore: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
