#!/usr/bin/env python3
"""
AI Assistant v13 - Drago Procedurale sul Desktop (FIXED)
=========================================================
FIX: Finestra si muove realmente sui bordi via pygame._sdl2 + fallback macOS
FIX: Cronologia si apre nel browser (HTML) senza bloccare il programma
- Drago procedurale con sistema particellare (fuoco, fumo, scintille)
- Widget desktop che cammina sui bordi dello schermo
- Riconoscimento emozioni dal volto → cambia umore del drago
- Animazioni reattive (fuoco quando parla, dorme quando in pausa)
- Modalità traduttore simultaneo (italiano ↔ inglese)
- Cronologia ricercabile delle conversazioni (HTML nel browser)
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
import webbrowser
import tempfile

load_dotenv()

try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False

import logging
logging.getLogger("face_recognition").setLevel(logging.WARNING)
logging.getLogger("dlib").setLevel(logging.WARNING)

# Prova a importare pygame._sdl2 per spostamento finestra a runtime
SDL2_AVAILABLE = False
try:
    from pygame._sdl2.video import Window as SDL2Window
    SDL2_AVAILABLE = True
except ImportError:
    SDL2_AVAILABLE = False


# =============================================================================
# SISTEMA PARTICELLARE
# =============================================================================

class Particle:
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
    def __init__(self, max_particles=250):
        self.particles = []
        self.max_particles = max_particles
    
    def emit_fire(self, x, y, count=5, intensity=1.0):
        for _ in range(count):
            vx = random.uniform(-1.5, 1.5) * intensity
            vy = random.uniform(-4, -1.5) * intensity
            life = random.uniform(15, 40)
            size = random.uniform(2, 6) * intensity
            color_start = (255, random.randint(180, 255), 0, 255)
            color_end = (200, 0, 0, 0)
            self._add(Particle(x, y, vx, vy, life, size, color_start, color_end, "fire"))
    
    def emit_smoke(self, x, y, count=2):
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
                    p.vy -= 0.05
                alive.append(p)
        self.particles = alive
    
    def draw(self, surface):
        for p in self.particles:
            t = 1 - (p.life / p.max_life)
            r = int(p.color_start[0] + (p.color_end[0] - p.color_start[0]) * t)
            g = int(p.color_start[1] + (p.color_end[1] - p.color_start[1]) * t)
            b = int(p.color_start[2] + (p.color_end[2] - p.color_start[2]) * t)
            a = int(p.color_start[3] + (p.color_end[3] - p.color_start[3]) * t)
            r, g, b, a = [max(0, min(255, v)) for v in (r, g, b, a)]
            size = max(1, int(p.size * (p.life / p.max_life)))
            
            if p.particle_type == "zzz" and p.life > 10:
                font = pygame.font.Font(None, int(size * 4))
                z_text = font.render("Z", True, (r, g, b))
                z_text.set_alpha(a)
                surface.blit(z_text, (int(p.x), int(p.y)))
            else:
                ps = pygame.Surface((size * 2, size * 2), pygame.SRCALPHA)
                pygame.draw.circle(ps, (r, g, b, a), (size, size), size)
                surface.blit(ps, (int(p.x - size), int(p.y - size)))


# =============================================================================
# DRAGO PROCEDURALE
# =============================================================================

class ProceduralDragon:
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
        if is_sleeping:
            self.wing_angle = 5 * math.sin(self.frame * 0.02)
        else:
            speed = 0.15 if is_speaking else 0.08
            amplitude = 25 if is_speaking else 15
            self.wing_angle = amplitude * math.sin(self.frame * speed)
        self.tail_wave = 10 * math.sin(self.frame * 0.06)
        if is_speaking:
            self.mouth_open = 8 + 5 * abs(math.sin(self.frame * 0.3))
            self.breath_intensity = min(1.0, self.breath_intensity + 0.1)
        else:
            self.mouth_open = max(0, self.mouth_open - 1)
            self.breath_intensity = max(0, self.breath_intensity - 0.05)
        self.eye_blink_timer += 1
        if not self.eye_is_blinking and self.eye_blink_timer > random.randint(80, 200):
            self.eye_is_blinking = True
            self.eye_blink_timer = 0
        if self.eye_is_blinking:
            self.eye_blink_timer += 1
            if self.eye_blink_timer > 6:
                self.eye_is_blinking = False
                self.eye_blink_timer = 0
        if is_sleeping:
            self.set_mood("sleeping")
        elif is_speaking:
            self.set_mood("speaking")
        elif is_processing:
            self.set_mood("thinking")
    
    def draw(self, surface, cx, cy, scale=1.0):
        colors = self.MOODS.get(self.mood, self.MOODS["neutral"])
        body_col = colors["body"]
        belly_col = colors["belly"]
        eye_col = colors["eye"]
        flip = 1 if self.facing_right else -1
        bob = int(4 * math.sin(self.frame * 0.04))
        s = scale
        
        # CODA
        tail_points = []
        for i in range(8):
            t = i / 7
            tx = cx - flip * int((30 + i * 12) * s)
            ty = cy + int((10 + i * 5) * s) + bob + int(self.tail_wave * t * math.sin(self.frame * 0.06 + t * 3))
            tail_points.append((tx, ty))
        if len(tail_points) > 1:
            dark_body = tuple(max(0, c - 30) for c in body_col)
            pygame.draw.lines(surface, dark_body, False, tail_points, max(1, int(4 * s)))
            last = tail_points[-1]
            spike_size = int(6 * s)
            spike = [last, (last[0] - flip * spike_size, last[1] - spike_size), (last[0] - flip * spike_size, last[1] + spike_size)]
            pygame.draw.polygon(surface, (180, 50, 20), spike)
        
        # ALI
        wbx = cx - flip * int(5 * s)
        wby = cy - int(15 * s) + bob
        wto = self.wing_angle * s
        wing_pts = [
            (wbx, wby),
            (wbx - flip * int(10 * s), wby - int((35 + wto) * s)),
            (wbx + flip * int(15 * s), wby - int((20 + wto * 0.6) * s)),
            (wbx + flip * int(5 * s), wby - int(5 * s)),
        ]
        wing_membrane = tuple(min(255, c + 40) for c in body_col) + (120,)
        wing_surf = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
        pygame.draw.polygon(wing_surf, wing_membrane, wing_pts)
        surface.blit(wing_surf, (0, 0))
        pygame.draw.lines(surface, body_col, False, wing_pts, max(1, int(2 * s)))
        
        # CORPO
        body_rect = pygame.Rect(cx - int(22*s), cy - int(18*s) + bob, int(44*s), int(36*s))
        pygame.draw.ellipse(surface, body_col, body_rect)
        belly_rect = pygame.Rect(cx - int(14*s), cy - int(6*s) + bob, int(28*s), int(22*s))
        pygame.draw.ellipse(surface, belly_col, belly_rect)
        
        # SCAGLIE
        for i in range(5):
            sx = cx + flip * int((-10 + i * 6) * s)
            sy = cy - int(18*s) + bob - int(3*s)
            sh = int((4 + 2 * math.sin(self.frame * 0.1 + i)) * s)
            pts = [(sx, sy), (sx - int(3*s), sy + sh), (sx + int(3*s), sy + sh)]
            pygame.draw.polygon(surface, (180, 50, 20), pts)
        
        # ZAMPE
        for lo in [-12, 8]:
            lx = cx + int(lo * s)
            ly = cy + int(16*s) + bob
            lsw = int(3 * math.sin(self.frame * 0.08 + lo))
            pygame.draw.ellipse(surface, tuple(max(0, c-20) for c in body_col), (lx - int(5*s), ly + lsw, int(10*s), int(8*s)))
        
        # TESTA
        head_x = cx + flip * int(28 * s)
        head_y = cy - int(12 * s) + bob
        head_size = int(20 * s)
        pygame.draw.line(surface, body_col, (cx + flip*int(15*s), cy - int(10*s) + bob), (head_x - flip*int(5*s), head_y + int(5*s)), max(1, int(10*s)))
        pygame.draw.circle(surface, body_col, (head_x, head_y), head_size)
        
        # MUSO
        muzzle_x = head_x + flip * int(14 * s)
        muzzle_y = head_y + int(4 * s)
        pygame.draw.ellipse(surface, tuple(min(255, c+20) for c in body_col), (muzzle_x - int(10*s), muzzle_y - int(6*s), int(20*s), int(12*s)))
        
        # NARICI
        ng = int(80 * self.breath_intensity)
        nc = (200 + min(55, ng), 80, 20)
        for nyo in [-2, 2]:
            pygame.draw.circle(surface, nc, (muzzle_x + flip*int(7*s), muzzle_y + int(nyo*s)), max(1, int(2*s)))
        
        # OCCHIO
        eye_x = head_x + flip * int(6*s)
        eye_y = head_y - int(5*s)
        esz = max(1, int(6*s))
        if self.eye_is_blinking or self.mood == "sleeping":
            pygame.draw.line(surface, (30,30,30), (eye_x - esz, eye_y), (eye_x + esz, eye_y), 2)
        else:
            pygame.draw.circle(surface, (240,240,220), (eye_x, eye_y), esz)
            pygame.draw.circle(surface, eye_col, (eye_x, eye_y), max(1, int(3*s)))
            pygame.draw.circle(surface, (0,0,0), (eye_x, eye_y), max(1, int(2*s)))
            pygame.draw.circle(surface, (255,255,255), (eye_x - max(1,int(2*s)), eye_y - max(1,int(2*s))), max(1, int(1.5*s)))
        
        # CORNA
        for hs in [-1, 1]:
            hx = head_x + int(hs*8*s)
            hy = head_y - int(14*s)
            horn = [(hx, head_y - int(8*s)), (hx + int(hs*4*s), hy), (hx - int(hs*2*s), hy + int(4*s))]
            pygame.draw.polygon(surface, (180, 150, 80), horn)
        
        # BOCCA
        if self.mouth_open > 2:
            my = muzzle_y + int(3*s)
            mw = int(self.mouth_open * s)
            pygame.draw.ellipse(surface, (80,20,20), (muzzle_x + flip*int(2*s) - mw//2, my - int(3*s), mw, int(6*s)))
        
        fire_x = muzzle_x + flip * int(15*s)
        fire_y = muzzle_y + int(2*s)
        return fire_x, fire_y


# =============================================================================
# RICONOSCIMENTO EMOZIONI
# =============================================================================

class EmotionDetector:
    def __init__(self):
        self.smile_cascade = None
        self.eye_cascade = None
        self.current_emotion = "neutral"
        try:
            sc = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
            self.smile_cascade = sc if not sc.empty() else None
        except Exception:
            pass
        try:
            ec = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
            self.eye_cascade = ec if not ec.empty() else None
        except Exception:
            pass
    
    def detect(self, frame, face_rect):
        if face_rect is None:
            return "neutral"
        x, y, w, h = face_rect
        try:
            roi = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
        except Exception:
            return "neutral"
        smile = False
        eyes = True
        if self.smile_cascade is not None:
            smiles = self.smile_cascade.detectMultiScale(roi, 1.7, 22, minSize=(25, 25))
            smile = len(smiles) > 0
        if self.eye_cascade is not None:
            e = self.eye_cascade.detectMultiScale(roi, 1.1, 5, minSize=(20, 20))
            eyes = len(e) >= 1
        if smile:
            self.current_emotion = "happy"
        elif not eyes:
            self.current_emotion = "sleeping"
        else:
            self.current_emotion = "neutral"
        return self.current_emotion


# =============================================================================
# CRONOLOGIA CONVERSAZIONI → HTML NEL BROWSER (no tkinter)
# =============================================================================

class ConversationHistory:
    """Genera una pagina HTML con la cronologia e la apre nel browser"""
    
    def __init__(self, log_dir="log"):
        self.log_dir = log_dir
    
    def open(self):
        """Genera HTML e apri nel browser — non bloccante"""
        threading.Thread(target=self._generate_and_open, daemon=True).start()
    
    def _generate_and_open(self):
        sessions = self._load_all()
        html = self._render_html(sessions)
        
        # Salva in un file temporaneo
        html_path = os.path.join(self.log_dir, "cronologia.html")
        try:
            os.makedirs(self.log_dir, exist_ok=True)
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html)
            webbrowser.open(f"file://{os.path.abspath(html_path)}")
            print(f"📜 Cronologia aperta nel browser: {html_path}")
        except Exception as e:
            print(f"❌ Errore apertura cronologia: {e}")
    
    def _load_all(self):
        sessions = []
        if not os.path.exists(self.log_dir):
            return sessions
        for f in sorted(os.listdir(self.log_dir), reverse=True):
            if f.endswith(".json"):
                try:
                    with open(os.path.join(self.log_dir, f), 'r', encoding='utf-8') as fh:
                        sessions.append(json.load(fh))
                except Exception:
                    pass
        return sessions
    
    def _render_html(self, sessions):
        convs_html = ""
        total = 0
        for session in sessions:
            sid = session.get("session_id", "?")
            start = session.get("start_time", "?")[:19]
            convs_html += f'<div class="session-header">Sessione {sid} — {start}</div>\n'
            for conv in session.get("conversations", []):
                ts = conv.get("timestamp", "")[:19]
                q = conv.get("domanda", "")
                r = conv.get("risposta", "")
                tipo = conv.get("tipo", "text")
                img = conv.get("immagine", "")
                if not q and not r:
                    continue
                total += 1
                # Escape HTML
                q = q.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
                r = r.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
                
                icon = "🎨" if tipo == "image" else "🤖"
                img_line = f'<div class="image-path">📁 {img}</div>' if img else ""
                
                convs_html += f'''<div class="conv" data-q="{q.lower()}" data-r="{r.lower()}">
  <div class="ts">{ts}</div>
  <div class="question">🗣️ {q}</div>
  <div class="answer">{icon} {r}</div>
  {img_line}
</div>\n'''
        
        return f'''<!DOCTYPE html>
<html lang="it">
<head>
<meta charset="UTF-8">
<title>🐉 Dragon AI - Cronologia</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ background: #0d1117; color: #c9d1d9; font-family: 'Segoe UI', sans-serif; padding: 20px; }}
  h1 {{ color: #e94560; text-align: center; margin-bottom: 5px; font-size: 28px; }}
  .subtitle {{ text-align: center; color: #666; margin-bottom: 20px; font-size: 14px; }}
  .search-box {{ display: block; width: 100%; max-width: 600px; margin: 0 auto 25px; padding: 12px 16px;
    background: #161b22; border: 1px solid #30363d; border-radius: 8px; color: #c9d1d9;
    font-size: 16px; outline: none; }}
  .search-box:focus {{ border-color: #e94560; }}
  .session-header {{ background: #161b22; padding: 10px 16px; margin: 20px 0 8px; border-radius: 6px;
    border-left: 3px solid #e94560; color: #8b949e; font-size: 13px; font-weight: 600; }}
  .conv {{ background: #161b22; margin: 6px 0; padding: 14px 18px; border-radius: 8px;
    border: 1px solid #21262d; transition: border-color 0.2s; }}
  .conv:hover {{ border-color: #30363d; }}
  .conv.hidden {{ display: none; }}
  .ts {{ color: #484f58; font-size: 11px; margin-bottom: 6px; }}
  .question {{ color: #e94560; font-weight: 600; margin-bottom: 8px; font-size: 15px; }}
  .answer {{ color: #7ee8fa; line-height: 1.5; font-size: 14px; white-space: pre-wrap; }}
  .image-path {{ color: #ffc107; font-size: 12px; margin-top: 6px; }}
  .no-results {{ text-align: center; color: #484f58; padding: 40px; font-size: 16px; }}
</style>
</head>
<body>
<h1>🐉 Dragon AI — Cronologia Conversazioni</h1>
<div class="subtitle">{total} conversazioni in {len(sessions)} sessioni</div>
<input type="text" class="search-box" placeholder="🔍 Cerca nelle conversazioni..." oninput="filterConvs(this.value)">
<div id="conversations">
{convs_html}
</div>
<div id="no-results" class="no-results" style="display:none;">Nessun risultato trovato.</div>
<script>
function filterConvs(query) {{
  query = query.toLowerCase().trim();
  const convs = document.querySelectorAll('.conv');
  let visible = 0;
  convs.forEach(c => {{
    if (!query || c.dataset.q.includes(query) || c.dataset.r.includes(query)) {{
      c.classList.remove('hidden');
      visible++;
    }} else {{
      c.classList.add('hidden');
    }}
  }});
  document.getElementById('no-results').style.display = visible === 0 ? 'block' : 'none';
}}
</script>
</body>
</html>'''


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
        
        self.reference_image_path = reference_image_path
        self.authorized_face_encoding = None
        self.face_recognition_enabled = FACE_RECOGNITION_AVAILABLE
        self.load_reference_face()
        
        self.log_dir = "log"
        self.current_session = None
        self.setup_logging()
        
        self.images_dir = "generated_images"
        os.makedirs(self.images_dir, exist_ok=True)
        
        self.load_face_cascade()
        self.emotion_detector = EmotionDetector()
        self.current_emotion = "neutral"
        self.last_face_rect = None
        
        # Cronologia (HTML, no tkinter)
        self.history = ConversationHistory(self.log_dir)
        
        self.dragon = ProceduralDragon()
        self.particles = ParticleSystem(300)
        self.translator_mode = False
        
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
        
        self.draw_keywords = [
            'disegna', 'disegnami', 'genera immagine', "genera un'immagine",
            'crea immagine', "crea un'immagine", 'illustra', 'illustrami',
            'fammi vedere', 'mostrami', 'dipingi', 'dipingimi',
            'draw', 'generate image', 'create image', 'paint',
            'fai un disegno', "fai un'immagine", 'produci immagine'
        ]
        
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
        
        self.pause_words = ['pausa', 'stop', 'fermati', 'silenzio', 'basta', 'dormi', 'sleep', 'pause', 'zitto', 'taci']
        self.wake_words = ['risveglia', 'sveglia', 'wake up', 'continua', 'torna', 'genio', 'ehi genio', 'ciao genio', 'hey', 'start', 'riprendi', 'attivati', 'ci sei', 'drago']
        
        # Desktop border walking
        self.border_position = 0.0
        self.border_speed = 0.0005
        self.border_direction = 1
        self.sdl2_window = None
        
        self.setup_pygame()
        
        print("✅ Dragon AI Assistant v13 inizializzato!")
        if self.authorized_face_encoding is not None:
            print("🔐 Riconoscimento facciale personale ATTIVO")
        if SDL2_AVAILABLE:
            print("✅ Movimento finestra: pygame._sdl2 (nativo)")
        else:
            print("⚠️ Movimento finestra: fallback AppleScript/wmctrl")
    
    def load_face_cascade(self):
        try:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
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
            img = face_recognition.load_image_file(self.reference_image_path)
            enc = face_recognition.face_encodings(img)
            self.authorized_face_encoding = enc[0] if enc else None
            if self.authorized_face_encoding is not None:
                print("✅ Viso di riferimento caricato!")
        except Exception:
            self.authorized_face_encoding = None
    
    def setup_logging(self):
        try:
            os.makedirs(self.log_dir, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.current_session = {
                "session_id": ts, "start_time": datetime.now().isoformat(),
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
        entry = {"timestamp": datetime.now().isoformat(), "domanda": question, "risposta": response, "tipo": response_type}
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
    
    # =========================================================================
    # PYGAME SETUP + SPOSTAMENTO FINESTRA REALE
    # =========================================================================
    
    def setup_pygame(self):
        pygame.init()
        info = pygame.display.Info()
        self.desktop_w = info.current_w
        self.desktop_h = info.current_h
        self.widget_w, self.widget_h = 280, 200
        
        # Posizione iniziale (env var funziona SOLO al primo set_mode)
        self.widget_x = self.desktop_w - self.widget_w - 50
        self.widget_y = self.desktop_h - self.widget_h - 80
        os.environ['SDL_VIDEO_WINDOW_POS'] = f"{self.widget_x},{self.widget_y}"
        
        self.screen = pygame.display.set_mode((self.widget_w, self.widget_h), pygame.NOFRAME)
        pygame.display.set_caption("Dragon AI")
        
        # Ottieni handle SDL2 per spostamento a runtime
        if SDL2_AVAILABLE:
            try:
                self.sdl2_window = SDL2Window.from_display_module()
                self.sdl2_window.position = (self.widget_x, self.widget_y)
                print(f"🖥️  Finestra posizionata: ({self.widget_x}, {self.widget_y})")
            except Exception as e:
                print(f"⚠️ SDL2 Window init fallito: {e}")
                self.sdl2_window = None
        
        # macOS: always on top
        self._set_always_on_top()
    
    def _set_always_on_top(self):
        """Imposta la finestra always-on-top (macOS con pyobjc, fallback con osascript)"""
        try:
            from AppKit import NSApplication, NSFloatingWindowLevel
            app = NSApplication.sharedApplication()
            for w in app.windows():
                w.setLevel_(NSFloatingWindowLevel)
            print("✅ Always-on-top via AppKit")
            return
        except Exception:
            pass
        
        # Fallback: AppleScript
        try:
            script = '''
            tell application "System Events"
                set frontmost of every process whose name contains "Python" to true
            end tell
            '''
            subprocess.Popen(['osascript', '-e', script], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass
    
    def move_window(self, x, y):
        """Sposta la finestra pygame alla posizione (x, y) sul desktop"""
        self.widget_x = x
        self.widget_y = y
        
        # Metodo 1: pygame._sdl2 (più affidabile)
        if self.sdl2_window is not None:
            try:
                self.sdl2_window.position = (int(x), int(y))
                return
            except Exception:
                pass
        
        # Metodo 2: AppleScript (macOS fallback)
        try:
            title = "Dragon AI"
            script = f'''
            tell application "System Events"
                tell (first process whose name contains "Python")
                    try
                        set position of window 1 to {{{int(x)}, {int(y)}}}
                    end try
                end tell
            end tell
            '''
            subprocess.Popen(['osascript', '-e', script], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass
    
    def update_widget_position(self):
        """Calcola posizione lungo il perimetro e sposta la finestra"""
        self.border_position += self.border_speed * self.border_direction
        if self.border_position > 1.0:
            self.border_position -= 1.0
        elif self.border_position < 0.0:
            self.border_position += 1.0
        
        margin = 5
        perim = 2 * (self.desktop_w + self.desktop_h)
        pos_px = self.border_position * perim
        
        if pos_px < self.desktop_w:
            x = int(pos_px)
            y = self.desktop_h - self.widget_h - margin
            self.dragon.facing_right = True
        elif pos_px < self.desktop_w + self.desktop_h:
            offset = pos_px - self.desktop_w
            x = self.desktop_w - self.widget_w - margin
            y = self.desktop_h - int(offset) - self.widget_h
            self.dragon.facing_right = False
        elif pos_px < 2 * self.desktop_w + self.desktop_h:
            offset = pos_px - self.desktop_w - self.desktop_h
            x = self.desktop_w - int(offset) - self.widget_w
            y = margin
            self.dragon.facing_right = False
        else:
            offset = pos_px - 2 * self.desktop_w - self.desktop_h
            x = margin
            y = int(offset)
            self.dragon.facing_right = True
        
        x = max(0, min(int(x), self.desktop_w - self.widget_w))
        y = max(0, min(int(y), self.desktop_h - self.widget_h))
        
        self.move_window(x, y)
    
    # =========================================================================
    # FACE DETECTION
    # =========================================================================
    
    def detect_faces(self, frame):
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
                self.last_face_rect = tuple(faces[0])
            
            if not self.face_recognition_enabled or self.authorized_face_encoding is None:
                authorized_user = faces_detected
                for (x,y,w,h) in faces:
                    cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,255), 2)
                return faces_detected, authorized_user, frame
            
            if faces_detected:
                try:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    locs = face_recognition.face_locations(rgb)
                    encs = face_recognition.face_encodings(rgb, locs)
                    for (top,right,bottom,left), enc in zip(locs, encs):
                        matches = face_recognition.compare_faces([self.authorized_face_encoding], enc, tolerance=0.6)
                        dists = face_recognition.face_distance([self.authorized_face_encoding], enc)
                        if matches[0]:
                            authorized_user = True
                            conf = (1 - dists[0]) * 100
                            cv2.rectangle(frame, (left,top), (right,bottom), (0,255,0), 2)
                            cv2.putText(frame, f"OK ({conf:.0f}%)", (left,top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                        else:
                            cv2.rectangle(frame, (left,top), (right,bottom), (0,0,255), 2)
                except Exception:
                    pass
        except Exception:
            pass
        return faces_detected, authorized_user, frame
    
    # =========================================================================
    # VOCE
    # =========================================================================
    
    def speak_text(self, text):
        if self.shutdown:
            return None
        def speak():
            with self.speech_lock:
                if self.shutdown: return
                self.is_speaking = True
                try:
                    subprocess.run(['say', text], check=True)
                except Exception:
                    pass
                finally:
                    self.is_speaking = False
        t = threading.Thread(target=speak, daemon=True)
        try:
            t.start()
            return t
        except RuntimeError:
            return None
    
    def ask_random_question(self):
        if (not self.is_listening and not self.is_speaking and not self.is_processing and 
            not self.is_paused and self.authorized_user_detected):
            q = random.choice(self.random_questions)
            print(f"\n🐉 Drago: {q}")
            self.speak_text(q)
            self.last_question_time = time.time()
            self.is_processing = True
    
    def check_for_control_commands(self, text):
        tl = text.lower()
        if any(w in tl for w in self.pause_words):
            if not self.is_paused:
                print("😴 Pausa!")
                self.is_paused = True
                self.speak_text("Mi metto a dormire. Dimmi drago o sveglia per riattivarmi.")
                return "PAUSA"
        if any(w in tl for w in self.wake_words):
            if self.is_paused:
                print("🔥 Sveglia!")
                self.is_paused = False
                self.speak_text("Eccomi! Sono tornato. Come posso aiutarti?")
                return "RISVEGLIO"
        return None
    
    def listen_for_wake_command(self):
        if self.shutdown or not self.is_paused: return None
        if not self.microphone_lock.acquire(blocking=False): return None
        try:
            with self.microphone as src:
                if self.shutdown or not self.is_paused: return None
                self.recognizer.adjust_for_ambient_noise(src, duration=0.3)
                audio = self.recognizer.listen(src, timeout=3, phrase_time_limit=3)
            if self.shutdown: return None
            try:
                txt = self.recognizer.recognize_google(audio, language='it-IT')
                return self.check_for_control_commands(txt)
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
        if not self.microphone_lock.acquire(blocking=False): return None
        try:
            print("\n🎤 Ascoltando...")
            self.is_listening = True
            with self.microphone as src:
                if self.shutdown: return None
                self.recognizer.adjust_for_ambient_noise(src, duration=1)
                audio = self.recognizer.listen(src, timeout=10)
            if self.shutdown: return None
            try:
                txt = self.recognizer.recognize_google(audio, language='it-IT')
                print(f"📝 Sentito: '{txt}'")
                if self.check_for_control_commands(txt): return None
                return self.get_voice_confirmation(txt)
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
            with self.microphone as src:
                if self.shutdown: return None
                self.recognizer.adjust_for_ambient_noise(src, duration=0.5)
                audio = self.recognizer.listen(src, timeout=8)
            if self.shutdown: return None
            try:
                conf = self.recognizer.recognize_google(audio, language='it-IT').lower()
                print(f"🔊 Conferma: '{conf}'")
                if any(w in conf for w in ['sì','si','yes','vai','ok','okay','invia','perfetto']):
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
    # IMMAGINI + TRADUTTORE + OPENAI
    # =========================================================================
    
    def is_draw_request(self, q):
        return any(k in q.lower() for k in self.draw_keywords)
    
    def extract_image_prompt(self, q):
        p = q
        for kw in sorted(self.draw_keywords, key=len, reverse=True):
            p = re.sub(re.escape(kw), '', p, flags=re.IGNORECASE).strip()
        p = re.sub(r'^(un |una |uno |il |la |lo |i |le |gli |di |del |della |dello )', '', p, flags=re.IGNORECASE).strip()
        return p if p else q
    
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
                    sc = 600 / max(img.shape[:2])
                    img = cv2.resize(img, (int(img.shape[1]*sc), int(img.shape[0]*sc)))
                    cv2.imshow(f'Immagine - {prompt[:30]}', img)
            except Exception:
                pass
            return path, revised, url
        except Exception as e:
            return None, str(e), None
        finally:
            self.is_generating_image = False
    
    def translate_response(self, text, target_lang="inglese"):
        try:
            resp = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": f"Sei un traduttore. Traduci in {target_lang}. Restituisci SOLO la traduzione."},
                    {"role": "user", "content": text}
                ],
                max_tokens=800, temperature=0.2
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            return f"[Errore traduzione: {e}]"
    
    def get_openai_response(self, question):
        try:
            prompt = """Sei un assistente AI con competenze scientifiche avanzate. 
            Rispondi in italiano, dettagliato e scientificamente accurato con esempi concreti."""
            q_lower = question.lower()
            is_sci = any(w in q_lower for w in ['come','perché','meccanismo','funziona','fisica','chimica','biologia','matematica'])
            resp = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role":"system","content":prompt}, {"role":"user","content":question}],
                max_tokens=600 if is_sci else 400,
                temperature=0.3 if is_sci else 0.5,
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
                self.speak_text(f"Genero un'immagine di {prompt}.")
                path, revised, url = self.generate_image(prompt)
                if path:
                    print(f"  🎨 RISPOSTA: Immagine generata!")
                    print(f"  📁 {path}")
                    print(f"{'='*70}\n")
                    self.log_conversation(user_question, f"Immagine: '{prompt}' → {path}", "image", path)
                    self.speak_text(f"Fatto! Immagine di {prompt} pronta.")
                else:
                    print(f"  ❌ Errore: {revised}")
                    print(f"{'='*70}\n")
                    self.log_conversation(user_question, f"Errore: {revised}", "image_error")
                    self.speak_text("Non sono riuscito a generare l'immagine.")
            else:
                response = self.get_openai_response(user_question)
                translation = self.translate_response(response, "inglese") if self.translator_mode else None
                
                print(f"  🤖 RISPOSTA: {response}")
                if translation:
                    print(f"  🌐 ENGLISH: {translation}")
                print(f"{'='*70}\n")
                
                log_r = response + (f"\n\n[EN] {translation}" if translation else "")
                self.log_conversation(user_question, log_r, "text")
                
                speech = response[:800] + "..." if len(response) > 800 else response
                if not self.shutdown:
                    t = self.speak_text(speech)
                    if t: t.join()
                if translation and not self.shutdown:
                    time.sleep(0.5)
                    eng = translation[:500] if len(translation) > 500 else translation
                    t2 = self.speak_text("In English: " + eng)
                    if t2: t2.join()
            
            print(f"💤 Pausa {self.cooldown_after_interaction}s...")
            time.sleep(self.cooldown_after_interaction)
        finally:
            self.is_processing = False
            self.last_question_time = time.time()
    
    # =========================================================================
    # RENDERING DRAGO
    # =========================================================================
    
    def render_dragon_widget(self):
        self.screen.fill((10, 10, 25))
        
        border_col = (40, 40, 80)
        if self.is_paused: border_col = (60, 30, 80)
        elif self.is_speaking: border_col = (120, 50, 20)
        elif self.authorized_user_detected: border_col = (30, 80, 40)
        pygame.draw.rect(self.screen, border_col, (0, 0, self.widget_w, self.widget_h), 2, border_radius=8)
        
        dcx = self.widget_w // 2
        dcy = self.widget_h // 2 + 10
        
        self.dragon.update(is_speaking=self.is_speaking, is_sleeping=self.is_paused, is_processing=self.is_processing)
        
        if not self.is_paused and not self.is_speaking and not self.is_processing:
            if self.current_emotion == "happy":
                self.dragon.set_mood("happy")
            elif self.current_emotion == "sleeping":
                self.dragon.set_mood("sleeping")
            else:
                self.dragon.set_mood("neutral")
        
        fire_x, fire_y = self.dragon.draw(self.screen, dcx, dcy, scale=1.2)
        
        if self.is_speaking:
            self.particles.emit_fire(fire_x, fire_y, count=8, intensity=1.5)
            self.particles.emit_sparkle(fire_x, fire_y, count=2)
        elif self.is_processing or self.is_generating_image:
            self.particles.emit_smoke(dcx, dcy - 30, count=2)
            self.particles.emit_sparkle(dcx, dcy - 40, count=1)
        elif self.is_paused:
            if self.dragon.frame % 20 == 0:
                self.particles.emit_zzz(dcx + 30, dcy - 40)
        elif self.authorized_user_detected:
            if self.dragon.frame % 8 == 0:
                self.particles.emit_fire(fire_x, fire_y, count=1, intensity=0.4)
        
        self.particles.update()
        self.particles.draw(self.screen)
        
        small = pygame.font.Font(None, 16)
        tiny = pygame.font.Font(None, 14)
        
        if self.is_paused:
            self.screen.blit(small.render("💤 Dormendo...", True, (180,130,220)), (8, 6))
        elif not self.authorized_user_detected and self.authorized_face_encoding is not None:
            self.screen.blit(small.render("🚫 Accesso negato", True, (255,80,80)), (8, 6))
        elif self.is_generating_image:
            self.screen.blit(small.render("🎨 Disegnando...", True, (255,200,0)), (8, 6))
        elif self.is_speaking:
            self.screen.blit(small.render("🔥 Parlando...", True, (255,150,50)), (8, 6))
        elif self.is_listening:
            self.screen.blit(small.render("🎤 Ascoltando...", True, (255,255,100)), (8, 6))
        elif self.is_processing:
            self.screen.blit(small.render("🤔 Elaborando...", True, (100,150,255)), (8, 6))
        elif self.authorized_user_detected:
            self.screen.blit(small.render("👤✅ Autorizzato", True, (80,220,80)), (8, 6))
        
        inds = []
        if self.translator_mode: inds.append("🌐 IT↔EN")
        inds.append("H:storia | T:trad | P:pausa")
        self.screen.blit(tiny.render(" | ".join(inds), True, (100,100,130)), (8, self.widget_h - 18))
        
        pygame.display.flip()
    
    # =========================================================================
    # LOOP PRINCIPALE
    # =========================================================================
    
    def run(self):
        clock = pygame.time.Clock()
        last_wake_check = 0
        move_tick = 0
        
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
                            print(f"⏯️ {'Pausa 😴' if self.is_paused else 'Attivo 🔥'}")
                        elif event.key == pygame.K_t:
                            self.translator_mode = not self.translator_mode
                            print(f"🌐 Traduttore: {'ON' if self.translator_mode else 'OFF'}")
                            self.speak_text(f"Traduttore {'attivato' if self.translator_mode else 'disattivato'}.")
                        elif event.key == pygame.K_h:
                            print("📜 Apro cronologia nel browser...")
                            self.history.open()
                        elif event.key == pygame.K_SPACE and self.face_detected and not self.is_processing and not self.is_paused and self.authorized_user_detected:
                            def manual():
                                q = self.listen_to_user()
                                if q and not self.shutdown:
                                    self.complete_interaction(q)
                            threading.Thread(target=manual, daemon=True).start()
                
                if self.is_paused:
                    if current_time - last_wake_check > 2:
                        threading.Thread(target=self.listen_for_wake_command, daemon=True).start()
                        last_wake_check = current_time
                else:
                    ret, frame = self.cap.read()
                    if ret:
                        fd, au, pf = self.detect_faces(frame)
                        self.face_detected = fd
                        self.authorized_user_detected = au
                        if self.last_face_rect and au:
                            self.current_emotion = self.emotion_detector.detect(frame, self.last_face_rect)
                        if self.show_video:
                            cv2.imshow('Webcam', pf)
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                break
                        else:
                            cv2.waitKey(1)
                        
                        if (au and current_time - self.last_question_time > self.question_interval and
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
                
                # Muovi finestra lungo i bordi (ogni 5 frame per fluidità)
                move_tick += 1
                if move_tick >= 5:
                    self.update_widget_position()
                    move_tick = 0
                
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


def main():
    print("🐉 Dragon AI Assistant v13 (FIXED)")
    print("="*50)
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ OPENAI_API_KEY non trovata nel .env!")
        api_key = input("🔑 API key: ").strip()
        if not api_key:
            return
    
    ref_img = "anto2025.png"
    if os.path.exists(ref_img):
        print(f"🔐 Sicurezza ATTIVA ({ref_img})")
    else:
        print(f"⚠️ {ref_img} non trovata - modalità aperta")
        if input("Continuare? (s/n): ").lower() not in ['s','si','sì','y','yes']:
            return
    
    try:
        assistant = DragonAIAssistant(api_key, ref_img)
        print("\n" + "="*50)
        print("  🐉 DRAGON AI v13 - CONTROLLI")
        print("  " + "-"*46)
        print("  SPAZIO  Parla           Q/ESC  Esci")
        print("  V       Video on/off    P      Pausa")
        print("  T       Traduttore IT↔EN")
        print("  H       Cronologia (apre nel browser)")
        print("  " + "-"*46)
        print("  🎤 Voce: PAUSA/STOP | DRAGO/SVEGLIA")
        print("  🎨 Di: 'disegna...' per generare immagini")
        print("  🌐 Traduttore: risponde IT + EN")
        print("="*50 + "\n")
        assistant.run()
    except Exception as e:
        print(f"❌ Errore: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
