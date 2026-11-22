"""
AIR DRAWING STUDIO  -  hand-tracking drawing + voice control + AI calculator
==========================================================================

GESTURES (per hand - both hands work at the same time)
  1 finger  (index)                 -> DRAW
  2 fingers (index + middle)        -> SELECT   (hover a button ~0.8 s to press it)
  3 fingers (index+middle+ring)     -> ERASE    (big eraser follows your finger)
  open palm (hold 1.5 s)            -> CLEAR CANVAS

VOICE (needs internet for Google speech recognition)
  "clear canvas" "save drawing" "undo" "redo" "change color" "change color to red"
  "brush small / medium / large" "bigger" "smaller" "eraser" "rainbow"
  "dark mode" "camera mode" "glow on/off" "shape mode on/off" "auto solve on/off"
  "solve" / "calculate"            -> solves the expression you DREW
  "what is 25 times 4", "square root of 144", "12 plus 7", "5 squared" ...
                                   -> solves the expression you SAID
  "start recording" "stop recording" "mute" "unmute" "help" "quit"

KEYBOARD
  s save | space/enter solve | c clear | u undo | y redo | d dark/camera
  g glow | p shape-snap | a auto-solve | r record | m mic on/off | i help | q quit

DRAWING MATH
  Write e.g.  12 + 7 =   (use a drawn "x" for multiply, a "dash with two dots"
  for divide).  With auto-solve ON the answer appears when you draw "=" and
  pause for ~1.5 s.  Write BIG and with the medium (M) brush for best results.
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # quieter TensorFlow

import ast
import math
import operator
import queue
import re
import threading
import time
from collections import deque
from datetime import datetime

import cv2
import numpy as np
import tensorflow as tf

try:
    import mediapipe as mp          # only needed for the desktop app, not for the web server
except ImportError:
    mp = None

SR_ERROR = None
try:
    import speech_recognition as sr
except Exception as e:              # show the REAL reason instead of guessing
    sr = None
    SR_ERROR = f"{type(e).__name__}: {e}"
    print("[voice] speech_recognition could not be imported ->", SR_ERROR)
    print("[voice] python being used:", __import__('sys').executable)

try:
    import pyttsx3
except ImportError:
    pyttsx3 = None

# ============================================================================
# CONFIG
# ============================================================================
MODEL_PATH = "symbol_model.keras"
LABELS = list("0123456789+-*/=")
W, H = 640, 480
DISPLAY_SIZE = (960, 720)
DWELL_SECONDS = 0.8
PALM_HOLD_SECONDS = 1.5
AUTO_SOLVE_DELAY = 1.5
ERASE_RADIUS = 28
SPEECH_LANG = "en-IN"          # change to "en-US", "hi-IN", ... if you like
SAVE_DIR = "drawings"
REC_DIR = "recordings"
FONT = cv2.FONT_HERSHEY_SIMPLEX

COLORS = {                      # BGR
    "white": (255, 255, 255),
    "red": (60, 60, 255),
    "green": (0, 255, 0),
    "blue": (255, 150, 50),
    "yellow": (0, 255, 255),
    "pink": (180, 105, 255),
    "orange": (0, 165, 255),
    "purple": (200, 50, 160),
    "cyan": (255, 255, 0),
}
PALETTE = ["white", "red", "green", "blue", "yellow", "pink", "rainbow"]
COLOR_CYCLE = ["white", "red", "green", "blue", "yellow", "pink",
               "orange", "purple", "cyan", "rainbow"]
BRUSH_SIZES = [5, 10, 20]


# ============================================================================
# HELPERS - math / speech text
# ============================================================================
_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Mod: operator.mod,
        ast.Pow: operator.pow, ast.FloorDiv: operator.floordiv}
_UN = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def safe_eval(expr):
    """Evaluate an arithmetic expression WITHOUT using eval()."""
    expr = re.sub(r"\b0+(\d)", r"\1", expr)           # 007 -> 7

    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in _BIN:
            a, b = ev(n.left), ev(n.right)
            if isinstance(n.op, ast.Pow) and (abs(b) > 200 or abs(a) > 1e6):
                raise ValueError("number too large")
            return _BIN[type(n.op)](a, b)
        if isinstance(n, ast.UnaryOp) and type(n.op) in _UN:
            return _UN[type(n.op)](ev(n.operand))
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                and n.func.id == "sqrt" and len(n.args) == 1 and not n.keywords):
            return math.sqrt(ev(n.args[0]))
        raise ValueError("unsupported expression")

    value = ev(ast.parse(expr, mode="eval"))
    if isinstance(value, complex):
        raise ValueError("result is not a real number")
    return value


def fmt_value(v):
    if isinstance(v, float):
        if v.is_integer() and abs(v) < 1e15:
            return str(int(v))
        return f"{v:.6g}"
    return str(v)


def pretty_expr(e):
    return e.replace("**", "^").replace("*", "x")


_UNITS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
          "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
          "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
          "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
         "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}


def words_to_numbers(text):
    """'twenty five times four' -> '25 times 4'.  'five eight twelve' stays three numbers."""
    def can_add(cur, v, tens):
        if cur is None:
            return True
        if tens or v >= 10:                       # tens / teens only after nothing or a hundred
            return cur % 100 == 0
        return cur % 100 == 0 or (cur % 100 >= 20 and cur % 10 == 0)

    out, cur = [], None
    text = re.sub(r"(?<=[a-z])-(?=[a-z])", " ", text)
    for tok in text.split():
        val = _UNITS.get(tok, _TENS.get(tok))
        if val is not None:
            if not can_add(cur, val, tok in _TENS):
                out.append(str(cur))
                cur = None
            cur = (cur or 0) + val
        elif tok == "hundred" and cur is not None:
            cur = (cur // 100) * 100 + (cur % 100) * 100
        elif tok == "thousand" and cur is not None:
            cur *= 1000
        else:
            if cur is not None:
                out.append(str(cur))
                cur = None
            out.append(tok)
    if cur is not None:
        out.append(str(cur))
    return " ".join(out)


def spoken_to_expression(t):
    """'what is 25 times 4' -> '25*4'.  Returns None when it is not a math sentence."""
    s = t.lower().replace(",", "")
    s = s.replace("\u00d7", "*").replace("\u00f7", "/").replace("\u2212", "-").replace("\u221a", " square root ")
    s = re.sub(r"[?!]", " ", s)
    s = re.sub(r"\.(?!\d)", " ", s)                           # sentence dots, keep 2.5
    s = re.sub(r"(?<=\d)\s*x\s*(?=\d)", " * ", s)           # 25x4 / 25 x 4
    s = re.sub(r"^\s*((please|hey|ok|okay)\s+)*(what is|what's|whats|what are|calculate|compute|"
               r"solve|find|tell me|how much is|give me)\s+(the\s+)?", "", s)
    s = re.sub(r"\b(equals|equal to|is equal to|equal|is)\b\s*$", "", s)
    num = r"(\d+(?:\.\d+)?)"
    subs = [
        (r"(?:square\s*root|sqrt|root)\s*(?:of\s*)?" + num, r"sqrt(\1)"),
        (num + r"\s*(?:square\s*root|sqrt)", r"sqrt(\1)"),
        (num + r"\s*squared", r"(\1)**2"),
        (num + r"\s*cubed", r"(\1)**3"),
        (num + r"\s*(?:percent|per cent|%)\s*of\s*" + num, r"(\1/100*\2)"),
        (r"to the power of|to the power|raised to|\bpower\b", "**"),
        (r"multiplied by|multiply by|multiplied with|multiplied|\bmultiply\b|\btimes\b|\binto\b", "*"),
        (r"divided by|divide by|divided|\bover\b", "/"),
        (r"\bplus\b|\badd\b", "+"),
        (r"\bminus\b|\bsubtract\b|\bless\b", "-"),
        (r"\bmodulo\b|\bmod\b|\bremainder\b", "%"),
    ]
    for pat, rep in subs:
        s = re.sub(pat, rep, s)
    s = re.sub(r"\s+", "", s)
    if not s or not re.search(r"\d", s) or not re.search(r"[+\-*/%]|sqrt", s):
        return None
    if not re.fullmatch(r"[0-9+\-*/%.()]+", s.replace("sqrt", "")):
        return None
    return s


# ============================================================================
# HELPERS - vision
# ============================================================================
def prep_symbol(crop):
    """IDENTICAL to train_model.py: uint8 mask of ONE symbol -> 28x28 uint8."""
    h, w = crop.shape[:2]
    s = max(h, w)
    sq = np.zeros((s, s), np.uint8)
    y0, x0 = (s - h) // 2, (s - w) // 2
    sq[y0:y0 + h, x0:x0 + w] = crop
    small = cv2.resize(sq, (20, 20), interpolation=cv2.INTER_AREA)
    return np.pad(small, 4)


def _should_merge(a, b):
    """Merge boxes stacked vertically (for '=' and the division sign)."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x_overlap = min(ax + aw, bx + bw) - max(ax, bx)
    if x_overlap <= 0.5 * min(aw, bw):
        return False
    gap = max(ay, by) - min(ay + ah, by + bh)
    return gap < 0.6 * max(aw, bw, ah, bh) + 10


def merge_boxes(boxes):
    boxes = [tuple(b) for b in boxes]
    changed = True
    while changed:
        changed = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                if _should_merge(boxes[i], boxes[j]):
                    ax, ay, aw, ah = boxes[i]
                    bx, by, bw, bh = boxes[j]
                    x1, y1 = min(ax, bx), min(ay, by)
                    x2, y2 = max(ax + aw, bx + bw), max(ay + ah, by + bh)
                    boxes[i] = (x1, y1, x2 - x1, y2 - y1)
                    del boxes[j]
                    changed = True
                    break
            if changed:
                break
    return boxes


def rounded_rect(img, p1, p2, color, r=8):
    x1, y1 = p1
    x2, y2 = p2
    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
    cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1)
    for cx, cy in ((x1 + r, y1 + r), (x2 - r, y1 + r), (x1 + r, y2 - r), (x2 - r, y2 - r)):
        cv2.circle(img, (cx, cy), r, color, -1)


def put(img, text, org, scale=0.5, color=(255, 255, 255), thick=1):
    cv2.putText(img, text, (org[0] + 1, org[1] + 1), FONT, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, FONT, scale, color, thick, cv2.LINE_AA)


def center_text(img, text, rect, scale=0.45, color=(255, 255, 255), thick=1):
    x1, y1, x2, y2 = rect
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, thick)
    cv2.putText(img, text, (x1 + (x2 - x1 - tw) // 2, y1 + (y2 - y1 + th) // 2),
                FONT, scale, color, thick, cv2.LINE_AA)


def panel(img, p1, p2, color=(0, 0, 0), alpha=0.55):
    x1, y1 = max(0, p1[0]), max(0, p1[1])
    x2, y2 = min(img.shape[1], p2[0]), min(img.shape[0], p2[1])
    if x2 <= x1 or y2 <= y1:
        return
    sub = img[y1:y2, x1:x2]
    img[y1:y2, x1:x2] = cv2.addWeighted(sub, 1 - alpha, np.full_like(sub, color), alpha, 0)


def ascii_only(s):
    return s.encode("ascii", "ignore").decode()


# ============================================================================
# BACKGROUND THREADS - voice in / voice out
# ============================================================================
class Speaker(threading.Thread):
    """Text-to-speech in its own thread (optional: pip install pyttsx3)."""

    def __init__(self):
        super().__init__(daemon=True)
        self.q = queue.Queue()
        self.muted = False
        self.available = pyttsx3 is not None
        self.busy = threading.Event()
        self.start()

    def say(self, text):
        if self.available and not self.muted:
            self.q.put(text)

    def run(self):
        if not self.available:
            return
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", 165)
        except Exception:
            self.available = False
            return
        while True:
            text = self.q.get()
            self.busy.set()
            try:
                engine.say(text)
                engine.runAndWait()
            except Exception:
                pass
            time.sleep(0.6)           # let the echo die so the mic does not hear us
            if self.q.empty():
                self.busy.clear()


class VoiceListener(threading.Thread):
    """Continuously listens to the microphone and pushes recognised text to a queue."""

    def __init__(self, out_queue, speaker):
        super().__init__(daemon=True)
        self.q = out_queue
        self.speaker = speaker
        self.running = True
        self.muted = False
        self.status = "starting"

    def run(self):
        if sr is None:
            self.status = "voice library missing - see terminal"
            return
        recognizer = sr.Recognizer()
        recognizer.dynamic_energy_threshold = True
        recognizer.pause_threshold = 0.6
        try:
            mic = sr.Microphone()
        except Exception as e:                      # no PyAudio / no microphone
            self.status = "no microphone (" + type(e).__name__ + ")"
            return
        try:
            with mic as source:
                recognizer.adjust_for_ambient_noise(source, duration=1)
                self.status = "listening"
                while self.running:
                    try:
                        audio = recognizer.listen(source, timeout=2, phrase_time_limit=5)
                    except sr.WaitTimeoutError:
                        continue
                    except Exception:
                        continue
                    if self.muted or self.speaker.busy.is_set():
                        continue
                    try:
                        text = recognizer.recognize_google(audio, language=SPEECH_LANG)
                        self.q.put(text)
                        self.status = "listening"
                    except sr.UnknownValueError:
                        pass
                    except sr.RequestError:
                        self.status = "offline - voice needs internet"
        except Exception as e:
            self.status = "mic error (" + type(e).__name__ + ")"


# ============================================================================
# UI
# ============================================================================
class Button:
    def __init__(self, name, label, x, y, w, h, fill=None):
        self.name, self.label, self.fill = name, label, fill
        self.rect = (x, y, x + w, y + h)

    def contains(self, x, y, margin=0):
        x1, y1, x2, y2 = self.rect
        return x1 - margin <= x <= x2 + margin and y1 - margin <= y <= y2 + margin


class HandState:
    def __init__(self):
        self.pos = None
        self.prev = None
        self.stroke = []
        self.drawing = False
        self.kind = "pen"
        self.mode = "idle"
        self.hover = None
        self.hover_t = 0.0
        self.hover_done = False
        self.palm_t = None
        self.palm_done = False
        self.progress = 0.0


# ============================================================================
# MAIN APPLICATION
# ============================================================================
class AirStudio:
    def __init__(self):
        if not os.path.exists(MODEL_PATH):
            raise SystemExit(f"\n'{MODEL_PATH}' not found.\n"
                             "Run  python train_model.py  first (takes a few minutes).\n")
        self.model = tf.keras.models.load_model(MODEL_PATH)
        if self.model.output_shape[-1] != len(LABELS):
            raise SystemExit("\nThe saved model has the wrong number of classes.\n"
                             "Delete it and run  python train_model.py  again.\n")
        self.model(np.zeros((1, 28, 28, 1), np.float32), training=False)   # warm-up
        print("AI model loaded.")

        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(max_num_hands=2,
                                         min_detection_confidence=0.6,
                                         min_tracking_confidence=0.6)

        # drawing state
        self.canvas = np.zeros((H, W, 3), np.uint8)
        self.undo_stack = deque(maxlen=30)
        self.redo_stack = []
        self.hand_states = {}
        self.current_lms = []
        self.tool = "pen"
        self.color_name = "white"
        self.brush = 10
        self.background = "camera"
        self.glow = True
        self.shape_mode = False
        self.auto_solve = True
        self.dirty = False
        self.last_stroke_end = 0.0

        # outputs
        self.result = None                 # (expr, value, timestamp)
        self.history = deque(maxlen=5)
        self.rec_overlay = None            # (symbols, expiry)
        self.toast_text, self.toast_until = "", 0.0
        self.last_heard, self.last_heard_t = "", 0.0
        self.show_help = False
        self.save_pending = False
        self.recording, self.writer = False, None
        self.running = True
        self.fps, self._t_prev = 0.0, time.time()

        # voice
        self.voice_q = queue.Queue()
        self.speaker = Speaker()
        self.listener = VoiceListener(self.voice_q, self.speaker)
        self.listener.start()
        self._last_status = ""

        self.build_ui()

    # ------------------------------------------------------------------ UI --
    def build_ui(self):
        self.buttons = []
        x = 100
        for name in PALETTE:
            self.buttons.append(Button(f"color:{name}", "", x, 10, 56, 38, COLORS.get(name)))
            x += 60
        x = 530
        for label, size in (("S", 5), ("M", 10), ("L", 20)):
            self.buttons.append(Button(f"size:{size}", label, x, 10, 32, 38))
            x += 36
        y = 60
        for name, label in [("eraser", "ERASER"), ("undo", "UNDO"), ("redo", "REDO"),
                            ("clear", "CLEAR"), ("save", "SAVE"), ("solve", "SOLVE"),
                            ("bg", "DARK"), ("shape", "SHAPE"), ("glow", "GLOW"),
                            ("rec", "REC")]:
            self.buttons.append(Button(name, label, 10, y, 80, 32))
            y += 37

        hsv = np.zeros((38, 56, 3), np.uint8)
        hsv[:, :, 0] = np.linspace(0, 179, 56).astype(np.uint8)[None, :]
        hsv[:, :, 1] = 255
        hsv[:, :, 2] = 255
        self.rainbow_strip = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def is_active(self, name):
        if name.startswith("color:"):
            return self.tool == "pen" and self.color_name == name[6:]
        if name.startswith("size:"):
            return self.brush == int(name[5:])
        return {"eraser": self.tool == "eraser", "bg": self.background == "dark",
                "shape": self.shape_mode, "glow": self.glow,
                "rec": self.recording}.get(name, False)

    def draw_buttons(self, out):
        for b in self.buttons:
            x1, y1, x2, y2 = b.rect
            active = self.is_active(b.name)
            if b.name == "color:rainbow":
                out[y1:y2, x1:x2] = self.rainbow_strip
            elif b.fill is not None:
                rounded_rect(out, (x1, y1), (x2, y2), b.fill, 6)
            else:
                rounded_rect(out, (x1, y1), (x2, y2), (40, 120, 40) if active else (70, 70, 70), 6)
                label = b.label
                if b.name == "bg":
                    label = "CAMERA" if self.background == "dark" else "DARK"
                elif b.name == "rec":
                    label = "STOP" if self.recording else "REC"
                center_text(out, label, b.rect)
            if active:
                cv2.rectangle(out, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), (255, 255, 255), 2)

    def button_at(self, x, y, margin=0):
        for b in self.buttons:
            if b.contains(x, y, margin):
                return b
        return None

    def toast(self, text, seconds=2.2):
        self.toast_text, self.toast_until = ascii_only(text), time.time() + seconds
        print("[info]", text)

    # ------------------------------------------------------------- actions --
    def current_color(self):
        if self.tool == "eraser":
            return (0, 0, 0)
        if self.color_name == "rainbow":
            hue = int((time.time() * 70) % 180)
            return tuple(int(c) for c in cv2.cvtColor(np.uint8([[[hue, 255, 255]]]),
                                                      cv2.COLOR_HSV2BGR)[0, 0])
        return COLORS[self.color_name]

    def set_color(self, name):
        if name == "rainbow" or name in COLORS:
            self.tool, self.color_name = "pen", name
            self.toast(f"Color: {name}")

    def cycle_color(self):
        i = COLOR_CYCLE.index(self.color_name) if self.tool == "pen" else -1
        self.set_color(COLOR_CYCLE[(i + 1) % len(COLOR_CYCLE)])

    def use_eraser(self):
        self.tool = "eraser"
        self.toast("Eraser")

    def set_brush(self, size):
        self.brush = size
        self.tool = "pen" if self.tool == "eraser" else self.tool
        self.toast(f"Brush size: {size}")

    def push_undo(self):
        self.undo_stack.append(self.canvas.copy())
        self.redo_stack.clear()

    def undo_action(self):
        if self.undo_stack:
            self.redo_stack.append(self.canvas.copy())
            self.canvas = self.undo_stack.pop()
            self.toast("Undo")
        else:
            self.toast("Nothing to undo")

    def redo_action(self):
        if self.redo_stack:
            self.undo_stack.append(self.canvas.copy())
            self.canvas = self.redo_stack.pop()
            self.toast("Redo")
        else:
            self.toast("Nothing to redo")

    def clear_canvas(self):
        self.push_undo()
        self.canvas[:] = 0
        for hs in self.hand_states.values():
            hs.drawing, hs.stroke, hs.prev = False, [], None
        self.result = None
        self.rec_overlay = None
        self.dirty = False
        self.toast("Canvas cleared (say 'undo' to bring it back)")

    def request_save(self):
        self.save_pending = True

    def toggle_bg(self, value=None):
        self.background = ("dark" if self.background == "camera" else "camera") if value is None else value
        self.toast("Dark mode" if self.background == "dark" else "Camera mode")

    def toggle_shape(self, value=None):
        self.shape_mode = (not self.shape_mode) if value is None else value
        self.toast("Shape snapping " + ("ON" if self.shape_mode else "OFF"))

    def toggle_glow(self, value=None):
        self.glow = (not self.glow) if value is None else value
        self.toast("Neon glow " + ("ON" if self.glow else "OFF"))

    def toggle_auto(self, value=None):
        self.auto_solve = (not self.auto_solve) if value is None else value
        self.toast("Auto-solve " + ("ON" if self.auto_solve else "OFF"))

    def toggle_mic(self):
        self.listener.muted = not self.listener.muted
        self.toast("Microphone " + ("muted" if self.listener.muted else "listening"))

    def toggle_record(self, value=None):
        want = (not self.recording) if value is None else value
        if want and not self.recording:
            os.makedirs(REC_DIR, exist_ok=True)
            path = os.path.join(REC_DIR, datetime.now().strftime("session_%Y%m%d_%H%M%S.mp4"))
            self.writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (W, H))
            self.recording = self.writer.isOpened()
            self.toast(f"Recording -> {path}" if self.recording else "Could not start recording")
        elif not want and self.recording:
            self.recording = False
            self.writer.release()
            self.writer = None
            self.toast("Recording saved")

    def press(self, name):
        if name.startswith("color:"):
            self.set_color(name[6:])
        elif name.startswith("size:"):
            self.set_brush(int(name[5:]))
        else:
            {"eraser": self.use_eraser, "undo": self.undo_action, "redo": self.redo_action,
             "clear": self.clear_canvas, "save": self.request_save, "solve": self.solve_canvas,
             "bg": self.toggle_bg, "shape": self.toggle_shape, "glow": self.toggle_glow,
             "rec": self.toggle_record}[name]()

    # --------------------------------------------------------- recognition --
    def recognize(self):
        """Segment the canvas into symbols and classify them. -> [(box, label, conf)]"""
        mask = (self.canvas.max(axis=2) > 40).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = [cv2.boundingRect(c) for c in contours]
        boxes = [b for b in boxes if b[2] >= 4 and b[3] >= 4]
        if not boxes:
            return []
        boxes = merge_boxes(boxes)
        biggest = max(max(b[2], b[3]) for b in boxes)
        boxes = [b for b in boxes if max(b[2], b[3]) >= 0.12 * biggest]   # drop stray dots
        boxes.sort(key=lambda b: b[0])
        batch = np.stack([prep_symbol(mask[y:y + h, x:x + w]) for x, y, w, h in boxes])
        batch = (batch.astype("float32") / 255.0).reshape(-1, 28, 28, 1)
        probs = self.model(batch, training=False).numpy()
        result = []
        for box, p in zip(boxes, probs):
            i = int(np.argmax(p))
            result.append((box, LABELS[i], float(p[i])))
        return result

    def solve_canvas(self, auto=False):
        symbols = self.recognize()
        if not symbols:
            if not auto:
                self.toast("Nothing to solve - draw an expression first")
            return
        if auto and symbols[-1][1] != "=":
            return
        self.rec_overlay = (symbols, time.time() + 6)
        expr = "".join(s[1] for s in symbols)
        if "=" in expr:
            expr = expr.split("=")[0]
        if not expr:
            self.toast("Write something before the '='")
            return
        try:
            value = fmt_value(safe_eval(expr))
        except ZeroDivisionError:
            self.show_result(pretty_expr(expr), "cannot divide by zero")
            return
        except Exception:
            self.toast(f"Could not understand: {pretty_expr(expr)}", 3.5)
            return
        self.show_result(pretty_expr(expr), value)
        if min(s[2] for s in symbols) < 0.6:
            self.toast("Some symbols were unclear (orange) - draw bigger / cleaner", 3.5)

    def solve_spoken(self, expr):
        try:
            value = fmt_value(safe_eval(expr))
        except ZeroDivisionError:
            self.show_result(pretty_expr(expr), "cannot divide by zero")
            return
        except Exception:
            self.toast("Sorry, I could not calculate that")
            return
        self.show_result(pretty_expr(expr), value)

    def show_result(self, expr, value):
        self.result = (expr, value, time.time())
        self.history.appendleft(f"{expr} = {value}")
        print(f"[calc] {expr} = {value}")
        self.speaker.say("The answer is " + value.replace("-", "minus "))

    # --------------------------------------------------------------- voice --
    def drain_voice(self):
        while True:
            try:
                text = self.voice_q.get_nowait()
            except queue.Empty:
                break
            self.handle_voice(text)
        status = self.listener.status
        if status != self._last_status:
            self._last_status = status
            if status not in ("starting", "listening"):
                self.toast("Voice: " + status, 4)

    def handle_voice(self, raw):
        self.last_heard, self.last_heard_t = raw, time.time()
        t = words_to_numbers(raw.lower().strip())
        print("[voice]", t)

        expr = spoken_to_expression(t)
        if expr:
            self.solve_spoken(expr)
            return

        words = set(re.findall(r"[a-z']+", t))
        off = bool(words & {"off", "disable", "disabled"})

        if words & {"quit", "exit"} or "close the app" in t:
            self.running = False
        elif "help" in words or "commands" in words:
            self.show_help = not self.show_help
        elif "unmute" in words:
            self.speaker.muted = False
            self.toast("Voice replies ON")
        elif "mute" in words:
            self.speaker.muted = True
            self.toast("Voice replies OFF")
        elif words & {"clear", "wipe", "reset"} or "erase all" in t or "erase everything" in t:
            self.clear_canvas()
        elif "undo" in words:
            self.undo_action()
        elif "redo" in words:
            self.redo_action()
        elif words & {"save", "screenshot", "snapshot", "capture"}:
            self.request_save()
        elif words & {"record", "recording"}:
            self.toggle_record(not (words & {"stop", "end", "finish"}))
        elif words & {"solve", "calculate", "compute", "answer", "result", "equals", "equal"}:
            self.solve_canvas()
        elif "auto" in words:
            self.toggle_auto(not off)
        elif "glow" in words or "neon" in words:
            self.toggle_glow(not off)
        elif words & {"shape", "shapes"}:
            self.toggle_shape(not off)
        elif words & {"eraser", "erase", "rubber"}:
            self.use_eraser()
        elif words & {"dark", "black"}:
            self.toggle_bg("dark")
        elif words & {"camera", "webcam", "normal"}:
            self.toggle_bg("camera")
        elif "rainbow" in words:
            self.set_color("rainbow")
        elif words & {"bigger", "larger", "increase", "thicker"}:
            i = BRUSH_SIZES.index(self.brush) if self.brush in BRUSH_SIZES else 1
            self.set_brush(BRUSH_SIZES[min(i + 1, 2)])
        elif words & {"smaller", "decrease", "thinner", "reduce"}:
            i = BRUSH_SIZES.index(self.brush) if self.brush in BRUSH_SIZES else 1
            self.set_brush(BRUSH_SIZES[max(i - 1, 0)])
        elif words & {"small", "thin", "fine"}:
            self.set_brush(5)
        elif words & {"medium", "normal", "regular"}:
            self.set_brush(10)
        elif words & {"large", "big", "thick"}:
            self.set_brush(20)
        else:
            named = [c for c in COLORS if c in words]
            if named:
                self.set_color(named[-1])
            elif words & {"color", "colour"}:
                self.cycle_color()
            else:
                self.toast(f"Heard: '{raw}' - say 'help' for commands", 3)

    # ------------------------------------------------------ hand / drawing --
    def over_ui(self, x, y):
        return self.button_at(x, y, margin=6) is not None

    def process_hands(self, results, now):
        seen = set()
        self.current_lms = []
        if results.multi_hand_landmarks:
            handed = results.multi_handedness or []
            for i, lms in enumerate(results.multi_hand_landmarks):
                label = handed[i].classification[0].label if i < len(handed) else str(i)
                key = label if label not in seen else f"{label}{i}"
                seen.add(key)
                hs = self.hand_states.setdefault(key, HandState())
                self.current_lms.append(lms)
                self.update_hand(lms, hs, now)
        for key in list(self.hand_states):
            if key not in seen:
                self.end_stroke(self.hand_states[key])
                del self.hand_states[key]

    def update_hand(self, lms, hs, now):
        lm = lms.landmark
        raw = np.array([lm[8].x * W, lm[8].y * H], np.float32)
        hs.pos = raw if hs.pos is None else 0.5 * hs.pos + 0.5 * raw     # smoothing
        x, y = int(hs.pos[0]), int(hs.pos[1])

        up = [lm[t].y < lm[p].y for t, p in ((8, 6), (12, 10), (16, 14), (20, 18))]
        if all(up):
            mode = "palm"
        elif up[0] and up[1] and up[2]:
            mode = "erase"
        elif up[0] and up[1]:
            mode = "select"
        elif up[0]:
            mode = "draw"
        else:
            mode = "idle"
        hs.mode = mode
        hs.progress = 0.0

        if mode != "palm":
            hs.palm_t, hs.palm_done = None, False
        if mode not in ("draw", "erase"):
            self.end_stroke(hs)

        if mode == "select":
            self.hover_logic(hs, x, y, now)
        else:
            hs.hover, hs.hover_done = None, False

        if mode == "palm":
            if hs.palm_t is None:
                hs.palm_t = now
            if not hs.palm_done:
                hs.progress = (now - hs.palm_t) / PALM_HOLD_SECONDS
                if hs.progress >= 1:
                    self.clear_canvas()
                    hs.palm_done, hs.progress = True, 0.0
        elif mode == "draw":
            self.draw_step(hs, x, y)
        elif mode == "erase":
            self.erase_step(hs, x, y)

    def hover_logic(self, hs, x, y, now):
        b = self.button_at(x, y)
        name = b.name if b else None
        if name != hs.hover:
            hs.hover, hs.hover_t, hs.hover_done = name, now, False
        if b and not hs.hover_done:
            hs.progress = (now - hs.hover_t) / DWELL_SECONDS
            if hs.progress >= 1:
                self.press(b.name)
                hs.hover_done, hs.progress = True, 0.0

    def draw_step(self, hs, x, y):
        if self.over_ui(x, y):
            self.end_stroke(hs)
            return
        if hs.drawing and hs.kind != "pen":
            self.end_stroke(hs)
        erasing = self.tool == "eraser"
        color = (0, 0, 0) if erasing else self.current_color()
        thick = self.brush * 2 if erasing else self.brush
        ltype = cv2.LINE_8 if erasing else cv2.LINE_AA
        preview_only = self.shape_mode and not erasing
        if not hs.drawing:
            self.push_undo()
            hs.drawing, hs.kind, hs.stroke, hs.prev = True, "pen", [(x, y)], (x, y)
            if not preview_only:
                cv2.circle(self.canvas, (x, y), max(1, thick // 2), color, -1, ltype)
        else:
            hs.stroke.append((x, y))
            if not preview_only:
                cv2.line(self.canvas, hs.prev, (x, y), color, thick, ltype)
            hs.prev = (x, y)

    def erase_step(self, hs, x, y):
        if not hs.drawing or hs.kind != "erase":
            self.end_stroke(hs)
            self.push_undo()
            hs.drawing, hs.kind, hs.prev = True, "erase", (x, y)
        cv2.line(self.canvas, hs.prev, (x, y), (0, 0, 0), ERASE_RADIUS * 2)
        cv2.circle(self.canvas, (x, y), ERASE_RADIUS, (0, 0, 0), -1)
        hs.prev = (x, y)

    def end_stroke(self, hs):
        if hs.drawing:
            if hs.kind == "pen" and self.shape_mode and self.tool == "pen":
                self.commit_shape(hs.stroke)
            self.dirty = True
            self.last_stroke_end = time.time()
        hs.drawing, hs.stroke, hs.prev = False, [], None

    def commit_shape(self, stroke):
        """Turn a rough stroke into a clean line / circle / triangle / rectangle."""
        color, th = self.current_color(), self.brush
        if len(stroke) < 2:
            return
        pts = np.array(stroke, np.int32)
        x, y, w, h = cv2.boundingRect(pts)
        diag = math.hypot(w, h)

        def freeform():
            cv2.polylines(self.canvas, [pts.reshape(-1, 1, 2)], False, color, th, cv2.LINE_AA)

        if len(stroke) < 8 or diag < 25:
            freeform()
            return
        closed = math.dist(stroke[0], stroke[-1]) < 0.25 * diag
        if closed:
            cnt = pts.reshape(-1, 1, 2)
            area = cv2.contourArea(cnt)
            per = cv2.arcLength(cnt, True)
            (cx, cy), r = cv2.minEnclosingCircle(cnt)
            if area > 0 and area / (math.pi * r * r) > 0.8:
                cv2.circle(self.canvas, (int(cx), int(cy)), int(r), color, th, cv2.LINE_AA)
                return
            approx = cv2.approxPolyDP(cnt, 0.04 * per, True)
            if len(approx) == 3:
                cv2.polylines(self.canvas, [approx], True, color, th, cv2.LINE_AA)
                return
            rect = cv2.minAreaRect(cnt)
            rect_area = rect[1][0] * rect[1][1]
            if len(approx) == 4 or (rect_area > 0 and area / rect_area > 0.8):
                box = cv2.boxPoints(rect).astype(np.int32)
                cv2.polylines(self.canvas, [box.reshape(-1, 1, 2)], True, color, th, cv2.LINE_AA)
                return
            freeform()
        else:
            vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
            dev = np.abs((pts[:, 0] - x0) * vy - (pts[:, 1] - y0) * vx).max()
            if dev < 0.08 * diag:
                cv2.line(self.canvas, stroke[0], stroke[-1], color, th, cv2.LINE_AA)
            else:
                freeform()

    # -------------------------------------------------------------- render --
    def render(self, frame, now):
        out = np.zeros_like(frame) if self.background == "dark" else frame
        mask = self.canvas.max(axis=2) > 12
        out[mask] = self.canvas[mask]
        if self.glow:
            out = cv2.addWeighted(out, 1.0, cv2.GaussianBlur(self.canvas, (0, 0), 9), 0.8, 0)

        # live preview of strokes that will be snapped to shapes
        if self.shape_mode and self.tool == "pen":
            for hs in self.hand_states.values():
                if hs.drawing and hs.kind == "pen" and len(hs.stroke) > 1:
                    cv2.polylines(out, [np.array(hs.stroke, np.int32).reshape(-1, 1, 2)],
                                  False, self.current_color(), self.brush, cv2.LINE_AA)

        if self.save_pending:                       # clean image, no toolbar
            self.save_pending = False
            os.makedirs(SAVE_DIR, exist_ok=True)
            scene = out.copy()
            stamp = datetime.now().strftime("%d-%m-%Y  %H:%M:%S")
            cv2.rectangle(scene, (5, H - 28), (185, H - 5), (0, 0, 0), -1)
            cv2.putText(scene, stamp, (10, H - 10), FONT, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
            path = os.path.join(SAVE_DIR, datetime.now().strftime("drawing_%Y%m%d_%H%M%S.png"))
            cv2.imwrite(path, scene)
            self.toast(f"Saved: {path}")

        for lms in self.current_lms:
            self.mp_draw.draw_landmarks(out, lms, self.mp_hands.HAND_CONNECTIONS)

        # pointers
        for hs in self.hand_states.values():
            if hs.pos is None:
                continue
            x, y = int(hs.pos[0]), int(hs.pos[1])
            if hs.mode == "erase":
                cv2.circle(out, (x, y), ERASE_RADIUS, (255, 255, 255), 2)
            elif hs.mode == "select":
                cv2.circle(out, (x, y), 12, (0, 255, 255), 2)
            elif hs.mode == "draw":
                cv2.circle(out, (x, y), max(5, self.brush // 2 + 2), (255, 255, 255), 1)
                cv2.circle(out, (x, y), max(3, self.brush // 2), self.current_color(), -1)
            else:
                cv2.circle(out, (x, y), 6, (160, 160, 160), 1)
            if hs.progress > 0:
                cv2.ellipse(out, (x, y), (22, 22), -90, 0, int(360 * min(hs.progress, 1)),
                            (0, 255, 0), 3)
            put(out, hs.mode.upper(), (x + 16, y - 14), 0.4)

        # recognised symbols overlay
        if self.rec_overlay and now < self.rec_overlay[1]:
            for (x, y, w, h), label, conf in self.rec_overlay[0]:
                col = (0, 255, 0) if conf >= 0.6 else (0, 140, 255)
                cv2.rectangle(out, (x - 4, y - 4), (x + w + 4, y + h + 4), col, 2)
                put(out, f"{label.replace('*', 'x')} {int(conf * 100)}%",
                    (x, max(14, y - 10)), 0.45, col)

        self.draw_buttons(out)
        self.draw_status(out, now)
        return out

    def draw_status(self, out, now):
        # microphone indicator
        ok = self.listener.status == "listening" and not self.listener.muted
        col = (0, 200, 0) if ok else ((120, 120, 120) if self.listener.muted else (0, 0, 220))
        cv2.circle(out, (24, 24), 9, col, -1)
        put(out, "MIC", (40, 29), 0.45, col)

        if self.recording and int(now * 2) % 2 == 0:
            cv2.circle(out, (112, 66), 6, (0, 0, 255), -1)
            put(out, "REC", (124, 71), 0.45, (0, 0, 255))

        # result panel
        if self.result and now < self.result[2] + 25:
            expr, val, _ = self.result
            panel(out, (100, H - 82), (W - 10, H - 24), alpha=0.6)
            put(out, "Expression: " + expr, (110, H - 60), 0.55, (0, 255, 255))
            put(out, "Answer: " + val, (110, H - 33), 0.8, (0, 255, 0), 2)

        # history
        if self.history:
            n = len(self.history)
            panel(out, (W - 250, 58), (W - 10, 80 + 18 * n), alpha=0.45)
            put(out, "History", (W - 244, 74), 0.42, (180, 180, 180))
            for i, item in enumerate(self.history):
                put(out, item[:34], (W - 244, 94 + 18 * i), 0.45)

        if now - self.last_heard_t < 6 and self.last_heard:
            put(out, 'Heard: "' + ascii_only(self.last_heard)[:60] + '"', (100, H - 8), 0.45, (200, 200, 255))

        put(out, f"{self.fps:.0f} FPS", (W - 62, H - 8), 0.4, (180, 180, 180))

        if now < self.toast_until:
            (tw, th), _ = cv2.getTextSize(self.toast_text, FONT, 0.6, 2)
            x = max(100, (W - tw) // 2)
            panel(out, (x - 12, 54), (x + tw + 12, 54 + th + 22), alpha=0.7)
            put(out, self.toast_text, (x, 54 + th + 10), 0.6, (255, 255, 255), 2)

        if self.show_help:
            lines = [
                "GESTURES",
                " 1 finger = draw    2 fingers = select (hover)",
                " 3 fingers = erase  open palm (hold) = clear",
                "VOICE",
                ' "clear canvas"  "save drawing"  "undo"  "redo"',
                ' "change color [to red]"  "rainbow"  "eraser"',
                ' "brush small/medium/large"  "bigger"  "smaller"',
                ' "dark mode"  "camera mode"  "glow on/off"',
                ' "shape mode on/off"  "auto solve on/off"',
                ' "solve"  -> solves what you drew',
                ' "what is 25 times 4"  "square root of 144"',
                ' "start/stop recording"  "mute"  "unmute"  "quit"',
                "KEYS",
                " s save  space solve  c clear  u undo  y redo  d dark",
                " g glow  p shapes  a auto  r record  m mic  i help  q quit",
            ]
            panel(out, (100, 100), (W - 10, 100 + 18 * len(lines) + 14), alpha=0.8)
            for i, ln in enumerate(lines):
                head = ln.isupper()
                put(out, ln, (110, 122 + 18 * i), 0.45, (0, 255, 255) if head else (255, 255, 255))

    # ------------------------------------------------------------ main loop --
    def handle_key(self, key):
        if key == 255:
            return
        if key in (ord("q"), 27):
            self.running = False
        elif key == ord("s"):
            self.request_save()
        elif key in (32, 13):
            self.solve_canvas()
        elif key == ord("c"):
            self.clear_canvas()
        elif key == ord("u"):
            self.undo_action()
        elif key == ord("y"):
            self.redo_action()
        elif key == ord("d"):
            self.toggle_bg()
        elif key == ord("g"):
            self.toggle_glow()
        elif key == ord("p"):
            self.toggle_shape()
        elif key == ord("a"):
            self.toggle_auto()
        elif key == ord("r"):
            self.toggle_record()
        elif key == ord("m"):
            self.toggle_mic()
        elif key == ord("i"):
            self.show_help = not self.show_help

    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
        if not cap.isOpened():
            raise SystemExit("Could not open the webcam.")
        print(__doc__)
        self.toast("Say 'help' or press I for commands", 4)

        while self.running:
            ok, frame = cap.read()
            if not ok:
                print("Camera frame not received - stopping.")
                break
            frame = cv2.resize(cv2.flip(frame, 1), (W, H))
            now = time.time()
            self.fps = 0.9 * self.fps + 0.1 / max(now - self._t_prev, 1e-3)
            self._t_prev = now

            results = self.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            self.process_hands(results, now)
            self.drain_voice()

            # auto-solve: an '=' was drawn and the user paused
            if (self.auto_solve and self.dirty
                    and not any(h.drawing for h in self.hand_states.values())
                    and now - self.last_stroke_end > AUTO_SOLVE_DELAY):
                self.dirty = False
                self.solve_canvas(auto=True)

            out = self.render(frame, now)
            if self.recording and self.writer is not None:
                self.writer.write(out)

            cv2.imshow("Air Drawing Studio", cv2.resize(out, DISPLAY_SIZE))
            self.handle_key(cv2.waitKey(1) & 0xFF)

        self.listener.running = False
        if self.recording:
            self.toggle_record(False)
        cap.release()
        self.hands.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    AirStudio().run()
