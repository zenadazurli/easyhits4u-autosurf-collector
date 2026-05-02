#!/usr/bin/env python3
# multi_surf_collector.py - Multi-account con FAISS + raccolta captcha matematici

import os
import sys
import time
import json
import threading
import signal
import requests
import numpy as np
import cv2
from datetime import datetime
from supabase import create_client
from datasets import load_dataset
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from config import ACCOUNTS, MAX_CONCURRENT_ACCOUNTS, STAGGERED_START_DELAY, DATASET_REPO

# ================ GLOBALS ====================
X_fast = None
y_fast = None
classes_fast = None
DIM = 64

# ================ LOG ====================
def log(account_name, msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}][{account_name}] {msg}", flush=True)

# ================ CARICAMENTO DATASET FAISS ====================
def load_faiss_dataset():
    global X_fast, y_fast, classes_fast
    
    print("📥 Caricamento dataset FAISS da Hugging Face...")
    
    try:
        dataset = load_dataset(DATASET_REPO, trust_remote_code=True)
        data = dataset["train"] if "train" in dataset else dataset
        
        X = []
        y = []
        class_to_idx = {}
        
        for item in data:
            features = item.get("X")
            label_idx = item.get("y")
            
            if features is None or label_idx is None:
                continue
            
            if hasattr(data.features['y'], 'names'):
                class_name = data.features['y'].names[label_idx]
            else:
                class_name = str(label_idx)
            
            if class_name not in class_to_idx:
                class_to_idx[class_name] = len(class_to_idx)
            
            X.append(np.array(features, dtype=np.float32))
            y.append(class_to_idx[class_name])
        
        if not X:
            print("❌ Nessun dato valido nel dataset")
            return False
        
        X_fast = np.vstack(X).astype(np.float32)
        y_fast = np.array(y, dtype=np.int32)
        classes_fast = {v: k for k, v in class_to_idx.items()}
        
        print(f"✅ Dataset caricato: {X_fast.shape[0]} vettori, {len(classes_fast)} classi")
        return True
        
    except Exception as e:
        print(f"❌ Errore caricamento dataset: {e}")
        return False

# ================ FUNZIONI PER FIGURE ====================
def centra_figura(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return cv2.resize(image, (DIM, DIM))
    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    crop = image[y:y+h, x:x+w]
    return cv2.resize(crop, (DIM, DIM))

def estrai_descrittori(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    circularity = 0.0
    aspect_ratio = 0.0
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        peri = cv2.arcLength(cnt, True)
        area = cv2.contourArea(cnt)
        if peri != 0:
            circularity = 4.0 * np.pi * area / (peri * peri)
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w)/h if h != 0 else 0.0

    moments = cv2.moments(thresh)
    hu = cv2.HuMoments(moments).flatten().tolist()

    h, w = img.shape[:2]
    cx, cy = w//2, h//2
    raggi = [int(min(h,w)*r) for r in (0.2, 0.4, 0.6, 0.8)]
    radiale = []
    for r in raggi:
        mask = np.zeros((h,w), np.uint8)
        cv2.circle(mask, (cx,cy), r, 255, -1)
        mean = cv2.mean(img, mask=mask)[:3]
        radiale.extend([m/255.0 for m in mean])

    spaziale = []
    quadranti = [(0,0,cx,cy), (cx,0,w,cy), (0,cy,cx,h), (cx,cy,w,h)]
    for (x1,y1,x2,y2) in quadranti:
        roi = img[y1:y2, x1:x2]
        if roi.size > 0:
            mean = cv2.mean(roi)[:3]
            spaziale.extend([m/255.0 for m in mean])

    vettore = radiale + spaziale + [circularity, aspect_ratio] + hu
    return np.array(vettore, dtype=float)

def get_features(img):
    img_centrata = centra_figura(img)
    return estrai_descrittori(img_centrata)

def predict_figure(img_crop):
    global X_fast, y_fast, classes_fast
    
    if X_fast is None or img_crop is None or img_crop.size == 0:
        return None
    
    features = get_features(img_crop)
    distances = np.linalg.norm(X_fast - features, axis=1)
    best_idx = np.argmin(distances)
    return classes_fast.get(int(y_fast[best_idx]), "errore")

def crop_safe(img, coords):
    try:
        x1, y1, x2, y2 = map(int, coords.split(","))
    except:
        return None
    h, w = img.shape[:2]
    x1 = max(0, min(w-1, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h-1, y1))
    y2 = max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2]

def upload_captcha_to_supabase(image_path, surfses, urlid, qpic, email):
    """Upload captcha matematico su Supabase Storage"""
    try:
        MATH_URL = os.environ.get("MATH_SUPABASE_URL")
        MATH_KEY = os.environ.get("MATH_SUPABASE_KEY")
        
        if not MATH_URL or not MATH_KEY:
            return False
        
        supabase_math = create_client(MATH_URL, MATH_KEY)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")[:-3]
        
        with open(image_path, "rb") as f:
            file_data = f.read()
        
        file_path = f"{timestamp}/captcha.jpg"
        supabase_math.storage.from_("math-captchas").upload(file_path, file_data,
                                                           {"content-type": "image/jpeg"})
        return True
    except Exception as e:
        return False

def salva_captcha_matematico(surfses, urlid, qpic, image_path, email, account_name):
    """Salva captcha matematico non riconosciuto"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")[:-3]
    
    if image_path and os.path.exists(image_path):
        upload_captcha_to_supabase(image_path, surfses, urlid, qpic, email)
        log(account_name, f"📤 Captcha matematico salvato")
    return True

# ================ COLLECTOR PER SINGOLO ACCOUNT ====================
class AccountSurfer:
    def __init__(self, email, account_name):
        self.email = email
        self.account_name = account_name
        self.supabase_url = os.environ.get("SUPABASE_URL")
        self.supabase_key = os.environ.get("SUPABASE_KEY")
        self.cookie_string = None
        self.session = None
    
    def get_cookie_from_supabase(self):
        try:
            supabase = create_client(self.supabase_url, self.supabase_key)
            resp = supabase.table('account_cookies')\
                .select('cookies_string')\
                .eq('email', self.email)\
                .eq('status', 'active')\
                .execute()
            
            if resp.data:
                return resp.data[0]['cookies_string']
            return None
        except Exception as e:
            log(self.account_name, f"❌ Errore lettura cookie: {e}")
            return None
    
    def run(self):
        log(self.account_name, f"🚀 Avvio surf")
        
        self.cookie_string = self.get_cookie_from_supabase()
        if not self.cookie_string:
            log(self.account_name, "❌ Cookie non trovato")
            return
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Cookie": self.cookie_string
        }
        self.session = requests.Session()
        self.session.headers.update(headers)
        
        captcha_count = 0
        
        while True:
            try:
                r = self.session.post("https://www.easyhits4u.com/surf/?ajax=1&try=1",
                                      verify=False, timeout=15)
                
                if r.status_code != 200:
                    time.sleep(5)
                    continue
                
                data = r.json()
                urlid = data.get("surfses", {}).get("urlid")
                qpic = data.get("surfses", {}).get("qpic")
                seconds = int(data.get("surfses", {}).get("seconds", 20))
                picmap = data.get("picmap")
                
                if not urlid or not qpic:
                    log(self.account_name, "⚠️ Cookie scaduto")
                    break
                
                # CAPTCHA A FIGURE (con FAISS)
                if picmap is not None and len(picmap) > 0:
                    img_data = self.session.get(f"https://www.easyhits4u.com/simg/{qpic}.jpg", 
                                                verify=False).content
                    img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
                    
                    crops = [crop_safe(img, p.get("coords", "")) for p in picmap]
                    labels = [predict_figure(c) for c in crops]
                    
                    seen = {}
                    chosen_idx = None
                    for i, label in enumerate(labels):
                        if label and label != "errore":
                            if label in seen:
                                chosen_idx = seen[label]
                                break
                            seen[label] = i
                    
                    if chosen_idx is None:
                        log(self.account_name, "❌ Nessun duplicato - FERMO")
                        return
                    
                    time.sleep(seconds)
                    word = picmap[chosen_idx]["value"]
                    resp = self.session.get(
                        f"https://www.easyhits4u.com/surf/?f=surf&urlid={urlid}&surftype=2"
                        f"&ajax=1&word={word}&screen_width=1024&screen_height=768",
                        verify=False
                    )
                    
                    if resp.json().get("warning") == "wrong_choice":
                        log(self.account_name, "❌ Wrong choice - FERMO")
                        return
                    
                    captcha_count += 1
                    log(self.account_name, f"✅ OK #{captcha_count}")
                    time.sleep(2)
                    
                else:
                    # CAPTCHA MATEMATICO - SALVA E CONTINUA
                    log(self.account_name, "🧮 Captcha matematico - SALVO")
                    
                    surfses = data.get("surfses", {})
                    img_data = self.session.get(f"https://www.easyhits4u.com/simg/{qpic}.jpg", 
                                                verify=False).content
                    temp_path = f"temp_math_{self.account_name}.jpg"
                    with open(temp_path, "wb") as f:
                        f.write(img_data)
                    
                    salva_captcha_matematico(surfses, urlid, qpic, temp_path, 
                                            self.email, self.account_name)
                    
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                    
                    time.sleep(seconds)
                    continue
                    
            except Exception as e:
                log(self.account_name, f"❌ Errore: {e}")
                time.sleep(5)
                break

# ================ MAIN ====================
def run_account(account):
    surfer = AccountSurfer(account['email'], account['name'])
    surfer.run()

def main():
    print("=" * 60)
    print("🚀 MULTI-ACCOUNT SURF COLLECTOR")
    print("   FAISS per figure + salvataggio captcha matematici")
    print("=" * 60)
    
    # Carica dataset FAISS
    if not load_faiss_dataset():
        print("❌ Impossibile caricare dataset FAISS")
        return
    
    print(f"📋 Account configurati: {len(ACCOUNTS)}")
    print(f"🔢 Massimo simultanei: {MAX_CONCURRENT_ACCOUNTS}")
    print("=" * 60)
    
    threads = []
    for account in ACCOUNTS:
        while len(threads) >= MAX_CONCURRENT_ACCOUNTS:
            threads = [t for t in threads if t.is_alive()]
            time.sleep(1)
        
        print(f"📧 Avvio: {account['email']}")
        t = threading.Thread(target=run_account, args=(account,))
        t.daemon = True
        t.start()
        threads.append(t)
        time.sleep(STAGGERED_START_DELAY)
    
    for t in threads:
        t.join()
    
    print("\n✅ Raccolta completata!")

if __name__ == "__main__":
    main()