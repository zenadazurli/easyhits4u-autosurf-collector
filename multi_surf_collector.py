#!/usr/bin/env python3
# multi_surf_collector.py - Legge i cookie da Supabase

import os
import time
import threading
import requests
import numpy as np
import cv2
from datetime import datetime
from supabase import create_client
from datasets import load_dataset
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ==================== CONFIG ====================
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

# Account da usare (solo email, le password non servono)
ACCOUNTS = [
    {'email': 'dangiopiera+filippomesherda@gmail.com', 'name': 'acc1'},
    {'email': 'piersilviogarrini+linadarini@gmail.com', 'name': 'acc2'},
    {'email': 'sandrominori50+ucecelu@gmail.com', 'name': 'acc3'},
    {'email': 'sandrominori50+ulonomizano@gmail.com', 'name': 'acc4'},
    {'email': 'sandrominori50+uzakabechi@gmail.com', 'name': 'acc5'},
    {'email': 'sandrominori50+uisnrnafwttvvceer@gmail.com', 'name': 'acc6'},
    {'email': 'sandrominori50+ulimugekalochinefo@gmail.com', 'name': 'acc7'},
    {'email': 'sandrominori50+ukaxixigalilo@gmail.com', 'name': 'acc8'},
    {'email': 'sandrominori50+usaparmzogg@gmail.com', 'name': 'acc9'},
    {'email': 'sandrominori50+umifomixirmncgg@gmail.com', 'name': 'acc10'},
    {'email': 'sandrominori50+ukukamulurmgaka@gmail.com', 'name': 'acc11'},
    {'email': 'sandrominori50+udizageku@gmail.com', 'name': 'acc12'},
    {'email': 'sandrominori50+uzalifolusageneka@gmail.com', 'name': 'acc13'},
    {'email': 'sandrominori50+ulugarecexisa@gmail.com', 'name': 'acc14'},
]

MAX_CONCURRENT = 5
DIM = 64
REQUEST_TIMEOUT = 15

# ==================== CARICAMENTO DATASET ====================
def load_faiss_dataset():
    print("📥 Caricamento dataset FAISS...")
    dataset = load_dataset("zenadazurli/easyhits4u-dataset", trust_remote_code=True)
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
    
    X_fast = np.vstack(X).astype(np.float32)
    y_fast = np.array(y, dtype=np.int32)
    classes_fast = {v: k for k, v in class_to_idx.items()}
    
    print(f"✅ Dataset caricato: {X_fast.shape[0]} vettori, {len(classes_fast)} classi")
    return X_fast, y_fast, classes_fast

# ==================== LEGGI COOKIE DA SUPABASE ====================
def get_cookie_from_supabase(email):
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        resp = supabase.table('account_cookies')\
            .select('cookies_string')\
            .eq('email', email)\
            .eq('status', 'active')\
            .execute()
        
        if resp.data:
            return resp.data[0]['cookies_string']
        return None
    except Exception as e:
        print(f"❌ Errore lettura cookie: {e}")
        return None

# ==================== FUNZIONI FIGURE ====================
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

def predict_figure(img_crop, X_fast, y_fast, classes_fast):
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

# ==================== SURF ACCOUNT ====================
def surf_account(account, X_fast, y_fast, classes_fast):
    email = account['email']
    account_name = account['name']
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}][{account_name}] 🚀 Avvio surf")
    
    # Leggi cookie da Supabase
    cookie_str = get_cookie_from_supabase(email)
    
    if not cookie_str:
        print(f"[{datetime.now().strftime('%H:%M:%S')}][{account_name}] ❌ Cookie non trovato")
        return
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Cookie": cookie_str
    }
    session = requests.Session()
    session.headers.update(headers)
    
    captcha_count = 0
    
    while True:
        try:
            r = session.post("https://www.easyhits4u.com/surf/?ajax=1&try=1", verify=False, timeout=REQUEST_TIMEOUT)
            if r.status_code != 200:
                time.sleep(5)
                continue
            
            data = r.json()
            urlid = data.get("surfses", {}).get("urlid")
            qpic = data.get("surfses", {}).get("qpic")
            seconds = int(data.get("surfses", {}).get("seconds", 20))
            picmap = data.get("picmap")
            
            if not urlid or not qpic:
                print(f"[{datetime.now().strftime('%H:%M:%S')}][{account_name}] ⚠️ Cookie scaduto")
                break
            
            # Captcha figure
            if picmap and len(picmap) > 0:
                img_data = session.get(f"https://www.easyhits4u.com/simg/{qpic}.jpg", verify=False).content
                img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
                
                crops = [crop_safe(img, p.get("coords", "")) for p in picmap]
                labels = [predict_figure(c, X_fast, y_fast, classes_fast) for c in crops]
                
                seen = {}
                chosen_idx = None
                for i, label in enumerate(labels):
                    if label and label != "errore":
                        if label in seen:
                            chosen_idx = seen[label]
                            break
                        seen[label] = i
                
                if chosen_idx is None:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}][{account_name}] ❌ Nessun duplicato")
                    break
                
                time.sleep(seconds)
                word = picmap[chosen_idx]["value"]
                resp = session.get(
                    f"https://www.easyhits4u.com/surf/?f=surf&urlid={urlid}&surftype=2"
                    f"&ajax=1&word={word}&screen_width=1024&screen_height=768",
                    verify=False
                )
                
                if resp.json().get("warning") == "wrong_choice":
                    print(f"[{datetime.now().strftime('%H:%M:%S')}][{account_name}] ❌ Wrong choice")
                    break
                
                captcha_count += 1
                print(f"[{datetime.now().strftime('%H:%M:%S')}][{account_name}] ✅ OK #{captcha_count}")
                time.sleep(2)
            
            # Captcha matematico - salva e continua
            else:
                print(f"[{datetime.now().strftime('%H:%M:%S')}][{account_name}] 🧮 Captcha matematico - SALVO")
                # Qui va la logica di salvataggio su Supabase Storage
                time.sleep(seconds)
                continue
                
        except Exception as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}][{account_name}] ❌ Errore: {e}")
            time.sleep(5)
            break

# ==================== MAIN ====================
def main():
    print("="*60)
    print("🚀 MULTI-ACCOUNT SURF COLLECTOR (Supabase)")
    print("="*60)
    
    # Carica dataset FAISS
    X_fast, y_fast, classes_fast = load_faiss_dataset()
    
    # Avvia thread (max 5 alla volta)
    threads = []
    for account in ACCOUNTS:
        while len(threads) >= MAX_CONCURRENT:
            threads = [t for t in threads if t.is_alive()]
            time.sleep(1)
        
        print(f"📧 Avvio: {account['email']}")
        t = threading.Thread(target=surf_account, args=(account, X_fast, y_fast, classes_fast))
        t.start()
        threads.append(t)
        time.sleep(2)
    
    for t in threads:
        t.join()
    
    print("✅ Raccolta completata!")

if __name__ == "__main__":
    main()
