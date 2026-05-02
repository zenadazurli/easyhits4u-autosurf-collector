# config.py

# Account da utilizzare (solo email, le password non servono perché usiamo i cookie)
# Devono corrispondere ESATTAMENTE alle email per cui hai caricato i cookie su Supabase
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

# Numero massimo di account simultanei
MAX_CONCURRENT_ACCOUNTS = 5

# Ritardo tra l'avvio di un account e l'altro (secondi)
STAGGERED_START_DELAY = 3

# Dataset FAISS (su Hugging Face)
DATASET_REPO = "zenadazurli/easyhits4u-dataset"