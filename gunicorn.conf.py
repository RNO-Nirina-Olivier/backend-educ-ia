# Configuration Gunicorn pour Render + FastAPI
# Compatible avec les instances gratuites (512MB RAM)

import multiprocessing

# ========================================
# CONNEXION (OBLIGATOIRE pour Render)
# ========================================
bind = "0.0.0.0:10000"
backlog = 2048

# ========================================
# WORKERS (léger pour Render gratuit)
# ========================================
workers = 1  # 1 seul worker pour éviter OOM
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000
max_requests = 1000  # Restart après 1000 req
max_requests_jitter = 100

# ========================================
# TIMEOUTS (essentiels pour Render)
# ========================================
timeout = 120
keepalive = 2

# ========================================
# LOGGING (pour debug)
# ========================================
loglevel = "info"
accesslog = "-"
errorlog = "-"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s"'

# ========================================
# PROCESSUS (optimisé Render)
# ========================================
preload_app = True
daemon = False
pidfile = None
user = None
group = None
tmp_upload_dir = None

# ========================================
# PERF (léger)
# ========================================
worker_tmp_dir = None
max_client_wbody_size = 104857600  # 100MB
