# Utilisation d'une image Python légère
FROM python:3.10-slim

# Définition du répertoire de travail dans le conteneur
WORKDIR /app

# Installation des dépendances système nécessaires
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copie du fichier de dépendances
COPY requirements.txt .

# Installation des bibliothèques Python
RUN pip install --no-cache-dir -r requirements.txt

# Copie de l'intégralité du code source
COPY . .

# Exposition du port par défaut de Streamlit
EXPOSE 8501

# Commande pour lancer l'application au démarrage du conteneur
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]