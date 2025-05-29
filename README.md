# Pasos para ejecutar

## 1. En su carpeta donde esté el .py cree un entorno virtual para ejecutar los scripts de deepface (y deepface fine tuneado) y otro para la cnn (preferible, no obligatorio):
```bash
    python -m venv <nombre de su entorno virtual (no use tildes ni espacios de preferencia, dejelo suave)>
```
## 2. Cambie al entorno virtual respectivo (si aplicó el paso 1):
```bash
    <nombre de su entorno virtual>\Scripts\activate
```

## 3. Instale:
```bash
pip install -r <archivo de requisitos.txt>
```
Para el entorno de cnn instale el requirements.txt para los de deepface instale DFrequirements.txt
## 4. Ejecute:
```bash
    cd demos
    python <script de su eleccion.py>
```
dependiendo de cual de los 3 scripts python(dentro de la carpeta demos) quiere ejecutar
