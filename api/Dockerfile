# Usamos la imagen oficial de Python como base
FROM python:3.10

# Copiamos el código de la aplicación en el contenedor
COPY . /app

# Establecemos el directorio de trabajo
WORKDIR /app

# Instalamos las dependencias de la aplicación
RUN pip install --upgrade pip && pip install -r requirements.txt

# Exponemos el puerto que usará Flask
EXPOSE 5000

# Ejecutamos la aplicación
CMD ["python", "app.py"]

