# Usamos una imagen base de Python
FROM python:3.10

# Establecemos el directorio de trabajo
WORKDIR /app

# Copiamos el código de la aplicación en el contenedor
COPY . /app/

# Instalamos las dependencias de la aplicación
RUN pip install -r requirements.txt

# Exponemos el puerto que usará el servicio modelo (opcional, si lo necesitas)
# EXPOSE 8000

# Ejecutamos el archivo modelo.py
CMD ["python", "modelo.py"]
