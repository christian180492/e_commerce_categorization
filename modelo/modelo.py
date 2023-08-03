import redis
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import RandomizedSearchCV

# Cargar el modelo al iniciar el script
ruta_modelo = '/app/modelos/best_moc.pkl'  # Ruta del modelo dentro del contenedor
modelo = None

def cargar_modelo(ruta_modelo):
    global modelo  # Utiliza la variable global del modelo
    with open(ruta_modelo, 'rb') as file:
        modelo = pickle.load(file)
    print(f"Modelo cargado exitosamente.\n Estos son sus mejores parámetros: ¡¡¡{modelo.best_params_}!!!")

def procesar_datos(nombre, descripcion, imagen):
    # Verificar si el modelo ya ha sido cargado
    if modelo is None:
        print("El modelo aún no ha sido cargado. Cargando...")
        cargar_modelo(ruta_modelo)

    # Lógica para procesar los datos recibidos y obtener las categorías usando el modelo
    # Ejemplo de uso del modelo:
    # categorias = modelo.predict([datos_a_procesar])

    categorias = ['Categoria 1', 'Categoria 2', 'Categoria 3']  # Ejemplo: categorías predichas por el modelo
    return categorias

if __name__ == '__main__':
    # Conectar a Redis
    redis_client = redis.StrictRedis(host='redis', port=6379, db=0)

    # Esperar a que se envíen los datos desde Flask a Redis
    while not redis_client.exists('producto'):
        pass

    # Obtener los datos desde Redis
    datos = redis_client.hgetall('producto')
    nombre = datos.get(b'nombre').decode('utf-8')
    descripcion = datos.get(b'descripcion').decode('utf-8')
    imagen = datos.get(b'imagen').decode('utf-8') if datos.get(b'imagen') else None

    # Procesar los datos y obtener las categorías
    categorias = procesar_datos(nombre, descripcion, imagen)

    # Guardar las categorías en Redis
    redis_client.set('categorias', ','.join(categorias))

