import os
import redis
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Connect to Redis
redis_client = redis.StrictRedis(host='redis', port=6379, db=0)

# Path to store the images
app.config['UPLOAD_FOLDER'] = 'api/static/imagenes'
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# List to store the entered products
productos = []

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        nombre_producto = request.form['nombre']
        descripcion = request.form['descripcion']
        imagen = request.files['imagen']

        # Check if an image was uploaded
        if imagen.filename:
            # Make sure the file name is safe
            filename = secure_filename(imagen.filename)
            # Save the image in the configured path
            imagen.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        else:
            filename = None

        # Store the data in Redis
        producto = {'nombre': nombre_producto, 'descripcion': descripcion}
        if filename is not None:
            producto['imagen'] = filename

        redis_client.hmset('producto', producto)

        # Wait for the response from Redis
        while not redis_client.exists('categorias'):
            pass

        # Get the list of categories from Redis
        categorias = redis_client.get('categorias').decode('utf-8').split(',')

        # Store the data in the product list
        productos.append({
            'nombre': nombre_producto,
            'descripcion': descripcion,
            'imagen': filename,
            'categorias': categorias
        })

        # Redirect to the page that shows the last product entered
        return redirect(url_for('mostrar_producto'))

    return render_template('index.html')

@app.route('/productos')
def mostrar_producto():
    # Get the last product entered
    ultimo_producto = productos[-1] if productos else None
    # Show the page with the last product entered
    return render_template('productos.html', producto=ultimo_producto)

@app.route('/historial')
def mostrar_historial():
    # Show the page with the history of entered products
    return render_template('historial.html', productos=productos)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
