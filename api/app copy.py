import os
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Ruta para almacenar las imágenes
app.config['UPLOAD_FOLDER'] = 'api/static/imagenes'
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Lista para almacenar los productos ingresados
productos = []

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        nombre_producto = request.form['nombre']
        descripcion = request.form['descripcion']
        imagen = request.files['imagen']

        # Verificar si se subió una imagen
        if imagen.filename:
            # Asegurarse de que el nombre del archivo sea seguro
            filename = secure_filename(imagen.filename)
            # Guardar la imagen en la ruta configurada
            imagen.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        else:
            filename = None

        # Almacenar los datos en la lista de productos
        productos.append({
            'nombre': nombre_producto,
            'descripcion': descripcion,
            'imagen': filename
        })

        # Redireccionar a la página que muestra el último producto ingresado
        return redirect(url_for('mostrar_ultimo_producto'))

    return render_template('index.html')

@app.route('/productos')
def mostrar_ultimo_producto():
    # Obtener el último producto ingresado
    ultimo_producto = productos[-1] if productos else None
    # Mostrar la página con el último producto ingresado
    return render_template('productos.html', producto=ultimo_producto)

@app.route('/historial')
def mostrar_historial():
    # Mostrar la página con el historial de productos ingresados
    return render_template('historial.html', productos=productos)

if __name__ == '__main__':
    app.run(debug=True)


