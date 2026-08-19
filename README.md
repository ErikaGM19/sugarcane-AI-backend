# Sugarcane Disease Classification Backend

Este proyecto es el backend para una aplicación de clasificación de enfermedades en hojas de caña de azúcar utilizando Inteligencia Artificial. Está construido con **FastAPI** y utiliza un modelo de aprendizaje profundo **ResNet50** entrenado con TensorFlow/Keras para detectar diferentes tipos de enfermedades a partir de imágenes.

## Características Principales

*   **API RESTful:** Desarrollada con FastAPI para un alto rendimiento y fácil integración.
*   **Modelo de IA:** Integración de un modelo ResNet50 para la clasificación de 10 clases de condiciones en las hojas (incluyendo hojas sanas y diversas enfermedades como Roya, Carbón, Mosaico, etc.).
*   **Procesamiento de Imágenes:** Las imágenes enviadas a la API son preprocesadas y adaptadas para cumplir con los requerimientos del modelo.
*   **Modo Simulación:** Si el archivo de pesos del modelo no se encuentra disponible, la API responde con predicciones simuladas para facilitar las pruebas de integración.

## Requisitos Previos

Para ejecutar este proyecto, necesitas tener instalado:

*   Python 3.8 o superior.
*   `pip` (gestor de paquetes de Python).
*   *Nota:* El proyecto usa SQLite por defecto, por lo que no necesitas instalar PostgreSQL ni otros motores de bases de datos.

## Instalación

Sigue estos pasos para configurar el entorno y descargar las dependencias:

1.  Navega a la carpeta del proyecto (si no estás ya en ella):
    ```bash
    cd d:/proyecto/sugarcane-AI-backend
    ```

2.  (Opcional pero recomendado) Crea y activa un entorno virtual:
    ```bash
    python -m venv tesis_app
    # En Windows:
    tesis_app\Scripts\activate
    # En macOS/Linux:
    source tesis_app/bin/activate
    ```

3.  Instala las dependencias necesarias mediante el archivo `requirements.txt`:
    ```bash
    pip install -r requirements.txt
    ```

4.  Configura la base de datos local (SQLite) y genera el cliente de Prisma:
    ```bash
    prisma db push
    prisma generate
    ```

## Ejecución Local

Para poner en marcha el servidor de desarrollo localmente, utiliza el siguiente comando desde la raíz del proyecto:

```bash
uvicorn app.main:app --reload --port 8000
```

*   `--reload`: Reinicia el servidor automáticamente cuando detecta cambios en el código.
*   `--port 8000`: Especifica el puerto en el que correrá el servidor.

Una vez que el servidor esté en marcha, puedes acceder a la **Documentación Interactiva (Swagger UI)** para probar los endpoints en tu navegador:
`http://localhost:8000/docs`
