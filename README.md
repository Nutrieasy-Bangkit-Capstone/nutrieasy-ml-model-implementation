# Fruit and Vegetable Detection API

This is a Flask-based API for classifying images of fruits and vegetables using a pre-trained TensorFlow model.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Model and Labels](#model-and-labels)
- [Acknowledgements](#acknowledgements)

## Installation

### Prerequisites

- Python 3.10 or higher
- Flask
- TensorFlow
- Pillow

### Steps

1. Clone the repository:
    ```sh
    git clone <repository-url>
    cd <repository-directory>
    ```

2. Create a virtual environment and activate it:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

4. Ensure you have the TensorFlow model (`fruit_and_vegetable_detection.h5`) and the class labels file (`metadata_arr.json`) in the `models` directory.

## Usage

1. Run the Flask application:
    ```sh
    python app.py
    ```

2. The application will be accessible at `http://localhost:5000`.

## API Endpoints

### GET /

- **Description:** Health check endpoint to ensure the API is running.
- **Response:**
    ```json
    {
        "status": "ok"
    }
    ```

### GET /health

- **Description:** Returns the health status and uptime of the API.
- **Response:**
    ```json
    {
        "status": "UP",
        "uptime": "<uptime>"
    }
    ```

### POST /classify

- **Description:** Classifies an uploaded image of a fruit or vegetable.
- **Request:**
    - Content-Type: multipart/form-data
    - File field name: `file`
- **Response:**
    ```json
    {
        "result": "<predicted_class>"
    }
    ```
- **Error Responses:**
    - If no file is part of the request:
        ```json
        {
            "error": "No file part"
        }
        ```
    - If no file is selected:
        ```json
        {
            "error": "No selected file"
        }
        ```
    - If an error occurs during processing:
        ```json
        {
            "error": "<error_message>"
        }
        ```

## Model and Labels

- **Model:** The TensorFlow model file (`fruit_and_vegetable_detection.h5`) should be placed in the `models` directory.
- **Labels:** The class labels file (`metadata_arr.json`) should also be placed in the `models` directory. This file contains the mapping of class indices to class names.

## Acknowledgements

This project uses the following libraries and frameworks:
- [Flask](https://flask.palletsprojects.com/)
- [TensorFlow](https://www.tensorflow.org/)
- [Pillow](https://python-pillow.org/)
