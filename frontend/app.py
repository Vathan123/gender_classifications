from flask import Flask, render_template, request
import requests

app = Flask(__name__)

FASTAPI_URL = "http://127.0.0.1:8000/predict"  # change to your Render URL when deployed

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        file = request.files['file']
        response = requests.post(FASTAPI_URL, files={'file': file})
        if response.status_code == 200:
            result = response.json()
        else:
            result = {"predicted_class": "Error", "confidence": 0}
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
