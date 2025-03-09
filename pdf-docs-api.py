from flask import Flask, request, send_file
from werkzeug.utils import secure_filename
import os
from pdf2docx import Converter

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "/tmp"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route('/convert', methods=['POST'])
def convert_pdf_to_docx():
    if 'file' not in request.files:
        return "No file part", 400
    
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400
    
    filename = secure_filename(file.filename)
    pdf_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(pdf_path)
    
    docx_filename = filename.rsplit('.', 1)[0] + '.docx'
    docx_path = os.path.join(OUTPUT_FOLDER, docx_filename)

    try:
        cv = Converter(pdf_path)
        cv.convert(docx_path)
        cv.close()
        
        if not os.path.exists(docx_path):
            return "DOCX file was not created", 500

        return send_file(docx_path, as_attachment=True)
    
    except Exception as e:
        return f"Conversion error: {str(e)}", 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
