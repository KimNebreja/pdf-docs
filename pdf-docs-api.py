from flask import Flask, request, send_file, jsonify
from werkzeug.utils import secure_filename
import os
import requests
from pdf2docx import Converter
from docx import Document
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "/tmp"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

LANGUAGETOOL_API_URL = "https://api.languagetool.org/v2/check"

def extract_text_from_docx(docx_path):
    """Extracts text from a DOCX file."""
    doc = Document(docx_path)
    return "\n".join([para.text for para in doc.paragraphs])

def proofread_text(text):
    """Sends extracted text to LanguageTool API for grammar checking."""
    data = {"text": text, "language": "en-US"}
    response = requests.post(LANGUAGETOOL_API_URL, data=data)
    result = response.json()

    errors = []
    for match in result.get("matches", []):
        error_info = {
            "message": match["message"],
            "suggestions": [r["value"] for r in match["replacements"]],
            "offset": match["offset"],
            "length": match["length"]
        }
        errors.append(error_info)

    return errors

@app.route('/convert', methods=['POST'])
def convert_pdf_to_docx():
    """Handles PDF to DOCX conversion."""
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

@app.route('/proofread', methods=['POST'])
def proofread_docx():
    """Handles DOCX proofreading."""
    if 'file' not in request.files:
        return "No file uploaded", 400
    
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400
    
    docx_path = os.path.join(OUTPUT_FOLDER, secure_filename(file.filename))
    file.save(docx_path)

    text = extract_text_from_docx(docx_path)
    if not text.strip():
        return "The document is empty or could not be read", 400

    errors = proofread_text(text)
    
    return jsonify({"errors": errors})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000, debug=True)
