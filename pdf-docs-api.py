from flask import Flask, request, send_file, jsonify
from werkzeug.utils import secure_filename
import os
from pdf2docx import Converter
from flask_cors import CORS
import language_tool_python
import requests
import docx
import tempfile

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "/tmp"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

tool = language_tool_python.LanguageToolPublicAPI('en-US')  # English language

def extract_text_from_docx(docx_path):
    doc = docx.Document(docx_path)
    return "\n".join([para.text for para in doc.paragraphs])

def proofread_text(text):
    matches = tool.check(text)
    corrected_text = language_tool_python.utils.correct(text, matches)
    return corrected_text

def save_text_to_docx(text, docx_path):
    doc = docx.Document()
    for line in text.split("\n"):
        doc.add_paragraph(line)
    doc.save(docx_path)

def generate_audio(text):
    try:
        url = "https://translate.google.com/translate_tts"
        params = {
            "ie": "UTF-8",
            "q": text,
            "tl": "en",
            "client": "tw-ob",
            "ttsspeed": "1"
        }
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, params=params, headers=headers)
        
        if response.status_code == 200:
            audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3", dir=OUTPUT_FOLDER)
            audio_file.write(response.content)
            audio_file.close()
            return os.path.basename(audio_file.name)
        return None
    except Exception as e:
        print(f"Error generating audio: {str(e)}")
        return None

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

        # Extract text from DOCX
        extracted_text = extract_text_from_docx(docx_path)

        # Proofread the text
        proofread_text_content = proofread_text(extracted_text)

        # Save proofread text back to DOCX
        proofread_docx_path = os.path.join(OUTPUT_FOLDER, "proofread_" + docx_filename)
        save_text_to_docx(proofread_text_content, proofread_docx_path)

        audio_filename = generate_audio(proofread_text_content)
        response_data = {
            "original_text": extracted_text,
            "proofread_text": proofread_text_content,
            "download_url": "/download/" + "proofread_" + docx_filename,
        }
        if audio_filename:
            response_data["audio_url"] = "/audio/" + audio_filename
        return jsonify(response_data)

    except Exception as e:
        return f"Conversion error: {str(e)}", 500

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(os.path.join(OUTPUT_FOLDER, filename), as_attachment=True)

@app.route('/audio/<filename>')
def get_audio(filename):
    return send_file(
        os.path.join(OUTPUT_FOLDER, filename),
        mimetype='audio/mpeg'
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000, debug=True)
