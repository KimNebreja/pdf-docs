from flask import Flask, request, send_file, jsonify
from werkzeug.utils import secure_filename
import os
import docx
from docx.shared import Pt, RGBColor
from docx.enum.text import WD_PARAGRAPH_ALIGNMENT
import language_tool_python
from pdf2docx import Converter
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "/tmp"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Using local LanguageTool instance for more accuracy
tool = language_tool_python.LanguageToolPublicAPI('en-US')  # Uses the online API

def extract_text_from_docx(docx_path):
    """Extracts text and formatting from a DOCX file."""
    doc = docx.Document(docx_path)
    formatted_paragraphs = []
    
    for para in doc.paragraphs:
        if para.text.strip():
            # Get paragraph formatting
            para_format = {
                'text': para.text,
                'alignment': para.alignment,
                'style': para.style.name,
                'runs': []
            }
            
            # Get run-level formatting
            for run in para.runs:
                run_format = {
                    'text': run.text,
                    'bold': run.bold,
                    'italic': run.italic,
                    'underline': run.underline,
                    'font_size': run.font.size.pt if run.font.size else None,
                    'font_name': run.font.name,
                    'color': run.font.color.rgb if run.font.color else None
                }
                para_format['runs'].append(run_format)
            
            formatted_paragraphs.append(para_format)
    
    return formatted_paragraphs

def proofread_text(formatted_paragraphs):
    """Proofreads text while preserving formatting."""
    # Combine all text for proofreading
    full_text = "\n".join([p['text'] for p in formatted_paragraphs])
    matches = tool.check(full_text)
    corrected_text = language_tool_python.utils.correct(full_text, matches)
    
    # Split corrected text back into paragraphs
    corrected_paragraphs = corrected_text.split("\n")
    
    # Preserve formatting while applying corrections
    corrected_formatted = []
    current_pos = 0
    
    for i, para in enumerate(formatted_paragraphs):
        if i < len(corrected_paragraphs):
            corrected_para = {
                'text': corrected_paragraphs[i],
                'alignment': para['alignment'],
                'style': para['style'],
                'runs': para['runs']  # Preserve original run formatting
            }
            corrected_formatted.append(corrected_para)
    
    # Collect grammar errors
    errors = []
    for match in matches:
        errors.append({
            "message": match.message,
            "suggestions": match.replacements,
            "offset": match.offset,
            "length": match.errorLength
        })
    
    return corrected_formatted, errors

def save_text_to_docx(formatted_paragraphs, docx_path):
    """Saves formatted text to a new DOCX file."""
    doc = docx.Document()
    
    for para_format in formatted_paragraphs:
        paragraph = doc.add_paragraph()
        paragraph.alignment = para_format['alignment']
        paragraph.style = para_format['style']
        
        # Apply run-level formatting
        for run_format in para_format['runs']:
            run = paragraph.add_run(run_format['text'])
            run.bold = run_format['bold']
            run.italic = run_format['italic']
            run.underline = run_format['underline']
            
            if run_format['font_size']:
                run.font.size = Pt(run_format['font_size'])
            if run_format['font_name']:
                run.font.name = run_format['font_name']
            if run_format['color']:
                run.font.color.rgb = run_format['color']
    
    doc.save(docx_path)

@app.route('/convert', methods=['POST'])
def convert_pdf_to_docx():
    """Handles PDF-to-DOCX conversion and proofreading."""
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(file.filename)
    pdf_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(pdf_path)

    docx_filename = filename.rsplit('.', 1)[0] + '.docx'
    docx_path = os.path.join(OUTPUT_FOLDER, docx_filename)

    try:
        # Convert PDF to DOCX with formatting
        cv = Converter(pdf_path)
        cv.convert(docx_path, start=0, end=None)
        cv.close()

        if not os.path.exists(docx_path):
            return jsonify({"error": "DOCX file was not created"}), 500

        # Extract formatted text from DOCX
        formatted_paragraphs = extract_text_from_docx(docx_path)

        # Proofread the text while preserving formatting
        proofread_formatted, grammar_errors = proofread_text(formatted_paragraphs)

        # Save proofread text back to DOCX with formatting
        proofread_docx_path = os.path.join(OUTPUT_FOLDER, "proofread_" + docx_filename)
        save_text_to_docx(proofread_formatted, proofread_docx_path)

        # Extract plain text for display
        original_text = "\n".join([p['text'] for p in formatted_paragraphs])
        proofread_text = "\n".join([p['text'] for p in proofread_formatted])

        return jsonify({
            "original_text": original_text,
            "proofread_text": proofread_text,
            "grammar_errors": grammar_errors,
            "download_url": "/download/" + "proofread_" + docx_filename
        })

    except Exception as e:
        return jsonify({"error": f"Conversion error: {str(e)}"}), 500

@app.route('/download/<filename>')
def download_file(filename):
    """Handles file download."""
    file_path = os.path.join(OUTPUT_FOLDER, filename)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    return jsonify({"error": "File not found"}), 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000, debug=True)
