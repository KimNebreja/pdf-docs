from flask import Flask, request, send_file, jsonify
from werkzeug.utils import secure_filename
import os
import docx
import language_tool_python
from pdf2docx import Converter
import subprocess
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
        if not para.text.strip():
            continue
            
        # Extract formatting for each paragraph
        para_format = {
            'text': para.text,
            'style': para.style.name if para.style else 'Normal',
            'alignment': para.alignment,
            'runs': []
        }
        
        # Extract formatting for each run (text segment with consistent formatting)
        for run in para.runs:
            if run.text.strip():
                run_format = {
                    'text': run.text,
                    'bold': run.bold,
                    'italic': run.italic,
                    'underline': run.underline,
                    'font_size': run.font.size.pt if run.font.size else None,
                    'font_name': run.font.name if run.font.name else None,
                    'color': str(run.font.color.rgb) if run.font.color and run.font.color.rgb else None
                }
                para_format['runs'].append(run_format)
        
        formatted_paragraphs.append(para_format)
    
    return formatted_paragraphs

def proofread_text(text):
    """Proofreads text using LanguageTool and returns corrected text with details."""
    # Join all paragraph texts for proofreading
    full_text = "\n".join([para['text'] for para in text])
    matches = tool.check(full_text)
    corrected_text = language_tool_python.utils.correct(full_text, matches)

    # Collect detailed grammar mistakes
    errors = []
    for match in matches:
        errors.append({
            "message": match.message,
            "suggestions": match.replacements,
            "offset": match.offset,
            "length": match.errorLength
        })

    # Split the corrected text back into paragraphs
    corrected_paragraphs = corrected_text.split("\n")
    
    # Create a new formatted text with corrections
    formatted_corrected = []
    for i, para in enumerate(text):
        if i < len(corrected_paragraphs):
            corrected_para = para.copy()
            corrected_para['text'] = corrected_paragraphs[i]
            
            # Update the text in each run to maintain formatting
            if corrected_para.get('runs') and len(corrected_para['runs']) > 0:
                # If we have runs, we need to update them to match the corrected text
                # This is a simplified approach - for more complex cases, you might need
                # a more sophisticated algorithm to map corrections to specific runs
                
                # If there's only one run, simply update its text
                if len(corrected_para['runs']) == 1:
                    corrected_para['runs'][0]['text'] = corrected_paragraphs[i]
                else:
                    # For multiple runs, we need a more sophisticated approach
                    # Here we'll use a simple approach where we update the first run
                    # and remove the rest, but in a real implementation, you might want
                    # to use a more sophisticated algorithm to map corrections to specific runs
                    corrected_para['runs'] = [{
                        'text': corrected_paragraphs[i],
                        'bold': corrected_para['runs'][0].get('bold', False),
                        'italic': corrected_para['runs'][0].get('italic', False),
                        'underline': corrected_para['runs'][0].get('underline', False),
                        'font_size': corrected_para['runs'][0].get('font_size'),
                        'font_name': corrected_para['runs'][0].get('font_name'),
                        'color': corrected_para['runs'][0].get('color')
                    }]
            
            formatted_corrected.append(corrected_para)
        else:
            formatted_corrected.append(para)
    
    return formatted_corrected, errors

def save_text_to_docx(formatted_text, docx_path):
    """Saves proofread text with formatting to a new DOCX file."""
    doc = docx.Document()
    
    for para_data in formatted_text:
        p = doc.add_paragraph(para_data['text'])
        
        # Apply paragraph-level formatting
        if para_data.get('style'):
            p.style = para_data['style']
        if para_data.get('alignment'):
            p.alignment = para_data['alignment']
        
        # Apply run-level formatting
        if para_data.get('runs'):
            # Clear the paragraph text to add formatted runs
            p.clear()
            for run_data in para_data['runs']:
                run = p.add_run(run_data['text'])
                run.bold = run_data.get('bold', False)
                run.italic = run_data.get('italic', False)
                run.underline = run_data.get('underline', False)
                
                # Apply font properties if available
                if run_data.get('font_size'):
                    run.font.size = docx.shared.Pt(run_data['font_size'])
                if run_data.get('font_name'):
                    run.font.name = run_data['font_name']
                if run_data.get('color'):
                    run.font.color.rgb = docx.shared.RGBColor.from_string(run_data['color'])
    
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
        # Convert PDF to DOCX
        cv = Converter(pdf_path)
        cv.convert(docx_path, start=0, end=None)  # Ensure full document conversion
        cv.close()

        if not os.path.exists(docx_path):
            return jsonify({"error": "DOCX file was not created"}), 500

        # Extract text and formatting from DOCX
        formatted_text = extract_text_from_docx(docx_path)

        # Proofread the text
        proofread_formatted_text, grammar_errors = proofread_text(formatted_text)

        # Save proofread text back to DOCX
        proofread_docx_path = os.path.join(OUTPUT_FOLDER, "proofread_" + docx_filename)
        save_text_to_docx(proofread_formatted_text, proofread_docx_path)

        return jsonify({
            "original_text": formatted_text,
            "proofread_text": proofread_formatted_text,
            "grammar_errors": grammar_errors,  # Provide detailed grammar corrections
            "download_url": "/download/" + "proofread_" + docx_filename
        })

    except Exception as e:
        return jsonify({"error": f"Conversion error: {str(e)}"}), 500

@app.route('/download/<filename>')
def download_file(filename):
    """Handles file download."""
    # First check in the OUTPUT_FOLDER
    file_path = os.path.join(OUTPUT_FOLDER, filename)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    
    # If not found in OUTPUT_FOLDER, check in UPLOAD_FOLDER
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    
    return jsonify({"error": "File not found"}), 404

@app.route('/convert-to-pdf/<filename>')
def convert_to_pdf(filename):
    """Converts a DOCX file to PDF and returns the PDF file."""
    # Check if the file exists in the OUTPUT_FOLDER
    docx_path = os.path.join(OUTPUT_FOLDER, filename)
    if not os.path.exists(docx_path):
        return jsonify({"error": "DOCX file not found"}), 404
    
    # Create a PDF filename
    pdf_filename = filename.rsplit('.', 1)[0] + '.pdf'
    pdf_path = os.path.join(OUTPUT_FOLDER, pdf_filename)
    
    try:
        # Convert DOCX to PDF using pandoc
        subprocess.run(['pandoc', docx_path, '-o', pdf_path], check=True)
        
        # Return the PDF file
        return send_file(pdf_path, as_attachment=True)
    except subprocess.CalledProcessError as e:
        return jsonify({"error": f"Conversion error: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000, debug=True)
