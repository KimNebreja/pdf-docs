from flask import Flask, request, send_file, jsonify
from werkzeug.utils import secure_filename
import os
import docx
from docx.shared import Pt, RGBColor
from docx.enum.text import WD_PARAGRAPH_ALIGNMENT
import language_tool_python
from pdf2docx import Converter
from flask_cors import CORS
import traceback
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

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
    try:
        doc = docx.Document(docx_path)
        formatted_paragraphs = []
        
        for para in doc.paragraphs:
            if para.text.strip():
                # Get paragraph formatting
                para_format = {
                    'text': para.text,
                    'alignment': para.alignment,
                    'style': para.style.name if para.style else 'Normal',
                    'runs': []
                }
                
                # Get run-level formatting
                for run in para.runs:
                    try:
                        run_format = {
                            'text': run.text,
                            'bold': run.bold,
                            'italic': run.italic,
                            'underline': run.underline,
                            'font_size': run.font.size.pt if run.font.size else None,
                            'font_name': run.font.name,
                            'color': run.font.color.rgb if run.font.color and run.font.color.rgb else None
                        }
                        para_format['runs'].append(run_format)
                    except Exception as e:
                        logger.error(f"Error processing run: {str(e)}")
                        # Add a simplified run format as fallback
                        para_format['runs'].append({'text': run.text})
                
                formatted_paragraphs.append(para_format)
        
        return formatted_paragraphs
    except Exception as e:
        logger.error(f"Error extracting text from DOCX: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def proofread_text(formatted_paragraphs):
    """Proofreads text while preserving formatting."""
    try:
        # Combine all text for proofreading
        full_text = "\n".join([p['text'] for p in formatted_paragraphs])
        matches = tool.check(full_text)
        corrected_text = language_tool_python.utils.correct(full_text, matches)
        
        # Split corrected text back into paragraphs
        corrected_paragraphs = corrected_text.split("\n")
        
        # Preserve formatting while applying corrections
        corrected_formatted = []
        
        for i, para in enumerate(formatted_paragraphs):
            if i < len(corrected_paragraphs):
                corrected_para = {
                    'text': corrected_paragraphs[i],
                    'alignment': para['alignment'],
                    'style': para['style'],
                    'runs': para['runs']  # Preserve original run formatting
                }
                corrected_formatted.append(corrected_para)
            else:
                # If we have more original paragraphs than corrected ones, 
                # just copy the original formatting
                corrected_formatted.append(para)
        
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
    except Exception as e:
        logger.error(f"Error proofreading text: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def save_text_to_docx(formatted_paragraphs, docx_path):
    """Saves formatted text to a new DOCX file."""
    try:
        doc = docx.Document()
        
        for para_format in formatted_paragraphs:
            paragraph = doc.add_paragraph()
            try:
                paragraph.alignment = para_format['alignment']
                paragraph.style = para_format['style']
            except Exception as e:
                logger.warning(f"Error setting paragraph format: {str(e)}")
            
            # Apply run-level formatting
            for run_format in para_format['runs']:
                try:
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
                except Exception as e:
                    logger.warning(f"Error applying run format: {str(e)}")
                    # Add text without formatting as fallback
                    paragraph.add_run(run_format['text'])
        
        doc.save(docx_path)
    except Exception as e:
        logger.error(f"Error saving DOCX: {str(e)}")
        logger.error(traceback.format_exc())
        raise

@app.route('/convert', methods=['POST'])
def convert_pdf_to_docx():
    """Handles PDF-to-DOCX conversion and proofreading."""
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        filename = secure_filename(file.filename)
        pdf_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(pdf_path)
        
        logger.info(f"File saved to {pdf_path}")

        docx_filename = filename.rsplit('.', 1)[0] + '.docx'
        docx_path = os.path.join(OUTPUT_FOLDER, docx_filename)

        try:
            # Convert PDF to DOCX with formatting
            logger.info("Starting PDF to DOCX conversion")
            cv = Converter(pdf_path)
            cv.convert(docx_path, start=0, end=None)
            cv.close()
            logger.info(f"PDF converted to DOCX at {docx_path}")

            if not os.path.exists(docx_path):
                return jsonify({"error": "DOCX file was not created"}), 500

            # Extract formatted text from DOCX
            logger.info("Extracting formatted text from DOCX")
            formatted_paragraphs = extract_text_from_docx(docx_path)
            logger.info(f"Extracted {len(formatted_paragraphs)} paragraphs")

            # Proofread the text while preserving formatting
            logger.info("Proofreading text")
            proofread_formatted, grammar_errors = proofread_text(formatted_paragraphs)
            logger.info(f"Found {len(grammar_errors)} grammar errors")

            # Save proofread text back to DOCX with formatting
            proofread_docx_path = os.path.join(OUTPUT_FOLDER, "proofread_" + docx_filename)
            logger.info(f"Saving proofread text to {proofread_docx_path}")
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
            logger.error(f"Error during conversion process: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({"error": f"Conversion error: {str(e)}"}), 500
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

@app.route('/download/<filename>')
def download_file(filename):
    """Handles file download."""
    try:
        file_path = os.path.join(OUTPUT_FOLDER, filename)
        if os.path.exists(file_path):
            return send_file(file_path, as_attachment=True)
        return jsonify({"error": "File not found"}), 404
    except Exception as e:
        logger.error(f"Error downloading file: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Download error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000, debug=True)
