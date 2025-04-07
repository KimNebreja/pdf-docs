from flask import Flask, request, send_file, jsonify
from werkzeug.utils import secure_filename
import os
import docx
import language_tool_python
from pdf2docx import Converter
from flask_cors import CORS
import fitz  # PyMuPDF
import io

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "/tmp"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Using local LanguageTool instance for more accuracy
tool = language_tool_python.LanguageToolPublicAPI('en-US')  # Uses the online API

def extract_text_from_docx(docx_path):
    """Extracts text from a DOCX file."""
    doc = docx.Document(docx_path)
    return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])

def proofread_text(text):
    """Proofreads text using LanguageTool and returns corrected text with details."""
    matches = tool.check(text)
    corrected_text = language_tool_python.utils.correct(text, matches)

    # Collect detailed grammar mistakes
    errors = []
    for match in matches:
        errors.append({
            "message": match.message,
            "suggestions": match.replacements,
            "offset": match.offset,
            "length": match.errorLength
        })

    return corrected_text, errors

def save_text_to_pdf(text, pdf_path, original_pdf_path):
    """
    Saves proofread text to a new PDF file while preserving the original PDF formatting.
    Uses PyMuPDF to maintain the exact positioning and formatting of the original PDF.
    """
    try:
        # Open the original PDF
        doc = fitz.open(original_pdf_path)
        
        # Create a copy of the original PDF
        doc.save(pdf_path)
        doc.close()
        
        # Reopen the new PDF for editing
        doc = fitz.open(pdf_path)
        
        # Get the text blocks from the original PDF to maintain formatting
        original_blocks = []
        for page in doc:
            blocks = page.get_text("dict")["blocks"]
            original_blocks.extend([(block, page.number) for block in blocks if "lines" in block])
        
        # Split the proofread text into words
        proofread_words = text.split()
        word_index = 0
        
        # For each text block in the original PDF
        for block, page_num in original_blocks:
            if word_index >= len(proofread_words):
                break
                
            page = doc[page_num]
            
            # Get the block's position and formatting
            x0 = block["bbox"][0]
            y0 = block["bbox"][1]
            
            # Create a new text block with proofread content
            block_words = []
            for line in block["lines"]:
                for span in line["spans"]:
                    span_word_count = len(span["text"].split())
                    # Get the corresponding number of words from proofread text
                    if word_index + span_word_count <= len(proofread_words):
                        new_text = " ".join(proofread_words[word_index:word_index + span_word_count])
                        block_words.append(new_text)
                        word_index += span_word_count
            
            if block_words:
                # Create a new text insertion
                text_to_insert = " ".join(block_words)
                # Use the original position and font
                page.insert_text(
                    (x0, y0),
                    text_to_insert,
                    fontname="helv",  # Use Helvetica as default
                    fontsize=11,  # Default size, adjust if needed
                    color=(0, 0, 0)  # Black color
                )
        
        # Save the modified PDF
        doc.save(pdf_path, garbage=4, deflate=True)
        doc.close()
            
    except Exception as e:
        print(f"Error creating PDF: {str(e)}")
        raise e

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

        # Extract text from DOCX
        extracted_text = extract_text_from_docx(docx_path)

        # Proofread the text
        proofread_text_content, grammar_errors = proofread_text(extracted_text)

        # Save proofread text back to PDF instead of DOCX
        proofread_pdf_filename = "proofread_" + filename.rsplit('.', 1)[0] + '.pdf'
        proofread_pdf_path = os.path.join(OUTPUT_FOLDER, proofread_pdf_filename)
        save_text_to_pdf(proofread_text_content, proofread_pdf_path, pdf_path)

        return jsonify({
            "original_text": extracted_text,
            "proofread_text": proofread_text_content,
            "grammar_errors": grammar_errors,  # Provide detailed grammar corrections
            "download_url": "/download/" + proofread_pdf_filename
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
