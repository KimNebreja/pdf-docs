from flask import Flask, request, send_file, jsonify
from werkzeug.utils import secure_filename
import os
import language_tool_python
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

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file."""
    doc = fitz.open(pdf_path)
    text = []
    for page in doc:
        text.append(page.get_text())
    doc.close()
    return "\n".join(text)

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
        # Create a new PDF
        doc = fitz.open()
        
        # Get formatting from original PDF
        orig_doc = fitz.open(original_pdf_path)
        
        # For each page in original
        for page_num in range(len(orig_doc)):
            # Create new page with same dimensions
            orig_page = orig_doc[page_num]
            page = doc.new_page(width=orig_page.rect.width, height=orig_page.rect.height)
            
            # Get original formatting
            blocks = orig_page.get_text("dict")["blocks"]
            
            # Split the proofread text into words if not already split
            if isinstance(text, str):
                proofread_words = text.split()
            else:
                proofread_words = text
            word_index = 0
            
            # Process each text block
            for block in blocks:
                if "lines" not in block or word_index >= len(proofread_words):
                    continue
                
                # Calculate total words in this block
                block_word_count = sum(
                    len(span["text"].split())
                    for line in block["lines"]
                    for span in line["spans"]
                )
                
                # Get the text for this block
                if word_index + block_word_count <= len(proofread_words):
                    block_text = " ".join(proofread_words[word_index:word_index + block_word_count])
                    
                    # Get the first span's formatting as reference
                    first_span = block["lines"][0]["spans"][0]
                    x0 = first_span["origin"][0]
                    y0 = first_span["origin"][1]
                    
                    try:
                        # Insert text with original properties
                        page.insert_text(
                            (x0, y0),
                            block_text,
                            fontname=first_span.get("font", "helv"),
                            fontsize=first_span.get("size", 11),
                            color=first_span.get("color", (0, 0, 0))
                        )
                    except Exception as e:
                        print(f"Warning: Could not insert text with original font, using fallback. Error: {str(e)}")
                        # Fallback to basic font if original font fails
                        page.insert_text(
                            (x0, y0),
                            block_text,
                            fontname="helv",
                            fontsize=first_span.get("size", 11),
                            color=first_span.get("color", (0, 0, 0))
                        )
                    
                    word_index += block_word_count
        
        # Save the modified PDF with maximum quality
        doc.save(pdf_path, garbage=4, deflate=True, clean=True)
        doc.close()
        orig_doc.close()
            
    except Exception as e:
        print(f"Error creating PDF: {str(e)}")
        raise e

@app.route('/convert', methods=['POST'])
def convert_and_proofread():
    """Handles PDF proofreading."""
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(file.filename)
    pdf_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(pdf_path)

    try:
        # Extract text directly from PDF
        extracted_text = extract_text_from_pdf(pdf_path)

        # Proofread the text
        proofread_text_content, grammar_errors = proofread_text(extracted_text)

        # Save proofread text back to PDF
        proofread_pdf_filename = "proofread_" + filename
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
