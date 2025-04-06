from flask import Flask, request, send_file, jsonify
from werkzeug.utils import secure_filename
import os
import docx
import language_tool_python
from pdf2docx import Converter
from flask_cors import CORS
import fitz  # PyMuPDF for better PDF handling
import re

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

def extract_formatted_content_from_pdf(pdf_path):
    """Extracts formatted content from PDF with styling information."""
    doc = fitz.open(pdf_path)
    formatted_content = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        blocks = page.get_text("dict")["blocks"]
        
        for block in blocks:
            if "lines" in block:
                for line in block["lines"]:
                    for span in line["spans"]:
                        # Extract text with formatting
                        text = span["text"]
                        font = span["font"]
                        size = span["size"]
                        color = span["color"]
                        
                        # Convert color to hex
                        color_hex = "#{:02x}{:02x}{:02x}".format(
                            int(color[0] * 255), 
                            int(color[1] * 255), 
                            int(color[2] * 255)
                        )
                        
                        # Create HTML with styling
                        formatted_content.append({
                            "text": text,
                            "font": font,
                            "size": size,
                            "color": color_hex,
                            "bold": "bold" in font.lower(),
                            "italic": "italic" in font.lower(),
                            "underline": "underline" in font.lower()
                        })
    
    return formatted_content

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

def save_text_to_docx(text, docx_path):
    """Saves proofread text to a new DOCX file."""
    doc = docx.Document()
    for line in text.split("\n"):
        doc.add_paragraph(line)
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
        # Extract formatted content from PDF
        formatted_content = extract_formatted_content_from_pdf(pdf_path)
        
        # Extract plain text for proofreading
        plain_text = " ".join([item["text"] for item in formatted_content])
        
        # Proofread the text
        proofread_text_content, grammar_errors = proofread_text(plain_text)
        
        # Create proofread formatted content
        proofread_formatted_content = []
        current_pos = 0
        
        for item in formatted_content:
            text = item["text"]
            text_len = len(text)
            
            # Find if this text has any corrections
            corrections = []
            for error in grammar_errors:
                if error["offset"] >= current_pos and error["offset"] < current_pos + text_len:
                    # Calculate relative position within this text
                    rel_offset = error["offset"] - current_pos
                    if rel_offset + error["length"] <= text_len:
                        corrections.append({
                            "offset": rel_offset,
                            "length": error["length"],
                            "suggestion": error["suggestions"][0] if error["suggestions"] else ""
                        })
            
            # Create a copy of the item with corrections
            proofread_item = item.copy()
            if corrections:
                # Apply corrections to the text
                corrected_text = text
                for correction in sorted(corrections, key=lambda x: x["offset"], reverse=True):
                    start = correction["offset"]
                    end = start + correction["length"]
                    corrected_text = corrected_text[:start] + correction["suggestion"] + corrected_text[end:]
                
                proofread_item["text"] = corrected_text
                proofread_item["has_correction"] = True
            else:
                proofread_item["has_correction"] = False
                
            proofread_formatted_content.append(proofread_item)
            current_pos += text_len + 1  # +1 for the space we added between items

        # Save proofread text back to DOCX
        proofread_docx_path = os.path.join(OUTPUT_FOLDER, "proofread_" + docx_filename)
        save_text_to_docx(proofread_text_content, proofread_docx_path)

        return jsonify({
            "original_content": formatted_content,
            "proofread_content": proofread_formatted_content,
            "grammar_errors": grammar_errors,
            "download_url": "/download/" + "proofread_" + docx_filename,
            "file_name": filename
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
