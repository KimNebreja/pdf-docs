from flask import Flask, request, send_file, jsonify
from werkzeug.utils import secure_filename
import os
import docx
import language_tool_python
from pdf2docx import Converter
from flask_cors import CORS
from PyPDF2 import PdfReader, PdfWriter
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
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
    This is a simplified approach - for a production system, you might want to use
    more sophisticated PDF manipulation libraries.
    """
    try:
        # Create a new PDF with the proofread text
        packet = io.BytesIO()
        can = canvas.Canvas(packet, pagesize=letter)
        width, height = letter
        
        # Set font and size
        can.setFont("Helvetica", 12)
        
        # Split text into lines and write to PDF
        lines = text.split('\n')
        y = height - 50  # Start from top with margin
        
        for line in lines:
            if y < 50:  # If we're near the bottom, start a new page
                can.showPage()
                can.setFont("Helvetica", 12)
                y = height - 50
            
            can.drawString(50, y, line)
            y -= 15  # Line spacing
        
        can.save()
        
        # Move to the beginning of the BytesIO buffer
        packet.seek(0)
        
        # Create a new PDF with the proofread text
        new_pdf = PdfReader(packet)
        
        # Get the original PDF
        original_pdf = PdfReader(original_pdf_path)
        
        # Create a PDF writer object
        output = PdfWriter()
        
        # Add the first page from the new PDF (with our text)
        output.add_page(new_pdf.pages[0])
        
        # If the original PDF has more than one page, add them
        if len(original_pdf.pages) > 1:
            for i in range(1, len(original_pdf.pages)):
                output.add_page(original_pdf.pages[i])
        
        # Write the output to a file
        with open(pdf_path, "wb") as output_file:
            output.write(output_file)
            
    except Exception as e:
        print(f"Error creating PDF: {str(e)}")
        # Fallback to a simple PDF creation if the above method fails
        c = canvas.Canvas(pdf_path, pagesize=letter)
        c.setFont("Helvetica", 12)
        
        lines = text.split('\n')
        y = 750  # Start from top
        
        for line in lines:
            if y < 50:  # If we're near the bottom, start a new page
                c.showPage()
                c.setFont("Helvetica", 12)
                y = 750
            
            c.drawString(50, y, line)
            y -= 15  # Line spacing
        
        c.save()

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
