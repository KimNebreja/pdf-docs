from flask import Flask, request, send_file, jsonify
from werkzeug.utils import secure_filename
import os
import language_tool_python
from flask_cors import CORS
import fitz  # PyMuPDF
import pdfplumber
from reportlab.pdfgen import canvas
from reportlab.lib.colors import Color
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

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file with advanced formatting preservation."""
    doc = fitz.open(pdf_path)
    text = []
    for page in doc:
        # Extract text with formatting information
        text.append(page.get_text())
    doc.close()
    return "\n".join(text)

def get_text_color(page, bbox):
    """Extracts the color of text at a specific position with advanced precision."""
    try:
        # Get text spans in the area with more detailed extraction
        spans = page.get_text("dict", clip=bbox)["blocks"]
        for block in spans:
            if "lines" in block:
                for line in block["lines"]:
                    if "spans" in line:
                        for span in line["spans"]:
                            # Check for multiple color attributes with priority
                            if "color" in span:
                                return span["color"]
                            elif "fill" in span:
                                return span["fill"]
                            elif "stroke" in span:
                                return span["stroke"]
        return None
    except Exception as e:
        print(f"Error getting text color: {str(e)}")
        return None

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

def normalize_color(color):
    """
    Normalizes color values to RGB format in range 0-1 with advanced handling.
    Returns a tuple of (r, g, b) values.
    """
    try:
        if color is None:
            return (0, 0, 0)  # Default to black
            
        # If color is already a tuple/list of RGB values
        if isinstance(color, (tuple, list)):
            if len(color) >= 3:
                # Convert to range 0-1 if needed
                r = float(color[0]) / 255 if color[0] > 1 else float(color[0])
                g = float(color[1]) / 255 if color[1] > 1 else float(color[1])
                b = float(color[2]) / 255 if color[2] > 1 else float(color[2])
                # Ensure values are in range 0-1
                r = max(0.0, min(1.0, r))
                g = max(0.0, min(1.0, g))
                b = max(0.0, min(1.0, b))
                return (r, g, b)
        
        # If color is a single value (grayscale)
        if isinstance(color, (int, float)):
            val = float(color) / 255 if color > 1 else float(color)
            val = max(0.0, min(1.0, val))
            return (val, val, val)
        
        # If color is a hex string
        if isinstance(color, str) and color.startswith('#'):
            # Convert hex to RGB
            color = color.lstrip('#')
            r = int(color[0:2], 16) / 255.0
            g = int(color[2:4], 16) / 255.0
            b = int(color[4:6], 16) / 255.0
            return (r, g, b)
        
        # If color is a CMYK value
        if isinstance(color, (tuple, list)) and len(color) == 4:
            c, m, y, k = color
            # Convert CMYK to RGB
            r = 1 - min(1, c * (1 - k) + k)
            g = 1 - min(1, m * (1 - k) + k)
            b = 1 - min(1, y * (1 - k) + k)
            return (r, g, b)
        
        # Default to black for unknown types
        return (0, 0, 0)
    except Exception as e:
        print(f"Error normalizing color {color}: {str(e)}")
        return (0, 0, 0)  # Default to black on error

def get_font_name(font_name):
    """Normalizes font names with advanced mapping."""
    font_map = {
        "helv": "Helvetica",
        "tiro": "Times-Roman",
        "helvetica-bold": "Helvetica-Bold",
        "times-bold": "Times-Bold",
        "times-italic": "Times-Italic",
        "times-bolditalic": "Times-BoldItalic",
        "courier": "Courier",
        "courier-bold": "Courier-Bold",
        "courier-italic": "Courier-Oblique",
        "courier-bolditalic": "Courier-BoldOblique",
        "symbol": "Symbol",
        "zapfdingbats": "ZapfDingbats"
    }
    
    # Check if font name is in our mapping
    if font_name.lower() in font_map:
        return font_map[font_name.lower()]
    
    # Check if font name contains any of our mapped names
    for key in font_map:
        if key.lower() in font_name.lower():
            return font_map[key]
    
    # Default to Helvetica if no match
    return "Helvetica"

def save_text_to_pdf(text, pdf_path, original_pdf_path):
    """
    Saves proofread text to a new PDF file using pdfplumber for text extraction
    and reportlab for PDF generation with advanced positioning and color handling.
    """
    try:
        # Open the original PDF with both pdfplumber and PyMuPDF
        with pdfplumber.open(original_pdf_path) as pdf:
            doc = fitz.open(original_pdf_path)
            
            # Get page dimensions from the first page
            first_page = pdf.pages[0]
            page_width = first_page.width
            page_height = first_page.height
            
            # Create a new PDF with reportlab
            c = canvas.Canvas(pdf_path, pagesize=(page_width, page_height))
            
            # Split text into paragraphs
            if isinstance(text, str):
                paragraphs = text.split('\n')
            else:
                paragraphs = text
            
            # Track current paragraph
            current_paragraph = 0
            
            # Process each page
            for page_num, (page, mupdf_page) in enumerate(zip(pdf.pages, doc)):
                print(f"\nProcessing page {page_num + 1}")
                
                # Extract text objects with their formatting
                text_objects = page.extract_words(
                    keep_blank_chars=True,
                    x_tolerance=3,
                    y_tolerance=3,
                    extra_attrs=['fontname', 'size']
                )
                
                # Process each text object
                for obj in text_objects:
                    if current_paragraph >= len(paragraphs):
                        break
                    
                    # Save canvas state
                    c.saveState()
                    
                    # Get text properties
                    x0, y0 = obj['x0'], obj['top']
                    font = get_font_name(obj.get('fontname', 'Helvetica'))
                    fontsize = float(obj.get('size', 11))
                    
                    # Get color from original PDF with improved extraction
                    bbox = fitz.Rect(x0, y0, x0 + obj['width'], y0 + obj['height'])
                    color = get_text_color(mupdf_page, bbox)
                    if color:
                        r, g, b = normalize_color(color)
                        fill_color = Color(r, g, b)
                    else:
                        fill_color = Color(0, 0, 0)  # Default to black
                    
                    # Calculate text metrics for better positioning
                    c.setFont(font, fontsize)
                    text_width = c.stringWidth(paragraphs[current_paragraph], font, fontsize)
                    text_height = fontsize * 1.2  # Approximate height
                    
                    # Calculate baseline position for better text alignment
                    baseline_offset = fontsize * 0.2  # Approximate baseline offset
                    
                    # Calculate leading (line spacing) based on font size
                    leading = fontsize * 1.5  # Standard leading is 1.5x font size
                    
                    # Adjust position for better alignment
                    y_pos = page_height - y0 - text_height/2 + baseline_offset  # Center text vertically with baseline adjustment
                    x_pos = x0  # Keep original x position
                    
                    # Draw text with improved positioning
                    text_object = c.beginText(x_pos, y_pos)
                    text_object.setFont(font, fontsize)
                    text_object.setFillColor(fill_color)
                    text_object.textLine(paragraphs[current_paragraph])
                    c.drawText(text_object)
                    current_paragraph += 1
                    
                    # Restore canvas state
                    c.restoreState()
                
                # Move to next page
                c.showPage()
            
            # Save the PDF
            c.save()
            doc.close()
            print("PDF saved successfully")
            
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
