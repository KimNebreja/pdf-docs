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
import logging
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
    try:
        doc = fitz.open(pdf_path)
        text = []
        for page in doc:
            # Extract text with formatting information
            text.append(page.get_text())
        doc.close()
        return "\n".join(text)
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {str(e)}")
        raise

def extract_text_with_formatting(pdf_path):
    """
    Extracts text with detailed formatting information using pdfplumber.
    Returns a list of dictionaries with text and formatting details.
    """
    try:
        result = []
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                # Extract text with detailed formatting
                words = page.extract_words(
                    keep_blank_chars=True,
                    x_tolerance=3,
                    y_tolerance=3,
                    extra_attrs=['fontname', 'size', 'upright']
                )
                
                # Group words into lines based on y-position
                lines = {}
                for word in words:
                    y_pos = round(word['top'], 1)  # Round to 1 decimal place for grouping
                    if y_pos not in lines:
                        lines[y_pos] = []
                    lines[y_pos].append(word)
                
                # Sort lines by y-position (top to bottom)
                sorted_lines = sorted(lines.items(), key=lambda x: x[0])
                
                # Process each line
                for y_pos, line_words in sorted_lines:
                    # Sort words in line by x-position (left to right)
                    line_words.sort(key=lambda x: x['x0'])
                    
                    # Create a line object with formatting
                    line_obj = {
                        'text': ' '.join([w['text'] for w in line_words]),
                        'words': line_words,
                        'y_pos': y_pos,
                        'page': page_num
                    }
                    result.append(line_obj)
        
        return result
    except Exception as e:
        logger.error(f"Error extracting text with formatting: {str(e)}")
        raise

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
        logger.warning(f"Error getting text color: {str(e)}")
        return None

def proofread_text(text):
    """Proofreads text using LanguageTool and returns corrected text with details."""
    try:
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
    except Exception as e:
        logger.error(f"Error proofreading text: {str(e)}")
        raise

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
        logger.warning(f"Error normalizing color {color}: {str(e)}")
        return (0, 0, 0)  # Default to black on error

def get_font_name(font_name):
    """Normalizes font names with advanced mapping and fallback mechanism."""
    if not font_name:
        return "Helvetica"  # Default font
        
    # Clean up font name
    font_name = font_name.lower().strip()
    
    # Remove common prefixes and suffixes
    font_name = re.sub(r'^[a-z]+[_-]', '', font_name)
    font_name = re.sub(r'[_-][a-z]+$', '', font_name)
    
    # Map common font names
    font_map = {
        "helv": "Helvetica",
        "helvetica": "Helvetica",
        "arial": "Helvetica",
        "tiro": "Times-Roman",
        "times": "Times-Roman",
        "timesroman": "Times-Roman",
        "times new roman": "Times-Roman",
        "timesnewroman": "Times-Roman",
        "helvetica-bold": "Helvetica-Bold",
        "helveticabold": "Helvetica-Bold",
        "arial-bold": "Helvetica-Bold",
        "arialbold": "Helvetica-Bold",
        "times-bold": "Times-Bold",
        "timesbold": "Times-Bold",
        "times-italic": "Times-Italic",
        "timesitalic": "Times-Italic",
        "times-bolditalic": "Times-BoldItalic",
        "timesbolditalic": "Times-BoldItalic",
        "courier": "Courier",
        "courier-bold": "Courier-Bold",
        "courierbold": "Courier-Bold",
        "courier-italic": "Courier-Oblique",
        "courieritalic": "Courier-Oblique",
        "courier-bolditalic": "Courier-BoldOblique",
        "courierbolditalic": "Courier-BoldOblique",
        "symbol": "Symbol",
        "zapfdingbats": "ZapfDingbats"
    }
    
    # Check if font name is in our mapping
    if font_name in font_map:
        return font_map[font_name]
    
    # Check if font name contains any of our mapped names
    for key in font_map:
        if key in font_name:
            return font_map[key]
    
    # Check for bold/italic variants
    if "bold" in font_name and "italic" in font_name:
        return "Times-BoldItalic"
    elif "bold" in font_name:
        return "Helvetica-Bold"
    elif "italic" in font_name or "oblique" in font_name:
        return "Times-Italic"
    
    # Default to Helvetica if no match
    return "Helvetica"

def register_fonts():
    """Registers common fonts with ReportLab."""
    try:
        # Register standard fonts
        pdfmetrics.registerFontFamily(
            'Helvetica',
            normal='Helvetica',
            bold='Helvetica-Bold',
            italic='Helvetica-Oblique',
            boldItalic='Helvetica-BoldOblique'
        )
        
        pdfmetrics.registerFontFamily(
            'Times-Roman',
            normal='Times-Roman',
            bold='Times-Bold',
            italic='Times-Italic',
            boldItalic='Times-BoldItalic'
        )
        
        pdfmetrics.registerFontFamily(
            'Courier',
            normal='Courier',
            bold='Courier-Bold',
            italic='Courier-Oblique',
            boldItalic='Courier-BoldOblique'
        )
        
        # Try to register additional fonts if available
        try:
            # Check if Arial font is available
            arial_path = os.path.join(os.environ.get('WINDIR', ''), 'Fonts', 'arial.ttf')
            if os.path.exists(arial_path):
                TTFont('Arial', arial_path)
                logger.info("Registered Arial font")
        except Exception as e:
            logger.warning(f"Could not register additional fonts: {str(e)}")
            
    except Exception as e:
        logger.warning(f"Error registering fonts: {str(e)}")

def save_text_to_pdf(text, pdf_path, original_pdf_path):
    """
    Saves proofread text to a new PDF file using pdfplumber for text extraction
    and reportlab for PDF generation with advanced positioning and color handling.
    """
    try:
        # Register fonts
        register_fonts()
        
        # Open the original PDF with both pdfplumber and PyMuPDF
        with pdfplumber.open(original_pdf_path) as pdf:
            doc = fitz.open(original_pdf_path)
            
            # Get page dimensions from the first page
            first_page = pdf.pages[0]
            page_width = first_page.width
            page_height = first_page.height
            
            # Create a new PDF with reportlab
            c = canvas.Canvas(pdf_path, pagesize=(page_width, page_height))
            
            # Extract text with formatting
            formatted_text = extract_text_with_formatting(original_pdf_path)
            
            # Split text into paragraphs for proofreading
            if isinstance(text, str):
                paragraphs = text.split('\n')
            else:
                paragraphs = text
            
            # Track current paragraph
            current_paragraph = 0
            
            # Process each page
            for page_num, (page, mupdf_page) in enumerate(zip(pdf.pages, doc)):
                logger.info(f"Processing page {page_num + 1}")
                
                # Get lines for this page
                page_lines = [line for line in formatted_text if line['page'] == page_num]
                
                # Process each line
                for line in page_lines:
                    if current_paragraph >= len(paragraphs):
                        break
                    
                    # Save canvas state
                    c.saveState()
                    
                    # Get line properties
                    y_pos = line['y_pos']
                    
                    # Process each word in the line
                    x_pos = line['words'][0]['x0']  # Start with the first word's x position
                    
                    # Get the first word's properties for the line
                    first_word = line['words'][0]
                    font = get_font_name(first_word.get('fontname', 'Helvetica'))
                    fontsize = float(first_word.get('size', 11))
                    
                    # Get color from original PDF
                    bbox = fitz.Rect(first_word['x0'], first_word['top'], 
                                    first_word['x0'] + first_word['width'], 
                                    first_word['top'] + first_word['height'])
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
                    y_pos_adjusted = page_height - y_pos - text_height/2 + baseline_offset
                    
                    # Draw text with improved positioning
                    text_object = c.beginText(x_pos, y_pos_adjusted)
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
            logger.info("PDF saved successfully")
            
    except Exception as e:
        logger.error(f"Error creating PDF: {str(e)}")
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
        logger.error(f"Conversion error: {str(e)}")
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
