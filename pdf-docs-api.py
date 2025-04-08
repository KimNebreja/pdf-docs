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
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
import io
import logging
import re
import json
import numpy as np
from collections import defaultdict

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

def detect_tables(page):
    """
    Detects tables in a PDF page using pdfplumber.
    Returns a list of table objects with their positions and content.
    """
    try:
        tables = page.find_tables()
        result = []
        
        for table in tables:
            # Extract table data
            table_data = table.extract()
            
            # Get table position
            bbox = table.bbox
            
            # Create table object
            table_obj = {
                'data': table_data,
                'bbox': bbox,
                'rows': len(table_data),
                'cols': len(table_data[0]) if table_data else 0
            }
            
            result.append(table_obj)
            
        return result
    except Exception as e:
        logger.warning(f"Error detecting tables: {str(e)}")
        return []

def detect_columns(page):
    """
    Detects columns in a PDF page using text positioning.
    Returns a list of column objects with their positions.
    """
    try:
        # Extract words with their positions
        words = page.extract_words(
            keep_blank_chars=True,
            x_tolerance=3,
            y_tolerance=3
        )
        
        if not words:
            return []
            
        # Sort words by y-position (top to bottom)
        words.sort(key=lambda x: x['top'])
        
        # Group words by y-position (with some tolerance)
        lines = defaultdict(list)
        for word in words:
            y_pos = round(word['top'], 1)  # Round to 1 decimal place for grouping
            lines[y_pos].append(word)
            
        # Sort words in each line by x-position (left to right)
        for y_pos in lines:
            lines[y_pos].sort(key=lambda x: x['x0'])
            
        # Detect columns based on x-position gaps
        columns = []
        for y_pos, line_words in lines.items():
            if len(line_words) < 2:
                continue
                
            # Find gaps between words
            gaps = []
            for i in range(len(line_words) - 1):
                gap = line_words[i+1]['x0'] - (line_words[i]['x0'] + line_words[i]['width'])
                if gap > 50:  # Threshold for column gap
                    gaps.append((i, gap))
                    
            # If we found gaps, create column objects
            if gaps:
                # Sort gaps by position
                gaps.sort(key=lambda x: x[0])
                
                # Create column objects
                for i, (gap_pos, gap_size) in enumerate(gaps):
                    # Calculate column boundaries
                    if i == 0:
                        x0 = line_words[0]['x0']
                    else:
                        x0 = line_words[gaps[i-1][0] + 1]['x0']
                        
                    if i == len(gaps) - 1:
                        x1 = line_words[-1]['x0'] + line_words[-1]['width']
                    else:
                        x1 = line_words[gap_pos]['x0'] + line_words[gap_pos]['width']
                        
                    # Create column object
                    column_obj = {
                        'x0': x0,
                        'x1': x1,
                        'y0': y_pos,
                        'y1': y_pos + line_words[0]['height'],
                        'width': x1 - x0
                    }
                    
                    columns.append(column_obj)
                    
        return columns
    except Exception as e:
        logger.warning(f"Error detecting columns: {str(e)}")
        return []

def detect_headers_footers(pdf_path):
    """
    Detects headers and footers in a PDF document.
    Returns a dictionary with header and footer information.
    """
    try:
        headers = []
        footers = []
        
        with pdfplumber.open(pdf_path) as pdf:
            # Get page dimensions
            first_page = pdf.pages[0]
            page_height = first_page.height
            
            # Define header and footer regions (top and bottom 10% of page)
            header_region = (0, 0, first_page.width, page_height * 0.1)
            footer_region = (0, page_height * 0.9, first_page.width, page_height)
            
            # Process first few pages to detect consistent headers/footers
            sample_pages = min(5, len(pdf.pages))
            
            for i in range(sample_pages):
                page = pdf.pages[i]
                
                # Extract text from header region
                header_text = page.crop(header_region).extract_text()
                if header_text and header_text.strip():
                    headers.append({
                        'text': header_text.strip(),
                        'page': i,
                        'region': header_region
                    })
                
                # Extract text from footer region
                footer_text = page.crop(footer_region).extract_text()
                if footer_text and footer_text.strip():
                    footers.append({
                        'text': footer_text.strip(),
                        'page': i,
                        'region': footer_region
                    })
        
        # Find consistent headers and footers across pages
        consistent_headers = find_consistent_text(headers)
        consistent_footers = find_consistent_text(footers)
        
        return {
            'headers': consistent_headers,
            'footers': consistent_footers
        }
    except Exception as e:
        logger.warning(f"Error detecting headers and footers: {str(e)}")
        return {'headers': [], 'footers': []}

def find_consistent_text(text_blocks):
    """
    Finds text blocks that appear consistently across pages.
    """
    if not text_blocks:
        return []
        
    # Group text blocks by similarity
    groups = []
    for block in text_blocks:
        matched = False
        for group in groups:
            # Check if this block is similar to any in the group
            for existing in group:
                if text_similarity(block['text'], existing['text']) > 0.8:
                    group.append(block)
                    matched = True
                    break
            if matched:
                break
        if not matched:
            groups.append([block])
    
    # Find the most common group
    if not groups:
        return []
        
    most_common = max(groups, key=len)
    return most_common

def text_similarity(text1, text2):
    """
    Calculates similarity between two text strings.
    Returns a value between 0 and 1.
    """
    if not text1 or not text2:
        return 0
        
    # Simple character-based similarity
    set1 = set(text1.lower())
    set2 = set(text2.lower())
    
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    if union == 0:
        return 0
        
    return intersection / union

def detect_lists(lines):
    """
    Detects lists (bullet points, numbered lists) in a list of lines.
    Returns a list of list objects.
    """
    try:
        lists = []
        current_list = []
        
        # Regular expressions for list detection
        bullet_pattern = re.compile(r'^[\s]*[•\-\*\+]\s+')
        number_pattern = re.compile(r'^[\s]*\d+[\.\)]\s+')
        
        for i, line in enumerate(lines):
            text = line['text'].strip()
            
            # Check if this line is a list item
            is_bullet = bool(bullet_pattern.match(text))
            is_numbered = bool(number_pattern.match(text))
            
            if is_bullet or is_numbered:
                # If this is the first item in a list
                if not current_list:
                    current_list = {
                        'items': [],
                        'type': 'bullet' if is_bullet else 'numbered',
                        'start_line': i,
                        'indentation': len(text) - len(text.lstrip())
                    }
                
                # Add item to current list
                current_list['items'].append({
                    'text': text,
                    'line_index': i,
                    'indentation': len(text) - len(text.lstrip())
                })
            else:
                # If we have a current list and this line is not a list item
                if current_list:
                    # Check if this line is a continuation of the previous list item
                    if (i > 0 and 
                        lines[i-1]['text'].strip() and 
                        line['indentation'] > current_list['indentation']):
                        # This is a continuation of the previous list item
                        current_list['items'][-1]['text'] += ' ' + text
                    else:
                        # End of the list
                        current_list['end_line'] = i - 1
                        lists.append(current_list)
                        current_list = []
        
        # Add the last list if there is one
        if current_list:
            current_list['end_line'] = len(lines) - 1
            lists.append(current_list)
            
        return lists
    except Exception as e:
        logger.warning(f"Error detecting lists: {str(e)}")
        return []

def detect_paragraphs(lines):
    """
    Detects paragraphs in a list of lines based on spacing and formatting.
    Returns a list of paragraph objects.
    """
    try:
        paragraphs = []
        current_paragraph = []
        
        for i, line in enumerate(lines):
            # Check if this is a new paragraph
            is_new_paragraph = False
            
            # Check for empty line (double line break)
            if i > 0 and line['y_pos'] - lines[i-1]['y_pos'] > lines[i-1]['words'][0]['height'] * 2:
                is_new_paragraph = True
                
            # Check for indentation
            if i > 0 and line['words'][0]['x0'] > lines[i-1]['words'][0]['x0'] + 20:
                is_new_paragraph = True
                
            # Check for different formatting (font, size, color)
            if i > 0 and len(current_paragraph) > 0:
                prev_word = current_paragraph[-1]['words'][0]
                curr_word = line['words'][0]
                
                if (prev_word.get('fontname') != curr_word.get('fontname') or
                    abs(prev_word.get('size', 0) - curr_word.get('size', 0)) > 2):
                    is_new_paragraph = True
                    
            # If this is a new paragraph, save the current one and start a new one
            if is_new_paragraph and current_paragraph:
                paragraphs.append({
                    'lines': current_paragraph,
                    'text': ' '.join([l['text'] for l in current_paragraph])
                })
                current_paragraph = []
                
            # Add this line to the current paragraph
            current_paragraph.append(line)
            
        # Add the last paragraph
        if current_paragraph:
            paragraphs.append({
                'lines': current_paragraph,
                'text': ' '.join([l['text'] for l in current_paragraph])
            })
            
        return paragraphs
    except Exception as e:
        logger.warning(f"Error detecting paragraphs: {str(e)}")
        return [{'lines': lines, 'text': ' '.join([l['text'] for l in lines])}]

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
        "zapfdingbats": "ZapfDingbats",
        "calibri": "Helvetica",
        "verdana": "Helvetica",
        "tahoma": "Helvetica",
        "georgia": "Times-Roman",
        "garamond": "Times-Roman",
        "bookman": "Times-Roman",
        "palatino": "Times-Roman",
        "goudy": "Times-Roman",
        "century": "Times-Roman",
        "avantgarde": "Helvetica",
        "futura": "Helvetica",
        "optima": "Helvetica",
        "gill": "Helvetica",
        "franklin": "Helvetica",
        "lucida": "Courier",
        "consolas": "Courier",
        "monaco": "Courier",
        "andale": "Courier"
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
                
            # Check for other common fonts
            font_paths = {
                'Calibri': 'calibri.ttf',
                'Verdana': 'verdana.ttf',
                'Tahoma': 'tahoma.ttf',
                'Georgia': 'georgia.ttf',
                'Times New Roman': 'times.ttf',
                'Courier New': 'cour.ttf'
            }
            
            for font_name, font_file in font_paths.items():
                font_path = os.path.join(os.environ.get('WINDIR', ''), 'Fonts', font_file)
                if os.path.exists(font_path):
                    TTFont(font_name, font_path)
                    logger.info(f"Registered {font_name} font")
                    
        except Exception as e:
            logger.warning(f"Could not register additional fonts: {str(e)}")
            
    except Exception as e:
        logger.warning(f"Error registering fonts: {str(e)}")

def create_table(c, table_data, x, y, width, height, font, fontsize, fill_color):
    """
    Creates a table in the PDF using ReportLab.
    """
    try:
        # Calculate cell dimensions
        rows = len(table_data)
        cols = len(table_data[0]) if table_data else 0
        
        if rows == 0 or cols == 0:
            return
            
        cell_width = width / cols
        cell_height = height / rows
        
        # Create table style
        style = [
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTNAME', (0, 0), (-1, -1), font),
            ('FONTSIZE', (0, 0), (-1, -1), fontsize),
            ('TEXTCOLOR', (0, 0), (-1, -1), fill_color),
            ('GRID', (0, 0), (-1, -1), 0.5, Color(0.5, 0.5, 0.5)),
            ('BACKGROUND', (0, 0), (-1, 0), Color(0.9, 0.9, 0.9)),  # Header row
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
            ('RIGHTPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 3),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ]
        
        # Create table
        table = Table(table_data, colWidths=[cell_width] * cols, rowHeights=[cell_height] * rows)
        table.setStyle(TableStyle(style))
        
        # Draw table
        table.wrapOn(c, width, height)
        table.drawOn(c, x, y)
        
    except Exception as e:
        logger.warning(f"Error creating table: {str(e)}")

def extract_text_with_formatting(pdf_path):
    """
    Extracts text with detailed formatting information using pdfplumber.
    Returns a list of dictionaries with text and formatting details.
    """
    try:
        result = []
        
        # Detect headers and footers
        header_footer_info = detect_headers_footers(pdf_path)
        
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                # Detect tables
                tables = detect_tables(page)
                
                # Detect columns
                columns = detect_columns(page)
                
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
                        'page': page_num,
                        'is_table': False,
                        'is_column': False,
                        'is_header': False,
                        'is_footer': False
                    }
                    
                    # Check if this line is part of a table
                    for table in tables:
                        if (table['bbox'][1] <= y_pos <= table['bbox'][3] and
                            table['bbox'][0] <= line_words[0]['x0'] <= table['bbox'][2]):
                            line_obj['is_table'] = True
                            line_obj['table_data'] = table['data']
                            break
                            
                    # Check if this line is part of a column
                    for column in columns:
                        if (column['y0'] <= y_pos <= column['y1'] and
                            column['x0'] <= line_words[0]['x0'] <= column['x1']):
                            line_obj['is_column'] = True
                            line_obj['column'] = column
                            break
                            
                    # Check if this line is a header
                    for header in header_footer_info['headers']:
                        if (header['region'][1] <= y_pos <= header['region'][3] and
                            header['region'][0] <= line_words[0]['x0'] <= header['region'][2]):
                            line_obj['is_header'] = True
                            break
                            
                    # Check if this line is a footer
                    for footer in header_footer_info['footers']:
                        if (footer['region'][1] <= y_pos <= footer['region'][3] and
                            footer['region'][0] <= line_words[0]['x0'] <= footer['region'][2]):
                            line_obj['is_footer'] = True
                            break
                            
                    result.append(line_obj)
        
        # Detect paragraphs
        paragraphs = detect_paragraphs(result)
        
        # Detect lists
        lists = detect_lists(result)
        
        # Add paragraph and list information to the result
        for line in result:
            # Find which paragraph this line belongs to
            for i, paragraph in enumerate(paragraphs):
                if line in paragraph['lines']:
                    line['paragraph_index'] = i
                    break
                    
            # Find which list this line belongs to
            for i, list_obj in enumerate(lists):
                for item in list_obj['items']:
                    if line['line_index'] == item['line_index']:
                        line['list_index'] = i
                        line['list_type'] = list_obj['type']
                        break
                        
        return result
    except Exception as e:
        logger.error(f"Error extracting text with formatting: {str(e)}")
        raise

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
            
            # Detect paragraphs
            paragraphs = detect_paragraphs(formatted_text)
            
            # Detect lists
            lists = detect_lists(formatted_text)
            
            # Detect headers and footers
            header_footer_info = detect_headers_footers(original_pdf_path)
            
            # Split text into paragraphs for proofreading
            if isinstance(text, str):
                proofread_paragraphs = text.split('\n')
            else:
                proofread_paragraphs = text
            
            # Track current paragraph
            current_paragraph = 0
            
            # Process each page
            for page_num, (page, mupdf_page) in enumerate(zip(pdf.pages, doc)):
                logger.info(f"Processing page {page_num + 1}")
                
                # Get lines for this page
                page_lines = [line for line in formatted_text if line['page'] == page_num]
                
                # Get paragraphs for this page
                page_paragraphs = [p for p in paragraphs if p['lines'][0]['page'] == page_num]
                
                # Get lists for this page
                page_lists = [l for l in lists if l['items'][0]['line_index'] < len(page_lines) and 
                             page_lines[l['items'][0]['line_index']]['page'] == page_num]
                
                # Process each paragraph
                for paragraph in page_paragraphs:
                    if current_paragraph >= len(proofread_paragraphs):
                        break
                        
                    # Check if this is a table
                    if paragraph['lines'][0].get('is_table', False):
                        # Get table data
                        table_data = paragraph['lines'][0].get('table_data', [])
                        
                        # Get table position
                        first_line = paragraph['lines'][0]
                        last_line = paragraph['lines'][-1]
                        
                        x0 = first_line['words'][0]['x0']
                        y0 = first_line['y_pos']
                        x1 = last_line['words'][-1]['x0'] + last_line['words'][-1]['width']
                        y1 = last_line['y_pos'] + last_line['words'][-1]['height']
                        
                        # Get font and color
                        font = get_font_name(first_line['words'][0].get('fontname', 'Helvetica'))
                        fontsize = float(first_line['words'][0].get('size', 11))
                        
                        # Get color
                        bbox = fitz.Rect(x0, y0, x1, y1)
                        color = get_text_color(mupdf_page, bbox)
                        if color:
                            r, g, b = normalize_color(color)
                            fill_color = Color(r, g, b)
                        else:
                            fill_color = Color(0, 0, 0)  # Default to black
                            
                        # Create table
                        create_table(c, table_data, x0, page_height - y1, x1 - x0, y1 - y0, 
                                    font, fontsize, fill_color)
                                    
                        # Update current paragraph
                        current_paragraph += 1
                        continue
                        
                    # Check if this is a list
                    is_list = False
                    list_type = None
                    for list_obj in page_lists:
                        if paragraph['lines'][0] in [page_lines[item['line_index']] for item in list_obj['items']]:
                            is_list = True
                            list_type = list_obj['type']
                            break
                            
                    # Process each line in the paragraph
                    for line in paragraph['lines']:
                        if current_paragraph >= len(proofread_paragraphs):
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
                        text_width = c.stringWidth(proofread_paragraphs[current_paragraph], font, fontsize)
                        text_height = fontsize * 1.2  # Approximate height
                        
                        # Calculate baseline position for better text alignment
                        baseline_offset = fontsize * 0.2  # Approximate baseline offset
                        
                        # Calculate leading (line spacing) based on font size
                        leading = fontsize * 1.5  # Standard leading is 1.5x font size
                        
                        # Adjust position for better alignment
                        y_pos_adjusted = page_height - y_pos - text_height/2 + baseline_offset
                        
                        # Handle lists
                        if is_list:
                            # Add bullet or number
                            if list_type == 'bullet':
                                bullet = '•'
                                bullet_width = c.stringWidth(bullet, font, fontsize)
                                c.setFillColor(fill_color)
                                c.drawString(x_pos - bullet_width - 10, y_pos_adjusted, bullet)
                                x_pos += 10  # Indent the text
                            elif list_type == 'numbered':
                                # Find the list item number
                                for list_obj in page_lists:
                                    for i, item in enumerate(list_obj['items']):
                                        if line == page_lines[item['line_index']]:
                                            number = f"{i+1}."
                                            number_width = c.stringWidth(number, font, fontsize)
                                            c.setFillColor(fill_color)
                                            c.drawString(x_pos - number_width - 10, y_pos_adjusted, number)
                                            x_pos += 10  # Indent the text
                                            break
                        
                        # Draw text with improved positioning
                        text_object = c.beginText(x_pos, y_pos_adjusted)
                        text_object.setFont(font, fontsize)
                        text_object.setFillColor(fill_color)
                        text_object.textLine(proofread_paragraphs[current_paragraph])
                        c.drawText(text_object)
                        current_paragraph += 1
                        
                        # Restore canvas state
                        c.restoreState()
                
                # Add headers and footers
                for header in header_footer_info['headers']:
                    if header['page'] == page_num:
                        c.setFont('Helvetica', 10)
                        c.setFillColor(Color(0.5, 0.5, 0.5))  # Gray color for headers
                        c.drawString(50, page_height - 30, header['text'])
                        
                for footer in header_footer_info['footers']:
                    if footer['page'] == page_num:
                        c.setFont('Helvetica', 10)
                        c.setFillColor(Color(0.5, 0.5, 0.5))  # Gray color for footers
                        c.drawString(50, 30, footer['text'])
                
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
