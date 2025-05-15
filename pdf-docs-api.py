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
    
    # Map common font names with style detection
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
    
    # Check for style indicators in the original font name
    is_bold = 'bold' in font_name or 'heavy' in font_name or 'black' in font_name
    is_italic = 'italic' in font_name or 'oblique' in font_name or 'slant' in font_name
    
    # Check if font name is in our mapping
    if font_name in font_map:
        base_font = font_map[font_name]
    else:
        # Check if font name contains any of our mapped names
        for key in font_map:
            if key in font_name:
                base_font = font_map[key]
                break
        else:
            # Default to Helvetica if no match
            base_font = "Helvetica"
    
    # Apply styles to base font
    if is_bold and is_italic:
        if base_font == "Helvetica":
            return "Helvetica-BoldOblique"
        elif base_font == "Times-Roman":
            return "Times-BoldItalic"
        elif base_font == "Courier":
            return "Courier-BoldOblique"
    elif is_bold:
        if base_font == "Helvetica":
            return "Helvetica-Bold"
        elif base_font == "Times-Roman":
            return "Times-Bold"
        elif base_font == "Courier":
            return "Courier-Bold"
    elif is_italic:
        if base_font == "Helvetica":
            return "Helvetica-Oblique"
        elif base_font == "Times-Roman":
            return "Times-Italic"
        elif base_font == "Courier":
            return "Courier-Oblique"
    
    return base_font

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
    Saves proofread text to a new PDF file while preserving the exact formatting of the original PDF.
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
            
            # Create a new PDF with reportlab using exact dimensions
            c = canvas.Canvas(pdf_path, pagesize=(page_width, page_height))
            
            # Extract text with formatting
            formatted_text = extract_text_with_formatting(original_pdf_path)
            
            # Split text into paragraphs while preserving line breaks
            if isinstance(text, str):
                proofread_paragraphs = text.split('\n')
            else:
                proofread_paragraphs = text
            
            # Track current paragraph
            current_paragraph = 0
            
            # Process each page
            for page_num, (page, mupdf_page) in enumerate(zip(pdf.pages, doc)):
                logger.info(f"Processing page {page_num + 1}")
                
                # Get words with formatting for this page
                words = page.extract_words(
                    keep_blank_chars=True,
                    x_tolerance=3,
                    y_tolerance=3,
                    extra_attrs=['fontname', 'size', 'upright', 'top', 'bottom', 'x0', 'x1', 'width']
                )
                
                # Group words by line based on vertical position
                lines = defaultdict(list)
                for word in words:
                    y_pos = round(word['top'], 1)
                    lines[y_pos].append(word)
                
                # Sort lines by vertical position
                sorted_lines = sorted(lines.items(), key=lambda x: x[0])
                
                # Process each line
                for idx, (y_pos, line_words) in enumerate(sorted_lines):
                    if current_paragraph >= len(proofread_paragraphs):
                        break
                    
                    # Sort words in line by horizontal position
                    line_words.sort(key=lambda x: x['x0'])
                    
                    # Get formatting from first word in line
                    first_word = line_words[0]
                    font_name = get_font_name(first_word.get('fontname', 'Helvetica'))
                    font_size = float(first_word.get('size', 11))
                    
                    # Get text color
                    bbox = fitz.Rect(first_word['x0'], first_word['top'], 
                                   first_word['x0'] + first_word['width'], 
                                   first_word['bottom'])
                    color = get_text_color(mupdf_page, bbox)
                    r, g, b = normalize_color(color) if color else (0, 0, 0)
                    
                    # Calculate text position
                    x_pos = first_word['x0']
                    y_pos_adjusted = page_height - first_word['top'] - font_size/3
                    
                    # Calculate line width for alignment
                    line_width = line_words[-1]['x1'] - line_words[0]['x0']
                    
                    # Get text alignment from original
                    text_align = 'left'  # Default
                    if len(line_words) > 1:
                        # Check if text is centered
                        page_center = page_width / 2
                        line_center = (line_words[0]['x0'] + line_words[-1]['x1']) / 2
                        if abs(page_center - line_center) < 20:  # 20pt tolerance
                            text_align = 'center'
                        # Check if text is right-aligned
                        elif line_words[0]['x0'] > page_width * 0.7:
                            text_align = 'right'
                        # Check if text is justified
                        else:
                            # Calculate average word spacing
                            total_spacing = 0
                            word_count = len(line_words)
                            if word_count > 1:
                                for i in range(word_count - 1):
                                    spacing = line_words[i + 1]['x0'] - (line_words[i]['x0'] + line_words[i]['width'])
                                    total_spacing += spacing
                                avg_spacing = total_spacing / (word_count - 1)
                                
                                # Check if spacing is relatively uniform (justified text)
                                is_uniform = True
                                for i in range(word_count - 1):
                                    spacing = line_words[i + 1]['x0'] - (line_words[i]['x0'] + line_words[i]['width'])
                                    if abs(spacing - avg_spacing) > avg_spacing * 0.2:  # 20% tolerance
                                        is_uniform = False
                                        break
                                
                                if is_uniform and word_count > 2:  # Need at least 3 words for justified text
                                    text_align = 'justify'
                    
                    text = proofread_paragraphs[current_paragraph]
                    c.setFont(font_name, font_size)
                    c.setFillColorRGB(r, g, b)
                    
                    # Handle different text alignments
                    if text_align == 'center':
                        c.drawCentredString(x_pos + line_width/2, y_pos_adjusted, text)
                    elif text_align == 'right':
                        c.drawRightString(x_pos + line_width, y_pos_adjusted, text)
                    elif text_align == 'justify':
                        # For justified text, calculate word spacing
                        words = text.split()
                        if len(words) > 1:
                            # Calculate total width of text
                            total_text_width = 0
                            for word in words:
                                total_text_width += c.stringWidth(word, font_name, font_size)
                            # Calculate space between words
                            space_width = (line_width - total_text_width) / (len(words) - 1)
                            # Draw each word with calculated spacing
                            current_x = x_pos
                            for i, word in enumerate(words):
                                c.drawString(current_x, y_pos_adjusted, word)
                                if i < len(words) - 1:  # Don't add space after last word
                                    current_x += c.stringWidth(word, font_name, font_size) + space_width
                        else:
                            c.drawString(x_pos, y_pos_adjusted, text)  # Single word, no justification needed
                    else:
                        # Left align (default)
                        c.drawString(x_pos, y_pos_adjusted, text)
                    
                    # Add line spacing based on original
                    if current_paragraph < len(proofread_paragraphs) - 1 and idx + 1 < len(sorted_lines):
                        next_line = sorted_lines[idx + 1][1][0]
                        line_spacing = next_line['top'] - first_word['bottom']
                        if line_spacing > 0:
                            c.translate(0, -line_spacing)
                    
                    current_paragraph += 1
                
                # Add page numbers if present in original
                if page.page_number is not None:
                    c.setFont('Helvetica', 10)
                    c.drawString(page_width/2, 30, str(page.page_number))
                
                # Save the current page and start a new one
                c.showPage()
            
            # Save the PDF
            c.save()
            doc.close()
            logger.info("PDF saved successfully with original formatting")
            
    except Exception as e:
        logger.error(f"Error creating PDF: {str(e)}")
        raise e

@app.route('/convert', methods=['POST'])
def convert_and_proofread():
    """Handles PDF proofreading."""
    try:
        logger.info("Received file upload request")
        
        if 'file' not in request.files and 'text' not in request.form:
            logger.error("No file or text provided in request")
            return jsonify({"error": "No file or text provided"}), 400

        # Handle text and suggestions if provided directly
        if 'text' in request.form:
            logger.info("Processing text-based request")
            proofread_text_content = request.form['text']
            original_filename = request.form.get('filename', 'document.pdf')
            sanitized_filename = secure_filename(original_filename)
            
            # Store the original file path in a session-like dictionary
            original_pdf_path = os.path.join(UPLOAD_FOLDER, sanitized_filename)
            
            # Ensure the original file exists
            if not os.path.exists(original_pdf_path):
                # Try to find the file with "proofread_" prefix removed
                base_filename = sanitized_filename.replace("proofread_", "", 1)
                original_pdf_path = os.path.join(UPLOAD_FOLDER, base_filename)
                
                if not os.path.exists(original_pdf_path):
                    logger.error(f"Original file not found: {sanitized_filename}")
                    return jsonify({"error": f"Original file not found: {sanitized_filename}"}), 404
            
            # Get selected suggestions
            selected_suggestions = {}
            if 'selected_suggestions' in request.form:
                try:
                    selected_suggestions = dict(json.loads(request.form['selected_suggestions']))
                    # Apply selected suggestions
                    for original_word, selected_word in selected_suggestions.items():
                        proofread_text_content = proofread_text_content.replace(original_word, selected_word)
                except Exception as e:
                    logger.warning(f"Failed to parse selected suggestions: {str(e)}")

            # Generate PDF with the updated text
            output_filename = "proofread_" + sanitized_filename.replace("proofread_", "", 1)  # Avoid duplicate prefix
            output_path = os.path.join(OUTPUT_FOLDER, output_filename)
            
            # Use the original PDF as a template for formatting
            save_text_to_pdf(proofread_text_content, output_path, original_pdf_path)
            
            return jsonify({
                "download_url": "/download/" + output_filename,
                "file_name": sanitized_filename
            })

        # Handle file upload case
        file = request.files['file']
        if file.filename == '':
            logger.error("No file selected in upload")
            return jsonify({"error": "No selected file"}), 400

        filename = secure_filename(file.filename)
        pdf_path = os.path.join(UPLOAD_FOLDER, filename)
        
        # Create UPLOAD_FOLDER if it doesn't exist
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        
        # Save the uploaded file
        try:
            file.save(pdf_path)
            logger.info(f"Successfully saved uploaded file: {filename}")
        except Exception as e:
            logger.error(f"Error saving uploaded file: {str(e)}")
            return jsonify({"error": f"Error saving file: {str(e)}"}), 500

        # Extract text from PDF
        try:
            extracted_text = extract_text_from_pdf(pdf_path)
            logger.info("Successfully extracted text from PDF")
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            return jsonify({"error": f"Error extracting text: {str(e)}"}), 500

        # Proofread the text
        try:
            proofread_text_content, grammar_errors = proofread_text(extracted_text)
            logger.info("Successfully proofread text")
        except Exception as e:
            logger.error(f"Error proofreading text: {str(e)}")
            return jsonify({"error": f"Error proofreading text: {str(e)}"}), 500

        # Save proofread text back to PDF
        proofread_pdf_filename = "proofread_" + filename
        proofread_pdf_path = os.path.join(OUTPUT_FOLDER, proofread_pdf_filename)
        
        # Create OUTPUT_FOLDER if it doesn't exist
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
        
        try:
            # Save the proofread PDF using the original as template
            save_text_to_pdf(proofread_text_content, proofread_pdf_path, pdf_path)
            logger.info("Successfully saved proofread PDF")
        except Exception as e:
            logger.error(f"Error saving proofread PDF: {str(e)}")
            return jsonify({"error": f"Error saving proofread PDF: {str(e)}"}), 500

        return jsonify({
            "original_text": extracted_text,
            "proofread_text": proofread_text_content,
            "grammar_errors": grammar_errors,
            "download_url": "/download/" + proofread_pdf_filename,
            "file_name": filename  # Always return sanitized filename
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
