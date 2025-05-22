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
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Frame, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
import io
import logging
import re
import json
import numpy as np
from collections import defaultdict
import difflib
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT, TA_CENTER, TA_RIGHT
from PIL import Image, ImageDraw, ImageFont

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
            ('ALIGN', (0, 0), (-1, -1), 'JUSTIFY'),
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
                    extra_attrs=['fontname', 'size', 'upright', 'top', 'bottom', 'x0', 'x1', 'y0', 'y1', 'width', 'height', 'upright', 'text']
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
                        'is_footer': False,
                        'line_height': max(w['height'] for w in line_words) if line_words else 0,
                        'line_width': sum(w['width'] for w in line_words) if line_words else 0,
                        'line_spacing': 0,  # Will be calculated later
                        'alignment': 'left',  # Default alignment
                        'indentation': min(w['x0'] for w in line_words) if line_words else 0,
                        'right_margin': page.width - max(w['x1'] for w in line_words) if line_words else 0
                    }
                    
                    # Calculate line spacing
                    line_obj['line_spacing'] = 0 # Default to 0
                    if len(sorted_lines) > 1:
                        current_idx = sorted_lines.index((y_pos, line_words))
                        if current_idx > 0:
                            # Calculate space from the bottom of the previous line to the top of the current line
                            prev_line_words = sorted_lines[current_idx - 1][1]
                            if prev_line_words:
                                prev_bottom = max(w['bottom'] for w in prev_line_words)
                                current_top = min(w['top'] for w in line_words)
                                line_obj['line_spacing'] = current_top - prev_bottom
                    
                    # Determine alignment
                    alignment = 'left' # Default alignment
                    if line_words:
                        page_width = page.width
                        text_width = sum(w['width'] for w in line_words) + sum(line_words[i+1]['x0'] - (line_words[i]['x0'] + line_words[i]['width']) for i in range(len(line_words)-1))
                        line_x0 = min(w['x0'] for w in line_words)
                        line_x1 = max(w['x1'] for w in line_words)
                        
                        # Simple check for centered text: left margin approx equals right margin
                        left_margin = line_x0
                        right_margin = page_width - line_x1
                        
                        if abs(left_margin - right_margin) < 10: # Tolerance for centering
                            alignment = 'center'
                        # Check for right alignment: right margin is small and left margin is large
                        elif right_margin < 10 and left_margin > page_width * 0.1: # Tolerance and threshold for right alignment
                             alignment = 'right'
                        # Check for justified text (if text fills most of the line width)
                        elif text_width > page_width * 0.8: # If text fills more than 80% of the line width
                             alignment = 'justify'
                        
                    line_obj['alignment'] = alignment
                    
                    # Calculate indentation and right margin relative to the page content area
                    # Assuming content area roughly within margins (e.g., 72 points)
                    content_x0 = 72 # Example left margin
                    content_x1 = page_width - 72 # Example right margin
                    
                    line_obj['indentation'] = line_x0 - content_x0
                    line_obj['right_margin'] = content_x1 - line_x1

                    # Check if this line is part of a table
                    for table in tables:
                        if (table['bbox'][1] <= y_pos <= table['bbox'][3] and
                            table['bbox'][0] <= line_words[0]['x0'] <= table['bbox'][2]):
                            line_obj['is_table'] = True
                            line_obj['table_data'] = table['data']
                            line_obj['table_position'] = {
                                'row': int((y_pos - table['bbox'][1]) / (table['bbox'][3] - table['bbox'][1]) * table['rows']),
                                'col': int((line_words[0]['x0'] - table['bbox'][0]) / (table['bbox'][2] - table['bbox'][0]) * table['cols'])
                            }
                            break
                            
                    # Check if this line is part of a column
                    for column in columns:
                        if (column['y0'] <= y_pos <= column['y1'] and
                            column['x0'] <= line_words[0]['x0'] <= column['x1']):
                            line_obj['is_column'] = True
                            line_obj['column'] = column
                            line_obj['column_position'] = {
                                'x0': column['x0'],
                                'x1': column['x1'],
                                'width': column['width']
                            }
                            break
                            
                    # Check if this line is a header
                    for header in header_footer_info['headers']:
                        if (header['region'][1] <= y_pos <= header['region'][3] and
                            header['region'][0] <= line_words[0]['x0'] <= header['region'][2]):
                            line_obj['is_header'] = True
                            line_obj['header_text'] = header['text']
                            break
                            
                    # Check if this line is a footer
                    for footer in header_footer_info['footers']:
                        if (footer['region'][1] <= y_pos <= footer['region'][3] and
                            footer['region'][0] <= line_words[0]['x0'] <= footer['region'][2]):
                            line_obj['is_footer'] = True
                            line_obj['footer_text'] = footer['text']
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
                    line['paragraph_spacing'] = paragraph.get('spacing', 0)
                    line['paragraph_indentation'] = paragraph.get('indentation', 0)
                    break
                    
            # Find which list this line belongs to
            for i, list_obj in enumerate(lists):
                for item in list_obj['items']:
                    if line['line_index'] == item['line_index']:
                        line['list_index'] = i
                        line['list_type'] = list_obj['type']
                        line['list_indentation'] = item['indentation']
                        line['list_marker'] = item.get('marker', '')
                        break
                        
        return result
    except Exception as e:
        logger.error(f"Error extracting text with formatting: {str(e)}")
        raise

def save_text_to_pdf(text, pdf_path, original_pdf_path):
    """
    Saves proofread text to a new PDF file by converting pages to images,
    overlaying corrected text, and converting back to PDF.
    """
    try:
        # Open the original PDF with PyMuPDF
        original_doc = fitz.open(original_pdf_path)
        img_info_list = [] # Store (image, page_num, original_page_rect)

        # Convert each page to an image and store original page dimensions
        for page_num in range(len(original_doc)):
            page = original_doc.load_page(page_num)
            original_page_rect = page.rect
            # Render page to a high-resolution PNG image
            # We'll use a fixed DPI (e.g., 300) for consistent scaling
            dpi = 300
            matrix = fitz.Matrix(dpi / 72, dpi / 72)
            pix = page.get_pixmap(matrix=matrix)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            img_info_list.append((img, page_num, original_page_rect))

        # original_doc.close() # Keep open to get text info

        # Ensure text is a single string
        if isinstance(text, list):
            text = ' '.join(text)

        # --- Image Processing: Clear Original Text Areas ---
        processed_img_list = [] # Store images with cleared text

        # Re-extract text information with bounding boxes from the original PDF
        # Scale these bounding boxes to the image dimensions
        original_spans_by_page = defaultdict(list)
        for page_num in range(len(original_doc)):
             page = original_doc.load_page(page_num)
             # Get text blocks and spans with their bounding boxes
             blocks = page.get_text("dict")['blocks']
             for b in blocks:
                 if b['type'] == 0: # text block
                     for l in b['lines']:
                         for s in l['spans']:
                             # Scale bbox from PDF points to image pixels
                             original_page_rect = original_doc[page_num].rect # Get rect again
                             img_width, img_height = img_info_list[page_num][0].size
                             x_scale = img_width / original_page_rect.width
                             y_scale = img_height / original_page_rect.height

                             scaled_bbox = fitz.Rect(s['bbox'])
                             scaled_bbox.x0 *= x_scale
                             scaled_bbox.y0 *= y_scale
                             scaled_bbox.x1 *= x_scale
                             scaled_bbox.y1 *= y_scale

                             original_spans_by_page[page_num].append({
                                 'bbox': scaled_bbox,
                                 'font': s.get('font', 'Helvetica'),
                                 'size': s.get('size', 11) * x_scale, # Scale font size too
                                 'color': s.get('color', 0),
                                 'flags': s.get('flags', 0),
                                 'text': s['text']
                             })

        original_doc.close() # Close the original doc after extracting info

        # Process each image: draw white rectangles over text areas
        for img, page_num, _ in img_info_list:
            draw = ImageDraw.Draw(img)
            
            # Draw white rectangles over original text span bounding boxes
            # Add a small buffer to ensure full coverage
            buffer_pixels = 2 # pixels
            for span_info in original_spans_by_page[page_num]:
                 bbox = span_info['bbox']
                 # Create a slightly buffered rectangle
                 erase_rect = [
                     bbox.x0 - buffer_pixels,
                     bbox.y0 - buffer_pixels,
                     bbox.x1 + buffer_pixels,
                     bbox.y1 + buffer_pixels,
                 ]
                 draw.rectangle(erase_rect, fill="white")

            processed_img_list.append((img, page_num)) # Store image with cleared text

        # --- Image Processing: Draw Corrected Text ---
        final_img_list = [] # Store images with corrected text drawn

        # Prepare original and corrected text for mapping
        # Concatenate text from all original spans across all pages
        original_full_text_concat = "".join([s['text'] for page_num in sorted(original_spans_by_page.keys()) for s in original_spans_by_page[page_num]])
        corrected_full_text = text

        # Use sequence matcher to find correspondences
        sm = difflib.SequenceMatcher(None, original_full_text_concat, corrected_full_text)
        opcodes = sm.get_opcodes()

        # Create a mapping from original character index to corrected character index
        # This map helps in translating positions from the original concatenated text to the corrected one.
        original_to_corrected_map = {}
        original_cursor_map = 0
        corrected_cursor_map = 0
        for tag, i1, i2, j1, j2 in opcodes:
            if tag == 'equal':
                for i in range(i1, i2):
                    original_to_corrected_map[i] = corrected_cursor_map + (i - i1)
                original_cursor_map += (i2 - i1)
                corrected_cursor_map += (j2 - j1)
            elif tag == 'replace':
                 # Map the start of the original replaced segment to the start of the corrected replacement segment
                 if i1 < i2: # Ensure original segment is not empty
                     original_to_corrected_map[i1] = j1
                 # Map the end of the original replaced segment to the end of the corrected replacement segment
                 if i2 > i1:
                      original_to_corrected_map[i2] = j2 # Point after the segment

                 original_cursor_map += (i2 - i1)
                 corrected_cursor_map += (j2 - j1)
            elif tag == 'insert':
                 # An insertion at original index i1 corresponds to corrected range j1 to j2
                 # Map the original index i1 to the start of the inserted text j1
                 original_to_corrected_map[i1] = j1
                 corrected_cursor_map += (j2 - j1)
            elif tag == 'delete':
                 # A deletion of original range i1 to i2 has no corresponding text in corrected version.
                 # Map the start of the deleted segment i1 to the corrected position j1 (end of preceding segment)
                 if i1 < i2: # Ensure original segment is not empty
                      original_to_corrected_map[i1] = j1
                 # Map the end of the deleted segment i2 to the same corrected position j1
                 if i2 > i1:
                       original_to_corrected_map[i2] = j1

                 original_cursor_map += (i2 - i1)
                 # corrected_cursor_map does not advance for deletions


        # Iterate through images with cleared text and draw corrected text
        original_char_pos_for_drawing = 0 # Track position in the concatenated original text

        for img, page_num in processed_img_list:
            draw = ImageDraw.Draw(img)
            
            # Get original spans for this page, sorted by their position to ensure correct processing order
            page_spans = sorted(original_spans_by_page[page_num], key=lambda s: (s['bbox'].y0, s['bbox'].x0))

            for span_info in page_spans:
                 original_span_text = span_info['text']
                 original_span_length = len(original_span_text)
                 span_bbox = span_info['bbox'] # Already scaled to image pixels
                 font_name = span_info.get('font', 'Helvetica')
                 font_size = span_info.get('size', 11) # Already scaled to image size
                 color_int = span_info.get('color', 0)
                 # Convert integer color to RGB tuple for Pillow
                 color_rgb = ((color_int >> 16) & 255, (color_int >> 8) & 255, color_int & 255)

                 is_bold = bool(span_info.get('flags', 0) & 1)
                 is_italic = bool(span_info.get('flags', 0) & 2)
                 
                 # Find the corresponding corrected text segment for this original span
                 start_original_index = original_char_pos_for_drawing
                 end_original_index = original_char_pos_for_drawing + original_span_length

                 # Use the character map to get the corresponding range in the corrected full text
                 # Get the start index in the corrected text. Use .get with default to handle cases not in map.
                 # If the start index of the original span is in the map, use its mapping.
                 # Otherwise, try to infer from the previous mapped position.

                 corrected_segment_start = original_to_corrected_map.get(start_original_index)

                 corrected_segment_parts = []
                 original_span_start_in_concat = original_char_pos_for_drawing
                 original_span_end_in_concat = original_char_pos_for_drawing + original_span_length

                 # Iterate through opcodes to find the corresponding corrected text
                 current_original_pos_in_opcodes = 0 # Tracks position within the original text as we iterate through opcodes
                 current_corrected_pos_in_opcodes = 0 # Tracks position within the corrected text as we iterate through opcodes

                 for tag, i1, i2, j1, j2 in sm.get_opcodes():
                     # Original range of the current opcode: i1 to i2 in original_full_text_concat
                     # Corrected range of the current opcode: j1 to j2 in corrected_full_text

                     # Calculate the overlap between the original span's range and the opcode's original range
                     overlap_start_in_original_span = max(original_span_start_in_concat, i1)
                     overlap_end_in_original_span = min(original_span_end_in_concat, i2)

                     # If there is an overlap in the original text ranges
                     if overlap_start_in_original_span < overlap_end_in_original_span:
                         # Determine the corresponding segment in the corrected text
                         corrected_overlap_start = -1
                         corrected_overlap_end = -1

                         if tag == 'equal' or tag == 'replace':
                             # For equal/replace, map the overlapping range from original to corrected opcode range
                             pos_in_opcode_original = overlap_start_in_original_span - i1
                             corrected_overlap_start = j1 + pos_in_opcode_original

                             # Calculate the length of the overlap in the original text
                             overlap_length = overlap_end_in_original_span - overlap_start_in_original_span

                             # The corresponding end in corrected depends on the tag
                             if tag == 'equal':
                                 corrected_overlap_end = corrected_overlap_start + overlap_length # Length is the same
                             elif tag == 'replace' and (i2 - i1) > 0:
                                 # For replace, scale the overlap length by the ratio of corrected to original opcode lengths
                                 proportion_of_opcode = (overlap_end_in_original_span - overlap_start_in_original_span) / (i2 - i1)
                                 corrected_overlap_end = j1 + int(proportion_of_opcode * (j2 - j1)) # Approximate end
                             else: # Handle replace with empty original segment (shouldn't overlap this way)
                                  corrected_overlap_end = corrected_overlap_start

                         elif tag == 'insert':
                             # For an insertion, if the insertion point (i1) is within the original span's range,
                             # we consider the entire inserted segment (j1 to j2) as potentially belonging to this span.
                             # This is a heuristic; a more precise method is very complex.
                             # Let's check if the insertion point is within the span's *original* boundaries.
                             if i1 >= original_span_start_in_concat and i1 < original_span_end_in_concat:
                                 corrected_overlap_start = j1
                                 corrected_overlap_end = j2

                         elif tag == 'delete':
                             # For a deletion, the corrected range is empty. If the deletion range overlaps the original span,
                             # it means part of the original span was deleted. There is no corresponding corrected text.
                             corrected_overlap_start = j1 # End of previous segment in corrected text
                             corrected_overlap_end = j1 # No text added


                         # Append the corresponding corrected text segment if valid
                         if corrected_overlap_start != -1 and corrected_overlap_start <= corrected_overlap_end:
                              corrected_segment_parts.append(corrected_full_text[corrected_overlap_start:corrected_overlap_end])
                         elif corrected_overlap_start != -1 and corrected_overlap_start > corrected_overlap_end:
                             logger.warning(f"Calculated reversed corrected overlap range: {corrected_overlap_start}-{corrected_overlap_end} for tag {tag}, original span range {original_span_start_in_concat}-{original_span_end_in_concat}")
                             # Append empty string to avoid errors with reversed ranges
                             corrected_segment_parts.append("")
                         # If corrected_overlap_start is still -1, no valid corrected segment found for this overlap


                     # If the original range of the current opcode starts after the original span's end,
                     # we can break the opcode iteration for this span.
                     if i1 >= original_span_end_in_concat:
                         break


                 corrected_segment_for_span = "".join(corrected_segment_parts)

                 # If the corrected segment is empty or only whitespace, and the original span had text,
                 # this might indicate a deletion or a mapping issue. We should skip drawing this segment.
                 # Only skip if the original span had content and the corrected one is empty/whitespace.
                 if not corrected_segment_for_span.strip() and original_span_length > 0 and original_span_text.strip():
                      original_char_pos_for_drawing += original_span_length
                      continue # Skip drawing empty or whitespace-only corrected segments that replaced non-empty original text
                 # If the original span was empty/whitespace, and corrected is also empty/whitespace, we also skip.
                 if not corrected_segment_for_span.strip() and not original_span_text.strip():
                      original_char_pos_for_drawing += original_span_length
                      continue


                 # Calculate text position within the span bbox to preserve original alignment
                 text_x = span_bbox.x0 # Default horizontal position
                 text_y = span_bbox.y0 # Default vertical position (top of bbox)

                 # Get font for drawing
                 image_font = None
                 pillow_font_size = int(font_size)

                 # Calculate text dimensions before alignment adjustments
                 if image_font:
                     try:
                         text_render_bbox = draw.textbbox((0, 0), corrected_segment_for_span, font=image_font)
                         text_width_pixels = text_render_bbox[2] - text_render_bbox[0]
                         text_height_pixels = text_render_bbox[3] - text_render_bbox[1]
                         # Adjust vertical position based on font metrics
                         text_y = span_bbox.y0 - text_render_bbox[1]
                     except Exception as bbox_e:
                         logger.warning(f"Error calculating text bbox: {bbox_e}")
                         text_width_pixels = len(corrected_segment_for_span) * (pillow_font_size * 0.6)
                         text_height_pixels = pillow_font_size
                 else:
                     text_width_pixels = len(corrected_segment_for_span) * (pillow_font_size * 0.6)
                     text_height_pixels = pillow_font_size

                 # Enhanced alignment detection using multiple heuristics
                 img_width, img_height = img.size
                 span_center_x = (span_bbox.x0 + span_bbox.x1) / 2
                 img_center_x = img_width / 2
                 
                 # Calculate margins and text block width
                 left_margin = span_bbox.x0
                 right_margin = img_width - span_bbox.x1
                 text_block_width = span_bbox.x1 - span_bbox.x0
                 
                 # Calculate alignment score for each type
                 center_score = 1 - min(1, abs(span_center_x - img_center_x) / (img_width * 0.1))
                 right_score = 1 - min(1, right_margin / (left_margin + 1))
                 justify_score = 1 - min(1, abs(text_block_width - text_width_pixels) / text_block_width)
                 
                 # Determine alignment based on scores
                 alignment = 'left'  # Default
                 max_score = 0
                 
                 if center_score > 0.8 and center_score > max_score:
                     alignment = 'center'
                     max_score = center_score
                 if right_score > 0.8 and right_score > max_score:
                     alignment = 'right'
                     max_score = right_score
                 if justify_score > 0.9 and justify_score > max_score:
                     alignment = 'justify'
                     max_score = justify_score

                 # Apply alignment to text position
                 if alignment == 'right':
                     text_x = span_bbox.x1 - text_width_pixels
                 elif alignment == 'center':
                     text_x = span_bbox.x0 + (text_block_width - text_width_pixels) / 2
                 elif alignment == 'justify':
                     # For justified text, we'll use the full width of the span
                     text_x = span_bbox.x0
                     # Note: Actual justification would require word spacing adjustment
                     # which is complex to implement with PIL
                 else:  # left alignment
                     text_x = span_bbox.x0

                 # Ensure text doesn't go beyond span bounds
                 text_x = max(span_bbox.x0, min(text_x, span_bbox.x1 - text_width_pixels))

                 # Draw the corrected text onto the image
                 try:
                     if image_font:
                         draw.text((text_x, text_y), corrected_segment_for_span, fill=color_rgb, font=image_font)
                     else:
                         draw.text((text_x, text_y), corrected_segment_for_span, fill=color_rgb)
                 except Exception as drawing_e:
                     logger.error(f"Error drawing text segment '{corrected_segment_for_span}': {drawing_e}")
                     pass

                 original_char_pos_for_drawing += original_span_length

            final_img_list.append((img, page_num)) # Store image with corrected text drawn and its page_num

        # --- Final PDF Creation from Processed Images ---
        new_doc = fitz.open()
        # We need the original page dimensions for each image to set the new PDF page size correctly
        # img_info_list contains (original_image, page_num, original_page_rect)
        # final_img_list contains the processed images and their page_num in the same order

        for img, page_num in final_img_list:
             # Get the original page info for this image using the stored page_num
             original_page_rect = None
             for img_info, info_page_num, original_rect in img_info_list:
                  if info_page_num == page_num:
                       original_page_rect = original_rect
                       break

             img_byte_arr = io.BytesIO()
             img.save(img_byte_arr, format='PNG')
             img_bytes = img_byte_arr.getvalue()

             # Insert image into a new PDF page with original dimensions
             if original_page_rect:
                  original_page_width = original_page_rect.width
                  original_page_height = original_page_rect.height
                  img_page = new_doc.new_page(width=original_page_width, height=original_page_height)

                  # Calculate the image size in points based on DPI (used during creation)
                  dpi = 300
                  img_width_pts = img.width * 72 / dpi
                  img_height_pts = img.height * 72 / dpi

                  # Create a rectangle on the new PDF page to place the image
                  # It should cover the whole page, matching original dimensions in points
                  image_rect_on_pdf = fitz.Rect(0, 0, original_page_width, original_page_height)

                  # Insert the image into the PDF page, scaled to fit the defined rectangle
                  img_page.insert_image(image_rect_on_pdf, stream=img_bytes)

             else:
                  # Fallback if original dimensions not found
                  img_page = new_doc.new_page(width=img.width, height=img.height) # Use image dimensions in pixels as points
                  img_page.insert_image(img_page.rect, stream=img_bytes)


        new_doc.save(pdf_path)
        new_doc.close()

        logger.info("PDF saved successfully using image overlay method")

    except Exception as e:
        logger.error(f"Error in image-based PDF creation: {str(e)}")
        raise e

@app.route('/convert', methods=['POST'])
def convert_and_proofread():
    """Handles PDF proofreading."""
    try:
        if 'file' not in request.files and 'text' not in request.form:
            return jsonify({"error": "No file or text provided"}), 400

        # Handle text and suggestions if provided directly
        if 'text' in request.form:
            proofread_text_content = request.form['text']
            original_filename = request.form.get('filename', 'document.pdf')
            
            # Store the original file path in a session-like dictionary
            original_pdf_path = os.path.join(UPLOAD_FOLDER, original_filename)
            
            # Ensure the original file exists
            if not os.path.exists(original_pdf_path):
                # Try to find the file with "proofread_" prefix removed
                base_filename = original_filename.replace("proofread_", "", 1)
                original_pdf_path = os.path.join(UPLOAD_FOLDER, base_filename)
                
                if not os.path.exists(original_pdf_path):
                    return jsonify({"error": "Original file not found"}), 404
            
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
            output_filename = "proofread_" + original_filename.replace("proofread_", "", 1)  # Avoid duplicate prefix
            output_path = os.path.join(OUTPUT_FOLDER, output_filename)
            
            # Use the original PDF as a template for formatting
            save_text_to_pdf(proofread_text_content, output_path, original_pdf_path)
            
            return jsonify({
                "download_url": "/download/" + output_filename
            })

        # Handle file upload case
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        filename = secure_filename(file.filename)
        pdf_path = os.path.join(UPLOAD_FOLDER, filename)
        
        # Create UPLOAD_FOLDER if it doesn't exist
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        
        # Save the uploaded file
        file.save(pdf_path)

        # Extract text from PDF
        extracted_text = extract_text_from_pdf(pdf_path)

        # Proofread the text
        proofread_text_content, grammar_errors = proofread_text(extracted_text)

        # Save proofread text back to PDF
        proofread_pdf_filename = "proofread_" + filename
        proofread_pdf_path = os.path.join(OUTPUT_FOLDER, proofread_pdf_filename)
        
        # Create OUTPUT_FOLDER if it doesn't exist
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
        
        # Save the proofread PDF using the original as template
        save_text_to_pdf(proofread_text_content, proofread_pdf_path, pdf_path)

        return jsonify({
            "original_text": extracted_text,
            "proofread_text": proofread_text_content,
            "grammar_errors": grammar_errors,
            "download_url": "/download/" + proofread_pdf_filename,
            "file_name": filename  # Add filename to response for frontend reference
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
