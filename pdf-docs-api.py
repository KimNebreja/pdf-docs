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
import statistics

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "processed_files"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Using local LanguageTool instance for more accuracy
tool = language_tool_python.LanguageToolPublicAPI('en-US')  # Uses the online API

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file with advanced formatting preservation."""
    try:
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        doc = fitz.open(pdf_path)
        text = []
        
        if doc.page_count == 0:
            raise ValueError("PDF file is empty or corrupted")
            
        for page_num in range(doc.page_count):
            try:
                page = doc[page_num]
                # Extract text with formatting information
                page_text = page.get_text()
                if isinstance(page_text, str):
                    text.append(page_text)
                else:
                    logger.warning(f"Unexpected text type on page {page_num}: {type(page_text)}")
                    text.append("")
            except Exception as page_error:
                logger.error(f"Error extracting text from page {page_num}: {str(page_error)}")
                text.append("")
                continue
                
        doc.close()
        
        # Join text with newlines and ensure it's a string
        final_text = "\n".join(text)
        if not isinstance(final_text, str):
            raise TypeError(f"Expected string output, got {type(final_text)}")
            
        if not final_text.strip():
            return "No text content found in PDF"
            
        return final_text
        
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {str(e)}")
        raise ValueError(f"Failed to extract text from PDF: {str(e)}")

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

def detect_paragraphs(formatted_text):
    """
    Groups text blocks into paragraphs based on their position and formatting.
    Returns a list of paragraph objects with text and formatting information.
    """
    paragraphs = []
    current_paragraph = None
    
    for block in formatted_text:
        block_info = block.get('block_info', {})
        lines = block.get('lines', [])
        alignment = block.get('alignment', {})
        
        # Skip empty blocks
        if not lines:
            continue
        
        # Start a new paragraph if:
        # 1. No current paragraph
        # 2. Different alignment
        # 3. Significant vertical gap
        # 4. First line is indented but current paragraph's last line isn't
        if current_paragraph is None:
            current_paragraph = {
                'text': block['text'],
                'lines': lines,
                'alignment': alignment,
                'block_info': block_info
            }
        else:
            prev_block_info = current_paragraph.get('block_info', {})
            vertical_gap = block_info.get('y0', 0) - prev_block_info.get('y1', 0)
            
            # Check if this is a new paragraph
            if (vertical_gap > 20 or  # Significant vertical gap
                alignment['type'] != current_paragraph['alignment']['type'] or  # Different alignment
                (alignment.get('indented', False) and not current_paragraph['alignment'].get('indented', False))):  # Indentation change
                
                # Save current paragraph
                paragraphs.append(current_paragraph)
                
                # Start new paragraph
                current_paragraph = {
                    'text': block['text'],
                    'lines': lines,
                    'alignment': alignment,
                    'block_info': block_info
                }
            else:
                # Append to current paragraph
                current_paragraph['text'] += '\n' + block['text']
                current_paragraph['lines'].extend(lines)
                
                # Update block info to span both blocks
                current_paragraph['block_info'] = {
                    'x0': min(prev_block_info.get('x0', float('inf')), block_info.get('x0', float('inf'))),
                    'x1': max(prev_block_info.get('x1', 0), block_info.get('x1', 0)),
                    'y0': prev_block_info.get('y0', 0),
                    'y1': block_info.get('y1', 0),
                    'width': max(prev_block_info.get('width', 0), block_info.get('width', 0)),
                    'page_width': block_info.get('page_width', 0)
                }
    
    # Add the last paragraph
    if current_paragraph:
        paragraphs.append(current_paragraph)
    
    return paragraphs

def get_text_color(page, bbox):
    """
    Get the text color from a specific region in the PDF page.
    Returns RGB tuple or None if color cannot be determined.
    """
    try:
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            logger.warning(f"Invalid bbox format in get_text_color: {bbox}")
            return None
            
        x0, y0, x1, y1 = map(float, bbox)
        rect = fitz.Rect(x0, y0, x1, y1)
        
        # Extract text blocks in the region
        blocks = page.get_text("dict", clip=rect).get('blocks', [])
        
        if not blocks:
            return None
            
        # Get spans from the first block
        spans = blocks[0].get('lines', [{}])[0].get('spans', [])
        
        if not spans:
            return None
            
        # Get color from the first span
        color = spans[0].get('color')
        
        if not color:
            return None
            
        return color
        
    except Exception as e:
        logger.error(f"Error getting text color: {str(e)}")
        return None

def normalize_color(color):
    """
    Normalize color values to RGB format (0-1 range).
    Returns tuple of (r, g, b) values.
    """
    try:
        if not isinstance(color, (list, tuple)) or len(color) < 3:
            logger.warning(f"Invalid color format: {color}")
            return (0, 0, 0)  # Default to black
            
        # Extract RGB values
        r, g, b = color[:3]
        
        # Ensure values are numeric
        try:
            r = float(r)
            g = float(g)
            b = float(b)
        except (ValueError, TypeError):
            logger.warning(f"Invalid color values: {color}")
            return (0, 0, 0)
            
        # Normalize values to 0-1 range
        r = max(0, min(1, r))
        g = max(0, min(1, g))
        b = max(0, min(1, b))
        
        return (r, g, b)
        
    except Exception as e:
        logger.error(f"Error normalizing color: {str(e)}")
        return (0, 0, 0)  # Default to black

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
    """Extract text with formatting information from a PDF file."""
    try:
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        doc = fitz.open(pdf_path)
        formatted_text = []
        
        for page_num in range(len(doc)):
            try:
                page = doc[page_num]
                page_width = page.rect.width
                page_dict = page.get_text("dict")
                
                if not isinstance(page_dict, dict) or "blocks" not in page_dict:
                    logger.warning(f"Invalid page dictionary format on page {page_num}")
                    continue
                    
                blocks = page_dict["blocks"]
                
                for block in blocks:
                    if not isinstance(block, dict):
                        continue
                        
                    if block.get("type") == 0:  # text block
                        lines = []
                        text_lines = []
                        
                        for line in block.get("lines", []):
                            if not isinstance(line, dict):
                                continue
                                
                            line_text = ""
                            line_spans = []
                            
                            for span in line.get("spans", []):
                                if not isinstance(span, dict):
                                    continue
                                    
                                span_text = span.get("text", "").strip()
                                if span_text:
                                    font_info = {
                                        "size": float(span.get("size", 0)),
                                        "font": str(span.get("font", "")),
                                        "color": span.get("color", 0)
                                    }
                                    line_spans.append({
                                        "text": span_text,
                                        "font_info": font_info,
                                        "bbox": span.get("bbox", [0, 0, 0, 0])
                                    })
                                    line_text += span_text + " "
                            
                            if line_text.strip():
                                text_lines.append(line_text.strip())
                                lines.append({
                                    "text": line_text.strip(),
                                    "spans": line_spans,
                                    "bbox": line.get("bbox", [0, 0, 0, 0])
                                })
                        
                        if lines:
                            block_text = "\n".join(text_lines)
                            formatted_text.append({
                                "text": block_text,
                                "lines": lines,
                                "block_info": {
                                    "page_num": page_num,
                                    "page_width": float(page_width),
                                    "bbox": block.get("bbox", [0, 0, 0, 0])
                                }
                            })
            except Exception as page_error:
                logger.error(f"Error processing page {page_num}: {str(page_error)}")
                continue
        
        doc.close()
        return formatted_text
    except Exception as e:
        logger.error(f"Error extracting formatted text from PDF: {str(e)}")
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
                
                try:
                    # Get paragraphs for this page
                    page_paragraphs = [p for p in paragraphs if isinstance(p, dict) and 
                                     p.get('block_info', {}).get('page_num') == page_num]
                    
                    # Process each paragraph
                    for paragraph in page_paragraphs:
                        if current_paragraph >= len(proofread_paragraphs):
                            break
                        
                        try:
                            # Process each line in the paragraph
                            for line in paragraph.get('lines', []):
                                if not isinstance(line, dict):
                                    continue
                                    
                                if current_paragraph >= len(proofread_paragraphs):
                                    break
                                
                                # Save canvas state
                                c.saveState()
                                
                                try:
                                    # Get line properties from bbox
                                    bbox = line.get('bbox', [0, 0, 0, 0])
                                    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                                        logger.warning(f"Invalid bbox format: {bbox}")
                                        continue
                                        
                                    x_pos = float(bbox[0])
                                    y_pos = float(bbox[1])
                                    
                                    # Get the first span's properties
                                    spans = line.get('spans', [])
                                    if not spans or not isinstance(spans[0], dict):
                                        continue
                                        
                                    first_span = spans[0]
                                    
                                    # Get font and size with defaults
                                    font_info = first_span.get('font_info', {})
                                    font = get_font_name(font_info.get('font', 'Helvetica'))
                                    fontsize = float(font_info.get('size', 11))
                                    
                                    # Get color with default black
                                    color = get_text_color(mupdf_page, bbox)
                                    if color:
                                        r, g, b = normalize_color(color)
                                        fill_color = Color(r, g, b)
                                    else:
                                        fill_color = Color(0, 0, 0)  # Default to black
                                    
                                    # Calculate text metrics
                                    c.setFont(font, fontsize)
                                    text_width = c.stringWidth(proofread_paragraphs[current_paragraph], font, fontsize)
                                    text_height = fontsize * 1.2
                                    
                                    # Calculate position
                                    baseline_offset = fontsize * 0.2
                                    y_pos_adjusted = page_height - y_pos - text_height/2 + baseline_offset
                                    
                                    # Get alignment with default left
                                    alignment = paragraph.get('alignment', {}).get('type', 'left')
                                    
                                    # Adjust position based on alignment
                                    if alignment == 'center':
                                        x_pos = (page_width - text_width) / 2
                                    elif alignment == 'right':
                                        x_pos = page_width - text_width - 50
                                    elif alignment == 'justified':
                                        # Simple justification
                                        pass
                                    
                                    # Draw text
                                    text_object = c.beginText(x_pos, y_pos_adjusted)
                                    text_object.setFont(font, fontsize)
                                    text_object.setFillColor(fill_color)
                                    text_object.textLine(proofread_paragraphs[current_paragraph])
                                    c.drawText(text_object)
                                    current_paragraph += 1
                                    
                                finally:
                                    # Restore canvas state
                                    c.restoreState()
                                    
                        except Exception as para_error:
                            logger.error(f"Error processing paragraph: {str(para_error)}")
                            continue
                    
                    # Move to next page
                    c.showPage()
                    
                except Exception as page_error:
                    logger.error(f"Error processing page {page_num}: {str(page_error)}")
                    c.showPage()  # Still move to next page even if there's an error
                    continue
            
            # Save the PDF
            c.save()
            doc.close()
            logger.info("PDF saved successfully")
            
    except Exception as e:
        logger.error(f"Error creating PDF: {str(e)}")
        raise

def detect_single_line_alignment(line, page_width=None):
    """
    Detect alignment for a single line.
    
    Args:
        line: A line dictionary containing text and position information
        page_width: Optional page width override
    """
    try:
        if not isinstance(line, dict):
            return 'left'
            
        line_width = line.get('width', 0)
        line_left = line.get('x', 0)
        if page_width is None:
            page_width = line.get('page_width', 0)
        line_right = page_width - (line_left + line_width)
        
        # Check for center alignment
        if abs(line_left - line_right) < 2:
            return 'center'
        
        # Check for right alignment
        if line_right < line_left * 0.7:
            return 'right'
        
        # Default to left alignment
        return 'left'
    except Exception as e:
        logger.warning(f"Error detecting single line alignment: {str(e)}")
        return 'left'

def detect_alignment(lines, page_width):
    """Detect text alignment based on line positions and widths."""
    if not lines or not isinstance(lines, list):
        return {
            "type": "left",
            "confidence": 0.5,
            "scores": {"left": 0.5, "right": 0.0, "center": 0.0, "justified": 0.0},
            "indented": False,
            "mixed": False
        }
    
    # Extract line information
    line_info = []
    for line in lines:
        if isinstance(line, dict):
            bbox = line.get("bbox", [0, 0, 0, 0])
            text = line.get("text", "").strip()
            if text and len(bbox) == 4:
                left_margin = bbox[0]
                right_margin = page_width - bbox[2]
                width = bbox[2] - bbox[0]
                
                # Enhanced text analysis for justified detection
                words = text.split()
                word_count = len(words)
                space_count = text.count(" ")
                avg_word_length = sum(len(word) for word in words) / word_count if word_count > 0 else 0
                
                # Calculate space distribution
                spaces = [len(space) for space in text.split(" ") if space]
                space_variance = statistics.variance(spaces) if len(spaces) > 1 else 0
                
                line_info.append({
                    "text": text,
                    "left_margin": left_margin,
                    "right_margin": right_margin,
                    "width": width,
                    "word_count": word_count,
                    "space_count": space_count,
                    "avg_word_length": avg_word_length,
                    "space_variance": space_variance,
                    "avg_space_per_word": space_count / word_count if word_count > 0 else 0
                })
    
    if not line_info:
        return {
            "type": "left",
            "confidence": 0.5,
            "scores": {"left": 0.5, "right": 0.0, "center": 0.0, "justified": 0.0},
            "indented": False,
            "mixed": False
        }
    
    # Calculate statistics with improved precision
    avg_left_margin = sum(l["left_margin"] for l in line_info) / len(line_info)
    avg_right_margin = sum(l["right_margin"] for l in line_info) / len(line_info)
    avg_width = sum(l["width"] for l in line_info) / len(line_info)
    
    # Enhanced variance calculations
    left_margin_var = statistics.variance([l["left_margin"] for l in line_info]) if len(line_info) > 1 else 0
    right_margin_var = statistics.variance([l["right_margin"] for l in line_info]) if len(line_info) > 1 else 0
    width_var = statistics.variance([l["width"] for l in line_info]) if len(line_info) > 1 else 0
    
    # Calculate margin ratios with improved precision
    left_ratio = avg_left_margin / page_width
    right_ratio = avg_right_margin / page_width
    
    # Enhanced heading detection
    is_heading = False
    if len(line_info) <= 2:
        text = line_info[0]["text"]
        # More comprehensive heading patterns
        heading_patterns = [
            r'^[A-Z][a-z]*\s+\d',  # Chapter 1, Section 2
            r'^(Chapter|Section|Part|Title)\s+\d',  # Common heading prefixes
            r'^[A-Z][A-Za-z\s]+$',  # All caps or title case
            r'^[A-Z][a-z]+\s+[A-Z][a-z]+'  # Title Case Words
        ]
        if any(re.match(pattern, text) for pattern in heading_patterns):
            is_heading = True
    
    # Special handling for short text with improved thresholds
    if len(line_info) <= 2:
        margin_diff = abs(left_ratio - right_ratio)
        
        # Enhanced center alignment detection
        if margin_diff < 0.03:  # More precise threshold
            return {
                "type": "center",
                "confidence": 0.95,
                "scores": {"left": 0.2, "right": 0.2, "center": 0.95, "justified": 0.0},
                "indented": False,
                "mixed": False
            }
        
        # Enhanced right alignment detection
        if right_ratio < 0.08 and left_ratio > right_ratio * 2.5:
            return {
                "type": "right",
                "confidence": 0.95,
                "scores": {"left": 0.2, "right": 0.95, "center": 0.2, "justified": 0.0},
                "indented": False,
                "mixed": False
            }
        
        # Enhanced left alignment detection
        if left_ratio < 0.08 and right_ratio > left_ratio * 2.5:
            return {
                "type": "left",
                "confidence": 0.95,
                "scores": {"left": 0.95, "right": 0.2, "center": 0.2, "justified": 0.0},
                "indented": False,
                "mixed": False
            }
    
    # Enhanced justified text detection
    width_consistency = 1.0 - min(width_var / (avg_width * 0.05), 1.0)
    space_consistency = 1.0 - min(statistics.mean([l["space_variance"] for l in line_info]) * 5, 1.0)
    margin_consistency = 1.0 - max(left_margin_var, right_margin_var) / (page_width * 0.05)
    
    # Calculate word length consistency with improved precision
    word_lengths = [l["avg_word_length"] for l in line_info]
    word_length_var = statistics.variance(word_lengths) if len(word_lengths) > 1 else 0
    word_length_consistency = 1.0 - min(word_length_var * 0.3, 1.0)
    
    # Enhanced justified score calculation
    justified_score = (
        width_consistency * 0.35 +
        space_consistency * 0.35 +
        margin_consistency * 0.15 +
        word_length_consistency * 0.15
    ) * (1.0 + min(len(line_info) / 8.0, 0.6))  # Adjusted boost factor
    
    # Early return for high-confidence justified text
    if justified_score > 0.75 and len(line_info) > 2 and not is_heading:
        return {
            "type": "justified",
            "confidence": justified_score,
            "scores": {
                "left": 0.1,
                "right": 0.1,
                "center": 0.1,
                "justified": justified_score
            },
            "indented": False,
            "mixed": False
        }
    
    # Enhanced alignment score calculations
    left_score = 1.0 - min(left_margin_var / (page_width * 0.05), 1.0)
    right_score = 1.0 - min(right_margin_var / (page_width * 0.05), 1.0)
    center_score = 1.0 - min(abs(avg_left_margin - avg_right_margin) / (page_width * 0.05), 1.0)
    
    # Context-based score adjustments
    if justified_score > 0.4:
        left_score *= 0.6
        right_score *= 0.6
        center_score *= 0.6
    
    # Enhanced right alignment detection
    if right_ratio < left_ratio * 0.65 and right_margin_var < left_margin_var * 0.8:
        right_score = max(right_score, 0.95)
    
    # Enhanced heading handling
    if is_heading:
        center_score = max(center_score, 0.95)
        left_score *= 0.6
        right_score *= 0.6
        justified_score = 0.0
    
    # Normalize scores with improved precision
    scores = {
        "left": max(0.0, min(1.0, left_score)),
        "right": max(0.0, min(1.0, right_score)),
        "center": max(0.0, min(1.0, center_score)),
        "justified": max(0.0, min(1.0, justified_score))
    }
    
    # Determine alignment type with improved confidence
    alignment_type = max(scores.items(), key=lambda x: x[1])[0]
    confidence = scores[alignment_type]
    
    # Enhanced indentation detection
    first_line = line_info[0]
    other_lines = line_info[1:] if len(line_info) > 1 else []
    is_indented = False
    
    if other_lines:
        other_left_margin = sum(l["left_margin"] for l in other_lines) / len(other_lines)
        indent_diff = first_line["left_margin"] - other_left_margin
        is_indented = abs(indent_diff) > page_width * 0.03  # More precise threshold
    
    # Enhanced mixed alignment detection
    is_mixed = False
    if len(line_info) > 1:
        alignment_types = set()
        for i in range(len(line_info) - 1):
            curr_line = line_info[i]
            next_line = line_info[i + 1]
            if abs(curr_line["left_margin"] - next_line["left_margin"]) > page_width * 0.03:
                is_mixed = True
                break
    
    return {
        "type": alignment_type,
        "confidence": confidence,
        "scores": scores,
        "indented": is_indented,
        "mixed": is_mixed
    }

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

        # Save proofread text back to PDF (without proofreading for now)
        proofread_pdf_filename = "proofread_" + filename
        proofread_pdf_path = os.path.join(OUTPUT_FOLDER, proofread_pdf_filename)
        save_text_to_pdf(extracted_text, proofread_pdf_path, pdf_path)  # Using original text instead of proofread text

        # Get the base URL for the download link
        base_url = request.host_url.rstrip('/')
        
        return jsonify({
            "original_text": extracted_text,
            "proofread_text": extracted_text,  # Using original text for now
            "grammar_errors": [],  # Empty list since we're not proofreading
            "download_url": f"{base_url}/download/{proofread_pdf_filename}",
            "alignment_info": {
                'left': 0,
                'right': 0,
                'center': 0,
                'justified': 0,
                'indented': 0,
                'mixed': 0
            }
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
