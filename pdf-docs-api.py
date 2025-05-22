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
from reportlab.lib.enums import TA_JUSTIFY

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

    # Map common font names (add more as needed)
    font_map = {
        "helvetica": "Helvetica",
        "arial": "Helvetica",
        "arial-bold": "Helvetica-Bold",
        "arialitalic": "Helvetica-Oblique",
        "arial-bolditalic": "Helvetica-BoldOblique",
        "calibri": "Calibri",
        "calibri-bold": "Calibri-Bold",
        "calibri-italic": "Calibri-Italic",
        "calibri-bolditalic": "Calibri-BoldItalic",
        "verdana": "Verdana",
        "tahoma": "Tahoma",
        "georgia": "Georgia",
        "times": "Times-Roman",
        "times new roman": "Times-Roman",
        "times-bold": "Times-Bold",
        "times-italic": "Times-Italic",
        "times-bolditalic": "Times-BoldItalic",
        "courier": "Courier",
        "courier new": "Courier",
        "symbol": "Symbol",
        "zapfdingbats": "ZapfDingbats",
    }
    for key in font_map:
        if key in font_name:
            return font_map[key]
    # Fallback to Helvetica
    return "Helvetica"

def register_fonts():
    """Registers common fonts with ReportLab if available."""
    try:
        from reportlab.pdfbase.ttfonts import TTFont
        from reportlab.pdfbase import pdfmetrics
        import os
        # Register standard built-in fonts
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
        font_candidates = {
            'Arial': 'arial.ttf',
            'Arial-Bold': 'arialbd.ttf',
            'Arial-Italic': 'ariali.ttf',
            'Arial-BoldItalic': 'arialbi.ttf',
            'Calibri': 'calibri.ttf',
            'Calibri-Bold': 'calibrib.ttf',
            'Calibri-Italic': 'calibrii.ttf',
            'Calibri-BoldItalic': 'calibriz.ttf',
            'Verdana': 'verdana.ttf',
            'Tahoma': 'tahoma.ttf',
            'Georgia': 'georgia.ttf',
            'Times New Roman': 'times.ttf',
            'Times-Bold': 'timesbd.ttf',
            'Times-Italic': 'timesi.ttf',
            'Times-BoldItalic': 'timesbi.ttf',
            'Courier New': 'cour.ttf',
            'Courier-Bold': 'courbd.ttf',
            'Courier-Oblique': 'couri.ttf',
            'Courier-BoldOblique': 'courbi.ttf',
        }
        font_dir = os.path.join(os.environ.get('WINDIR', ''), 'Fonts')
        for font_name, font_file in font_candidates.items():
            font_path = os.path.join(font_dir, font_file)
            if os.path.exists(font_path):
                try:
                    pdfmetrics.registerFont(TTFont(font_name, font_path))
                except Exception as e:
                    logger.warning(f"Could not register font {font_name}: {str(e)}")
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
    Extracts text with detailed formatting information and image positions using pdfplumber and PyMuPDF.
    Returns a list of dictionaries with text, formatting details, and image information.
    """
    try:
        result = []
        images = []

        # Open PDF with PyMuPDF to get image information
        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            mupdf_page = doc[page_num]
            image_list = mupdf_page.get_images(full=True)
            for img in image_list:
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_rect = mupdf_page.get_image_bbox(img)
                images.append({
                    'page': page_num,
                    'rect': image_rect,
                    'xref': xref,
                    'ext': base_image["ext"]
                })

        # Use pdfplumber for text extraction
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, plumber_page in enumerate(pdf.pages):
                # Use extract_text_blocks if available, else fallback to extract_words
                if hasattr(plumber_page, "extract_text_blocks"):
                    text_blocks = plumber_page.extract_text_blocks()
                else:
                    words = plumber_page.extract_words()
                    text_blocks = [{
                        'text': ' '.join([w['text'] for w in words]),
                        'x0': min([w['x0'] for w in words]) if words else 0,
                        'top': min([w['top'] for w in words]) if words else 0,
                        'x1': max([w['x1'] for w in words]) if words else 0,
                        'bottom': max([w['bottom'] for w in words]) if words else 0,
                    }]
                for block in text_blocks:
                    # Add image information if this block contains an image
                    block_images = [
                        img for img in images
                        if img['page'] == page_num and
                        img['rect'].intersects(fitz.Rect(block['x0'], block['top'], block['x1'], block['bottom']))
                    ]
                    block['images'] = block_images
                    result.append(block)

        doc.close()
        return result

    except Exception as e:
        logger.error(f"Error extracting text with formatting: {str(e)}")
        raise

def apply_proofread_text(original_text, proofread_text, selected_suggestions=None):
    """
    Applies proofread text and suggestions to the original text while maintaining formatting.
    Returns a dictionary mapping original text positions to their proofread versions.
    """
    try:
        # Initialize the mapping
        text_mapping = {}
        
        # First apply any selected suggestions
        if selected_suggestions:
            for original_word, selected_word in selected_suggestions.items():
                # Find all occurrences of the original word
                start = 0
                while True:
                    start = original_text.find(original_word, start)
                    if start == -1:
                        break
                    text_mapping[start] = {
                        'original': original_word,
                        'proofread': selected_word,
                        'length': len(original_word)
                    }
                    start += len(original_word)
        
        # Use difflib to find differences between original and proofread text
        matcher = difflib.SequenceMatcher(None, original_text, proofread_text)
        
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'equal':
                # Text is the same, no changes needed
                continue
            elif tag in ('replace', 'insert'):
                # Store the mapping for changed text
                text_mapping[i1] = {
                    'original': original_text[i1:i2],
                    'proofread': proofread_text[j1:j2],
                    'length': i2 - i1
                }
            elif tag == 'delete':
                # Mark deleted text
                text_mapping[i1] = {
                    'original': original_text[i1:i2],
                    'proofread': '',
                    'length': i2 - i1
                }
        
        return text_mapping
    except Exception as e:
        logger.error(f"Error applying proofread text: {str(e)}")
        raise

def save_text_to_pdf(text, pdf_path, original_pdf_path):
    """
    Saves proofread text to a new PDF file while preserving the exact formatting, images, and structure of the original PDF.
    """
    try:
        # Open the original PDF to get its structure
        original_doc = fitz.open(original_pdf_path)
        new_doc = fitz.open()
        
        # Extract the original text for comparison
        original_text = ""
        text_blocks = []
        for page in original_doc:
            blocks = page.get_text("dict")["blocks"]
            text_blocks.extend(blocks)
            original_text += page.get_text()
        
        # Apply proofreading changes
        text_mapping = apply_proofread_text(original_text, text)
        
        # Process each page
        for page_num in range(len(original_doc)):
            # Get the original page
            original_page = original_doc[page_num]
            
            # Create a new page with the same dimensions
            new_page = new_doc.new_page(width=original_page.rect.width, height=original_page.rect.height)
            
            # Copy all images from the original page
            image_list = original_page.get_images(full=True)
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = original_doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                
                # Insert the image at the same position
                image_rect = original_page.get_image_bbox(img)
                new_page.insert_image(image_rect, stream=image_bytes)
            
            # Get the text blocks from the original page
            page_blocks = original_page.get_text("dict")["blocks"]
            
            # Process each text block
            for block in page_blocks:
                if "lines" in block:
                    for line in block["lines"]:
                        if "spans" in line:
                            for span in line["spans"]:
                                # Get the original text and its position
                                original_span_text = span["text"]
                                bbox = fitz.Rect(span["bbox"])
                                
                                if original_span_text.strip():
                                    # Find the corresponding proofread text
                                    span_start = original_text.find(original_span_text)
                                    if span_start != -1:
                                        # Get the proofread version from our mapping
                                        proofread_span = original_span_text
                                        for pos, mapping in sorted(text_mapping.items()):
                                            if span_start <= pos < span_start + len(original_span_text):
                                                # Calculate the relative position within this span
                                                rel_pos = pos - span_start
                                                # Replace the text at this position
                                                proofread_span = (
                                                    proofread_span[:rel_pos] +
                                                    mapping['proofread'] +
                                                    proofread_span[rel_pos + mapping['length']:]
                                                )
                                        
                                        # Insert the proofread text with the same formatting
                                        font_name = get_font_name(span.get("font", "Helvetica"))
                                        try:
                                            new_page.insert_text(
                                                bbox.tl,  # Top-left point of the original text
                                                proofread_span,
                                                fontname=font_name,
                                                fontsize=span.get("size", 11),
                                                color=span.get("color", 0),
                                                render_mode=span.get("render_mode", 0)
                                            )
                                        except Exception as e:
                                            logger.warning(f"Font error with '{font_name}', falling back to Helvetica: {e}")
                                            new_page.insert_text(
                                                bbox.tl,
                                                proofread_span,
                                                fontname="Helvetica",
                                                fontsize=span.get("size", 11),
                                                color=span.get("color", 0),
                                                render_mode=span.get("render_mode", 0)
                                            )
            
            # Copy any other elements (like annotations, links, etc.)
            for annot in original_page.annots():
                new_page.insert_annot(annot)
            
            # Copy any links
            for link in original_page.get_links():
                new_page.insert_link(link)
        
        # Save the new PDF
        new_doc.save(pdf_path)
        new_doc.close()
        original_doc.close()
        
        logger.info("PDF saved successfully with original formatting and images preserved")
        
    except Exception as e:
        logger.error(f"Error creating PDF: {str(e)}")
        raise e

@app.route('/convert', methods=['POST'])
def convert_and_proofread():
    """Handles PDF proofreading while preserving formatting and images."""
    try:
        if 'file' not in request.files and 'text' not in request.form:
            return jsonify({"error": "No file or text provided"}), 400

        # Handle text and suggestions if provided directly
        if 'text' in request.form:
            proofread_text_content = request.form['text']
            original_filename = request.form.get('filename', 'document.pdf')
            
            # Store the original file path
            original_pdf_path = os.path.join(UPLOAD_FOLDER, original_filename)
            
            # Ensure the original file exists
            if not os.path.exists(original_pdf_path):
                base_filename = original_filename.replace("proofread_", "", 1)
                original_pdf_path = os.path.join(UPLOAD_FOLDER, base_filename)
                
                if not os.path.exists(original_pdf_path):
                    return jsonify({"error": "Original file not found"}), 404
            
            # Get selected suggestions
            selected_suggestions = {}
            if 'selected_suggestions' in request.form:
                try:
                    selected_suggestions = dict(json.loads(request.form['selected_suggestions']))
                except Exception as e:
                    logger.warning(f"Failed to parse selected suggestions: {str(e)}")

            # Generate PDF with the updated text while preserving formatting and images
            output_filename = "proofread_" + original_filename.replace("proofread_", "", 1)
            output_path = os.path.join(OUTPUT_FOLDER, output_filename)
            
            # Save the proofread PDF while preserving all formatting and images
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

        # Extract text with formatting information
        formatted_text = extract_text_with_formatting(pdf_path)
        extracted_text = " ".join([block['text'] for block in formatted_text])

        # Proofread the text
        proofread_text_content, grammar_errors = proofread_text(extracted_text)

        # Save proofread text back to PDF while preserving formatting and images
        proofread_pdf_filename = "proofread_" + filename
        proofread_pdf_path = os.path.join(OUTPUT_FOLDER, proofread_pdf_filename)
        
        # Create OUTPUT_FOLDER if it doesn't exist
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
        
        # Save the proofread PDF while preserving all formatting and images
        save_text_to_pdf(proofread_text_content, proofread_pdf_path, pdf_path)

        return jsonify({
            "original_text": extracted_text,
            "proofread_text": proofread_text_content,
            "grammar_errors": grammar_errors,
            "download_url": "/download/" + proofread_pdf_filename,
            "file_name": filename
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
