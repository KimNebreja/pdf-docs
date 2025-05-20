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
                
            # Check for indentation of the first line relative to the previous line
            if i > 0 and line['words'] and lines[i-1]['words']:
                 # Calculate the horizontal difference between the start of the current and previous line
                 horizontal_diff = line['words'][0]['x0'] - lines[i-1]['words'][0]['x0']
                 # Consider it a new paragraph if indented significantly more than previous line
                 if horizontal_diff > 10 and len(current_paragraph) > 0: # Threshold of 10 points
                     is_new_paragraph = True
                 # Also consider it a new paragraph if the previous line was indented more and this one less (outdent)
                 if horizontal_diff < -10 and len(current_paragraph) > 0: # Threshold of 10 points
                      is_new_paragraph = True


            # Check for different formatting (font, size, color) - this logic was commented out but is important
            # Re-adding basic check for formatting changes as a potential paragraph break
            if i > 0 and len(current_paragraph) > 0:
                prev_line_first_word = current_paragraph[-1]['words'][0] if current_paragraph[-1]['words'] else None
                curr_line_first_word = line['words'][0] if line['words'] else None

                if prev_line_first_word and curr_line_first_word:
                    # Check font name or size change
                    if (prev_line_first_word.get('fontname') != curr_line_first_word.get('fontname') or
                        abs(prev_line_first_word.get('size', 0) - curr_line_first_word.get('size', 0)) > 1): # Use smaller threshold
                        is_new_paragraph = True
                    # Basic color check (more precise color check is in get_text_color)
                    # You might need a more sophisticated color comparison here if needed
                    # if prev_line_first_word.get('color') != curr_line_first_word.get('color'):
                    #      is_new_paragraph = True


            # If this is a new paragraph, save the current one and start a new one
            if is_new_paragraph and current_paragraph:
                # Ensure the current paragraph is not just empty lines
                if any(line['text'].strip() for line in current_paragraph):
                    paragraphs.append({
                        'lines': current_paragraph,
                        'text': ' '.join([l['text'] for l in current_paragraph])
                    })
                current_paragraph = []
                
            # Add this line to the current paragraph
            current_paragraph.append(line)
            
        # Add the last paragraph if it's not empty
        if current_paragraph and any(line['text'].strip() for line in current_paragraph):
            paragraphs.append({
                'lines': current_paragraph,
                'text': ' '.join([l['text'] for l in current_paragraph])
            })
            
        return paragraphs
    except Exception as e:
        logger.warning(f"Error detecting paragraphs: {str(e)}")
        # Fallback: return all lines as a single paragraph if detection fails
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
                                # Convert integer color to hex string if needed
                                if isinstance(span["color"], int):
                                     # This is a heuristic, assuming typical sRGB integer colors
                                     return f'#{span["color"]:06x}'
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
            
        # If color is already a tuple/list of RGB values (0-1 range)
        if isinstance(color, (tuple, list)) and len(color) >= 3 and all(0.0 <= c <= 1.0 for c in color[:3]):
             return tuple(float(c) for c in color[:3]) # Return directly if already in 0-1 range

        # If color is a tuple/list of RGB values (0-255 range)
        if isinstance(color, (tuple, list)) and len(color) >= 3 and all(0 <= c <= 255 for c in color[:3]):
             return (float(color[0]) / 255, float(color[1]) / 255, float(color[2]) / 255)


        # If color is a single value (grayscale 0-255 or 0-1)
        if isinstance(color, (int, float)):
            val = float(color) / 255 if color > 1 else float(color)
            val = max(0.0, min(1.0, val))
            return (val, val, val)

        # If color is a hex string
        if isinstance(color, str) and color.startswith('#'):
            # Convert hex to RGB
            color = color.lstrip('#')
            if len(color) == 6:
                r = int(color[0:2], 16) / 255.0
                g = int(color[2:4], 16) / 255.0
                b = int(color[4:6], 16) / 255.0
                return (r, g, b)
            # Handle short hex colors (e.g., #RGB)
            if len(color) == 3:
                r = int(color[0]*2, 16) / 255.0
                g = int(color[1]*2, 16) / 255.0
                b = int(color[2]*2, 16) / 255.0
                return (r, g, b)


        # If color is a CMYK value
        if isinstance(color, (tuple, list)) and len(color) == 4:
            c, m, y, k = color
            # Convert CMYK to RGB
            # Ensure CMYK values are in 0-1 range
            c, m, y, k = max(0, min(1, c)), max(0, min(1, m)), max(0, min(1, y)), max(0, min(1, k))
            r = 1 - min(1, c * (1 - k) + k)
            g = 1 - min(1, m * (1 - k) + k)
            b = 1 - min(1, y * (1 - k) + k)
            return (r, g, b)

        # Default to black for unknown types
        logger.warning(f"Unknown color format: {color}")
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
    
    # Check if font name contains any of our mapped names (case-insensitive)
    for key in font_map:
        if key in font_name:
            return font_map[key]
    
    # Check for bold/italic variants (case-insensitive)
    is_bold = "bold" in font_name
    is_italic = "italic" in font_name or "oblique" in font_name

    if is_bold and is_italic:
        return "Times-BoldItalic" # Common fallback for bold italic
    elif is_bold:
        return "Helvetica-Bold" # Common fallback for bold
    elif is_italic:
        return "Times-Italic" # Common fallback for italic

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
        # This part depends on the system where the code is run
        # It's good to have, but failures here should not stop the process
        try:
            # Example: Registering Arial if available on Windows
            if os.name == 'nt': # Check if running on Windows
                arial_path = os.path.join(os.environ.get('WINDIR', ''), 'Fonts', 'arial.ttf')
                if os.path.exists(arial_path):
                    pdfmetrics.registerFont(TTFont('Arial', arial_path))
                    logger.info("Registered Arial font")
                    # Register Arial family if needed, mapping variants
                    pdfmetrics.registerFontFamily(
                        'Arial',
                        normal='Arial',
                        bold='Arial-Bold', # Assuming Arial Bold is also available
                        italic='Arial-Italic', # Assuming Arial Italic is also available
                        boldItalic='Arial-BoldItalic' # Assuming Arial Bold Italic is also available
                    )
            # Add checks for other OS or common font locations if necessary
            # e.g., for Linux: /usr/share/fonts/, /usr/local/share/fonts/

        except Exception as e:
            logger.warning(f"Could not register additional system fonts: {str(e)}")
            
    except Exception as e:
        logger.warning(f"Error registering fonts: {str(e)}")

def create_table(c, table_data, x, y, width, height, font, fontsize, fill_color):
    """
    Creates a table in the PDF using ReportLab.
    This function is not currently used in save_text_to_pdf but might be for future table rendering.
    """
    logger.warning("create_table function called but not fully implemented for ReportLab flowables.")
    # This function would need to be adapted to create ReportLab Table flowables
    # and add them to the story list for use with SimpleDocTemplate.
    # It cannot directly draw on the canvas when using SimpleDocTemplate for main content flow.
    pass # Placeholder

def extract_text_with_formatting(pdf_path):
    """
    Extracts text with detailed formatting information using pdfplumber.
    Returns a list of dictionaries with text and formatting details.
    Includes page number information.
    """
    try:
        result = []
        
        # Detect headers and footers - Note: This is currently detected but not used in rendering
        # header_footer_info = detect_headers_footers(pdf_path)
        
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                # Detect tables - Note: This is currently detected but not used in rendering
                # tables = detect_tables(page)
                
                # Detect columns - Note: This is currently detected but not used in rendering
                # columns = detect_columns(page)
                
                # Extract words with detailed formatting
                words = page.extract_words(
                    keep_blank_chars=True,
                    x_tolerance=2, # Reduced tolerance for potentially better alignment
                    y_tolerance=2, # Reduced tolerance for potentially better alignment
                    extra_attrs=['fontname', 'size', 'upright', 'color', 'x0', 'x1', 'top', 'bottom'] # Include coordinate and color info
                )
                
                # Group words into lines based on y-position
                lines = {}
                # Use a slightly larger tolerance for grouping words into lines if needed
                line_y_tolerance = 5 # Adjust this value if lines are not grouped correctly
                for word in words:
                    # Round to the nearest tolerance for grouping
                    y_pos_group = round(word['top'] / line_y_tolerance) * line_y_tolerance
                    if y_pos_group not in lines:
                        # Store min/max x0/x1 and average y for the line group
                        lines[y_pos_group] = {
                            'words': [],
                            'y_pos_group': y_pos_group,
                            'page': page_num,
                            'min_x0': float('inf'),
                            'max_x1': float('-inf'),
                            'avg_y': 0, # To calculate average vertical position
                            'line_count': 0 # To count words for average y calculation
                        }
                    lines[y_pos_group]['words'].append(word)
                    lines[y_pos_group]['min_x0'] = min(lines[y_pos_group]['min_x0'], word['x0'])
                    lines[y_pos_group]['max_x1'] = max(lines[y_pos_group]['max_x1'], word['x1'])
                    lines[y_pos_group]['avg_y'] += word['top'] # Use top as a reference for average y
                    lines[y_pos_group]['line_count'] += 1 # Count words for average y
                
                # Sort lines by y-position group (top to bottom)
                sorted_lines = sorted(lines.items(), key=lambda x: x[0])
                
                # Process each line group
                for y_pos_group, line_info in sorted_lines:
                    # Sort words in line group by x-position (left to right)
                    line_info['words'].sort(key=lambda x: x['x0'])
                    
                    # Calculate average y-position for the line
                    avg_y = line_info['avg_y'] / line_info['line_count'] if line_info['line_count'] > 0 else y_pos_group
                    
                    # Create a simplified line object with formatting summary
                    line_obj = {
                        'text': ' '.join([w['text'] for w in line_info['words']]),
                        'words': line_info['words'], # Keep original word data for detail
                        'y_pos': avg_y, # Use calculated average y-position
                        'page': page_num,
                        'min_x0': line_info['min_x0'],
                        'max_x1': line_info['max_x1'],
                        # Include formatting details from the first word as a summary for the line
                        'fontname': line_info['words'][0].get('fontname') if line_info['words'] else None,
                        'size': line_info['words'][0].get('size') if line_info['words'] else None,
                        'color': line_info['words'][0].get('color') if line_info['words'] else None,
                         'is_table': False, # Placeholder, need more sophisticated detection/handling
                         'is_column': False, # Placeholder
                         'is_header': False, # Placeholder
                         'is_footer': False # Placeholder
                    }
                    
                    # Note: Table, Column, Header, Footer detection info is currently not fully integrated
                    # into the line objects in a way that's used by the rendering part.
                    # If preserving these elements is critical, the rendering logic would need to be updated
                    # to create specific flowables (e.g., Table, HeaderFooter) instead of just Paragraphs.


                    result.append(line_obj)
        
        # Now that we have all lines with their simplified info, detect paragraphs across pages
        # Need to pass the full result list to detect_paragraphs
        paragraphs = detect_paragraphs(result)

        # Add paragraph information back to the result lines (for potential later use if needed)
        # This step is not strictly necessary for the current rendering logic with SimpleDocTemplate
        # but could be useful for debugging or future features.
        # for line in result:
        #     for i, paragraph in enumerate(paragraphs):
        #         if line in paragraph['lines']:
        #             line['paragraph_index'] = i
        #             break

        return paragraphs # Return the detected paragraphs directly

    except Exception as e:
        logger.error(f"Error extracting text with formatting: {str(e)}")
        raise

def save_text_to_pdf(text, pdf_path, original_pdf_path):
    """
    Saves proofread text to a new PDF file while preserving the exact formatting of the original PDF.
    """
    try:
        register_fonts()
        # Use standard letter size for output
        # Keep margins for the overall document structure
        doc_template = SimpleDocTemplate(
            pdf_path,
            pagesize=letter,
            leftMargin=72, rightMargin=72, topMargin=72, bottomMargin=72
        )
        page_width, page_height = letter
        with pdfplumber.open(original_pdf_path) as pdf:
            doc = fitz.open(original_pdf_path)
            # extract_text_with_formatting now returns paragraphs directly
            paragraphs_with_original_lines = extract_text_with_formatting(original_pdf_path)


            if isinstance(text, str):
                proofread_words = text.split()
            else:
                # Assuming text is already a list of corrected words/phrases or similar
                # Need to join it to match the structure of the text used for difflib
                proofread_text_combined = ' '.join(text)
                proofread_words = proofread_text_combined.split()


            original_words = []
            line_word_indices = []
            idx = 0
            # Build original words list and line indices from the extracted paragraph lines
            for para in paragraphs_with_original_lines:
                 for line in para['lines']:
                    words = line['text'].split()
                    start = idx
                    idx += len(words)
                    end = idx
                    line_word_indices.append((start, end))
                    original_words.extend(words)

            sm = difflib.SequenceMatcher(None, original_words, proofread_words)
            corrected_words = list(original_words) # Start with original
            opcodes = sm.get_opcodes()
            for tag, i1, i2, j1, j2 in opcodes:
                if tag in ('replace', 'insert'):
                    corrected_words[i1:i2] = proofread_words[j1:j2]
                elif tag == 'delete':
                    corrected_words[i1:i2] = [] # Remove deleted words

            # Reconstruct corrected lines based on the original line structure indices
            corrected_lines = []
            for start, end in line_word_indices:
                corrected_lines.append(' '.join(corrected_words[start:end]))


            # --- Determine the absolute minimum x0 across ALL paragraphs ---
            # This will be our most reliable baseline for zero indentation
            absolute_min_x0_all_paragraphs = float('inf')
            for para in paragraphs_with_original_lines:
                 if para['lines']:
                      # Find the minimum x0 within this paragraph's lines
                      para_min_x0 = float('inf')
                      for line in para['lines']:
                           if line['words']:
                                para_min_x0 = min(para_min_x0, line['words'][0]['x0'])
                      if para_min_x0 != float('inf'):
                           absolute_min_x0_all_paragraphs = min(absolute_min_x0_all_paragraphs, para_min_x0)

            # Use doc template margin as a fallback if no text found
            if absolute_min_x0_all_paragraphs == float('inf'):
                 absolute_min_x0_all_paragraphs = doc_template.leftMargin


            story = []

            # Iterate through the detected paragraphs (which are already grouped by original page implicitly by extract_text_with_formatting)
            current_page_num = None
            for para_idx, para in enumerate(paragraphs_with_original_lines):
                 para_lines = para['lines']
                 if not para_lines:
                      continue

                 # Get the original page number for this paragraph
                 original_page_num = para_lines[0]['page']

                 # Add a page break if the page number changes, except before the first page
                 if current_page_num is not None and original_page_num != current_page_num:
                      story.append(PageBreak())
                      # Add space at the top of the new page if needed (optional, handled by topMargin)
                      # story.append(Spacer(1, doc_template.topMargin)) # Example

                 current_page_num = original_page_num # Update current page number


                 para_text = ' '.join([corrected_lines[l['line_index']] for l in para_lines]) # Get corrected text for the paragraph

                 # Calculate the minimum x0 and maximum x1 for this paragraph from its original lines
                 min_x0 = float('inf')
                 max_x1 = float('-inf')
                 has_words = False
                 for line in para_lines:
                     if line['words']:
                          has_words = True
                          min_x0 = min(min_x0, line['words'][0]['x0'])
                          max_x1 = max(max_x1, line['words'][-1]['x0'] + line['words'][-1]['width'])

                 if not has_words:
                      continue # Skip if the paragraph has no words after all

                 first_word = para_lines[0]['words'][0] if para_lines[0]['words'] else None
                 if not first_word:
                      continue # Should not happen if has_words is true, but as a safeguard

                 font_name = get_font_name(first_word.get('fontname', 'Helvetica'))
                 font_size = float(first_word.get('size', 11))
                 mupdf_page = doc[original_page_num] # Use original page num for color detection

                 # Use the bounding box of the first line for color detection (simplification)
                 bbox = fitz.Rect(first_word['x0'], first_word['top'], first_word['x0'] + first_word['width'], first_word['bottom'])
                 color = get_text_color(mupdf_page, bbox)
                 r, g, b = normalize_color(color) if color else (0, 0, 0)


                 # Calculate left indent relative to the absolute minimum x0 found in the document
                 # This is the additional indent beyond the document's standard left margin
                 # Ensure indent is non-negative
                 left_indent = max(0, min_x0 - absolute_min_x0_all_paragraphs)

                 # Calculate right indent from the right edge of the text block to the right margin
                 # Use page_width based on the template, and subtract document right margin and paragraph's width from the right
                 # This calculation needs to be careful about the reference point
                 # A simpler way might be to calculate the width and use that with left indent
                 # Let's calculate the intended width based on original max_x1 and min_x0
                 intended_width = max_x1 - min_x0
                 # The right indent will then be the usable width minus the left indent and intended width
                 usable_width = page_width - doc_template.leftMargin - doc_template.rightMargin
                 calculated_right_indent = max(0, usable_width - left_indent - intended_width)


                 # Calculate first line indent if it differs from the paragraph's overall left indent
                 # This is relative to the paragraph's calculated left indent
                 first_line_x0 = para_lines[0]['words'][0]['x0'] if para_lines[0]['words'] else min_x0
                 calculated_first_line_indent = max(0, first_line_x0 - min_x0)


                 # Determine if there's a significant first line indent vs a block indent
                 # Use a heuristic threshold
                 indent_threshold = font_size * 0.2 # Further reduced threshold


                 # Check if there's a first line indent significantly different from zero (relative to paragraph's min_x0)
                 if calculated_first_line_indent > indent_threshold and abs(calculated_first_line_indent - left_indent) > indent_threshold:
                      # If the first line indent is significantly different from the block indent, apply it
                      style = ParagraphStyle(
                           name='JustifiedWithComplexIndent',
                           fontName=font_name,
                           fontSize=font_size,
                           leading=font_size * 1.2,
                           textColor=Color(r, g, b),
                           alignment=TA_JUSTIFY,
                           spaceAfter=font_size * 0.5,
                           spaceBefore=0,
                           leftIndent=doc_template.leftMargin + left_indent, # Base margin + block indent
                           rightIndent=calculated_right_indent, # Apply calculated right indent
                           firstLineIndent=calculated_first_line_indent, # Additional indent for the first line
                      )
                 else:
                     # Otherwise, assume it's a regular block indent or no indent
                      style = ParagraphStyle(
                           name='JustifiedWithBlockIndent',
                           fontName=font_name,
                           fontSize=font_size,
                           leading=font_size * 1.2,
                           textColor=Color(r, g, b),
                           alignment=TA_JUSTIFY,
                           spaceAfter=font_size * 0.5,
                           spaceBefore=0,
                           leftIndent=doc_template.leftMargin + left_indent, # Base margin + calculated left indent
                           rightIndent=calculated_right_indent, # Apply the calculated right indent
                           firstLineIndent=0, # No special first line indent
                      )


                 para_obj = Paragraph(para_text, style)
                 story.append(para_obj)
                 # Add space after the paragraph
                 # Only add spacer if it's not the last paragraph on the original page
                 # This is tricky because Platypus handles flow, but adding a small space after each
                 # paragraph based on font size is a reasonable default.
                 # If precise vertical spacing from original PDF is needed, it's much more complex.
                 # story.append(Spacer(1, style.spaceAfter)) # Keep existing spacer for consistency


            # Add page numbers using onPage callback
            def add_page_number(canvas, doc):
                # Position page number at the bottom center
                canvas.drawCentredString(page_width/2, 30, str(doc.page))

            # Build the document with the story and page number callback
            # Pass the canvas and document to the callback for positioning
            doc_template.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)


            doc.close()
            logger.info("PDF saved successfully with original formatting")
    except Exception as e:
        logger.error(f"Error creating PDF: {str(e)}")
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
                    # Need to be careful here to apply suggestions to the proofread_text_content correctly
                    # This might need a more sophisticated replacement that considers word boundaries
                    # For simplicity now, doing a basic replace, but could lead to issues
                    for original_word, selected_word in selected_suggestions.items():
                         # Use regex to replace whole words only
                         proofread_text_content = re.sub(r'\b' + re.escape(original_word) + r'\b', selected_word, proofread_text_content)

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
            "proofread_text": proofread_text_content, # Return the combined string text
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
