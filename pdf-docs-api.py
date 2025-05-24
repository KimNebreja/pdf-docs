from flask import Flask, request, send_file, jsonify
from werkzeug.utils import secure_filename
import os
import language_tool_python
import spacy
from flask_cors import CORS
import fitz  # PyMuPDF
import pdfplumber
from reportlab.pdfgen import canvas
from reportlab.lib.colors import Color
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Frame, PageBreak, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
import io
import logging
import re
import json
import numpy as np
from collections import defaultdict
import difflib
from reportlab.lib.enums import TA_JUSTIFY
import tempfile

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "/tmp"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Initialize language tools
tool = language_tool_python.LanguageToolPublicAPI('en-US')
nlp = spacy.load("en_core_web_sm")

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

def apply_custom_grammar_rules(text):
    """
    Applies custom grammar rules using spaCy and regex patterns to detect additional grammar issues.
    Includes enhanced auxiliary verb checking, verb phrase analysis, and contraction handling.
    """
    issues = []
    lines = text.split('\n')
    sentence_end = re.compile(r"[.!?]$")
    passive_voice = re.compile(r"\b(be|is|are|was|were|been|being)\s+\w+ed\b", re.IGNORECASE)
    double_negative = re.compile(r"\b(not|never|no)\b.*\b(none|nothing|nowhere|no one|n't)\b", re.IGNORECASE)

    # Auxiliary verb patterns with their contractions
    aux_verb_patterns = {
        'singular': {
            'is': ["is", "isn't", "'s"],
            'was': ["was", "wasn't"],
            'has': ["has", "hasn't", "'s"],
            'does': ["does", "doesn't"]
        },
        'plural': {
            'are': ["are", "aren't", "'re"],
            'were': ["were", "weren't"],
            'have': ["have", "haven't", "'ve"],
            'do': ["do", "don't"]
        },
        'modal': {
            'can': ["can", "can't", "cannot"],
            'could': ["could", "couldn't"],
            'may': ["may", "mayn't"],
            'might': ["might", "mightn't"],
            'shall': ["shall", "shan't"],
            'should': ["should", "shouldn't"],
            'will': ["will", "won't", "'ll"],
            'would': ["would", "wouldn't", "'d"],
            'must': ["must", "mustn't"]
        },
        'perfect': {
            'have': ["have", "haven't", "'ve"],
            'has': ["has", "hasn't", "'s"],
            'had': ["had", "hadn't", "'d"]
        },
        'progressive': {
            'am': ["am", "ain't", "'m"],
            'is': ["is", "isn't", "'s"],
            'are': ["are", "aren't", "'re"],
            'was': ["was", "wasn't"],
            'were': ["were", "weren't"],
            'be': ["be"],
            'been': ["been"],
            'being': ["being"]
        }
    }

    # Subject-verb agreement patterns with special cases
    singular_subjects = {
        'pronouns': ['he', 'she', 'it', 'this', 'that'],
        'indefinite': ['each', 'every', 'either', 'neither', 'one'],
        'compound': ['anyone', 'everyone', 'someone', 'nobody', 'anybody', 'somebody', 'everybody'],
        'collective': ['team', 'group', 'committee', 'family', 'class', 'company', 'organization', 'government'],
        'quantities': ['each of', 'every one of', 'either of', 'neither of', 'one of']
    }
    
    plural_subjects = {
        'pronouns': ['they', 'we', 'you', 'these', 'those'],
        'quantifiers': ['both', 'few', 'many', 'several'],
        'collective': ['teams', 'groups', 'committees', 'families', 'classes', 'companies', 'organizations', 'governments']
    }

    # Contraction patterns for special cases
    contraction_patterns = {
        'is': {
            'singular': ["he's", "she's", "it's", "that's", "this's"],
            'plural': ["they're", "we're", "you're", "these're", "those're"]
        },
        'has': {
            'singular': ["he's", "she's", "it's", "that's", "this's"],
            'plural': ["they've", "we've", "you've"]
        },
        'have': {
            'singular': ["he's", "she's", "it's", "that's", "this's"],
            'plural': ["they've", "we've", "you've"]
        }
    }

    def get_auxiliary_forms(aux_type, aux_word):
        """Helper function to get all forms of an auxiliary verb."""
        for category in aux_verb_patterns.values():
            if aux_word in category:
                return category[aux_word]
        return [aux_word]

    def check_contraction_agreement(token, subject, aux_type):
        """Helper function to check contraction agreement with subject."""
        if not subject or not token:
            return None
            
        subject_text = subject.text.lower()
        token_text = token.text.lower()
        
        # Check for special contraction cases
        if aux_type in contraction_patterns:
            if subject_text in singular_subjects['pronouns']:
                if token_text in contraction_patterns[aux_type]['plural']:
                    return f"Use '{contraction_patterns[aux_type]['singular'][singular_subjects['pronouns'].index(subject_text)]}' for singular subject."
            elif subject_text in plural_subjects['pronouns']:
                if token_text in contraction_patterns[aux_type]['singular']:
                    return f"Use '{contraction_patterns[aux_type]['plural'][plural_subjects['pronouns'].index(subject_text)]}' for plural subject."
        return None

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue

        # Basic sentence structure checks
        if stripped and not sentence_end.search(stripped):
            issues.append({
                "message": "Incomplete or improperly punctuated sentence.",
                "suggestions": ["Ensure the sentence ends with proper punctuation."],
                "offset": 0,
                "length": len(stripped)
            })

        if stripped and not stripped[0].isupper():
            issues.append({
                "message": "Sentence does not start with a capital letter.",
                "suggestions": ["Capitalize the first word of the sentence."],
                "offset": 0,
                "length": len(stripped)
            })

        # Advanced NLP analysis using spaCy
        doc = nlp(stripped)
        
        # Track verb phrases and their components
        verb_phrases = []
        current_vp = []
        
        for token in doc:
            # Collect verb phrases including contractions
            if token.pos_ in ["VERB", "AUX"] or (token.pos_ == "PART" and token.text == "n't"):
                current_vp.append(token)
            elif current_vp:
                verb_phrases.append(current_vp)
                current_vp = []
        
        if current_vp:
            verb_phrases.append(current_vp)

        # Analyze each verb phrase
        for vp in verb_phrases:
            # Get the main verb and its auxiliaries
            auxiliaries = [t for t in vp if t.pos_ == "AUX" or (t.pos_ == "PART" and t.text == "n't")]
            main_verb = next((t for t in vp if t.pos_ == "VERB" and t.text != "n't"), None)
            
            if not main_verb and not auxiliaries:
                continue

            # Check subject-verb agreement
            subject = None
            for token in doc:
                if token.dep_ in ["nsubj", "nsubjpass"] and token.head in vp:
                    subject = token
                    break

            if subject and auxiliaries:
                # Check auxiliary verb agreement
                aux = auxiliaries[0]
                aux_text = aux.text.lower()
                
                # Check for contraction agreement
                contraction_suggestion = None
                for aux_type in ['is', 'has', 'have']:
                    if aux_type in contraction_patterns:
                        suggestion = check_contraction_agreement(aux, subject, aux_type)
                        if suggestion:
                            contraction_suggestion = suggestion
                            break

                if contraction_suggestion:
                    issues.append({
                        "message": f"Contraction agreement error with '{aux_text}'.",
                        "suggestions": [contraction_suggestion],
                        "offset": aux.idx,
                        "length": len(aux_text)
                    })

                # Check for don't/doesn't agreement
                if aux_text in ["don't", "doesn't"]:
                    subject_text = subject.text.lower()
                    is_singular = any(subject_text in category for category in singular_subjects.values())
                    if is_singular and aux_text == "don't":
                        issues.append({
                            "message": f"Use 'doesn't' with singular subject '{subject.text}'.",
                            "suggestions": ["Use 'doesn't' for singular subjects."],
                            "offset": aux.idx,
                            "length": len(aux_text)
                        })
                    elif not is_singular and aux_text == "doesn't":
                        issues.append({
                            "message": f"Use 'don't' with plural subject '{subject.text}'.",
                            "suggestions": ["Use 'don't' for plural subjects."],
                            "offset": aux.idx,
                            "length": len(aux_text)
                        })

                # Check for didn't agreement
                if aux_text == "didn't":
                    if subject.text.lower() in singular_subjects['pronouns']:
                        issues.append({
                            "message": f"Consider using 'wasn't' or 'hadn't' with singular subject '{subject.text}'.",
                            "suggestions": ["Use appropriate past tense auxiliary based on context."],
                            "offset": aux.idx,
                            "length": len(aux_text)
                        })

            # Check modal verb usage with contractions
            for aux in auxiliaries:
                aux_text = aux.text.lower()
                if aux_text in ["can't", "cannot", "couldn't", "won't", "wouldn't", "shouldn't", "mustn't"]:
                    if main_verb and main_verb.tag_ not in ["VB"]:
                        issues.append({
                            "message": f"Modal verb contraction '{aux_text}' should be followed by base form of verb.",
                            "suggestions": [f"Use base form after '{aux_text}'."],
                            "offset": main_verb.idx,
                            "length": len(main_verb.text)
                        })

            # Check perfect tense construction with contractions
            if any(aux.text.lower() in ["'ve", "'s", "haven't", "hasn't", "hadn't"] for aux in auxiliaries):
                if main_verb and main_verb.tag_ not in ["VBN"]:
                    issues.append({
                        "message": "Perfect tense requires past participle form.",
                        "suggestions": ["Use past participle form after perfect tense auxiliaries."],
                        "offset": main_verb.idx,
                        "length": len(main_verb.text)
                    })

            # Check progressive tense construction with contractions
            if any(aux.text.lower() in ["'m", "'s", "'re", "isn't", "aren't", "wasn't", "weren't"] for aux in auxiliaries):
                if main_verb and main_verb.tag_ not in ["VBG"]:
                    issues.append({
                        "message": "Progressive tense requires present participle form.",
                        "suggestions": ["Use present participle form (-ing) after progressive tense auxiliaries."],
                        "offset": main_verb.idx,
                        "length": len(main_verb.text)
                    })

            # Check for double auxiliaries with contractions
            if len(auxiliaries) > 1:
                aux_texts = [aux.text.lower() for aux in auxiliaries]
                modal_contractions = ["can't", "couldn't", "won't", "wouldn't", "shouldn't", "mustn't"]
                if any(aux in modal_contractions for aux in aux_texts) and \
                   any(aux in modal_contractions for aux in aux_texts):
                    issues.append({
                        "message": "Multiple modal verb contractions detected.",
                        "suggestions": ["Use only one modal verb contraction in a verb phrase."],
                        "offset": auxiliaries[1].idx,
                        "length": len(auxiliaries[1].text)
                    })

        # Check for passive voice with contractions
        passive_with_contractions = re.compile(r"\b(isn't|aren't|wasn't|weren't)\s+\w+ed\b", re.IGNORECASE)
        if passive_with_contractions.search(line):
            issues.append({
                "message": "Passive voice with contraction detected.",
                "suggestions": ["Consider rewriting in active voice for clarity."],
                "offset": 0,
                "length": len(line)
            })

        # Check for double negatives with contractions
        double_negative_with_contractions = re.compile(r"\b(not|never|no|n't)\b.*\b(none|nothing|nowhere|no one|n't)\b", re.IGNORECASE)
        if double_negative_with_contractions.search(line):
            issues.append({
                "message": "Double negative with contraction detected.",
                "suggestions": ["Revise to remove the double negative for clarity."],
                "offset": 0,
                "length": len(line)
            })

    return issues

def proofread_text(text):
    """Proofreads text using LanguageTool and custom grammar rules, returns corrected text with details."""
    try:
        # Get LanguageTool matches
        matches = tool.check(text)
        corrected_text = language_tool_python.utils.correct(text, matches)

        # Collect LanguageTool grammar mistakes
        lt_errors = [{
            "message": match.message,
            "suggestions": match.replacements,
            "offset": match.offset,
            "length": match.errorLength
        } for match in matches]

        # Get custom grammar rule issues
        custom_errors = apply_custom_grammar_rules(text)
        
        # Combine all errors
        all_errors = lt_errors + custom_errors

        return corrected_text, all_errors
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

def save_text_to_pdf(text, pdf_path, original_pdf_path, proofread_text_content=None, selected_suggestions=None):
    """
    Saves proofread text to a new PDF file, re-justifying each line after proofreading so the line is always justified even if the text changes.
    """
    try:
        from reportlab.pdfgen import canvas as rl_canvas
        from reportlab.lib.pagesizes import letter
        import tempfile
        import difflib
        register_fonts()
        # Apply selected suggestions to the proofread text before mapping spans
        if selected_suggestions:
            for original_word, selected_word in selected_suggestions.items():
                text = text.replace(original_word, selected_word)
        doc = fitz.open(original_pdf_path)
        all_original_spans = []
        all_span_meta = []
        for page_num in range(len(doc)):
            mupdf_page = doc[page_num]
            blocks = mupdf_page.get_text("dict")["blocks"]
            for block in blocks:
                if "lines" in block:
                    for line in block["lines"]:
                        if "spans" in line:
                            for span in line["spans"]:
                                all_original_spans.append(span["text"])
                                all_span_meta.append((page_num, span))
        proofread_spans = text.split()
        original_spans_flat = ' '.join(all_original_spans).split()
        sm = difflib.SequenceMatcher(None, original_spans_flat, proofread_spans)
        aligned_spans = []
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == 'equal':
                aligned_spans.extend(proofread_spans[j1:j2])
            elif tag in ('replace', 'insert'):
                aligned_spans.extend(proofread_spans[j1:j2])
            elif tag == 'delete':
                continue
        c = rl_canvas.Canvas(pdf_path, pagesize=letter)
        page_width, page_height = letter
        span_word_idx = 0
        for page_num in range(len(doc)):
            mupdf_page = doc[page_num]
            image_list = mupdf_page.get_images(full=True)
            for img in image_list:
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                bbox = mupdf_page.get_image_bbox(img)
                temp_img = tempfile.NamedTemporaryFile(delete=False, suffix=f'.{image_ext}')
                temp_img.write(image_bytes)
                temp_img.close()
                x0, y0, x1, y1 = bbox
                img_width = x1 - x0
                img_height = y1 - y0
                y0_rl = page_height - y1
                try:
                    c.drawImage(temp_img.name, x0, y0_rl, width=img_width, height=img_height)
                except Exception as e:
                    logger.warning(f"Could not draw image at ({x0},{y0_rl}): {e}")
            blocks = mupdf_page.get_text("dict")["blocks"]
            for block in blocks:
                if "lines" in block:
                    for line in block["lines"]:
                        if "spans" in line:
                            # Gather all span info for this line
                            line_spans = line["spans"]
                            # Calculate available width for justification
                            x0s = [span["bbox"][0] for span in line_spans]
                            x1s = [span["bbox"][2] for span in line_spans]
                            line_x0 = min(x0s)
                            line_x1 = max(x1s)
                            # Prepare the full line text from aligned spans
                            line_words = []
                            for span in line_spans:
                                original_words = span["text"].split()
                                n_words = len(original_words)
                                draw_words = aligned_spans[span_word_idx:span_word_idx + n_words]
                                line_words.extend(draw_words if draw_words else original_words)
                                span_word_idx += n_words
                            # Calculate y position (use first span's y)
                            y = page_height - line_spans[0]["bbox"][1]
                            font_name = get_font_name(line_spans[0].get("font", "Helvetica"))
                            font_size = line_spans[0].get("size", 11)
                            c.setFont(font_name, font_size)
                            # Calculate text width
                            total_word_width = sum([c.stringWidth(w, font_name, font_size) for w in line_words])
                            n_spaces = len(line_words) - 1
                            available_width = line_x1 - line_x0
                            if n_spaces > 0 and total_word_width < available_width:
                                # Justify: distribute extra space between words
                                extra_space = (available_width - total_word_width) / n_spaces
                                x = line_x0
                                for i, word in enumerate(line_words):
                                    c.drawString(x, y, word)
                                    word_width = c.stringWidth(word, font_name, font_size)
                                    if i < n_spaces:
                                        x += word_width + extra_space
                            else:
                                # Left align if only one word or no extra space
                                x = line_x0
                                c.drawString(x, y, ' '.join(line_words))
            if page_num < len(doc) - 1:
                c.showPage()
        c.save()
        doc.close()
        logger.info("PDF saved successfully with re-justified lines after proofreading.")
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
            save_text_to_pdf(proofread_text_content, output_path, original_pdf_path, selected_suggestions=selected_suggestions)
            
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

@app.route('/proofread', methods=['POST'])
def proofread():
    """Handles direct text proofreading without PDF conversion."""
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({"error": "No text provided"}), 400

        text = data.get("text", "")
        corrected_text, errors = proofread_text(text)
        
        return jsonify({
            "corrected_text": corrected_text,
            "errors": errors
        })
    except Exception as e:
        logger.error(f"Proofreading error: {str(e)}")
        return jsonify({"error": f"Proofreading error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000, debug=True)
