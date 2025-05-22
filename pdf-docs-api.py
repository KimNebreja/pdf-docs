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

# --- Helper Functions ---
def register_fonts():
    pdfmetrics.registerFontFamily(
        'Helvetica',
        normal='Helvetica',
        bold='Helvetica-Bold',
        italic='Helvetica-Oblique',
        boldItalic='Helvetica-BoldOblique'
    )

def extract_text_from_pdf(pdf_path):
    with fitz.open(pdf_path) as doc:
        return "\n".join([page.get_text() for page in doc])

def extract_text_with_formatting(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        result = []
        for page_num, page in enumerate(pdf.pages):
            words = page.extract_words(keep_blank_chars=True, extra_attrs=['fontname', 'size'])
            lines = defaultdict(list)
            for w in words:
                lines[round(w['top'], 1)].append(w)
            for y_pos, words_line in sorted(lines.items()):
                words_line.sort(key=lambda x: x['x0'])
                result.append({
                    'text': ' '.join(w['text'] for w in words_line),
                    'words': words_line,
                    'y_pos': y_pos,
                    'page': page_num
                })
        return result

def detect_paragraphs(lines):
    paragraphs = []
    para = []
    for i, line in enumerate(lines):
        if para and line['y_pos'] - para[-1]['y_pos'] > 15:
            paragraphs.append({'lines': para})
            para = []
        para.append(line)
    if para:
        paragraphs.append({'lines': para})
    return paragraphs

def get_font_name(font_name):
    if not font_name:
        return 'Helvetica'
    font_name = font_name.lower()
    if 'bold' in font_name and 'italic' in font_name:
        return 'Times-BoldItalic'
    elif 'bold' in font_name:
        return 'Helvetica-Bold'
    elif 'italic' in font_name:
        return 'Times-Italic'
    return 'Helvetica'

def get_text_color(page, bbox):
    try:
        spans = page.get_text("dict", clip=bbox).get("blocks", [])
        for block in spans:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    return span.get("color") or span.get("fill") or span.get("stroke")
    except Exception:
        pass
    return None

def normalize_color(color):
    if isinstance(color, (list, tuple)) and len(color) >= 3:
        return tuple(min(1.0, max(0.0, float(c) / 255)) for c in color[:3])
    if isinstance(color, (int, float)):
        val = min(1.0, max(0.0, float(color) / 255))
        return (val, val, val)
    return (0, 0, 0)

def detect_text_alignment(page, line_words):
    if not line_words:
        return 'JUSTIFY'
    page_width = page.width
    left_margin = 72
    right_margin = page_width - 72
    line_start = line_words[0]['x0']
    line_end = line_words[-1]['x0'] + line_words[-1]['width']
    center = (left_margin + right_margin) / 2
    line_center = (line_start + line_end) / 2
    if abs(line_center - center) < 15:
        return 'CENTER'
    if abs(line_end - right_margin) < 20:
        return 'RIGHT'
    if abs(line_start - left_margin) < 20:
        return 'LEFT'
    return 'JUSTIFY'

def proofread_text(text):
    tool = language_tool_python.LanguageToolPublicAPI('en-US')
    matches = tool.check(text)
    corrected = language_tool_python.utils.correct(text, matches)
    return corrected, [
        {
            "message": m.message,
            "suggestions": m.replacements,
            "offset": m.offset,
            "length": m.errorLength
        } for m in matches
    ]

def save_text_to_pdf(text, pdf_path, original_pdf_path):
    register_fonts()
    page_width, page_height = letter
    doc_template = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        leftMargin=72, rightMargin=72, topMargin=72, bottomMargin=72
    )
    doc = fitz.open(original_pdf_path)
    formatted_lines = extract_text_with_formatting(original_pdf_path)
    proofread_words = text.split()
    original_words = []
    line_word_indices = []
    idx = 0
    for line in formatted_lines:
        words = line['text'].split()
        start = idx
        idx += len(words)
        end = idx
        line_word_indices.append((start, end))
        original_words.extend(words)
    sm = difflib.SequenceMatcher(None, original_words, proofread_words)
    corrected_words = list(original_words)
    opcodes = sm.get_opcodes()
    for tag, i1, i2, j1, j2 in opcodes:
        if tag in ('replace', 'insert'):
            corrected_words[i1:i2] = proofread_words[j1:j2]
        elif tag == 'delete':
            corrected_words[i1:i2] = []
    corrected_lines = []
    for start, end in line_word_indices:
        corrected_lines.append(' '.join(corrected_words[start:end]))
    lines_by_page = defaultdict(list)
    for i, line in enumerate(formatted_lines):
        line = dict(line)
        line['line_index'] = i
        lines_by_page[line.get('page', 0)].append(line)
    story = []
    page_numbers = sorted(lines_by_page.keys())
    with pdfplumber.open(original_pdf_path) as pdf:
        for page_idx, page_num in enumerate(page_numbers):
            page_lines = lines_by_page[page_num]
            paragraphs = detect_paragraphs(page_lines)
            for para in paragraphs:
                para_lines = para['lines']
                para_indices = [l['line_index'] for l in para_lines]
                para_text = ' '.join([corrected_lines[i] for i in para_indices])
                if not para_lines:
                    continue
                first_word = para_lines[0]['words'][0]
                font_name = get_font_name(first_word.get('fontname', 'Helvetica'))
                font_size = float(first_word.get('size', 11))
                mupdf_page = doc[para_lines[0]['page']]
                bbox = fitz.Rect(first_word['x0'], first_word['top'], first_word['x0'] + first_word['width'], first_word['bottom'])
                color = get_text_color(mupdf_page, bbox)
                r, g, b = normalize_color(color) if color else (0, 0, 0)
                alignment = detect_text_alignment(pdf.pages[para_lines[0]['page']], para_lines[0]['words'])
                alignment_map = {
                    'LEFT': TA_LEFT,
                    'CENTER': TA_CENTER,
                    'RIGHT': TA_RIGHT,
                    'JUSTIFY': TA_JUSTIFY
                }
                alignment_value = alignment_map.get(alignment, TA_JUSTIFY)
                style = ParagraphStyle(
                    name=f'Custom_{page_num}_{len(story)}',
                    fontName=font_name,
                    fontSize=font_size,
                    leading=font_size * 1.2,
                    textColor=Color(r, g, b),
                    alignment=alignment_value,
                    spaceAfter=font_size * 0.5
                )
                para_obj = Paragraph(para_text, style)
                story.append(para_obj)
                story.append(Spacer(1, font_size * 0.5))
            if page_idx < len(page_numbers) - 1:
                story.append(PageBreak())
        def on_page(canvas, doc):
            try:
                page_idx = doc.page - 1
                if page_idx < len(doc):
                    pdf_doc = fitz.open(original_pdf_path)
                    pdf_page = pdf_doc[page_idx]
                    for img in pdf_page.get_images(full=True):
                        xref = img[0]
                        base_image = pdf_doc.extract_image(xref)
                        pix = fitz.Pixmap(pdf_doc, xref)
                        if pix.n > 4:
                            pix = fitz.Pixmap(fitz.csRGB, pix)
                        img_io = io.BytesIO(pix.tobytes("png"))
                        canvas.drawImage(img_io, img[1], page_height - img[4], width=img[3] - img[1], height=img[4] - img[2])
            except Exception as e:
                logging.warning(f"Image rendering failed: {e}")
            canvas.setFont("Helvetica", 10)
            canvas.drawString(page_width / 2, 30, str(doc.page))
        doc_template.build(story, onFirstPage=on_page, onLaterPages=on_page)
        doc.close()

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "/tmp"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route('/convert', methods=['POST'])
def convert_and_proofread():
    try:
        if 'file' not in request.files and 'text' not in request.form:
            return jsonify({"error": "No file or text provided"}), 400

        if 'text' in request.form:
            proofread_text_content = request.form['text']
            original_filename = request.form.get('filename', 'document.pdf')
            original_pdf_path = os.path.join(UPLOAD_FOLDER, original_filename)
            if not os.path.exists(original_pdf_path):
                base_filename = original_filename.replace("proofread_", "", 1)
                original_pdf_path = os.path.join(UPLOAD_FOLDER, base_filename)
                if not os.path.exists(original_pdf_path):
                    return jsonify({"error": "Original file not found"}), 404

            selected_suggestions = {}
            if 'selected_suggestions' in request.form:
                try:
                    selected_suggestions = dict(json.loads(request.form['selected_suggestions']))
                    for original_word, selected_word in selected_suggestions.items():
                        proofread_text_content = proofread_text_content.replace(original_word, selected_word)
                except Exception as e:
                    logging.warning(f"Failed to parse selected suggestions: {str(e)}")

            output_filename = "proofread_" + original_filename.replace("proofread_", "", 1)
            output_path = os.path.join(OUTPUT_FOLDER, output_filename)
            save_text_to_pdf(proofread_text_content, output_path, original_pdf_path)

            return jsonify({
                "download_url": "/download/" + output_filename
            })

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        filename = secure_filename(file.filename)
        pdf_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(pdf_path)

        extracted_text = extract_text_from_pdf(pdf_path)
        proofread_text_content, grammar_errors = proofread_text(extracted_text)

        proofread_pdf_filename = "proofread_" + filename
        proofread_pdf_path = os.path.join(OUTPUT_FOLDER, proofread_pdf_filename)
        save_text_to_pdf(proofread_text_content, proofread_pdf_path, pdf_path)

        return jsonify({
            "original_text": extracted_text,
            "proofread_text": proofread_text_content,
            "grammar_errors": grammar_errors,
            "download_url": "/download/" + proofread_pdf_filename,
            "file_name": filename
        })

    except Exception as e:
        logging.error(f"Conversion error: {str(e)}")
        return jsonify({"error": f"Conversion error: {str(e)}"}), 500

@app.route('/download/<filename>')
def download_file(filename):
    file_path = os.path.join(OUTPUT_FOLDER, filename)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    return jsonify({"error": "File not found"}), 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000, debug=True)
  
