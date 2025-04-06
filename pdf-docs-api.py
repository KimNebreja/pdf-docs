from flask import Flask, request, send_file, jsonify
from werkzeug.utils import secure_filename
import os
from pdf2docx import Converter
from flask_cors import CORS
import language_tool_python
import docx
import re
import logging

app = Flask(__name__)
CORS(app)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "/tmp"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Initialize LanguageTool with error handling
try:
    tool = language_tool_python.LanguageToolPublicAPI('en-US')  # English language
    language_tool_available = True
    logger.info("LanguageTool API initialized successfully")
except Exception as e:
    language_tool_available = False
    logger.error(f"Failed to initialize LanguageTool API: {str(e)}")

def extract_text_from_docx(docx_path):
    doc = docx.Document(docx_path)
    text_with_formatting = []
    
    for para in doc.paragraphs:
        # Get paragraph text
        para_text = para.text
        
        # Check for paragraph formatting
        if para.alignment == docx.enum.text.WD_ALIGN_PARAGRAPH.CENTER:
            # Add spaces to indicate centered text
            para_text = " " * 10 + para_text + " " * 10
        elif para.alignment == docx.enum.text.WD_ALIGN_PARAGRAPH.RIGHT:
            # Add spaces to indicate right-aligned text
            para_text = " " * 20 + para_text
        
        # Check for bold text
        for run in para.runs:
            if run.bold:
                # Mark bold text with **
                run_text = run.text
                if run_text:
                    para_text = para_text.replace(run_text, f"**{run_text}**")
        
        # Check for justified text (multiple spaces between words)
        if para.alignment == docx.enum.text.WD_ALIGN_PARAGRAPH.JUSTIFY:
            # Add extra spaces between words to indicate justified text
            words = para_text.split()
            if len(words) > 3:
                para_text = "  ".join(words)
        
        text_with_formatting.append(para_text)
    
    return "\n".join(text_with_formatting)

def proofread_text(text):
    # Split text into paragraphs to preserve formatting
    paragraphs = text.split('\n')
    proofread_paragraphs = []
    
    for para in paragraphs:
        # Skip empty paragraphs
        if not para.strip():
            proofread_paragraphs.append(para)
            continue
        
        # Check for formatting markers
        has_formatting = bool(re.search(r'\*\*.*\*\*|^\s{10,}|^\s{20,}', para))
        
        if has_formatting:
            # For formatted paragraphs, only proofread the content, not the formatting
            # Extract the actual text content
            content = re.sub(r'\*\*|\s{2,}', ' ', para).strip()
            
            # Proofread the content with error handling
            try:
                if language_tool_available:
                    matches = tool.check(content)
                    corrected_content = language_tool_python.utils.correct(content, matches)
                else:
                    # Fallback: return the original content if LanguageTool is not available
                    corrected_content = content
            except Exception as e:
                logger.error(f"Error during proofreading: {str(e)}")
                corrected_content = content
            
            # Reapply the formatting
            if re.search(r'^\s{10,}', para):
                # Centered text
                corrected_para = " " * 10 + corrected_content + " " * 10
            elif re.search(r'^\s{20,}', para):
                # Right-aligned text
                corrected_para = " " * 20 + corrected_content
            else:
                # Just reapply the original formatting
                corrected_para = para.replace(content, corrected_content)
            
            proofread_paragraphs.append(corrected_para)
        else:
            # For regular paragraphs, proofread normally with error handling
            try:
                if language_tool_available:
                    matches = tool.check(para)
                    corrected_para = language_tool_python.utils.correct(para, matches)
                else:
                    # Fallback: return the original content if LanguageTool is not available
                    corrected_para = para
            except Exception as e:
                logger.error(f"Error during proofreading: {str(e)}")
                corrected_para = para
                
            proofread_paragraphs.append(corrected_para)
    
    return "\n".join(proofread_paragraphs)

def save_text_to_docx(text, docx_path):
    doc = docx.Document()
    for line in text.split("\n"):
        # Check for formatting markers
        if re.search(r'\*\*.*\*\*', line):
            # Bold text
            p = doc.add_paragraph()
            parts = re.split(r'(\*\*.*?\*\*)', line)
            for part in parts:
                if re.match(r'\*\*.*\*\*', part):
                    # Bold text
                    run = p.add_run(part.strip('*'))
                    run.bold = True
                else:
                    # Regular text
                    p.add_run(part)
        elif re.search(r'^\s{10,}', line):
            # Centered text
            p = doc.add_paragraph(line.strip())
            p.alignment = docx.enum.text.WD_ALIGN_PARAGRAPH.CENTER
        elif re.search(r'^\s{20,}', line):
            # Right-aligned text
            p = doc.add_paragraph(line.strip())
            p.alignment = docx.enum.text.WD_ALIGN_PARAGRAPH.RIGHT
        elif re.search(r'\s{2,}', line):
            # Justified text
            p = doc.add_paragraph(line)
            p.alignment = docx.enum.text.WD_ALIGN_PARAGRAPH.JUSTIFY
        else:
            # Regular paragraph
            doc.add_paragraph(line)
    
    doc.save(docx_path)

@app.route('/convert', methods=['POST'])
def convert_pdf_to_docx():
    if 'file' not in request.files:
        return "No file part", 400

    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    filename = secure_filename(file.filename)
    pdf_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(pdf_path)

    docx_filename = filename.rsplit('.', 1)[0] + '.docx'
    docx_path = os.path.join(OUTPUT_FOLDER, docx_filename)

    try:
        cv = Converter(pdf_path)
        cv.convert(docx_path)
        cv.close()

        if not os.path.exists(docx_path):
            return "DOCX file was not created", 500

        # Extract text from DOCX with formatting
        extracted_text = extract_text_from_docx(docx_path)

        # Proofread the text while preserving formatting
        proofread_text_content = proofread_text(extracted_text)

        # Save proofread text back to DOCX
        proofread_docx_path = os.path.join(OUTPUT_FOLDER, "proofread_" + docx_filename)
        save_text_to_docx(proofread_text_content, proofread_docx_path)

        return jsonify({
            "original_text": extracted_text,
            "proofread_text": proofread_text_content,
            "download_url": "/download/" + "proofread_" + docx_filename
        })

    except Exception as e:
        logger.error(f"Conversion error: {str(e)}")
        return f"Conversion error: {str(e)}", 500

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(os.path.join(OUTPUT_FOLDER, filename), as_attachment=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000, debug=True)
    
