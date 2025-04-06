from flask import Flask, request, send_file, jsonify
from werkzeug.utils import secure_filename
import os
import docx
import language_tool_python
from pdf2docx import Converter
from flask_cors import CORS
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.colors import black, blue, red
import io
import PyPDF2
import fitz  # PyMuPDF
import re

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "/tmp"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Using local LanguageTool instance for more accuracy
tool = language_tool_python.LanguageToolPublicAPI('en-US')  # Uses the online API

def extract_text_from_docx(docx_path):
    """Extracts text and formatting from a DOCX file."""
    doc = docx.Document(docx_path)
    formatted_paragraphs = []
    
    for para in doc.paragraphs:
        if not para.text.strip():
            continue
            
        # Extract formatting for each paragraph
        para_format = {
            'text': para.text,
            'style': para.style.name if para.style else 'Normal',
            'alignment': para.alignment,
            'runs': [],
            'indent': para.paragraph_format.left_indent.pt if para.paragraph_format.left_indent else 0,
            'line_spacing': para.paragraph_format.line_spacing if para.paragraph_format.line_spacing else 1.0,
            'space_before': para.paragraph_format.space_before.pt if para.paragraph_format.space_before else 0,
            'space_after': para.paragraph_format.space_after.pt if para.paragraph_format.space_after else 0,
            'first_line_indent': para.paragraph_format.first_line_indent.pt if para.paragraph_format.first_line_indent else 0,
            'keep_together': para.paragraph_format.keep_together if hasattr(para.paragraph_format, 'keep_together') else False,
            'keep_with_next': para.paragraph_format.keep_with_next if hasattr(para.paragraph_format, 'keep_with_next') else False,
            'page_break_before': para.paragraph_format.page_break_before if hasattr(para.paragraph_format, 'page_break_before') else False
        }
        
        # Extract formatting for each run (text segment with consistent formatting)
        for run in para.runs:
            if run.text.strip():
                run_format = {
                    'text': run.text,
                    'bold': run.bold,
                    'italic': run.italic,
                    'underline': run.underline,
                    'font_size': run.font.size.pt if run.font.size else None,
                    'font_name': run.font.name if run.font.name else None,
                    'color': str(run.font.color.rgb) if run.font.color and run.font.color.rgb else None,
                    'highlight': str(run.font.highlight_color) if run.font.highlight_color else None,
                    'strike': run.strike if hasattr(run, 'strike') else False,
                    'subscript': run.subscript if hasattr(run, 'subscript') else False,
                    'superscript': run.superscript if hasattr(run, 'superscript') else False
                }
                para_format['runs'].append(run_format)
        
        formatted_paragraphs.append(para_format)
    
    return formatted_paragraphs

def proofread_text(text):
    """Proofreads text using LanguageTool and returns corrected text with details."""
    # Join all paragraph texts for proofreading
    full_text = "\n".join([para['text'] for para in text])
    matches = tool.check(full_text)
    corrected_text = language_tool_python.utils.correct(full_text, matches)

    # Collect detailed grammar mistakes
    errors = []
    for match in matches:
        errors.append({
            "message": match.message,
            "suggestions": match.replacements,
            "offset": match.offset,
            "length": match.errorLength
        })

    # Split the corrected text back into paragraphs
    corrected_paragraphs = corrected_text.split("\n")
    
    # Create a new formatted text with corrections
    formatted_corrected = []
    for i, para in enumerate(text):
        if i < len(corrected_paragraphs):
            corrected_para = para.copy()
            corrected_para['text'] = corrected_paragraphs[i]
            formatted_corrected.append(corrected_para)
        else:
            formatted_corrected.append(para)
    
    return formatted_corrected, errors

def save_text_to_docx(formatted_text, docx_path):
    """Saves proofread text with formatting to a new DOCX file."""
    doc = docx.Document()
    
    for para_data in formatted_text:
        p = doc.add_paragraph(para_data['text'])
        
        # Apply paragraph-level formatting
        if para_data.get('style'):
            p.style = para_data['style']
        if para_data.get('alignment'):
            p.alignment = para_data['alignment']
        
        # Apply run-level formatting
        if para_data.get('runs'):
            # Clear the paragraph text to add formatted runs
            p.clear()
            for run_data in para_data['runs']:
                run = p.add_run(run_data['text'])
                run.bold = run_data.get('bold', False)
                run.italic = run_data.get('italic', False)
                run.underline = run_data.get('underline', False)
                
                # Apply font properties if available
                if run_data.get('font_size'):
                    run.font.size = docx.shared.Pt(run_data['font_size'])
                if run_data.get('font_name'):
                    run.font.name = run_data['font_name']
                if run_data.get('color'):
                    run.font.color.rgb = docx.shared.RGBColor.from_string(run_data['color'])
    
    doc.save(docx_path)

def extract_pdf_formatting(pdf_path):
    """Extracts formatting information from a PDF file."""
    formatting = {
        'pages': [],
        'fonts': set(),
        'page_size': None
    }
    
    try:
        # Open the PDF with PyMuPDF
        doc = fitz.open(pdf_path)
        
        # Get page size
        if doc.page_count > 0:
            page = doc[0]
            formatting['page_size'] = (page.rect.width, page.rect.height)
        
        # Extract text and formatting from each page
        for page_num in range(doc.page_count):
            page = doc[page_num]
            page_info = {
                'text_blocks': [],
                'images': []
            }
            
            # Extract text blocks with their formatting
            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                if block.get("type") == 0:  # Text block
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            text_block = {
                                'text': span.get("text", ""),
                                'font': span.get("font", ""),
                                'size': span.get("size", 0),
                                'color': span.get("color", 0),
                                'bbox': span.get("bbox", (0, 0, 0, 0)),
                                'bold': "bold" in span.get("font", "").lower(),
                                'italic': "italic" in span.get("font", "").lower()
                            }
                            page_info['text_blocks'].append(text_block)
                            formatting['fonts'].add(span.get("font", ""))
                
                elif block.get("type") == 1:  # Image block
                    page_info['images'].append({
                        'bbox': block.get("bbox", (0, 0, 0, 0)),
                        'xref': block.get("xref", 0)
                    })
            
            formatting['pages'].append(page_info)
        
        doc.close()
        return formatting
    
    except Exception as e:
        print(f"Error extracting PDF formatting: {str(e)}")
        return formatting

def apply_pdf_formatting_to_docx(docx_path, pdf_formatting, output_path):
    """Applies PDF formatting to a DOCX file and saves it as a new DOCX."""
    doc = docx.Document(docx_path)
    
    # Create a new document with the same content
    new_doc = docx.Document()
    
    # Copy paragraphs with formatting
    for i, para in enumerate(doc.paragraphs):
        if not para.text.strip():
            continue
        
        # Create a new paragraph
        new_para = new_doc.add_paragraph()
        
        # Copy paragraph formatting
        if para.style:
            new_para.style = para.style
        if para.alignment:
            new_para.alignment = para.alignment
        
        # Copy paragraph format properties
        if para.paragraph_format:
            new_para.paragraph_format.left_indent = para.paragraph_format.left_indent
            new_para.paragraph_format.right_indent = para.paragraph_format.right_indent
            new_para.paragraph_format.first_line_indent = para.paragraph_format.first_line_indent
            new_para.paragraph_format.line_spacing = para.paragraph_format.line_spacing
            new_para.paragraph_format.space_before = para.paragraph_format.space_before
            new_para.paragraph_format.space_after = para.paragraph_format.space_after
        
        # Copy runs with formatting
        for run in para.runs:
            new_run = new_para.add_run(run.text)
            
            # Copy run formatting
            new_run.bold = run.bold
            new_run.italic = run.italic
            new_run.underline = run.underline
            
            # Copy font properties
            if run.font:
                if run.font.name:
                    new_run.font.name = run.font.name
                if run.font.size:
                    new_run.font.size = run.font.size
                if run.font.color and run.font.color.rgb:
                    new_run.font.color.rgb = run.font.color.rgb
        
        # Try to match with PDF formatting if available
        if pdf_formatting and pdf_formatting['pages'] and i < len(pdf_formatting['pages'][0]['text_blocks']):
            pdf_block = pdf_formatting['pages'][0]['text_blocks'][i]
            
            # Apply font size if available
            if pdf_block.get('size'):
                for run in new_para.runs:
                    run.font.size = docx.shared.Pt(pdf_block['size'])
            
            # Apply bold/italic if available
            if pdf_block.get('bold'):
                for run in new_para.runs:
                    run.bold = True
            
            if pdf_block.get('italic'):
                for run in new_para.runs:
                    run.italic = True
    
    # Save the new document
    new_doc.save(output_path)
    return output_path

@app.route('/convert', methods=['POST'])
def convert_pdf_to_docx():
    """Handles PDF-to-DOCX conversion and proofreading."""
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(file.filename)
    pdf_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(pdf_path)

    docx_filename = filename.rsplit('.', 1)[0] + '.docx'
    docx_path = os.path.join(OUTPUT_FOLDER, docx_filename)

    try:
        # Convert PDF to DOCX
        cv = Converter(pdf_path)
        cv.convert(docx_path, start=0, end=None)  # Ensure full document conversion
        cv.close()

        if not os.path.exists(docx_path):
            return jsonify({"error": "DOCX file was not created"}), 500

        # Extract text and formatting from DOCX
        formatted_text = extract_text_from_docx(docx_path)

        # Proofread the text
        proofread_formatted_text, grammar_errors = proofread_text(formatted_text)

        # Save proofread text back to DOCX
        proofread_docx_path = os.path.join(OUTPUT_FOLDER, "proofread_" + docx_filename)
        save_text_to_docx(proofread_formatted_text, proofread_docx_path)

        return jsonify({
            "original_text": formatted_text,
            "proofread_text": proofread_formatted_text,
            "grammar_errors": grammar_errors,  # Provide detailed grammar corrections
            "download_url": "/download/" + "proofread_" + docx_filename
        })

    except Exception as e:
        return jsonify({"error": f"Conversion error: {str(e)}"}), 500

@app.route('/download/<filename>')
def download_file(filename):
    """Handles file download."""
    file_path = os.path.join(OUTPUT_FOLDER, filename)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    return jsonify({"error": "File not found"}), 404

@app.route('/download-pdf/<filename>')
def download_pdf(filename):
    """Converts a DOCX file to PDF with formatting similar to the original PDF."""
    # Extract the original filename from the proofread filename
    # The format is "proofread_originalname.docx"
    original_filename = filename.replace("proofread_", "")
    original_filename = original_filename.replace(".docx", ".pdf")
    
    # Check if the original PDF exists in the UPLOAD_FOLDER
    original_pdf_path = os.path.join(UPLOAD_FOLDER, original_filename)
    if not os.path.exists(original_pdf_path):
        return jsonify({"error": "Original PDF file not found"}), 404
    
    # Check if the DOCX file exists in the OUTPUT_FOLDER
    docx_path = os.path.join(OUTPUT_FOLDER, filename)
    if not os.path.exists(docx_path):
        return jsonify({"error": "DOCX file not found"}), 404
    
    try:
        # Extract text from the proofread DOCX
        doc = docx.Document(docx_path)
        proofread_text = []
        
        for para in doc.paragraphs:
            if para.text.strip():
                proofread_text.append(para.text)
        
        # Open the original PDF
        pdf_doc = fitz.open(original_pdf_path)
        
        # Create a new PDF with the same page size and layout
        output_pdf = fitz.open()
        
        # Process each page of the original PDF
        for page_num in range(pdf_doc.page_count):
            # Get the original page
            original_page = pdf_doc[page_num]
            
            # Create a new page with the same size
            new_page = output_pdf.new_page(width=original_page.rect.width, height=original_page.rect.height)
            
            # Copy the original page content (background, images, etc.)
            new_page.show_pdf_page(new_page.rect, pdf_doc, page_num)
            
            # Extract text blocks from the original page
            text_blocks = original_page.get_text("dict")["blocks"]
            
            # Create a list to store text positions
            text_positions = []
            
            # Extract text positions from the original page
            for block in text_blocks:
                if block.get("type") == 0:  # Text block
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            text_positions.append({
                                'text': span.get("text", ""),
                                'bbox': span.get("bbox", (0, 0, 0, 0)),
                                'font': span.get("font", ""),
                                'size': span.get("size", 0),
                                'color': span.get("color", 0),
                                'bold': "bold" in span.get("font", "").lower(),
                                'italic': "italic" in span.get("font", "").lower()
                            })
            
            # If we have proofread text and text positions, replace the text
            if proofread_text and text_positions:
                # Create a mapping of original text to proofread text
                text_mapping = {}
                
                # Improved mapping: try to match text by similarity
                # This helps when the text has been corrected and doesn't match exactly
                for i, pos in enumerate(text_positions):
                    original_text = pos['text'].strip()
                    if not original_text:
                        continue
                    
                    # Try to find the best match in the proofread text
                    best_match_index = -1
                    best_match_score = 0
                    
                    for j, proofread_para in enumerate(proofread_text):
                        # Simple similarity score based on word overlap
                        original_words = set(original_text.lower().split())
                        proofread_words = set(proofread_para.lower().split())
                        
                        # Calculate Jaccard similarity
                        intersection = len(original_words.intersection(proofread_words))
                        union = len(original_words.union(proofread_words))
                        
                        if union > 0:
                            similarity = intersection / union
                            if similarity > best_match_score and similarity > 0.3:  # Threshold to avoid false matches
                                best_match_score = similarity
                                best_match_index = j
                    
                    # If we found a good match, add it to the mapping
                    if best_match_index >= 0:
                        text_mapping[original_text] = proofread_text[best_match_index]
                    # Fallback to index-based mapping if no good match found
                    elif i < len(proofread_text):
                        text_mapping[original_text] = proofread_text[i]
                
                # Replace text on the page
                for pos in text_positions:
                    original_text = pos['text'].strip()
                    if original_text in text_mapping:
                        # Create a white rectangle to cover the original text
                        new_page.draw_rect(pos['bbox'], color=(1, 1, 1), fill=(1, 1, 1))
                        
                        # Use a default font if the original font is not available
                        font_name = "helv"  # Default to Helvetica
                        
                        # Insert the proofread text at the same position
                        try:
                            new_page.insert_text(
                                (pos['bbox'][0], pos['bbox'][1] + pos['size'] * 0.8),  # Position text at the top of the bbox
                                text_mapping[original_text],
                                fontsize=pos['size'],
                                color=pos['color'],
                                fontname=font_name
                            )
                        except Exception as e:
                            # If there's an error with the font, try with a different font
                            print(f"Font error: {str(e)}")
                            try:
                                new_page.insert_text(
                                    (pos['bbox'][0], pos['bbox'][1] + pos['size'] * 0.8),
                                    text_mapping[original_text],
                                    fontsize=pos['size'],
                                    color=pos['color']
                                )
                            except Exception as e2:
                                print(f"Second font error: {str(e2)}")
                                # If still failing, just use the most basic approach
                                new_page.insert_text(
                                    (pos['bbox'][0], pos['bbox'][1] + pos['size'] * 0.8),
                                    text_mapping[original_text]
                                )
        
        # Save the modified PDF to a buffer
        pdf_buffer = io.BytesIO()
        output_pdf.save(pdf_buffer)
        pdf_buffer.seek(0)
        
        # Close the PDF documents
        pdf_doc.close()
        output_pdf.close()
        
        # Return the PDF file
        return send_file(
            pdf_buffer,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=original_filename
        )
    except Exception as e:
        return jsonify({"error": f"Conversion error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000, debug=True)
