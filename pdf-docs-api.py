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
    """Converts a DOCX file to PDF and returns the PDF file."""
    # Check if the file exists in the OUTPUT_FOLDER
    docx_path = os.path.join(OUTPUT_FOLDER, filename)
    if not os.path.exists(docx_path):
        return jsonify({"error": "DOCX file not found"}), 404
    
    try:
        # Extract text and formatting from DOCX
        formatted_text = extract_text_from_docx(docx_path)
        
        # Create a PDF in memory with proper margins
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer, 
            pagesize=letter,
            leftMargin=72,  # 1 inch
            rightMargin=72,  # 1 inch
            topMargin=72,    # 1 inch
            bottomMargin=72  # 1 inch
        )
        styles = getSampleStyleSheet()
        
        # Create custom styles with better formatting
        normal_style = ParagraphStyle(
            'CustomNormal',
            parent=styles['Normal'],
            fontSize=11,
            leading=14,  # Line spacing
            spaceBefore=6,
            spaceAfter=6
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading1'],
            fontSize=16,
            leading=20,
            spaceBefore=12,
            spaceAfter=12
        )
        
        # Build the PDF content
        story = []
        
        for para in formatted_text:
            # Process runs to handle text formatting
            formatted_text = ""
            for run in para.get('runs', []):
                text = run.get('text', '')
                if run.get('bold'):
                    text = f"<b>{text}</b>"
                if run.get('italic'):
                    text = f"<i>{text}</i>"
                if run.get('underline'):
                    text = f"<u>{text}</u>"
                if run.get('strike'):
                    text = f"<strike>{text}</strike>"
                if run.get('subscript'):
                    text = f"<sub>{text}</sub>"
                if run.get('superscript'):
                    text = f"<sup>{text}</sup>"
                
                # Handle font size if available
                font_size = run.get('font_size')
                if font_size:
                    text = f"<font size='{font_size}'>{text}</font>"
                
                # Handle font color if available
                color = run.get('color')
                if color:
                    # Convert RGB color to hex
                    try:
                        rgb = color.replace('RGBColor(', '').replace(')', '').split(',')
                        if len(rgb) == 3:
                            hex_color = '#{:02x}{:02x}{:02x}'.format(
                                int(rgb[0]), int(rgb[1]), int(rgb[2])
                            )
                            text = f"<font color='{hex_color}'>{text}</font>"
                    except:
                        pass
                
                formatted_text += text
            
            # If no runs or empty formatted text, use the paragraph text
            if not formatted_text:
                formatted_text = para.get('text', '')
            
            # Create a custom style for this paragraph based on its properties
            para_style = ParagraphStyle(
                f'CustomStyle_{len(story)}',
                parent=normal_style,
                fontSize=11,  # Default font size
                leading=14,   # Default line spacing
                spaceBefore=para.get('space_before', 6),
                spaceAfter=para.get('space_after', 6),
                leftIndent=para.get('indent', 0),
                firstLineIndent=para.get('first_line_indent', 0)
            )
            
            # Apply paragraph-level formatting
            if para.get('style') and 'Heading' in para['style']:
                para_style.fontSize = 16
                para_style.leading = 20
                para_style.spaceBefore = 12
                para_style.spaceAfter = 12
            
            # Create the paragraph with the custom style
            p = Paragraph(formatted_text, para_style)
            
            # Apply alignment if available
            if para.get('alignment'):
                alignment = para['alignment']
                if alignment == 0:  # Left
                    p.alignment = 0
                elif alignment == 1:  # Center
                    p.alignment = 1
                elif alignment == 2:  # Right
                    p.alignment = 2
                elif alignment == 3:  # Justify
                    p.alignment = 4
            
            # Add page break if needed
            if para.get('page_break_before'):
                story.append(PageBreak())
            
            story.append(p)
            
            # Add keep with next if needed
            if para.get('keep_with_next'):
                # This is a simplified approach - in a real implementation,
                # you would need to handle this more carefully
                pass
        
        # Build the PDF
        doc.build(story)
        
        # Reset buffer position
        buffer.seek(0)
        
        # Return the PDF file
        return send_file(
            buffer,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=filename.replace('.docx', '.pdf')
        )
    except Exception as e:
        return jsonify({"error": f"Conversion error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000, debug=True)
