from flask import Flask, request, send_file, jsonify
from werkzeug.utils import secure_filename
import os
import docx
import language_tool_python
from pdf2docx import Converter
from flask_cors import CORS
import traceback
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "/tmp"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Using local LanguageTool instance for more accuracy
tool = language_tool_python.LanguageToolPublicAPI('en-US')  # Uses the online API

def extract_text_from_docx(docx_path):
    """Extracts text from a DOCX file."""
    doc = docx.Document(docx_path)
    return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])

def extract_formatted_content_from_pdf(pdf_path):
    """Extracts formatted content from PDF with styling information."""
    try:
        # Try to use PyMuPDF if available
        import fitz
        doc = fitz.open(pdf_path)
        formatted_content = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            try:
                # Get text blocks with formatting information
                blocks = page.get_text("dict")
                
                # Check if blocks is a dictionary and has the expected structure
                if not isinstance(blocks, dict) or "blocks" not in blocks:
                    logger.warning(f"Unexpected structure in PDF page {page_num}: {blocks}")
                    continue
                
                for block in blocks["blocks"]:
                    # Check if block has the expected structure
                    if not isinstance(block, dict) or "lines" not in block:
                        continue
                        
                    for line in block["lines"]:
                        # Check if line has the expected structure
                        if not isinstance(line, dict) or "spans" not in line:
                            continue
                            
                        for span in line["spans"]:
                            # Check if span has the expected structure
                            if not isinstance(span, dict):
                                continue
                                
                            # Extract text with formatting
                            text = span.get("text", "")
                            font = span.get("font", "Arial")
                            size = span.get("size", 12)
                            color = span.get("color", [0, 0, 0])
                            
                            # Ensure color is a list with at least 3 elements
                            if not isinstance(color, (list, tuple)) or len(color) < 3:
                                color = [0, 0, 0]
                            
                            # Convert color to hex
                            try:
                                color_hex = "#{:02x}{:02x}{:02x}".format(
                                    int(color[0] * 255), 
                                    int(color[1] * 255), 
                                    int(color[2] * 255)
                                )
                            except (TypeError, ValueError, IndexError):
                                color_hex = "#000000"
                            
                            # Create HTML with styling
                            formatted_content.append({
                                "text": text,
                                "font": font,
                                "size": size,
                                "color": color_hex,
                                "bold": "bold" in str(font).lower(),
                                "italic": "italic" in str(font).lower(),
                                "underline": "underline" in str(font).lower()
                            })
            except Exception as page_error:
                logger.error(f"Error processing page {page_num}: {str(page_error)}")
                logger.error(traceback.format_exc())
                # Continue with next page
        
        # Check if we extracted any content
        if not formatted_content:
            logger.warning(f"No text content extracted from PDF using PyMuPDF: {pdf_path}")
            # Try fallback method
            return extract_formatted_content_fallback(pdf_path)
            
        return formatted_content
    except ImportError:
        # Fallback to pdf2docx if PyMuPDF is not available
        logger.warning("PyMuPDF not available, falling back to pdf2docx")
        return extract_formatted_content_fallback(pdf_path)
    except Exception as e:
        logger.error(f"Error extracting formatted content with PyMuPDF: {str(e)}")
        logger.error(traceback.format_exc())
        # Fallback to pdf2docx
        return extract_formatted_content_fallback(pdf_path)

def extract_formatted_content_fallback(pdf_path):
    """Fallback method to extract content from PDF using pdf2docx."""
    try:
        # Convert PDF to DOCX
        docx_filename = os.path.basename(pdf_path).rsplit('.', 1)[0] + '_temp.docx'
        docx_path = os.path.join(OUTPUT_FOLDER, docx_filename)
        
        cv = Converter(pdf_path)
        cv.convert(docx_path, start=0, end=None)
        cv.close()
        
        # Extract text from DOCX
        doc = docx.Document(docx_path)
        
        # Create formatted content with basic styling
        formatted_content = []
        for para in doc.paragraphs:
            if para.text.strip():
                # Check if paragraph has runs with different formatting
                if len(para.runs) > 1:
                    for run in para.runs:
                        if run.text.strip():
                            formatted_content.append({
                                "text": run.text,
                                "font": run.font.name if run.font.name else "Arial",
                                "size": 12,  # Default size
                                "color": "#000000",  # Default color
                                "bold": run.bold if run.bold is not None else False,
                                "italic": run.italic if run.italic is not None else False,
                                "underline": run.underline if run.underline is not None else False
                            })
                else:
                    # Single run paragraph
                    formatted_content.append({
                        "text": para.text,
                        "font": "Arial",
                        "size": 12,
                        "color": "#000000",
                        "bold": False,
                        "italic": False,
                        "underline": False
                    })
        
        # Clean up temporary file
        if os.path.exists(docx_path):
            os.remove(docx_path)
        
        # Check if we extracted any content
        if not formatted_content:
            logger.warning(f"No text content extracted from PDF using fallback method: {pdf_path}")
            # Return a message indicating no content could be extracted
            return [{
                "text": "No text content could be extracted from this PDF. The file might be scanned, image-based, or password-protected.",
                "font": "Arial",
                "size": 12,
                "color": "#000000",
                "bold": False,
                "italic": False,
                "underline": False
            }]
            
        return formatted_content
    except Exception as e:
        logger.error(f"Error in fallback extraction: {str(e)}")
        logger.error(traceback.format_exc())
        # Last resort: return plain text with error message
        return [{
            "text": f"Error extracting content: {str(e)}. The PDF might be corrupted or in an unsupported format.",
            "font": "Arial",
            "size": 12,
            "color": "#000000",
            "bold": False,
            "italic": False,
            "underline": False
        }]

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
        logger.error(f"Error in proofreading: {str(e)}")
        # Return original text if proofreading fails
        return text, []

def save_text_to_docx(text, docx_path):
    """Saves proofread text to a new DOCX file."""
    try:
        doc = docx.Document()
        for line in text.split("\n"):
            doc.add_paragraph(line)
        doc.save(docx_path)
    except Exception as e:
        logger.error(f"Error saving to DOCX: {str(e)}")
        # Create a simple text file as fallback
        with open(docx_path, 'w', encoding='utf-8') as f:
            f.write(text)

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
    
    try:
        # Save the uploaded file
        file.save(pdf_path)
        
        # Check if the file is a valid PDF
        try:
            import fitz
            # Try to open the PDF to validate it
            doc = fitz.open(pdf_path)
            if len(doc) == 0:
                raise ValueError("PDF has no pages")
            doc.close()
        except Exception as pdf_error:
            logger.error(f"Invalid PDF file: {str(pdf_error)}")
            # Create a fallback response with error information
            error_message = f"Invalid PDF file: {str(pdf_error)}"
            fallback_content = [{
                "text": error_message,
                "font": "Arial",
                "size": 12,
                "color": "#000000",
                "bold": False,
                "italic": False,
                "underline": False
            }]
            
            # Return a response with the expected structure but with error information
            return jsonify({
                "original_content": fallback_content,
                "proofread_content": fallback_content,
                "grammar_errors": [],
                "download_url": "",
                "file_name": filename,
                "error": error_message
            }), 200
        
        # Extract formatted content from PDF
        formatted_content = extract_formatted_content_from_pdf(pdf_path)
        
        # Check if we got valid content
        if not formatted_content or len(formatted_content) == 0:
            logger.warning(f"No content extracted from PDF: {filename}")
            # Create a fallback content with a message
            formatted_content = [{
                "text": "No text content could be extracted from this PDF. The file might be scanned, image-based, or password-protected.",
                "font": "Arial",
                "size": 12,
                "color": "#000000",
                "bold": False,
                "italic": False,
                "underline": False
            }]
        
        # Extract plain text for proofreading
        plain_text = " ".join([item["text"] for item in formatted_content])
        
        # Check if we have enough text to proofread
        if len(plain_text.strip()) < 10:
            logger.warning(f"Insufficient text for proofreading: {filename}")
            # Create a fallback content with a message
            formatted_content = [{
                "text": "Insufficient text content for proofreading. The PDF might contain mostly images or be password-protected.",
                "font": "Arial",
                "size": 12,
                "color": "#000000",
                "bold": False,
                "italic": False,
                "underline": False
            }]
            plain_text = formatted_content[0]["text"]
        
        # Proofread the text
        proofread_text_content, grammar_errors = proofread_text(plain_text)
        
        # Create proofread formatted content
        proofread_formatted_content = []
        current_pos = 0
        
        for item in formatted_content:
            text = item["text"]
            text_len = len(text)
            
            # Find if this text has any corrections
            corrections = []
            for error in grammar_errors:
                if error["offset"] >= current_pos and error["offset"] < current_pos + text_len:
                    # Calculate relative position within this text
                    rel_offset = error["offset"] - current_pos
                    if rel_offset + error["length"] <= text_len:
                        corrections.append({
                            "offset": rel_offset,
                            "length": error["length"],
                            "suggestion": error["suggestions"][0] if error["suggestions"] else ""
                        })
            
            # Create a copy of the item with corrections
            proofread_item = item.copy()
            if corrections:
                # Apply corrections to the text
                corrected_text = text
                for correction in sorted(corrections, key=lambda x: x["offset"], reverse=True):
                    start = correction["offset"]
                    end = start + correction["length"]
                    corrected_text = corrected_text[:start] + correction["suggestion"] + corrected_text[end:]
                
                proofread_item["text"] = corrected_text
                proofread_item["has_correction"] = True
            else:
                proofread_item["has_correction"] = False
                
            proofread_formatted_content.append(proofread_item)
            current_pos += text_len + 1  # +1 for the space we added between items

        # Save proofread text back to DOCX
        docx_filename = filename.rsplit('.', 1)[0] + '.docx'
        docx_path = os.path.join(OUTPUT_FOLDER, "proofread_" + docx_filename)
        save_text_to_docx(proofread_text_content, docx_path)

        # Ensure we always return the expected data structure
        return jsonify({
            "original_content": formatted_content,
            "proofread_content": proofread_formatted_content,
            "grammar_errors": grammar_errors,
            "download_url": "/download/" + "proofread_" + docx_filename,
            "file_name": filename
        })

    except Exception as e:
        logger.error(f"Conversion error: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Create a fallback response with error information
        error_message = f"Error processing PDF: {str(e)}"
        fallback_content = [{
            "text": error_message,
            "font": "Arial",
            "size": 12,
            "color": "#000000",
            "bold": False,
            "italic": False,
            "underline": False
        }]
        
        # Return a response with the expected structure but with error information
        return jsonify({
            "original_content": fallback_content,
            "proofread_content": fallback_content,
            "grammar_errors": [],
            "download_url": "",
            "file_name": filename,
            "error": error_message
        }), 200  # Return 200 to avoid triggering the frontend error handler

@app.route('/download/<filename>')
def download_file(filename):
    """Handles file download."""
    file_path = os.path.join(OUTPUT_FOLDER, filename)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    return jsonify({"error": "File not found"}), 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000, debug=True)
