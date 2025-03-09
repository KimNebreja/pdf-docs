from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import os
from pdf2docx import Converter
from tempfile import NamedTemporaryFile

app = FastAPI()
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "output"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.post("/convert")
async def convert_pdf_to_docx(file: UploadFile = File(...)):
    filename = file.filename
    pdf_path = os.path.join(UPLOAD_FOLDER, filename)
    docx_filename = filename.rsplit('.', 1)[0] + '.docx'
    docx_path = os.path.join(OUTPUT_FOLDER, docx_filename)
    
    with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
        temp_pdf.write(file.file.read())
        temp_pdf_path = temp_pdf.name
    
    cv = Converter(temp_pdf_path)
    cv.convert(docx_path)
    cv.close()
    os.remove(temp_pdf_path)
    
    return FileResponse(docx_path, media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document", filename=docx_filename)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
