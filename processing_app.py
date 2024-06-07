import os
from flask import Flask, request, jsonify
from PyPDF2 import PdfReader
from transformers import T5Tokenizer, T5ForConditionalGeneration

app = Flask(__name__)

def extract_text_from_pdf(file_path, num_pages_to_extract=5):
    reader = PdfReader(file_path)
    text = ""
    num_pages_to_extract = min(len(reader.pages), num_pages_to_extract)
    for page_num in range(num_pages_to_extract):
        page = reader.pages[page_num]
        text += page.extract_text()
    return text


tokenizer = T5Tokenizer.from_pretrained('t5-small')
model = T5ForConditionalGeneration.from_pretrained('t5-small')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file:
        file_path = os.path.join("uploads", file.filename)
        file.save(file_path)
        pdf_text = extract_text_from_pdf(file_path)
        return jsonify({"extracted_text": pdf_text})

if __name__ == '__main__':
    os.makedirs("uploads", exist_ok=True)  
    app.run(host='0.0.0.0', port=5000, debug=True)
