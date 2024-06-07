import os
from flask import Flask, request, jsonify
from PyPDF2 import PdfReader
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from google.cloud import storage

app = Flask(__name__)


tokenizer = T5Tokenizer.from_pretrained('t5-small')
model = T5ForConditionalGeneration.from_pretrained('t5-small')

def extract_text_from_pdf(file_path, num_pages_to_extract=5):
    reader = PdfReader(file_path)
    text = ""
    num_pages_to_extract = min(len(reader.pages), num_pages_to_extract)
    for page_num in range(num_pages_to_extract):
        page = reader.pages[page_num]
        text += page.extract_text()
    return text

def upload_to_gcs(bucket_name, file_name, content):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(file_name)
    blob.upload_from_string(content)
    print(f"Successfully uploaded {file_name} to {bucket_name}")

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"})
    
    if file:
       
        file_path = os.path.join('uploads', file.filename)
        file.save(file_path)

       
        pdf_text = extract_text_from_pdf(file_path)

       
        inputs = tokenizer(pdf_text, return_tensors="pt", max_length=512, truncation=True)

       
        bucket_name = 'your-gcs-bucket-name'
        file_name = 'extracted_text.txt'

   
        upload_to_gcs(bucket_name, file_name, pdf_text)

        
        tokenized_file_name = 'tokenized_inputs.json'
        upload_to_gcs(bucket_name, tokenized_file_name, str(inputs))

        return jsonify({"message": "File processed and uploaded successfully"})

if __name__ == '__main__':
    os.makedirs("uploads", exist_ok=True)  
    app.run(host='0.0.0.0', port=5000, debug=True)