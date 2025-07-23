import os
import uuid
from base64 import b64encode

from flask import Flask, request, jsonify

from processor import LicensePlateProcessor

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10MB максимальный размер файла

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    """Проверяет, разрешен ли тип файла"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def image_to_base64(image_path):
    """Конвертирует изображение в base64 строку"""
    try:
        with open(image_path, 'rb') as image_file:
            encoded_string = b64encode(image_file.read()).decode('utf-8')
            return encoded_string
    except Exception as e:
        print(f"Ошибка при конвертации в base64: {str(e)}")
        return None


@app.route('/anpr/', methods=['POST'])
def anpr():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'Файл не найден в запросе'}), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({'error': 'Файл не выбран'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Недопустимый тип файла. Разрешены: png, jpg, jpeg'}), 400
        
        if file:
            file_extension = file.filename.rsplit('.', 1)[1].lower()
            unique_filename = f"{uuid.uuid4()}.{file_extension}"
            
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(file_path)
            
            try:
                result_filename = f"result_{uuid.uuid4()}.jpg"
                result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
                
                result = processor.process_and_visualize(
                    file_path, 
                    output_path=result_path
                )

                output_file = image_to_base64(result_path)
                
                print(f"Результат обработки: {result}")
                
                response_data = {
                    'success': True,
                    'results': result,
                    'output_file': output_file
                }
                
                return jsonify(response_data), 200
                
            except Exception as e:                
                return jsonify({
                    'error': f'Ошибка при обработке изображения: {str(e)}'
                }), 500
            
            finally:
                for path in [file_path, result_path]:
                    if os.path.exists(path):
                        os.remove(path)
    
    except Exception as e:
        return jsonify({'error': f'Внутренняя ошибка сервера: {str(e)}'}), 500


@app.errorhandler(413)
def too_large(e):
    """Обработчик ошибки превышения размера файла"""
    return jsonify({'error': 'Файл слишком большой. Максимальный размер: 10MB'}), 413


if __name__ == '__main__':
    yolo_model_path = "/home/user/anpr/models/anpr_detect.pt"
    ocr_model_path = "/home/user/anpr/models/anpr_ocr.pth"
    
    processor = LicensePlateProcessor(yolo_model_path, ocr_model_path)
    
    app.run(host='127.0.0.1', port=4000, debug=False)
