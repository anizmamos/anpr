from yolo_detector import PlateDetector
from ocr import PlateRecognizer

class LicensePlateProcessor:
    def __init__(self, yolo_model_path, ocr_model_path):
        """
        Инициализация полного пайплайна распознавания номеров
        
        Args:
            yolo_model_path (str): Путь к модели YOLO для детекции
            ocr_model_path (str): Путь к модели CRNN для распознавания
        """
        self.detector = PlateDetector(yolo_model_path)
        self.recognizer = PlateRecognizer(ocr_model_path)
    
    def process_image(self, image_path, confidence_threshold=0.5):
        """
        Полная обработка изображения: детекция + распознавание
        
        Args:
            image_path (str): Путь к изображению
            confidence_threshold (float): Порог уверенности для детекции
            
        Returns:
            list: Список словарей с результатами
        """
  
        plates = self.detector.detect_plates(image_path, confidence_threshold)
        
        results = []
        for i, plate_data in enumerate(plates):
            text = self.recognizer.recognize_from_image(plate_data['image'])
            
            result = {
                'plate_number': i,
                'text': text,
                'bbox': plate_data['bbox'],
                'confidence': plate_data['confidence'],
                'image': plate_data['image']
            }
            results.append(result)
        
        return results
    
    def process_and_visualize(self, image_path, output_path=None, confidence_threshold=0.5):
        """
        Обрабатывает изображение и возвращает результат с рамочками и текстом
        
        Args:
            image_path (str): Путь к изображению
            output_path (str): Путь для сохранения результата (если None, не сохраняет)
            confidence_threshold (float): Порог уверенности
            show_image (bool): Показать изображение с помощью cv2
            
        Returns:
            PIL.Image: Изображение с нарисованными результатами
        """
        results = self.process_image(image_path, confidence_threshold)
        
        if not results:
            print("Номерные знаки не обнаружены")
            return None
        
        self.detector.draw_results_on_image(image_path, results, output_path)
        
        return [result['text'] for result in results]


def main():
    yolo_model_path = "/home/user/anpr/models/anpr_detect.pt"
    ocr_model_path = "/home/user/anpr/models/anpr_ocr.pth"
    
    processor = LicensePlateProcessor(yolo_model_path, ocr_model_path)
    
    image_path = "test.jpg"

    result = processor.process_and_visualize(
        image_path, 
        output_path='result.jpg'
    )

    print(result)


if __name__ == "__main__":
    main()
