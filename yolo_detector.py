import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont

class PlateDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
    
    def detect_plates(self, image_path, confidence_threshold=0.5):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Не удалось загрузить изображение: {image_path}")
        
        results = self.model(image)
        
        plates = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    confidence = box.conf.cpu().numpy()[0]
                    if confidence >= confidence_threshold:
                        x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0].astype(int)
                        
                        plate_crop = image[y1:y2, x1:x2]
                        
                        plate_crop_rgb = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2RGB)
                        plate_image = Image.fromarray(plate_crop_rgb)
                        
                        plates.append({
                            'image': plate_image,
                            'bbox': (x1, y1, x2, y2),
                            'confidence': confidence
                        })
        
        return plates
    
    def detect_and_save(self, image_path, output_dir="plates", confidence_threshold=0.5):
        import os
        
        os.makedirs(output_dir, exist_ok=True)
        
        plates = self.detect_plates(image_path, confidence_threshold)
        saved_paths = []
        
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        for i, plate_data in enumerate(plates):
            plate_path = os.path.join(output_dir, f"{base_name}_plate_{i}.jpg")
            plate_data['image'].save(plate_path)
            saved_paths.append(plate_path)
            
    def draw_results_on_image(self, image_path, plates_with_text, output_path=None):
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        
        draw = ImageDraw.Draw(pil_image)
        
        font = ImageFont.load_default(size=30)
        
        for plate_data in plates_with_text:
            x1, y1, x2, y2 = plate_data['bbox']
            text = plate_data['text']
            
            color = (0, 255, 0)
            
            draw.rectangle([x1, y1, x2, y2], outline=color, width=6)
            
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            text_x = x1
            text_y = max(0, y1 - text_height - 10)
            
            draw.rectangle([text_x, text_y, text_x + text_width + 10, text_y + text_height + 10], 
                         fill=color, outline=color, width=4)
            
            draw.text((text_x + 5, text_y + 2), text, fill=(0, 0, 0), font=font)
        
        if output_path:
            pil_image.save(output_path)
        
        return pil_image