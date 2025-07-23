import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

class Config:
    CHARS = ['A', 'B', 'E', 'K', 'M', 'H', 'O', 'P', 'C', 'T', 'Y', 'X'] + \
            ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    BLANK_TOKEN = len(CHARS)
    VOCAB_SIZE = len(CHARS) + 1
    IMG_HEIGHT = 64
    IMG_WIDTH = 256
    HIDDEN_SIZE = 256
    NUM_LAYERS = 2

    @classmethod
    def idx_to_char(cls, idx):
        if 0 <= idx < len(cls.CHARS):
            return cls.CHARS[idx]
        return ''

class CRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers):
        super(CRNN, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d((2, 1), (2, 1)),

            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d((2, 1), (2, 1)),

            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d((4, 1), (4, 1)),
        )

        self.rnn = nn.LSTM(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True
        )

        self.classifier = nn.Linear(hidden_size * 2, vocab_size)

    def forward(self, x):
        conv_feat = self.cnn(x)
        conv_feat = conv_feat.squeeze(2)
        conv_feat = conv_feat.permute(0, 2, 1)
        rnn_out, _ = self.rnn(conv_feat)
        output = self.classifier(rnn_out)
        return output


class PlateRecognizer:
    def __init__(self, model_path):
        """
        Инициализация распознавателя текста номеров
        
        Args:
            model_path (str): Путь к весам модели CRNN
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = CRNN(Config.VOCAB_SIZE, Config.HIDDEN_SIZE, Config.NUM_LAYERS)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((Config.IMG_HEIGHT, Config.IMG_WIDTH)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def ctc_decode(self, predictions, blank_idx=Config.BLANK_TOKEN):
        """
        Декодирование CTC выхода
        
        Args:
            predictions: Массив предсказаний
            blank_idx: Индекс пустого токена
            
        Returns:
            str: Декодированный текст
        """
        decoded = []
        prev_idx = blank_idx

        for idx in predictions:
            if idx != blank_idx and idx != prev_idx:
                decoded.append(Config.idx_to_char(idx))
            prev_idx = idx

        return ''.join(decoded)
    
    def recognize_from_path(self, image_path):
        """
        Распознает текст номера из файла изображения
        
        Args:
            image_path (str): Путь к изображению номера
            
        Returns:
            str: Распознанный текст
        """
        image = Image.open(image_path).convert('RGB')
        return self.recognize_from_image(image)
    
    def recognize_from_image(self, image):
        """
        Распознает текст номера из PIL изображения
        
        Args:
            image (PIL.Image): Изображение номера
            
        Returns:
            str: Распознанный текст
        """
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(image_tensor)
            pred_indices = torch.argmax(output[0], dim=1).cpu().numpy()
            return self.ctc_decode(pred_indices)
    
    def recognize_batch(self, images):
        """
        Распознает текст для батча изображений
        
        Args:
            images (list): Список PIL изображений
            
        Returns:
            list: Список распознанных текстов
        """
        results = []
        for image in images:
            text = self.recognize_from_image(image)
            results.append(text)
        return results
