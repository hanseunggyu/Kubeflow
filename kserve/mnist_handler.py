import torch
import torch.nn as nn
from ts.torch_handler.base_handler import BaseHandler
import json
import numpy as np


class MNISTHandler(BaseHandler):
    """MNIST 숫자 분류를 위한 커스텀 핸들러"""
    
    def initialize(self, context):
        """모델 초기화"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 모델 구조 정의 (trainer.py와 동일)
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )
        
        # 저장된 모델 가중치 로드
        model_data = torch.load(context.manifest['model']['modelFile'], map_location=self.device)
        self.model.load_state_dict(model_data['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"MNIST model loaded on {self.device}")
        
    def preprocess(self, data):
        """입력 데이터 전처리"""
        # JSON에서 이미지 데이터 추출
        images = []
        for item in data:
            if isinstance(item, dict):
                # REST API 요청 처리
                instances = item.get("instances", [item.get("data", [])])
            else:
                # 직접 배열 입력 처리
                instances = [item]
                
            for instance in instances:
                # 784 크기의 1차원 배열을 28x28로 변환
                image_array = np.array(instance, dtype=np.float32)
                if image_array.shape != (784,):
                    image_array = image_array.reshape(784)
                images.append(image_array)
        
        # PyTorch 텐서로 변환
        tensor = torch.FloatTensor(images).to(self.device)
        return tensor
    
    def inference(self, data, *args, **kwargs):
        """모델 추론 실행"""
        with torch.no_grad():
            outputs = self.model(data)
            # 가장 높은 확률의 클래스 선택
            predictions = torch.argmax(outputs, dim=1)
        return predictions.cpu().numpy()
    
    def postprocess(self, data):
        """추론 결과 후처리"""
        return {"predictions": data.tolist()}