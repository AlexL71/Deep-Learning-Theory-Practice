from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model
import numpy as np
import os
from tqdm import tqdm

# VGG16 모델 불러오기 (마지막 분류기 제외, 평균 풀링 사용 -> 512차원 벡터 출력)
base_model = VGG16(weights='imagenet', include_top=False, pooling='avg')
model = Model(inputs=base_model.input, outputs=base_model.output)

# 이미지 특징 추출 함수
def extract_features(directory):
    features = {}
    print("이미지 특징 추출을 시작")
    
    # 이미지 파일 목록 가져오기
    files = [f for f in os.listdir(directory) if f.endswith('.jpg')]
    
    for name in tqdm(files):
        # 이미지 로드 및 크기 조정 (VGG16 기본 크기: 224x224)
        filename = directory + '/' + name
        image = load_img(filename, target_size=(224, 224))
        
        # 배열 변환 및 전처리
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image)
        
        # 특징 추출 (예측)
        feature = model.predict(image, verbose=0)
        
        # 이미지 ID(파일명)를 키로 저장, 1차원 벡터로 변형
        features[name] = feature.flatten() # (512,) 형태로 저장
        
    return features

# 실행 및 저장
directory = 'archive/Images' 
features = extract_features(directory)

print('추출된 특징 개수:', len(features))

# 파일로 저장
np.save('archive/features.npy', features)
print("저장 완료: archive/features.npy")