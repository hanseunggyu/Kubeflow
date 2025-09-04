# Kubeflow MNIST MLOps Pipeline

자동화된 MNIST 손글씨 숫자 분류 MLOps 파이프라인

## 프로젝트 개요

**데이터 준비**: MNIST 데이터셋을 활용해 손글씨 숫자 이미지를 학습. 전체 데이터셋의 일부(30%)를 사용하여 실험 속도를 최적화.

**모델 학습**: 간단한 2층 신경망(784 → 128 → 10)을 구성, Katib를 활용해 학습률, 배치 크기 등 하이퍼파라미터를 자동 탐색.

**자동화 파이프라인**: Kubeflow Pipelines로 데이터 전처리 → 모델 학습 → 모델 저장/배포까지 자동화.

**모델 서빙**: 최적화된 모델을 KServe로 배포하여 REST API 형태의 예측 서비스 제공.

**클라우드 인프라**: AWS EC2(t2.2xlarge, 32GiB RAM, 100GiB Disk)에 Kubeflow 환경을 구성, 모델/데이터 저장소는 GCS(서울 리전) 활용.

## 프로젝트 구조

```
kubeflow/
├── katib/                          # 하이퍼파라미터 최적화
│   ├── template.yaml               # Katib Experiment 정의
│   ├── trainer.py                  # MNIST 모델 훈련 코드
│   ├── Dockerfile                  # 컨테이너 이미지 빌드
│   └── requirements.txt            # Python 의존성
├── pipeline/                       # ML 파이프라인
│   ├── integrated_mnist_pipeline.py # 통합 MLOps 파이프라인
│   └── requirements.txt            # 파이프라인 의존성
└── kserve/                         # 모델 서빙
    └── deployment_mnist/           # MNIST 모델 서빙
        ├── inference-service.yaml  # KServe 배포 설정
        └── mnist-input.json        # 테스트용 입력 데이터
```

## 워크플로우

### 1. 데이터 전처리 & 모델 학습
```bash
# Katib 하이퍼파라미터 최적화 실행
kubectl apply -f katib/template.yaml

# 통합 파이프라인 실행
python pipeline/integrated_mnist_pipeline.py
```

### 2. 모델 저장 위치
- **GCS 경로**: `gs://240924_gkstmdrb/mnist-model/v1/model.pt`
- 모델 구조: PyTorch Sequential (784→128→10)
- 메타데이터: 학습률, 배치 크기, 정확도, 손실값 포함

### 3. 모델 서빙 & API 테스트
```bash
# KServe InferenceService 배포
kubectl apply -f kserve/deployment_mnist/inference-service.yaml

# 예측 API 테스트
curl -X POST http://mnist-classifier.kubeflow/v1/models/mnist-classifier:predict \
  -H "Content-Type: application/json" \
  -d @kserve/deployment_mnist/mnist-input.json
```

## 주요 기능

### Katib 하이퍼파라미터 최적화
- **탐색 범위**: 학습률(0.01-0.05), 배치 크기(1-64)
- **목표**: Loss 최소화 (목표값: 0.001)
- **알고리즘**: Random Search
- **병렬 실험**: 3개, 최대 시행: 12회

### 통합 파이프라인 구성요소
1. **데이터 전처리**: MNIST 다운로드 & 30% 서브셋 생성
2. **모델 학습**: 2층 신경망 훈련
3. **모델 업로드**: GCS 저장소에 자동 업로드
4. **서빙 배포**: KServe InferenceService 자동 생성

### API 응답 예시
```json
{
  "predictions": [7]  // 0-9 숫자 예측 결과
}
```

##  효과

- **자동화**: 수동 튜닝 대비 실험 시간 단축 및 최적 성능 조합 자동 탐색
- **재현성**: ML Workflow를 코드/파이프라인 단위로 관리해 재현성과 확장성 확보  
- **MLOps**: 모델 학습부터 서빙까지 완전 자동화된 MLOps 구현

## 요구사항

- Kubeflow 1.7+
- Kubernetes 1.21+
- Python 3.9+
- PyTorch 1.12+
- Google Cloud Storage 접근 권한

---