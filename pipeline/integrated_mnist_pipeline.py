import kfp
from kfp import dsl
from kfp.dsl import component, Input, Output, Artifact, Model
from typing import NamedTuple


@component(
    base_image="python:3.9",
    packages_to_install=["torch", "torchvision", "pydantic-settings", "tqdm", "numpy"]
)
def data_preprocessing_op(
    dataset_path: Output[Artifact],
    subset_ratio: float = 0.3
) -> NamedTuple('DataStats', [('total_samples', int), ('subset_samples', int)]):
    """데이터 전처리: MNIST 데이터셋 다운로드 및 서브셋 생성"""
    import torch
    from torchvision import datasets, transforms
    import numpy as np
    import os
    
    # MNIST 데이터셋 다운로드
    full_dataset = datasets.MNIST(
        "./data", train=True, download=True, transform=transforms.ToTensor()
    )
    
    # 서브셋 생성 (30%)
    subset_size = int(subset_ratio * len(full_dataset))
    indices = np.random.choice(len(full_dataset), subset_size, replace=False)
    
    # 데이터셋 정보 저장
    os.makedirs(dataset_path.path, exist_ok=True)
    torch.save({
        'indices': indices,
        'subset_size': subset_size,
        'total_size': len(full_dataset)
    }, f"{dataset_path.path}/dataset_info.pt")
    
    from collections import namedtuple
    DataStats = namedtuple('DataStats', ['total_samples', 'subset_samples'])
    return DataStats(len(full_dataset), subset_size)


@component(
    base_image="python:3.9",
    packages_to_install=["torch", "torchvision", "pydantic-settings", "tqdm", "numpy", "google-cloud-storage"]
)
def model_training_op(
    dataset_info: Input[Artifact],
    trained_model: Output[Model],
    learning_rate: float = 0.01,
    batch_size: int = 64,
    epochs: int = 10
) -> NamedTuple('TrainingResults', [('final_loss', float), ('accuracy', float)]):
    """모델 학습"""
    import torch
    import torch.nn as nn
    from torch.utils.data import Subset, DataLoader
    from torchvision import datasets, transforms
    from tqdm import tqdm
    import numpy as np
    import os
    
    # 데이터셋 정보 로드
    dataset_info_data = torch.load(f"{dataset_info.path}/dataset_info.pt")
    indices = dataset_info_data['indices']
    
    # 데이터셋 준비
    full_dataset = datasets.MNIST(
        "./data", train=True, download=True, transform=transforms.ToTensor()
    )
    subset_dataset = Subset(full_dataset, indices)
    train_loader = DataLoader(subset_dataset, batch_size=batch_size, shuffle=True)
    
    # 모델 정의
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    
    # 학습
    model.train()
    total_loss, correct, total = 0, 0, 0
    
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_correct = 0
        epoch_total = 0
        
        for data, target in train_loader:
            optimizer.zero_grad()
            outputs = model(data)
            loss = loss_fn(outputs, target)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * data.size(0)
            _, predicted = torch.max(outputs.data, 1)
            epoch_total += target.size(0)
            epoch_correct += (predicted == target).sum().item()
        
        total_loss += epoch_loss
        correct += epoch_correct
        total += epoch_total
    
    final_loss = total_loss / total
    accuracy = correct / total
    
    # 모델 저장
    os.makedirs(trained_model.path, exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'input_size': 784,
            'hidden_size': 128,
            'output_size': 10
        },
        'training_stats': {
            'final_loss': final_loss,
            'accuracy': accuracy,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'epochs': epochs
        }
    }, f"{trained_model.path}/model.pt")
    
    from collections import namedtuple
    TrainingResults = namedtuple('TrainingResults', ['final_loss', 'accuracy'])
    return TrainingResults(final_loss, accuracy)


@component(
    base_image="google/cloud-sdk:latest",
    packages_to_install=["google-cloud-storage"]
)
def model_upload_op(
    trained_model: Input[Model],
    model_version: str = "v1"
) -> str:
    """훈련된 모델을 GCS에 업로드"""
    from google.cloud import storage
    import os
    
    # GCS 설정
    bucket_name = "240924_gkstmdrb"
    model_path = f"mnist-model/{model_version}/model.pt"
    
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(model_path)
    blob.upload_from_filename(f"{trained_model.path}/model.pt")
    
    gcs_uri = f"gs://{bucket_name}/{model_path}"
    print(f"Model uploaded to: {gcs_uri}")
    return gcs_uri


@component(
    base_image="python:3.9",
    packages_to_install=["kubernetes", "pyyaml"]
)
def deploy_kserve_op(
    model_gcs_uri: str,
    service_name: str = "mnist-classifier",
    namespace: str = "kubeflow-user-example-com"
) -> str:
    """KServe InferenceService 배포"""
    import yaml
    
    inference_service = {
        'apiVersion': 'serving.kserve.io/v1beta1',
        'kind': 'InferenceService',
        'metadata': {
            'name': service_name,
            'namespace': namespace
        },
        'spec': {
            'predictor': {
                'serviceAccountName': 'kserve-storage-access',
                'pytorch': {
                    'storageUri': model_gcs_uri,
                    'resources': {
                        'requests': {
                            'cpu': '1',
                            'memory': '1Gi'
                        },
                        'limits': {
                            'cpu': '1',
                            'memory': '1Gi'
                        }
                    }
                }
            }
        }
    }
    
    # YAML 파일로 저장
    with open('/tmp/inference-service.yaml', 'w') as f:
        yaml.dump(inference_service, f)
    
    print("InferenceService manifest created at /tmp/inference-service.yaml")
    print("Apply with: kubectl apply -f /tmp/inference-service.yaml")
    
    return f"InferenceService {service_name} ready for deployment"


@dsl.pipeline(
    name="integrated-mnist-mlops-pipeline",
    description="Complete MLOps pipeline: Data → Training → Model Upload → Serving"
)
def integrated_mnist_pipeline(
    subset_ratio: float = 0.3,
    learning_rate: float = 0.01,
    batch_size: int = 64,
    epochs: int = 10,
    model_version: str = "v1"
):
    """통합 MNIST MLOps 파이프라인 - gs://240924_gkstmdrb/mnist-model/ 사용"""
    
    # 1. 데이터 전처리
    data_prep_task = data_preprocessing_op(subset_ratio=subset_ratio)
    
    # 2. 모델 학습
    training_task = model_training_op(
        dataset_info=data_prep_task.outputs['dataset_path'],
        learning_rate=learning_rate,
        batch_size=batch_size,
        epochs=epochs
    )
    
    # 3. 모델 GCS 업로드
    upload_task = model_upload_op(
        trained_model=training_task.outputs['trained_model'],
        model_version=model_version
    )
    
    # 4. KServe 배포
    deploy_task = deploy_kserve_op(
        model_gcs_uri=upload_task.output,
        service_name="mnist-classifier"
    )


if __name__ == "__main__":
    # 파이프라인 컴파일
    kfp.compiler.Compiler().compile(
        pipeline_func=integrated_mnist_pipeline,
        package_path="build/integrated_mnist_pipeline.tar.gz"
    )
    print("Pipeline compiled to: build/integrated_mnist_pipeline.tar.gz")