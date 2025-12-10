import sys
import os
import config
import logging
import webbrowser

# 1. 텐서플로우 및 시스템 로그 레벨 조정 (최우선)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

import torch
import tensorflow as tf
from tensorflow.keras import mixed_precision

# 성능 최적화: Mixed Precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

from tensorboard import program
from config import SessionManager
from strategies.pipeline_trainer import PipelineTrainer

def setup_global_logging(log_file):
    # 루트 로거 초기화
    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)
            
    # 포맷 설정
    formatter = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%H:%M:%S')

    # 파일 핸들러
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setFormatter(formatter)
    
    # 콘솔 핸들러
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    
    logging.basicConfig(
        level=logging.INFO,
        handlers=[file_handler, stream_handler],
        force=True
    )

def silence_annoying_loggers():
    """소음을 유발하는 로거들의 입을 강제로 막는 함수"""
    noisy_loggers = [
        'werkzeug', 'tensorboard', 'tensorflow', 
        'h5py', 'matplotlib', 'absl', 'urllib3', 'requests'
    ]
    
    for name in noisy_loggers:
        logger = logging.getLogger(name)
        # 1. 레벨을 CRITICAL보다 높은 수준으로 설정 (아무것도 출력 안 함)
        logger.setLevel(logging.CRITICAL + 1)
        # 2. 전파 차단
        logger.propagate = False
        # 3. 기존 핸들러 모두 제거
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

def launch_tensorboard(log_dir):
    try:
        # [1차 차단]
        silence_annoying_loggers()
        
        tb = program.TensorBoard()
        # reload_interval을 크게 설정하여 로그 갱신 빈도 줄임
        tb.configure(argv=[None, '--logdir', log_dir, '--port', '6006', '--reload_interval', '300'])
        url = tb.launch()
        
        # [2차 차단 - 핵심] 텐서보드 실행 직후 다시 한번 로거를 죽임 (재설정 방지)
        silence_annoying_loggers()
        
        # 접속 로그가 아닌 단순 안내 메시지만 출력
        logging.info(f">> TensorBoard started: {url}")
        webbrowser.open(url)
    except Exception as e:
        logging.warning(f">> TB Launch failed: {e}")

if __name__ == "__main__":
    gpus = tf.config.list_physical_devices('GPU')
    gpu_msg = f"GPU: {gpus[0].name}" if gpus else "GPU: None"

    session = SessionManager()
    paths = session.create()
    
    setup_global_logging(paths['log_file'])
    
    # 시작 전에도 차단 실행
    silence_annoying_loggers()
    
    logging.info(f"{'='*50}")
    logging.info(f" SESSION: {paths['id']} | {gpu_msg}")
    logging.info(f"{'='*50}")

    launch_tensorboard(config.LOG_BASE_DIR)

    trainer = PipelineTrainer(paths)
    trainer.run_all()