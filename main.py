import sys
import os
import logging
import webbrowser
# [잡음 제거 1] TensorFlow 로그 레벨 조정 (Warning 이상만 출력)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

import torch # PyTorch 먼저 임포트 (DLL 충돌 방지)
import tensorflow as tf
from tensorflow.keras import mixed_precision

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

from tensorboard import program
from config import SessionManager
from strategies.pipeline_trainer import PipelineTrainer

def setup_global_logging(log_file):
    root = logging.getLogger()
    if root.handlers:
        for handler in root.handlers:
            root.removeHandler(handler)
            
    # 포맷을 깔끔하게 변경 (시간 | 레벨 | 메시지)
    formatter = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%H:%M:%S')

    # 파일 핸들러 (모든 중요 로그 저장)
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

    # [잡음 제거 2] TensorBoard(Werkzeug) 및 기타 라이브러리 로그 차단
    logging.getLogger('werkzeug').setLevel(logging.ERROR)
    logging.getLogger('tensorflow').setLevel(logging.ERROR)
    logging.getLogger('h5py').setLevel(logging.ERROR)

def launch_tensorboard(log_dir):
    try:
        # TensorBoard 자체 로그도 숨김
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        tb = program.TensorBoard()
        tb.configure(argv=[None, '--logdir', log_dir, '--port', '6006'])
        url = tb.launch()
        logging.info(f">> TensorBoard started: {url}") # 깔끔하게 한 줄만
        webbrowser.open(url)
    except Exception as e:
        logging.warning(f">> TB Launch failed: {e}")

if __name__ == "__main__":
    # GPU 정보도 간략하게 한 줄로
    gpus = tf.config.list_physical_devices('GPU')
    gpu_msg = f"GPU: {gpus[0].name}" if gpus else "GPU: None"

    session = SessionManager()
    paths = session.create()
    
    setup_global_logging(paths['log_file'])
    
    logging.info(f"{'='*50}")
    logging.info(f" SESSION: {paths['id']} | {gpu_msg}")
    logging.info(f"{'='*50}")

    launch_tensorboard(paths['tb'])

    trainer = PipelineTrainer(paths)
    trainer.run_all()