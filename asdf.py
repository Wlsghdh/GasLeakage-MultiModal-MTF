import torch
from torch.utils.mobile_optimizer import optimize_for_mobile
from src.models.multitask_fusion_model import MultitaskFusionModel
from src.config import DEVICE

# ==============================
# 1. 모델 로드
# ==============================
MODEL_PATH = 'Multitask_fusion_model.pt'

model = MultitaskFusionModel().to(DEVICE)
state = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state, strict=True)
model.eval()

# ==============================
# 2. 더미 입력 생성
# ==============================
# Sensor input: [batch, features]
dummy_sensor = torch.randn(1, 3,244,244).to(DEVICE)  

# Thermal input: [batch, channels, height, width]
# 학습할 때 사용한 채널, H, W로 맞춰야 함
dummy_thermal = torch.randn(1, 3, 224, 224).to(DEVICE)

# ==============================
# 3. TorchScript 변환
# ==============================
ts_model = torch.jit.trace(model, (dummy_sensor, dummy_thermal))

# ==============================
# 4. 모바일 최적화
# ==============================
ts_model_optimized = optimize_for_mobile(ts_model)
ts_model_optimized._save_for_lite_interpreter("Multitask_fusion_model.ptl")

print("✅ 변환 완료: Multitask_fusion_model.ptl 저장됨")
