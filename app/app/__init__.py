"""PCB Defect Inspector 애플리케이션 패키지."""
from .inference import CLASS_KOR, CLASS_NAMES, Detection, PCBDetector, verdict

__all__ = ["PCBDetector", "verdict", "Detection", "CLASS_NAMES", "CLASS_KOR"]
