from pathlib import Path
from src.presentation.vivo_theme_engine import build_vivo_style_presentation

payload = Path("examples/demo_payload.json")
output = Path("outputs/presentations/PMTS_Rollout_Report_Demo.pptx")

result = build_vivo_style_presentation(payload, output)
print(f"Apresentação criada: {result}")
