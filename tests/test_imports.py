"""Test training manager import"""
import sys
sys.path.insert(0, ".")

try:
    from mars_lite.server.training_manager import TrainingManager, TrainingConfig
    print("TrainingManager import: OK")
    
    tc = TrainingConfig()
    print(f"TrainingConfig: {tc.to_dict()}")
    
    from mars_lite.server.metrics_server import create_app
    print("metrics_server import: OK")
    
    print("\nAll imports successful!")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
