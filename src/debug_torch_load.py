import torch
from pathlib import Path

def debug_load():
    """
    一個簡單的腳本，專門用來測試直接使用 torch.load 載入模型檔案。
    這有助於將問題與 pyannote/lightning 的載入器隔離。
    """
    model_path = Path(__file__).parent.parent / "models" / "direct" / "segmentation-3.0" / "pytorch_model.bin"
    
    print("="*50)
    print("🚀 Torch.load() 直接載入測試 🚀")
    print("="*50)
    
    print(f"[*] 準備載入模型檔案: {model_path}")
    
    if not model_path.exists():
        print(f"[!] 錯誤: 檔案不存在！")
        return
        
    print(f"[*] 檔案大小: {model_path.stat().st_size / (1024*1024):.2f} MB")
    
    try:
        print("[*] 正在執行 torch.load()...")
        # 嘗試用 CPU 載入，避免任何 CUDA 問題
        checkpoint = torch.load(model_path, map_location="cpu")
        print("\n[✅] torch.load() 成功！")
        
        if isinstance(checkpoint, dict):
            print("[*] 檔案內容是一個字典 (標準的 PyTorch checkpoint 格式)。")
            print("[*] 主要鍵 (Keys):")
            for key in checkpoint.keys():
                print(f"    - {key}")
            if 'state_dict' in checkpoint:
                print("[*] 找到了 'state_dict'，這是模型的權重。")
        else:
            print(f"[*] 檔案內容類型: {type(checkpoint)}")

    except Exception as e:
        print(f"\n[❌] torch.load() 失敗！")
        print(f"[*] 錯誤類型: {type(e).__name__}")
        print(f"[*] 錯誤訊息: {e}")
        print("\n[*] 這證實了問題出在 PyTorch 載入層面，與 pyannote 無關。")
        print("[*] 這極有可能是模型檔案與你目前的 PyTorch/Python 版本不相容所致。")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_load()
