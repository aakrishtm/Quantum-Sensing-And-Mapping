"""Thin wrapper: run GraviQ Flask demo (delegates to graviq.app_flask)."""
from graviq.app_flask import create_app
import torch

app = create_app()

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("\n" + "=" * 60)
    print("GraviQ Tunnel Detection Demo Server")
    print("=" * 60)
    print(f"Device: {device}")
    print("Model: U-Net (380K params)")
    print(f"\nStarting server at http://localhost:{port}")
    print("(Set PORT=5001 if 5000 is already in use)")
    print("Press Ctrl+C to stop")
    print("=" * 60 + "\n")
    app.run(debug=True, host='0.0.0.0', port=port)
