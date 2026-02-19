#!/usr/bin/env python3
"""Thin wrapper: run GraviQ Flask demo app."""

import os
import sys

# Ensure project root is on path and templates are found next to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Run from project root so paths like checkpoints/, training_data/ work
os.chdir(PROJECT_ROOT)

from graviq.app_flask import create_app

app = create_app()

if __name__ == '__main__':
    import torch
    port = int(os.environ.get('PORT', 5000))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("\n" + "=" * 60)
    print("GraviQ Tunnel Detection Demo Server")
    print("=" * 60)
    print(f"Device: {device}")
    print("Model: U-Net (380K params)")
    print(f"\nStarting server at http://localhost:{port}")
    print("(Use PORT=5001 make run-app if 5000 is already in use)")
    print("Press Ctrl+C to stop")
    print("=" * 60 + "\n")
    app.run(debug=True, host='0.0.0.0', port=port)
