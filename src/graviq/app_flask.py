"""Flask web app for GraviQ tunnel detection demo."""

import os
import random
from flask import Flask, render_template, request, jsonify
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64

from graviq.models import UNet
from graviq.data import make_grid
from graviq.sim import gzz_approximation
from graviq.inference import predict as model_predict


def create_app(template_folder=None):
    if template_folder is None:
        # Use absolute path relative to this module (not cwd) so templates work
        # regardless of where the app is started from
        template_folder = os.path.abspath(
            os.path.join(os.path.dirname(__file__), 'templates')
        )
    app = Flask(__name__, template_folder=template_folder)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = UNet(in_channels=1, out_channels=1)
    checkpoint_path = os.path.join(os.getcwd(), 'checkpoints', 'best_model.pth')
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded! (Epoch {checkpoint.get('epoch', 'N/A')}, Val Dice: {checkpoint.get('val_dice', 'N/A'):.4f})")
    else:
        print("WARNING: No trained model found. Please train first!")
    model = model.to(device)
    model.eval()

    def predict_tunnel(density_grid, threshold=0.5):
        prob_map, binary_mask, has_tunnel = model_predict(model, density_grid, device, threshold)
        confidence = float(prob_map.max())
        tunnel_pixels = int(binary_mask.sum())
        return prob_map, binary_mask, has_tunnel, confidence, tunnel_pixels

    def create_visualization(density_grid, gzz_grid, prob_map, binary_mask, tunnel_mask=None):
        #We keep 4 plots, but the 4th one becomes the "X-Ray" if ground truth exists
        fig, axes = plt.subplots(1, 4, figsize=(18, 4))

        #1. Density Grid
        im0 = axes[0].imshow(density_grid, cmap='inferno', origin='upper')
        axes[0].set_title('Density Grid', fontsize=12, fontweight='bold')
        axes[0].axis('off')

        #2. Quantum Sensor Data
        im1 = axes[1].imshow(gzz_grid, cmap='viridis', origin='upper')
        axes[1].set_title('Gzz Grid (Quantum)', fontsize=12, fontweight='bold')
        axes[1].axis('off')

        #3. Raw AI Confidence
        im2 = axes[2].imshow(prob_map, cmap='hot', origin='upper', vmin=0, vmax=1)
        axes[2].set_title('AI Confidence Map', fontsize=12, fontweight='bold')
        axes[2].axis('off')

        #4. X-Ray of Detected Tunnels
        if tunnel_mask is not None:
            #Create RGB overlay: Green = Correct, Red = Missed, Blue = False Positive
            h, w = binary_mask.shape
            xray = np.zeros((h, w, 3), dtype=np.uint8)
            pred = (binary_mask > 0.5)
            gt = (tunnel_mask > 0.5)

            xray[..., 1] = (pred & gt) * 255 #Green: True Positive
            xray[..., 0] = (gt & ~pred) * 255 #Red: False Negative
            xray[..., 2] = (pred & ~gt) * 255 #Blue: False Positive

            axes[3].imshow(xray.astype(np.uint8), origin='upper')
            axes[3].set_title('Quantum X-Ray Analysis', fontsize=12, fontweight='bold')
        else:
            axes[3].imshow(binary_mask, cmap='binary', origin='upper', vmin=0, vmax=1)
            axes[3].set_title('Detected Tunnels', fontsize=12, fontweight='bold')

        axes[3].axis('off')
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        return img_base64

    def generate_random_grid():
        seed = random.randint(0, 10000)
        grid, tunnel_mask, metadata = make_grid(seed)
        return grid, tunnel_mask, metadata

    @app.route('/')
    def index():
        return render_template('index.html')

    @app.route('/generate', methods=['POST'])
    def generate():
        try:
            density_grid, tunnel_mask, metadata = generate_random_grid()
            gzz_grid = gzz_approximation(density_grid, t_evolution=30e-6)
            prob_map, binary_mask, has_tunnel, confidence, tunnel_pixels = predict_tunnel(density_grid)
            img_base64 = create_visualization(density_grid, gzz_grid, prob_map, binary_mask, tunnel_mask)
            return jsonify({
                'success': True,
                'image': img_base64,
                'has_tunnel': bool(has_tunnel),
                'confidence': float(confidence),
                'tunnel_pixels': int(tunnel_pixels),
                'ground_truth': {
                    'has_tunnel': metadata['has_tunnel'],
                    'num_tunnels': metadata['num_tunnels']
                }
            })
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500

    @app.route('/upload', methods=['POST'])
    def upload():
        try:
            if 'file' not in request.files:
                return jsonify({'success': False, 'error': 'No file uploaded'}), 400
            file = request.files['file']
            if file.filename == '':
                return jsonify({'success': False, 'error': 'No file selected'}), 400
            if not file.filename.endswith('.npy'):
                return jsonify({'success': False, 'error': 'Please upload a .npy file'}), 400
            density_grid = np.load(file)
            if density_grid.shape != (60, 150):
                return jsonify({
                    'success': False,
                    'error': f'Invalid grid shape {density_grid.shape}. Expected (60, 150)'
                }), 400
            gzz_grid = None
            if file.filename.startswith('density_grid_'):
                sample_id = file.filename.replace('density_grid_', '').replace('.npy', '')
                gzz_path = os.path.join('training_data', f'gzz_grid_{sample_id}.npy')
                if os.path.exists(gzz_path):
                    gzz_grid = np.load(gzz_path)
            if gzz_grid is None:
                gzz_grid = gzz_approximation(density_grid, t_evolution=30e-6)
            prob_map, binary_mask, has_tunnel, confidence, tunnel_pixels = predict_tunnel(density_grid)
            img_base64 = create_visualization(density_grid, gzz_grid, prob_map, binary_mask)
            return jsonify({
                'success': True,
                'image': img_base64,
                'has_tunnel': bool(has_tunnel),
                'confidence': float(confidence),
                'tunnel_pixels': int(tunnel_pixels)
            })
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500

    @app.route('/predict', methods=['POST'])
    def predict_route():
        try:
            data = request.get_json()
            threshold = float(data.get('threshold', 0.5))
            return jsonify({'success': True, 'threshold': threshold})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500

    return app
