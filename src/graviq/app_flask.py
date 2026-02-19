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
from graviq.baselines import baseline_predict


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

    def compute_metrics(pred_mask, gt_mask):
        """Compute Dice and IoU metrics."""
        pred = (pred_mask > 0.5).astype(np.float32)
        gt = (gt_mask > 0.5).astype(np.float32)
        intersection = (pred * gt).sum()
        union = pred.sum() + gt.sum() - intersection
        dice = (2 * intersection + 1e-7) / (pred.sum() + gt.sum() + 1e-7)
        iou = (intersection + 1e-7) / (union + 1e-7)
        return float(dice), float(iou)

    def create_visualization(density_grid, gzz_grid, prob_map, binary_mask, baseline_mask=None, tunnel_mask=None):
        """Create comparison visualization: AI vs Baseline."""
        if baseline_mask is not None and tunnel_mask is not None:
            # Comparison view: AI vs Baseline side-by-side
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            fig.patch.set_facecolor('#0b0e14')
            
            # Row 1: Input data
            im0 = axes[0, 0].imshow(density_grid, cmap='inferno', origin='upper')
            axes[0, 0].set_title('Density Grid', fontsize=11, fontweight='bold', color='#e0e6ed')
            axes[0, 0].axis('off')
            axes[0, 0].set_facecolor('#1a1f29')
            
            im1 = axes[0, 1].imshow(gzz_grid, cmap='viridis', origin='upper')
            axes[0, 1].set_title('Quantum Sensor (Gzz)', fontsize=11, fontweight='bold', color='#e0e6ed')
            axes[0, 1].axis('off')
            axes[0, 1].set_facecolor('#1a1f29')
            
            im2 = axes[0, 2].imshow(prob_map, cmap='hot', origin='upper', vmin=0, vmax=1)
            axes[0, 2].set_title('AI Confidence Map', fontsize=11, fontweight='bold', color='#4ade80')
            axes[0, 2].axis('off')
            axes[0, 2].set_facecolor('#1a1f29')
            
            # Row 2: Predictions comparison
            # AI prediction with overlay
            h, w = binary_mask.shape
            ai_overlay = np.zeros((h, w, 3), dtype=np.uint8)
            ai_pred = (binary_mask > 0.5)
            gt = (tunnel_mask > 0.5)
            ai_overlay[..., 1] = (ai_pred & gt) * 255  # Green: TP
            ai_overlay[..., 0] = (gt & ~ai_pred) * 255  # Red: FN
            ai_overlay[..., 2] = (ai_pred & ~gt) * 255  # Blue: FP
            
            axes[1, 0].imshow(ai_overlay.astype(np.uint8), origin='upper')
            axes[1, 0].set_title('AI Prediction (U-Net)', fontsize=11, fontweight='bold', color='#4ade80')
            axes[1, 0].axis('off')
            axes[1, 0].set_facecolor('#1a1f29')
            
            # Baseline prediction with overlay
            baseline_overlay = np.zeros((h, w, 3), dtype=np.uint8)
            baseline_pred = (baseline_mask > 0.5)
            baseline_overlay[..., 1] = (baseline_pred & gt) * 255  # Green: TP
            baseline_overlay[..., 0] = (gt & ~baseline_pred) * 255  # Red: FN
            baseline_overlay[..., 2] = (baseline_pred & ~gt) * 255  # Blue: FP
            
            axes[1, 1].imshow(baseline_overlay.astype(np.uint8), origin='upper')
            axes[1, 1].set_title('Baseline (Threshold)', fontsize=11, fontweight='bold', color='#8892b0')
            axes[1, 1].axis('off')
            axes[1, 1].set_facecolor('#1a1f29')
            
            # Ground truth
            axes[1, 2].imshow(tunnel_mask, cmap='binary', origin='upper', vmin=0, vmax=1)
            axes[1, 2].set_title('Ground Truth', fontsize=11, fontweight='bold', color='#e0e6ed')
            axes[1, 2].axis('off')
            axes[1, 2].set_facecolor('#1a1f29')
            
        else:
            # Simple view without comparison
            fig, axes = plt.subplots(1, 4, figsize=(18, 4))
            fig.patch.set_facecolor('#0b0e14')
            
            im0 = axes[0].imshow(density_grid, cmap='inferno', origin='upper')
            axes[0].set_title('Density Grid', fontsize=12, fontweight='bold', color='#e0e6ed')
            axes[0].axis('off')
            axes[0].set_facecolor('#1a1f29')
            
            im1 = axes[1].imshow(gzz_grid, cmap='viridis', origin='upper')
            axes[1].set_title('Gzz Grid (Quantum)', fontsize=12, fontweight='bold', color='#e0e6ed')
            axes[1].axis('off')
            axes[1].set_facecolor('#1a1f29')
            
            im2 = axes[2].imshow(prob_map, cmap='hot', origin='upper', vmin=0, vmax=1)
            axes[2].set_title('AI Confidence Map', fontsize=12, fontweight='bold', color='#4ade80')
            axes[2].axis('off')
            axes[2].set_facecolor('#1a1f29')
            
            axes[3].imshow(binary_mask, cmap='binary', origin='upper', vmin=0, vmax=1)
            axes[3].set_title('Detected Tunnels', fontsize=12, fontweight='bold', color='#e0e6ed')
            axes[3].axis('off')
            axes[3].set_facecolor('#1a1f29')
        
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='#0b0e14')
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
            
            # AI prediction
            prob_map, binary_mask, has_tunnel, confidence, tunnel_pixels = predict_tunnel(gzz_grid)
            ai_dice, ai_iou = compute_metrics(binary_mask, tunnel_mask)
            
            # Baseline prediction
            baseline_mask = baseline_predict(gzz_grid, {'z': 2.0, 'min_size': 50})
            baseline_dice, baseline_iou = compute_metrics(baseline_mask, tunnel_mask)
            
            img_base64 = create_visualization(density_grid, gzz_grid, prob_map, binary_mask, baseline_mask, tunnel_mask)
            return jsonify({
                'success': True,
                'image': img_base64,
                'ai': {
                    'has_tunnel': bool(has_tunnel),
                    'confidence': float(confidence),
                    'tunnel_pixels': int(tunnel_pixels),
                    'dice': ai_dice,
                    'iou': ai_iou
                },
                'baseline': {
                    'dice': baseline_dice,
                    'iou': baseline_iou,
                    'tunnel_pixels': int(baseline_mask.sum())
                },
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
            
            # Try to load as Gzz grid first (for trained data), fallback to density
            uploaded_grid = np.load(file)
            if uploaded_grid.shape != (60, 150):
                return jsonify({
                    'success': False,
                    'error': f'Invalid grid shape {uploaded_grid.shape}. Expected (60, 150)'
                }), 400
            
            # Determine if this is a density or gzz grid
            if file.filename.startswith('gzz_grid_'):
                gzz_grid = uploaded_grid
                density_grid = None  # Will approximate if needed
            elif file.filename.startswith('density_grid_'):
                density_grid = uploaded_grid
                sample_id = file.filename.replace('density_grid_', '').replace('.npy', '')
                gzz_path = os.path.join('training_data', f'gzz_grid_{sample_id}.npy')
                if os.path.exists(gzz_path):
                    gzz_grid = np.load(gzz_path)
                else:
                    gzz_grid = gzz_approximation(density_grid, t_evolution=30e-6)
            else:
                # Assume it's a density grid
                density_grid = uploaded_grid
                gzz_grid = gzz_approximation(density_grid, t_evolution=30e-6)
            
            # Use density_grid for visualization if available, otherwise use gzz_grid
            viz_grid = density_grid if density_grid is not None else gzz_grid
            
            # AI prediction on Gzz grid
            prob_map, binary_mask, has_tunnel, confidence, tunnel_pixels = predict_tunnel(gzz_grid)
            
            # Baseline prediction
            baseline_mask = baseline_predict(gzz_grid, {'z': 2.0, 'min_size': 50})
            
            # Try to load ground truth if available
            tunnel_mask = None
            if file.filename.startswith('gzz_grid_') or file.filename.startswith('density_grid_'):
                sample_id = file.filename.replace('gzz_grid_', '').replace('density_grid_', '').replace('.npy', '')
                mask_path = os.path.join('training_data', f'tunnel_mask_{sample_id}.npy')
                if os.path.exists(mask_path):
                    tunnel_mask = np.load(mask_path)
            
            if tunnel_mask is not None:
                ai_dice, ai_iou = compute_metrics(binary_mask, tunnel_mask)
                baseline_dice, baseline_iou = compute_metrics(baseline_mask, tunnel_mask)
                img_base64 = create_visualization(viz_grid, gzz_grid, prob_map, binary_mask, baseline_mask, tunnel_mask)
                return jsonify({
                    'success': True,
                    'image': img_base64,
                    'ai': {
                        'has_tunnel': bool(has_tunnel),
                        'confidence': float(confidence),
                        'tunnel_pixels': int(tunnel_pixels),
                        'dice': ai_dice,
                        'iou': ai_iou
                    },
                    'baseline': {
                        'dice': baseline_dice,
                        'iou': baseline_iou,
                        'tunnel_pixels': int(baseline_mask.sum())
                    }
                })
            else:
                # No ground truth available
                img_base64 = create_visualization(viz_grid, gzz_grid, prob_map, binary_mask, baseline_mask)
                return jsonify({
                    'success': True,
                    'image': img_base64,
                    'ai': {
                        'has_tunnel': bool(has_tunnel),
                        'confidence': float(confidence),
                        'tunnel_pixels': int(tunnel_pixels)
                    },
                    'baseline': {
                        'tunnel_pixels': int(baseline_mask.sum())
                    }
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
