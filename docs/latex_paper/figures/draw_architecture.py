"""
Generate DP-FedRAG Architecture Diagram (Academic Style)
Outputs: architecture_dp_fedrag.png (300 DPI)
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# ============================================================
# CONFIG
# ============================================================
FIG_W, FIG_H = 18, 14
DPI = 300
BG = '#FFFFFF'

# Colors
C_FL = '#2d6a4f'
C_FL_BG = '#d8f3dc'
C_KD = '#0f3460'
C_KD_BG = '#d6e4f0'
C_DP = '#c62828'
C_DP_BG = '#ffcdd2'
C_DP_BOX = '#e94560'
C_HE = '#6a1b9a'
C_HE_BG = '#e1bee7'
C_SERVER = '#e65100'
C_SERVER_BG = '#fff3e0'
C_GRAY = '#455a64'
C_LIGHT = '#f5f5f5'
C_ROUTE = '#e65100'
C_ROUTE_BG = '#fff8e1'
C_LLM = '#4a148c'
C_LLM_BG = '#ede7f6'
C_GEN = '#37474f'
C_GEN_BG = '#eceff1'

def rounded_box(ax, x, y, w, h, color, bg, lw=1.5, alpha=1.0, zorder=1):
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle="round,pad=0.08",
                         facecolor=bg, edgecolor=color,
                         linewidth=lw, alpha=alpha, zorder=zorder)
    ax.add_patch(box)
    return box

def text_center(ax, x, y, txt, size=7, color='black', weight='normal', zorder=10):
    ax.text(x, y, txt, ha='center', va='center', fontsize=size,
            color=color, fontweight=weight, zorder=zorder,
            fontfamily='sans-serif')

def arrow(ax, x1, y1, x2, y2, color=C_GRAY, lw=1.2, style='->', zorder=5):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw),
                zorder=zorder)

def arrow_label(ax, x1, y1, x2, y2, label, color=C_GRAY, lw=1.0, fontsize=5.5, label_offset=(0,0.08)):
    arrow(ax, x1, y1, x2, y2, color=color, lw=lw)
    mx, my = (x1+x2)/2 + label_offset[0], (y1+y2)/2 + label_offset[1]
    ax.text(mx, my, label, ha='center', va='center', fontsize=fontsize,
            color=color, fontstyle='italic', zorder=10, fontfamily='sans-serif')


# ============================================================
# FIGURE
# ============================================================
fig, ax = plt.subplots(1, 1, figsize=(FIG_W, FIG_H), dpi=DPI)
ax.set_xlim(0, 18)
ax.set_ylim(0, 14)
ax.set_aspect('equal')
ax.axis('off')
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

# ============================================================
# UPSTREAM SECTION
# ============================================================
# Section background
rounded_box(ax, 0.2, 5.9, 17.6, 7.7, '#b0bec5', '#fafafa', lw=1, alpha=0.3)
text_center(ax, 9, 13.35, 'Upstream: Privacy-Preserving Federated Embedding Pre-training',
            size=12, color='#1a1a2e', weight='bold')

# ---- CLIENT A ----
rounded_box(ax, 0.5, 6.1, 5.5, 7.0, C_FL, C_FL_BG, lw=1.8, alpha=0.25)
text_center(ax, 3.25, 12.85, '[Hospital A] Client 1', size=9, color=C_FL, weight='bold')
text_center(ax, 3.25, 12.5, 'Private Medical Data', size=6, color=C_GRAY)

# Global/Local Embedding
rounded_box(ax, 0.8, 11.6, 2.0, 0.65, C_KD, C_KD_BG, lw=1.2)
text_center(ax, 1.8, 11.92, 'Global RAG\nEmbedding', size=6.5, color=C_KD)

rounded_box(ax, 3.6, 11.6, 2.0, 0.65, C_KD, C_KD_BG, lw=1.2)
text_center(ax, 4.6, 11.92, 'Local RAG\nEmbedding', size=6.5, color=C_KD)

# KD arrows
ax.annotate('', xy=(3.5, 12.0), xytext=(2.9, 12.0),
            arrowprops=dict(arrowstyle='<->', color=C_KD, lw=1.2, linestyle='dashed'))
text_center(ax, 3.2, 11.4, 'logits (KD-GLE)', size=5.5, color=C_KD)

# Loss
rounded_box(ax, 1.0, 10.4, 4.5, 0.65, C_GRAY, '#fff3e0', lw=1)
text_center(ax, 3.25, 10.72, 'Contrastive Loss + KD Loss\n$\\mathcal{L} = \\mathcal{L}_{RAG} + \\alpha \\cdot \\mathcal{L}_{KD}$', size=6, color=C_GRAY)

arrow(ax, 3.25, 11.55, 3.25, 11.1, C_GRAY, 1)

# DP-SGD Block (HIGHLIGHTED)
rounded_box(ax, 0.7, 8.2, 5.1, 1.9, C_DP_BOX, C_DP_BG, lw=2.5, alpha=0.5)
text_center(ax, 3.25, 9.9, '[*] DP-SGD (Differential Privacy)', size=7, color=C_DP, weight='bold')

rounded_box(ax, 0.9, 8.9, 2.2, 0.7, C_DP, '#ffebee', lw=1.2)
text_center(ax, 2.0, 9.25, 'Gradient Clipping\n$\\|\\mathbf{g}\\|_2 \\leq C$', size=6, color=C_DP)

rounded_box(ax, 3.4, 8.9, 2.2, 0.7, C_DP, '#ffebee', lw=1.2)
text_center(ax, 4.5, 9.25, 'Noise Injection\n$\\tilde{\\mathbf{g}} = \\mathbf{g} + \\mathcal{N}(0, \\sigma^2 C^2 \\mathbf{I})$', size=5.5, color=C_DP)

arrow(ax, 3.15, 9.25, 3.35, 9.25, C_DP, 1.5)

text_center(ax, 3.25, 8.55, 'RDP Accounting: $(\\varepsilon=20,\\, \\delta=10^{-5})$-DP', size=5.5, color=C_DP)

arrow_label(ax, 3.25, 10.35, 3.25, 10.15, 'backward', C_DP, 1.2, 5)

# CKKS
rounded_box(ax, 1.0, 6.7, 4.5, 0.65, C_HE, C_HE_BG, lw=1.2)
text_center(ax, 3.25, 7.02, 'CKKS Homomorphic Encryption\nEnc(Δ) with public key', size=5.5, color=C_HE)

arrow(ax, 3.25, 8.15, 3.25, 7.4, C_HE, 1.2)
text_center(ax, 3.9, 7.85, 'Δ = W_local − W_global', size=5, color=C_HE)

# ---- SERVER ----
rounded_box(ax, 6.5, 8.5, 5, 3.2, C_SERVER, C_SERVER_BG, lw=1.5, alpha=0.4)
text_center(ax, 9, 11.45, 'FL Server (Aggregator)', size=9, color=C_SERVER, weight='bold')

rounded_box(ax, 7.0, 10.2, 4.0, 0.7, C_SERVER, '#ffe0b2', lw=1.2)
text_center(ax, 9, 10.55, 'FedAvg Aggregation\n$\\mathbf{W} \\leftarrow \\mathbf{W} + \\frac{1}{N}\\sum_i \\text{Dec}(\\Delta_i)$', size=6, color=C_SERVER)

rounded_box(ax, 7.3, 9.0, 3.4, 0.65, C_FL, '#c8e6c9', lw=1.2)
text_center(ax, 9, 9.32, 'Trained Global\nEmbedding Model', size=6.5, color=C_FL, weight='bold')

arrow(ax, 9, 10.15, 9, 9.7, C_SERVER, 1)

# Dots
text_center(ax, 9, 12.0, '· · ·', size=14, color='#bdbdbd')

# ---- CLIENT N ----
rounded_box(ax, 12.0, 6.1, 5.5, 7.0, C_FL, C_FL_BG, lw=1.8, alpha=0.25)
text_center(ax, 14.75, 12.85, '[Hospital N] Client N', size=9, color=C_FL, weight='bold')
text_center(ax, 14.75, 12.5, 'Private Medical Data', size=6, color=C_GRAY)

rounded_box(ax, 12.3, 11.6, 2.0, 0.65, C_KD, C_KD_BG, lw=1.2)
text_center(ax, 13.3, 11.92, 'Global RAG\nEmbedding', size=6.5, color=C_KD)
rounded_box(ax, 15.1, 11.6, 2.0, 0.65, C_KD, C_KD_BG, lw=1.2)
text_center(ax, 16.1, 11.92, 'Local RAG\nEmbedding', size=6.5, color=C_KD)
ax.annotate('', xy=(15.0, 12.0), xytext=(14.4, 12.0),
            arrowprops=dict(arrowstyle='<->', color=C_KD, lw=1.2, linestyle='dashed'))
text_center(ax, 14.7, 11.4, 'logits (KD-GLE)', size=5.5, color=C_KD)

rounded_box(ax, 12.5, 10.4, 4.5, 0.65, C_GRAY, '#fff3e0', lw=1)
text_center(ax, 14.75, 10.72, 'Contrastive Loss + KD Loss', size=6, color=C_GRAY)
arrow(ax, 14.75, 11.55, 14.75, 11.1, C_GRAY, 1)

rounded_box(ax, 12.2, 8.2, 5.1, 1.9, C_DP_BOX, C_DP_BG, lw=2.5, alpha=0.5)
text_center(ax, 14.75, 9.9, '[*] DP-SGD (Differential Privacy)', size=7, color=C_DP, weight='bold')
rounded_box(ax, 12.4, 8.9, 2.2, 0.7, C_DP, '#ffebee', lw=1.2)
text_center(ax, 13.5, 9.25, 'Gradient Clipping\n$\\|\\mathbf{g}\\|_2 \\leq C$', size=6, color=C_DP)
rounded_box(ax, 14.9, 8.9, 2.2, 0.7, C_DP, '#ffebee', lw=1.2)
text_center(ax, 16.0, 9.25, 'Noise Injection\n$\\tilde{\\mathbf{g}} + \\mathcal{N}(0, \\sigma^2 C^2 \\mathbf{I})$', size=5.5, color=C_DP)
arrow(ax, 14.65, 9.25, 14.85, 9.25, C_DP, 1.5)
text_center(ax, 14.75, 8.55, 'RDP: $(\\varepsilon=20, \\delta=10^{-5})$-DP', size=5.5, color=C_DP)
arrow_label(ax, 14.75, 10.35, 14.75, 10.15, 'backward', C_DP, 1.2, 5)

rounded_box(ax, 12.5, 6.7, 4.5, 0.65, C_HE, C_HE_BG, lw=1.2)
text_center(ax, 14.75, 7.02, 'CKKS Homomorphic Encryption\nEnc(Δ) with public key', size=5.5, color=C_HE)
arrow(ax, 14.75, 8.15, 14.75, 7.4, C_HE, 1.2)

# Arrows: Clients <-> Server
arrow_label(ax, 5.5, 7.0, 7.2, 9.5, 'Enc(Δ)', C_HE, 1.5, 5.5, (0.1, 0.1))
arrow_label(ax, 7.2, 11.0, 5.6, 12.0, 'W_global', C_FL, 1.2, 5.5, (-0.1, 0.1))
arrow_label(ax, 12.0, 7.0, 10.8, 9.5, 'Enc(Δ)', C_HE, 1.5, 5.5, (-0.1, 0.1))
arrow_label(ax, 10.8, 11.0, 12.0, 12.0, 'W_global', C_FL, 1.2, 5.5, (0.1, 0.1))

# ============================================================
# CONNECTOR: Upstream -> Downstream
# ============================================================
arrow(ax, 9, 8.95, 9, 5.55, C_FL, 2, '->')
text_center(ax, 9.8, 7.0, 'model\ncheckpoint', size=5.5, color=C_FL)

# ============================================================
# DOWNSTREAM SECTION
# ============================================================
rounded_box(ax, 0.2, 0.8, 17.6, 4.7, '#b0bec5', '#fafafa', lw=1, alpha=0.2)
text_center(ax, 9, 5.25, 'Downstream: Federated Retrieval-Augmented Generation Inference',
            size=12, color='#1a1a2e', weight='bold')

# ---- Left Client ----
rounded_box(ax, 0.6, 1.8, 4.8, 3.0, C_FL, C_FL_BG, lw=1.5, alpha=0.25)
text_center(ax, 3.0, 4.55, '[Hospital A] FL Client', size=8, color=C_FL, weight='bold')

rounded_box(ax, 1.0, 3.9, 4.0, 0.45, C_FL, '#c8e6c9', lw=1)
text_center(ax, 3.0, 4.12, 'DP-Trained Embedding Model', size=6, color=C_FL)

rounded_box(ax, 1.0, 3.2, 4.0, 0.5, C_KD, C_KD_BG, lw=1)
text_center(ax, 3.0, 3.45, 'FAISS Index + LlamaIndex\nMulti-Retriever (Dense + Sparse)', size=5.5, color=C_KD)

rounded_box(ax, 1.0, 2.55, 4.0, 0.45, C_DP, '#ffebee', lw=1)
text_center(ax, 3.0, 2.77, 'Privacy-aware Summarization', size=6, color=C_DP)

rounded_box(ax, 1.0, 1.95, 4.0, 0.45, C_GEN, C_GEN_BG, lw=1)
text_center(ax, 3.0, 2.17, 'Generation Model', size=6, color=C_GEN)

arrow(ax, 3.0, 3.85, 3.0, 3.75, C_GRAY, 0.8)
arrow(ax, 3.0, 3.15, 3.0, 3.05, C_GRAY, 0.8)
arrow(ax, 3.0, 2.5, 3.0, 2.45, C_GRAY, 0.8)

# ---- Right Client ----
rounded_box(ax, 12.6, 1.8, 4.8, 3.0, C_FL, C_FL_BG, lw=1.5, alpha=0.25)
text_center(ax, 15.0, 4.55, '[Hospital N] FL Client', size=8, color=C_FL, weight='bold')

rounded_box(ax, 13.0, 3.9, 4.0, 0.45, C_FL, '#c8e6c9', lw=1)
text_center(ax, 15.0, 4.12, 'DP-Trained Embedding Model', size=6, color=C_FL)

rounded_box(ax, 13.0, 3.2, 4.0, 0.5, C_KD, C_KD_BG, lw=1)
text_center(ax, 15.0, 3.45, 'FAISS Index + LlamaIndex\nMulti-Retriever (Dense + Sparse)', size=5.5, color=C_KD)

rounded_box(ax, 13.0, 2.55, 4.0, 0.45, C_DP, '#ffebee', lw=1)
text_center(ax, 15.0, 2.77, 'Privacy-aware Summarization', size=6, color=C_DP)

rounded_box(ax, 13.0, 1.95, 4.0, 0.45, C_GEN, C_GEN_BG, lw=1)
text_center(ax, 15.0, 2.17, 'Generation Model', size=6, color=C_GEN)

arrow(ax, 15.0, 3.85, 15.0, 3.75, C_GRAY, 0.8)
arrow(ax, 15.0, 3.15, 15.0, 3.05, C_GRAY, 0.8)
arrow(ax, 15.0, 2.5, 15.0, 2.45, C_GRAY, 0.8)

# ---- RagRoute (Center) ----
rounded_box(ax, 6.8, 2.8, 4.4, 1.5, C_ROUTE, C_ROUTE_BG, lw=2.5)
text_center(ax, 9.0, 3.85, 'RagRoute', size=9, color=C_ROUTE, weight='bold')
text_center(ax, 9.0, 3.45, 'Intelligent Query Routing', size=7, color=C_ROUTE)
text_center(ax, 9.0, 3.1, '[OK] relevant       [X] not relevant', size=6, color=C_GRAY)

# Arrows: Clients -> RagRoute
arrow_label(ax, 5.05, 2.77, 6.75, 3.3, 'retrieved docs', C_ROUTE, 1.5, 5.5, (0, 0.12))
arrow_label(ax, 12.95, 2.77, 11.25, 3.3, 'retrieved docs', C_ROUTE, 1.5, 5.5, (0, 0.12))

# ---- Global LLM Fusion ----
rounded_box(ax, 5.5, 1.0, 7.0, 0.9, C_LLM, C_LLM_BG, lw=1.5)
text_center(ax, 9.0, 1.6, 'Global LLM Fusion  +  Prompt Privacy', size=7, color=C_LLM, weight='bold')
text_center(ax, 9.0, 1.25, 'PII Detection (Presidio)  |  Embedding Model', size=5.5, color=C_LLM)

# Arrows to LLM
arrow(ax, 9.0, 2.75, 9.0, 1.95, C_LLM, 1.5)
arrow(ax, 3.0, 1.9, 5.5, 1.5, C_GRAY, 0.8)
arrow(ax, 15.0, 1.9, 12.5, 1.5, C_GRAY, 0.8)

# ---- User ----
user_circle = plt.Circle((16.5, 1.45), 0.4, facecolor='#e0e0e0', edgecolor=C_GEN, lw=1.5, zorder=5)
ax.add_patch(user_circle)
text_center(ax, 16.5, 1.45, 'User\nUser', size=6, color=C_GEN)

# Arrows LLM <-> User
arrow_label(ax, 12.55, 1.5, 16.05, 1.5, 'Response', C_GEN, 1.5, 6, (0, 0.12))
ax.annotate('', xy=(12.55, 1.3), xytext=(16.05, 1.3),
            arrowprops=dict(arrowstyle='->', color='#9e9e9e', lw=0.8, linestyle='dashed'))
text_center(ax, 14.3, 1.15, 'Prompt', size=5.5, color='#9e9e9e')

# ============================================================
# LEGEND
# ============================================================
leg_y = 0.35
text_center(ax, 1.5, leg_y, 'Defense-in-Depth:', size=7, color='#1a1a2e', weight='bold')

rounded_box(ax, 3.0, leg_y-0.2, 2.8, 0.4, C_FL, C_FL_BG, lw=1.2, alpha=0.6)
text_center(ax, 4.4, leg_y, 'Layer 1: FL — Data stays local', size=5.5, color=C_FL)

rounded_box(ax, 6.1, leg_y-0.2, 2.8, 0.4, C_KD, C_KD_BG, lw=1.2, alpha=0.6)
text_center(ax, 7.5, leg_y, 'Layer 2: KD — Only logits shared', size=5.5, color=C_KD)

rounded_box(ax, 9.2, leg_y-0.2, 3.3, 0.4, C_DP_BOX, C_DP_BG, lw=2, alpha=0.6)
text_center(ax, 10.85, leg_y, '[*] Layer 3: DP-SGD — (ε,δ)-DP', size=5.5, color=C_DP, weight='bold')

rounded_box(ax, 12.8, leg_y-0.2, 3.2, 0.4, C_HE, C_HE_BG, lw=1.2, alpha=0.6)
text_center(ax, 14.4, leg_y, 'Layer 4: HE — Encrypted transit', size=5.5, color=C_HE)

# ============================================================
# SAVE
# ============================================================
output = 'architecture_dp_fedrag.png'
plt.tight_layout(pad=0.3)
plt.savefig(output, dpi=DPI, bbox_inches='tight', facecolor=BG, edgecolor='none')
print(f"Saved: {output}")
plt.close()
