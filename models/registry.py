"""Model registry with decorator pattern — auto-discovers models.

Each model registers itself with:

    @register_model('ModelName', expects_4d=True, hidden_dim=64)
    def factory(num_classes, bands, patch_size=11, hidden_dim=64, **kwargs):
        return ModelName(...)

Only two kinds of keyword go in the decorator:
  * ``expects_4d`` — stripped out by ``create_model`` and used to decide whether
    the 5D loader output (B, 1, bands, H, W) is squeezed to 4D via
    ``InputShapeWrapper``.
  * model hyper-parameter defaults — these are forwarded straight to the
    factory, so anything listed here must be accepted by it.

Bibliographic metadata (paper, code, year, venue) does NOT belong in the
decorator — passing it there would forward it into the model constructor and
break instantiation. It lives in ``MODEL_CATALOG`` below, keyed by model name.
"""

import os
import importlib

_model_registry = {}
_model_configs = {}
_models_loaded = False


# ──────────────────────────────────────────────
# Model metadata catalog
# Keyed by registered model name.
# ──────────────────────────────────────────────
MODEL_CATALOG = {
    "SpectralFormer": {
        'full_name': "SpectralFormer",
        'paper_title': "SpectralFormer: Rethinking Hyperspectral Image Classification With Transformers",
        'paper': "https://doi.org/10.1109/TGRS.2021.3130716",
        'code': "https://github.com/danfenghong/IEEE_TGRS_SpectralFormer",
        'year': 2021,
        'venue': "IEEE TGRS",
    },
    "MFT": {
        'full_name': "Multimodal Fusion Transformer",
        'paper_title': "Multimodal Fusion Transformer for Remote Sensing Image Classification",
        'paper': "https://doi.org/10.1109/TGRS.2023.3286826",
        'code': "https://github.com/AnkurDeria/MFT",
        'year': 2023,
        'venue': "IEEE TGRS",
    },
    "GAHT": {
        'full_name': "Group-Aware Hierarchical Transformer",
        'paper_title': "Hyperspectral Image Classification Using Group-Aware Hierarchical Transformer",
        'paper': "https://doi.org/10.1109/TGRS.2022.3207933",
        'code': "https://github.com/MeiShaohui/Group-Aware-Hierarchical-Transformer",
        'year': 2023,
        'venue': "IEEE TGRS",
    },
    "MASSFormer": {
        'full_name': "Memory-Augmented Spectral-Spatial Transformer",
        'paper_title': "MASSFormer: Memory-Augmented Spectral-Spatial Transformer for Hyperspectral Image Classification",
        'paper': "https://doi.org/10.1109/TGRS.2024.3392264",
        'code': "https://github.com/hz63/MASSFormer",
        'year': 2024,
        'venue': "IEEE TGRS",
    },
    "MorphFormer": {
        'full_name': "MorphFormer",
        'paper_title': "Spectral-Spatial Morphological Attention Transformer for Hyperspectral Image Classification",
        'paper': "https://doi.org/10.1109/TGRS.2023.3242346",
        'code': "https://github.com/mhaut/morphFormer",
        'year': 2023,
        'venue': "IEEE TGRS",
    },
    "SSFTTNet": {
        'full_name': "Spatial-Spectral Feature Tokenization Transformer",
        'paper_title': "Spectral-Spatial Feature Tokenization Transformer for Hyperspectral Image Classification",
        'paper': "https://doi.org/10.1109/TGRS.2022.3144158",
        'code': "https://github.com/zgr6010/HSI_SSFTT",
        'year': 2022,
        'venue': "IEEE TGRS",
    },
    "3DConvSST": {
        'full_name': "3D Convolution Spectral-Spatial Transformer",
        'paper_title': "3D-Convolution Guided Spectral-Spatial Transformer for Hyperspectral Image Classification",
        'paper': "https://doi.org/10.1109/CAI59869.2024.00011",
        'code': "https://github.com/ShyamVarahagiri/3D-ConvSST",
        'year': 2024,
        'venue': "IEEE CAI",
    },
    "DBCTNet": {
        'full_name': "Double Branch Convolution-Transformer Network",
        'paper_title': "DBCTNet: Double Branch Convolution-Transformer Network for Hyperspectral Image Classification",
        'paper': "https://doi.org/10.1109/TGRS.2024.3368141",
        'code': "https://github.com/xurui-joei/DBCTNet",
        'year': 2024,
        'venue': "IEEE TGRS",
    },
    "DSFormer": {
        'full_name': "Dual Selective Fusion Transformer Network",
        'paper_title': "Dual selective fusion transformer network for hyperspectral image classification",
        'paper': "https://doi.org/10.1016/j.neunet.2025.107311",
        'code': "https://github.com/YichuXu/DSFormer",
        'year': 2025,
        'venue': "Neural Networks",
    },
    "WaveMamba": {
        'full_name': "Wavelet-enhanced State Space Model",
        'paper_title': "WaveMamba: Spatial-Spectral Wavelet Mamba for Hyperspectral Image Classification",
        'paper': "https://doi.org/10.1109/LGRS.2024.3506034",
        'code': "https://github.com/mahmad000/WaveMamba",
        'year': 2024,
        'venue': "IEEE GRSL",
    },
    "MiM": {
        'full_name': "Mamba-in-Mamba: Centralized Mamba-Cross-Scan",
        'paper_title': "Mamba-in-Mamba: Centralized Mamba-Cross-Scan in Tokenized Mamba Model for Hyperspectral Image Classification",
        'paper': "https://doi.org/10.1016/j.neucom.2024.128751",
        'code': "https://github.com/zhouweilian1904/Mamba-in-Mamba",
        'year': 2025,
        'venue': "Neurocomputing",
    },
    "MambaHSI": {
        'full_name': "MambaHSI",
        'paper_title': "MambaHSI: Spatial-Spectral Mamba for Hyperspectral Image Classification",
        'paper': "https://doi.org/10.1109/TGRS.2024.3430985",
        'code': "https://github.com/li-yapeng/MambaHSI",
        'year': 2024,
        'venue': "IEEE TGRS",
    },
    "MambaHSI_Plus": {
        'full_name': "MambaHSI+",
        'paper_title': "MambaHSI+: Multidirectional State Propagation for Efficient Hyperspectral Image Classification",
        'paper': "https://doi.org/10.1109/TGRS.2025.3576656",
        'code': "https://github.com/RockAilab/MambaHSI_Plus",
        'year': 2024,
        'venue': "IEEE TGRS",
    },
    "SSMamba": {
        'full_name': "Spectral-Spatial Mamba",
        'paper_title': "Spectral-Spatial Mamba for Hyperspectral Image Classification",
        'paper': "https://doi.org/10.3390/rs16132449",
        'code': "https://github.com/mengduanjinghua/Spectral-spatial-Mamba-for-HSIC",
        'year': 2024,
        'venue': "MDPI Remote Sensing",
    },
    "S2Mamba": {
        'full_name': "S2Mamba: Spectral-Spatial Mamba",
        'paper_title': "S2Mamba: A Spatial-Spectral State Space Model for Hyperspectral Image Classification",
        'paper': "https://doi.org/10.1109/TGRS.2025.3530993",
        'code': "https://github.com/PURE-melo/S2Mamba",
        'year': 2024,
        'venue': "IEEE TGRS",
    },
    "PHDMamba": {
        'full_name': "Progressive Hybrid Mamba",
        'paper_title': "PHDMamba: Progressive Hybrid Mamba for Hyperspectral Image Classification",
        'paper': "https://doi.org/10.1109/LGRS.2025.3626712",
        'code': "https://github.com/YichuXu/PHDMamba",
        'year': 2025,
        'venue': "IEEE GRSL",
    },
    "HybridSN": {
        'full_name': "HybridSN: 3D-2D CNN for HSI Classification",
        'paper_title': "HybridSN: Exploring 3-D-2-D CNN Feature Hierarchy for Hyperspectral Image Classification",
        'paper': "https://doi.org/10.1109/LGRS.2019.2918719",
        'code': "https://github.com/gokriznastic/HybridSN",
        'year': 2019,
        'venue': "IEEE GRSL",
    },
    "S3ANet": {
        'full_name': "Spectral-Spatial Self-Attention Network",
        'paper_title': "S3ANet: Spatial-Spectral Self-Attention Learning Network for Defending Against Adversarial Attacks in Hyperspectral Image Classification",
        'paper': "https://doi.org/10.1109/TGRS.2024.3381824",
        'code': "https://github.com/YichuXu/S3ANet",
        'year': 2024,
        'venue': "IEEE TGRS",
    },
    "SACNet": {
        'full_name': "Self-Attention Context Network",
        'paper_title': "Self-Attention Context Network: Addressing the Threat of Adversarial Attacks for Hyperspectral Image Classification",
        'paper': "https://doi.org/10.1109/TIP.2021.3118977",
        'code': "https://github.com/YonghaoXu/SACNet",
        'year': 2021,
        'venue': "IEEE TIP",
    },
    "FETNet": {
        'full_name': "Fuzzy Enhanced Transformer Network",
        'paper_title': "Fuzzy enhanced transformer network for classification of hyperspectral image combined with light detection and ranging data",
        'paper': "https://doi.org/10.1016/j.engappai.2025.113343",
        'code': "https://github.com/lplxtp/FETNet",
        'year': 2026,
        'venue': "Engineering Applications of Artificial Intelligence",
    },
    "GTCFN": {
        'full_name': "Graph Transformer Convolution Fusion Network",
        'paper_title': "GTCFN: A Graph-Based Transformer and Convolution Fusion Network for Hyperspectral Image Classification",
        'paper': "https://doi.org/10.1109/TGRS.2025.3618962",
        'code': "https://github.com/Majunyi310321/GTCFN",
        'year': 2025,
        'venue': "IEEE TGRS",
    },
    "GraphGST": {
        'full_name': "Graph-Guided Spectral Transformer",
        'paper_title': "GraphGST: Graph Generative Structure-Aware Transformer for Hyperspectral Image Classification",
        'paper': "https://doi.org/10.1109/TGRS.2023.3349076",
        'code': "https://github.com/yuanchaosu/TGRS-graphGST",
        'year': 2024,
        'venue': "IEEE TGRS",
    },
    "MCTGCL": {
        'full_name': "Mixed CNN-Transformer with Graph Contrastive Learning",
        'paper_title': "MCTGCL: Mixed CNN-Transformer for Mars Hyperspectral Image Classification With Graph Contrastive Learning",
        'paper': "https://doi.org/10.1109/TGRS.2025.3529996",
        'code': "https://github.com/B-Xi/TGRS_2025_MCTGCL",
        'year': 2025,
        'venue': "IEEE TGRS",
    },
    "GraphMamba": {
        'full_name': "Graph-enhanced Mamba for HSI Classification",
        'paper_title': "GraphMamba: An Efficient Graph Structure Learning Vision Mamba for Hyperspectral Image Classification",
        'paper': "https://doi.org/10.1109/TGRS.2024.3493101",
        'code': "https://github.com/ahappyyang/GraphMamba",
        'year': 2024,
        'venue': "IEEE TGRS",
    },
    "IGroupSS-Mamba": {
        'full_name': "Interval Group Spatial-Spectral Mamba",
        'paper_title': "IGroupSS-Mamba: Interval Group Spatial-Spectral Mamba for Hyperspectral Image Classification",
        'paper': "https://doi.org/10.1109/TGRS.2024.3502055",
        'code': "https://github.com/IIP-Team/IGroupSS-Mamba",
        'year': 2024,
        'venue': "IEEE TGRS",
    },
    "GSCViT": {
        'full_name': "Groupwise Separable Convolutional Vision Transformer",
        'paper_title': "Hyperspectral Image Classification Using Groupwise Separable Convolutional Vision Transformer Network",
        'paper': "https://doi.org/10.1109/TGRS.2024.3377610",
        'code': "https://github.com/flyzzie/TGRS-GSC-VIT",
        'year': 2024,
        'venue': "IEEE TGRS",
    },
    "FAHM": {
        'full_name': "Frequency-Aware Hierarchical Mamba",
        'paper_title': "FAHM: Frequency-Aware Hierarchical Mamba for Hyperspectral Image Classification",
        'paper': "https://doi.org/10.1109/JSTARS.2025.3539791",
        'code': "https://github.com/zhangxc0105/FAHM",
        'year': 2025,
        'venue': "IEEE JSTARS",
    },
    "S2Gformer": {
        'full_name': "Spectral-Spatial Graph Transformer",
        'paper_title': "S2GFormer: A Transformer and Graph Convolution Combining Framework for Hyperspectral Image Classification",
        'paper': "https://doi.org/10.1109/TGRS.2024.3488202",
        'code': "https://github.com/DY-HYX",
        'year': 2024,
        'venue': "IEEE TGRS",
    },
    "FuzzySpectralMamba": {
        'full_name': "Fuzzy Spectral Mamba",
        'paper_title': "Learnable Fuzzy Spectral Mamba for Uncertainty Aware Hyperspectral Image Classification",
        'paper': "https://doi.org/10.1109/LGRS.2026.3687386",
        'code': "https://github.com/Tanishq251/FSM-HSIC",
        'year': 2026,
        'venue': "IEEE GRSL",
    },
    "SSRN": {
        'full_name': "Spectral-Spatial Residual Network",
        'paper_title': "Spectral-Spatial Residual Network for Hyperspectral Image Classification: A 3-D Deep Learning Framework",
        'paper': "https://doi.org/10.1109/TGRS.2017.2755542",
        'code': "https://github.com/zilongzhong/SSRN",
        'year': 2017,
        'venue': "IEEE TGRS",
    },
    "DBDA": {
        'full_name': "Double-branch Dual-attention Mechanism Network",
        'paper_title': "Classification of Hyperspectral Image Based on Double-Branch Dual-Attention Mechanism Network",
        'paper': "https://doi.org/10.3390/rs12030582",
        'code': "https://github.com/lironui/Double-Branch-Dual-Attention-Mechanism-Network",
        'year': 2020,
        'venue': "MDPI Remote Sensing",
    },
    "ENL_FCN": {
        'full_name': "Efficient Non-local Fully Convolutional Network",
        'paper_title': "Efficient Deep Learning of Nonlocal Features for Hyperspectral Image Classification",
        'paper': "https://doi.org/10.1109/TGRS.2020.3014286",
        'code': "https://github.com/ShaneShen/ENL-FCN",
        'year': 2020,
        'venue': "IEEE TGRS",
    },
    "SSTN": {
        'full_name': "Spatial-Spectral Transformer Network",
        'paper_title': "Spectral-Spatial Transformer Network for Hyperspectral Image Classification: A Factorized Architecture Search Framework",
        'paper': "https://doi.org/10.1109/TGRS.2021.3115699",
        'code': "https://github.com/zilongzhong/SSTN",
        'year': 2021,
        'venue': "IEEE TGRS",
    },
    "HSIC_FM": {
        'full_name': "Hyperspectral Image Classification Full Model",
        'paper_title': "Overcoming the Barrier of Incompleteness: A Hyperspectral Image Classification Full Model",
        'paper': "https://doi.org/10.1109/TNNLS.2023.3279377",
        'code': "https://github.com/jqyang22/HSIC-FM",
        'year': 2024,
        'venue': "IEEE TNNLS",
    },
    "HSIMAE": {
        'full_name': "Spatial-Spectral Vision Transformer (HSIMAE)",
        'paper_title': "HSIMAE: A Unified Masked Autoencoder With Large-Scale Pretraining for Hyperspectral Image Classification",
        'paper': "https://doi.org/10.1109/JSTARS.2024.3432743",
        'code': "https://github.com/Ryan21wy/HSIMAE",
        'year': 2024,
        'venue': "arXiv",
    },
    "LFSMIM": {
        'full_name': "Low-frequency Spatial-spectral Masked Image Modeling",
        'paper_title': "LFSMIM: A Low-Frequency Spectral Masked Image Modeling Method for Hyperspectral Image Classification",
        'paper': "https://doi.org/10.1109/LGRS.2024.3360184",
        'code': "https://github.com/yuweikong/LFSMIM",
        'year': 2024,
        'venue': "IEEE GRSL",
    },
    "HyPyraMamba": {
        'full_name': "Pyramid Spectral Attention and Mamba-Based Architecture",
        'paper_title': "HyPyraMamba: A Pyramid Spectral Attention and Mamba-Based Architecture for Robust Hyperspectral Image Classification",
        'paper': "https://doi.org/10.1109/TGRS.2025.3650350",
        'code': "https://github.com/dekai-li/HyPyraMamba",
        'year': 2026,
        'venue': "IEEE TGRS",
    },
    "MLFMamba": {
        'full_name': "Multi-Level Feature Mamba",
        'paper_title': "Mamba unleashed: a multi-level feature modeling framework for enhanced hyperspectral image classification",
        'paper': "https://doi.org/10.1007/s00371-026-04496-w",
        'code': "https://github.com/foopy113/MLF",
        'year': 2026,
        'venue': "The Visual Computer (Springer)",
    },
    "MS2GCAN": {
        'full_name': "Multi-Scale Spiking Graph Convolution Aggregation Network",
        'paper_title': "Multiscale Spiking Graph Convolution Aggregation Network for Hyperspectral Image Classification",
        'paper': "https://doi.org/10.1109/TGRS.2026.3678343",
        'code': "https://github.com/fwx0314/MS2GCAN",
        'year': 2026,
        'venue': "IEEE TGRS",
    },
    "R2Mamba": {
        'full_name': "Route-Reliability Mamba",
        'paper_title': "R2Mamba: Route-Reliability Mamba for Hyperspectral Image Classification",
        'paper': "https://doi.org/10.1109/JSTARS.2026.3728152",
        'code': "https://github.com/xunshang111/R2Mamba-HSI",
        'year': 2026,
        'venue': "IEEE JSTARS",
    },
    "MambaMoE": {
        'full_name': "Mixture-of-Experts Mamba for HSI Classification",
        'paper_title': "MambaMoE: Mixture-of-spectral-spatial-experts state space model for hyperspectral image classification",
        'paper': "https://doi.org/10.1016/j.inffus.2025.103811",
        'code': "https://github.com/YichuXu/MambaMoE",
        'year': 2025,
        'venue': "Information Fusion (Elsevier)",
    },
    "MMFormer": {
        'full_name': "Macro-Micro Transformer for HSI Classification",
        'paper_title': "MMFormer: Macro-Micro Transformer for Small-Sample Classification of Mars Hyperspectral Image",
        'paper': "https://doi.org/10.1109/TGRS.2026.3696892",
        'code': "https://github.com/tingruifeng/TGRS_2026_MMFormer",
        'year': 2026,
        'venue': "IEEE TGRS",
    },
    "pResNet": {
        'full_name': "Deep Pyramidal Residual Network for HSI",
        'paper_title': "Deep Pyramidal Residual Networks for Spectral-Spatial Hyperspectral Image Classification",
        'paper': "https://doi.org/10.1109/TGRS.2018.2860125",
        'code': "https://github.com/mhaut/pResNet-HSI",
        'year': 2019,
        'venue': "IEEE TGRS",
    },
    "MVAHN": {
        'full_name': "Multiple Vision Architectures-based Hybrid Network",
        'paper_title': "Multiple vision architectures-based hybrid network for hyperspectral image classification",
        'paper': "https://doi.org/10.1016/j.eswa.2023.121032",
        'code': "https://github.com/ZJier/MVAHN",
        'year': 2023,
        'venue': "IEEE TGRS",
    },
    "DKDMN": {
        'full_name': "Data and Knowledge-Driven Multiview Fusion Network",
        'paper_title': "Data and knowledge-driven deep multiview fusion network based on diffusion model for hyperspectral image classification",
        'paper': "https://doi.org/10.1016/j.eswa.2024.123796",
        'code': "https://github.com/ZJier/DKDMN",
        'year': 2024,
        'venue': "Expert Systems with Applications (Elsevier)",
    },
    "CTMixer": {
        'full_name': "Convolution Transformer Mixer",
        'paper_title': "Convolution Transformer Mixer for Hyperspectral Image Classification",
        'paper': "https://ieeexplore.ieee.org/document/9924229",
        'code': "https://github.com/ZJier/CTMixer",
        'year': 2022,
        'venue': "IEEE GRSL",
    },
    "HyperKAN": {
        'full_name': "HyperKAN (SSFTT-KAN variant)",
        'paper_title': "HyperKAN: Kolmogorov-Arnold Networks Make Hyperspectral Image Classifiers Smarter",
        'paper': "https://doi.org/10.3390/s24237683",
        'code': "https://github.com/f-neumann77/HyperKAN",
        'year': 2024,
        'venue': "Sensors (MDPI)",
    },
    "HSIConvKAN": {
        'full_name': "Hybrid 3D-2D Convolutional KAN",
        'paper_title': "How to Learn More? Exploring Kolmogorov-Arnold Networks for Hyperspectral Image Classification",
        'paper': "https://doi.org/10.3390/rs16214015",
        'code': "https://github.com/aj1365/HSIConvKAN",
        'year': 2024,
        'venue': "Remote Sensing (MDPI)",
    },
    "HyperMamba": {
        'full_name': "Spectral-Spatial Adaptive Mamba",
        'paper_title': "HyperMamba: A Spectral-Spatial Adaptive Mamba for Hyperspectral Image Classification",
        'paper': "https://ieeexplore.ieee.org/document/10614183",
        'code': "https://github.com/chiangliu/HyperMamba",
        'year': 2024,
        'venue': "IEEE TGRS",
    },
    "MambaLG": {
        'full_name': "Local-Global Mamba",
        'paper_title': "MambaLG: A Local-Global Mamba for Hyperspectral Image Classification",
        'paper': "https://doi.org/10.1109/TGRS.2024.3521411",
        'code': "https://github.com/danfenghong/IEEE_TGRS_MambaLG",
        'year': 2024,
        'venue': "IEEE TGRS",
    },
    "ConvVitMamba": {
        'full_name': "Convolutional Vision Transformer with Mamba Block",
        'paper_title': "ConvViTMamba: A Convolutional Vision Transformer with Mamba block for Hyperspectral Image Classification",
        'paper': "https://doi.org/10.1016/j.knosys.2025.113282",
        'code': "https://github.com/mqalkhatib/ConvVitMamba",
        'year': 2025,
        'venue': "Knowledge-Based Systems (Elsevier)",
    },
    "HSIC_SClusterFormer": {
        'full_name': "Spectral-Spatial Cluster Attention Hierarchical Transformer",
        'paper_title': "Deformable Convolution-Enhanced Hierarchical Transformer With Spectral-Spatial Cluster Attention for Hyperspectral Image Classification",
        'paper': "https://doi.org/10.1109/TIP.2024.3522809",
        'code': "https://github.com/Fang666666/HSIC_SClusterFormer",
        'year': 2025,
        'venue': "IEEE TIP",
    },
    "MHSSMamba": {
        'full_name': "Multi-head Spatial-Spectral Mamba",
        'paper_title': "MHSSMamba: Multi-head Spatial-Spectral Mamba for Hyperspectral Image Classification",
        'paper': "https://doi.org/10.1080/2150704X.2025.2461330",
        'code': "https://github.com/mahmad000/MHSSMamba",
        'year': 2025,
        'venue': "Remote Sensing Letters",
    },
    "MorpMamba": {
        'full_name': "Spatial-Spectral Morphological Mamba",
        'paper_title': "MorpMamba: Spatial-Spectral Morphological Mamba for Hyperspectral Image Classification",
        'paper': "https://doi.org/10.1016/j.neucom.2025.129990",
        'code': "https://github.com/mahmad000/MorpMamba",
        'year': 2025,
        'venue': "Neurocomputing (Elsevier)",
    },
    "EMamba": {
        'full_name': "Efficient Cross-Modal Mamba (single-modality HSI variant)",
        'paper_title': "E-Mamba: Efficient Cross-Modal Mamba for HSI-LiDAR Fusion Classification",
        'paper': "https://doi.org/10.1016/j.inffus.2025.103328",
        'code': "https://github.com/zhangyiyan001/E-Mamba",
        'year': 2026,
        'venue': "Information Fusion (Elsevier)",
    },
}


# ──────────────────────────────────────────────
# Registration decorator
# ──────────────────────────────────────────────

def register_model(name, expects_4d=False, **default_cfg):
    """Decorator to register a model factory function.

    Usage:
        @register_model('MyModel', expects_4d=True, dim=64)
        def my_model(**kwargs):
            return MyModelClass(**kwargs)
    """
    def decorator(cls):
        _model_registry[name] = cls
        config = default_cfg.copy()
        config['expects_4d'] = expects_4d
        _model_configs[name] = config
        return cls
    return decorator


# ──────────────────────────────────────────────
# Auto-discovery
# ──────────────────────────────────────────────

def _load_models():
    """Auto-discover and load all models from this directory and subdirectories."""
    global _models_loaded
    if _models_loaded:
        return

    models_dir = os.path.dirname(__file__)
    skip_files = {'__init__.py', 'registry.py', '__pycache__'}

    # Root-level .py files
    for filename in sorted(os.listdir(models_dir)):
        if filename in skip_files or not filename.endswith('.py'):
            continue
        try:
            importlib.import_module(f'models.{filename[:-3]}')
        except Exception as e:
            print(f"Warning: Could not load {filename}: {e}")

    # Subdirectory packages (e.g., y2024/) — import each .py individually
    for item in sorted(os.listdir(models_dir)):
        item_path = os.path.join(models_dir, item)
        if not os.path.isdir(item_path) or item.startswith(('.', '_')):
            continue
        for fname in sorted(os.listdir(item_path)):
            if fname in skip_files or not fname.endswith('.py'):
                continue
            module_name = f'models.{item}.{fname[:-3]}'
            try:
                importlib.import_module(module_name)
            except Exception as e:
                print(f"Warning: Could not load {module_name}: {e}")

    _models_loaded = True


# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────

def create_model(model_name, **kwargs):
    """Create a model by registered name.

    Standard kwargs accepted by most models:
        num_classes, bands, patch_size
    """
    _load_models()

    if model_name not in _model_registry:
        available = sorted(_model_registry.keys())
        raise ValueError(f"Model '{model_name}' not found. Available: {available}")

    return_config = kwargs.pop('return_config', False)

    if model_name in _model_configs:
        cfg = _model_configs[model_name].copy()
        cfg.update(kwargs)
        model_args = {k: v for k, v in cfg.items() if k != 'expects_4d'}
        model = _model_registry[model_name](**model_args)
        if return_config:
            return cfg, model
        return model

    model = _model_registry[model_name](**kwargs)
    if return_config:
        return kwargs, model
    return model


def list_models():
    """Return sorted list of all registered model names."""
    _load_models()
    return sorted(_model_registry.keys())


def get_model_config(model_name):
    """Get the default config dict for a model."""
    _load_models()
    return _model_configs.get(model_name, {})


def get_model_info(model_name):
    """Get catalog metadata for a model.

    Returns a dict with keys: full_name, paper_title, paper (DOI/arXiv URL),
    code, year, venue. Returns an empty dict if the model is not in the
    catalog. Unpublished entries have None for paper_title/paper/code.
    """
    return MODEL_CATALOG.get(model_name, {})


_ABLATION_PREFIXES = ('HSSFN_', 'FuzzySpectralMamba-')

_VENUE_W = 34   # longest real venue is ~51 chars, so long ones are elided


def _is_ablation(name):
    return name.startswith(_ABLATION_PREFIXES)


def print_model_catalog(sort_by='year'):
    """Pretty-print the full model catalog.

    Args:
        sort_by: 'year' (default, then name) or 'name'.
    """
    _load_models()
    registered = sorted(_model_registry.keys())

    main_models = [m for m in registered if not _is_ablation(m)]
    ablation_models = [m for m in registered if _is_ablation(m)]

    if sort_by == 'year':
        main_models.sort(key=lambda m: (MODEL_CATALOG.get(m, {}).get('year') or 0, m.lower()))
    else:
        main_models.sort(key=str.lower)

    width = 25 + _VENUE_W + 60
    print(f"\n{'='*width}")
    print(f"  Model Catalog ({len(main_models)} models"
          + (f" + {len(ablation_models)} ablation variants" if ablation_models else "") + ")")
    print(f"{'='*width}")
    print(f"  {'Model':<25} {'4D':>3}  {'Year':>4}  {'Venue':<{_VENUE_W}}  {'Paper / Code'}")
    print(f"  {'-'*(width-5)}")

    for name in main_models:
        info = MODEL_CATALOG.get(name, {})
        cfg = _model_configs.get(name, {})
        is_4d = 'yes' if cfg.get('expects_4d', False) else '-'
        year = str(info.get('year') or '—')
        venue = info.get('venue') or '—'
        if len(venue) > _VENUE_W:
            venue = venue[:_VENUE_W - 1] + '…'
        link = info.get('paper') or info.get('code') or '—'
        print(f"  {name:<25} {is_4d:>3}  {year:>4}  {venue:<{_VENUE_W}}  {link}")

    if ablation_models:
        print(f"\n  Ablation variants: {', '.join(ablation_models)}")

    # Surface drift between the catalog and what actually registered.
    missing = sorted(set(main_models) - set(MODEL_CATALOG))
    unregistered = sorted(set(MODEL_CATALOG) - set(registered))
    if missing:
        print(f"\n  Warning: registered but missing catalog metadata: {', '.join(missing)}")
    if unregistered:
        print(f"  Warning: in catalog but not registered: {', '.join(unregistered)}")

    print(f"{'='*width}\n")
