import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class StructuralSignature:
    """Encapsulates the structural properties of a vector field."""
    entropy: torch.Tensor      # Structure tensor entropy field
    alignment: torch.Tensor    # Local directional consistency
    curvature: torch.Tensor    # Directional change intensity
    
    def __post_init__(self):
        """Validate tensor shapes match."""
        shapes = [self.entropy.shape, self.alignment.shape, self.curvature.shape]
        if not all(shape == shapes[0] for shape in shapes):
            raise ValueError("All structural metrics must have matching shapes")
    
    @property
    def device(self) -> torch.device:
        return self.entropy.device
    
    def to(self, device: torch.device) -> 'StructuralSignature':
        """Move all tensors to specified device."""
        return StructuralSignature(
            entropy=self.entropy.to(device),
            alignment=self.alignment.to(device),
            curvature=self.curvature.to(device)
        )
    
    def global_statistics(self) -> Dict[str, torch.Tensor]:
        """Compute global statistical summary of structural properties."""
        stats = {}
        for name, field in [('entropy', self.entropy), ('alignment', self.alignment), ('curvature', self.curvature)]:
            stats.update({
                f'{name}_mean': field.mean(dim=(-2, -1)),     # Spatial mean
                f'{name}_std': field.std(dim=(-2, -1)),       # Spatial std
                f'{name}_max': field.amax(dim=(-2, -1)),      # Spatial max
                f'{name}_min': field.amin(dim=(-2, -1)),      # Spatial min
            })
        return stats


class VectorNeuronLayer(nn.Module):
    """
    2D Polar Vector Neuron Layer where each neuron outputs (magnitude, angle).
    
    The core building block for structure-first networks. Each neuron represents
    a 2D polar vector that contributes to the overall vector field structure.
    """
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, 
                 stride: int = 1, padding: int = 1, magnitude_activation: str = 'relu',
                 angle_activation: str = 'none'):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Magnitude branch - learns vector strengths
        self.magnitude_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.magnitude_bn = nn.BatchNorm2d(out_channels)
        
        # Angle branch - learns vector directions  
        self.angle_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.angle_bn = nn.BatchNorm2d(out_channels)
        
        # Activation functions
        self.magnitude_activation = self._get_activation(magnitude_activation)
        self.angle_activation = self._get_activation(angle_activation)
        
        self._initialize_weights()
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            'relu': nn.ReLU(inplace=True),
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh(),
            'none': nn.Identity()
        }
        if activation not in activations:
            raise ValueError(f"Unknown activation: {activation}")
        return activations[activation]
    
    def _initialize_weights(self):
        """Initialize weights for stable vector field generation."""
        # Xavier initialization for magnitude branch
        nn.init.xavier_uniform_(self.magnitude_conv.weight)
        nn.init.zeros_(self.magnitude_conv.bias)
        
        # Uniform initialization for angle branch to cover full angular range
        nn.init.uniform_(self.angle_conv.weight, -np.pi, np.pi)
        nn.init.zeros_(self.angle_conv.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass producing polar vector field.
        
        Args:
            x: Input tensor [B, in_channels, H, W]
            
        Returns:
            Polar vector field [B, out_channels*2, H, W] where 
            channels 0:out_channels are magnitudes, out_channels:2*out_channels are angles
        """
        # Compute magnitude components
        mag = self.magnitude_conv(x)
        mag = self.magnitude_bn(mag)
        mag = self.magnitude_activation(mag)
        
        # Compute angle components  
        angle = self.angle_conv(x)
        angle = self.angle_bn(angle)
        angle = self.angle_activation(angle)
        
        # Normalize angles to [-π, π] range
        angle = torch.atan2(torch.sin(angle), torch.cos(angle))
        
        # Concatenate magnitude and angle channels
        return torch.cat([mag, angle], dim=1)


class StructuralAnalyzer(nn.Module):
    """
    Analyzes vector fields to extract structural signatures.
    
    This is the core component that transforms vector fields into 
    structural representations for classification.
    """
    
    def __init__(self, window_size: int = 5, sigma: float = 1.0):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.eps = 1e-10
        
        # Create and register analysis kernels
        gaussian_kernel = self._create_gaussian_kernel()
        self.register_buffer('gaussian_kernel', gaussian_kernel)
        
        sobel_x, sobel_y = self._create_sobel_kernels()
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
    
    def _create_gaussian_kernel(self) -> torch.Tensor:
        """Create normalized Gaussian kernel."""
        kernel = torch.zeros(self.window_size, self.window_size, dtype=torch.float32)
        center = self.window_size // 2
        
        y, x = torch.meshgrid(
            torch.arange(self.window_size, dtype=torch.float32) - center,
            torch.arange(self.window_size, dtype=torch.float32) - center,
            indexing='ij'
        )
        
        kernel = torch.exp(-(x**2 + y**2) / (2 * self.sigma**2))
        kernel = kernel / kernel.sum()
        
        return kernel.view(1, 1, self.window_size, self.window_size)
    
    def _create_sobel_kernels(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create Sobel kernels for gradient computation."""
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32) / 8.0
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32) / 8.0
        
        return sobel_x.view(1, 1, 3, 3), sobel_y.view(1, 1, 3, 3)
    
    def analyze_vector_field(self, vector_field: torch.Tensor) -> StructuralSignature:
        """
        Extract complete structural signature from vector field.
        
        Args:
            vector_field: [B, 2*C, H, W] where first C channels are magnitudes,
                         last C channels are angles
        
        Returns:
            StructuralSignature containing entropy, alignment, and curvature
        """
        B, channels, H, W = vector_field.shape
        if channels % 2 != 0:
            raise ValueError("Vector field must have even number of channels (magnitude/angle pairs)")
        
        num_vectors = channels // 2
        
        # Process each vector component and aggregate
        all_entropy = []
        all_alignment = []
        all_curvature = []
        
        for i in range(num_vectors):
            mag_idx = i
            angle_idx = i + num_vectors
            
            # Extract single vector field
            single_field = torch.stack([
                vector_field[:, mag_idx:mag_idx+1],
                vector_field[:, angle_idx:angle_idx+1]
            ], dim=1).squeeze(2)  # [B, 2, H, W]
            
            # Compute structural metrics
            entropy = self._compute_entropy(single_field)
            alignment = self._compute_alignment(single_field)
            curvature = self._compute_curvature(single_field)
            
            all_entropy.append(entropy)
            all_alignment.append(alignment)
            all_curvature.append(curvature)
        
        # Aggregate across vector components
        entropy_field = torch.stack(all_entropy, dim=1).mean(dim=1, keepdim=True)
        alignment_field = torch.stack(all_alignment, dim=1).mean(dim=1, keepdim=True)
        curvature_field = torch.stack(all_curvature, dim=1).mean(dim=1, keepdim=True)
        
        return StructuralSignature(
            entropy=entropy_field,
            alignment=alignment_field,
            curvature=curvature_field
        )
    
    def _compute_entropy(self, vector_field: torch.Tensor) -> torch.Tensor:
        """Compute structure tensor entropy."""
        magnitudes = vector_field[:, 0:1]
        angles = vector_field[:, 1:2]
        
        # Convert to Cartesian
        vx = magnitudes * torch.cos(angles)
        vy = magnitudes * torch.sin(angles)
        
        # Structure tensor components
        vxx = vx * vx
        vyy = vy * vy
        vxy = vx * vy
        
        # Smooth with Gaussian kernel
        pad = self.window_size // 2
        vxx_smooth = F.conv2d(vxx, self.gaussian_kernel, padding=pad)
        vyy_smooth = F.conv2d(vyy, self.gaussian_kernel, padding=pad)
        vxy_smooth = F.conv2d(vxy, self.gaussian_kernel, padding=pad)
        
        # Eigenvalues
        trace = vxx_smooth + vyy_smooth
        det = vxx_smooth * vyy_smooth - vxy_smooth * vxy_smooth
        
        discriminant = torch.sqrt(torch.clamp(trace * trace - 4 * det, min=0.0) + self.eps)
        lambda1 = torch.clamp((trace + discriminant) / 2, min=self.eps)
        lambda2 = torch.clamp((trace - discriminant) / 2, min=self.eps)
        
        # Normalize and compute entropy
        sum_eig = lambda1 + lambda2 + self.eps
        p1, p2 = lambda1 / sum_eig, lambda2 / sum_eig
        
        entropy = -(p1 * torch.log(p1 + self.eps) + p2 * torch.log(p2 + self.eps)) / np.log(2.0)
        return torch.clamp(entropy, 0.0, 1.0)
    
    def _compute_alignment(self, vector_field: torch.Tensor) -> torch.Tensor:
        """Compute local alignment field."""
        angles = vector_field[:, 1:2]
        
        ux = torch.cos(angles)
        uy = torch.sin(angles)
        
        pad = self.window_size // 2
        ux_avg = F.conv2d(ux, self.gaussian_kernel, padding=pad)
        uy_avg = F.conv2d(uy, self.gaussian_kernel, padding=pad)
        
        alignment = torch.sqrt(ux_avg**2 + uy_avg**2 + self.eps)
        return torch.clamp(alignment, 0.0, 1.0)
    
    def _compute_curvature(self, vector_field: torch.Tensor) -> torch.Tensor:
        """Compute curvature field."""
        angles = vector_field[:, 1:2]
        
        ux = torch.cos(angles)
        uy = torch.sin(angles)
        
        # Compute gradients
        dudx = F.conv2d(ux, self.sobel_x, padding=1)
        dudy = F.conv2d(ux, self.sobel_y, padding=1)
        dvdx = F.conv2d(uy, self.sobel_x, padding=1)
        dvdy = F.conv2d(uy, self.sobel_y, padding=1)
        
        # Frobenius norm
        curvature = torch.sqrt(dudx**2 + dudy**2 + dvdx**2 + dvdy**2 + self.eps)
        
        # Optional smoothing
        pad = self.window_size // 2
        curvature = F.conv2d(curvature, self.gaussian_kernel, padding=pad)
        
        return curvature


class StructuralClassifier(nn.Module):
    """
    Classifier that operates on structural signatures.
    
    Takes structural signatures and produces class predictions based on
    the principle that similar structures should yield similar classifications.
    """
    
    def __init__(self, num_classes: int, signature_channels: int = 3, 
                 hidden_dim: int = 128, use_spatial_pooling: bool = True):
        super().__init__()
        
        self.num_classes = num_classes
        self.signature_channels = signature_channels  # entropy, alignment, curvature
        self.use_spatial_pooling = use_spatial_pooling
        
        if use_spatial_pooling:
            # Global spatial pooling + MLP classifier
            self.global_pool = nn.AdaptiveAvgPool2d(1)
            self.classifier = nn.Sequential(
                nn.Linear(signature_channels * 4, hidden_dim),  # 4 stats per signature
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim // 2, num_classes)
            )
        else:
            # Convolutional classifier preserving spatial structure
            self.classifier = nn.Sequential(
                nn.Conv2d(signature_channels, hidden_dim, 3, padding=1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(4),
                nn.Flatten(),
                nn.Linear(hidden_dim * 16, num_classes)
            )
    
    def forward(self, signature: StructuralSignature) -> torch.Tensor:
        """
        Classify based on structural signature.
        
        Args:
            signature: StructuralSignature containing entropy, alignment, curvature
            
        Returns:
            Class logits [B, num_classes]
        """
        if self.use_spatial_pooling:
            # Extract global statistics for classification
            stats = signature.global_statistics()
            features = torch.cat([
                stats['entropy_mean'], stats['entropy_std'], 
                stats['entropy_max'], stats['entropy_min'],
                stats['alignment_mean'], stats['alignment_std'],
                stats['alignment_max'], stats['alignment_min'],
                stats['curvature_mean'], stats['curvature_std'],
                stats['curvature_max'], stats['curvature_min']
            ], dim=1)
            
            return self.classifier(features)
        else:
            # Concatenate structural fields for spatial processing
            structure_tensor = torch.cat([
                signature.entropy, signature.alignment, signature.curvature
            ], dim=1)
            
            return self.classifier(structure_tensor)


class StructureFirstNetwork(nn.Module):
    """
    Complete Structure-First Neural Network.
    
    Implements the principle: "Structure makes classification trivial via vector neuron processing"
    
    Architecture:
    1. Vector neuron layers build hidden vector fields
    2. Structural analyzer extracts entropy/alignment/curvature signatures  
    3. Structural classifier predicts classes based on field similarities
    """
    
    def __init__(self, 
                 input_channels: int,
                 num_classes: int,
                 vector_channels: List[int] = [32, 64, 128],
                 window_size: int = 5,
                 sigma: float = 1.0,
                 classifier_hidden: int = 128,
                 use_spatial_pooling: bool = True):
        super().__init__()
        
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.vector_channels = vector_channels
        
        # Build vector neuron backbone
        self.vector_layers = nn.ModuleList()
        in_ch = input_channels
        
        for i, out_ch in enumerate(vector_channels):
            stride = 2 if i > 0 else 1  # Downsample after first layer
            self.vector_layers.append(
                VectorNeuronLayer(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    stride=stride,
                    magnitude_activation='relu',
                    angle_activation='tanh'
                )
            )
            in_ch = out_ch * 2  # Account for magnitude + angle channels
        
        # Structural analysis
        self.structural_analyzer = StructuralAnalyzer(window_size, sigma)
        
        # Classification head
        self.classifier = StructuralClassifier(
            num_classes=num_classes,
            signature_channels=3,
            hidden_dim=classifier_hidden,
            use_spatial_pooling=use_spatial_pooling
        )
    
    def forward(self, x: torch.Tensor, return_signature: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, StructuralSignature]]:
        """
        Forward pass through structure-first network.
        
        Args:
            x: Input tensor [B, C, H, W]
            return_signature: Whether to return structural signature
            
        Returns:
            Class logits [B, num_classes] or (logits, signature) if return_signature=True
        """
        # Pass through vector neuron layers
        vector_field = x
        for layer in self.vector_layers:
            vector_field = layer(vector_field)
        
        # Extract structural signature
        signature = self.structural_analyzer.analyze_vector_field(vector_field)
        
        # Classify based on structure
        logits = self.classifier(signature)
        
        if return_signature:
            return logits, signature
        return logits
    
    def compute_structural_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract structural embeddings for similarity analysis.
        
        Returns flattened structural signatures that can be used for 
        computing similarities, clustering, or visualization.
        """
        with torch.no_grad():
            _, signature = self.forward(x, return_signature=True)
            
            # Flatten spatial dimensions and concatenate metrics
            B = signature.entropy.size(0)
            entropy_flat = signature.entropy.view(B, -1)
            alignment_flat = signature.alignment.view(B, -1)  
            curvature_flat = signature.curvature.view(B, -1)
            
            # Concatenate all structural metrics
            embeddings = torch.cat([entropy_flat, alignment_flat, curvature_flat], dim=1)
            
            return embeddings
    
    def get_structural_similarity(self, x1: torch.Tensor, x2: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute structural similarity between two inputs.
        
        This validates the core principle: similar inputs should have similar structures.
        """
        with torch.no_grad():
            _, sig1 = self.forward(x1, return_signature=True)
            _, sig2 = self.forward(x2, return_signature=True)
            
            # Compute MSE between structural signatures
            return {
                'entropy_similarity': F.mse_loss(sig1.entropy, sig2.entropy),
                'alignment_similarity': F.mse_loss(sig1.alignment, sig2.alignment),
                'curvature_similarity': F.mse_loss(sig1.curvature, sig2.curvature)
            }


class StructuralContrastiveLoss(nn.Module):
    """
    Contrastive loss that enforces structural similarity for same-class examples
    and structural dissimilarity for different-class examples.
    
    Core principle: "Similar examples should have similar structure,
                     dissimilar examples should have dissimilar structure"
    """
    
    def __init__(self, margin: float = 1.0, temperature: float = 0.1):
        super().__init__()
        self.margin = margin
        self.temperature = temperature
    
    def structural_distance(self, sig1: StructuralSignature, sig2: StructuralSignature) -> torch.Tensor:
        """Compute structural distance between two signatures."""
        entropy_dist = F.mse_loss(sig1.entropy, sig2.entropy, reduction='none').mean(dim=(-2, -1))
        alignment_dist = F.mse_loss(sig1.alignment, sig2.alignment, reduction='none').mean(dim=(-2, -1))
        curvature_dist = F.mse_loss(sig1.curvature, sig2.curvature, reduction='none').mean(dim=(-2, -1))
        
        # Weighted combination of structural distances
        return entropy_dist + alignment_dist + curvature_dist
    
    def forward(self, signatures: StructuralSignature, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute contrastive loss over batch of structural signatures.
        
        Args:
            signatures: Batch of structural signatures [B, ...]
            targets: Class labels [B]
        
        Returns:
            Contrastive loss encouraging structural similarity within classes
        """
        batch_size = targets.size(0)
        device = targets.device
        
        # Create pairwise label comparison matrix
        labels_equal = targets.unsqueeze(1) == targets.unsqueeze(0)  # [B, B]
        mask = ~torch.eye(batch_size, dtype=torch.bool, device=device)  # Exclude diagonal
        
        contrastive_loss = 0.0
        num_pairs = 0
        
        for i in range(batch_size):
            for j in range(i + 1, batch_size):
                # Extract signatures for pair (i, j)
                sig_i = StructuralSignature(
                    entropy=signatures.entropy[i:i+1],
                    alignment=signatures.alignment[i:i+1], 
                    curvature=signatures.curvature[i:i+1]
                )
                sig_j = StructuralSignature(
                    entropy=signatures.entropy[j:j+1],
                    alignment=signatures.alignment[j:j+1],
                    curvature=signatures.curvature[j:j+1]
                )
                
                distance = self.structural_distance(sig_i, sig_j)
                
                if labels_equal[i, j]:
                    # Same class: minimize structural distance
                    contrastive_loss += distance.mean()
                else:
                    # Different class: maximize structural distance (with margin)
                    contrastive_loss += torch.clamp(self.margin - distance, min=0.0).mean()
                
                num_pairs += 1
        
        return contrastive_loss / num_pairs if num_pairs > 0 else torch.tensor(0.0, device=device)


class StructuralTripletLoss(nn.Module):
    """
    Triplet loss for structural signatures: anchor-positive-negative.
    
    Ensures anchor-positive structural distance < anchor-negative structural distance + margin
    """
    
    def __init__(self, margin: float = 1.0, mining_strategy: str = 'hard'):
        super().__init__()
        self.margin = margin
        self.mining_strategy = mining_strategy
    
    def structural_distance(self, sig1: StructuralSignature, sig2: StructuralSignature) -> torch.Tensor:
        """Compute structural distance between signatures."""
        entropy_dist = F.mse_loss(sig1.entropy, sig2.entropy, reduction='none').mean(dim=(-2, -1))
        alignment_dist = F.mse_loss(sig1.alignment, sig2.alignment, reduction='none').mean(dim=(-2, -1))
        curvature_dist = F.mse_loss(sig1.curvature, sig2.curvature, reduction='none').mean(dim=(-2, -1))
        
        return entropy_dist + alignment_dist + curvature_dist
    
    def mine_triplets(self, signatures: StructuralSignature, targets: torch.Tensor) -> List[Tuple[int, int, int]]:
        """Mine triplets based on structural distances."""
        batch_size = targets.size(0)
        triplets = []
        
        for anchor_idx in range(batch_size):
            anchor_label = targets[anchor_idx]
            
            # Find positives (same class)
            positive_indices = (targets == anchor_label).nonzero(as_tuple=True)[0]
            positive_indices = positive_indices[positive_indices != anchor_idx]
            
            # Find negatives (different class)
            negative_indices = (targets != anchor_label).nonzero(as_tuple=True)[0]
            
            if len(positive_indices) == 0 or len(negative_indices) == 0:
                continue
            
            anchor_sig = StructuralSignature(
                entropy=signatures.entropy[anchor_idx:anchor_idx+1],
                alignment=signatures.alignment[anchor_idx:anchor_idx+1],
                curvature=signatures.curvature[anchor_idx:anchor_idx+1]
            )
            
            if self.mining_strategy == 'hard':
                # Hard positive: most dissimilar same-class example
                pos_distances = []
                for pos_idx in positive_indices:
                    pos_sig = StructuralSignature(
                        entropy=signatures.entropy[pos_idx:pos_idx+1],
                        alignment=signatures.alignment[pos_idx:pos_idx+1],
                        curvature=signatures.curvature[pos_idx:pos_idx+1]
                    )
                    pos_distances.append(self.structural_distance(anchor_sig, pos_sig).item())
                
                hard_positive_idx = positive_indices[np.argmax(pos_distances)]
                
                # Hard negative: most similar different-class example
                neg_distances = []
                for neg_idx in negative_indices:
                    neg_sig = StructuralSignature(
                        entropy=signatures.entropy[neg_idx:neg_idx+1],
                        alignment=signatures.alignment[neg_idx:neg_idx+1],
                        curvature=signatures.curvature[neg_idx:neg_idx+1]
                    )
                    neg_distances.append(self.structural_distance(anchor_sig, neg_sig).item())
                
                hard_negative_idx = negative_indices[np.argmin(neg_distances)]
                
                triplets.append((anchor_idx, hard_positive_idx.item(), hard_negative_idx.item()))
            
            else:  # Random mining
                positive_idx = positive_indices[torch.randint(len(positive_indices), (1,))].item()
                negative_idx = negative_indices[torch.randint(len(negative_indices), (1,))].item()
                triplets.append((anchor_idx, positive_idx, negative_idx))
        
        return triplets
    
    def forward(self, signatures: StructuralSignature, targets: torch.Tensor) -> torch.Tensor:
        """Compute triplet loss for structural signatures."""
        triplets = self.mine_triplets(signatures, targets)
        
        if not triplets:
            return torch.tensor(0.0, device=targets.device)
        
        total_loss = 0.0
        
        for anchor_idx, positive_idx, negative_idx in triplets:
            # Extract signatures
            anchor_sig = StructuralSignature(
                entropy=signatures.entropy[anchor_idx:anchor_idx+1],
                alignment=signatures.alignment[anchor_idx:anchor_idx+1],
                curvature=signatures.curvature[anchor_idx:anchor_idx+1]
            )
            positive_sig = StructuralSignature(
                entropy=signatures.entropy[positive_idx:positive_idx+1],
                alignment=signatures.alignment[positive_idx:positive_idx+1],
                curvature=signatures.curvature[positive_idx:positive_idx+1]
            )
            negative_sig = StructuralSignature(
                entropy=signatures.entropy[negative_idx:negative_idx+1],
                alignment=signatures.alignment[negative_idx:negative_idx+1],
                curvature=signatures.curvature[negative_idx:negative_idx+1]
            )
            
            # Compute distances
            pos_distance = self.structural_distance(anchor_sig, positive_sig)
            neg_distance = self.structural_distance(anchor_sig, negative_sig)
            
            # Triplet loss: d(a,p) + margin < d(a,n)
            triplet_loss = torch.clamp(pos_distance - neg_distance + self.margin, min=0.0)
            total_loss += triplet_loss.mean()
        
        return total_loss / len(triplets)


class StructuralInfoNCELoss(nn.Module):
    """
    InfoNCE-style contrastive loss for structural signatures.
    
    Treats same-class examples as positives and different-class as negatives
    in a softmax-normalized contrastive framework.
    """
    
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature
    
    def structural_similarity(self, sig1: StructuralSignature, sig2: StructuralSignature) -> torch.Tensor:
        """Compute structural similarity (negative distance)."""
        entropy_sim = -F.mse_loss(sig1.entropy, sig2.entropy, reduction='none').mean(dim=(-2, -1))
        alignment_sim = -F.mse_loss(sig1.alignment, sig2.alignment, reduction='none').mean(dim=(-2, -1))
        curvature_sim = -F.mse_loss(sig1.curvature, sig2.curvature, reduction='none').mean(dim=(-2, -1))
        
        return (entropy_sim + alignment_sim + curvature_sim) / 3.0
    
    def forward(self, signatures: StructuralSignature, targets: torch.Tensor) -> torch.Tensor:
        """Compute InfoNCE loss for structural signatures."""
        batch_size = targets.size(0)
        device = targets.device
        
        # Compute all pairwise similarities
        similarities = torch.zeros(batch_size, batch_size, device=device)
        
        for i in range(batch_size):
            for j in range(batch_size):
                if i != j:
                    sig_i = StructuralSignature(
                        entropy=signatures.entropy[i:i+1],
                        alignment=signatures.alignment[i:i+1],
                        curvature=signatures.curvature[i:i+1]
                    )
                    sig_j = StructuralSignature(
                        entropy=signatures.entropy[j:j+1],
                        alignment=signatures.alignment[j:j+1],
                        curvature=signatures.curvature[j:j+1]
                    )
                    
                    similarities[i, j] = self.structural_similarity(sig_i, sig_j)
        
        # Apply temperature scaling
        similarities = similarities / self.temperature
        
        # Create positive mask (same class, excluding diagonal)
        labels_equal = targets.unsqueeze(1) == targets.unsqueeze(0)
        mask = ~torch.eye(batch_size, dtype=torch.bool, device=device)
        positive_mask = labels_equal & mask
        
        # InfoNCE loss
        total_loss = 0.0
        num_valid = 0
        
        for i in range(batch_size):
            if positive_mask[i].sum() > 0:  # Has at least one positive
                # Numerator: sum over all positives
                positive_similarities = similarities[i][positive_mask[i]]
                numerator = torch.logsumexp(positive_similarities, dim=0)
                
                # Denominator: sum over all negatives (excluding self)
                denominator = torch.logsumexp(similarities[i][mask[i]], dim=0)
                
                total_loss += denominator - numerator
                num_valid += 1
        
        return total_loss / num_valid if num_valid > 0 else torch.tensor(0.0, device=device)


# Enhanced composite loss with contrastive learning
def create_structure_first_loss(alpha_classification: float = 10.0,
                               alpha_contrastive: float = 1.0,
                               alpha_triplet: float = 0.5,
                               alpha_infonce: float = 0.5,
                               contrastive_type: str = 'all') -> nn.Module:
    """
    Create comprehensive structure-first loss with contrastive learning.
    
    Args:
        alpha_classification: Weight for classification loss
        alpha_contrastive: Weight for pairwise contrastive loss
        alpha_triplet: Weight for triplet loss
        alpha_infonce: Weight for InfoNCE loss
        contrastive_type: Which contrastive losses to use ('contrastive', 'triplet', 'infonce', 'all')
    """
    class StructureFirstLoss(nn.Module):
        def __init__(self):
            super().__init__()
            self.classification_loss = nn.CrossEntropyLoss()
            
            # Contrastive loss components
            if contrastive_type in ['contrastive', 'all']:
                self.contrastive_loss = StructuralContrastiveLoss(margin=1.0)
            if contrastive_type in ['triplet', 'all']:
                self.triplet_loss = StructuralTripletLoss(margin=1.0, mining_strategy='hard')
            if contrastive_type in ['infonce', 'all']:
                self.infonce_loss = StructuralInfoNCELoss(temperature=0.1)
        
        def forward(self, logits: torch.Tensor, targets: torch.Tensor, 
                   signatures: StructuralSignature) -> Dict[str, torch.Tensor]:
            
            # Classification loss
            cls_loss = self.classification_loss(logits, targets)
            
            losses = {
                'classification': cls_loss,
                'total': alpha_classification * cls_loss
            }
            
            # Add contrastive losses
            if contrastive_type in ['contrastive', 'all'] and alpha_contrastive > 0:
                contrastive = self.contrastive_loss(signatures, targets)
                losses['contrastive'] = contrastive
                losses['total'] += alpha_contrastive * contrastive
            
            if contrastive_type in ['triplet', 'all'] and alpha_triplet > 0:
                triplet = self.triplet_loss(signatures, targets)
                losses['triplet'] = triplet
                losses['total'] += alpha_triplet * triplet
            
            if contrastive_type in ['infonce', 'all'] and alpha_infonce > 0:
                infonce = self.infonce_loss(signatures, targets)
                losses['infonce'] = infonce
                losses['total'] += alpha_infonce * infonce
            
            return losses
    
    return StructureFirstLoss()


def visualize_structural_evolution(model: StructureFirstNetwork, 
                                 dataloader: torch.utils.data.DataLoader,
                                 num_samples: int = 4) -> Dict[str, torch.Tensor]:
    """
    Visualize how structural signatures evolve during training.
    
    Useful for validating the structure-first principle.
    """
    model.eval()
    signatures_by_class = {}
    
    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(dataloader):
            if batch_idx >= num_samples:
                break
                
            logits, signature = model(data, return_signature=True)
            
            for i, target in enumerate(targets):
                class_id = target.item()
                if class_id not in signatures_by_class:
                    signatures_by_class[class_id] = []
                
                signatures_by_class[class_id].append({
                    'entropy': signature.entropy[i],
                    'alignment': signature.alignment[i],
                    'curvature': signature.curvature[i]
                })
    
    return signatures_by_class


# Training function with contrastive learning
def train_structure_first_model(model: StructureFirstNetwork,
                               train_loader: torch.utils.data.DataLoader,
                               val_loader: torch.utils.data.DataLoader,
                               num_epochs: int = 100,
                               lr: float = 1e-3,
                               device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    """
    Train structure-first model with contrastive learning.
    
    Monitors both classification accuracy and structural consistency.
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Create loss function with contrastive learning
    loss_fn = create_structure_first_loss(
        alpha_classification=10.0,
        alpha_contrastive=1.0,
        alpha_triplet=0.5,
        alpha_infonce=0.5,
        contrastive_type='all'
    )
    
    best_val_acc = 0.0
    training_history = {
        'train_acc': [], 'val_acc': [],
        'classification_loss': [], 'contrastive_loss': [],
        'triplet_loss': [], 'infonce_loss': []
    }
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_correct = 0
        train_total = 0
        epoch_losses = {'classification': 0, 'contrastive': 0, 'triplet': 0, 'infonce': 0}
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass with structural signatures
            logits, signatures = model(data, return_signature=True)
            
            # Compute losses
            losses = loss_fn(logits, targets, signatures)
            
            # Backward pass
            losses['total'].backward()
            optimizer.step()
            
            # Accumulate metrics
            _, predicted = logits.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
            
            for key in epoch_losses:
                if key in losses:
                    epoch_losses[key] += losses[key].item()
        
        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(device), targets.to(device)
                logits = model(data)
                
                _, predicted = logits.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
        
        # Update learning rate
        scheduler.step()
        
        # Record metrics
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        
        training_history['train_acc'].append(train_acc)
        training_history['val_acc'].append(val_acc)
        
        for key in epoch_losses:
            if key not in training_history:
                training_history[key] = []
            training_history[key].append(epoch_losses[key] / len(train_loader))
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, 'best_structure_first_model.pth')
        
        # Print progress
        if epoch % 10 == 0:
            print(f'Epoch {epoch:3d}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%')
            print(f'  Losses - Cls: {epoch_losses["classification"]:.4f}, ' +
                  f'Cont: {epoch_losses["contrastive"]:.4f}, ' +
                  f'Trip: {epoch_losses["triplet"]:.4f}, ' +
                  f'InfoNCE: {epoch_losses["infonce"]:.4f}')
    
    print(f'Training complete. Best validation accuracy: {best_val_acc:.2f}%')
    return training_history


def analyze_structural_consistency(model: StructureFirstNetwork,
                                 dataloader: torch.utils.data.DataLoader,
                                 num_classes: int,
                                 device: torch.device) -> Dict[str, float]:
    """
    Analyze how well the model achieves structural consistency.
    
    Validates the core principle: similar examples should have similar structures.
    """
    model.eval()
    
    # Collect structural embeddings by class
    embeddings_by_class = {i: [] for i in range(num_classes)}
    
    with torch.no_grad():
        for data, targets in dataloader:
            data, targets = data.to(device), targets.to(device)
            embeddings = model.compute_structural_embeddings(data)
            
            for i, target in enumerate(targets):
                embeddings_by_class[target.item()].append(embeddings[i])
    
    # Compute intra-class and inter-class distances
    intra_class_distances = []
    inter_class_distances = []
    
    for class_id in range(num_classes):
        if len(embeddings_by_class[class_id]) < 2:
            continue
            
        class_embeddings = torch.stack(embeddings_by_class[class_id])
        
        # Intra-class distances (should be small)
        for i in range(len(class_embeddings)):
            for j in range(i + 1, len(class_embeddings)):
                dist = F.mse_loss(class_embeddings[i], class_embeddings[j], reduction='mean')
                intra_class_distances.append(dist.item())
        
        # Inter-class distances (should be large)
        for other_class_id in range(class_id + 1, num_classes):
            if len(embeddings_by_class[other_class_id]) == 0:
                continue
                
            other_embeddings = torch.stack(embeddings_by_class[other_class_id])
            
            for class_emb in class_embeddings:
                for other_emb in other_embeddings:
                    dist = F.mse_loss(class_emb, other_emb, reduction='mean')
                    inter_class_distances.append(dist.item())
    
    # Compute metrics
    avg_intra_dist = np.mean(intra_class_distances) if intra_class_distances else 0.0
    avg_inter_dist = np.mean(inter_class_distances) if inter_class_distances else 0.0
    
    # Structural consistency score (higher is better)
    consistency_score = avg_inter_dist / (avg_intra_dist + 1e-8)
    
    return {
        'intra_class_distance': avg_intra_dist,
        'inter_class_distance': avg_inter_dist,
        'consistency_score': consistency_score,
        'separation_ratio': avg_inter_dist / avg_intra_dist if avg_intra_dist > 0 else float('inf')
    }
    # Create structure-first network
    model = StructureFirstNetwork(
        input_channels=3,
        num_classes=10,
        vector_channels=[32, 64, 128],
        window_size=5,
        sigma=1.0
    )
    
    # Test forward pass
    test_input = torch.randn(2, 3, 64, 64)
    logits, signature = model(test_input, return_signature=True)
    
    print(f"Input shape: {test_input.shape}")
    print(f"Output logits shape: {logits.shape}")
    print(f"Entropy field shape: {signature.entropy.shape}")
    print(f"Alignment field shape: {signature.alignment.shape}")
    print(f"Curvature field shape: {signature.curvature.shape}")
    
    # Test structural similarity
    test_input2 = torch.randn(2, 3, 64, 64)
    similarity = model.get_structural_similarity(test_input, test_input2)
    print(f"Structural similarity: {similarity}")
    
    # Test loss function
    loss_fn = create_structure_first_loss()
    targets = torch.randint(0, 10, (2,))
    
    # Get intermediate vector field for loss computation
    with torch.no_grad():
        vector_field = test_input
        for layer in model.vector_layers:
            vector_field = layer(vector_field)
    
    losses = loss_fn(logits, targets, vector_field)
    print(f"Training losses: {losses}")
