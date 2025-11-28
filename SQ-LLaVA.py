import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    CLIPVisionModel,
    CLIPImageProcessor,
    AutoTokenizer,
    AutoModelForCausalLM,
    get_cosine_schedule_with_warmup
)

from torch.utils.data import Dataset, DataLoader
from peft import LoraConfig, get_peft_model, TaskType
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import math

@dataclass
class SQConfig:
    """Configuration for SQ-LLaVA"""
    # Model configs
    vision_model: str = "openai/clip-vit-base-patch32"  # Smaller CLIP
    llm_model: str = "microsoft/phi-2"  # Small but capable LLM (2.7B)
    
    # Architecture configs
    num_clusters: int = 256
    em_iterations: int = 2
    projection_hidden: int = 512
    
    # LoRA configs
    llm_lora_rank: int = 128
    llm_lora_alpha: int = 256
    vit_lora_rank: int = 32
    vit_lora_alpha: int = 64
    
    # Training configs
    pretrain_lr: float = 2e-3
    finetune_lora_lr: float = 2e-4
    finetune_other_lr: float = 2e-5
    pretrain_batch_size: int = 32  # Smaller for limited resources
    finetune_batch_size: int = 16
    pretrain_epochs: int = 1
    finetune_epochs: int = 1
    
    # Self-questioning configs
    sq_probability: float = 0.5  # δ threshold
    max_length: int = 512
    
    # Special tokens
    user_token: str = "[usr]"
    assistant_token: str = "[aswr]"
    vuser_token: str = "[vusr]"  # New token for self-questioning
    delimiter_token: str = "<o_d>"

class PrototypeExtractor(nn.Module):
    """
    Prototype Extractor for enhanced visual representations
    Uses EM clustering to extract semantic prototypes
    """
    def __init__(self,embed_dim:int,num_clusters:int=256,em_iterations:int=2):
        super().__init__()
        self.num_clusters = num_clusters
        self.em_iterations = em_iterations

        #Randomly initialize cluster centers.
        self.register_buffer('cluster_centers',torch.randn(num_clusters,embed_dim))

        #Query,Key,Value projections for EM Clustering
        self.query_proj = nn.Linear(embed_dim,embed_dim)
        self.key_proj = nn.Linear(embed_dim,embed_dim)
        self.value_proj = nn.Linear(embed_dim,embed_dim)

        #Projection to map cluster info back to embeddings
        self.cluster_to_embed = nn.Linear(embed_dim,embed_dim)

        #Layer Norms
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_k = nn.LayerNorm(embed_dim)
        self.norm_v = nn.LayerNorm(embed_dim)

    def em_clustering(self,z_v:torch.Tensor) -> Tuple[torch.Tensor,torch.Tensor]:
        """
        Perform EM clustering on visual tokens
        Args:
            z_v: Visual tokens [batch, num_tokens, embed_dim]
        Returns:
            cluster_map: Soft assignment [batch, num_clusters, num_tokens]
            updated_centers: Updated cluster centers [num_clusters, embed_dim]
        """

        batch_size,num_tokens,embed_dim = z_v.shape
        #Project Inputs
        q = self.norm_q(self.query_proj(self.cluster_centers))  #[num_clusters,embed_dim]
        k = self.norm_k(self.key_proj(z_v))  # [batch, num_tokens, embed_dim]
        v = self.norm_v(self.value_proj(z_v)) # [batch, num_tokens, embed_dim]

        centers = q.unsqueeze(0).expand(batch_size,-1,-1)  #[batch,num_clusters,embed_dim]

        for _ in range(self.em_iterations):
            #E-step:Compute soft assignments
            #[batch,num_cluseters,embed_dim] @ [batch,embed_dim,num_tokens]
            similarity = torch.bmm(centers,k.transpose(1,2)) #[batch,num_clusters,num_tokens]
            cluster_map = F.softmax(similarity / math.sqrt(embed_dim)) #Soft assignment

            #M-step: Update cluster centers
            #[batch, num_clusters, num_tokens] @ [batch, num_tokens, embed_dim]
            centers = torch.bmm(cluster_map, v)  # [batch, num_clusters, embed_dim]
        
        return cluster_map, centers  
    
    def forward(self, z_v: torch.Tensor) -> torch.Tensor:
        """
        Enhance visual embeddings with prototype information
        Args:
            z_v: Visual tokens [batch, num_tokens, embed_dim]
        Returns:
            enhanced_z_v: Enhanced visual tokens [batch, num_tokens, embed_dim]
        """
        batch_size, num_tokens, embed_dim = z_v.shape
        
        # Perform EM clustering
        cluster_map, centers = self.em_clustering(z_v)  # [batch, num_clusters, num_tokens], [batch, num_clusters, embed_dim]
        
        # Compute cosine similarity for each token with all prototypes
        z_v_norm = F.normalize(z_v, dim=-1)  # [batch, num_tokens, embed_dim]
        centers_norm = F.normalize(centers, dim=-1)  # [batch, num_clusters, embed_dim]
        
        # [batch, num_tokens, embed_dim] @ [batch, embed_dim, num_clusters]
        cosine_sim = torch.bmm(z_v_norm, centers_norm.transpose(1, 2))  # [batch, num_tokens, num_clusters]
        
        # Normalize similarity weights
        weights = F.softmax(cosine_sim, dim=-1)  # [batch, num_tokens, num_clusters]
        
        # Weighted sum of prototypes for each token
        # [batch, num_tokens, num_clusters] @ [batch, num_clusters, embed_dim]
        weighted_prototypes = torch.bmm(weights, centers)  # [batch, num_tokens, embed_dim]
        
        # Map prototype info and add to original embeddings (residual connection)
        prototype_contribution = self.cluster_to_embed(weighted_prototypes)
        enhanced_z_v = z_v + prototype_contribution
        
        return enhanced_z_v

class VisionLanguageProjector(nn.Module):
    """
    Two-layer MLP projector from vision to language space
    """
    def __init__(self,vision_dim:int,llm_dim:int,hidden_dim:int=512):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(vision_dim,hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim,llm_dim)
        )

    def forward(self,x:torch.Tensor) -> torch.Tensor:
        return self.proj(x)
    
class SQLLaVA(nn.Module):
    """
    SQ-LLaVA: Self-Questioning Large Vision-Language Assistant
    """
    def __init__(self, config: SQConfig):
        super().__init__()
        self.config = config
        
        # Load pre-trained models
        print(f"Loading vision encoder: {config.vision_model}")
        self.vision_encoder = CLIPVisionModel.from_pretrained(config.vision_model)
        self.image_processor = CLIPImageProcessor.from_pretrained(config.vision_model)
        
        print(f"Loading LLM: {config.llm_model}")
        self.llm = AutoModelForCausalLM.from_pretrained(
            config.llm_model,
            torch_dtype=torch.float32,
            trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.llm_model,
            trust_remote_code=True
        )
        
        # Add special tokens
        special_tokens = {
            'additional_special_tokens': [
                config.user_token,
                config.assistant_token,
                config.vuser_token,
                config.delimiter_token
            ]
        }
        self.tokenizer.add_special_tokens(special_tokens)
        self.llm.resize_token_embeddings(len(self.tokenizer))
        
        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Get dimensions
        vision_dim = self.vision_encoder.config.hidden_size
        llm_dim = self.llm.config.hidden_size

        #Prototype Extractor
        self.prototype_extractor =  PrototypeExtractor(
            embed_dim=vision_dim,
            num_clusters=config.num_clusters,
            em_iterations=config.em_iterations
        )

        #Vision-to-language projector
        self.projector = VisionLanguageProjector(
            vision_dim=vision_dim,
            llm_dim=llm_dim,
            hidden_dim=config.projection_hidden
        )
        # Store token IDs for special tokens
        self.user_token_id = self.tokenizer.convert_tokens_to_ids(config.user_token)
        self.assistant_token_id = self.tokenizer.convert_tokens_to_ids(config.assistant_token)
        self.vuser_token_id = self.tokenizer.convert_tokens_to_ids(config.vuser_token)
        self.delimiter_token_id = self.tokenizer.convert_tokens_to_ids(config.delimiter_token)

    def add_lora(self):
        """Add LoRA adapters to vision encoder and LLM"""
        print("Adding LoRA adapters...")
        
        # LoRA for LLM
        llm_lora_config = LoraConfig(
            r=self.config.llm_lora_rank,
            lora_alpha=self.config.llm_lora_alpha,
            target_modules=["q_proj", "v_proj"],  # Adjust based on model architecture
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        self.llm = get_peft_model(self.llm, llm_lora_config)
        
        # LoRA for Vision Encoder
        vit_lora_config = LoraConfig(
            r=self.config.vit_lora_rank,
            lora_alpha=self.config.vit_lora_alpha,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none"
        )
        # Apply LoRA to vision encoder (manual application needed)
        # For simplicity, we'll keep vision encoder frozen in stage 1
        # and unfreeze with LoRA in stage 2
        
        print("LoRA adapters added successfully!")

