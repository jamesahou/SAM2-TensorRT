import torch
import torch.nn as nn

from sam2.build_sam import build_sam2_video_predictor
from sam2.modeling.sam2_base import SAM2Base
from sam2.sam2_video_predictor import SAM2VideoPredictor

from typing import List, Optional, Tuple, Union
import os

class ImageEncoder(nn.Module):
    def __init__(self, model: SAM2VideoPredictor):
        super(ImageEncoder, self).__init__()
        self.image_encoder = model.image_encoder
        self.conv0 = model.sam_mask_decoder.conv_s0
        self.conv1 = model.sam_mask_decoder.conv_s1

    @torch.no_grad()
    def forward(self, x):
        backbone_out = self.image_encoder(x)
        # vision_features (1) top onnx::Shape_9495
        # vision_pos_enc (3) 9996, 9829, 9662
        # backbone_fpn (3) 9997, 9998, bottom onnx::Shape_9495
        
        backbone_out["backbone_fpn"][0] = self.conv0(
            backbone_out["backbone_fpn"][0]
        ) # 9997
        backbone_out["backbone_fpn"][1] = self.conv1(
            backbone_out["backbone_fpn"][1]
        ) # 9998

        return backbone_out
    
class Trunk(nn.Module):
    def __init__(self, model):
        super(Trunk, self).__init__()
        self.trunk = model.image_encoder.trunk
    
    @torch.no_grad()
    def forward(self, x):
        return self.trunk(x)

# class MemoryAttention(nn.Module):
#     def __init__(self, model):
#         super(MemoryAttention, self).__init__()
#         self.memory_attention = model.memory_attention

#     @torch.no_grad()
#     def forward(self, current_vision_feats, current_vision_pos_embeds, memory, memory_pos_embed, num_obj_ptr_tokens):
#         return self.memory_attention(
#             curr=current_vision_feats,
#             curr_pos=current_vision_pos_embeds,
#             memory=memory,
#             memory_pos=memory_pos_embed,
#             num_obj_ptr_tokens=num_obj_ptr_tokens,
#         )
        
# class MaskDecoder(nn.Module):
#     def __init__(self, model):
#         super(MaskDecoder, self).__init__()
#         self.mask_decoder = model.sam_mask_decoder
    
#     @torch.no_grad()
#     def forward(
#         self,
#         image_embeddings: torch.Tensor,
#         image_pe: torch.Tensor,
#         sparse_prompt_embeddings: torch.Tensor,
#         dense_prompt_embeddings: torch.Tensor,
#         multimask_output: bool,
#         repeat_image: bool,
#         high_res_features: Optional[List[torch.Tensor]] = None,
#         ):
        
#         return self.mask_decoder(image_embeddings, image_pe, sparse_prompt_embeddings, dense_prompt_embeddings, multimask_output, repeat_image, high_res_features)


def export_model(model: SAM2VideoPredictor, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    dummy_input = torch.zeros([1, 3, 1024, 1024]).to("cuda")

    image_encoder = ImageEncoder(model)
    image_encoder.eval()


    torch.onnx.export(image_encoder, dummy_input, out_dir + "/hiera_l_image_encoder.onnx", export_params=True, opset_version=17, output_names=["vision_features", "vision_pos_enc_0", "vision_pos_enc_1", "vision_pos_enc_2", "backbone_fpn_0", "backbone_fpn_1", "backbone_fpn_2"],verbose=True)
    
    # trunk = Trunk(model)
    # trunk.eval()
    
    # torch.onnx.export(trunk, dummy_input, out_dir + "/hiera_l_trunk.onnx", export_params=True, opset_version=17, output_names=["trunk_out_0", "trunk_out_1", "trunk_out_2", "trunk_out_3"], verbose=True)

    


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sam2_checkpoint = "../checkpoints/sam2_hiera_large.pt"
    model_cfg = "sam2_hiera_l.yaml"

    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

    export_model(predictor, "sam2_onnx")