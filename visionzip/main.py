from .utils import CLIP_EncoderLayer_forward, CLIPAttention_forward, apply_info
from .clip_encoder import CLIPVisionTower_VisionZip
from .dino_encoder import DINOVisionTower_VisionZip
from .llava_arch import prepare_inputs_labels_for_multimodal_visionzip, prepare_inputs_labels_for_multimodal_withdino_visionzip, encode_images_visionzip, encode_images_visionzip_clip, encode_images_visionzip_dino, encode_images_visionzip_multi, restore_image_features_sorted
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model

def visionzip(model, dominant=191, contextual=30):

    apply_info(model.model.vision_tower.vision_tower, dominant_num=dominant-1, contextual_num=contextual)


    from transformers.models.clip.modeling_clip import CLIPEncoderLayer, CLIPAttention

    CLIPEncoderLayer.forward = CLIP_EncoderLayer_forward
    CLIPAttention.forward = CLIPAttention_forward

    from llava.model.multimodal_encoder.clip_encoder import CLIPVisionTower
    CLIPVisionTower.forward = CLIPVisionTower_VisionZip.forward

    from llava.model.multimodal_encoder.clip_encoder import CLIPVisionTower

    from llava.model.llava_arch import LlavaMetaForCausalLM
    if hasattr(LlavaMetaForCausalLM, 'prepare_inputs_labels_for_multimodal'):
        LlavaMetaForCausalLM.prepare_inputs_labels_for_multimodal = prepare_inputs_labels_for_multimodal_visionzip
        LlavaMetaForCausalLM.restore_image_features_sorted = restore_image_features_sorted
        LlavaMetaForCausalLM.encode_images_visionzip_multi = encode_images_visionzip_multi
        LlavaMetaForCausalLM.encode_images_visionzip = encode_images_visionzip


    return model

def visionzip_mof(model, dominant=191, contextual=30):

    apply_info(model.model.vision_tower.vision_tower, dominant_num=dominant-1, contextual_num=contextual)
    apply_info(model.model.dino_tower.vision_tower, dominant_num=dominant-1, contextual_num=contextual)


    from transformers.models.clip.modeling_clip import CLIPEncoderLayer, CLIPAttention

    CLIPEncoderLayer.forward = CLIP_EncoderLayer_forward
    CLIPAttention.forward = CLIPAttention_forward
            
    from llava_mof.model.multimodal_encoder.clip_encoder import CLIPVisionTower
    CLIPVisionTower.forward = CLIPVisionTower_VisionZip.forward

    from llava_mof.model.multimodal_encoder.dino_encoder import DINOVisionTower
    DINOVisionTower.forward = DINOVisionTower_VisionZip.forward

    from llava_mof.model.llava_arch import LlavaMetaForCausalLM
    if hasattr(LlavaMetaForCausalLM, 'prepare_inputs_labels_for_multimodal_withdino'):
        LlavaMetaForCausalLM.prepare_inputs_labels_for_multimodal_withdino = prepare_inputs_labels_for_multimodal_withdino_visionzip
        LlavaMetaForCausalLM.restore_image_features_sorted = restore_image_features_sorted
        LlavaMetaForCausalLM.encode_images_visionzip_multi = encode_images_visionzip_multi
        LlavaMetaForCausalLM.encode_images_visionzip_clip = encode_images_visionzip_clip
        LlavaMetaForCausalLM.encode_images_visionzip_dino = encode_images_visionzip_dino

    return model

def __main__():
    model_path = "liuhaotian/llava-v1.5-7b"

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name=get_model_name_from_path(model_path)
    )
    ## VisoinZip retains 54 dominant tokens and 10 contextual tokens
    model = visionzip(model, dominant=54, contextual=10)

