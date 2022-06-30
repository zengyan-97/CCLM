import torch
from torch import nn
import torch.nn.functional as F

from transformers.models.bert.modeling_bert import BertModel
from models import XVLMBase, build_mlp


class XVLM4XVNLI(XVLMBase):
    def __init__(self, config):
        super().__init__(config, load_vision_params=False, load_text_params=False,
                         use_contrastive_loss=False, use_matching_loss=False, use_mlm_loss=False, use_bbox_loss=False)

        self.cls_head = build_mlp(input_dim=self.text_width, output_dim=config['num_labels'])
        self.init_params = ['cls_head.' + n for n, _ in self.cls_head.named_parameters()]

    def forward(self, image, text_ids, text_atts, targets, train=True):
        image_embeds, image_atts = self.get_vision_embeds(image)
        text_embeds = self.get_cross_embeds(image_embeds, image_atts, text_ids, text_atts=text_atts)
        prediction = self.cls_head(text_embeds[:, 0, :])

        return F.cross_entropy(prediction, targets) if train else prediction
