import torch

from typing import Any
from src.models.modeling_vlt5 import VLT5


class TextClozeImageTextVLT5Model(VLT5):
    def __init__(self, config: Any, device: torch.device):
        model_config = VLT5.create_model_config(config)
        super().__init__(model_config)
        self.m_device = device
        pretrained_w = torch.load(
            config.pretrained_weights,
            map_location=device
        )
        self.load_checkpoint(pretrained_w)

    def run(self, *args, **kwargs):
        device = self.m_device
        input_ids = kwargs['input_ids'].to(device)
        B = len(input_ids)
        V_L = kwargs['vis_feats'].size(2)
        vis_feats = kwargs['vis_feats'].to(device).view(B, 4*V_L, 2048)
        vis_pos = kwargs['boxes'].to(device).view(B, 4*V_L, 4)

        lm_labels = kwargs["target_ids"].to(device)

        img_order_ids = [0] * V_L + [1] * V_L + [2] * V_L + [3] * V_L
        img_order_ids = torch.tensor(
            img_order_ids, dtype=torch.long, device=device)
        img_order_ids = img_order_ids.view(1, 4*V_L).expand(B, -1)

        obj_order_ids = torch.arange(V_L, dtype=torch.long, device=device)
        obj_order_ids = obj_order_ids.view(1, 1, V_L).expand(
            B, 4, -1).contiguous().view(B, 4*V_L)

        output = self(
            input_ids=input_ids,
            vis_inputs=(vis_feats, vis_pos, img_order_ids, obj_order_ids),
            labels=lm_labels,
            return_dict=True
        )

        if "target_ids" in kwargs:
            lm_mask = (lm_labels != -100).float()
            B, L = lm_labels.size()

            loss = output['loss']
            loss = loss.view(B, L) * lm_mask
            loss = loss.sum(dim=1) / lm_mask.sum(dim=1).clamp(min=1)  # B
            loss = loss.mean()
            output['loss'] = loss

        logits = output['logits'].detach()[:, 0]
        logits = logits.view(B, self.lm_head.out_features)
        confidence = torch.softmax(logits, dim=1)
        prediction = confidence.argmax(dim=1)
        output["prediction"] = prediction

        return output
