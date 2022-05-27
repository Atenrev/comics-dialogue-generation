import torch

from typing import Any, Optional
from src.models.modeling_vlt5 import VLT5


class TextClozeImageTextVLT5Model(VLT5):
    def __init__(self, config: Any, device: torch.device):
        model_config = VLT5.create_model_config(config)
        super().__init__(model_config)
        self.m_device = device
        pretrained_w = torch.load(
            './pretrained_weights/vlt5_epoch30.pth',
            map_location=device
        )
        self.load_checkpoint(pretrained_w)

    def forward(self, input_ids: torch.Tensor, vis_feats: torch.Tensor,
                boxes: torch.Tensor, target: Optional[torch.Tensor] = None
                ) -> Any:
        device = self.m_device
        input_ids = input_ids.to(device)
        B = len(input_ids)
        V_L = vis_feats.size(2)
        vis_feats = vis_feats.to(device).view(B, 4*V_L, 2048)
        vis_pos = boxes.to(device).view(B, 4*V_L, 4)

        lm_labels = target.to(device) if target is not None else None

        img_order_ids = [0] * V_L + [1] * V_L + [2] * V_L + [3] * V_L
        img_order_ids = torch.tensor(
            img_order_ids, dtype=torch.long, device=device)
        img_order_ids = img_order_ids.view(1, 4*V_L).expand(B, -1)

        obj_order_ids = torch.arange(V_L, dtype=torch.long, device=device)
        obj_order_ids = obj_order_ids.view(1, 1, V_L).expand(
            B, 4, -1).contiguous().view(B, 4*V_L)

        output = super().forward(
            input_ids=input_ids,
            vis_inputs=(vis_feats, vis_pos, img_order_ids, obj_order_ids),
            labels=lm_labels,
            return_dict=True
        )

        if lm_labels is not None:
            lm_mask = (lm_labels != -100).float()
            B, L = lm_labels.size()

            loss = output['loss']
            output['loss'] = loss

        output["prediction"] = output["logits"]

        return output
