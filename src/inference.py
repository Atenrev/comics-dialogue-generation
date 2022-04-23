import os
import torch
import logging

from tqdm import tqdm
from typing import Any
from torch.utils.data import DataLoader
from transformers.models.beit.modeling_beit import BeitModelOutputWithPooling
from transformers.modeling_outputs import Seq2SeqLMOutput


class InferenceEngine:
    """
    Inference Engine
    """
    def __init__(self, model: torch.nn.Module, device: torch.device) -> None:
        """
        Constructor of the InferenceEngine.
        """
        self.model = model
        self.device = device
        self.output_dir = "inference_output"
        os.makedirs(self.output_dir, exist_ok=True)

    def run(self, dataloader: DataLoader[Any]) -> None:
        """
        Run the inference engine through the dataloader and
        save the results to the output directory.
        """
        logging.info(f"Running inference on {len(dataloader)} samples.")
        self.model.eval()

        for local_batch in tqdm(dataloader):
            sample_ids = local_batch["sample_id"]
            batch = local_batch["data"]
            batch_len = len(sample_ids)

            with torch.no_grad():
                batch_output = self.model(**batch)

            for i in range(batch_len):
                sample_id = sample_ids[i]

                if type(batch_output) is BeitModelOutputWithPooling:
                    output = batch_output.last_hidden_state[i]
                elif type(batch_output) is Seq2SeqLMOutput:
                    raise NotImplementedError("Seq2SeqLMOutput is not implemented yet.")
                else:
                    logging.warning(f"Unhandled output type: {type(batch_output)}")
                    output = batch_output[i]

                save_path = os.path.join(self.output_dir, f"{sample_id}.pt")
                torch.save(output, save_path)