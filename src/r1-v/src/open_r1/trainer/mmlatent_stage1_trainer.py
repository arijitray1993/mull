# pylint: disable=all
# from https://github.com/UMass-Embodied-AGI/Mirage/blob/main/src/trainer.py 

from trl import SFTTrainer, SFTConfig
import torch
import pdb

class CustomTrainerStage1(SFTTrainer):

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute training loss and additionally compute token accuracies
        """
        (ce_loss, outputs) = super().compute_loss(
            model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
        )
        if "image_out_mask" in inputs:
            predict_embeddings = outputs.hidden_states
            image_out_mask = inputs["image_out_mask"]
            shift_image_mask = image_out_mask[:, -(predict_embeddings.shape[1] - 1) :].to(predict_embeddings.device)
            shift_predict_embeddings = predict_embeddings[..., :-1, :][shift_image_mask.to(predict_embeddings.device) != 0].contiguous()

            input_embeddings = outputs.inputs_embeds
            gt_embeddings = input_embeddings[..., 1:, :][shift_image_mask.to(input_embeddings.device) != 0].contiguous()
            # pdb.set_trace()
            sim_loss = torch.nn.functional.cosine_similarity(gt_embeddings, shift_predict_embeddings).mean()
            sim_loss = 1 - sim_loss
            loss = 0.5*ce_loss + 0.5*sim_loss
        else:
            loss = ce_loss

        return (loss, outputs) if return_outputs else loss

class CustomTrainerStage2(SFTTrainer):

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute training loss and additionally compute token accuracies
        """
        (ce_loss, outputs) = super().compute_loss(
            model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
        )

        loss = ce_loss
        return (loss, outputs) if return_outputs else loss

