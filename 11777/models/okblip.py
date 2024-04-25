import torch
import copy
from torch import nn
from typing import Optional
from transformers import BlipForQuestionAnswering, BlipTextModel, BertModel


class OKBLIP(nn.Module):
    """
    Adapted from https://github.com/huggingface/transformers/blob/8c12690cecbb97e187861e386f7a0ac790e4236c/src/transformers/models/blip/modeling_blip.py#L1092-L1303
    """

    def __init__(self, blip: BlipForQuestionAnswering, bert: BertModel):
        super().__init__()
        self.blip = blip
        self.bert = bert
        # self.text_encoder_kn: BlipTextModel = copy.deepcopy(blip.text_encoder)

        # self.mlp = nn.Sequential(
        #     nn.Linear(2 * hidden_size, hidden_size),
        #     nn.ReLU(),
        #     nn.Linear(hidden_size, hidden_size),
        # )
        # Change 1: Add Projection
        #hidden_size = self.blip.config.text_config.hidden_size

        #self.mlp = nn.Sequential(
        #    nn.Linear(hidden_size, 4 * hidden_size),
        #    nn.ReLU(),
        #    nn.Linear(4 * hidden_size, hidden_size),
        #)
        # Change 1 Ends
    def forward(
        self,
        input_ids: torch.LongTensor,
        kn_input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        kn_attention_mask: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
    ):

        if labels is None and decoder_input_ids is None:
            raise ValueError(
                "Either `decoder_input_ids` or `labels` should be passed when calling `forward` with"
                " `BlipForQuestionAnswering`. if you are training the model make sure that `labels` is passed, if you"
                " are using the model for inference make sure that `decoder_input_ids` is passed or call `generate`"
            )

        return_dict = (
            return_dict if return_dict is not None else self.blip.config.use_return_dict
        )
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.blip.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.blip.config.output_hidden_states
        )

        vision_outputs = self.blip.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        image_embeds = vision_outputs[0]
        
        # Change 1
        #image_embeds = self.mlp(image_embeds)
        # Change 1 ends


        image_attention_mask = torch.ones(
            image_embeds.size()[:-1], dtype=torch.long
        ).to(image_embeds.device)

        knowledge_embeds = self.bert(
            input_ids=kn_input_ids, attention_mask=kn_attention_mask
        ).last_hidden_state

        img_kn_embeds = torch.cat((image_embeds, knowledge_embeds), dim=1)
        img_kn_attention_mask = torch.cat(
            (image_attention_mask, kn_attention_mask), axis=1
        )

        question_embeds = self.blip.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=img_kn_embeds,
            encoder_attention_mask=img_kn_attention_mask,
            return_dict=return_dict,
        )

        # knowledge_embeds = self.text_encoder_kn(
        #     input_ids=kn_input_ids,
        #     attention_mask=kn_attention_mask,
        #     encoder_hidden_states=image_embeds,
        #     encoder_attention_mask=image_attention_mask,
        #     return_dict=return_dict,
        # )

        if labels is not None and decoder_input_ids is None:
            # labels are already shifted right, see: https://github.com/huggingface/transformers/pull/23153
            decoder_input_ids = labels

        question_embeds = (
            question_embeds[0] if not return_dict else question_embeds.last_hidden_state
        )
        # print(question_embeds.shape) # [batch_size, seq_len, embed_size]
        # knowledge_embeds = (
        #     knowledge_embeds[0]
        #     if not return_dict
        #     else knowledge_embeds.last_hidden_state
        # )
        # print(knowledge_embeds.shape)
        # text_embeds = torch.cat((question_embeds, knowledge_embeds), dim=1)

        answer_output = self.blip.text_decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=question_embeds,
            encoder_attention_mask=attention_mask,
            labels=labels,
            return_dict=return_dict,
            reduction="mean",
        )

        if labels is not None:
            decoder_loss = (
                answer_output.loss.mean() if return_dict else answer_output[0].mean()
            )
        else:
            decoder_loss = None

        return decoder_loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        kn_input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        kn_attention_mask: Optional[torch.LongTensor] = None,
        **generate_kwargs,
    ) -> torch.LongTensor:

        vision_outputs = self.blip.vision_model(pixel_values=pixel_values)

        image_embeds = vision_outputs[0]

        image_attention_mask = torch.ones(
            image_embeds.size()[:-1], dtype=torch.long
        ).to(image_embeds.device)

        if isinstance(input_ids, list):
            input_ids = torch.LongTensor(input_ids)

        knowledge_embeds = self.bert(
            input_ids=kn_input_ids, attention_mask=kn_attention_mask
        ).last_hidden_state

        img_kn_embeds = torch.cat((image_embeds, knowledge_embeds), dim=1)
        img_kn_attention_mask = torch.cat(
            (image_attention_mask, kn_attention_mask), axis=1
        )

        question_outputs = self.blip.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=img_kn_embeds,
            encoder_attention_mask=img_kn_attention_mask,
            return_dict=False,
        )

        # knowledge_outputs = self.text_encoder_kn(
        #     input_ids=kn_input_ids,
        #     attention_mask=kn_attention_mask,
        #     encoder_hidden_states=image_embeds,
        #     encoder_attention_mask=image_attention_mask,
        #     return_dict=False,
        # )

        question_embeds = question_outputs[0]
        # knowledge_embeds = knowledge_outputs[0]

        # text_embeds = self.mlp(torch.cat((question_embeds, knowledge_embeds), dim=-1))

        # text_attention_mask = torch.ones(text_embeds.size()[:-1], dtype=torch.long).to(
        #     text_embeds.device
        # )

        bos_ids = torch.full(
            (question_embeds.size(0), 1),
            fill_value=self.blip.decoder_start_token_id,
            device=question_embeds.device,
        )

        outputs = self.blip.text_decoder.generate(
            input_ids=bos_ids,
            eos_token_id=self.blip.config.text_config.sep_token_id,
            pad_token_id=self.blip.config.text_config.pad_token_id,
            encoder_hidden_states=question_embeds,
            encoder_attention_mask=attention_mask,
            **generate_kwargs,
        )

        return outputs
