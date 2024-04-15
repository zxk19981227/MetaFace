from typing import Optional,Tuple,Union
import torch
from transformers import HubertModel,HubertForCTC
from transformers.modeling_outputs import BaseModelOutput
from config import cfg
from utils import linear_interpolation


class Hubert2Vec(HubertModel):
    def __init__(self, config):
        super().__init__(config)



    def forward(
            self,
            input_values,
            input_fps=50,
            output_fps=30,
            attention_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            mask_time_indices: Optional[torch.FloatTensor] = None,
            frame_num=None
    ) -> Union[Tuple, BaseModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        extract_features = self.feature_extractor(input_values)
        extract_features = extract_features.transpose(1, 2)

        if attention_mask is not None:
            # compute reduced attention_mask corresponding to feature vectors
            attention_mask = self._get_feature_vector_attention_mask(extract_features.shape[1], attention_mask)

        if cfg.dataset == "vocaset":
            extract_features = linear_interpolation(extract_features, input_fps, output_fps)
        hidden_states = self.feature_projection(extract_features)

        hidden_states = self._mask_hidden_states(hidden_states, mask_time_indices=mask_time_indices)

        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = encoder_outputs[0]

        if not return_dict:
            return (hidden_states,) + encoder_outputs[1:]

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )