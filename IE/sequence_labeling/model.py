# -*- coding: utf-8 -*-
"""
@author: longshuicui
@date  :   2021/3/25
@function:
"""
import torch
import torch.nn as nn
from torch.nn import BCELoss, CrossEntropyLoss
from transformers import BertPreTrainedModel, BertModel


class TokenClassifierOutput:
    def __init__(self, loss=None, logits=None, hidden_states=None, attentions=None):
        self.loss = loss
        self.logits = logits
        self.hidden_states = hidden_states
        self.attentions = attentions


class BertForTokenClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output) # [batch_size, seq_len, num_labels]

        loss = None
        if labels is not None:
            loss_fct = BCELoss(reduction="none")
            logits=nn.Sigmoid()(logits)
            loss=loss_fct(logits, labels.float())
            attention_mask=attention_mask.view(-1)
            loss=torch.mean(loss.view(-1, self.num_labels), dim=-1)
            loss=torch.sum(loss*attention_mask)/torch.sum(attention_mask)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

