from pathlib import Path

import torch
import torch.nn as nn

# Modified from the project 'trl' and 'DeepSpeed-Chat'
class RewardModel1(nn.Module):
    def __init__(self,
                 tokenizer,
                 base_model,
                 num_padding_at_beginning = 0,
                 compute_fp32_loss = False,
                 inds = 10):
        super().__init__()
        self.config = base_model.config
        self.num_padding_at_beginning = num_padding_at_beginning
        self.inds = inds
        self.index = 0
        if hasattr(self.config, "word_embed_proj_dim"):
            # `OPT` models use word_embed_proj_dim as final output
            # https://github.com/huggingface/transformers/blob/main/src/transformers/models/opt/modeling_opt.py#L497
            self.v_head = nn.ModuleList([nn.Linear(self.config.word_embed_proj_dim,
                                         1,
                                         bias=False) for _ in range(inds)])
        else:
            # `gpt-neo(x)` models use `hidden_size` attribute names instead of `n_embd``
            self.config.n_embd = self.config.hidden_size if hasattr(
                self.config, "hidden_size") else self.config.n_embd
            self.v_head = nn.ModulesList([nn.Linear(self.config.n_embd, 1, bias=False) for _ in range(inds)])
        self.weight = nn.Parameter(torch.zeros([inds,]), requires_grad = True).to(device)
        self.rwtransformer = base_model
        self.PAD_ID = tokenizer.pad_token_id
        self.compute_fp32_loss = compute_fp32_loss

    def gradient_checkpointing_enable(self):
        self.rwtransformer.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        self.rwtransformer.gradient_checkpointing_disable()

    def forward(self,
                input_ids=None,
                past_key_values=None,
                attention_mask=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                use_cache=False):

        if self.config.model_type in ["llama", "mistral"]:
            kwargs = dict()
        else:
            kwargs = dict(head_mask=head_mask)

        transformer_outputs = self.rwtransformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs)

        hidden_states = transformer_outputs[0]
        bs = hidden_states.size(0)
        rewards = self.v_head[self.index](hidden_states).squeeze(-1) # batch, length

        p_list = []
        for i in range(bs):
            sig = torch.nn.functional.logsigmoid(rewards[i])
            mask = attention_mask[i].float()
            mask_sig = torch.dot(sig, mask) / torch.sum(mask)
            p_list.append(torch.exp(mask_sig))
        p = torch.stack(p_list)

        return {
            "probability": p,
        }


class RewardModel2(nn.Module):
    def __init__(self,
                 tokenizer,
                 base_model,
                 num_padding_at_beginning = 0,
                 compute_fp32_loss = False,
                 inds = 10):
        super().__init__()
        self.config = base_model.config
        self.num_padding_at_beginning = num_padding_at_beginning
        self.inds = inds
        self.index = 0
        if hasattr(self.config, "word_embed_proj_dim"):
            # `OPT` models use word_embed_proj_dim as final output
            # https://github.com/huggingface/transformers/blob/main/src/transformers/models/opt/modeling_opt.py#L497
            self.v_head = nn.ModuleList([nn.Linear(self.config.word_embed_proj_dim,
                                         1,
                                         bias=False) for _ in range(inds)])
        else:
            # `gpt-neo(x)` models use `hidden_size` attribute names instead of `n_embd``
            self.config.n_embd = self.config.hidden_size if hasattr(
                self.config, "hidden_size") else self.config.n_embd
            self.v_head = nn.ModulesList([nn.Linear(self.config.n_embd, 1, bias=False) for _ in range(inds)])
        self.weight = nn.Parameter(torch.zeros([inds,]))
        self.rwtransformer = base_model
        self.PAD_ID = tokenizer.pad_token_id
        self.compute_fp32_loss = compute_fp32_loss
        self.SEP_ID = tokenizer.convert_tokens_to_ids('[SEP]')

    def gradient_checkpointing_enable(self):
        self.rwtransformer.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        self.rwtransformer.gradient_checkpointing_disable()

    def forward(self,
                input_ids=None,
                past_key_values=None,
                attention_mask=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                use_cache=False):

        if self.config.model_type in ["llama", "mistral"]:
            kwargs = dict()
        else:
            kwargs = dict(head_mask=head_mask)

        transformer_outputs = self.rwtransformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs)


        hidden_states = transformer_outputs[0]
        bs = hidden_states.size(0)
        rewards = self.v_head[self.index](hidden_states).squeeze(-1) # batch, length
        SEP_mask = (input_ids == self.SEP_ID).int()

        p_list = []
        for i in range(bs):
            sig = torch.nn.functional.logsigmoid(rewards[i])
            SEP_pos = SEP_mask[i].nonzero()[-1][0]
            mask = attention_mask[i]
            mask[:SEP_pos + 1] = 0
            mask = mask.float()
            mask_sig = torch.dot(sig, mask) / torch.sum(mask)
            p_list.append(torch.exp(mask_sig))
        p = torch.stack(p_list)

        return {
            "probability": p,
        }

class RewardModel(nn.Module):

    def __init__(self,
                 tokenizer,
                 base_model,
                 num_padding_at_beginning=0,
                 compute_fp32_loss=False,
                 inds = 10,
                 device = 'cpu'):
        super().__init__()
        self.config = base_model.config
        self.num_padding_at_beginning = num_padding_at_beginning
        self.inds = inds
        self.index = 0
        self.device = device
        if hasattr(self.config, "word_embed_proj_dim"):
            # `OPT` models use word_embed_proj_dim as final output
            # https://github.com/huggingface/transformers/blob/main/src/transformers/models/opt/modeling_opt.py#L497
            self.v_head = nn.ModuleList([nn.Linear(self.config.word_embed_proj_dim,
                                         1,
                                         bias=False) for _ in range(inds)]).to(device)
        else:
            # `gpt-neo(x)` models use `hidden_size` attribute names instead of `n_embd``
            self.config.n_embd = self.config.hidden_size if hasattr(
                self.config, "hidden_size") else self.config.n_embd
            self.v_head = nn.ModuleList([nn.Linear(self.config.n_embd, 1, bias=False) for _ in range(inds)]).to(device)
        self.weight = nn.Parameter(torch.zeros([inds,]).to(device))
        self.rwtransformer = base_model
        self.PAD_ID = tokenizer.pad_token_id
        self.compute_fp32_loss = compute_fp32_loss

    def gradient_checkpointing_enable(self):
        self.rwtransformer.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        self.rwtransformer.gradient_checkpointing_disable()

    def save_pretrained(self, folder):
        if folder[-1] != '/':
            folder = folder + '/'
        directory = Path(folder)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save({
            'weight': self.weight,
            'v_head': self.v_head.state_dict(),
        }, folder + 'weight_and_vhead.pt')
        for i in range(self.inds):
            self.rwtransformer.set_adapter('adapter_' + str(i))
            self.rwtransformer.save_pretrained(folder + 'adapter_' + str(i) + '.ckpt')
        self.rwtransformer.set_adapter(['adapter_' + str(i) for i in range(self.inds)])
    
    def load_pretrained(self, folder):
        if folder[-1] != '/':
            folder = folder + '/'
        x = torch.load(folder + 'weight_and_vhead.pt')
        self.weight = x['weight'].to(self.device)
        self.v_head.load_state_dict(x['v_head'])
        self.v_head = self.v_head.to(self.device)
        for i in range(self.inds):
            self.rwtransformer.load_adapter(folder + 'adapter_' + str(i) + '.ckpt', 'adapter_' + str(i))
        self.rwtransformer.set_adapter(['adapter_' + str(i) for i in range(self.inds)])    

    def forward(self,
                input_ids=None,
                past_key_values=None,
                attention_mask=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                use_cache=False):
        loss = None

        if self.config.model_type in ["llama", "mistral"]:
            kwargs = dict()
        else:
            kwargs = dict(head_mask=head_mask)

        transformer_outputs = self.rwtransformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs)

        hidden_states = transformer_outputs[0]
        rewards = self.v_head[self.index](hidden_states).squeeze(-1)
        chosen_mean_scores = []
        rejected_mean_scores = []

        # Split the inputs and rewards into two parts, chosen and rejected
        assert len(input_ids.shape) == 2
        bs = input_ids.shape[0] // 2
        seq_len = input_ids.shape[1]

        chosen_ids = input_ids[:bs]  # bs x seq x 1
        rejected_ids = input_ids[bs:]
        chosen_rewards = rewards[:bs]
        rejected_rewards = rewards[bs:]

        # Compute pairwise loss. Only backprop on the different tokens before padding
        loss = 0.
        p_list = []
        chosen_scores_bs = []
        rejected_scores_bs = []
        for i in range(bs):
            chosen_id = chosen_ids[i]
            rejected_id = rejected_ids[i]
            chosen_reward = chosen_rewards[i]
            rejected_reward = rejected_rewards[i]

            c_inds = (chosen_id == self.PAD_ID).nonzero()
            c_ind = c_inds[self.num_padding_at_beginning].item() if len(
                c_inds
            ) > self.num_padding_at_beginning else seq_len  # OPT model pads the first token, so we need to use the second padding token as the end of the sequence
            check_divergence = (chosen_id != rejected_id).nonzero()

            if len(check_divergence) == 0:
                end_ind = rejected_reward.size(-1)
                divergence_ind = end_ind - 1
                r_ind = c_ind
            else:
                # Check if there is any padding otherwise take length of sequence
                r_inds = (rejected_id == self.PAD_ID).nonzero()
                r_ind = r_inds[self.num_padding_at_beginning].item(
                ) if len(r_inds) > self.num_padding_at_beginning else seq_len
                end_ind = max(c_ind, r_ind)
                divergence_ind = check_divergence[0]
            assert divergence_ind > 0
            c_truncated_reward = chosen_reward[divergence_ind:end_ind]
            chosen_scores_bs.append(c_truncated_reward)
            r_truncated_reward = rejected_reward[divergence_ind:end_ind]
            rejected_scores_bs.append(r_truncated_reward)
            chosen_mean_scores.append(
                chosen_reward[c_ind - 1])  #use the end score for reference
            rejected_mean_scores.append(rejected_reward[r_ind - 1])

            loss += -torch.nn.functional.logsigmoid(c_truncated_reward -
                                                    r_truncated_reward).mean()
            p_list.append(torch.exp(torch.nn.functional.logsigmoid(c_truncated_reward -
                                                    r_truncated_reward).mean()))
        p = torch.stack(p_list)

        loss = loss / bs
        chosen_mean_scores = torch.stack(chosen_mean_scores)
        rejected_mean_scores = torch.stack(rejected_mean_scores)
        return {
            "loss": loss,
            "probability": p,
            "chosen_scores": chosen_scores_bs,
            "rejected_scores": rejected_scores_bs,
        }

    def forward_value(self,
                      input_ids=None,
                      attention_mask=None,
                      past_key_values=None,
                      position_ids=None,
                      head_mask=None,
                      inputs_embeds=None,
                      return_value_only=False,
                      prompt_length=0,
                      use_cache=False):

        if self.config.model_type in ["llama", "mistral"]:
            kwargs = dict()
        else:
            kwargs = dict(head_mask=head_mask)

        transformer_outputs = self.rwtransformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs)
        hidden_states = transformer_outputs[0]
        values = self.v_head[self.index](hidden_states).squeeze(-1)
        if return_value_only:
            return values
        else:
            # [0 0 0 0 prompt, answer, 0 0 0 0 ] for step 3, we have padding at the beginning
            # [prompt, answer, 0, 0, 0, 0] this is normal
            #assert prompt_length > 1, "prompt_length must be greater than 1 to help select the end score"
            bs = values.size(0)
            seq_len = input_ids.shape[1]
            chosen_end_scores = [
            ]  # we use this name for consistency with the original forward function
            p_list = []
            for i in range(bs):
                input_id = input_ids[i]
                value = values[i]
                divergence_ind = prompt_length + self.num_padding_at_beginning
                c_inds = (input_id[prompt_length:] == self.PAD_ID).nonzero()
                # here we only use the answer part of the sequence so we do not need to care about the padding at the beginning
                c_ind = c_inds[0].item() + prompt_length if len(
                    c_inds) > 0 else seq_len
                chosen_end_scores.append(value[c_ind - 1])

                p = torch.exp(torch.nn.functional.logsigmoid(value[divergence_ind:c_ind]).mean())
                p_list.append(p)
            p = torch.stack(p_list)
            return {
                "values": values,
                "probability": p,
                "chosen_end_scores": torch.stack(chosen_end_scores),
            }