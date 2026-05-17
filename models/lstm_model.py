import torch
import torch.nn as nn
import random
from models.attention import BahdanauAttention, build_attention

class EncoderLSTM(nn.Module):
    def __init__(self,vocab_size,embed_size,hidden_size,pad_id,dropout):
        super().__init__()
        self.embedding=nn.Embedding(vocab_size,embed_size,padding_idx=pad_id)
        self.dropout=nn.Dropout(dropout)
        self.lstm=nn.LSTM(embed_size,hidden_size,batch_first=True)
    def forward(self,src):
        embedded=self.dropout(self.embedding(src))
        outputs,(hidden,cell)=self.lstm(embedded)
        return outputs,(hidden,cell)

class AttentionDecoderLSTM(nn.Module):
    def __init__(self,vocab_size,embed_size,hidden_size,pad_id,dropout,attention_type="bahdanau"):
        super().__init__()
        self.embedding=nn.Embedding(vocab_size,embed_size,padding_idx=pad_id)
        self.dropout=nn.Dropout(dropout)
        self.attention=build_attention(attention_type,hidden_size)
        self.lstm=nn.LSTM(embed_size+hidden_size,hidden_size,batch_first=True)
        self.out=nn.Linear(hidden_size,vocab_size)
    def forward_step(self,input_token,state,encoder_outputs,src_mask=None):
        hidden,cell=state
        embedded=self.dropout(self.embedding(input_token))
        query=hidden.permute(1,0,2)
        context,attn_weights=self.attention(query,encoder_outputs,src_mask)
        lstm_input=torch.cat([embedded,context],dim=-1)
        output,(hidden,cell)=self.lstm(lstm_input,(hidden,cell))
        logits=self.out(output)
        return logits,(hidden,cell),attn_weights

class Seq2SeqCoupletModel(nn.Module):
    def __init__(self,vocab_size,embed_size,hidden_size,pad_id,sos_id,eos_id,dropout,attention_type="bahdanau"):
        super().__init__()
        self.pad_id=pad_id
        self.sos_id=sos_id
        self.eos_id=eos_id
        self.attention_type=attention_type
        self.encoder=EncoderLSTM(vocab_size,embed_size,hidden_size,pad_id,dropout)
        self.decoder=AttentionDecoderLSTM(vocab_size,embed_size,hidden_size,pad_id,dropout,attention_type)

    def encode(self,src):
        src_mask=src.eq(self.pad_id)
        encoder_outputs,state=self.encoder(src)
        return {
            "encoder_outputs": encoder_outputs,
            "state": state,
            "src_mask": src_mask,
        }

    def decode_step(self,input_token,state,encoder_cache):
        return self.decoder.forward_step(
            input_token,
            state,
            encoder_cache["encoder_outputs"],
            encoder_cache["src_mask"],
        )

    def forward(self,src,tgt=None,teacher_forcing_ratio=0.5,max_len=None):
        batch_size=src.size(0)
        encoder_cache=self.encode(src)
        state=encoder_cache["state"]
        decode_len=tgt.size(1)-1 if tgt is not None else max_len if max_len is not None else src.size(1)
        input_token=torch.full((batch_size,1),self.sos_id,dtype=torch.long,device=src.device)
        logits_list=[]
        attentions=[]
        for t in range(decode_len):
            logits,state,attn_weights=self.decode_step(input_token,state,encoder_cache)
            logits_list.append(logits)
            attentions.append(attn_weights)
            if tgt is not None and random.random()<teacher_forcing_ratio:
                input_token=tgt[:,t+1].unsqueeze(1)
            else:
                input_token=logits.argmax(dim=-1).detach()
        logits=torch.cat(logits_list,dim=1)
        attentions=torch.cat(attentions,dim=1)
        return logits,attentions
