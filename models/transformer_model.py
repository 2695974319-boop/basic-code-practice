import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self,d_model,max_len=5000,dropout=0.1):
        super().__init__()
        self.dropout=nn.Dropout(dropout)
        pe=torch.zeros(max_len,d_model)
        position=torch.arange(0,max_len,dtype=torch.float).unsqueeze(1)
        div_term=torch.exp(torch.arange(0,d_model,2).float()*(-math.log(10000.0)/d_model))
        pe[:,0::2]=torch.sin(position*div_term)
        pe[:,1::2]=torch.cos(position*div_term)
        pe=pe.unsqueeze(0)
        self.register_buffer("pe",pe)
    def forward(self,x):
        seq_len=x.size(1)
        x=x+self.pe[:,:seq_len,:]
        return self.dropout(x)

class TransformerCoupletModel(nn.Module):
    def __init__(self,vocab_size,pad_id,sos_id,eos_id,d_model,nhead,num_encoder_layers,
                 num_decoder_layers,dim_feedforward,dropout,max_len):
        super().__init__()
        self.pad_id=pad_id
        self.sos_id=sos_id
        self.eos_id=eos_id
        self.d_model=d_model
        self.src_embedding=nn.Embedding(vocab_size,d_model,padding_idx=pad_id)
        self.tgt_embedding=nn.Embedding(vocab_size,d_model,padding_idx=pad_id)
        self.positional_encoding=PositionalEncoding(d_model,max_len,dropout)
        self.transformer=nn.Transformer(d_model=d_model,nhead=nhead,
                                        num_encoder_layers=num_encoder_layers,
                                        num_decoder_layers=num_decoder_layers,
                                        dim_feedforward=dim_feedforward,
                                        dropout=dropout,
                                        batch_first=True)
        if hasattr(self.transformer.encoder,"enable_nested_tensor"):
            self.transformer.encoder.enable_nested_tensor=False
            self.transformer.encoder.use_nested_tensor=False
        self.output_layer=nn.Linear(d_model,vocab_size)

    def make_tgt_mask(self,tgt_len,device):
        mask=torch.triu(torch.ones(tgt_len,tgt_len,device=device,dtype=torch.bool),diagonal=1)
        return mask

    def forward(self,src,tgt=None,teacher_forcing_ratio=0.0,max_len=None):
        if tgt is not None:
            tgt_input=tgt[:,:-1]
            return self._forward_with_tgt_input(src,tgt_input),None

        batch_size=src.size(0)
        decode_len=max_len if max_len is not None else src.size(1)
        tgt_input=torch.full((batch_size,1),self.sos_id,dtype=torch.long,device=src.device)
        logits_list=[]
        encoder_cache=self.encode(src)

        for _ in range(decode_len):
            logits=self.decode_from_memory(encoder_cache,tgt_input)
            next_logits=logits[:,-1:,:]
            logits_list.append(next_logits)
            next_token=next_logits.argmax(dim=-1)
            tgt_input=torch.cat([tgt_input,next_token],dim=1)

        return torch.cat(logits_list,dim=1),None

    def encode(self,src):
        src_key_padding_mask=src.eq(self.pad_id)
        src_emb=self.src_embedding(src)*math.sqrt(self.d_model)
        src_emb=self.positional_encoding(src_emb)
        memory=self.transformer.encoder(
            src_emb,
            src_key_padding_mask=src_key_padding_mask,
        )
        return {
            "memory": memory,
            "src_key_padding_mask": src_key_padding_mask,
        }

    def decode_from_memory(self,encoder_cache,tgt_input):
        tgt_key_padding_mask=tgt_input.eq(self.pad_id)
        tgt_mask=self.make_tgt_mask(tgt_input.size(1),tgt_input.device)
        tgt_emb=self.tgt_embedding(tgt_input)*math.sqrt(self.d_model)
        tgt_emb=self.positional_encoding(tgt_emb)
        transformer_output=self.transformer.decoder(
            tgt=tgt_emb,
            memory=encoder_cache["memory"],
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=encoder_cache["src_key_padding_mask"],
        )
        return self.output_layer(transformer_output)

    def _forward_with_tgt_input(self,src,tgt_input):
        encoder_cache=self.encode(src)
        return self.decode_from_memory(encoder_cache,tgt_input)
