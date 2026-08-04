import torch
import torch.nn as nn
import torch.nn.functional as F

from recbole.model.abstract_recommender import SequentialRecommender


class MySASRec(SequentialRecommender):
    # 1 initialize model
    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        # model hyperparameters
        self.hidden_size = config['hidden_size']
        self.inner_size = config['inner_size']
        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        self.dropout_prob = config['dropout_prob']
        self.loss_type = config['loss_type']

        self.item_embedding = nn.Embedding(
            self.n_items, 
            self.hidden_size, 
        padding_idx=0,
        )

        self.position_embedding = nn.Embedding(
            self.max_seq_length, 
            self.hidden_size,
        )

        # self attentinon
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size, 
            nhead=self.n_heads, 
            dropout=self.dropout_prob,
            dim_feedforward = self.inner_size, # feedforward dimension
            activation = "gelu", # gelu activation
            batch_first = True,
            norm_first = True, # layer normalization before
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.n_layers,
        )

        self.layer_norm = nn.LayerNorm(self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_prob
        )

        self.apply(self._init_weights)

    # 2 initialize weights
    def _init_weights(self, module):
        # embedding init
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

            # padding idx weight to zero
            if module.padding_idx is not None: 
                with torch.no_grad():
                    module.weight[module.padding_idx].fill_(0)

        # linear init
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
            if module.bias is not None: # bias init to zero
                nn.init.zeros_(module.bias)

        # layer norm init
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    # 3 forward function
    def forward(self, item_seq, item_seq_len):
        """
        item_seq: [batch_size, max_seq_length]
        item_seq_len: [batch_size]
        """

        batch_size, seq_len = item_seq.size()
        device = item_seq.device

        # position ids E = M + P
        position_ids = torch.arange(seq_len, device=device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_len)   

        item_emb = self.item_embedding(item_seq)
        pos_emb = self.position_embedding(position_ids)

        x = item_emb + pos_emb
        x = self.layer_norm(x)
        x = self.dropout(x)

        # causal attention layers
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1
        )

        padding_mask = item_seq.eq(0)

        x = self.encoder(
            x, 
            mask=causal_mask, 
            src_key_padding_mask=padding_mask
        )

        # get the output of the last position
        seq_output = self.gather_indexes(
            x,
            item_seq_len - 1
        )

        return seq_output
    
    # 4 calculate loss (recbole)
    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        pos_items = interaction[self.POS_ITEM_ID]

        seq_output = self.forward(item_seq, item_seq_len)

        # compute logits last layer 
        logits = torch.matmul(
            seq_output, 
            self.item_embedding.weight.transpose(0, 1)
        )

        # cross entropy
        loss = F.cross_entropy(logits, pos_items)

        return loss
    
    # 5 predict function (recbole)
    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_items = interaction[self.ITEM_ID]

        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding(test_items)

        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)

        return scores
    
    # 6 full sort predict function (recbole)
    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]

        seq_output = self.forward(item_seq, item_seq_len)

        logits = torch.matmul(
            seq_output, 
            self.item_embedding.weight.transpose(0, 1)
        )

        return logits