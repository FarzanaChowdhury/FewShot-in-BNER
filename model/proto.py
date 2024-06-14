import sys
sys.path.append('..')
import util
import torch
from torch import autograd, optim, nn
from torch.autograd import Variable
from torch.nn import functional as F

class Proto(util.framework.FewShotNERModel):
    
    def __init__(self, word_encoder, dot=False, ignore_index=-1):
        util.framework.FewShotNERModel.__init__(self, word_encoder, ignore_index=ignore_index)
        self.drop = nn.Dropout()
        self.dot = dot

    def __dist__(self, x, y, dim):
        if self.dot:
            return (x * y).sum(dim)
        else:
            return -(torch.pow(x - y, 2)).sum(dim)

    def __batch_dist__(self, S, Q, q_mask):
        # S [class, embed_dim], Q [num_of_sent, num_of_tokens, embed_dim]
        assert Q.size()[:2] == q_mask.size()
        Q = Q[q_mask==1].view(-1, Q.size(-1)) # [num_of_all_text_tokens, embed_dim]
        return self.__dist__(S.unsqueeze(0), Q.unsqueeze(1), 2)

    def __get_proto__(self, embedding, tag, mask):
        proto = []
        embedding = embedding[mask==1].view(-1, embedding.size(-1))
        tag = torch.cat(tag, 0)
        assert tag.size(0) == embedding.size(0)
        for label in range(torch.max(tag)+1):
            proto.append(torch.mean(embedding[tag==label], 0))
        proto = torch.stack(proto)
        return proto

    def forward(self, support, query):
        '''
        support: Inputs of the support set.
        query: Inputs of the query set.
        N: Num of classes
        K: Num of instances for each class in the support set
        Q: Num of instances in the query set
        '''
        support_emb = self.word_encoder(support['word'], support['mask']) # [num_sent, number_of_tokens, 768]
        query_emb = self.word_encoder(query['word'], query['mask']) # [num_sent, number_of_tokens, 768]
        support_emb = self.drop(support_emb)
        query_emb = self.drop(query_emb)

        # Debugging: Print the shapes of support and query embeddings
        # print(f"Support Embedding Shape: {support_emb.shape}")
        # print(f"Query Embedding Shape: {query_emb.shape}")

        # Prototypical Networks
        logits = []
        current_support_num = 0
        current_query_num = 0
        assert support_emb.size()[:2] == support['mask'].size()
        assert query_emb.size()[:2] == query['mask'].size()

        max_class_num = 0

        for i, sent_support_num in enumerate(support['sentence_num']):
            sent_query_num = query['sentence_num'][i]

            # Debugging: Print the current support and query numbers
            # print(f"Current Support Num: {current_support_num}, Sent Support Num: {sent_support_num}")
            # print(f"Current Query Num: {current_query_num}, Sent Query Num: {sent_query_num}")

            # Calculate prototype for each class
            support_proto = self.__get_proto__(
                support_emb[current_support_num:current_support_num+sent_support_num], 
                support['label'][current_support_num:current_support_num+sent_support_num], 
                support['text_mask'][current_support_num: current_support_num+sent_support_num])

            # Debugging: Print the shape of support_proto
            # print(f"Support Proto Shape: {support_proto.shape}")

            # Calculate distance to each prototype
            batch_dist = self.__batch_dist__(
                support_proto, 
                query_emb[current_query_num:current_query_num+sent_query_num],
                query['text_mask'][current_query_num: current_query_num+sent_query_num]) # [num_of_query_tokens, class_num]

            # Debugging: Print the shape of batch_dist
            # print(f"Batch Dist Shape: {batch_dist.shape}")

            max_class_num = max(max_class_num, batch_dist.size(1))
            logits.append(batch_dist)
            current_query_num += sent_query_num
            current_support_num += sent_support_num

        # Pad logits to have the same size in dimension 1
        for i, logit in enumerate(logits):
            if logit.size(1) < max_class_num:
                padding = torch.zeros(logit.size(0), max_class_num - logit.size(1), device=logit.device)
                logits[i] = torch.cat([logit, padding], dim=1)

        # Debugging: Print shapes of logits before concatenation
        # for i, logit in enumerate(logits):
        #     print(f"Logits[{i}] Shape: {logit.shape}")

        logits = torch.cat(logits, 0)

        # Debugging: Print shape of logits after concatenation
        # print(f"Concatenated Logits Shape: {logits.shape}")

        _, pred = torch.max(logits, 1)
        return logits, pred
