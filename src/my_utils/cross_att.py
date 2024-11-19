import numpy as np
import torch
from torchvision.transforms import Resize, InterpolationMode
from einops import rearrange
import glob
from diffusers.utils import USE_PEFT_BACKEND


# def compute_attn(attn, query, reference_key, reference_value, attention_mask):
#     # 将 reference_key 和 reference_value 转换为多头格式
#     reference_key = reference_key.expand(5, -1, -1)  # 将 reference_key 扩展为 [5, sequence_length, embedding_dim]
#     reference_value = reference_value.expand(5, -1, -1)
#     reference_key = attn.head_to_batch_dim(reference_key)
#     reference_value = attn.head_to_batch_dim(reference_value)
#    # query = attn.head_to_batch_dim(query)

#     # 计算注意力分数
#     attention_probs = attn.get_attention_scores(query, reference_key, attention_mask)
#     # 计算加权特征
#     hidden_states_ref_cross = torch.bmm(attention_probs, reference_value)

#     return hidden_states_ref_cross


def compute_attn(attn, query, key, value, video_length, ref_frame_index, attention_mask):
    #print(ref_frame_index)
    ref_frame_index = torch.tensor(ref_frame_index,device='cuda:0')
    print(key.size(),'1111')
    key_ref_cross = rearrange(key, "(b f) d c -> b f d c", f=video_length)
    print(key_ref_cross.size(),'2222')
    #print(key_ref_cross.size()) #([1, 10, 1024, 320])
    key_ref_cross = key_ref_cross[:, ref_frame_index]
    print(key_ref_cross.size(),'3333')
    key_ref_cross = rearrange(key_ref_cross, "b f d c -> (b f) d c")
    print(key_ref_cross.size(),'ffff')

    value_ref_cross = rearrange(value, "(b f) d c -> b f d c", f=video_length)
    value_ref_cross = value_ref_cross[:, ref_frame_index]
    value_ref_cross = rearrange(value_ref_cross, "b f d c -> (b f) d c")

    key_ref_cross = attn.head_to_batch_dim(key_ref_cross)
    value_ref_cross = attn.head_to_batch_dim(value_ref_cross)
    attention_probs = attn.get_attention_scores(query, key_ref_cross, attention_mask)
    hidden_states_ref_cross = torch.bmm(attention_probs, value_ref_cross) 
    #print('-------------')
    return hidden_states_ref_cross







class CrossViewAttnProcessor:
    def __init__(self, self_attn_coeff, unet_chunk_size=2):
        self.unet_chunk_size = unet_chunk_size
        self.self_attn_coeff = self_attn_coeff

    def __call__(
            self,
            attn,
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None,
            temb=None,
            scale=1.0,):
      #  hidden_chunks = torch.chunk(hidden_states, self.unet_chunk_size, dim=0)
        hidden_chunks = torch.chunk(hidden_states, self.unet_chunk_size, dim=0)

    #    print(hidden_chunks.size(),'----------')
        if encoder_hidden_states is not None:
     #       enc_hidden_chunks = torch.chunk(encoder_hidden_states,self.unet_chunk_size,dim=0)
            enc_hidden_chunks = torch.chunk(encoder_hidden_states, self.unet_chunk_size, dim=0)

        else:
            enc_hidden_chunks = [None]*10

        final_result = []
        for i, (hidden_states,encoder_hidden_states) in enumerate(zip(hidden_chunks,enc_hidden_chunks)):
            residual = hidden_states    # batch size, height * width, dim    
            args = () if USE_PEFT_BACKEND else (scale,)

            if attn.spatial_norm is not None:
                hidden_states = attn.spatial_norm(hidden_states, temb)

            input_ndim = hidden_states.ndim

            if input_ndim == 4:
                batch_size, channel, height, width = hidden_states.shape
                hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2) # batch_size, height * width, dim
            batch_size, sequence_length, _ = ( # Sequence_length is attention vector length
                hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
            )
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

            if attn.group_norm is not None:
                hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
            query = attn.to_q(hidden_states, *args)
        #  print(query.size())
            is_cross_attention = encoder_hidden_states is not None
            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            elif attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

            if i == 0:
                key_first_frame = attn.to_k(encoder_hidden_states, *args)
                value_first_frame = attn.to_v(encoder_hidden_states, *args)

                
            key = attn.to_k(encoder_hidden_states, *args)
            value = attn.to_v(encoder_hidden_states, *args)

            query = attn.head_to_batch_dim(query) 
            # Batch_size, sequence_length, feature_dim -> Batch_size*num_heads,seq_length,head_feature_dim
            # For computational efficiency
            # Sparse Attention
            
            if not is_cross_attention:
                ################## Perform self attention
                key_self = attn.head_to_batch_dim(key)
                value_self = attn.head_to_batch_dim(value)
                attention_probs = attn.get_attention_scores(query, key_self, attention_mask)
                hidden_states_self = torch.bmm(attention_probs, value_self)
                #######################################
                video_length = key.size()[0] #// self.unet_chunk_size
                ref0_frame_index = [0,0,0,1,2] #* video_length
                hidden_states_ref_spatial = compute_attn(attn, query, key, value, video_length, ref0_frame_index, attention_mask)

                if i == 1:
                    #######################################
                    video_length = key.size()[0]# // self.unet_chunk_size
                    ref0_frame_index = [0,1,2,3,4]
                    hidden_states_ref_temporal = compute_attn(attn, query, key_first_frame, value_first_frame, video_length, ref0_frame_index, attention_mask)              

                    hidden_states_ref0 = torch.mean(torch.stack([hidden_states_ref_spatial,hidden_states_ref_temporal]),dim=0)
                else:
                    hidden_states_ref0 = hidden_states_ref_spatial

                if i == 2:
                    #######################################
                    video_length = key.size()[0]# // self.unet_chunk_size
                    ref0_frame_index = [0,1,2,3,4]
                    hidden_states_ref_temporal = compute_attn(attn, query, key_first_frame, value_first_frame, video_length, ref0_frame_index, attention_mask)              

                    hidden_states_ref0 = torch.mean(torch.stack([hidden_states_ref_spatial,hidden_states_ref_temporal]),dim=0)
                else:
                    hidden_states_ref0 = hidden_states_ref_spatial
                # hidden_states_ref0 = hidden_states_ref_spatial

# torch.mean(torch.stack([hidden_states_ref0, hidden_states_ref1, hidden_states_ref2, hidden_states_ref3]), dim=0)


            key = attn.head_to_batch_dim(key)
            value = attn.head_to_batch_dim(value)

            attention_probs = attn.get_attention_scores(query, key, attention_mask)
            hidden_states_ref4 = torch.bmm(attention_probs, value)


            hidden_states = self.self_attn_coeff * hidden_states_self + (1 - self.self_attn_coeff)* hidden_states_ref0 if not is_cross_attention else hidden_states_ref4 

            hidden_states = attn.batch_to_head_dim(hidden_states)

            # linear proj
            hidden_states = attn.to_out[0](hidden_states, *args)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)

            if input_ndim == 4:
                hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
           # print(attn.residual_connection)
            if attn.residual_connection:
                hidden_states = hidden_states + residual

            hidden_states = hidden_states / attn.rescale_output_factor
            #print(attn.rescale_output_factor)
            final_result.append(hidden_states)
        hidden_states = torch.cat(final_result,dim=0)
        # print(hidden_states.size(),'22222')
        return hidden_states

# class CrossViewAttnProcessor:
#     def __init__(self, self_attn_coeff, unet_chunk_size=2):
#         self.unet_chunk_size = unet_chunk_size
#         self.self_attn_coeff = self_attn_coeff

#     def __call__(
#             self,
#             attn,
#             hidden_states,
#             encoder_hidden_states=None,
#             attention_mask=None,
#             temb=None,
#             scale=1.0):

#         residual = hidden_states
        
#         args = () if USE_PEFT_BACKEND else (scale,)

#         if attn.spatial_norm is not None:
#             hidden_states = attn.spatial_norm(hidden_states, temb)

#         input_ndim = hidden_states.ndim

#         if input_ndim == 4:
#             batch_size, channel, height, width = hidden_states.shape
#             hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
#         print(hidden_states.size())
#         # print(channel)
#         batch_size, sequence_length, _ = (
#             hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
#         )
#         attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

#         if attn.group_norm is not None:
#             hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

#         query = attn.to_q(hidden_states, *args)

#         is_cross_attention = encoder_hidden_states is not None
#         if encoder_hidden_states is None:
#             encoder_hidden_states = hidden_states
#         elif attn.norm_cross:
#             encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

#         key = attn.to_k(encoder_hidden_states, *args)
#         value = attn.to_v(encoder_hidden_states, *args)
#         query = attn.head_to_batch_dim(query)

#         # 判断是执行自注意力还是跨注意力
#         if is_cross_attention:

#             # 跨注意力：`key` 和 `value` 使用第 3 张图像的特征
#             reference_key = attn.to_k(encoder_hidden_states[2:3])
#             reference_value = attn.to_v(encoder_hidden_states[2:3])
#             hidden_states_ref_cross = compute_attn(attn, query, reference_key, reference_value, attention_mask)
#         key = attn.head_to_batch_dim(key)
#         value = attn.head_to_batch_dim(value)
#         attention_probs_self = attn.get_attention_scores(query, key, attention_mask)
#         hidden_states_self = torch.bmm(attention_probs_self, value)
#         # 融合自注意力和跨注意力特征
#         if not is_cross_attention:
#             hidden_states = hidden_states_self
#         else:
#             # 融合自注意力和跨注意力
#             hidden_states = self.self_attn_coeff * hidden_states_self + (1 - self.self_attn_coeff) * hidden_states_ref_cross

#         # 恢复维度并输出结果
#         hidden_states = attn.batch_to_head_dim(hidden_states)
#         hidden_states = attn.to_out[0](hidden_states)
#         hidden_states = attn.to_out[1](hidden_states)
#         if input_ndim == 4:

#             hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

#         if attn.residual_connection:
#             hidden_states = hidden_states + residual

#         return hidden_states
