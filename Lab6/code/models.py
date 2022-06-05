import torch
from torch import nn
import torchvision
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
    """
    Encoder
    """
    def __init__(self, encoded_image_size=14):
        super(Encoder, self).__init__()
        self.enc_image_size = encoded_image_size

        # Pretrained ImageNet ResNet-101
        # Remove linear and pool layers
        resnet = torchvision.models.resnet101(pretrained=True)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

        # Resize image to fixed size to allow input images of variable size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))

        self.fine_tune(fine_tune=True)

    def forward(self, images):
        """
        Forward propagation.
        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images
        """
        out = self.resnet(images)  # (batch_size, 2048, image_size/32, image_size/32)
        out = self.adaptive_pool(out)  # (batch_size=32, 2048, encoded_image_size=14, encoded_image_size=14)
        out = out.permute(0, 2, 3, 1)  # (batch_size=32, encoded_image_size=14, encoded_image_size=14, 2048)
        return out

    def fine_tune(self, fine_tune=True):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.
        :param fine_tune: boolean
        """
        for p in self.resnet.parameters():
            p.requires_grad = False
        # If fine-tuning, only fine-tune convolutional blocks 2 through 4
        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune

class DecoderWithRNN(nn.Module):
    def __init__(self, cfg, encoder_dim=14*14*2048):
        """
        :param embed_dim: embedding size
        :param decoder_dim: size of decoder's RNN
        :param vocab_size: size of vocabulary
        :param encoder_dim: feature size of encoded images
        :param dropout: dropout
        """
        super(DecoderWithRNN, self).__init__()
        self.encoder_dim = encoder_dim
        self.decoder_dim = cfg['decoder_dim']
        self.embed_dim = cfg['embed_dim']
        self.vocab_size = cfg['vocab_size']
        self.dropout = cfg['dropout']
        self.device = cfg['device']
        ############################################################################
        # To Do: define some layers for decoder with RNN
        # self.embedding : Embedding layer
        # self.decode_step : decoding LSTMCell, using nn.LSTMCell
        # self.init : linear layer to find initial input of LSTMCell
        # self.bn : Batch Normalization for encoder's output
        # self.fc : linear layer to transform hidden state to scores over vocabulary
        # other layers you may need
        # Your Code Here!
        self.embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embed_dim)
        self.bn = nn.BatchNorm1d(num_features=self.decoder_dim)
        self.init = nn.Linear(in_features=self.encoder_dim,out_features=self.decoder_dim)
        self.decode_step = nn.LSTMCell(input_size=self.decoder_dim, hidden_size=self.decoder_dim, bias=True)
        self.fc = nn.Linear(in_features=self.decoder_dim,out_features=self.vocab_size)
        self.dropout = nn.Dropout(self.dropout)
        ############################################################################

        # initialize some layers with the uniform distribution
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def load_pretrained_embeddings(self, embeddings):
        """
        Loads embedding layer with pre-trained embeddings.
        :param embeddings: pre-trained embeddings
        """
        self.embedding.weight = nn.Parameter(embeddings)

    def fine_tune_embeddings(self, fine_tune=True):
        """
        Allow fine-tuning of embedding layer? (Only makes sense to not-allow if using pre-trained embeddings).
        :param fine_tune: Allow?
        """
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune
    
    def forward(self, encoder_out, encoded_captions, caption_lengths):
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
        :param encoded_captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
        :param caption_lengths: caption lengths, a tensor of dimension (batch_size, 1)
        :return: scores for vocabulary, sorted encoded captions, decode lengths, sort indices
        """

        batch_size = encoder_out.size(0)
        encoder_out = encoder_out.reshape(batch_size, -1)
        vocab_size = self.vocab_size
        
        # Sort input data by decreasing lengths;
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]
        
        # Embedding
        # the order of embeddings are the same as the sorted encoded captions
        embeddings = self.embedding(encoded_captions) # (batch_size=32, max_caption_length=52, embed_dim=512)
        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1
        decode_lengths = (caption_lengths - 1).tolist()
        # Create tensors to hold word predicion scores
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(self.device)
        # Initialize LSTM state
        init_input = self.bn(self.init(encoder_out))
        h, c = self.decode_step(init_input)  # (batch_size_t, decoder_dim)

        ############################################################################
        # To Do: Implement the main decode step for forward pass 
        # Hint: Decode words one by one
        # Teacher forcing is used.
        # At each time-step, generate a new word in the decoder with the previous word embedding
        # Your Code Here!
        for t in range(max(decode_lengths)):  # t denotes the t-th time step or the t-th word in a generating caption
          count = sum([1 if l > t else 0 for l in decode_lengths])
          # we can do the slicing here because embeddings and decode_lengths are both sorted in descending order
          preds, h, c = self.one_step(embeddings[:count,t,:], h[:count], c[:count])
          predictions[:count,t,:] = preds
        ############################################################################
        return predictions, encoded_captions, decode_lengths, sort_ind
    
    def one_step(self, embeddings, h, c):
        ############################################################################
        # To Do: Implement the one time decode step for forward pass 
        # this function can be used for test decode with beam search
        # return predicted scores over vocabs: preds
        # return hidden state and cell state: h, c
        # Your Code Here!
        h, c = self.decode_step(embeddings, (h,c))
        preds = self.fc(self.dropout(h))
        #logits = self.fc(self.dropout_layer(h))
        #probs = self.softmax(logits)
        #preds = torch.argmax(probs,1) # prediction is the index of the word prediced in the vocab
        ############################################################################
        return preds, h, c

class Attention(nn.Module):
    """
    Attention Network.
    """

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """
        :param encoder_dim: feature size of encoded images
        :param decoder_dim: size of decoder's RNN
        :param attention_dim: size of the attention network
        """
        super(Attention, self).__init__()
        #################################################################
        # To Do: you need to define some layers for attention module
        # Hint: Firstly, define linear layers to transform encoded tensor
        # and decoder's output tensor to attention dim; Secondly, define
        # attention linear layer to calculate values to be softmax-ed; 
        # Your Code Here!
        self.f_att1 = nn.Linear(encoder_dim, attention_dim)
        self.f_att2 = nn.Linear(decoder_dim, attention_dim)
        self.f_att3 = nn.Linear(attention_dim,1)
        #################################################################
        
    def forward(self, encoder_out, decoder_hidden):
        """
        Forward pass.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        """
        #################################################################
        # To Do: Implement the forward pass for attention module
        # Hint: follow the equation 
        # "e = f_att(relu(f_att(encoder_out)+f_att(decoder_hidden)))"
        # "alpha = softmax(e)"
        # "z = alpha * encoder_out"
        # Your Code Here!
        att1 = self.f_att1(encoder_out)
        att2 = self.f_att2(decoder_hidden)
        e = self.f_att3(F.relu(att1 + att2.unsqueeze(1))).squeeze(2)
        alpha = F.softmax(e)
        # z - the context vector
        # soft attention is used
        z = (encoder_out * alpha.unsqueeze(2)).sum(dim=1) 
        #################################################################
        return z, alpha

class DecoderWithAttention(nn.Module):
    """
    Decoder.
    """
    def __init__(self, cfg, encoder_dim=2048):
        """
        :param attention_dim: size of attention network
        :param embed_dim: embedding size
        :param decoder_dim: size of decoder's RNN
        :param vocab_size: size of vocabulary
        :param encoder_dim: feature size of encoded images
        :param dropout: dropout
        """
        super(DecoderWithAttention, self).__init__()

        self.encoder_dim = encoder_dim
        self.decoder_dim = cfg['decoder_dim']
        self.attention_dim = cfg['attention_dim']
        self.embed_dim = cfg['embed_dim']
        self.vocab_size = cfg['vocab_size']
        self.dropout = cfg['dropout']
        self.device = cfg['device']

        ############################################################################
        # To Do: define some layers for decoder with attention
        # self.attention : Attention layer
        # self.embedding : Embedding layer
        # self.decode_step : decoding LSTMCell, using nn.LSTMCell
        # self.init_h : linear layer to find initial hidden state of LSTMCell
        # self.init_c : linear layer to find initial cell state of LSTMCell
        # self.beta : linear layer to create a sigmoid-activated gate
        # self.fc : linear layer to transform hidden state to scores over vocabulary
        # other layers you may need
        # Your Code Here!
        self.attention = Attention(encoder_dim, self.decoder_dim, self.attention_dim)
        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim)
        self.dropout = nn.Dropout(p=self.dropout)
        self.decode_step = nn.LSTMCell(self.embed_dim + self.encoder_dim, self.decoder_dim, bias=True)
        self.init_h = nn.Linear(self.encoder_dim, self.decoder_dim)
        self.init_c = nn.Linear(self.encoder_dim, self.decoder_dim)
        self.f_beta = nn.Linear(self.decoder_dim, self.encoder_dim)
        self.fc = nn.Linear(self.decoder_dim, self.vocab_size)
        ############################################################################

        # initialize some layers with the uniform distribution
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def load_pretrained_embeddings(self, embeddings):
        """
        Loads embedding layer with pre-trained embeddings.
        :param embeddings: pre-trained embeddings
        """
        self.embedding.weight = nn.Parameter(embeddings)

    def fine_tune_embeddings(self, fine_tune=True):
        """
        Allow fine-tuning of embedding layer? (Only makes sense to not-allow if using pre-trained embeddings).
        :param fine_tune: Allow?
        """
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune

    def forward(self, encoder_out, encoded_captions, caption_lengths):
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size=32, enc_image_size=14, enc_image_size=14, encoder_dim=2048)
        :param encoded_captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
        :param caption_lengths: caption lengths, a tensor of dimension (batch_size, 1)
        :return: scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
        """

        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size

        # Flatten image
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)

        # Sort input data by decreasing lengths;
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        # Embedding
        embeddings = self.embedding(encoded_captions)  # (batch_size, max_caption_length, embed_dim)

        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1
        decode_lengths = (caption_lengths - 1).tolist()

        # Create tensors to hold word predicion scores and alphas
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(self.device)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(self.device)

        # Initialize LSTM state
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)    # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)

        ############################################################################
        # To Do: Implement the main decode step for forward pass 
        # Hint: Decode words one by one
        # Teacher forcing is used.
        # At each time-step, decode by attention-weighing the encoder's output based 
        # on the decoder's previous hidden state output
        # then generate a new word in the decoder with the previous word and the attention weighted encoding
        # Your Code Here!
        for t in range(max(decode_lengths)):
          count = sum([1 if l > t else 0 for l in decode_lengths])
          preds, alpha, h, c = self.one_step(embeddings[:count, t, :], encoder_out[:count], h[:count], c[:count])
          predictions[:count,t,:] = preds
          alphas[:count,t,:] = alpha
        ############################################################################
        return predictions, encoded_captions, decode_lengths, alphas, sort_ind

    def one_step(self, embeddings, encoder_out, h, c):
        ############################################################################
        # To Do: Implement the one time decode step for forward pass
        # this function can be used for test decode with beam search
        # return predicted scores over vocabs: preds
        # return attention weight: alpha
        # return hidden state and cell state: h, c
        # Your Code Here!
        z, alpha = self.attention(encoder_out, h)
        beta = F.sigmoid(self.f_beta(h))
        z = beta * z
        h, c = self.decode_step(torch.cat([embeddings,z], dim=1),(h,c))
        preds = self.fc(self.dropout(h))
        ############################################################################
        return preds, alpha, h, c


class EncoderWithGlobalFeature(nn.Module):
    """
    Encoder that emits global features
    """
    def __init__(self, cfg, encoded_image_size=7):
        super(EncoderWithGlobalFeature,self).__init__()

        self.enc_image_size = encoded_image_size
        resnet = torchvision.models.resnet101(pretrained = True)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules) 
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7,7))

        self.fine_tune(fine_tune=True)
    
    def forward(self,images):
        """
        The forward propagation function
        input: resized image of shape (batch_size,3,224,224)
        """
        out = self.resnet(images)         # (batch_size,2048,7,7)
        out = self.adaptive_pool(out)
        batch_size = out.size(0)
        features = out.size(1)
        num_pixels = out.size(2) * out.size(3)
        # Get the global features of the image
        global_features = out.view(batch_size, num_pixels, features).mean(1)   # (batch_size, 2048)
        enc_image = out.permute(0, 2, 3, 1)  #  (batch_size,7,7,2048)
        enc_image = enc_image.view(batch_size,num_pixels,features)          # (batch_size,num_pixels,2048)
        return enc_image, global_features
    
    def fine_tune(self, fine_tune=True):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.
        :param fine_tune: boolean
        """
        for p in self.resnet.parameters():
            p.requires_grad = False
        # If fine-tuning, only fine-tune convolutional blocks 2 through 4
        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune

class AdaptiveAttention(nn.Module):
    def __init__(self, decoder_dim, attention_dim):
        super(AdaptiveAttention,self).__init__()
        self.sen_transform = nn.Linear(decoder_dim, decoder_dim)  
        self.sen_att = nn.Linear(decoder_dim, decoder_dim)
        self.h_transform = nn.Linear(decoder_dim, decoder_dim)   
        self.h_att = nn.Linear(decoder_dim, decoder_dim)
        self.v_att = nn.Linear(decoder_dim, attention_dim)
        self.alphas = nn.Linear(attention_dim, 1)
        self.context_hidden = nn.Linear(decoder_dim, decoder_dim)

    def forward(self, spatial_image, decoder_out, s_t):
        num_pixels = spatial_image.size(1)
        visual_attn = self.v_att(spatial_image)           # (batch_size,num_pixels,att_dim)
        sentinel = F.relu(self.sen_transform(s_t))     # (batch_size,hidden_size)
        sentinel_attn = self.sen_att(sentinel)     # (batch_size,att_dim)

        hidden = F.tanh(self.h_transform(decoder_out))    # (batch_size,hidden_size)
        hidden_attn = self.h_att(hidden)               # (batch_size,att_dim)

        hidden_resized = hidden_attn.unsqueeze(1).expand(hidden_attn.size(0), num_pixels + 1, hidden_attn.size(1))

        concat_features = torch.cat([spatial_image, sentinel.unsqueeze(1)], dim = 1)   # (batch_size, num_pixels+1, hidden_size)
        attended_features = torch.cat([visual_attn, sentinel_attn.unsqueeze(1)], dim = 1)     # (batch_size, num_pixels+1, att_dim)

        attention = F.tanh(attended_features + hidden_resized)    # (batch_size, num_pixels+1, att_dim)
        
        alpha = self.alphas(attention).squeeze(2)                   # (batch_size, num_pixels+1)
        att_weights = F.softmax(alpha, dim=1)                              # (batch_size, num_pixels+1)

        context = (concat_features * att_weights.unsqueeze(2)).sum(dim=1)       # (batch_size, hidden_size)     
        beta_value = att_weights[:,-1].unsqueeze(1)                       # (batch_size, 1)

        out_l = F.tanh(self.context_hidden(context + hidden))

        return out_l, att_weights, beta_value


class DecoderWithAdaptiveAttention(nn.Module):
    """
    Decoder.
    """
    def __init__(self, cfg, encoder_dim=2048):
        super(DecoderWithAdaptiveAttention,self).__init__()

        self.encoder_dim = encoder_dim
        self.decoder_dim = cfg['decoder_dim']
        self.attention_dim = cfg['attention_dim']
        self.embed_dim = cfg['embed_dim']
        self.vocab_size = cfg['vocab_size']
        self.dropout = cfg['dropout']
        self.device = cfg['device']

        self.fc = nn.Linear(self.decoder_dim, self.vocab_size)

        self.enc_image_transform = nn.Linear(self.encoder_dim, self.decoder_dim)
        self.global_feature_transform = nn.Linear(self.encoder_dim, self.embed_dim)
        
        self.init_h = nn.Linear(self.encoder_dim, self.decoder_dim)
        self.init_c = nn.Linear(self.encoder_dim, self.decoder_dim)

        self.x_weight = nn.Linear(self.embed_dim*2, self.decoder_dim)
        self.h_weight = nn.Linear(self.decoder_dim, self.decoder_dim)
        
        self.decode_step = nn.LSTMCell(self.embed_dim*2, self.decoder_dim)
        self.adaptive_attention = AdaptiveAttention(self.decoder_dim, self.attention_dim)
        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim)  
        self.dropout = nn.Dropout(p=self.dropout)

        self.fc.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.embedding.weight.data.uniform_(-0.1, 0.1)

    
    def forward(self, encoder_out, global_features, encoded_captions, caption_lengths):
        
        """
        encoder_out: the encoded images from the encoder, of shape (batch_size, num_pixels, 2048)
        global_features: the global image features returned by the Encoder, of shape: (batch_size, 2048)
        encoded_captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
        caption_lengths: caption lengths, a tensor of dimension (batch_size, 1)
        """
        # to transform the encoder_out and global features to the desired dimension
        spatial_image = F.relu(self.enc_image_transform(encoder_out)) # (batch_size, 512)
        global_features = F.relu(self.global_feature_transform(global_features)) #(batch_size, 49, 512)
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size

        # Flatten image
        encoder_out = encoder_out.reshape(batch_size, -1, encoder_dim) # (batch_size, num_pixels, encoder_dim) 
        num_pixels = encoder_out.size(1)

        # sort the data by decreasing lengths
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]  
        encoded_captions = encoded_captions[sort_ind]    
        spatial_image = spatial_image[sort_ind]          
        global_image = global_features[sort_ind]                 

        # Embedding
        embeddings = self.embedding(encoded_captions)     # (batch_size, max_caption_length, embed_dim)
        decode_lengths = (caption_lengths - 1).tolist()

        # create tensors to hold word predicion scores,alphas and betas
        predictions = torch.zeros(batch_size, max(decode_lengths), self.vocab_size).to(self.device)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels+1).to(self.device)
        betas = torch.zeros(batch_size, max(decode_lengths),1).to(self.device) 
        
        # Initialize the LSTM state
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)
        c = self.init_c(mean_encoder_out)

        # Concatenate the embeddings and global image features for input to LSTM 
        global_features = global_features.unsqueeze(1).expand_as(embeddings)
        x = torch.cat((embeddings,global_features), dim = 2)

        for t in range(max(decode_lengths)):
            count = sum([1 if l > t else 0 for l in decode_lengths])
            h, c, alpha, beta, preds = self.one_step(x[:count,t,:],spatial_image[:count], h[:count], c[:count])
            predictions[:count,t,:] = preds
            alphas[:count,t,:] = alpha
            betas[:count,t,:] = beta
            
        return predictions, alphas, betas, encoded_captions, decode_lengths, sort_ind 

    def one_step(self, x, spatial_image, h, c):
        h_t, c_t = self.decode_step(x,(h,c))
        sentinal_gate = F.sigmoid(self.x_weight(x) + self.h_weight(h))
        s_t = sentinal_gate*F.tanh(c_t)
        context, alpha, beta = self.adaptive_attention(spatial_image, h_t, s_t)
        preds = self.fc(self.dropout(context + h_t))
        return h_t, c_t, alpha, beta, preds
