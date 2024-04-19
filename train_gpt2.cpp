#include <iostream>
#include <vector>
#include <string>
#include <math.h>

using namespace std;

vector<vector<float>> transpose(vector<vector<float>>& A){
    vector<vector<float>> AT;
    AT.resize(A[0].size());
    for(int i = 0; i < AT.size(); i++){
        AT[i].resize(A.size());
    }
    
    for(int i = 0; i < A[0].size(); i++){
        for(int j = 0; j < A.size(); j++){
            AT[i][j] = A[j][i];
        }
    }
    return AT;
}
vector<vector<float>> addition(vector<vector<float>> A, vector<vector<float>> B){
    vector<vector<float>> C;
    C.resize(A.size());
    for(int i = 0; i < C.size(); i++){
        C[i].resize(A[0].size());
    }
    
    for(int i = 0; i < A.size(); i++){
        for(int j = 0; j < A[0].size(); j++){
            C[i][j] = A[i][j] + B[i][j];
        }
    }
    return C;
}
vector<vector<float>> matmult(vector<vector<float>>& matrix, vector<vector<float>>& weight, vector<float>& bias){
    // matrix is (C,T), weight is (OC, C), bias is (OC)
    // out is (OC,T)
    int w_row = weight.size(), w_col = weight[0].size();
    int m_row = matrix.size(), m_col = matrix[0].size();
    vector<vector<float>> res(m_row, vector<float>(w_col));

    for(int i = 0; i < m_col; i++){ 
        for(int k = 0; k < w_row; k++){ 
            float val = bias[k];
            for(int j = 0; j < w_col; j++){ 
                val += weight[k][j] * matrix[j][i]; 
            } 
            res[k][i] = val;
        } 
    } 
    return res;
}

class LinearLayer{
private:
    int in_dim;
    int out_dim;
public:
    vector<vector<float>> weight;
    vector<float> bias;
    LinearLayer(int in_dim, int out_dim){
        this->in_dim = in_dim;
        this->out_dim = out_dim;
        weight.resize(out_dim, vector<float>(in_dim));
        bias.resize(out_dim);
    }
    vector<vector<vector<float>>>& forward(vector<vector<vector<float>>>& inputs){
        int B = inputs.size();
        int T = inputs[0].size();
        int C = inputs[0][0].size();
        vector<vector<vector<float>>> out;
        for(int b = 0; b < B; b++){
            vector<vector<float>> input_transpose = transpose(inputs[b]);
            vector<vector<float>> res = matmult(input_transpose, weight, bias);
            out.push_back(transpose(res));
        }
        return out;
    }
};

class EmbeddingLayer{
private:
    int vocab_size;
    int embedding_dim;
public:
    vector<vector<float>> weight;
    EmbeddingLayer(int vocab_size, int embedding_dim){
        this->vocab_size = vocab_size;
        this->embedding_dim = embedding_dim;
        weight.resize(vocab_size, vector<float>(embedding_dim));
    }
    vector<vector<vector<float>>>& forward(vector<vector<int>>& inputs){
        int B = inputs.size();
        int T = inputs[0].size();        
        vector<vector<vector<float>>> embedding(B, vector<vector<float>>(T, vector<float>(this->embedding_dim)));
        for(int b = 0; b < inputs.size(); b++){
            for(int t = 0; t < inputs[0].size(); t++){
                embedding[b][t] = weight[inputs[b][t]];
            }
        }
        return embedding;
    }
};

class LayerNorm{
private:
    int feature_size;
    float eps = 1e-5f;
public:
    vector<float> gamma;
    vector<float> beta;
    LayerNorm(int feature_size){
        this->feature_size = feature_size;
        gamma.resize(feature_size);
        beta.resize(feature_size);
    }
    vector<vector<vector<float>>>& forward(vector<vector<vector<float>>>& inputs){
        int B = inputs.size();
        int T = inputs[0].size();
        int C = inputs[0][0].size();
        vector<vector<vector<float>>> out(B, vector<vector<float>>(T, vector<float>(C)));
        for(int b = 0; b < B; b++){
            for(int t = 0; t < T; t++){
                // calculate the mean
                float mean = 0.0f;
                for (int i = 0; i < C; i++) {
                    mean += inputs[b][t][i];
                }
                mean = mean / C;
                // calculate the variance
                float var = 0.0f;
                for (int i = 0; i < C; i++) {
                    float xshift = inputs[b][t][i] - mean;
                    var += xshift * xshift;
                }
                var = var / C;
                // normalized, then scaled and shifted
                for (int i = 0; i < C; i++) {
                    float normalized_x = (inputs[b][t][i] - mean) / sqrtf(var+this->eps);
                    out[b][t][i] = normalized_x * gamma[i] + beta[i];
                }
            }
        }
        return out;
    }

};

class GeLU{
private:
    float scale_factor = sqrtf(2.0f / M_PI);
    float constant = 0.044715f;
public:
    GeLU(){}
    vector<vector<vector<float>>>& forward(vector<vector<vector<float>>>& inputs){
        // inputs is (B,T,4C)
        int B = inputs.size();
        int T = inputs[0].size();
        int C4 = inputs[0][0].size();         
        vector<vector<vector<float>>> outs(B, vector<vector<float>>(T, vector<float>(C4)));
        for(int b = 0; b < B; b++){
            for(int t = 0; t < T; t++){
                for(int c = 0; c < C4; c++){
                    float x = inputs[b][t][c];
                    float cube = constant * x * x * x;
                    outs[b][t][c] = 0.5f * x * (1.0f + tanhf(scale_factor * (x + cube)));
                }
            }
        }
        return outs;
    }
};

class MultiHeadSelfAttention{
private:
    int embedding_dim;
    int num_head;
public:
    LinearLayer* attn;
    LinearLayer* proj;
    MultiHeadSelfAttention(int embedding_dim, int num_head){
        this->embedding_dim = embedding_dim;
        this->num_head = num_head;
        LinearLayer attn(embedding_dim, embedding_dim*3);
        LinearLayer proj(embedding_dim, embedding_dim);
        this->attn = &attn;
        this->proj = &proj;
    }
    vector<vector<vector<float>>>& forward(vector<vector<vector<float>>>& inputs){
        // input is (B, T, C) 
        int B = inputs.size();
        int T = inputs[0].size();
        int C = inputs[0][0].size();   
        int head_size = C / this->num_head;   
        float scale = 1.0 / sqrtf(head_size);  
        vector<vector<vector<float>>> qkv = attn->forward(inputs);
        vector<vector<vector<float>>> out(B, vector<vector<float>>(T, vector<float>(C)));
        // qkv is (B, T, 3C) holding the query, key, value (Q, K, V) vectors
        for(int b = 0; b < B; b++){
            for(int nh = 0; nh < this->num_head; nh++){
                vector<vector<float>> q_head(T), k_head(T) ,v_head(T);
                // split q_head,k_head,v_head
                for(int t = 0; t < T; t++){
                    q_head[t].assign(qkv[b][t].begin()+(nh*head_size), qkv[b][t].begin()+(nh*head_size+head_size));
                    k_head[t].assign(qkv[b][t].begin()+(nh*head_size+C), qkv[b][t].begin()+(nh*head_size+head_size+C));
                    v_head[t].assign(qkv[b][t].begin()+(nh*head_size+2*C), qkv[b][t].begin()+(nh*head_size+head_size+2*C));
                }
                // calculate query dot key
                vector<float> bias(T, 0.0f);
                k_head = transpose(k_head);
                vector<vector<float>> pre_attn = matmult(q_head, k_head, bias);
                float maxval = -10000.0f;
                for(int i = 0; i < T; i++){
                    // lower triangular
                    for(int j = 0; j <= i; j++){ 
                        // calculate scaled pre_attn
                        pre_attn[i][j] *= scale; 
                        // calculate maxval
                        if(pre_attn[i][j] > maxval) maxval = pre_attn[i][j];
                    }
                }
                // calculate the exp and keep track of sum
                // maxval is being calculated and subtracted only for numerical stability
                // normalize to get the softmax
                vector<vector<float>> post_attn(T, vector<float>(T));                
                for(int i = 0; i < T; i++){
                    // lower triangular
                    float expsum = 0.0f;
                    for(int j = 0; j <= i; j++){ 
                        float expv = expf(pre_attn[i][j] - maxval);
                        expsum += expv;
                        post_attn[i][j] = expv;
                    }
                    float expsum_inv = expsum == 0.0f ? 0.0f : 1.0f / expsum;
                    for(int j = 0; j < T; j++){ 
                        if (j <= i){
                            post_attn[i][j] *= expsum_inv;
                        }
                        else{
                            post_attn[i][j] = 0.0f;
                        }
                    }
                }
                bias.assign(T, 0.0f);
                vector<vector<float>> res = matmult(post_attn, v_head, bias);
                for(int t = 0; t < T; t++){
                    out[b][t].insert(out[b][t].begin()+(nh*head_size), res[t].begin(), res[t].end());
                }
            }
        }
        vector<vector<vector<float>>> attn_proj = proj->forward(out);
        return attn_proj;
    }
};

class MLP{
private:
    int n_embed;
    LinearLayer* c_fc;
    LinearLayer* c_proj;
    GeLU* gelu;
public:
    MLP(int n_embed){
        this->n_embed = n_embed;
        LinearLayer c_fc(n_embed, 4*n_embed);
        LinearLayer c_proj(4*n_embed, n_embed);
        this->c_fc = &c_fc;
        this->c_proj = &c_proj;
        GeLU gelu;
        this->gelu = &gelu;
        int abc = 0;
    }
    vector<vector<vector<float>>>& forward(vector<vector<vector<float>>>& inputs){
        vector<vector<vector<float>>> x = c_fc->forward(inputs);
        x = gelu->forward(x);
        x = c_proj->forward(x);

        return x;
    }
};

class Block{
private:
    LayerNorm* ln_1;
    LayerNorm* ln_2;
    MultiHeadSelfAttention* msa;
    MLP* mlp;
public:
    Block(int embedding_dim, int num_head){
        LayerNorm ln_1(embedding_dim);
        LayerNorm ln_2(embedding_dim);
        this->ln_1 = &ln_1;
        this->ln_2 = &ln_2;
        MultiHeadSelfAttention msa(embedding_dim, num_head);
        this->msa = &msa;
        MLP mlp(embedding_dim);
        this->mlp = &mlp;
    }
    vector<vector<vector<float>>> forward(vector<vector<vector<float>>>& inputs){
        vector<vector<vector<float>>> x = msa->forward(ln_1->forward(inputs));
        for(int i = 0; i < inputs.size(); i++){
            x[i] = addition(inputs[i], x[i]);
        }
        vector<vector<vector<float>>> outs = mlp->forward(ln_2->forward(x));
        for(int i = 0; i < inputs.size(); i++){
            outs[i] = addition(x[i], outs[i]);
        }       
        return outs; 
    }
};

class CrossEntropy{
private:
    vector<vector<vector<float>>> probs;
    vector<vector<int>> targets;
public:
    CrossEntropy(){}
    vector<vector<vector<float>>>& softmax(vector<vector<vector<float>>>& logits){
        // input: logits is (B,T,V) of the unnormalized log probabilities
        int B = logits.size();
        int T = logits[0].size();
        int V = logits[0][0].size();   
        vector<vector<vector<float>>> probs(B, vector<vector<float>>(T, vector<float>(V)));
        for(int b = 0; b < B; b++){
            for(int t = 0; t < T; t++){
                // maxval is only calculated and subtracted for numerical stability
                float maxval = -10000.0f; // TODO something better
                for (int i = 0; i < V; i++) {
                    if (logits[b][t][i] > maxval) {
                        maxval = logits[b][t][i];
                    }
                }
                float sum = 0.0f;
                for (int i = 0; i < V; i++) {
                    probs[b][t][i] = expf(logits[b][t][i] - maxval);
                    sum += probs[b][t][i];
                }
                for (int i = 0; i < V; i++) {
                    probs[b][t][i] /= sum;
                }
            }
        }
    }
    vector<vector<vector<float>>> forward(vector<vector<vector<float>>>& logits, vector<vector<int>>& targets){
        // targets is (B,T) of integers giving the correct index in logits
        int B = logits.size();
        int T = logits[0].size();
        this->probs = softmax(logits);
        this->targets = targets;
        vector<vector<float>> losses(B, vector<float>(T));
        for(int b = 0; b < B; b++){
            for(int t = 0; t < T; t++){
                int idx = targets[b][t];
                losses[b][t] = -logf(probs[b][t][idx]);
            }
            
        }
    }
    vector<vector<vector<float>>> backward(vector<vector<float>> dlosses){
        int B = dlosses.size();
        int T = dlosses[0].size();        
        vector<vector<vector<float>>> outs(B, vector<vector<float>>(T, vector<float>(V)));
        for (int b = 0; b < B; b++) {
            for (int t = 0; t < T; t++) {
                float dloss = dlosses[b][t];

            }
        }
    }
};

struct GPT2Config{
    int max_seq_len; // max sequence length, e.g. 1024
    int vocab_size; // vocab size, e.g. 50257
    int num_layers; // number of layers, e.g. 12
    int num_heads; // number of heads in attention, e.g. 12
    int channels; // number of channels, e.g. 768    
};
class GPT2{
public:
    GPT2Config config;
    EmbeddingLayer* tokenizer;
    EmbeddingLayer* position;
    vector<Block*> blocks;
    LayerNorm* ln_f;
    LinearLayer* lm_head;
    GPT2(GPT2Config& config){
        this->config = config;
        EmbeddingLayer tokenizer(config.vocab_size, config.channels);
        EmbeddingLayer position(config.max_seq_len, config.channels);
        this->tokenizer = &tokenizer;
        this->position = &position;
        blocks.resize(config.num_layers);
        for(int l = 0; l < config.num_layers; l++){
            Block block(config.channels, config.num_heads);
            blocks[l] = &block;
        }
        LayerNorm ln_f(config.channels);
        this->ln_f = &ln_f;
        LinearLayer lm_head(config.channels, config.vocab_size);
        lm_head.weight = tokenizer.weight;
        this->lm_head = &lm_head;
    }
    vector<vector<vector<float>>> forward(vector<vector<int>>& inputs){
        vector<vector<vector<float>>> tok_emb = tokenizer->forward(inputs);
        vector<vector<vector<float>>> pos_emb = position->forward(inputs);
        vector<vector<vector<float>>> x(inputs.size());
        for(int i = 0; i < inputs.size(); i++){
            x[i] = addition(tok_emb[i], pos_emb[i]);
        }       
        for(int l = 0; l < config.num_layers; l++){
            x = blocks[l]->forward(x);
        }
        x = ln_f->forward(x);
        x = lm_head->forward(x);
        return x;
    }
};

void load_checkpoint(string checkpoint_path, GPT2& model){

    FILE* model_file = fopen(checkpoint_path.data(), "rb");
    if (model_file == NULL) { printf("Error opening model file\n"); exit(1); }
    int model_header[256];
    fread(model_header, sizeof(int), 256, model_file);
    if (model_header[0] != 20240326) { printf("Bad magic model file"); exit(1); }
    if (model_header[1] != 1) { printf("Bad version in model file"); exit(1); }
    
    // read in hyperparameters
    int maxT, V, L, NH, C;
    model.config.max_seq_len = maxT = model_header[2];
    model.config.vocab_size = V = model_header[3];
    model.config.num_layers = L = model_header[4];
    model.config.num_heads = NH = model_header[5];
    model.config.channels = C = model_header[6];
    cout << "[GPT-2]" << endl;
    cout << "max_seq_len: "<< maxT << endl;
    cout << "vocab_size: "<< V << endl;
    cout << "num_layers: "<< L << endl;
    cout << "num_heads: "<< NH << endl;
    cout << "channels: "<< C << endl;

}

int main(){
    string check_point = "gpt2_124M.bin";
    GPT2 model;
    load_checkpoint(check_point, model);
}