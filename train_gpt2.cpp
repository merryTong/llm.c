#include <iostream>
#include <vector>
#include <string>
#include <math.h>

using namespace std;

vector<vector<float>> transpose(vector<vector<float>> A){
    std::vector<std::vector<float>> AT;
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
    vector<vector<vector<float>>> forward(vector<vector<vector<float>>> inputs){
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
    vector<vector<vector<float>>> forward(vector<vector<int>> inputs){
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
    vector<vector<vector<float>>> forward(vector<vector<vector<float>>> inputs){
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

};

class SelfAttention{
private:
    int embedding_dim;
    int num_head;
public:
    LinearLayer* attn;
    LinearLayer* proj;
    SelfAttention(int embedding_dim, int num_head){
        LinearLayer attn(embedding_dim, embedding_dim*3);
        LinearLayer proj(embedding_dim, embedding_dim);
        this->attn = &attn;
        this->proj = &proj;
    }
    vector<vector<vector<float>>> forward(vector<vector<vector<float>>> inputs){
        // input is (B, T, C) 
        vector<vector<vector<float>>> qkv = attn->forward(inputs);
        // qkv is (B, T, 3C) holding the query, key, value (Q, K, V) vectors

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