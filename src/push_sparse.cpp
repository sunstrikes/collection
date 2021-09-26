//
// Created by Sun,Minqi on 2021/9/17.
//
#include <Eigen/Dense>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <random>
#include <unordered_map>
#include <vector>
#include "time_recorder.h"

DEFINE_int32(log_level, 4, "");
DEFINE_uint64(times, 1, "");
DEFINE_uint64(total, 10, "");
DEFINE_uint64(batch, 4, "");
struct FeatureValue {
    float scale;
    float embed_w;
    float embed_g2sum;
    std::vector<float> embedx_w; // w + g2sum 16
};
::std::unordered_map<uint64_t, FeatureValue> feature_value_map;
::std::unordered_map<uint64_t, ::std::vector<float>> feature_vector_map;
::std::vector<uint64_t> keys;
::std::vector<uint64_t> push_keys;
::std::vector<float> push_values; //0: scale 1:embed_w 2-9: embedx_w
::std::vector<::std::vector<float>> push_vectors;
const float initial_g2sum =  3.0;
const float learning_rate = 0.075;

#define unlikely(x) __builtin_expect((x), 0)
template<typename T>
void bound_value(T& w) {
    if (unlikely(w < -10.0)) {
        w = (T)-10.0;
    } else if (unlikely(w > 10.0)) {
        w = (T)10.0;
    }
}

void bound_value(int row, int col, float** w) {
    for (auto i = 0u; i < row; ++i) {
        Eigen::Map<Eigen::MatrixXf> mat_w(w[i], 1, col);
        mat_w = mat_w.array().min(10.0).max(-10.0);
    }
}

void push_sparse_inplace() {
    for (size_t i = 0; i < FLAGS_batch; ++i) {
        auto key = keys[i];
        auto& value = feature_value_map[key];
        auto& push_value = feature_vector_map[key];
        value.scale = push_value[0];
        VLOG(3) << "[inplace] before scale: " << value.scale << "; embed_w: " << value.embed_w << "; g2sum:" << value.embed_g2sum;
        {
            //embed_w更新
            float& g2sum = value.embed_g2sum;
            float add_g2sum = 0;
            float reciprocal_scale = 1.0 / push_value[0];
            float scaled_grad = push_value[1] * reciprocal_scale;
            value.embed_w -= learning_rate * scaled_grad * std::sqrt(initial_g2sum / (initial_g2sum + g2sum));
            bound_value(value.embed_w);
            add_g2sum += scaled_grad * scaled_grad;
            g2sum += add_g2sum;
        }
        {
            //embedx_w 更新
            float reciprocal_scale = 1.0 / push_value[0];
            const int _embedding_dim = 8;
            for (int j = 0; j < _embedding_dim; j++) {
                float& g2sum = value.embedx_w[8 + j];
                float scaled_grad = push_value[j + 2] * reciprocal_scale;
                value.embedx_w[j] -= learning_rate * scaled_grad * sqrt(initial_g2sum / (initial_g2sum + g2sum));
                bound_value(value.embedx_w[j]);
                g2sum += scaled_grad * scaled_grad;
            }
        }

        VLOG(3) << "[inplace] after scale: " << value.scale << "; embed_w: " << value.embed_w << "; g2sum:" << value.embed_g2sum;
        Eigen::Map<Eigen::VectorXf> print_embedx(value.embedx_w.data(), value.embedx_w.size());
        VLOG(3) << "[inplace] after embedx: " << print_embedx.transpose();
    }
}
void push_sparse_as_dense() {
    static thread_local ::std::vector<Eigen::MatrixXf> dense_value {
            Eigen::MatrixXf(1, FLAGS_batch),
            Eigen::MatrixXf(1, FLAGS_batch),
            Eigen::MatrixXf(1, FLAGS_batch),
            Eigen::MatrixXf(8, FLAGS_batch),
            Eigen::MatrixXf(8, FLAGS_batch)
    };
    static thread_local ::std::vector<Eigen::MatrixXf> push_value {
            Eigen::MatrixXf(1, FLAGS_batch),
            Eigen::MatrixXf(8, FLAGS_batch)
    };
    static thread_local ::std::vector<float*> sparse_value(FLAGS_batch);
    static thread_local Eigen::MatrixXf init_g2sum_mat = Eigen::MatrixXf::Constant(1, FLAGS_batch, initial_g2sum);
    //static thread_local Eigen::MatrixXf init_g2sum_mat2 = Eigen::MatrixXf::Constant(8, FLAGS_batch, initial_g2sum);
    // load
    size_t row = 8;
    size_t col = FLAGS_batch;
    auto* dense3_data = const_cast<float*>(dense_value[3].data());
    auto* dense4_data = const_cast<float*>(dense_value[4].data());
    auto* push_data = const_cast<float*>(push_value[1].data());
    std::vector<FeatureValue*> value_vec;
    for (size_t i = 0; i < FLAGS_batch; ++i) {
        auto key = keys[i];
        auto& value = feature_value_map[key];
        auto& push_v = feature_vector_map[key];
        value_vec.emplace_back(&value);
        dense_value[0].data()[i] = push_v[0];
        dense_value[1].data()[i] = value.embed_w;
        dense_value[2].data()[i] = value.embed_g2sum;
        memcpy(dense3_data + i * row, &(value.embedx_w[0]), sizeof(float) * 8);
        memcpy(dense4_data + i * row, &(value.embedx_w[8]), sizeof(float) * 8);
        memcpy(push_data + i * row, &(push_v[2]), sizeof(float) * 8);
        //for (int j = 0; j < 8; ++j) {        //embedx_grad
        //    dense3_data[i * row + j] = value.embedx_w[j];
        //    dense4_data[i * row + j] = value.embedx_w[j + 8];
        //    push_value[1].data()[i * row + j] = push_v[j + 1];
        //}
        push_value[1].data()[i] = push_v[1];
        push_value[0].data()[i] = push_v[0]; //embed_grad
    }
    VLOG(3) << "[dense]before scale: \n" << dense_value[0];
    VLOG(3) << "[dense]before embed_w: \n" << dense_value[1];
    VLOG(3) << "embed_g2sum:\n" << dense_value[2];
    VLOG(3) << "embedx_w:\n" << dense_value[3].transpose();
    VLOG(3) << "embedx_g2sum:\n" << dense_value[4].transpose();

    // compute
    {
        //embed_w更新
        auto& g2sum = dense_value[2];
        Eigen::MatrixXf scaled_grad = push_value[0].cwiseProduct(push_value[0].cwiseInverse());
        VLOG(3) << "[dense] scaled_grad:\n " << scaled_grad;
        //dense_value[1] -= learning_rate * scaled_grad * std::sqrt(init_g2sum_mat / (g2sum + init_g2sum_mat));
        Eigen::MatrixXf mid_v = g2sum + init_g2sum_mat;
        VLOG(3) << "[dense] mid_v: g2sum + init_g2sum_mat\n " << scaled_grad;
        mid_v = init_g2sum_mat.cwiseProduct(mid_v.cwiseInverse());
        VLOG(3) << "[dense] mid_v: init_g2sum_mat.cwiseProduct(mid_v.cwiseInverse())\n " << scaled_grad;
        mid_v = mid_v.cwiseSqrt();
        VLOG(3) << "[dense] mid_v: mid_v.cwiseSqrt()" << scaled_grad;
        dense_value[1] -= mid_v.cwiseProduct(learning_rate * scaled_grad);
        VLOG(3) << "[dense] dense_value: mid_v.cwiseProduct(learning_rate * scaled_grad)\n " << scaled_grad;
        dense_value[1] = dense_value[1].array().min(10.0).max(-10.0).matrix();
        g2sum += scaled_grad.cwiseProduct(scaled_grad);
        VLOG(3) << "[dense] g2sum: scaled_grad.cwiseProduct(scaled_grad)\n" << g2sum;
    }
    {
        //embedx_w 更新
        auto& embedx_g2sum = dense_value[4];
        Eigen::Map<Eigen::VectorXf> scale_v(dense_value[0].data(), dense_value[0].size());
        auto scaled_grad = (push_value[1].array().rowwise() * scale_v.array().cwiseInverse().transpose()).matrix();
        Eigen::MatrixXf mid_v = (embedx_g2sum.array() + initial_g2sum).matrix();
        mid_v = (mid_v.array().cwiseInverse() * initial_g2sum).matrix();
        mid_v = mid_v.cwiseSqrt();
        dense_value[3] -= mid_v.cwiseProduct(learning_rate * scaled_grad);
        dense_value[3] = dense_value[3].array().min(10.0).max(-10.0).matrix();
        embedx_g2sum += scaled_grad.cwiseProduct(scaled_grad);
    }
    // store
    for (size_t i = 0; i < FLAGS_batch; ++i) {
        auto &value = *value_vec[i];
        value.scale = dense_value[0].data()[i];
        value.embed_w = dense_value[1].data()[i];
        value.embed_g2sum = dense_value[2].data()[i];
        memcpy(&(value.embedx_w[0]), (dense3_data + i * FLAGS_batch), sizeof(float) * 8);
        memcpy(&(value.embedx_w[8]), (dense4_data + i * FLAGS_batch), sizeof(float) * 8);
        VLOG(3) << "[inplace] after scale: " << value.scale << "; embed_w: " << value.embed_w << "; g2sum:" << value.embed_g2sum;
        Eigen::Map<Eigen::VectorXf> print_embedx(value.embedx_w.data(), value.embedx_w.size());
        VLOG(3) << "[inplace] after embedx: " << print_embedx.transpose();

        //for (int j = 0; j < 8; ++j) {
        //    value.embedx_w[j] = dense3_data[i * FLAGS_batch + j];
        //    value.embedx_w[j + 8] = dense4_data[i * FLAGS_batch + j];
        //}
    }
}
int32_t main(int32_t argc, char** argv) {
    google::InitGoogleLogging(argv[0]);
    ::google::ParseCommandLineFlags(&argc, &argv, true);
    FLAGS_logtostderr = true;
    google::SetLogDestination(google::INFO,"./info.log");
    google::SetLogDestination(google::WARNING,"./info.log.wf");
    google::SetLogDestination(google::ERROR,"./info.log.wf");
    ::std::random_device rd;
    for (size_t i = 0; i < FLAGS_total; i++) {
        uint64_t key = rd();
        FeatureValue fv;
        fv.scale = 2.0;
        fv.embed_w = 1.0;
        fv.embed_g2sum = 1.0;
        fv.embedx_w.resize(16, 1.0);
        keys.emplace_back(key);
        feature_value_map[key] = fv;
        feature_vector_map[key].resize(10, 1.0);
        if (i < FLAGS_batch) {
            //push_values.emplace_back(fv);
        }
    }
    push_vectors.resize(19);
    for (size_t i = 0; i < 19; ++i) {
        push_vectors[i].resize(FLAGS_batch, 1.0);
    }
    {
        size_t pos = 0;
        for (size_t i = 0; i < FLAGS_times; ++i) {
            if (pos + FLAGS_batch >= keys.size()) {
                pos = 0;
            }
            push_keys.assign(&keys[pos], &keys[pos + FLAGS_batch]);
            push_sparse_inplace();
        }
    }
    {
        TimeRecorder time("push in place");
        size_t pos = 0;
        for (size_t i = 0; i < FLAGS_times; ++i) {
            if (pos + FLAGS_batch >= keys.size()) {
                pos = 0;
            }
            push_keys.assign(&keys[pos], &keys[pos + FLAGS_batch]);
            push_sparse_inplace();
        }
    }
    {
        TimeRecorder time("push as dense");
        size_t pos = 0;
        for (size_t i = 0; i < FLAGS_times; ++i) {
            if (pos + FLAGS_batch >= keys.size()) {
                pos = 0;
            }
            push_keys.assign(&keys[pos], &keys[pos + FLAGS_batch]);
            push_sparse_as_dense();
        }
    }
    return 0;
}