#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <random>
#include <unordered_map>
#include <numeric>
#include <cmath> // for exp, log

#define INNODE 4
#define HIDENODE 6
#define OUTNODE 3 // 输出节点改为3

double rate = 0.001;        // 学习率
double threshold = 0.01;   // 最大允许误差
size_t maxime = 90000;    // 最大训练轮数

struct Sample {
    std::vector<double> in, out;
};

struct Node {
    double value{}, bias{}, bias_delta{};
    std::vector<double> weight, weight_delta;
};

namespace utils {
    inline double sigmoid(double x) {
        return 1.0 / (1.0 + std::exp(-x));
    }

    inline double sigmoid_derivative(double x) {
        double sig = sigmoid(x);
        return sig * (1 - sig);
    }

    std::vector<double> softmax(const std::vector<double>& z) {
        std::vector<double> exp_z(z.size());
        double sum_exp = 0.0;
        for (auto val : z) {
            sum_exp += std::exp(val);
        }
        for (size_t i = 0; i < z.size(); ++i) {
            exp_z[i] = std::exp(z[i]) / sum_exp;
        }
        return exp_z;
    }

    double cross_entropy_loss(const std::vector<double>& y_true, const std::vector<double>& y_pred) {
        double loss = 0.0;
        for (size_t i = 0; i < y_true.size(); ++i) {
            loss -= y_true[i] * std::log(y_pred[i] + 1e-10); // 避免log(0)
        }
        return loss;
    }

    int label_to_one_hot(const std::string& label) {
        if (label == "Iris-setosa")
            return 0;
        else if (label == "Iris-versicolor")
            return 1;
        else
            return 2;
    }

    std::vector<Sample> get_file_data(std::string filename) {
        std::vector<Sample> samples;
        std::ifstream file(filename);
        if (file.is_open()) {
            std::string line;
            while (std::getline(file, line)) {
                std::stringstream ss(line);
                std::string value;

                Sample sample;
                for (int i = 0; i < 4; ++i) {
                    if (!std::getline(ss, value, ',')) {
                        throw std::runtime_error("Malformed line: " + line);
                    }
                    sample.in.push_back(std::stod(value));
                }
                if (!std::getline(ss, value)) {
                    throw std::runtime_error("Missing label in line: " + line);
                }

                int label = label_to_one_hot(value);
                sample.out.resize(OUTNODE, 0.0);
                sample.out[label] = 1.0; // 转为 one-hot 编码

                samples.push_back(sample);
            }
        }
        else {
            std::cout << "Failed to open file: " << filename << std::endl;
        }
        file.close();
        return samples;
    }

    std::pair<std::vector<Sample>, std::vector<Sample>>
        split_dataset(const std::vector<Sample>& samples, double train_ratio, unsigned int seed) {
        std::vector<int> indices(samples.size());
        std::iota(indices.begin(), indices.end(), 0); // [0, 1, ..., N-1]

        // 打乱索引，使用固定种子
        std::mt19937 g(seed);
        std::shuffle(indices.begin(), indices.end(), g);

        size_t train_size = static_cast<size_t>(samples.size() * train_ratio);

        std::vector<Sample> training_set;
        std::vector<Sample> testing_set;

        for (size_t i = 0; i < train_size; ++i) {
            training_set.push_back(samples[indices[i]]);
        }
        for (size_t i = train_size; i < samples.size(); ++i) {
            testing_set.push_back(samples[indices[i]]);
        }

        return { training_set, testing_set };
    }

    void normalize(std::vector<Sample>& samples) {
        size_t feature_count = samples[0].in.size();
        std::vector<double> mean(feature_count, 0.0);
        std::vector<double> stddev(feature_count, 0.0);

        // 计算均值
        for (const auto& sample : samples) {
            for (size_t i = 0; i < feature_count; ++i) {
                mean[i] += sample.in[i];
            }
        }
        for (auto& val : mean) val /= samples.size();

        // 计算标准差
        for (const auto& sample : samples) {
            for (size_t i = 0; i < feature_count; ++i) {
                stddev[i] += std::pow(sample.in[i] - mean[i], 2);
            }
        }
        for (auto& val : stddev) val = std::sqrt(val / samples.size());

        // 标准化数据
        for (auto& sample : samples) {
            for (size_t i = 0; i < feature_count; ++i) {
                sample.in[i] = (sample.in[i] - mean[i]) / stddev[i];
            }
        }
    }
}

// 网络部分
Node* input_layer[INNODE], * hide_layer[HIDENODE], * out_layer[OUTNODE];

void init(int seed) {
    std::mt19937 rd(seed);
    std::uniform_real_distribution<double> distribution(-1.0, 1.0);
    for (int i = 0; i < INNODE; i++) {
        input_layer[i] = new Node();
        for (int j = 0; j < HIDENODE; j++) {
            input_layer[i]->weight.push_back(distribution(rd));
            input_layer[i]->weight_delta.push_back(0.0);
        }
    }
    for (int i = 0; i < HIDENODE; i++) {
        hide_layer[i] = new Node();
        hide_layer[i]->bias = distribution(rd);
        for (int j = 0; j < OUTNODE; j++) {
            hide_layer[i]->weight.push_back(distribution(rd));
            hide_layer[i]->weight_delta.push_back(0.0);
        }
    }
    for (int i = 0; i < OUTNODE; i++) {
        out_layer[i] = new Node();
        out_layer[i]->bias = distribution(rd);
    }
}

void reset_delta() {
    for (int i = 0; i < INNODE; i++) {
        input_layer[i]->weight_delta.assign(input_layer[i]->weight_delta.size(), 0.0);
    }
    for (int i = 0; i < HIDENODE; i++) {
        hide_layer[i]->bias_delta = 0.0;
        hide_layer[i]->weight_delta.assign(hide_layer[i]->weight_delta.size(), 0.0);
    }
    for (int i = 0; i < OUTNODE; i++) {
        out_layer[i]->bias_delta = 0.0;
    }
}

void train(std::vector<Sample>& training_set) {
    for (size_t times = 0; times < maxime; times++) {
        reset_delta();
        double total_loss = 0.0;

        for (const auto& sample : training_set) {
            // Forward
            for (int i = 0; i < INNODE; i++) {
                input_layer[i]->value = sample.in[i];
            }

            for (int j = 0; j < HIDENODE; j++) {
                double sum = 0.0;
                for (int i = 0; i < INNODE; i++) {
                    sum += input_layer[i]->value * input_layer[i]->weight[j];
                }
                sum -= hide_layer[j]->bias;
                hide_layer[j]->value = utils::sigmoid(sum);
            }

            std::vector<double> output_values(OUTNODE, 0.0);
            for (int j = 0; j < OUTNODE; j++) {
                double sum = 0.0;
                for (int i = 0; i < HIDENODE; i++) {
                    sum += hide_layer[i]->value * hide_layer[i]->weight[j];
                }
                sum -= out_layer[j]->bias;
                output_values[j] = sum; // 未应用 softmax
            }
            output_values = utils::softmax(output_values);

            // Loss
            total_loss += utils::cross_entropy_loss(sample.out, output_values);

            // Backward
            std::vector<double> out_deltas(OUTNODE, 0.0);
            for (int j = 0; j < OUTNODE; j++) {
                out_deltas[j] = output_values[j] - sample.out[j];
                out_layer[j]->bias_delta += out_deltas[j];
            }

            std::vector<double> hide_deltas(HIDENODE, 0.0);
            for (int i = 0; i < HIDENODE; i++) {
                double sum = 0.0;
                for (int j = 0; j < OUTNODE; j++) {
                    sum += out_deltas[j] * hide_layer[i]->weight[j];
                }
                hide_deltas[i] = sum * hide_layer[i]->value * (1.0 - hide_layer[i]->value);
                hide_layer[i]->bias_delta += hide_deltas[i];
            }

            for (int i = 0; i < INNODE; i++) {
                for (int j = 0; j < HIDENODE; j++) {
                    input_layer[i]->weight_delta[j] += hide_deltas[j] * input_layer[i]->value;
                }
            }
            for (int i = 0; i < HIDENODE; i++) {
                for (int j = 0; j < OUTNODE; j++) {
                    hide_layer[i]->weight_delta[j] += out_deltas[j] * hide_layer[i]->value;
                }
            }
        }

        // 更新权重和偏置
        for (int i = 0; i < INNODE; i++) {
            for (int j = 0; j < HIDENODE; j++) {
                input_layer[i]->weight[j] -= rate * input_layer[i]->weight_delta[j] / training_set.size();
            }
        }
        for (int i = 0; i < HIDENODE; i++) {
            hide_layer[i]->bias -= rate * hide_layer[i]->bias_delta / training_set.size();
            for (int j = 0; j < OUTNODE; j++) {
                hide_layer[i]->weight[j] -= rate * hide_layer[i]->weight_delta[j] / training_set.size();
            }
        }
        for (int i = 0; i < OUTNODE; i++) {
            out_layer[i]->bias -= rate * out_layer[i]->bias_delta / training_set.size();
        }

        if (total_loss / training_set.size() < threshold) {
            std::cout << "Training completed in " << times + 1 << " iterations with loss: " << total_loss / training_set.size() << std::endl;
            break;
        }
        if (times % 1000 == 0) {
            std::cout << "Iteration: " << times << ", Loss: " << total_loss / training_set.size() << std::endl;
        }
    }
}

void validate(const std::vector<Sample>& testing_set) {
    int correct_predictions = 0;

    for (const auto& sample : testing_set) {
        // Forward
        for (int i = 0; i < INNODE; i++) {
            input_layer[i]->value = sample.in[i];
        }

        for (int j = 0; j < HIDENODE; j++) {
            double sum = 0.0;
            for (int i = 0; i < INNODE; i++) {
                sum += input_layer[i]->value * input_layer[i]->weight[j];
            }
            sum -= hide_layer[j]->bias;
            hide_layer[j]->value = utils::sigmoid(sum);
        }

        std::vector<double> output_values(OUTNODE, 0.0);
        for (int j = 0; j < OUTNODE; j++) {
            double sum = 0.0;
            for (int i = 0; i < HIDENODE; i++) {
                sum += hide_layer[i]->value * hide_layer[i]->weight[j];
            }
            sum -= out_layer[j]->bias;
            output_values[j] = sum;
        }
        output_values = utils::softmax(output_values);

        // 获取预测类别
        int predicted_label = std::distance(output_values.begin(), std::max_element(output_values.begin(), output_values.end()));

        // 获取真实类别
        int true_label = std::distance(sample.out.begin(), std::max_element(sample.out.begin(), sample.out.end()));

        if (predicted_label == true_label) {
            ++correct_predictions;
        }
    }

    double accuracy = static_cast<double>(correct_predictions) / testing_set.size();
    std::cout << "Validation Accuracy: " << accuracy * 100 << "%" << std::endl;
}

int main() {
    std::vector<Sample> samples = utils::get_file_data("Iris.data");
    utils::normalize(samples);

    unsigned int seed = 42;
    auto [training_set, testing_set] = utils::split_dataset(samples, 0.85, seed);

    init(seed);
    train(training_set);
    validate(testing_set);

    return 0;
}
