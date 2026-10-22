#include "llama.h"
#include "ggml-backend.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef WB_WEFT_CHECK
#include "weft/ggml.h"
#endif

static uint64_t observe(int step) {
    uint64_t calls = 0;
#ifdef WB_WEFT_CHECK
    for (int type = 0; type < GGML_TYPE_COUNT; ++type) {
        ggml_weft_counters counts{};
        if (ggml_weft_counters_get(static_cast<ggml_type>(type), &counts) &&
            (counts.vec_dot_calls || counts.weft_quantize_calls)) {
            calls += counts.vec_dot_calls + counts.weft_quantize_calls;
            std::cout << "WB_MODEL_HITS {\"step\":" << step << ",\"type\":\"" << ggml_type_name(static_cast<ggml_type>(type))
                      << "\",\"vec_dot\":" << counts.vec_dot_calls << ",\"quantize\":" << counts.weft_quantize_calls << "}\n";
        }
    }
    const auto ime = ggml_weft_ime1_calls();
    calls += ime;
    std::cout << "WB_MODEL_HITS {\"step\":" << step << ",\"type\":\"ime\",\"ime\":" << ime << "}\n";
#ifdef WB_WEFT_IME2_COUNTER
    const auto ime2 = ggml_weft_ime2_calls();
    calls += ime2;
    std::cout << "WB_MODEL_HITS {\"step\":" << step << ",\"type\":\"ime2\",\"ime\":" << ime2 << "}\n";
#endif
#ifdef WB_WEFT_AMX_COUNTER
    const auto amx = ggml_weft_amx_calls();
    calls += amx;
    std::cout << "WB_MODEL_HITS {\"step\":" << step << ",\"type\":\"amx\",\"amx\":" << amx << "}\n";
#endif
#else
    (void)step;
#endif
    return calls;
}

template<class T> void write(std::ofstream &file, const T *data, size_t count) {
    file.write(reinterpret_cast<const char *>(data), count * sizeof(T));
    if (!file) throw std::runtime_error("cannot write validation output");
}

int main(int argc, char **argv) {
    llama_model *model = nullptr;
    llama_context *context = nullptr;
    llama_batch batch{};
    try {
        if (argc != 8) throw std::runtime_error("expected model, output prefix, token reference or -, threads, prompt length, decode steps, offloaded layers");
        const std::string prefix = argv[2], reference = argv[3];
        const int threads = std::stoi(argv[4]), prompt = std::stoi(argv[5]), steps = std::stoi(argv[6]);
        if (threads < 1 || prompt < 1 || steps < 1) throw std::runtime_error("invalid validation dimensions");
        ggml_backend_load_all();
        llama_backend_init();
#ifdef WB_WEFT_CHECK
        ggml_weft_initialize();
#endif
        auto model_params = llama_model_default_params();
        model_params.n_gpu_layers = std::stoi(argv[7]);
        if (model_params.n_gpu_layers < -1) throw std::runtime_error("invalid offloaded layer count");
        model = llama_model_load_from_file(argv[1], model_params);
        if (!model) throw std::runtime_error("cannot load model");
        auto params = llama_context_default_params();
        params.n_ctx = std::max(128, prompt + steps + 1);
        params.n_batch = prompt;
        params.n_ubatch = prompt;
        params.n_threads = params.n_threads_batch = threads;
        context = llama_init_from_model(model, params);
        if (!context) throw std::runtime_error("cannot create context");
        const auto *vocab = llama_model_get_vocab(model);
        const int vocabulary = llama_vocab_n_tokens(vocab);
        std::vector<llama_token> tokens;
        if (reference == "-") {
            std::string text;
            for (int i = 0; i < prompt; ++i) text += "The compiler computes a matrix and returns a numerical result. ";
            tokens.resize(text.size() + 8);
            int count = llama_tokenize(vocab, text.data(), text.size(), tokens.data(), tokens.size(), true, true);
            if (count < prompt) throw std::runtime_error("fixed prompt did not yield enough tokens");
            tokens.resize(prompt);
        } else {
            std::ifstream file(reference, std::ios::binary);
            uint32_t header[2]{};
            file.read(reinterpret_cast<char *>(header), sizeof(header));
            if (header[0] != uint32_t(prompt) || header[1] != uint32_t(prompt + steps))
                throw std::runtime_error("token reference dimensions differ");
            tokens.resize(header[1]);
            file.read(reinterpret_cast<char *>(tokens.data()), tokens.size() * sizeof(llama_token));
            if (!file) throw std::runtime_error("truncated token reference");
        }
        batch = llama_batch_init(prompt, 0, 1);
        auto evaluate = [&](int start, int count) {
            batch.n_tokens = count;
            for (int i = 0; i < count; ++i) {
                batch.token[i] = tokens[start + i];
                batch.pos[i] = start + i;
                batch.n_seq_id[i] = 1;
                batch.seq_id[i][0] = 0;
                batch.logits[i] = (i == count - 1);
            }
            if (llama_decode(context, batch)) throw std::runtime_error("llama_decode failed");
        };
        std::ofstream logits(prefix + ".logits.bin", std::ios::binary);
        const uint32_t shape[] = {uint32_t(steps + 1), uint32_t(vocabulary)};
        write(logits, shape, 2);
        const uint64_t initial_calls = observe(-1);
        uint64_t observed_calls = initial_calls;
        evaluate(0, prompt);
        for (int row = 0; row <= steps; ++row) {
            observed_calls = observe(row);
            const float *values = llama_get_logits_ith(context, -1);
            if (!values) throw std::runtime_error("no output logits");
            for (int i = 0; i < vocabulary; ++i)
                if (!std::isfinite(values[i])) throw std::runtime_error("nonfinite output logits");
            write(logits, values, vocabulary);
            if (row < steps) {
                if (reference == "-") tokens.push_back(std::max_element(values, values + vocabulary) - values);
                evaluate(prompt + row, 1);
            }
        }
        std::ofstream token_file(prefix + ".tokens.bin", std::ios::binary);
        const uint32_t token_shape[] = {uint32_t(prompt), uint32_t(tokens.size())};
        write(token_file, token_shape, 2);
        write(token_file, tokens.data(), tokens.size());
        std::cout << "WB_MODEL_CHECK {\"prompt_tokens\":" << prompt << ",\"decode_steps\":" << steps
                  << ",\"rows\":" << steps + 1 << ",\"vocabulary\":" << vocabulary
                  << ",\"replacement_observed\":" << (observed_calls > initial_calls ? "true" : "false")
                  << ",\"replacement_call_delta\":" << observed_calls - initial_calls << "}\n";
        llama_batch_free(batch);
        llama_free(context);
        llama_model_free(model);
        llama_backend_free();
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "model validation: " << error.what() << '\n';
        if (batch.token) llama_batch_free(batch);
        if (context) llama_free(context);
        if (model) llama_model_free(model);
        llama_backend_free();
        return 1;
    }
}
