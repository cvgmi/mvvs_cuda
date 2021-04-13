#include <torch/extension.h>

torch::Tensor mvc(torch::Tensor x, 
                  torch::Tensor weights);

torch::Tensor d_mvc(torch::Tensor x, 
                  torch::Tensor weights);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &mvc, "MVC forward");
    m.def("backward", &d_mvc, "MVC backward");
}
