#include <torch/extension.h>

torch::Tensor mvc_forward_exp(torch::Tensor x, 
                              torch::Tensor m);

torch::Tensor mvc_forward_log(torch::Tensor x, 
                              torch::Tensor weights);

std::vector<torch::Tensor> mvc_backward_exp(torch::Tensor x, 
                                            torch::Tensor m,
                                            torch::Tensor d_exp);

torch::Tensor mvc_backward_log(torch::Tensor x, 
                                            torch::Tensor weights,
                                            torch::Tensor d_log);

torch::Tensor mvc_backward_log_weights(torch::Tensor x, 
                                            torch::Tensor weights,
                                            torch::Tensor d_log);

std::vector<torch::Tensor> mvc_exp_backward(torch::Tensor X, 
                                torch::Tensor d_Exp,
                                int batch, 
                                int c, int mid_c, 
                                int x, int y, int z,
                                int x_m, int y_m, int z_m);


torch::Tensor mvc_log_backward(torch::Tensor X, 
                                torch::Tensor d_Log,
                                int batch, 
                                int c, int mid_c, 
                                int x, int y, int z,
                                int x_m, int y_m, int z_m);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("exp_forward", &mvc_forward_exp, "MVC forward Exp");
    m.def("log_forward", &mvc_forward_log, "MVC forward Log");
    m.def("exp_backward", &mvc_backward_exp, "MVC backward Exp");
    m.def("log_backward", &mvc_backward_log, "MVC backward Log");
    m.def("log_weights_backward", &mvc_backward_log_weights, "MVC backward Log for weights");

    m.def("exp_backward_test", &mvc_exp_backward, "MVC Exponential Backward Test.");
    m.def("log_backward_test", &mvc_log_backward, "MVC Log Backward Test.");
}
