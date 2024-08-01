#include <pybind11/pybind11.h>
#include<iostream>
#include<pybind11/pybind11.h>
#include<pybind11/numpy.h>

namespace py = pybind11;
double sum(double a, double b) {
   return a + b;
}

PYBIND11_MODULE(SumFunction, var) {
   var.doc() = "pybind11 example module";
   var.def("sum", &sum, "This function adds two input numbers");
}

/*
1d Array + 1d Array
*/
py::array_t<double> add_arrays_1d(py::array_t<double>& input1, py::array_t<double>& input2) {

    // Get info from input1, input2
    py::buffer_info buf1 = input1.request();
    py::buffer_info buf2 = input2.request();

    if (buf1.ndim !=1 || buf2.ndim !=1)
    {
        throw std::runtime_error("Number of dimensions must be one");
    }

    if (buf1.size !=buf2.size)
    {
        throw std::runtime_error("Input shape must match");
    }

    //Apply resources
    auto result = py::array_t<double>(buf1.size);
    py::buffer_info buf3 = result.request();

    //Obtain numpy.ndarray data pointer
    double* ptr1 = (double*)buf1.ptr;
    double* ptr2 = (double*)buf2.ptr;
    double* ptr3 = (double*)buf3.ptr;

    //Pointer visits numpy.ndarray
    for (int i = 0; i < buf1.shape[0]; i++)
    {
        ptr3[i] = ptr1[i] + ptr2[i];
    }

    return result;

}

/*
2d矩阵相加
*/
py::array_t<double> add_arrays_2d(py::array_t<double>& input1, py::array_t<double>& input2) {

    py::buffer_info buf1 = input1.request();
    py::buffer_info buf2 = input2.request();

    if (buf1.ndim != 2 || buf2.ndim != 2)
    {
        throw std::runtime_error("numpy.ndarray dims must be 2!");
    }
    if ((buf1.shape[0] != buf2.shape[0])|| (buf1.shape[1] != buf2.shape[1]))
    {
        throw std::runtime_error("two array shape must be match!");
    }

    //Apply resources
    auto result = py::array_t<double>(buf1.size);
    //Resize to 2d array
    result.resize({buf1.shape[0],buf1.shape[1]});


    py::buffer_info buf_result = result.request();

    //Pointer reads and writes numpy.ndarray
    double* ptr1 = (double*)buf1.ptr;
    double* ptr2 = (double*)buf2.ptr;
    double* ptr_result = (double*)buf_result.ptr;

    for (int i = 0; i < buf1.shape[0]; i++)
    {
        for (int j = 0; j < buf1.shape[1]; j++)
        {
            auto value1 = ptr1[i*buf1.shape[1] + j];
            auto value2 = ptr2[i*buf2.shape[1] + j];

            ptr_result[i*buf_result.shape[1] + j] = value1 + value2;
        }
    }
    return result;
}

PYBIND11_MODULE(numpy_demo2, m) {
    m.doc() = "Simple demo using numpy!";
    m.def("add_arrays_2d", &add_arrays_2d);
}

PYBIND11_MODULE(numpy_demo1, m) {
    m.doc() = "Simple demo using numpy!";
    m.def("add_arrays_1d", &add_arrays_1d);
}