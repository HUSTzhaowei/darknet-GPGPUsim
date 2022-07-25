#include "cuda_runtime.h"
#include <bitset>
#include <stdint.h>
#include <cstring>
#include <cmath>
// #include "curand.h"
// #include "cublas_v2.h"
using namespace std;
extern "C" {
#include "activations.h"
#include "cuda.h"
}


__device__ float lhtan_activate_kernel(float x)
{
    if(x < 0) return .001f*x;
    if(x > 1) return .001f*(x-1.f) + 1.f;
    return x;
}
__device__ float lhtan_gradient_kernel(float x)
{
    if(x > 0 && x < 1) return 1;
    return .001;
}

__device__ float hardtan_activate_kernel(float x)
{
    if (x < -1) return -1;
    if (x > 1) return 1;
    return x;
}
__device__ float linear_activate_kernel(float x){return x;}
__device__ float logistic_activate_kernel(float x){return 1.f/(1.f + expf(-x));}
__device__ float loggy_activate_kernel(float x){return 2.f/(1.f + expf(-x)) - 1;}
__device__ float relu_activate_kernel(float x){return x*(x>0);}
__device__ float elu_activate_kernel(float x){return (x >= 0)*x + (x < 0)*(expf(x)-1);}
__device__ float selu_activate_kernel(float x){return (x >= 0)*1.0507f*x + (x < 0)*1.0507f*1.6732f*(expf(x)-1);}
__device__ float relie_activate_kernel(float x){return (x>0) ? x : .01f*x;}
__device__ float ramp_activate_kernel(float x){return x*(x>0)+.1f*x;}
__device__ float leaky_activate_kernel(float x){return (x>0) ? x : .1f*x;}
__device__ float tanh_activate_kernel(float x){return (2.f/(1 + expf(-2*x)) - 1);}
__device__ float plse_activate_kernel(float x)
{
    if(x < -4) return .01f * (x + 4);
    if(x > 4)  return .01f * (x - 4) + 1;
    return .125f*x + .5f;
}
__device__ float stair_activate_kernel(float x)
{
    int n = floorf(x);
    if (n%2 == 0) return floorf(x/2);
    else return (x - n) + floorf(x/2);
}
 

__device__ float hardtan_gradient_kernel(float x)
{
    if (x > -1 && x < 1) return 1;
    return 0;
}
__device__ float linear_gradient_kernel(float x){return 1;}
__device__ float logistic_gradient_kernel(float x){return (1-x)*x;}
__device__ float loggy_gradient_kernel(float x)
{
    float y = (x+1)/2;
    return 2*(1-y)*y;
}
__device__ float relu_gradient_kernel(float x){return (x>0);}
__device__ float elu_gradient_kernel(float x){return (x >= 0) + (x < 0)*(x + 1);}
__device__ float selu_gradient_kernel(float x){return (x >= 0)*1.0507 + (x < 0)*(x + 1.0507*1.6732);}
__device__ float relie_gradient_kernel(float x){return (x>0) ? 1 : .01f;}
__device__ float ramp_gradient_kernel(float x){return (x>0)+.1f;}
__device__ float leaky_gradient_kernel(float x){return (x>0) ? 1 : .1f;}
__device__ float tanh_gradient_kernel(float x){return 1-x*x;}
__device__ float plse_gradient_kernel(float x){return (x < 0 || x > 1) ? .01f : .125f;}
__device__ float stair_gradient_kernel(float x)
{
    if (floorf(x) == x) return 0;
    return 1;
}
// added by zw
// extern "C" void float_to_fix(float a, std::bitset<16> &fix, bool add, bool &enable){
//   std::bitset<32> float_bits;
//   std::bitset<32> get_M(0x7FF800);
//   std::bitset<32> get_E(0x7F800000);
//   std::bitset<16> fix_bits;
//   unsigned int k;
//   memcpy(&k, &a, 4);
//   float_bits = k;
//   std::bitset<16> E;
//   std::bitset<16> M;
//   M = ((float_bits & get_M) >> 11).to_ulong();
// //   cout<<"M is:"<<M<<endl;
//   E = ((float_bits & get_E) >> 23).to_ulong();
//   if((M!=0xfff)&&(add==true)) M = M.to_ulong() + 1;
//   if(a==0.0){
//      fix = 0x00;
//      enable = true;
//   }
//   if((E.to_ulong() <= 131)&&(E.to_ulong() >= 124)){
//     //enable fix-point representation
//     switch(a>=0){
//       case 0:
//         fix_bits[15] = 1;
//         break;
//       case 1:
//         fix_bits[15] = 0;
//         break;
//     }
//     fix_bits |= M;
//     fix_bits |= (E.to_ulong()-124)<<12;
//     fix = fix_bits.to_ulong();
//     enable = true;
//   }
//   else if(E.to_ulong() < 124){
//       fix = 0x00;
//      enable = true;
//   }
//   else enable = false;
// }
// extern "C" void fix_to_float(std::bitset<16> fix_value, float &a){
//   std::bitset<32> M;
//   std::bitset<32> E;
//   std::bitset<32> float_bits;
//   std::bitset<16> fix_bits(fix_value.to_ulong());
// //   float a;
//   if(fix_value==0x00){
//       a = 0.00;
//     //   return a;
//   }
//   unsigned int data;
//   M = (fix_value.to_ulong()) & 0x0FFF;
//   E = ((fix_value.to_ulong()) & 0x7000) >> 12;
//   switch(fix_bits[15]){
//     case 0:
//       float_bits[31] = 0;
//       break;
//     case 1:
//       float_bits[31] = 1;
//       break;
//   }
//   float_bits |= M<<11;
//   E = E.to_ulong()+124;
//   float_bits |= E<<23;
//   data = float_bits.to_ulong();
//   memcpy(&a, &data, 4);
// //   return a;
// }

//end
__device__ float activate_kernel(float x, ACTIVATION a)
{
    switch(a){
        case LINEAR:
            return linear_activate_kernel(x);
        case LOGISTIC:
            return logistic_activate_kernel(x);
        case LOGGY:
            return loggy_activate_kernel(x);
        case RELU:
            return relu_activate_kernel(x);
        case ELU:
            return elu_activate_kernel(x);
        case SELU:
            return selu_activate_kernel(x);
        case RELIE:
            return relie_activate_kernel(x);
        case RAMP:
            return ramp_activate_kernel(x);
        case LEAKY:
            return leaky_activate_kernel(x);
        case TANH:
            return tanh_activate_kernel(x);
        case PLSE:
            return plse_activate_kernel(x);
        case STAIR:
            return stair_activate_kernel(x);
        case HARDTAN:
            return hardtan_activate_kernel(x);
        case LHTAN:
            return lhtan_activate_kernel(x);
    }
    return 0;
}

__device__ float gradient_kernel(float x, ACTIVATION a)
{
    switch(a){
        case LINEAR:
            return linear_gradient_kernel(x);
        case LOGISTIC:
            return logistic_gradient_kernel(x);
        case LOGGY:
            return loggy_gradient_kernel(x);
        case RELU:
            return relu_gradient_kernel(x);
        case ELU:
            return elu_gradient_kernel(x);
        case SELU:
            return selu_gradient_kernel(x);
        case RELIE:
            return relie_gradient_kernel(x);
        case RAMP:
            return ramp_gradient_kernel(x);
        case LEAKY:
            return leaky_gradient_kernel(x);
        case TANH:
            return tanh_gradient_kernel(x);
        case PLSE:
            return plse_gradient_kernel(x);
        case STAIR:
            return stair_gradient_kernel(x);
        case HARDTAN:
            return hardtan_gradient_kernel(x);
        case LHTAN:
            return lhtan_gradient_kernel(x);
    }
    return 0;
}

__global__ void binary_gradient_array_kernel(float *x, float *dy, int n, int s, BINARY_ACTIVATION a, float *dx)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    int i = id % s;
    int b = id / s;
    float x1 = x[b*s + i];
    float x2 = x[b*s + s/2 + i];
    if(id < n) {
        float de = dy[id];
        dx[b*s + i] = x2*de;
        dx[b*s + s/2 + i] = x1*de; 
    }
}

extern "C" void binary_gradient_array_gpu(float *x, float *dx, int n, int size, BINARY_ACTIVATION a, float *y) 
{
    binary_gradient_array_kernel<<<cuda_gridsize(n/2), BLOCK>>>(x, dx, n/2, size, a, y);
    check_error(cudaGetLastError());
}
__global__ void binary_activate_array_kernel(float *x, int n, int s, BINARY_ACTIVATION a, float *y)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    int i = id % s;
    int b = id / s;
    float x1 = x[b*s + i];
    float x2 = x[b*s + s/2 + i];
    if(id < n) y[id] = x1*x2;
}

extern "C" void binary_activate_array_gpu(float *x, int n, int size, BINARY_ACTIVATION a, float *y) 
{
    binary_activate_array_kernel<<<cuda_gridsize(n/2), BLOCK>>>(x, n/2, size, a, y);
    check_error(cudaGetLastError());
}

__global__ void activate_array_kernel(float *x, int n, ACTIVATION a)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < n) 
    {   
        x[i] = activate_kernel(x[i], a);
        // bool value1, value2;
        // float out1, out2;
        // std::bitset<16> b, d;
        // float_to_fix<<<1,1>>>(a, d, true, value1);
        // float_to_fix<<<1,1>>>(a, b, false, value2);
        // fix_to_float<<<1,1>>>(b, out1);
        // fix_to_float<<<1,1>>>(d, out2);
        // if(value1||value2){
        //     x[i] = fabs(out1-x[i])>fabs(out2-x[i])?out2:out1;
        // }
        
        // printf("activation value is: %f\n", x[i]);
    }
}

__global__ void gradient_array_kernel(float *x, int n, ACTIVATION a, float *delta)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < n) delta[i] *= gradient_kernel(x[i], a);
}

extern "C" void activate_array_gpu(float *x, int n, ACTIVATION a) 
{
    activate_array_kernel<<<cuda_gridsize(n), BLOCK>>>(x, n, a);
    // float *output = (float*)malloc(4*n);
    // cudaMemcpy(output,x,4*n,cudaMemcpyDeviceToHost);
    // int i;
    // for(i=0;i<n;i++){
    //     // printf("activation value is: %f\n", output[i]);
    //     bool value1, value2;
    //     float out1, out2;
    //     std::bitset<16> b, d;
    //     float_to_fix(output[i], d, true, value1);
    //     float_to_fix(output[i], b, false, value2);
    //     fix_to_float(b, out1);
    //     // printf("float value1 : %f\n", out1);
    //     fix_to_float(d, out2);
    //     // printf("float value2 : %f\n", out2);
    //     if(value1||value2){
    //         // printf("enable\n");
    //         output[i] = fabs(out1-output[i])>fabs(out2-output[i])?out2:out1;
    //         // printf("output value: %f\n", output[i]);
    //     }
    // }
    // cudaMemcpy(x,output,4*n,cudaMemcpyHostToDevice);
    check_error(cudaGetLastError());
    // int i;
    // for(i=0;i<n;i++) printf("activation value is: %f\n", x[0]);
}

extern "C" void gradient_array_gpu(float *x, int n, ACTIVATION a, float *delta) 
{
    gradient_array_kernel<<<cuda_gridsize(n), BLOCK>>>(x, n, a, delta);
    check_error(cudaGetLastError());
}
