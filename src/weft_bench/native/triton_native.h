#pragma once
#include <weft/runtime.h>
#include <dlfcn.h>
#include <fstream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>

namespace wb {
class TritonNative {
  std::unique_ptr<weft_context, decltype(&weft_context_release)> context_{nullptr,weft_context_release};
  std::unique_ptr<weft_kernel, decltype(&weft_kernel_release)> kernel_{nullptr,weft_kernel_release};
  std::unique_ptr<void, decltype(&dlclose)> launcher_library_{nullptr,dlclose};
  std::unique_ptr<void, decltype(&dlclose)> symbols_library_{nullptr,dlclose};
  static void require(weft_status status,const weft_error &error) {
    if(status!=WEFT_OK)throw std::runtime_error(std::string(weft_status_name(status))+": "+error.message);
  }
public:
  TritonNative(const char *library,const char *manifest) {
    std::ifstream stream(manifest);
    if(!stream)throw std::runtime_error("cannot open native artifact manifest");
    std::string json((std::istreambuf_iterator<char>(stream)),std::istreambuf_iterator<char>());
    weft_error error{};
    weft_context *context=nullptr;
    require(weft_context_create(&context,&error),error);
    context_.reset(context);
    weft_kernel *kernel=nullptr;
    require(weft_kernel_open_native(context_.get(),library,json.c_str(),&kernel,&error),error);
    kernel_.reset(kernel);
    symbols_library_.reset(dlopen(library,RTLD_NOW|RTLD_LOCAL));
    if(!symbols_library_)throw std::runtime_error(dlerror());
  }
  void check_worker() const {
    char error[1024]{};
    if(weft_kernel_check_worker(kernel_.get(),error,sizeof(error)))throw std::runtime_error(error);
  }
  template<class Function> Function entry() const {
    weft_error error{};
    weft_native_entry pointer=nullptr;
    require(weft_kernel_get_entry(kernel_.get(),&pointer,&error),error);
    return reinterpret_cast<Function>(pointer);
  }
  template<class Function> Function symbol(const char *name) const {
    auto pointer=dlsym(symbols_library_.get(),name);
    if(!pointer)throw std::runtime_error("native artifact lacks its declared builder symbol");
    return reinterpret_cast<Function>(pointer);
  }
  template<class Function> Function launcher(const char *library,const char *symbol) {
    launcher_library_.reset(dlopen(library,RTLD_NOW|RTLD_LOCAL));
    if(!launcher_library_)throw std::runtime_error(dlerror());
    auto pointer=dlsym(launcher_library_.get(),symbol);
    if(!pointer)throw std::runtime_error("native artifact lacks its declared grid launcher");
    return reinterpret_cast<Function>(pointer);
  }
  template<class Function> void grid(Function launcher,weft_native_entry entry,void *const *arguments,size_t x,size_t y,size_t z,int threads) const {
    char error[1024]{};
    int status=launcher(entry,arguments,x,y,z,threads,kernel_.get(),weft_kernel_check_worker,error,sizeof(error));
    if(status)throw std::runtime_error(std::string("grid launch: ")+error);
  }
};
}
