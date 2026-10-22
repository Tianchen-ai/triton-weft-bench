#pragma once
#include "worker.h"
#include <dlfcn.h>

inline void wb_ggml_reference_info(const char *name,const char *entry,const void *address,const std::string &extra="") {
  Dl_info info{};dladdr(address,&info);
  std::printf("WB_DETAILS {\"implementation\":%s,\"reference_dispatch\":{\"library\":\"GGML\",\"entry\":%s,\"resolved_symbol\":%s,\"dso\":%s%s}}\n",
    wb::json(name).c_str(),wb::json(entry).c_str(),wb::json(info.dli_sname?info.dli_sname:"").c_str(),
    wb::json(info.dli_fname?info.dli_fname:"").c_str(),extra.c_str());
}
