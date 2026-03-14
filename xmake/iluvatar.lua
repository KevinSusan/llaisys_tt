target("llaisys-device-iluvatar")
    set_kind("static")
    add_deps("llaisys-utils")
    set_languages("cxx17")
    set_warnings("all", "error")
    add_cxflags("-fPIC", "-Wno-unknown-pragmas")

    -- Iluvatar CoreX uses clang++ with CUDA frontend
    set_toolchains("clang")
    add_cxflags("-x", "cuda", "--cuda-gpu-arch=ivcore10", "--cuda-path=/usr/local/corex", {force = true})
    add_includedirs("/usr/local/corex/include")
    add_linkdirs("/usr/local/corex/lib64")
    add_links("cudart")

    add_files("../src/device/iluvatar/iluvatar_runtime_api.cu")
    add_files("../src/device/iluvatar/iluvatar_resource.cu")
    on_install(function (target) end)
target_end()

target("llaisys-ops-iluvatar")
    set_kind("static")
    add_deps("llaisys-tensor")
    set_languages("cxx17")
    set_warnings("all", "error")
    add_cxflags("-fPIC", "-Wno-unknown-pragmas")

    -- Iluvatar CoreX uses clang++ with CUDA frontend
    set_toolchains("clang")
    add_cxflags("-x", "cuda", "--cuda-gpu-arch=ivcore10", "--cuda-path=/usr/local/corex", {force = true})
    add_includedirs("/usr/local/corex/include")
    add_linkdirs("/usr/local/corex/lib64")
    add_links("cudart")

    add_files("../src/ops/*/nvidia/*.cu")
    on_install(function (target) end)
target_end()
