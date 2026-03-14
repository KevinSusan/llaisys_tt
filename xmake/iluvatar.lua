target("llaisys-device-iluvatar")
    set_kind("static")
    add_deps("llaisys-utils")
    set_languages("cxx17")
    set_warnings("all", "error")
    set_policy("build.cuda.devlink", false)

    add_includedirs("/usr/local/corex/include")
    add_linkdirs("/usr/local/corex/lib64")
    add_links("cudart")

    add_files("../src/device/iluvatar/iluvatar_runtime_api.cu", {
        rule = "iluvatar_cu"
    })
    add_files("../src/device/iluvatar/iluvatar_resource.cu", {
        rule = "iluvatar_cu"
    })

    on_install(function (target) end)
target_end()

target("llaisys-ops-iluvatar")
    set_kind("static")
    add_deps("llaisys-tensor")
    set_languages("cxx17")
    set_warnings("all", "error")
    set_policy("build.cuda.devlink", false)

    add_includedirs("/usr/local/corex/include")
    add_linkdirs("/usr/local/corex/lib64")
    add_links("cudart")

    add_files("../src/ops/*/nvidia/*.cu", {
        rule = "iluvatar_cu"
    })

    on_install(function (target) end)
target_end()

rule("iluvatar_cu")
    set_extensions(".cu")
    on_build_file(function (target, sourcefile, opt)
        import("core.project.depend")
        import("core.tool.compiler")

        local objectfile = target:objectfile(sourcefile)
        local dependfile = target:dependfile(objectfile)

        depend.on_changed(function ()
            local argv = {
                "-x", "cuda",
                "--cuda-gpu-arch=ivcore10",
                "--cuda-path=/usr/local/corex",
                "-std=c++17",
                "-fPIC",
                "-O3",
                "-DENABLE_ILUVATAR_API",
                "-Iinclude",
                "-I/usr/local/corex/include",
                "-c",
                "-o", objectfile,
                sourcefile
            }

            os.vrunv("/usr/local/corex/bin/clang++", argv)
        end, {dependfile = dependfile, files = {sourcefile}})
    end)
rule_end()
