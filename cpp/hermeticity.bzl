"""Analysis-time checks for the production C++ solver dependency boundary."""

load("@rules_cc//cc/common:cc_info.bzl", "CcInfo")

def _artifact_path(artifact):
    return artifact.path if artifact else ""

def _solver_cc_boundary_manifest_impl(ctx):
    cc_info = ctx.attr.target[CcInfo]
    compilation = cc_info.compilation_context
    headers = sorted([header.path for header in compilation.headers.to_list()])
    include_dirs = sorted(depset(
        direct = (
            compilation.includes.to_list() +
            compilation.quote_includes.to_list() +
            compilation.system_includes.to_list() +
            compilation.framework_includes.to_list()
        ),
    ).to_list())

    libraries = []
    link_flags = []
    for linker_input in cc_info.linking_context.linker_inputs.to_list():
        link_flags.extend(linker_input.user_link_flags)
        for library in linker_input.libraries:
            for artifact in [
                library.static_library,
                library.pic_static_library,
                library.dynamic_library,
                library.interface_library,
            ]:
                path = _artifact_path(artifact)
                if path:
                    libraries.append(path)
    libraries = sorted(depset(direct = libraries).to_list())
    link_flags = sorted(depset(direct = link_flags).to_list())

    all_entries = headers + include_dirs + libraries + link_flags
    forbidden_prefixes = [
        "/include",
        "/lib",
        "/lib64",
        "/opt",
        "/usr",
    ]
    forbidden_fragments = [
        "dense_oracle",
        "amd_l",
        "btf_l",
        "colamd_l.c",
        "klu_l",
        "klu_zl",
        "libblas",
        "liblapack",
        "libopenblas",
        "libgfortran",
        "libsuperlu",
        "libumfpack",
    ]
    forbidden = []
    for entry in all_entries:
        if any([entry == prefix or entry.startswith(prefix + "/") for prefix in forbidden_prefixes]):
            forbidden.append(entry)
        if any([fragment.lower() in entry.lower() for fragment in forbidden_fragments]):
            forbidden.append(entry)
    if forbidden:
        fail("production CcInfo contains forbidden system/dense/long-index input: %s" % sorted(depset(direct = forbidden).to_list()))

    required_suffixes = [
        "KLU/Include/klu.h",
        "AMD/Include/amd.h",
        "BTF/Include/btf.h",
        "COLAMD/Include/colamd.h",
        "SuiteSparse_config/SuiteSparse_config.h",
    ]
    for suffix in required_suffixes:
        if not any([header.endswith(suffix) for header in headers]):
            fail("production CcInfo is missing pinned SuiteSparse header: %s" % suffix)
    if not any([path.endswith("libklu.a") for path in libraries]):
        fail("production CcInfo is missing the static KLU library")

    output = ctx.actions.declare_file(ctx.label.name + ".txt")
    ctx.actions.write(
        output,
        "status=zero_forbidden_entries\n" +
        "headers:\n" + "\n".join(headers) + "\n" +
        "include_dirs:\n" + "\n".join(include_dirs) + "\n" +
        "libraries:\n" + "\n".join(libraries) + "\n" +
        "link_flags:\n" + "\n".join(link_flags) + "\n",
    )
    return [DefaultInfo(files = depset([output]))]

solver_cc_boundary_manifest = rule(
    implementation = _solver_cc_boundary_manifest_impl,
    attrs = {
        "target": attr.label(mandatory = True, providers = [CcInfo]),
    },
)
