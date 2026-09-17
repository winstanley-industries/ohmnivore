"""Keep the reference runtime ABI independent of production sanitizer settings."""

def _reference_runtime_transition_impl(_settings, _attr):
    return {
        "@llvm//config:asan": False,
        "@llvm//config:ubsan": False,
    }

_reference_runtime_transition = transition(
    implementation = _reference_runtime_transition_impl,
    inputs = [],
    outputs = ["@llvm//config:asan", "@llvm//config:ubsan"],
)

def _reference_runtime_impl(ctx):
    library = ctx.attr.library[0]
    return [DefaultInfo(files = library[DefaultInfo].files)]

reference_runtime = rule(
    implementation = _reference_runtime_impl,
    attrs = {
        "library": attr.label(mandatory = True, cfg = _reference_runtime_transition),
        "_allowlist_function_transition": attr.label(
            default = "@bazel_tools//tools/allowlists/function_transition_allowlist",
        ),
    },
)
