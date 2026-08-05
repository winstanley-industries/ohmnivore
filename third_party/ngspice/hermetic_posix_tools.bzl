"""Creates a tree of checksum-pinned BusyBox POSIX build tools."""

_APPLETS = [
    "awk",
    "autoconf",
    "automake",
    "basename",
    "cat",
    "chmod",
    "cmp",
    "cp",
    "cut",
    "date",
    "dd",
    "diff",
    "dirname",
    "echo",
    "env",
    "expr",
    "false",
    "find",
    "file",
    "grep",
    "head",
    "install",
    "ln",
    "ls",
    "mkdir",
    "m4",
    "mv",
    "printf",
    "pkg-config",
    "pwd",
    "readlink",
    "rm",
    "rmdir",
    "sed",
    "sha256sum",
    "sh",
    "sleep",
    "sort",
    "tail",
    "test",
    "touch",
    "tr",
    "true",
    "uname",
    "wc",
    "which",
    "xargs",
]

def _hermetic_posix_tools_impl(ctx):
    output = ctx.actions.declare_directory(ctx.label.name)
    ctx.actions.run(
        arguments = [ctx.file.busybox.path, output.path] + _APPLETS,
        executable = ctx.executable.builder,
        inputs = [ctx.file.busybox],
        outputs = [output],
        progress_message = "Creating hermetic POSIX tool directory",
        mnemonic = "HermeticPosixTools",
    )
    return [DefaultInfo(files = depset([output]))]

hermetic_posix_tools = rule(
    implementation = _hermetic_posix_tools_impl,
    attrs = {
        "busybox": attr.label(
            allow_single_file = True,
            executable = True,
            mandatory = True,
            cfg = "exec",
        ),
        "builder": attr.label(
            executable = True,
            mandatory = True,
            cfg = "exec",
        ),
    },
)

def _hermetic_copy_impl(ctx):
    output = ctx.actions.declare_file(ctx.attr.output_name)
    ctx.actions.run(
        arguments = ["--copy", ctx.file.src.path, output.path],
        executable = ctx.executable.builder,
        inputs = [ctx.file.src],
        outputs = [output],
        progress_message = "Copying pinned static archive",
        mnemonic = "HermeticCopy",
    )
    return [DefaultInfo(files = depset([output]))]

hermetic_copy = rule(
    implementation = _hermetic_copy_impl,
    attrs = {
        "builder": attr.label(
            executable = True,
            mandatory = True,
            cfg = "exec",
        ),
        "output_name": attr.string(mandatory = True),
        "src": attr.label(allow_single_file = True, mandatory = True),
    },
)
