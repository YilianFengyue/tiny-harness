from harness.tools.bash import is_read_only_command


def test_bash_read_only_commands_are_parallel_safe():
    for command in [
        "ls",
        "dir",
        "rg TODO",
        "git status",
        "git diff -- app.py",
        "git log --oneline",
        "Get-ChildItem",
    ]:
        assert is_read_only_command({"command": command}), command


def test_bash_side_effect_commands_are_not_read_only():
    for command in [
        "python script.py",
        "node build.js",
        "npm test",
        "pip install x",
        "git commit -m hi",
        "echo hi > out.txt",
        "Set-Content out.txt hi",
        "Remove-Item out.txt",
    ]:
        assert not is_read_only_command({"command": command}), command
