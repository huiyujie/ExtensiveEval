import subprocess
import os

def generate_command(dimensions, file_name):
    dimensions.reverse()
    commands = []
    commands.append("/users/huiyujie/SZ3-install/bin/sz3")
    commands.append("-d")
    commands.append("-i")
    commands.append(f"./tput_dat/{file_name}.dat")
    commands.append("-c")
    commands.append("./sz3.conf")
    commands.append("-o")
    commands.append(f"test.data.sz")
    commands.append("-M")
    commands.append("REL")
    commands.append("0.1")
    commands.append(f"-{len(dimensions)}")
    for dim in dimensions:
        commands.append(f"{dim}")

    return commands

def run_sz3(commands):
    try:
        result = subprocess.run(commands, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True)

        if result.returncode == 0:
            msg = result.stdout
        else:
            print("Program encountered an error.")
            print("Error Output:")
            print(result.stderr)
    except FileNotFoundError:
        print("sz3 not found")

    return msg
