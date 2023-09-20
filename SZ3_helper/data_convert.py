import subprocess

def compile_converter():
    cpp_file = "converter.cpp"
    exe_file = "converter"

    try:
        compile_result = subprocess.run(['g++', '-o', exe_file, cpp_file], stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE, text=True)
        if compile_result.returncode == 0:
            print("compile converter.cpp successfully")
        else:
            print("compile converter.cpp failed")
            print(compile_result.stderr)
    except FileNotFoundError:
        print("g++ not found")


def run_converter(input_file, output_file):
    binary_path = "./converter"

    try:
        result = subprocess.run([binary_path, input_file, output_file], stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True)

        if result.returncode != 0:
            print("Program encountered an error.")
            print("Error Output:")
            print(result.stderr)
    except FileNotFoundError:
        print("converter not found")
