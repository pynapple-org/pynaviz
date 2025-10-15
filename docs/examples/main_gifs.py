import os
import subprocess

def run_gifs(directory):
    for filename in os.listdir(directory):
        if filename.endswith('.py') and filename != os.path.basename(__file__) and "example_" in filename:
            filepath = os.path.join(directory, filename)
            print(f'Running {filename}...')
            subprocess.run(['python', filepath])

if __name__ == '__main__':
    run_gifs(os.path.dirname(os.path.abspath(__file__)))