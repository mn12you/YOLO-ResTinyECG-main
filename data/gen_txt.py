import os

# Get the current working directory
directory = './data/INCART_10s/val'
file_paths=[]
# Get a list of all files and directories in the current folder
for dirpath, dirnames,files_in_directory in os.walk(directory):

    # Create a list of full paths for the files and directories
    file_paths = file_paths+[os.path.join(dirpath[2:], file) for file in files_in_directory if file.endswith('.png') ]

# Specify the output text file
output_file = 'val.txt'

# Write the file paths to the output text file
with open(output_file, 'w') as f:
    for path in file_paths:
        f.write(f"{path}\n")

print(f"File paths have been exported to {output_file}")
