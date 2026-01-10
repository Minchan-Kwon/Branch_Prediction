# Extracting Branch Traces Through Intel Pin
1. Download Intel Pin and clone this repository </br>
2. Under PinTool/source/tools, create a new directory (ideally the same name as the cpp source code) where the source code will belong to</br>
3. Move both sampler.cpp and Makefile to the new directory</br>
4. Build the tool
'''
make obj-intel64/sampler.so
'''
5. Once the executable has been built, run the target program through Pin using the .so file
'''
../../../pin -t obj-intel64/sampler.so -o history.csv -d output_dir -- ./your_program
'''

**Description of the command line options for this sampler**
'''
pin -t <path_to_so_file> \
    -o branch_trace.csv \
    -d branch_data_output \
    -per_branch 0 \
    -v 1 \
    -max_samples 2000000 \
    -- ./your_binary_program arg1 arg2
'''