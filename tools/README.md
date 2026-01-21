# Extracting Branch Traces Through Intel Pin

## Prerequisites 

* Intel Pin: [Download here](https://www.intel.com/content/www/us/en/developer/articles/tool/pin-a-binary-instrumentation-tool-downloads.html)
* Linux: Tested on Ubuntu (WSL2)
* g++ compiler 

## Installation 
1. Download Intel Pin and clone this repository </br>
2. Under PinTool/source/tools, create a new directory (ideally the same name as the cpp source code) where the source code will belong to</br>
3. Move both branchExt.cpp and Makefile to the new directory</br>

```
pin/
|--source/
   |--tools/
      |--branchExt/ (New directory)
         |--branchExt.cpp (Copy here)
         |--Makefile (Copy here)
```

4. Build the tool

```
export PIN_ROOT=/path/to/pin/directory #Edit this path to point to your local pin tool

cd $PIN_ROOT/source/tools/branchExt 

make
```

5. The executable will be built under $PIN_ROOT/source/tools/branchExt/obj-intel64

6. Once the executable has been built, run the target program through Pin using the .so file

```
$PIN_ROOT/pin -t obj-intel64/branchExt.so -- /path/to/program  program_args
```

**Description of the command line options for this sampler**

| Option | Default | Description | 
| -o <file> | branches.csv | Output CSV file |
| -sampling <N> | 10 | Collect 1 in N branches |
| -warmup <N> | 100000 | Skip the first N branches |
| -max_per_branch <N> | 10000 | Max number of unique branch PCs |
| -max_total <N> | 1000000 | Max number of branch samples |
| -v <0 or 1> | 0 | Verbose |

**Examples** 

To see the format of program arguments, refer to the .runconf file inside $PARSEC_ROOT/pkgs/apps/benchmark_name/parsec

```
$PIN_ROOT/pin -t obj-intel64/branchExt.so \
  -o blackscholes_simsmall.csv \
  -sampling 10 \
  -warmup 100000 \
  -max_per_branch 10000\
  -max_total 1000000 \
  -v 1 \
  -- $PARSEC_ROOT/pkgs/apps/blackscholes/inst/amd64-linux.gcc/bin/blackscholes 1 \
     $PARSEC_ROOT/pkgs/apps/blackscholes/inputs/in_4K.txt \
     prices.txt
```

```
$PIN_ROOT/pin -t obj-intel64/branchExt.so \
  -o fluidanimate_simmedium.csv \
  -sampling 1 \
  -warmup 100000 \
  -max_per_branch 10000\
  -max_total 1000000 \
  -v 1 \
  -- $PARSEC_ROOT/pkgs/apps/blackscholes/inst/amd64-linux.gcc/bin/fluidanimate 1 \
     5 $PARSEC_ROOT/pkgs/apps/fluidanimate/inputs/in_100K.fluid \
     out.fluid
```