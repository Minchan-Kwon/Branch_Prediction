/*
 Pin Tool for Extracting Branch History 
 This tool collects the PC of the current conditional branch PC and its corresponding branch direction (0 or 1).
 The output is in a CSV format. 
 
 Refer to README for detailed usage guide.
 */

#include "pin.H"
#include <iostream>
#include <fstream>
#include <string>
#include <map>

using std::string;
using std::ofstream;
using std::endl;
using std::cerr;

// GLOBAL VARIABLES
ofstream OutFile;

// Statistics
UINT64 totalBranches = 0;
UINT64 samplesCollected = 0;

// Per-branch sample counts
std::map<ADDRINT, UINT64> branchSampleCount;

// COMMAND LINE OPTIONS
KNOB<string> KnobOutputFile(KNOB_MODE_WRITEONCE, "pintool",
    "o", "branches.csv", 
    "Output CSV file");

KNOB<UINT64> KnobSamplingInterval(KNOB_MODE_WRITEONCE, "pintool",
    "sampling", "10", 
    "Sampling interval: collect 1 in N branches (1=all, 10=10%, 100=1%)");

KNOB<UINT64> KnobWarmup(KNOB_MODE_WRITEONCE, "pintool",
    "warmup", "100000", 
    "Warmup: skip first N branches (0=no warmup)");

KNOB<UINT64> KnobMaxPerBranch(KNOB_MODE_WRITEONCE, "pintool",
    "max_per_branch", "10000", 
    "Maximum samples per branch PC");

KNOB<UINT64> KnobMaxTotal(KNOB_MODE_WRITEONCE, "pintool",
    "max_total", "1000000", 
    "Maximum total samples to collect");

KNOB<BOOL> KnobVerbose(KNOB_MODE_WRITEONCE, "pintool",
    "v", "0", 
    "Verbose output (0=quiet, 1=verbose)");

// SAMPLING LOGIC
inline BOOL ShouldSample(ADDRINT pc) {
    // Filter 1: Warmup period
    if (totalBranches < KnobWarmup.Value()) {
        return false;
    }
    
    // Filter 2: Total sample limit
    if (samplesCollected >= KnobMaxTotal.Value()) {
        return false;
    }
    
    // Filter 3: Per-branch limit
    if (branchSampleCount[pc] >= KnobMaxPerBranch.Value()) {
        return false;
    }
    
    // Filter 4: Sampling interval
    UINT64 branchesAfterWarmup = totalBranches - KnobWarmup.Value();
    if (branchesAfterWarmup % KnobSamplingInterval.Value() != 0) {
        return false;
    }
    
    return true;
}

// ANALYSIS FUNCTION
VOID RecordBranch(ADDRINT pc, BOOL taken) {
    totalBranches++;
    
    if (ShouldSample(pc)) {
        // Write: PC,direction
        OutFile << "0x" << std::hex << pc << "," 
                << std::dec << (taken ? 1 : 0) << endl;
        
        branchSampleCount[pc]++;
        samplesCollected++;
        
        // Report Progress 
        if (KnobVerbose.Value() && samplesCollected % 10000 == 0) {
            cerr << "Progress: " << samplesCollected << " samples collected ("
                 << totalBranches << " branches executed)" << endl;
        }
    }
}

// INSTRUMENTATION
VOID Instruction(INS ins, VOID *v) {
    // Only track conditional branches
    if (INS_IsBranch(ins) && INS_HasFallThrough(ins)) {
        INS_InsertCall(
            ins, 
            IPOINT_BEFORE,
            (AFUNPTR)RecordBranch,
            IARG_INST_PTR,
            IARG_BRANCH_TAKEN,
            IARG_END
        );
    }
}

// FINALIZATION
VOID Fini(INT32 code, VOID *v) {
    OutFile.close();
    
    if (KnobVerbose.Value()) {
        cerr << "\n========================================" << endl;
        cerr << "Collection Complete!" << endl;
        cerr << "========================================" << endl;
        cerr << "Total branches executed: " << totalBranches << endl;
        cerr << "Samples collected: " << samplesCollected << endl;
        cerr << "Unique branch PCs: " << branchSampleCount.size() << endl;
        
        if (totalBranches > 0) {
            double samplingRate = (100.0 * samplesCollected) / totalBranches;
            cerr << "Effective sampling rate: " << samplingRate << "%" << endl;
        }
        
        cerr << "Output file: " << KnobOutputFile.Value() << endl;
        cerr << "========================================" << endl;
    }
}

// USAGE
INT32 Usage() {
    cerr << "Simple Branch Extractor for CNN Branch Prediction" << endl;
    cerr << "Collects: Branch PC + Direction (Taken/Not-Taken)" << endl;
    cerr << endl;
    cerr << "Usage:" << endl;
    cerr << "  pin -t obj-intel64/branchExt.so [OPTIONS] -- <program> [args]" << endl;
    cerr << endl;
    cerr << "Examples:" << endl;
    cerr << "  # Quick test (low warmup)" << endl;
    cerr << "  pin -t obj-intel64/branchExt.so -o test.csv -warmup 1000 -v 1 -- /bin/ls" << endl;
    cerr << endl;
    cerr << "  # PARSEC collection" << endl;
    cerr << "  pin -t obj-intel64/branchExt.so -o trace.csv -sampling 10 -warmup 100000 -v 1 -- ./benchmark" << endl;
    cerr << endl;
    cerr << "  # Maximum data (collect everything)" << endl;
    cerr << "  pin -t obj-intel64/branchExt.so -o full.csv -sampling 1 -warmup 0 -max_total 10000000 -- ./program" << endl;
    cerr << endl;
    cerr << KNOB_BASE::StringKnobSummary() << endl;
    return -1;
}

// MAIN
int main(int argc, char *argv[]) {
    PIN_InitSymbols();
    
    if (PIN_Init(argc, argv)) {
        return Usage();
    }
    
    // Open output file
    OutFile.open(KnobOutputFile.Value().c_str());
    if (!OutFile.is_open()) {
        cerr << "Error: Cannot open output file " << KnobOutputFile.Value() << endl;
        return -1;
    }
    
    // Write CSV header
    OutFile << "branch_pc,taken" << endl;
    
    // Print configuration
    if (KnobVerbose.Value()) {
        cerr << "========================================" << endl;
        cerr << "Branch Extractor Started" << endl;
        cerr << "========================================" << endl;
        cerr << "Configuration:" << endl;
        cerr << "  Sampling interval: 1/" << KnobSamplingInterval.Value() << endl;
        cerr << "  Warmup branches: " << KnobWarmup.Value() << endl;
        cerr << "  Max per branch: " << KnobMaxPerBranch.Value() << endl;
        cerr << "  Max total samples: " << KnobMaxTotal.Value() << endl;
        cerr << "  Output file: " << KnobOutputFile.Value() << endl;
        cerr << "========================================" << endl;
        cerr << endl;
    }
    
    // Register callbacks
    INS_AddInstrumentFunction(Instruction, 0);
    PIN_AddFiniFunction(Fini, 0);
    
    // Start instrumented program
    PIN_StartProgram();
    
    return 0;
}