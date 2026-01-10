/*
This tool collects ONLY CONDITIONAL BRANCHES with their global history


 */

#include "pin.H"
#include <iostream>
#include <fstream>
#include <string>
#include <deque>
#include <map>
#include <iomanip>

using std::string;
using std::ofstream;
using std::endl;
using std::hex;
using std::dec;
using std::cerr;
using std::cout;
using std::deque;
using std::map;

//CONFIGURATION PARAMETERS - EDIT THESE TO CUSTOMIZE BEHAVIOR

//History length: How many previous branches to track
#define HISTORY_LENGTH 200

//PC encoding: How many bits of branch PC to encode
#define PC_BITS 12
#define PC_MASK ((1 << PC_BITS) - 1)

//Sampling parameters to control dataset size
//WARMUP_BRANCHES: Number of branches to skip at start of program
#define WARMUP_BRANCHES 100000

//MAX_SAMPLES_PER_BRANCH: Maximum samples to collect per static branch
#define MAX_SAMPLES_PER_BRANCH 10000

//SAMPLING_INTERVAL: Collect 1 out of every N branches after warmup
#define SAMPLING_INTERVAL 1

//GLOBAL DATA STRUCTURES

//Output files
ofstream TraceFile;      //Main CSV file with all traces
ofstream SummaryFile;    //Statistics about branch behavior
map<ADDRINT, ofstream*> PerBranchFiles;  //Optional: separate file per branch

//Global branch history buffer
//Stores encoded (PC, direction) values for the last N branches
deque<UINT32> globalHistory;

//Per-branch statistics
struct BranchStats {
    UINT64 total_executions;     
    UINT64 taken_count;        
    UINT64 samples_collected;   
    
    BranchStats() : total_executions(0), taken_count(0), samples_collected(0) {}
};
map<ADDRINT, BranchStats> branchStats;

//Counters
UINT64 totalBranches = 0;        
UINT64 conditionalBranches = 0;  
UINT64 samplesCollected = 0;      

//COMMAND LINE OPTIONS

KNOB<string> KnobOutputFile(KNOB_MODE_WRITEONCE, "pintool",
    "o", "branch_trace.csv", 
    "Output CSV file for branch traces");

KNOB<string> KnobOutputDir(KNOB_MODE_WRITEONCE, "pintool",
    "d", "branch_data", 
    "Output directory for summary and optional per-branch files");

KNOB<BOOL> KnobPerBranch(KNOB_MODE_WRITEONCE, "pintool",
    "per_branch", "0", 
    "Create separate file for each branch PC (0=no, 1=yes)");

KNOB<BOOL> KnobVerbose(KNOB_MODE_WRITEONCE, "pintool",
    "v", "0", 
    "Print verbose progress information (0=no, 1=yes)");

KNOB<UINT64> KnobMaxTotalSamples(KNOB_MODE_WRITEONCE, "pintool",
    "max_samples", "1000000", 
    "Maximum total samples to collect across all branches");

// UTILS

/*
Format: [PC_BITS bits of PC][1 bit direction]
Example with PC_BITS=12:
  PC=0x400587, taken=1  -> (0x587 << 1) | 1 = 0xB0F
  PC=0x400587, taken=0  -> (0x587 << 1) | 0 = 0xB0E
*/
inline UINT32 EncodeBranch(ADDRINT pc, BOOL taken) {
    UINT32 pc_low = (UINT32)(pc & PC_MASK);
    return (pc_low << 1) | (taken ? 1 : 0);
}

/*
ShouldSample: Decide whether to save this branch instance
 */
inline BOOL ShouldSample(ADDRINT pc) {
    if (conditionalBranches < WARMUP_BRANCHES) {
        return false;
    }
    
    if (samplesCollected >= KnobMaxTotalSamples.Value()) {
        return false;
    }
    
    if (branchStats[pc].samples_collected >= MAX_SAMPLES_PER_BRANCH) {
        return false;
    }
    
    if ((conditionalBranches - WARMUP_BRANCHES) % SAMPLING_INTERVAL != 0) {
        return false;
    }
    
    return true;
}

//ANALYSIS FUNCTIONS (called during program execution)

VOID RecordConditionalBranch(ADDRINT pc, BOOL taken) {
    conditionalBranches++;
    
    //Update statistics for this branch
    branchStats[pc].total_executions++;
    if (taken) {
        branchStats[pc].taken_count++;
    }
    
    //Decide whether to save this instance
    if (ShouldSample(pc)) {
        branchStats[pc].samples_collected++;
        samplesCollected++;
        
        //Write to main CSV file
        //Format: branch_pc,taken,hist_0,hist_1,...,hist_N
        TraceFile << hex << "0x" << pc << "," 
                  << dec << (taken ? 1 : 0);
        
        //Write history
        for (auto it = globalHistory.begin(); it != globalHistory.end(); ++it) {
            TraceFile << "," << *it;
        }
        TraceFile << endl;
        
        //Optionally write to per-branch file
        if (KnobPerBranch.Value()) {
            if (PerBranchFiles.find(pc) == PerBranchFiles.end()) {
                std::stringstream filename;
                filename << KnobOutputDir.Value() << "/branch_0x" 
                        << hex << pc << ".csv";
                
                PerBranchFiles[pc] = new ofstream(filename.str().c_str());
                
                *(PerBranchFiles[pc]) << "taken,history" << endl;
            }
            
            *(PerBranchFiles[pc]) << (taken ? 1 : 0);
            for (auto h : globalHistory) {
                *(PerBranchFiles[pc]) << "," << h;
            }
            *(PerBranchFiles[pc]) << endl;
        }
        
        if (KnobVerbose.Value() && samplesCollected % 10000 == 0) {
            cerr << "Progress: " << samplesCollected << " samples collected, "
                 << conditionalBranches << " branches executed" << endl;
        }
    }
    
    //ALWAYS update global history
    UINT32 encoded = EncodeBranch(pc, taken);
    globalHistory.push_back(encoded);
    
    //Maintain fixed history length
    if (globalHistory.size() > HISTORY_LENGTH) {
        globalHistory.pop_front();
    }
}

//INSTRUMENTATION FUNCTIONS
VOID Instruction(INS ins, VOID *v) {
    // Only instrument conditional branches
    // HasFallThrough = true means there's a "not-taken" path
    if (INS_IsBranch(ins) && INS_HasFallThrough(ins)) {
        INS_InsertCall(
            ins, 
            IPOINT_BEFORE,                    // Call before branch executes
            (AFUNPTR)RecordConditionalBranch, // Function to call
            IARG_INST_PTR,                    // Pass branch PC
            IARG_BRANCH_TAKEN,                // Pass taken/not-taken
            IARG_END
        );
    }
}

//FINALIZATION 

VOID Fini(INT32 code, VOID *v) {
    TraceFile.close();
    
    //Close per-branch files
    for (auto& pair : PerBranchFiles) {
        pair.second->close();
        delete pair.second;
    }
    
    //Write summary statistics
    SummaryFile << "===============================================" << endl;
    SummaryFile << "Branch Prediction Data Collection Summary" << endl;
    SummaryFile << "===============================================" << endl;
    SummaryFile << endl;
    
    SummaryFile << "Configuration:" << endl;
    SummaryFile << "  History Length: " << HISTORY_LENGTH << endl;
    SummaryFile << "  PC Bits: " << PC_BITS << endl;
    SummaryFile << "  Warmup Branches: " << WARMUP_BRANCHES << endl;
    SummaryFile << "  Sampling Interval: 1/" << SAMPLING_INTERVAL << endl;
    SummaryFile << "  Max Samples Per Branch: " << MAX_SAMPLES_PER_BRANCH << endl;
    SummaryFile << endl;
    
    SummaryFile << "Execution Statistics:" << endl;
    SummaryFile << "  Total Branches Executed: " << totalBranches << endl;
    SummaryFile << "  Conditional Branches: " << conditionalBranches << endl;
    SummaryFile << "  Unique Branch PCs: " << branchStats.size() << endl;
    SummaryFile << "  Samples Collected: " << samplesCollected << endl;
    SummaryFile << "  Sampling Rate: " 
                << (double)samplesCollected / conditionalBranches * 100 
                << "%" << endl;
    SummaryFile << endl;
    
    //Per-branch statistics
    SummaryFile << "===============================================" << endl;
    SummaryFile << "Per-Branch Statistics" << endl;
    SummaryFile << "===============================================" << endl;
    SummaryFile << endl;
    
    SummaryFile << std::setw(18) << "Branch PC" << " | "
                << std::setw(12) << "Executions" << " | "
                << std::setw(12) << "Taken" << " | "
                << std::setw(8) << "Bias" << " | "
                << std::setw(10) << "Samples" << endl;
    SummaryFile << std::string(80, '-') << endl;
    
    //Sort branches by execution count
    std::vector<std::pair<ADDRINT, BranchStats*>> sortedBranches;
    for (auto& pair : branchStats) {
        sortedBranches.push_back({pair.first, &pair.second});
    }
    std::sort(sortedBranches.begin(), sortedBranches.end(),
              [](const auto& a, const auto& b) { 
                  return a.second->total_executions > b.second->total_executions; 
              });
    
    for (auto& pair : sortedBranches) {
        ADDRINT pc = pair.first;
        BranchStats* stats = pair.second;
        
        double bias = (double)stats->taken_count / stats->total_executions;
        
        SummaryFile << "0x" << hex << std::setw(16) << pc << " | "
                   << dec << std::setw(12) << stats->total_executions << " | "
                   << std::setw(12) << stats->taken_count << " | "
                   << std::fixed << std::setprecision(3) 
                   << std::setw(8) << bias << " | "
                   << std::setw(10) << stats->samples_collected << endl;
    }
    
    SummaryFile.close();
    
    //Console output
    if (KnobVerbose.Value()) {
        cerr << endl;
        cerr << "=== Collection Complete ===" << endl;
        cerr << "Conditional Branches: " << conditionalBranches << endl;
        cerr << "Samples Collected: " << samplesCollected << endl;
        cerr << "Unique Branches: " << branchStats.size() << endl;
        cerr << "Output: " << KnobOutputFile.Value() << endl;
        cerr << "Summary: " << KnobOutputDir.Value() << "/summary.txt" << endl;
    }
}

//MAIN

INT32 Usage() {
    cerr << "Branch Extractor with Sampling for CNN Training" << endl;
    cerr << endl;
    cerr << "Collects only conditional branches with global history." << endl;
    cerr << "Uses sampling to keep dataset size manageable." << endl;
    cerr << endl;
    cerr << KNOB_BASE::StringKnobSummary() << endl;
    cerr << endl;
    cerr << "Example usage:" << endl;
    cerr << "  pin -t branchExt_sampling.so -o trace.csv -d output \\" << endl;
    cerr << "      -max_samples 500000 -v 1 -- ./your_program" << endl;
    return -1;
}

int main(int argc, char *argv[]) {
    //Initialize Pin
    PIN_InitSymbols();
    if (PIN_Init(argc, argv)) {
        return Usage();
    }
    
    //Create output directory
    string mkdirCmd = "mkdir -p " + KnobOutputDir.Value();
    system(mkdirCmd.c_str());
    
    //Open main trace file
    TraceFile.open(KnobOutputFile.Value().c_str());
    if (!TraceFile.is_open()) {
        cerr << "Error: Cannot open output file " << KnobOutputFile.Value() << endl;
        return -1;
    }
    
    //Write CSV header
    TraceFile << "branch_pc,taken";
    for (int i = 0; i < HISTORY_LENGTH; i++) {
        TraceFile << ",hist_" << i;
    }
    TraceFile << endl;
    
    //Open summary file
    string summaryPath = KnobOutputDir.Value() + "/summary.txt";
    SummaryFile.open(summaryPath.c_str());
    if (!SummaryFile.is_open()) {
        cerr << "Error: Cannot open summary file" << endl;
        return -1;
    }
    
    if (KnobVerbose.Value()) {
        cerr << "=== Branch Extractor Started ===" << endl;
        cerr << "Configuration:" << endl;
        cerr << "  History Length: " << HISTORY_LENGTH << endl;
        cerr << "  PC Bits: " << PC_BITS << endl;
        cerr << "  Warmup: " << WARMUP_BRANCHES << " branches" << endl;
        cerr << "  Sampling: 1/" << SAMPLING_INTERVAL << endl;
        cerr << "  Max Samples Per Branch: " << MAX_SAMPLES_PER_BRANCH << endl;
        cerr << "  Max Total Samples: " << KnobMaxTotalSamples.Value() << endl;
        cerr << "  Output: " << KnobOutputFile.Value() << endl;
        cerr << endl;
    }
    
    //Register instrumentation callbacks
    INS_AddInstrumentFunction(Instruction, 0);
    PIN_AddFiniFunction(Fini, 0);
    
    //Start the program
    PIN_StartProgram();
    
    return 0;
}