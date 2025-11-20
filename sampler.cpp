/*
 * Branch Extractor for CNN Branch Prediction
 * 
 * This tool collects ONLY CONDITIONAL BRANCHES with their global history
 * for training neural network branch predictors.
 * 
 * Output format: CSV file with columns:
 *   branch_pc, taken, history_0, history_1, ..., history_N
 * 
 * Features:
 * - Samples branches to keep dataset manageable
 * - Configurable history length
 * - Multiple output formats (CSV and per-branch files)
 * - Statistics summary
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

// ================================================================
// CONFIGURATION PARAMETERS - EDIT THESE TO CUSTOMIZE BEHAVIOR
// ================================================================

// History length: How many previous branches to track
// Papers use 200 (Tarsa) or 37-603 (BranchNet)
#define HISTORY_LENGTH 200

// PC encoding: How many bits of branch PC to encode
// 12 bits = 4096 unique values, good for most programs
#define PC_BITS 12
#define PC_MASK ((1 << PC_BITS) - 1)

// Sampling parameters to control dataset size
// WARMUP_BRANCHES: Number of branches to skip at program start
//   (skips library initialization, focuses on main program)
#define WARMUP_BRANCHES 100000

// MAX_SAMPLES_PER_BRANCH: Maximum samples to collect per static branch
//   This prevents one hot branch from dominating the dataset
#define MAX_SAMPLES_PER_BRANCH 10000

// SAMPLING_INTERVAL: Collect 1 out of every N branches after warmup
//   1 = collect all (large dataset)
//   10 = collect 10% (recommended for most cases)
//   100 = collect 1% (for very long runs)
#define SAMPLING_INTERVAL 10

// ================================================================
// GLOBAL DATA STRUCTURES
// ================================================================

// Output files
ofstream TraceFile;      // Main CSV file with all traces
ofstream SummaryFile;    // Statistics about branch behavior
map<ADDRINT, ofstream*> PerBranchFiles;  // Optional: separate file per branch

//Base Address Variable
ADDRINT base_address = 0;
BOOL base_address_set = FALSE;

// Global branch history buffer
// Stores encoded (PC, direction) values for the last N branches
deque<UINT32> globalHistory;

// Per-branch statistics
struct BranchStats {
    UINT64 total_executions;     // Total times this branch executed
    UINT64 taken_count;          // Times it was taken
    UINT64 samples_collected;    // Samples we saved (respects max limit)
    
    BranchStats() : total_executions(0), taken_count(0), samples_collected(0) {}
};
map<ADDRINT, BranchStats> branchStats;

// Counters
UINT64 totalBranches = 0;           // All branches seen
UINT64 conditionalBranches = 0;     // Only conditional branches
UINT64 samplesCollected = 0;        // Samples actually saved

// ================================================================
// COMMAND LINE OPTIONS
// ================================================================

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

// ================================================================
// UTILITY FUNCTIONS
// ================================================================

/*
 * EncodeBranch: Combine PC and direction into single value
 * 
 * Format: [PC_BITS bits of PC][1 bit direction]
 * Example with PC_BITS=12:
 *   PC=0x400587, taken=1  -> (0x587 << 1) | 1 = 0xB0F
 *   PC=0x400587, taken=0  -> (0x587 << 1) | 0 = 0xB0E
 * 
 * This encoding allows CNNs to distinguish same PC with different directions
 */
inline UINT32 EncodeBranch(ADDRINT pc, BOOL taken) {
    UINT32 pc_low = (UINT32)(pc & PC_MASK);
    return (pc_low << 1) | (taken ? 1 : 0);
}

/*
 * ShouldSample: Decide whether to save this branch instance
 * 
 * Criteria:
 * 1. Must be past warmup period
 * 2. Respects sampling interval (collect 1 in N)
 * 3. Haven't exceeded per-branch sample limit
 * 4. Haven't exceeded total sample limit
 */
inline BOOL ShouldSample(ADDRINT pc) {
    // Check warmup
    if (conditionalBranches < WARMUP_BRANCHES) {
        return false;
    }
    
    // Check global limit
    if (samplesCollected >= KnobMaxTotalSamples.Value()) {
        return false;
    }
    
    // Check per-branch limit
    if (branchStats[pc].samples_collected >= MAX_SAMPLES_PER_BRANCH) {
        return false;
    }
    
    // Sample every Nth branch
    if ((conditionalBranches - WARMUP_BRANCHES) % SAMPLING_INTERVAL != 0) {
        return false;
    }
    
    return true;
}

// ================================================================
// ANALYSIS FUNCTIONS (called during program execution)
// ================================================================

/*
 * ImageLoad: Capture base address when main executable loads
 */
VOID ImageLoad(IMG img, VOID *v) {
    if (IMG_IsMainExecutable(img) && !base_address_set) {
        base_address = IMG_LowAddress(img);
        base_address_set = TRUE;
        
        if (KnobVerbose.Value()) {
            cerr << "Base address detected: 0x" 
                 << hex << base_address << dec << endl;
        }
        
        // Write to trace file as comment
        TraceFile << "# Base Address: 0x" 
                  << hex << base_address << dec << endl;
    }
}

/*
 * RecordConditionalBranch: Main analysis routine
 * 
 * Called every time a conditional branch executes.
 * Decides whether to sample it, updates statistics, and maintains history.
 * 
 * Parameters:
 *   pc: Program counter (address) of the branch instruction
 *   taken: Whether branch was taken (true) or not-taken (false)
 */
VOID RecordConditionalBranch(ADDRINT pc, BOOL taken) {
    conditionalBranches++;
    
    // Update statistics for this branch
    branchStats[pc].total_executions++;
    if (taken) {
        branchStats[pc].taken_count++;
    }
    
    // Decide whether to save this instance
    if (ShouldSample(pc)) {
        branchStats[pc].samples_collected++;
        samplesCollected++;

        ADDRINT pc_offset = pc - base_address;
        
        // Write to main CSV file
        // Format: branch_pc,taken,hist_0,hist_1,...,hist_N
        TraceFile << hex << "0x" << pc_offset << "," 
                  << dec << (taken ? 1 : 0);
        
        // Write history
        for (auto it = globalHistory.begin(); it != globalHistory.end(); ++it) {
            TraceFile << "," << *it;
        }
        TraceFile << endl;
        
        // Optionally write to per-branch file
        if (KnobPerBranch.Value()) {
            // Create file if first time seeing this branch
            if (PerBranchFiles.find(pc) == PerBranchFiles.end()) {
                std::stringstream filename;
                filename << KnobOutputDir.Value() << "/branch_0x" 
                        << hex << pc << ".csv";
                
                PerBranchFiles[pc] = new ofstream(filename.str().c_str());
                
                // Write header
                *(PerBranchFiles[pc]) << "taken,history" << endl;
            }
            
            // Write to per-branch file
            *(PerBranchFiles[pc]) << (taken ? 1 : 0);
            for (auto h : globalHistory) {
                *(PerBranchFiles[pc]) << "," << h;
            }
            *(PerBranchFiles[pc]) << endl;
        }
        
        // Print progress periodically
        if (KnobVerbose.Value() && samplesCollected % 10000 == 0) {
            cerr << "Progress: " << samplesCollected << " samples collected, "
                 << conditionalBranches << " branches executed" << endl;
        }
    }
    
    // ALWAYS update global history (even if not sampling this instance)
    // This ensures history is continuous and accurate
    UINT32 encoded = EncodeBranch(pc, taken);
    globalHistory.push_back(encoded);
    
    // Maintain fixed history length
    if (globalHistory.size() > HISTORY_LENGTH) {
        globalHistory.pop_front();
    }
}

// ================================================================
// INSTRUMENTATION FUNCTIONS (called during JIT compilation)
// ================================================================

/*
 * Instruction: Decide which instructions to instrument
 * 
 * Called once for each instruction during JIT compilation.
 * We insert a callback before each conditional branch.
 * 
 * Filter:
 * - INS_IsBranch: Is it a branch instruction?
 * - INS_HasFallThrough: Does it have a fall-through path?
 *   (true = conditional, false = unconditional)
 * 
 * This filters OUT:
 * - Unconditional jumps (jmp)
 * - Calls (we don't predict these)
 * - Returns (not predictable with history alone)
 */
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

// ================================================================
// FINALIZATION (called when program exits)
// ================================================================

/*
 * Fini: Write summary statistics and cleanup
 * 
 * Called once when the program exits (or Pin detaches).
 * Writes detailed statistics and closes all files.
 */
VOID Fini(INT32 code, VOID *v) {
    TraceFile.close();
    
    // Close per-branch files
    for (auto& pair : PerBranchFiles) {
        pair.second->close();
        delete pair.second;
    }
    
    // Write summary statistics
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
    
    // Per-branch statistics
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
    
    // Sort branches by execution count
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
    
    // Console output
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

// ================================================================
// MAIN
// ================================================================

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
    // Initialize Pin
    PIN_InitSymbols();
    if (PIN_Init(argc, argv)) {
        return Usage();
    }
    
    // Create output directory
    string mkdirCmd = "mkdir -p " + KnobOutputDir.Value();
    system(mkdirCmd.c_str());
    
    // Open main trace file
    TraceFile.open(KnobOutputFile.Value().c_str());
    if (!TraceFile.is_open()) {
        cerr << "Error: Cannot open output file " << KnobOutputFile.Value() << endl;
        return -1;
    }
    
    // Write CSV header
    TraceFile << "branch_pc,taken";
    for (int i = 0; i < HISTORY_LENGTH; i++) {
        TraceFile << ",hist_" << i;
    }
    TraceFile << endl;
    
    // Open summary file
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
    
    // Register instrumentation callbacks
    IMG_AddInstrumentFunction(ImageLoad, 0);
    INS_AddInstrumentFunction(Instruction, 0);
    PIN_AddFiniFunction(Fini, 0);
    
    // Start the program
    PIN_StartProgram();
    
    return 0;
}