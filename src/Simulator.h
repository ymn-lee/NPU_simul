#pragma once

#include "Common.h"
#include "Core.h"
#include "Dram.h"
#include "Interconnect.h"
#include "Model.h"
#include "scheduler/Scheduler.h"
#include "scheduler/LanguageScheduler.h"
#include <cstdint>
#include <queue>

#define CORE_MASK 0x1 << 1
#define DRAM_MASK 0x1 << 2
#define ICNT_MASK 0x1 << 3

class Simulator {
 public:
  Simulator(SimulationConfig config, bool language_mode);
  void register_model(std::unique_ptr<Model> model);
  void register_language_model(json info, std::unique_ptr<LanguageModel> model);
  void finish_language_model(uint32_t model_id);
  void run_simulator();
  const double get_tile_ops();
  const size_t get_number_tile() { return _tile_timestamp.size(); }
  std::vector<uint32_t> core_turn;
  std::vector<bool> idle_ld_cores;
  void turn_issue_core();
  uint32_t distribute_cycle = 0;
  uint32_t cur_core_turn = 0;
  uint32_t response_turn = 0;
  uint32_t print_state_once = 0;

  
  
  // void run_offline(std::string model_name, uint32_t sample_count);
  // void run_multistream(std::string model_name, uint32_t sample_count,
  // uint32_t ); void run_server(std::string trace_path);
 private:
  void cycle();
  bool running();
  void set_cycle_mask();
  void handle_model();
  uint32_t get_dest_node(MemoryAccess* access, std::vector<bool> idle_ld_cores, std::vector<uint32_t> &core_turn, uint32_t core_id);
  uint32_t get_dest_node(MemoryAccess* access);
  SimulationConfig _config;
  uint32_t _n_cores;
  uint32_t _n_memories;
  uint32_t _memory_req_size;

  //MMScheduler
  struct RequestEntry{
    MemoryAccess* req;
    std::vector<uint32_t> addr_vec;
    uint32_t row;
    uint32_t cnt;
    uint32_t timer;
  };
  struct RowState{
    std::vector<uint32_t> active_rows;
    uint32_t cnt;
    uint32_t timer; 
  };
  void mmcycle(uint32_t ch);
  void enqueue_request(MemoryAccess* request, uint32_t ch);
  void pop_request(uint32_t mem_id);
  MemoryAccess* top_request(uint32_t mem_id);
  std::vector<uint32_t> get_vecctor(MemoryAccess* request);
  std::vector<uint32_t> slice_vec = {4,1,2,2,15};  // col, pse, bg, bk, row
  std::deque<std::deque<RequestEntry>> request_queue_per_ch;
  // std::deque<std::deque<RequestEntry>> request_buffer_per_ch;
  uint32_t buffer_size = 256;
  uint32_t queue_size = 256;
  uint32_t rows_num = 16*2*4*4*16;
  std::vector<uint32_t> mm_rr;
  std::map<std::tuple<uint32_t, uint32_t, uint32_t, uint32_t>, RowState> active_row_map;

  // std::vector<std::vector<uint32_t>> cor

  // Components
  std::vector<std::unique_ptr<Core>> _cores;
  std::unique_ptr<Interconnect> _icnt;
  std::unique_ptr<Dram> _dram;
  std::unique_ptr<Scheduler> _scheduler;
  
  // period information (ps)
  uint64_t _core_period;
  uint64_t _icnt_period;
  uint64_t _dram_period;
  //
  uint64_t _core_time;
  uint64_t _icnt_time;
  uint64_t _dram_time;

  addr_type _dram_ch_stride_size;

  uint64_t _core_cycles;

  uint32_t _cycle_mask;
  bool _single_run;
  bool _language_mode;
  std::unique_ptr<LangScheduler> _lang_scheduler;

  // Icnt stat
  uint64_t _nr_from_core=0;
  uint64_t _nr_to_core=0;
  uint64_t _nr_from_mem=0;
  uint64_t _nr_to_mem=0;
  cycle_type _icnt_cycle=0;
  uint64_t _icnt_interval=0;

  struct CompareModel {
    bool operator()(const std::unique_ptr<Model>& a, const std::unique_ptr<Model>& b) const {
        return a->get_request_time() > b->get_request_time();
    }
  };
  robin_hood::unordered_map<std::string, 
    std::vector<std::unique_ptr<Tensor>>> _weight_table;
  std::vector<std::unique_ptr<Model>>  _models;
  robin_hood::unordered_map<std::string, std::unique_ptr<Model>> _language_models;
  std::vector<std::chrono::time_point<std::chrono::high_resolution_clock>> _tile_timestamp;

  bool check_defined_model(std::string model_name);
};
