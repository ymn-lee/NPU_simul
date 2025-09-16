#pragma once
#include "Common.h"

class Sram {
 public:
  Sram(SimulationConfig config, const cycle_type& core_cycle, bool accum, uint32_t core_id);

  bool check_hit(addr_type address, int buffer_id);
  bool check_hit(addr_type address, int buffer_id, bool reuse_input);
  bool check_full(int buffer_id);
  bool check_remain(size_t size, int buffer_id);
  bool check_allocated(addr_type address, int buffer_id);
  std::array<bool,2> is_valid={true,true};
  std::array<bool,2> has_input={false,false};
  uint32_t core_id;

  void cycle();
  void flush(int buffer_id);
  void flush_weight(int buffer_id);
  int prefetch(addr_type address, int buffer_id, size_t allocated_size, size_t count, bool is_input);
  void count_up(addr_type, int buffer_id);
  void fill(addr_type address, int buffer_id);
  void fill(addr_type address, addr_type dram_address, int buffer_id, uint32_t operand_id);
  int get_size() { return _size; }
  int get_current_size(int buffer_id) { return _current_size[buffer_id]; }
  void print_all(int buffer_id);
  uint32_t can_issue_second_tile = 2;
  uint32_t layer_num;
  uint32_t layer_num_check;

 private:
  struct SramEntry {
    bool valid;
    addr_type address;
    size_t size;
    size_t remain_req_count;
    bool is_input;
    cycle_type timestamp;
  };

  int _size;
  int _data_width;
  int _current_size[2];
  bool _accum;

  const cycle_type& _core_cycle;

  robin_hood::unordered_map<addr_type, SramEntry> _cache_table[2];
};
