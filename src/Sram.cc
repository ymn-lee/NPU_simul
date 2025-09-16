#include "Sram.h"
#define NUM_PORTS 3

Sram::Sram(SimulationConfig config, const cycle_type& core_cycle, bool accum, uint32_t core_id)
    : _core_cycle(core_cycle) {
  if (!accum) {
    _size = config.core_config[core_id].spad_size KB / 2;
  } else {
    _size = config.core_config[core_id].accum_spad_size KB / 2;
  }
  _data_width = config.dram_req_size;
  int precision = config.precision;
  _current_size[0] = 0; // multiple of _data_width
  _current_size[1] = 0;
  _accum = accum;
  this->core_id = core_id;
}

bool Sram::check_hit(addr_type address, int buffer_id) { // spad에 올라온 bias, weight 처리
  if (_cache_table[buffer_id].find(address) == _cache_table[buffer_id].end())
    return false;
  _cache_table[buffer_id].at(address).timestamp = _core_cycle;
  return _cache_table[buffer_id].at(address).valid;
}

// imp_5 reuse_spad
bool Sram::check_hit(addr_type address, int buffer_id, bool reuse_input) { // spad에 올라온 input 중복 처리
  if(reuse_input){
    for(int id=0; id<2; ++id){
      auto it = _cache_table[id].find(address);
      if(it !=_cache_table[id].end()){
        _cache_table[id].at(address).timestamp = _core_cycle;
        return _cache_table[id].at(address).valid;   
      }
    }
    return false;   // spad 둘 다 데이터 없으면 false
  }
  
  if (_cache_table[buffer_id].find(address) != _cache_table[buffer_id].end()){ // layer 첫 시작 + c 바뀌는 reuse 못할 때, 처리
    _cache_table[buffer_id].at(address).timestamp = _core_cycle;
    return _cache_table[buffer_id].at(address).valid;
  }
  else return false;
}

bool Sram::check_full(int buffer_id) {
  return _current_size[buffer_id] * _data_width < _size;
}

bool Sram::check_remain(size_t size, int buffer_id) {
  return (_current_size[buffer_id] + size) * _data_width <= _size;
}

bool Sram::check_allocated(addr_type address, int buffer_id) {
  return _cache_table[buffer_id].find(address) != _cache_table[buffer_id].end();
}

void Sram::cycle() {}

void Sram::flush_weight(int buffer_id) {
  auto& table = _cache_table[buffer_id]; 
  size_t freed = 0;                       

  for (auto it = table.begin(); it != table.end(); ) {
      if (!it->second.is_input) {
          freed += it->second.size;
          it = table.erase(it);  
      } else {
          ++it; 
      }
  }

   _current_size[buffer_id] = (_current_size[buffer_id] >= static_cast<int>(freed))
                           ? (_current_size[buffer_id] - static_cast<int>(freed))
                           : 0;
                           
  spdlog::trace("{}SRAM[{}] Flush", _accum? "Acc-": "", buffer_id);
}

void Sram::flush(int buffer_id) {
  _current_size[buffer_id] = 0;
  _cache_table[buffer_id].clear();
  spdlog::trace("{}SRAM[{}] Flush", _accum? "Acc-": "", buffer_id);
}

int Sram::prefetch(addr_type address, int buffer_id, size_t allocated_size, size_t count, bool is_input) { // imp_5 reuse_spad
  if (_cache_table[buffer_id].find(address) == _cache_table[buffer_id].end()) {
    if (!check_remain(allocated_size, buffer_id)) {
      print_all(buffer_id);
      assert(0);
      return 0;
    }
    _current_size[buffer_id] += allocated_size;
  } else if (_cache_table[buffer_id].find(address) !=
                 _cache_table[buffer_id].end() &&
             _accum) {
    assert(_cache_table[buffer_id].at(address).size == allocated_size);
    return 0;
  } else {
    assert(0);
    return 0;
  }
  
  _cache_table[buffer_id][address] = SramEntry{.valid = false,
                                               .size = allocated_size,
                                               .remain_req_count = count,
                                               .is_input = is_input,
                                               .timestamp = _core_cycle};
  return 1;
}

void Sram::fill(addr_type address, int buffer_id) {
  assert(check_allocated(address, buffer_id));
  assert(_cache_table[buffer_id].at(address).remain_req_count > 0 &&
         !_cache_table[buffer_id].at(address).valid);
  _cache_table[buffer_id].at(address).remain_req_count--;
  if (_cache_table[buffer_id].at(address).remain_req_count == 0) {
    _cache_table[buffer_id].at(address).valid = true;
    is_valid[buffer_id] = true;
    spdlog::trace("MAKE valid {} {}F", buffer_id, address);
  }
}

void Sram::fill(addr_type address, addr_type dram_address, int buffer_id, uint32_t operand_id) {
  assert(check_allocated(address, buffer_id));
  assert(_cache_table[buffer_id].at(address).remain_req_count > 0 &&
         !_cache_table[buffer_id].at(address).valid);
  _cache_table[buffer_id].at(address).remain_req_count--;
  if(layer_num==layer_num_check){
      if(address >= ACCUM_SPAD_BASE){
        spdlog::info("[{}]core exec valid {}, {}, {}, remain={}, cycle={}", core_id, buffer_id, address, dram_address, _cache_table[buffer_id].at(address).remain_req_count, _core_cycle);
      }else{
        spdlog::info("[{}]core load valid {}, {}, {}, remain={}, cycle={}", core_id, buffer_id, address, dram_address, _cache_table[buffer_id].at(address).remain_req_count, _core_cycle);
      }
    }
  if (_cache_table[buffer_id].at(address).remain_req_count == 0) {
    _cache_table[buffer_id].at(address).valid = true;
    is_valid[buffer_id] = true; // tile 내부의 inst가 순차적으로 요청되게 traffic 조절
    if(operand_id == 100 && can_issue_second_tile>0) { 
      can_issue_second_tile -- ; // 두 번째 input이 와야 두 번째 tile issue
    }
    // spdlog::info("MAKE valid {}, {}, {}, input={}, cycle={}, cyc={}", core_id, buffer_id, address, operand_id==100?true:false, _core_cycle, _core_cycle+256);
  }
}

void Sram::count_up(addr_type address, int buffer_id) {
  assert(check_allocated(address, buffer_id));
  _cache_table[buffer_id].at(address).remain_req_count++;
  if (_cache_table[buffer_id].at(address).valid) {
    _cache_table[buffer_id].at(address).valid = false;
    spdlog::trace("MAKE valid {} {}F", buffer_id, address);
  }
}

void Sram::print_all(int buffer_id) {
  for (auto& [key, val] : _cache_table[buffer_id]) {
    spdlog::info("0x{:x} : {} B", key, val.size * _data_width);
  }
  spdlog::info("Allocated size: {} B", _current_size[buffer_id]*_data_width);
  spdlog::info("Size: {} B", _size);
}