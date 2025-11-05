#include "Simulator.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <tuple>

#include "Common.h"
#include "SystolicOS.h"
#include "SystolicWS.h"
#include "scheduler/Scheduler.h"

namespace fs = std::filesystem;

Simulator::Simulator(SimulationConfig config, bool language_mode)
    : _config(config), _core_cycles(0), _language_mode(language_mode) {
  // Create dram object
  spdlog::info("Simulator Configuration:");
  for (int i=0; i<config.num_cores;i++)
    spdlog::info("[Core {}] Systolic Array Throughput: {} GFLOPS, Spad size: {} KB, Accumulator size: {} KB",
      i, config.max_systolic_flops(i), config.core_config[i].spad_size, config.core_config[i].accum_spad_size);
  spdlog::info("DRAM Bandwidth {} GB/s", config.max_dram_bandwidth());
  _core_period = 1000000 / (config.core_freq);
  _icnt_period = 1000000 / (config.icnt_freq);
  _dram_period = 1000000 / (config.dram_freq);
  _core_time = 0;
  _dram_time = 0;
  _icnt_time = 0;
  char* onnxim_path_env = std::getenv("ONNXIM_HOME");
  std::string onnxim_path = onnxim_path_env != NULL?
  std::string(onnxim_path_env) : std::string("./");
  if (config.dram_type == DramType::SIMPLE) {
    _dram = std::make_unique<SimpleDram>(config);
  } else if (config.dram_type == DramType::RAMULATOR1) {
    std::string ramulator_config = fs::path(onnxim_path)
                                       .append("configs")
                                       .append(config.dram_config_path)
                                       .string();
    spdlog::info("Ramulator config: {}", ramulator_config);
    config.dram_config_path = ramulator_config;
    _dram = std::make_unique<DramRamulator>(config);
  } 
  else if (config.dram_type == DramType::RAMULATOR2) 
  {
    std::string ramulator_config = fs::path(onnxim_path)
                                       .append("configs")
                                       .append(config.dram_config_path)
                                       .string();
    spdlog::info("Ramulator2 config: {}", ramulator_config);
    config.dram_config_path = ramulator_config;
    _dram = std::make_unique<DramRamulator2>(config);
  } 
  else {
    spdlog::error("[Configuration] Invalid DRAM type...!");
    exit(EXIT_FAILURE);
  }

  // Create interconnect object
  if (config.icnt_type == IcntType::SIMPLE) {
    _icnt = std::make_unique<SimpleInterconnect>(config);
  } else if (config.icnt_type == IcntType::BOOKSIM2) {
    _icnt = std::make_unique<Booksim2Interconnect>(config);
  } else {
    spdlog::error("[Configuration] {} Invalid interconnect type...!");
    exit(EXIT_FAILURE);
  }
  _icnt_interval = config.icnt_print_interval;

  // Create core objects
  _cores.resize(config.num_cores);
  _n_cores = config.num_cores;
  _n_memories = config.dram_channels;
  _memory_req_size = config.dram_req_size;
  for (int core_index = 0; core_index < _n_cores; core_index++) {
    _cores[core_index] = Core::create(core_index, config);
  }

  core_turn.resize(config.num_cores);
  for(int i=0; i<config.num_cores; ++i){
    core_turn[i]=0;
  }
  idle_ld_cores.resize(config.num_cores);
  for(int i=0; i<config.num_cores; ++i){
    idle_ld_cores[i]=true;
  }

  // MMScheduler
  request_queue_per_ch.resize(_config.dram_channels);
  active_row_map.resize(_config.dram_channels);
  active_row_order.resize(_config.dram_channels);
  // request_buffer_per_ch.resize(_config.dram_channels);
  mm_rr.resize(_config.dram_channels, 0);
  mm_flush.resize(_config.dram_channels, false);

  //Configure Hardware Scheduler
  
  _scheduler = Scheduler::create(_config, &_core_cycles, &_core_time, this);
  _scheduler->layer_finish.resize(config.num_cores, false);
  
  /* Create heap */
  std::make_heap(_models.begin(), _models.end(), CompareModel());
}

void Simulator::run_simulator() {
  spdlog::info("======Start Simulation=====");
  cycle();
}

void Simulator::handle_model() {
  if(_language_mode) {
    _lang_scheduler->cycle();
    if(_lang_scheduler->can_schedule_model()) {
      _models.push_back(_lang_scheduler->pop_model());
      std::push_heap(_models.begin(), _models.end(), CompareModel());
    }
  }
  while (!_models.empty() && _models.front()->get_request_time() <= _core_time) {
    std::unique_ptr<Model> launch_model = std::move(_models.front());
    std::pop_heap(_models.begin(), _models.end(), CompareModel());
    _models.pop_back();

    launch_model->initialize_model(_weight_table[launch_model->get_name()]);
    launch_model->set_request_time(_core_time);
    spdlog::info("Schedule model: {} at {} us", launch_model->get_name(), _core_time);
    _scheduler->schedule_model(std::move(launch_model), 1);
  }
}

void Simulator::cycle() {
  print_state_once = 0;
  OpStat op_stat;
  ModelStat model_stat;
  uint32_t tile_count;
  bool is_accum_tile;
  while (running()) {
    int model_id = 0;

    set_cycle_mask();
    // Core Cycle
    if (_cycle_mask & CORE_MASK) {
      /* Handle requested model */
      handle_model();

      for (int core_id = 0; core_id < _n_cores; core_id++) {
        _cores[core_id]->layer_num = _scheduler->layer_num;
        _cores[core_id]->layer_num_check = _scheduler->layer_num_check;
        std::unique_ptr<Tile> finished_tile = _cores[core_id]->pop_finished_tile();
        int cold=0;
        if (finished_tile->status == Tile::Status::FINISH) {
          _scheduler->finish_tile(core_id, finished_tile->layer_id);
          if(_scheduler->layer_finish[core_id]){
              for (int i = 0; i < _n_cores; i++){
                _cores[i]->flush_queue();
                cold = _cores[i]->m_i_queue[0]-_scheduler->prev_layer_start.front();
                if(cold<0){
                  cold = _cores[i]->v_i_queue[0]-_scheduler->prev_layer_start.front(); 
                }if(cold<0){
                  cold = 0; 
                }
                spdlog::info("cold = {}", cold);
              }
              _scheduler->prev_layer_start.pop_front();
              _scheduler->layer_finish[core_id] = false;
            }
        }
        // Issue new tile to core
        if (!_scheduler->empty()) {
          is_accum_tile = _scheduler->is_accum_tile(core_id, 0);
          if (_cores[core_id]->can_issue(is_accum_tile)) {
            std::unique_ptr<Tile> tile = _scheduler->get_tile(core_id);
            if (tile->status == Tile::Status::INITIALIZED) {
              _cores[core_id]->issue(std::move(tile));
              _tile_timestamp.push_back(std::chrono::high_resolution_clock::now());
            }
          }
        }
        _cores[core_id]->cycle();
      }
      _core_cycles++;
      _dram->_core_cycle = _core_cycles;
    }

    // DRAM cycle
    if (_cycle_mask & DRAM_MASK) {
      _dram->layer_num = _scheduler->layer_num;
      _dram->layer_num_check = _scheduler->layer_num_check;
      _dram->cycle();
    }
    // Interconnect cycle
    // imp
    if (_cycle_mask & ICNT_MASK) {
      _icnt_cycle++;
      _icnt->layer_num = _scheduler->layer_num;
      _icnt->layer_num_check = _scheduler->layer_num_check;

      for (int core_id = 0; core_id < _n_cores; core_id++) {
        // PUHS core to ICNT. memory request
        if (_cores[core_id]->has_memory_request()) {
          MemoryAccess *front = _cores[core_id]->top_memory_request();
          front->core_id = core_id;
          if (!_icnt->is_full(core_id, front)) {
            // if(_scheduler->layer_num == _scheduler->layer_num_check){
            //   spdlog::info("c2i,core={},buffer_id={},ch={},addr={},op={},cycle={}",core_id, front->buffer_id, get_dest_node(front)-4, front->dram_address, front->operand_id, _core_cycles);
            // }
            _icnt->push(core_id, get_dest_node(front), front);
            _cores[core_id]->pop_memory_request();
            _nr_from_core++;
            if(_scheduler->layer_num==_scheduler->layer_num_check){
              _dram->get_input_weight_req(get_dest_node(front)-_config.num_cores);
            }
          }
        }
        // Push response from ICNT. to Core.
        if (!_icnt->is_empty(core_id)) {
          _cores[core_id]->push_memory_response(_icnt->top(core_id));
          _icnt->pop(core_id);
          _nr_to_core++;
        }
      }

      for (int mem_id = 0; mem_id < _n_memories; mem_id++) {
        // ICNT to memory
        if(mem_id==0){
          int a=0;
        }
        if (!_icnt->is_empty(_n_cores + mem_id) && request_queue_per_ch[mem_id].size()<buffer_size) {
            // if(_scheduler->layer_num == _scheduler->layer_num_check){
            //   spdlog::info("i2d,core={},buffer_id={},ch={},addr={},op={},cycle={}",_icnt->top(_n_cores + mem_id)->core_id, _icnt->top(_n_cores + mem_id)->buffer_id, mem_id, _icnt->top(_n_cores + mem_id)->dram_address, _icnt->top(_n_cores + mem_id)->operand_id,_icnt_cycle);
            // }
          enqueue_request(_icnt->top(_n_cores + mem_id), mem_id);
          _icnt->pop(_n_cores + mem_id);
        }
        mmcycle(mem_id);
        if(_dram->is_available(mem_id)){
          for(int temp=0; temp<16; ++temp){
            if(request_queue_per_ch[mem_id].empty() || mm_flush[mem_id]==0) break;
              if(top_request(mem_id)->dram_address==8669376||top_request(mem_id)->dram_address==8667936||top_request(mem_id)->dram_address==12866112||top_request(mem_id)->dram_address==12862240){
                int a=3;
              }
              _dram->push(mem_id, top_request(mem_id));
              pop_request(mem_id);
              _nr_to_mem++;
          }
        }
        // Pop response to ICNT from dram
        if (!_dram->is_empty(mem_id) &&
            !_icnt->is_full(_n_cores + mem_id, _dram->top(mem_id))) {
          _icnt->push(_n_cores + mem_id, get_dest_node(_dram->top(mem_id)), _dram->top(mem_id));  // imp_1_separated_ch
          _dram->pop(mem_id);
          _nr_from_mem++;
        }
      }
      if (_icnt_interval!=0 && _icnt_cycle % _icnt_interval == 0) {
        spdlog::info("[ICNT] Core->ICNT request {}GB/Sec", ((_memory_req_size*_nr_from_core*(1000/_icnt_period)/_icnt_interval)));
        spdlog::info("[ICNT] Core<-ICNT request {}GB/Sec", ((_memory_req_size*_nr_to_core*(1000/_icnt_period)/_icnt_interval)));
        spdlog::info("[ICNT] ICNT->MEM request {}GB/Sec", ((_memory_req_size*_nr_to_mem*(1000/_icnt_period)/_icnt_interval)));
        spdlog::info("[ICNT] ICNT<-MEM request {}GB/Sec", ((_memory_req_size*_nr_from_mem*(1000/_icnt_period)/_icnt_interval)));
        _nr_from_core=0;
        _nr_to_core=0;
        _nr_to_mem=0;
        _nr_from_mem=0;
      }
      _icnt->cycle();
    }
  }
  spdlog::info("Simulation Finished at {} cycle {} us", _core_cycles, _core_cycles / (_config.core_freq) );
  /* Print simulation stats */
  for (int core_id = 0; core_id < _n_cores; core_id++) {
    _cores[core_id]->print_stats();
  }
  _icnt->print_stats();
  _dram->print_stat();
}

void Simulator::register_model(std::unique_ptr<Model> model) {
  if(_weight_table.find(model->get_name()) == _weight_table.end()) {
    model->initialize_weight(_weight_table[model->get_name()]);
  } 
  _models.push_back(std::move(model));
  std::push_heap(_models.begin(), _models.end(), CompareModel());
}

void Simulator::register_language_model(json info, std::unique_ptr<LanguageModel> model) {
  std::string name = info["name"];
  std::string trace_file = info["trace_file"];
  char* onnxim_path_env = std::getenv("ONNXIM_HOME");
  std::string onnxim_path = onnxim_path_env != NULL?
  std::string(onnxim_path_env) : std::string("./");
  trace_file = fs::path(onnxim_path).append("traces").append(trace_file).string();
  if(_weight_table.find(name) == _weight_table.end()) {
    model->initialize_weight(_weight_table[name]);
  }
  _lang_scheduler = LangScheduler::create(name, trace_file, std::move(model), _config, info);
}

void Simulator::finish_language_model(uint32_t model_id) {
  _lang_scheduler->finish_model(model_id);
}

bool Simulator::running() {
  bool running = false;
  running |= !_models.empty();
  for (auto &core : _cores) {
    running = running || core->running();
  }
  running = running || _icnt->running();
  running = running || _dram->running();
  running = running || !_scheduler->empty();
  if(_language_mode) {
    running = running || _lang_scheduler->busy();
  }
  return running;
}

void Simulator::set_cycle_mask() {
  _cycle_mask = 0x0;
  uint64_t minimum_time = MIN3(_core_time, _dram_time, _icnt_time);
  if (_core_time <= minimum_time) {
    _cycle_mask |= CORE_MASK;
    _core_time += _core_period;
  }
  if (_dram_time <= minimum_time) {
    _cycle_mask |= DRAM_MASK;
    _dram_time += _dram_period;
  }
  if (_icnt_time <= minimum_time) {
    _cycle_mask |= ICNT_MASK;
    _icnt_time += _icnt_period;
  }
}

uint32_t Simulator::get_dest_node(MemoryAccess *access) {
  if (access->request) {
    return _config.num_cores + _dram->get_channel_id(access);
  } else {
    return access->core_id;
  }
}

uint32_t Simulator::get_dest_node(MemoryAccess *access, std::vector<bool> idle_ld_cores, std::vector<uint32_t> &core_turn, uint32_t core_id) {
  uint32_t cur_turn = core_turn[core_id];
  uint32_t lower_2bit = _dram->get_channel_id(access) & 3;
  uint32_t upper_2bit = core_id;
  std::vector<bool> temp_idle_ld_cores = idle_ld_cores;
  temp_idle_ld_cores[core_id] = true;
  uint32_t result_id; 
  bool available = true;

  for(int j=0; j<4; ++j){
    available = _dram->is_available(core_id*4+j);
    if(!available) break;
  }

  if(!available){
    for(int i=1; i<_config.num_cores; ++i){
      int turn = (i+cur_turn)%_config.num_cores;
      if(!temp_idle_ld_cores[turn]) continue;
      for(int j=0; j<4; ++j){
        available = _dram->is_available(turn*4+j);
        if(!available) break;
      }
      if(!available) continue;
      core_turn[core_id] = turn;
      upper_2bit = turn;
      break;
    }
  }
  
  result_id = _config.num_cores + ((upper_2bit & 3)<<2)+(lower_2bit & 3);

  if (access->request) {
    return result_id;
  } else {
    return access->core_id;
  }
}

const double Simulator::get_tile_ops() {
  std::chrono::duration<double> duration = _tile_timestamp.back() - _tile_timestamp.front();
  if (_tile_timestamp.empty())
    return 0.0;
  else
    return _tile_timestamp.size() / duration.count();
}

void Simulator::turn_issue_core(){
  // uint32_t request_line = 16*2*4*4*16/4;
  uint32_t request_line = 8192;
  cur_core_turn = (distribute_cycle/request_line)%_config.num_cores;
  distribute_cycle = (distribute_cycle + 1) % (request_line*_config.num_cores);
}

std::vector<uint32_t> Simulator::get_vecctor(MemoryAccess* request){
  uint32_t _n_ch = _config.dram_channels;
  uint32_t _req_size = _config.dram_req_size;
  uint32_t _tx_log2 = log2(_req_size);
  uint32_t _tx_ch_log2 = log2(_n_ch) + _tx_log2;

  addr_type addr = (request->dram_address>>_tx_ch_log2);
  // uint32_t ch = get_dest_node(request) - _config.num_cores;
  uint32_t ch = get_dest_node(request);
  addr = addr >> slice_vec[0];
  std::vector<uint32_t> vec = {ch,0,0,0, 0}; // ch, pse, bg, bk, row
  for(int i=1; i<slice_vec.size() ; ++i){
    vec[i] = addr & ((1u<<slice_vec[i])-1);
    addr = addr >> slice_vec[i];
  }
  return vec;
}

MemoryAccess* Simulator::top_request(uint32_t mem_id){
  if (request_queue_per_ch[mem_id].empty()) return nullptr;
  return request_queue_per_ch[mem_id].front().req;
}

void Simulator::enqueue_request(MemoryAccess* request, uint32_t ch){
  std::vector<uint32_t> addr_vec = get_vecctor(request);
  RequestEntry req_entry = {request, addr_vec, addr_vec[4], 0, 0};
  if(_scheduler->layer_num==_scheduler->layer_num_check){
    spdlog::info("push:ch={},addr={},queue={},cycle={}",ch,request->dram_address,request_queue_per_ch[ch].size(),_core_cycles);
  }
  // spdlog::info("push:ch={},addr={},{},queue={},cycle={}",ch,request->dram_address,addr_vec, request_queue_per_ch[ch].size(),_core_cycles);

  if(request->dram_address==8669376 || request->dram_address==8667936 || request->dram_address==12866112){
    int a=0;
  }
  std::deque<RequestEntry>& request_queue = request_queue_per_ch[ch];
  
  auto key = std::make_tuple(addr_vec[1], addr_vec[2], addr_vec[3], addr_vec[4]); // pseudo, bg, bk, row
  uint32_t row = addr_vec[4];
  auto iter = active_row_map[ch].find(key);
  if (iter == active_row_map[ch].end()) {
        RowState rs{ {row}, 1, 0 };
        active_row_map[ch][key] = rs;
        active_row_order[ch].push_back(key);
  }else{
    active_row_map[ch][key].cnt++;
  }


  request_queue.push_back(req_entry);
}

void Simulator::mmcycle(uint32_t ch){
  
  std::deque<RequestEntry>& request_queue = request_queue_per_ch[ch];
  for (auto& entry : request_queue) {
    entry.timer += 1;
  }
  std::vector<std::tuple<uint32_t,uint32_t,uint32_t,uint32_t>>& row_order = active_row_order[ch];
  if(request_queue.empty()) return;
  // 각 active row key의 등장 순서대로 stable_partition 수행
  auto key = row_order.front();
  auto begin_it = std::stable_partition(
        request_queue.begin(),
        request_queue.end(),
        [&](const RequestEntry& e) {
            auto k = std::make_tuple(e.addr_vec[1], e.addr_vec[2], e.addr_vec[3], e.addr_vec[4]);
            return k == key;
        }
    );

  if (row_order.size() > 1) {
        auto key_second = row_order[1];
        std::stable_partition(
            begin_it, // 앞의 첫 번째 row 이후부터 시작
            request_queue.end(),
            [&](const RequestEntry& e) {
                auto k = std::make_tuple(e.addr_vec[1], e.addr_vec[2], e.addr_vec[3], e.addr_vec[4]);
                return k == key_second;
            }
        );
    }
  
  if(row_order.size()>_config.num_cores*2){
    // while(!row_order.empty()){
      auto key = row_order.front();
      auto iter = active_row_map[ch].find(key);
      auto count = iter->second.cnt;
      if(ch==0){
        int a= count;
      }

      // if (count == 0) {
      //     row_order.erase(row_order.begin());
      //     if (iter != active_row_map[ch].end())
      //       active_row_map[ch].erase(iter);
      //     continue;
      // }
      mm_flush[ch] = count;
      if(ch==0){
        int a= count;
      }
    // break;
    // }
  }else{
    bool has_write = std::any_of(
      request_queue.begin(),
      request_queue.end(),
      [](const RequestEntry& e) {
          return e.req && e.req->operand_id == 200;
      }
    );
    // if(has_write || request_queue.front().timer>1000){
    if(request_queue.front().timer>1000){
      mm_flush[ch] = request_queue.size();
      if(ch==0){
        int a= request_queue.size();
      }
    } 
  }

  

}

void Simulator::pop_request(uint32_t mem_id){
  std::deque<RequestEntry>& request_queue = request_queue_per_ch[mem_id];
  if(request_queue.empty()) return;
  // RequestEntry entry = std::move(request_queue.front());
  if(mem_id==0){
        int a=3;
      }

  RequestEntry entry = std::move(request_queue.front());
  request_queue.pop_front();
  mm_flush[mem_id]--;
  if(mem_id==0){
    int a=3;
  }

  auto key = std::make_tuple(
        entry.addr_vec[1],
        entry.addr_vec[2],
        entry.addr_vec[3],
        entry.addr_vec[4]
    );
  auto iter = active_row_map[mem_id].find(key);
      if (iter != active_row_map[mem_id].end()) {
          if (iter->second.cnt > 0)
              iter->second.cnt--;

          // 더 이상 남은 요청이 없으면 제거
          if (iter->second.cnt == 0) {
              active_row_map[mem_id].erase(iter);
                if(mem_id==0){
                  int a=3;
                }

              // row_order에서도 제거
              // row_order.erase(row_order.begin());
              auto& row_order = active_row_order[mem_id];
              row_order.erase(
                  std::remove(row_order.begin(), row_order.end(), key),
                  row_order.end()
              );
          }
      }



}

