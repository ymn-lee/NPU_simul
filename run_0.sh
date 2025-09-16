PROGRAM="./ONNXim/build/bin/Simulator"

CONFIG="./ONNXim/configs/systolic_ws_128x128_c4_booksim2_tpuv4.json"
MODELS="./ONNXim/example/language_models_llama3-8b.json"
TRACE="input_128_8.csv"
SAVE="1_separated_ch_128_8"

$PROGRAM \
  --config "$CONFIG" \
  --models_list "$MODELS" \
  --mode language \
  --trace_file "$TRACE" \
  --save_name "$SAVE"
