# B200 G2S 完成周期

每个测试点只分配一个新的 TensorMap。第一次测量是该地址第一次进入 TMAU，完成后在同一个 kernel 内立刻用完全相同的地址和内容测第二次。主测试没有 TensorMap prefetch；普通全局读取只把 descriptor 放进 L2。

表格分别显示第一次和第二次，不再把两次平均成一个数字。每个数字来自一轮固定的 640 点扫描：80 个静态 kernel bank 各负责 8 个点，bank 在计时外选择；计时路径没有索引跳转，相邻点静态增加一条比较、一条统一分支和一条不访问数据通路的 PM-event。请求步长为三周期，实际物理间隔由 clock64 记录，原始测量点、地址和指纹保存在同目录压缩 CSV。

测试组：`bulk-tensor__g2s__fresh-map-common`

## base_alignment

|测试项|类型|数据格式|rank|字节数|TensorMap 状态|第一次|第二次|第二次-第一次|
|---|---|---|---:|---:|---|---:|---:|---:|
|bulk_base_g2s_mod128_0|bulk|u8|2|4096|Bulk（没有 TensorMap）|379|370|-9|
|bulk_base_g2s_mod128_112|bulk|u8|2|4096|Bulk（没有 TensorMap）|394|397|3|
|bulk_base_g2s_mod128_16|bulk|u8|2|4096|Bulk（没有 TensorMap）|980|379|-601|
|bulk_base_g2s_mod128_32|bulk|u8|2|4096|Bulk（没有 TensorMap）|388|379|-9|
|bulk_base_g2s_mod128_64|bulk|u8|2|4096|Bulk（没有 TensorMap）|379|374|-5|
|tensor_base_g2s_mod128_0|tensor|u8|2|4096|在 L2，TMAU 第一次使用|863|504|-359|
|tensor_base_g2s_mod128_0|tensor|u8|2|4096|已经提前送入 TMAU|585|505|-80|
|tensor_base_g2s_mod128_112|tensor|u8|2|4096|在 L2，TMAU 第一次使用|922|555|-367|
|tensor_base_g2s_mod128_112|tensor|u8|2|4096|已经提前送入 TMAU|645|553|-92|
|tensor_base_g2s_mod128_16|tensor|u8|2|4096|在 L2，TMAU 第一次使用|1511|566|-945|
|tensor_base_g2s_mod128_16|tensor|u8|2|4096|已经提前送入 TMAU|1149|546|-603|
|tensor_base_g2s_mod128_32|tensor|u8|2|4096|在 L2，TMAU 第一次使用|887|522|-365|
|tensor_base_g2s_mod128_32|tensor|u8|2|4096|已经提前送入 TMAU|629|529|-100|
|tensor_base_g2s_mod128_64|tensor|u8|2|4096|在 L2，TMAU 第一次使用|906|529|-377|
|tensor_base_g2s_mod128_64|tensor|u8|2|4096|已经提前送入 TMAU|619|530|-89|

## bulk_capacity

|测试项|类型|数据格式|rank|字节数|TensorMap 状态|第一次|第二次|第二次-第一次|
|---|---|---|---:|---:|---|---:|---:|---:|
|bulk_capacity_g2s_b1024|bulk|u8|2|1024|Bulk（没有 TensorMap）|343|346|3|
|bulk_capacity_g2s_b128|bulk|u8|2|128|Bulk（没有 TensorMap）|313|312|-1|
|bulk_capacity_g2s_b16|bulk|u8|2|16|Bulk（没有 TensorMap）|298|312|14|
|bulk_capacity_g2s_b16384|bulk|u8|2|16384|Bulk（没有 TensorMap）|476|476|0|
|bulk_capacity_g2s_b2048|bulk|u8|2|2048|Bulk（没有 TensorMap）|353|360|7|
|bulk_capacity_g2s_b256|bulk|u8|2|256|Bulk（没有 TensorMap）|340|340|0|
|bulk_capacity_g2s_b32|bulk|u8|2|32|Bulk（没有 TensorMap）|298|298|0|
|bulk_capacity_g2s_b32768|bulk|u8|2|32768|Bulk（没有 TensorMap）|610|600|-10|
|bulk_capacity_g2s_b4096|bulk|u8|2|4096|Bulk（没有 TensorMap）|371|368|-3|
|bulk_capacity_g2s_b512|bulk|u8|2|512|Bulk（没有 TensorMap）|355|355|0|
|bulk_capacity_g2s_b64|bulk|u8|2|64|Bulk（没有 TensorMap）|313|298|-15|
|bulk_capacity_g2s_b8192|bulk|u8|2|8192|Bulk（没有 TensorMap）|408|409|1|

## dtype

|测试项|类型|数据格式|rank|字节数|TensorMap 状态|第一次|第二次|第二次-第一次|
|---|---|---|---:|---:|---|---:|---:|---:|
|dtype_g2s_bf16_b128|tensor|bf16|2|128|在 L2，TMAU 第一次使用|810|449|-361|
|dtype_g2s_bf16_b128|tensor|bf16|2|128|已经提前送入 TMAU|534|442|-92|
|dtype_g2s_bf16_b4096|tensor|bf16|2|4096|在 L2，TMAU 第一次使用|850|507|-343|
|dtype_g2s_bf16_b4096|tensor|bf16|2|4096|已经提前送入 TMAU|587|503|-84|
|dtype_g2s_f16_b128|tensor|f16|2|128|在 L2，TMAU 第一次使用|850|476|-374|
|dtype_g2s_f16_b128|tensor|f16|2|128|已经提前送入 TMAU|566|460|-106|
|dtype_g2s_f16_b4096|tensor|f16|2|4096|在 L2，TMAU 第一次使用|863|501|-362|
|dtype_g2s_f16_b4096|tensor|f16|2|4096|已经提前送入 TMAU|587|482|-105|
|dtype_g2s_f32_b128|tensor|f32|2|128|在 L2，TMAU 第一次使用|789|440|-349|
|dtype_g2s_f32_b128|tensor|f32|2|128|已经提前送入 TMAU|563|465|-98|
|dtype_g2s_f32_b4096|tensor|f32|2|4096|在 L2，TMAU 第一次使用|867|496|-371|
|dtype_g2s_f32_b4096|tensor|f32|2|4096|已经提前送入 TMAU|588|503|-85|
|dtype_g2s_f32_ftz_b128|tensor|f32_ftz|2|128|在 L2，TMAU 第一次使用|789|449|-340|
|dtype_g2s_f32_ftz_b128|tensor|f32_ftz|2|128|已经提前送入 TMAU|529|434|-95|
|dtype_g2s_f32_ftz_b4096|tensor|f32_ftz|2|4096|在 L2，TMAU 第一次使用|858|504|-354|
|dtype_g2s_f32_ftz_b4096|tensor|f32_ftz|2|4096|已经提前送入 TMAU|590|490|-100|
|dtype_g2s_f64_b128|tensor|f64|2|128|在 L2，TMAU 第一次使用|802|459|-343|
|dtype_g2s_f64_b128|tensor|f64|2|128|已经提前送入 TMAU|523|442|-81|
|dtype_g2s_f64_b4096|tensor|f64|2|4096|在 L2，TMAU 第一次使用|869|498|-371|
|dtype_g2s_f64_b4096|tensor|f64|2|4096|已经提前送入 TMAU|601|514|-87|
|dtype_g2s_s32_b128|tensor|s32|2|128|在 L2，TMAU 第一次使用|806|468|-338|
|dtype_g2s_s32_b128|tensor|s32|2|128|已经提前送入 TMAU|529|446|-83|
|dtype_g2s_s32_b4096|tensor|s32|2|4096|在 L2，TMAU 第一次使用|862|501|-361|
|dtype_g2s_s32_b4096|tensor|s32|2|4096|已经提前送入 TMAU|595|496|-99|
|dtype_g2s_s64_b128|tensor|s64|2|128|在 L2，TMAU 第一次使用|826|476|-350|
|dtype_g2s_s64_b128|tensor|s64|2|128|已经提前送入 TMAU|568|487|-81|
|dtype_g2s_s64_b4096|tensor|s64|2|4096|在 L2，TMAU 第一次使用|861|501|-360|
|dtype_g2s_s64_b4096|tensor|s64|2|4096|已经提前送入 TMAU|601|496|-105|
|dtype_g2s_tf32_b128|tensor|tf32|2|128|在 L2，TMAU 第一次使用|826|468|-358|
|dtype_g2s_tf32_b128|tensor|tf32|2|128|已经提前送入 TMAU|563|456|-107|
|dtype_g2s_tf32_b4096|tensor|tf32|2|4096|在 L2，TMAU 第一次使用|869|507|-362|
|dtype_g2s_tf32_b4096|tensor|tf32|2|4096|已经提前送入 TMAU|590|503|-87|
|dtype_g2s_tf32_ftz_b128|tensor|tf32_ftz|2|128|在 L2，TMAU 第一次使用|850|478|-372|
|dtype_g2s_tf32_ftz_b128|tensor|tf32_ftz|2|128|已经提前送入 TMAU|563|479|-84|
|dtype_g2s_tf32_ftz_b4096|tensor|tf32_ftz|2|4096|在 L2，TMAU 第一次使用|869|498|-371|
|dtype_g2s_tf32_ftz_b4096|tensor|tf32_ftz|2|4096|已经提前送入 TMAU|587|505|-82|
|dtype_g2s_u16_b128|tensor|u16|2|128|在 L2，TMAU 第一次使用|813|464|-349|
|dtype_g2s_u16_b128|tensor|u16|2|128|已经提前送入 TMAU|563|456|-107|
|dtype_g2s_u16_b4096|tensor|u16|2|4096|在 L2，TMAU 第一次使用|870|504|-366|
|dtype_g2s_u16_b4096|tensor|u16|2|4096|已经提前送入 TMAU|601|506|-95|
|dtype_g2s_u32_b128|tensor|u32|2|128|在 L2，TMAU 第一次使用|802|440|-362|
|dtype_g2s_u32_b128|tensor|u32|2|128|已经提前送入 TMAU|538|446|-92|
|dtype_g2s_u32_b4096|tensor|u32|2|4096|在 L2，TMAU 第一次使用|863|496|-367|
|dtype_g2s_u32_b4096|tensor|u32|2|4096|已经提前送入 TMAU|605|496|-109|
|dtype_g2s_u64_b128|tensor|u64|2|128|在 L2，TMAU 第一次使用|834|462|-372|
|dtype_g2s_u64_b128|tensor|u64|2|128|已经提前送入 TMAU|534|442|-92|
|dtype_g2s_u64_b4096|tensor|u64|2|4096|在 L2，TMAU 第一次使用|863|513|-350|
|dtype_g2s_u64_b4096|tensor|u64|2|4096|已经提前送入 TMAU|597|503|-94|
|dtype_g2s_u8_b128|tensor|u8|2|128|在 L2，TMAU 第一次使用|813|444|-369|
|dtype_g2s_u8_b128|tensor|u8|2|128|已经提前送入 TMAU|570|479|-91|
|dtype_g2s_u8_b4096|tensor|u8|2|4096|在 L2，TMAU 第一次使用|845|507|-338|
|dtype_g2s_u8_b4096|tensor|u8|2|4096|已经提前送入 TMAU|601|506|-95|

## interleave

|测试项|类型|数据格式|rank|字节数|TensorMap 状态|第一次|第二次|第二次-第一次|
|---|---|---|---:|---:|---|---:|---:|---:|
|interleave16_g2s_b1024_int|tensor|u16|3|1024|在 L2，TMAU 第一次使用|847|475|-372|
|interleave16_g2s_b1024_int|tensor|u16|3|1024|已经提前送入 TMAU|577|469|-108|
|interleave16_g2s_b1024_plain|tensor|u16|3|1024|在 L2，TMAU 第一次使用|882|539|-343|
|interleave16_g2s_b1024_plain|tensor|u16|3|1024|已经提前送入 TMAU|635|533|-102|
|interleave16_g2s_b16384_int|tensor|u16|3|16384|在 L2，TMAU 第一次使用|980|620|-360|
|interleave16_g2s_b16384_int|tensor|u16|3|16384|已经提前送入 TMAU|716|611|-105|
|interleave16_g2s_b16384_plain|tensor|u16|3|16384|在 L2，TMAU 第一次使用|1850|1490|-360|
|interleave16_g2s_b16384_plain|tensor|u16|3|16384|已经提前送入 TMAU|1577|1493|-84|
|interleave16_g2s_b4096_int|tensor|u16|3|4096|在 L2，TMAU 第一次使用|899|519|-380|
|interleave16_g2s_b4096_int|tensor|u16|3|4096|已经提前送入 TMAU|614|510|-104|
|interleave16_g2s_b4096_plain|tensor|u16|3|4096|在 L2，TMAU 第一次使用|1083|722|-361|
|interleave16_g2s_b4096_plain|tensor|u16|3|4096|已经提前送入 TMAU|819|720|-99|

## oob

|测试项|类型|数据格式|rank|字节数|TensorMap 状态|第一次|第二次|第二次-第一次|
|---|---|---|---:|---:|---|---:|---:|---:|
|oob_g2s_left100|tensor|u8|2|4096|在 L2，TMAU 第一次使用|869|501|-368|
|oob_g2s_left100|tensor|u8|2|4096|已经提前送入 TMAU|595|496|-99|
|oob_g2s_left12p5|tensor|u8|2|4096|在 L2，TMAU 第一次使用|906|553|-353|
|oob_g2s_left12p5|tensor|u8|2|4096|已经提前送入 TMAU|649|553|-96|
|oob_g2s_left25|tensor|u8|2|4096|在 L2，TMAU 第一次使用|874|518|-356|
|oob_g2s_left25|tensor|u8|2|4096|已经提前送入 TMAU|611|530|-81|
|oob_g2s_left50|tensor|u8|2|4096|在 L2，TMAU 第一次使用|893|535|-358|
|oob_g2s_left50|tensor|u8|2|4096|已经提前送入 TMAU|621|529|-92|
|oob_g2s_outer75|tensor|u8|2|4096|在 L2，TMAU 第一次使用|867|492|-375|
|oob_g2s_outer75|tensor|u8|2|4096|已经提前送入 TMAU|590|496|-94|
|oob_g2s_plain|tensor|u8|2|4096|在 L2，TMAU 第一次使用|863|504|-359|
|oob_g2s_plain|tensor|u8|2|4096|已经提前送入 TMAU|597|496|-101|
|oob_g2s_right25|tensor|u8|2|4096|在 L2，TMAU 第一次使用|878|525|-353|
|oob_g2s_right25|tensor|u8|2|4096|已经提前送入 TMAU|597|510|-87|

## oob_nan

|测试项|类型|数据格式|rank|字节数|TensorMap 状态|第一次|第二次|第二次-第一次|
|---|---|---|---:|---:|---|---:|---:|---:|
|oob_nan_bf16_nan|tensor|bf16|2|4096|在 L2，TMAU 第一次使用|906|525|-381|
|oob_nan_bf16_nan|tensor|bf16|2|4096|已经提前送入 TMAU|621|522|-99|
|oob_nan_bf16_zero|tensor|bf16|2|4096|在 L2，TMAU 第一次使用|902|531|-371|
|oob_nan_bf16_zero|tensor|bf16|2|4096|已经提前送入 TMAU|625|527|-98|
|oob_nan_f16_nan|tensor|f16|2|4096|在 L2，TMAU 第一次使用|891|528|-363|
|oob_nan_f16_nan|tensor|f16|2|4096|已经提前送入 TMAU|614|522|-92|
|oob_nan_f16_zero|tensor|f16|2|4096|在 L2，TMAU 第一次使用|882|528|-354|
|oob_nan_f16_zero|tensor|f16|2|4096|已经提前送入 TMAU|632|522|-110|
|oob_nan_f32_ftz_nan|tensor|f32_ftz|2|4096|在 L2，TMAU 第一次使用|887|518|-369|
|oob_nan_f32_ftz_nan|tensor|f32_ftz|2|4096|已经提前送入 TMAU|614|529|-85|
|oob_nan_f32_ftz_zero|tensor|f32_ftz|2|4096|在 L2，TMAU 第一次使用|882|537|-345|
|oob_nan_f32_ftz_zero|tensor|f32_ftz|2|4096|已经提前送入 TMAU|621|540|-81|
|oob_nan_f32_nan|tensor|f32|2|4096|在 L2，TMAU 第一次使用|882|528|-354|
|oob_nan_f32_nan|tensor|f32|2|4096|已经提前送入 TMAU|625|534|-91|
|oob_nan_f32_zero|tensor|f32|2|4096|在 L2，TMAU 第一次使用|874|525|-349|
|oob_nan_f32_zero|tensor|f32|2|4096|已经提前送入 TMAU|629|534|-95|
|oob_nan_f64_nan|tensor|f64|2|4096|在 L2，TMAU 第一次使用|885|535|-350|
|oob_nan_f64_nan|tensor|f64|2|4096|已经提前送入 TMAU|625|530|-95|
|oob_nan_f64_zero|tensor|f64|2|4096|在 L2，TMAU 第一次使用|902|528|-374|
|oob_nan_f64_zero|tensor|f64|2|4096|已经提前送入 TMAU|614|534|-80|
|oob_nan_tf32_ftz_nan|tensor|tf32_ftz|2|4096|在 L2，TMAU 第一次使用|887|531|-356|
|oob_nan_tf32_ftz_nan|tensor|tf32_ftz|2|4096|已经提前送入 TMAU|625|534|-91|
|oob_nan_tf32_ftz_zero|tensor|tf32_ftz|2|4096|在 L2，TMAU 第一次使用|887|538|-349|
|oob_nan_tf32_ftz_zero|tensor|tf32_ftz|2|4096|已经提前送入 TMAU|614|530|-84|
|oob_nan_tf32_nan|tensor|tf32|2|4096|在 L2，TMAU 第一次使用|902|535|-367|
|oob_nan_tf32_nan|tensor|tf32|2|4096|已经提前送入 TMAU|625|534|-91|
|oob_nan_tf32_zero|tensor|tf32|2|4096|在 L2，TMAU 第一次使用|893|529|-364|
|oob_nan_tf32_zero|tensor|tf32|2|4096|已经提前送入 TMAU|625|530|-95|

## rank

|测试项|类型|数据格式|rank|字节数|TensorMap 状态|第一次|第二次|第二次-第一次|
|---|---|---|---:|---:|---|---:|---:|---:|
|rank_g2s_r1_b128|tensor|u8|1|128|在 L2，TMAU 第一次使用|815|464|-351|
|rank_g2s_r1_b128|tensor|u8|1|128|已经提前送入 TMAU|547|460|-87|
|rank_g2s_r1_b256|tensor|u8|1|256|在 L2，TMAU 第一次使用|826|435|-391|
|rank_g2s_r1_b256|tensor|u8|1|256|已经提前送入 TMAU|547|442|-105|
|rank_g2s_r2_b128|tensor|u8|2|128|在 L2，TMAU 第一次使用|839|430|-409|
|rank_g2s_r2_b128|tensor|u8|2|128|已经提前送入 TMAU|553|438|-115|
|rank_g2s_r2_b256|tensor|u8|2|256|在 L2，TMAU 第一次使用|802|440|-362|
|rank_g2s_r2_b256|tensor|u8|2|256|已经提前送入 TMAU|538|458|-80|
|rank_g2s_r2_b32768|tensor|u8|2|32768|在 L2，TMAU 第一次使用|1101|734|-367|
|rank_g2s_r2_b32768|tensor|u8|2|32768|已经提前送入 TMAU|830|734|-96|
|rank_g2s_r2_b4096|tensor|u8|2|4096|在 L2，TMAU 第一次使用|869|498|-371|
|rank_g2s_r2_b4096|tensor|u8|2|4096|已经提前送入 TMAU|590|496|-94|
|rank_g2s_r3_b128|tensor|u8|3|128|在 L2，TMAU 第一次使用|803|469|-334|
|rank_g2s_r3_b128|tensor|u8|3|128|已经提前送入 TMAU|572|465|-107|
|rank_g2s_r3_b256|tensor|u8|3|256|在 L2，TMAU 第一次使用|847|481|-366|
|rank_g2s_r3_b256|tensor|u8|3|256|已经提前送入 TMAU|579|486|-93|
|rank_g2s_r3_b32768|tensor|u8|3|32768|在 L2，TMAU 第一次使用|1120|758|-362|
|rank_g2s_r3_b32768|tensor|u8|3|32768|已经提前送入 TMAU|856|751|-105|
|rank_g2s_r3_b4096|tensor|u8|3|4096|在 L2，TMAU 第一次使用|919|548|-371|
|rank_g2s_r3_b4096|tensor|u8|3|4096|已经提前送入 TMAU|611|546|-65|
|rank_g2s_r4_b128|tensor|u8|4|128|在 L2，TMAU 第一次使用|856|444|-412|
|rank_g2s_r4_b128|tensor|u8|4|128|已经提前送入 TMAU|569|453|-116|
|rank_g2s_r4_b256|tensor|u8|4|256|在 L2，TMAU 第一次使用|876|495|-381|
|rank_g2s_r4_b256|tensor|u8|4|256|已经提前送入 TMAU|604|495|-109|
|rank_g2s_r4_b32768|tensor|u8|4|32768|在 L2，TMAU 第一次使用|1343|987|-356|
|rank_g2s_r4_b32768|tensor|u8|4|32768|已经提前送入 TMAU|1108|1020|-88|
|rank_g2s_r4_b4096|tensor|u8|4|4096|在 L2，TMAU 第一次使用|959|615|-344|
|rank_g2s_r4_b4096|tensor|u8|4|4096|已经提前送入 TMAU|704|604|-100|
|rank_g2s_r5_b128|tensor|u8|5|128|在 L2，TMAU 第一次使用|865|494|-371|
|rank_g2s_r5_b128|tensor|u8|5|128|已经提前送入 TMAU|540|441|-99|
|rank_g2s_r5_b256|tensor|u8|5|256|在 L2，TMAU 第一次使用|838|505|-333|
|rank_g2s_r5_b256|tensor|u8|5|256|已经提前送入 TMAU|558|470|-88|
|rank_g2s_r5_b32768|tensor|u8|5|32768|在 L2，TMAU 第一次使用|1883|1492|-391|
|rank_g2s_r5_b32768|tensor|u8|5|32768|已经提前送入 TMAU|1607|1487|-120|
|rank_g2s_r5_b4096|tensor|u8|5|4096|在 L2，TMAU 第一次使用|1089|724|-365|
|rank_g2s_r5_b4096|tensor|u8|5|4096|已经提前送入 TMAU|815|723|-92|

## stride

|测试项|类型|数据格式|rank|字节数|TensorMap 状态|第一次|第二次|第二次-第一次|
|---|---|---|---:|---:|---|---:|---:|---:|
|stride_g2s_s128|tensor|u8|2|4096|在 L2，TMAU 第一次使用|863|496|-367|
|stride_g2s_s128|tensor|u8|2|4096|已经提前送入 TMAU|587|506|-81|
|stride_g2s_s144|tensor|u8|2|4096|在 L2，TMAU 第一次使用|893|542|-351|
|stride_g2s_s144|tensor|u8|2|4096|已经提前送入 TMAU|625|534|-91|
|stride_g2s_s256|tensor|u8|2|4096|在 L2，TMAU 第一次使用|869|507|-362|
|stride_g2s_s256|tensor|u8|2|4096|已经提前送入 TMAU|601|506|-95|
|stride_g2s_s512|tensor|u8|2|4096|在 L2，TMAU 第一次使用|867|507|-360|
|stride_g2s_s512|tensor|u8|2|4096|已经提前送入 TMAU|601|517|-84|

## subbox

|测试项|类型|数据格式|rank|字节数|TensorMap 状态|第一次|第二次|第二次-第一次|
|---|---|---|---:|---:|---|---:|---:|---:|
|subbox_g2s_large2d_offset|tensor|u8|2|4096|在 L2，TMAU 第一次使用|887|537|-350|
|subbox_g2s_large2d_offset|tensor|u8|2|4096|已经提前送入 TMAU|625|534|-91|
|subbox_g2s_large2d_plain|tensor|u8|2|4096|在 L2，TMAU 第一次使用|863|498|-365|
|subbox_g2s_large2d_plain|tensor|u8|2|4096|已经提前送入 TMAU|590|506|-84|
|subbox_g2s_large3d_offset|tensor|u8|3|4096|在 L2，TMAU 第一次使用|915|559|-356|
|subbox_g2s_large3d_offset|tensor|u8|3|4096|已经提前送入 TMAU|649|557|-92|
|subbox_g2s_large3d_plain|tensor|u8|3|4096|在 L2，TMAU 第一次使用|904|539|-365|
|subbox_g2s_large3d_plain|tensor|u8|3|4096|已经提前送入 TMAU|635|543|-92|
|subbox_g2s_small2d_offset|tensor|u8|2|128|在 L2，TMAU 第一次使用|813|462|-351|
|subbox_g2s_small2d_offset|tensor|u8|2|128|已经提前送入 TMAU|563|479|-84|
|subbox_g2s_small2d_plain|tensor|u8|2|128|在 L2，TMAU 第一次使用|802|435|-367|
|subbox_g2s_small2d_plain|tensor|u8|2|128|已经提前送入 TMAU|523|431|-92|

## swizzle

|测试项|类型|数据格式|rank|字节数|TensorMap 状态|第一次|第二次|第二次-第一次|
|---|---|---|---:|---:|---|---:|---:|---:|
|swizzle128_g2s_b1024_plain|tensor|u8|2|1024|在 L2，TMAU 第一次使用|810|473|-337|
|swizzle128_g2s_b1024_plain|tensor|u8|2|1024|已经提前送入 TMAU|560|462|-98|
|swizzle128_g2s_b1024_swz|tensor|u8|2|1024|在 L2，TMAU 第一次使用|802|487|-315|
|swizzle128_g2s_b1024_swz|tensor|u8|2|1024|已经提前送入 TMAU|544|474|-70|
|swizzle128_g2s_b16384_plain|tensor|u8|3|16384|在 L2，TMAU 第一次使用|991|614|-377|
|swizzle128_g2s_b16384_plain|tensor|u8|3|16384|已经提前送入 TMAU|721|629|-92|
|swizzle128_g2s_b16384_swz|tensor|u8|3|16384|在 L2，TMAU 第一次使用|998|624|-374|
|swizzle128_g2s_b16384_swz|tensor|u8|3|16384|已经提前送入 TMAU|716|624|-92|
|swizzle128_g2s_b4096_plain|tensor|u8|3|4096|在 L2，TMAU 第一次使用|910|515|-395|
|swizzle128_g2s_b4096_plain|tensor|u8|3|4096|已经提前送入 TMAU|611|519|-92|
|swizzle128_g2s_b4096_swz|tensor|u8|3|4096|在 L2，TMAU 第一次使用|884|524|-360|
|swizzle128_g2s_b4096_swz|tensor|u8|3|4096|已经提前送入 TMAU|607|515|-92|
|swizzle32_g2s_b1024_plain|tensor|u8|2|1024|在 L2，TMAU 第一次使用|854|473|-381|
|swizzle32_g2s_b1024_plain|tensor|u8|2|1024|已经提前送入 TMAU|544|479|-65|
|swizzle32_g2s_b1024_swz|tensor|u8|2|1024|在 L2，TMAU 第一次使用|850|459|-391|
|swizzle32_g2s_b1024_swz|tensor|u8|2|1024|已经提前送入 TMAU|566|490|-76|
|swizzle32_g2s_b16384_plain|tensor|u8|3|16384|在 L2，TMAU 第一次使用|1326|979|-347|
|swizzle32_g2s_b16384_plain|tensor|u8|3|16384|已经提前送入 TMAU|1067|967|-100|
|swizzle32_g2s_b16384_swz|tensor|u8|3|16384|在 L2，TMAU 第一次使用|1336|995|-341|
|swizzle32_g2s_b16384_swz|tensor|u8|3|16384|已经提前送入 TMAU|1087|989|-98|
|swizzle32_g2s_b4096_plain|tensor|u8|3|4096|在 L2，TMAU 第一次使用|971|583|-388|
|swizzle32_g2s_b4096_plain|tensor|u8|3|4096|已经提前送入 TMAU|683|596|-87|
|swizzle32_g2s_b4096_swz|tensor|u8|3|4096|在 L2，TMAU 第一次使用|971|590|-381|
|swizzle32_g2s_b4096_swz|tensor|u8|3|4096|已经提前送入 TMAU|697|605|-92|
|swizzle64_g2s_b1024_plain|tensor|u8|2|1024|在 L2，TMAU 第一次使用|815|476|-339|
|swizzle64_g2s_b1024_plain|tensor|u8|2|1024|已经提前送入 TMAU|560|458|-102|
|swizzle64_g2s_b1024_swz|tensor|u8|2|1024|在 L2，TMAU 第一次使用|850|478|-372|
|swizzle64_g2s_b1024_swz|tensor|u8|2|1024|已经提前送入 TMAU|560|465|-95|
|swizzle64_g2s_b16384_plain|tensor|u8|3|16384|在 L2，TMAU 第一次使用|1102|738|-364|
|swizzle64_g2s_b16384_plain|tensor|u8|3|16384|已经提前送入 TMAU|833|751|-82|
|swizzle64_g2s_b16384_swz|tensor|u8|3|16384|在 L2，TMAU 第一次使用|1102|746|-356|
|swizzle64_g2s_b16384_swz|tensor|u8|3|16384|已经提前送入 TMAU|836|744|-92|
|swizzle64_g2s_b4096_plain|tensor|u8|3|4096|在 L2，TMAU 第一次使用|899|546|-353|
|swizzle64_g2s_b4096_plain|tensor|u8|3|4096|已经提前送入 TMAU|627|543|-84|
|swizzle64_g2s_b4096_swz|tensor|u8|3|4096|在 L2，TMAU 第一次使用|908|539|-369|
|swizzle64_g2s_b4096_swz|tensor|u8|3|4096|已经提前送入 TMAU|635|543|-92|

## tensor_capacity

|测试项|类型|数据格式|rank|字节数|TensorMap 状态|第一次|第二次|第二次-第一次|
|---|---|---|---:|---:|---|---:|---:|---:|
|tensor_capacity_g2s_b1024|tensor|u8|2|1024|在 L2，TMAU 第一次使用|810|473|-337|
|tensor_capacity_g2s_b1024|tensor|u8|2|1024|已经提前送入 TMAU|529|482|-47|
|tensor_capacity_g2s_b128|tensor|u8|2|128|在 L2，TMAU 第一次使用|796|459|-337|
|tensor_capacity_g2s_b128|tensor|u8|2|128|已经提前送入 TMAU|568|484|-84|
|tensor_capacity_g2s_b16384|tensor|u8|2|16384|在 L2，TMAU 第一次使用|963|597|-366|
|tensor_capacity_g2s_b16384|tensor|u8|2|16384|已经提前送入 TMAU|693|594|-99|
|tensor_capacity_g2s_b2048|tensor|u8|2|2048|在 L2，TMAU 第一次使用|850|487|-363|
|tensor_capacity_g2s_b2048|tensor|u8|2|2048|已经提前送入 TMAU|581|474|-107|
|tensor_capacity_g2s_b256|tensor|u8|2|256|在 L2，TMAU 第一次使用|796|462|-334|
|tensor_capacity_g2s_b256|tensor|u8|2|256|已经提前送入 TMAU|544|462|-82|
|tensor_capacity_g2s_b32768|tensor|u8|2|32768|在 L2，TMAU 第一次使用|1085|721|-364|
|tensor_capacity_g2s_b32768|tensor|u8|2|32768|已经提前送入 TMAU|824|732|-92|
|tensor_capacity_g2s_b4096|tensor|u8|2|4096|在 L2，TMAU 第一次使用|850|496|-354|
|tensor_capacity_g2s_b4096|tensor|u8|2|4096|已经提前送入 TMAU|587|484|-103|
|tensor_capacity_g2s_b512|tensor|u8|2|512|在 L2，TMAU 第一次使用|850|476|-374|
|tensor_capacity_g2s_b512|tensor|u8|2|512|已经提前送入 TMAU|544|474|-70|
|tensor_capacity_g2s_b8192|tensor|u8|2|8192|在 L2，TMAU 第一次使用|906|535|-371|
|tensor_capacity_g2s_b8192|tensor|u8|2|8192|已经提前送入 TMAU|621|534|-87|

## 怎样理解两种 Tensor 结果

“在 L2，TMAU 第一次使用”的第一列包含从 L2 取回 TensorMap并交给 TMAU 处理的时间；第二列是同一个 TensorMap 的立即复用。“已经提前送入 TMAU”是独立控制项，第一次发令前显式 prefetch，第二次仍是相同地址的热复用。
