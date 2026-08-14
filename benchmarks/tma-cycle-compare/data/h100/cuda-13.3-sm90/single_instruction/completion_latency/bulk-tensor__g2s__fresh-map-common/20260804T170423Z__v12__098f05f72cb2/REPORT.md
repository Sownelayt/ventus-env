# H100 G2S 完成周期

每个测试点只分配一个新的 TensorMap。第一次测量是该地址第一次进入 TMAU，完成后在同一个 kernel 内立刻用完全相同的地址和内容测第二次。主测试没有 TensorMap prefetch；普通全局读取只把 descriptor 放进 L2。

表格分别显示第一次和第二次，不再把两次平均成一个数字。每个数字来自一轮固定的 640 点扫描：80 个静态 kernel bank 各负责 8 个点，bank 在计时外选择；计时路径没有索引跳转，相邻点静态增加一条比较、一条统一分支和一条不访问数据通路的 PM-event。请求步长为三周期，实际物理间隔由 clock64 记录，原始测量点、地址和指纹保存在同目录压缩 CSV。

测试组：`bulk-tensor__g2s__fresh-map-common`

## base_alignment

|测试项|类型|数据格式|rank|字节数|TensorMap 状态|第一次|第二次|第二次-第一次|
|---|---|---|---:|---:|---|---:|---:|---:|
|bulk_base_g2s_mod128_0|bulk|u8|2|4096|Bulk（没有 TensorMap）|306|306|0|
|bulk_base_g2s_mod128_112|bulk|u8|2|4096|Bulk（没有 TensorMap）|376|369|-7|
|bulk_base_g2s_mod128_16|bulk|u8|2|4096|Bulk（没有 TensorMap）|784|369|-415|
|bulk_base_g2s_mod128_32|bulk|u8|2|4096|Bulk（没有 TensorMap）|335|335|0|
|bulk_base_g2s_mod128_64|bulk|u8|2|4096|Bulk（没有 TensorMap）|325|326|1|
|tensor_base_g2s_mod128_0|tensor|u8|2|4096|在 L2，TMAU 第一次使用|763|439|-324|
|tensor_base_g2s_mod128_0|tensor|u8|2|4096|已经提前送入 TMAU|510|446|-64|
|tensor_base_g2s_mod128_112|tensor|u8|2|4096|在 L2，TMAU 第一次使用|830|504|-326|
|tensor_base_g2s_mod128_112|tensor|u8|2|4096|已经提前送入 TMAU|571|490|-81|
|tensor_base_g2s_mod128_16|tensor|u8|2|4096|在 L2，TMAU 第一次使用|1232|498|-734|
|tensor_base_g2s_mod128_16|tensor|u8|2|4096|已经提前送入 TMAU|1011|505|-506|
|tensor_base_g2s_mod128_32|tensor|u8|2|4096|在 L2，TMAU 第一次使用|766|463|-303|
|tensor_base_g2s_mod128_32|tensor|u8|2|4096|已经提前送入 TMAU|541|464|-77|
|tensor_base_g2s_mod128_64|tensor|u8|2|4096|在 L2，TMAU 第一次使用|782|450|-332|
|tensor_base_g2s_mod128_64|tensor|u8|2|4096|已经提前送入 TMAU|547|466|-81|

## bulk_capacity

|测试项|类型|数据格式|rank|字节数|TensorMap 状态|第一次|第二次|第二次-第一次|
|---|---|---|---:|---:|---|---:|---:|---:|
|bulk_capacity_g2s_b1024|bulk|u8|2|1024|Bulk（没有 TensorMap）|282|282|0|
|bulk_capacity_g2s_b128|bulk|u8|2|128|Bulk（没有 TensorMap）|276|276|0|
|bulk_capacity_g2s_b16|bulk|u8|2|16|Bulk（没有 TensorMap）|261|276|15|
|bulk_capacity_g2s_b16384|bulk|u8|2|16384|Bulk（没有 TensorMap）|421|422|1|
|bulk_capacity_g2s_b2048|bulk|u8|2|2048|Bulk（没有 TensorMap）|292|292|0|
|bulk_capacity_g2s_b256|bulk|u8|2|256|Bulk（没有 TensorMap）|263|263|0|
|bulk_capacity_g2s_b32|bulk|u8|2|32|Bulk（没有 TensorMap）|261|276|15|
|bulk_capacity_g2s_b32768|bulk|u8|2|32768|Bulk（没有 TensorMap）|568|568|0|
|bulk_capacity_g2s_b4096|bulk|u8|2|4096|Bulk（没有 TensorMap）|311|309|-2|
|bulk_capacity_g2s_b512|bulk|u8|2|512|Bulk（没有 TensorMap）|285|282|-3|
|bulk_capacity_g2s_b64|bulk|u8|2|64|Bulk（没有 TensorMap）|276|276|0|
|bulk_capacity_g2s_b8192|bulk|u8|2|8192|Bulk（没有 TensorMap）|345|345|0|

## dtype

|测试项|类型|数据格式|rank|字节数|TensorMap 状态|第一次|第二次|第二次-第一次|
|---|---|---|---:|---:|---|---:|---:|---:|
|dtype_g2s_bf16_b128|tensor|bf16|2|128|在 L2，TMAU 第一次使用|745|407|-338|
|dtype_g2s_bf16_b128|tensor|bf16|2|128|已经提前送入 TMAU|477|398|-79|
|dtype_g2s_bf16_b4096|tensor|bf16|2|4096|在 L2，TMAU 第一次使用|758|439|-319|
|dtype_g2s_bf16_b4096|tensor|bf16|2|4096|已经提前送入 TMAU|504|440|-64|
|dtype_g2s_f16_b128|tensor|f16|2|128|在 L2，TMAU 第一次使用|713|407|-306|
|dtype_g2s_f16_b128|tensor|f16|2|128|已经提前送入 TMAU|465|401|-64|
|dtype_g2s_f16_b4096|tensor|f16|2|4096|在 L2，TMAU 第一次使用|754|446|-308|
|dtype_g2s_f16_b4096|tensor|f16|2|4096|已经提前送入 TMAU|515|446|-69|
|dtype_g2s_f32_b128|tensor|f32|2|128|在 L2，TMAU 第一次使用|730|407|-323|
|dtype_g2s_f32_b128|tensor|f32|2|128|已经提前送入 TMAU|465|385|-80|
|dtype_g2s_f32_b4096|tensor|f32|2|4096|在 L2，TMAU 第一次使用|754|450|-304|
|dtype_g2s_f32_b4096|tensor|f32|2|4096|已经提前送入 TMAU|507|444|-63|
|dtype_g2s_f32_ftz_b128|tensor|f32_ftz|2|128|在 L2，TMAU 第一次使用|730|402|-328|
|dtype_g2s_f32_ftz_b128|tensor|f32_ftz|2|128|已经提前送入 TMAU|477|414|-63|
|dtype_g2s_f32_ftz_b4096|tensor|f32_ftz|2|4096|在 L2，TMAU 第一次使用|766|442|-324|
|dtype_g2s_f32_ftz_b4096|tensor|f32_ftz|2|4096|已经提前送入 TMAU|510|442|-68|
|dtype_g2s_f64_b128|tensor|f64|2|128|在 L2，TMAU 第一次使用|719|391|-328|
|dtype_g2s_f64_b128|tensor|f64|2|128|已经提前送入 TMAU|457|398|-59|
|dtype_g2s_f64_b4096|tensor|f64|2|4096|在 L2，TMAU 第一次使用|749|439|-310|
|dtype_g2s_f64_b4096|tensor|f64|2|4096|已经提前送入 TMAU|507|442|-65|
|dtype_g2s_s32_b128|tensor|s32|2|128|在 L2，TMAU 第一次使用|713|397|-316|
|dtype_g2s_s32_b128|tensor|s32|2|128|已经提前送入 TMAU|477|401|-76|
|dtype_g2s_s32_b4096|tensor|s32|2|4096|在 L2，TMAU 第一次使用|749|444|-305|
|dtype_g2s_s32_b4096|tensor|s32|2|4096|已经提前送入 TMAU|510|446|-64|
|dtype_g2s_s64_b128|tensor|s64|2|128|在 L2，TMAU 第一次使用|730|378|-352|
|dtype_g2s_s64_b128|tensor|s64|2|128|已经提前送入 TMAU|477|409|-68|
|dtype_g2s_s64_b4096|tensor|s64|2|4096|在 L2，TMAU 第一次使用|766|442|-324|
|dtype_g2s_s64_b4096|tensor|s64|2|4096|已经提前送入 TMAU|523|433|-90|
|dtype_g2s_tf32_b128|tensor|tf32|2|128|在 L2，TMAU 第一次使用|725|415|-310|
|dtype_g2s_tf32_b128|tensor|tf32|2|128|已经提前送入 TMAU|465|414|-51|
|dtype_g2s_tf32_b4096|tensor|tf32|2|4096|在 L2，TMAU 第一次使用|767|439|-328|
|dtype_g2s_tf32_b4096|tensor|tf32|2|4096|已经提前送入 TMAU|515|438|-77|
|dtype_g2s_tf32_ftz_b128|tensor|tf32_ftz|2|128|在 L2，TMAU 第一次使用|730|402|-328|
|dtype_g2s_tf32_ftz_b128|tensor|tf32_ftz|2|128|已经提前送入 TMAU|484|414|-70|
|dtype_g2s_tf32_ftz_b4096|tensor|tf32_ftz|2|4096|在 L2，TMAU 第一次使用|758|444|-314|
|dtype_g2s_tf32_ftz_b4096|tensor|tf32_ftz|2|4096|已经提前送入 TMAU|507|440|-67|
|dtype_g2s_u16_b128|tensor|u16|2|128|在 L2，TMAU 第一次使用|697|402|-295|
|dtype_g2s_u16_b128|tensor|u16|2|128|已经提前送入 TMAU|470|414|-56|
|dtype_g2s_u16_b4096|tensor|u16|2|4096|在 L2，TMAU 第一次使用|766|439|-327|
|dtype_g2s_u16_b4096|tensor|u16|2|4096|已经提前送入 TMAU|510|433|-77|
|dtype_g2s_u32_b128|tensor|u32|2|128|在 L2，TMAU 第一次使用|716|397|-319|
|dtype_g2s_u32_b128|tensor|u32|2|128|已经提前送入 TMAU|489|420|-69|
|dtype_g2s_u32_b4096|tensor|u32|2|4096|在 L2，TMAU 第一次使用|769|439|-330|
|dtype_g2s_u32_b4096|tensor|u32|2|4096|已经提前送入 TMAU|510|438|-72|
|dtype_g2s_u64_b128|tensor|u64|2|128|在 L2，TMAU 第一次使用|739|415|-324|
|dtype_g2s_u64_b128|tensor|u64|2|128|已经提前送入 TMAU|489|398|-91|
|dtype_g2s_u64_b4096|tensor|u64|2|4096|在 L2，TMAU 第一次使用|754|446|-308|
|dtype_g2s_u64_b4096|tensor|u64|2|4096|已经提前送入 TMAU|515|440|-75|
|dtype_g2s_u8_b128|tensor|u8|2|128|在 L2，TMAU 第一次使用|719|400|-319|
|dtype_g2s_u8_b128|tensor|u8|2|128|已经提前送入 TMAU|477|409|-68|
|dtype_g2s_u8_b4096|tensor|u8|2|4096|在 L2，TMAU 第一次使用|763|442|-321|
|dtype_g2s_u8_b4096|tensor|u8|2|4096|已经提前送入 TMAU|507|442|-65|

## interleave

|测试项|类型|数据格式|rank|字节数|TensorMap 状态|第一次|第二次|第二次-第一次|
|---|---|---|---:|---:|---|---:|---:|---:|
|interleave16_g2s_b1024_int|tensor|u16|3|1024|在 L2，TMAU 第一次使用|777|427|-350|
|interleave16_g2s_b1024_int|tensor|u16|3|1024|已经提前送入 TMAU|474|422|-52|
|interleave16_g2s_b1024_plain|tensor|u16|3|1024|在 L2，TMAU 第一次使用|783|477|-306|
|interleave16_g2s_b1024_plain|tensor|u16|3|1024|已经提前送入 TMAU|558|482|-76|
|interleave16_g2s_b16384_int|tensor|u16|3|16384|在 L2，TMAU 第一次使用|901|567|-334|
|interleave16_g2s_b16384_int|tensor|u16|3|16384|已经提前送入 TMAU|649|580|-69|
|interleave16_g2s_b16384_plain|tensor|u16|3|16384|在 L2，TMAU 第一次使用|1771|1437|-334|
|interleave16_g2s_b16384_plain|tensor|u16|3|16384|已经提前送入 TMAU|1523|1448|-75|
|interleave16_g2s_b4096_int|tensor|u16|3|4096|在 L2，TMAU 第一次使用|768|455|-313|
|interleave16_g2s_b4096_int|tensor|u16|3|4096|已经提前送入 TMAU|524|450|-74|
|interleave16_g2s_b4096_plain|tensor|u16|3|4096|在 L2，TMAU 第一次使用|988|654|-334|
|interleave16_g2s_b4096_plain|tensor|u16|3|4096|已经提前送入 TMAU|740|676|-64|

## oob

|测试项|类型|数据格式|rank|字节数|TensorMap 状态|第一次|第二次|第二次-第一次|
|---|---|---|---:|---:|---|---:|---:|---:|
|oob_g2s_left100|tensor|u8|2|4096|在 L2，TMAU 第一次使用|754|431|-323|
|oob_g2s_left100|tensor|u8|2|4096|已经提前送入 TMAU|497|428|-69|
|oob_g2s_left12p5|tensor|u8|2|4096|在 L2，TMAU 第一次使用|811|498|-313|
|oob_g2s_left12p5|tensor|u8|2|4096|已经提前送入 TMAU|571|499|-72|
|oob_g2s_left25|tensor|u8|2|4096|在 L2，TMAU 第一次使用|797|469|-328|
|oob_g2s_left25|tensor|u8|2|4096|已经提前送入 TMAU|547|466|-81|
|oob_g2s_left50|tensor|u8|2|4096|在 L2，TMAU 第一次使用|776|466|-310|
|oob_g2s_left50|tensor|u8|2|4096|已经提前送入 TMAU|541|466|-75|
|oob_g2s_outer75|tensor|u8|2|4096|在 L2，TMAU 第一次使用|739|436|-303|
|oob_g2s_outer75|tensor|u8|2|4096|已经提前送入 TMAU|515|433|-82|
|oob_g2s_plain|tensor|u8|2|4096|在 L2，TMAU 第一次使用|749|444|-305|
|oob_g2s_plain|tensor|u8|2|4096|已经提前送入 TMAU|510|442|-68|
|oob_g2s_right25|tensor|u8|2|4096|在 L2，TMAU 第一次使用|778|455|-323|
|oob_g2s_right25|tensor|u8|2|4096|已经提前送入 TMAU|544|466|-78|

## oob_nan

|测试项|类型|数据格式|rank|字节数|TensorMap 状态|第一次|第二次|第二次-第一次|
|---|---|---|---:|---:|---|---:|---:|---:|
|oob_nan_bf16_nan|tensor|bf16|2|4096|在 L2，TMAU 第一次使用|791|474|-317|
|oob_nan_bf16_nan|tensor|bf16|2|4096|已经提前送入 TMAU|534|466|-68|
|oob_nan_bf16_zero|tensor|bf16|2|4096|在 L2，TMAU 第一次使用|787|463|-324|
|oob_nan_bf16_zero|tensor|bf16|2|4096|已经提前送入 TMAU|534|466|-68|
|oob_nan_f16_nan|tensor|f16|2|4096|在 L2，TMAU 第一次使用|778|462|-316|
|oob_nan_f16_nan|tensor|f16|2|4096|已经提前送入 TMAU|541|462|-79|
|oob_nan_f16_zero|tensor|f16|2|4096|在 L2，TMAU 第一次使用|778|479|-299|
|oob_nan_f16_zero|tensor|f16|2|4096|已经提前送入 TMAU|544|464|-80|
|oob_nan_f32_ftz_nan|tensor|f32_ftz|2|4096|在 L2，TMAU 第一次使用|776|466|-310|
|oob_nan_f32_ftz_nan|tensor|f32_ftz|2|4096|已经提前送入 TMAU|534|464|-70|
|oob_nan_f32_ftz_zero|tensor|f32_ftz|2|4096|在 L2，TMAU 第一次使用|797|466|-331|
|oob_nan_f32_ftz_zero|tensor|f32_ftz|2|4096|已经提前送入 TMAU|534|470|-64|
|oob_nan_f32_nan|tensor|f32|2|4096|在 L2，TMAU 第一次使用|791|460|-331|
|oob_nan_f32_nan|tensor|f32|2|4096|已经提前送入 TMAU|544|475|-69|
|oob_nan_f32_zero|tensor|f32|2|4096|在 L2，TMAU 第一次使用|800|470|-330|
|oob_nan_f32_zero|tensor|f32|2|4096|已经提前送入 TMAU|541|466|-75|
|oob_nan_f64_nan|tensor|f64|2|4096|在 L2，TMAU 第一次使用|797|466|-331|
|oob_nan_f64_nan|tensor|f64|2|4096|已经提前送入 TMAU|541|462|-79|
|oob_nan_f64_zero|tensor|f64|2|4096|在 L2，TMAU 第一次使用|793|463|-330|
|oob_nan_f64_zero|tensor|f64|2|4096|已经提前送入 TMAU|541|462|-79|
|oob_nan_tf32_ftz_nan|tensor|tf32_ftz|2|4096|在 L2，TMAU 第一次使用|797|469|-328|
|oob_nan_tf32_ftz_nan|tensor|tf32_ftz|2|4096|已经提前送入 TMAU|534|475|-59|
|oob_nan_tf32_ftz_zero|tensor|tf32_ftz|2|4096|在 L2，TMAU 第一次使用|793|466|-327|
|oob_nan_tf32_ftz_zero|tensor|tf32_ftz|2|4096|已经提前送入 TMAU|541|464|-77|
|oob_nan_tf32_nan|tensor|tf32|2|4096|在 L2，TMAU 第一次使用|778|462|-316|
|oob_nan_tf32_nan|tensor|tf32|2|4096|已经提前送入 TMAU|544|464|-80|
|oob_nan_tf32_zero|tensor|tf32|2|4096|在 L2，TMAU 第一次使用|782|470|-312|
|oob_nan_tf32_zero|tensor|tf32|2|4096|已经提前送入 TMAU|541|466|-75|

## rank

|测试项|类型|数据格式|rank|字节数|TensorMap 状态|第一次|第二次|第二次-第一次|
|---|---|---|---:|---:|---|---:|---:|---:|
|rank_g2s_r1_b128|tensor|u8|1|128|在 L2，TMAU 第一次使用|723|407|-316|
|rank_g2s_r1_b128|tensor|u8|1|128|已经提前送入 TMAU|447|374|-73|
|rank_g2s_r1_b256|tensor|u8|1|256|在 L2，TMAU 第一次使用|726|374|-352|
|rank_g2s_r1_b256|tensor|u8|1|256|已经提前送入 TMAU|483|397|-86|
|rank_g2s_r2_b128|tensor|u8|2|128|在 L2，TMAU 第一次使用|704|378|-326|
|rank_g2s_r2_b128|tensor|u8|2|128|已经提前送入 TMAU|438|380|-58|
|rank_g2s_r2_b256|tensor|u8|2|256|在 L2，TMAU 第一次使用|710|374|-336|
|rank_g2s_r2_b256|tensor|u8|2|256|已经提前送入 TMAU|424|389|-35|
|rank_g2s_r2_b32768|tensor|u8|2|32768|在 L2，TMAU 第一次使用|1018|701|-317|
|rank_g2s_r2_b32768|tensor|u8|2|32768|已经提前送入 TMAU|768|704|-64|
|rank_g2s_r2_b4096|tensor|u8|2|4096|在 L2，TMAU 第一次使用|782|436|-346|
|rank_g2s_r2_b4096|tensor|u8|2|4096|已经提前送入 TMAU|515|442|-73|
|rank_g2s_r3_b128|tensor|u8|3|128|在 L2，TMAU 第一次使用|748|396|-352|
|rank_g2s_r3_b128|tensor|u8|3|128|已经提前送入 TMAU|474|420|-54|
|rank_g2s_r3_b256|tensor|u8|3|256|在 L2，TMAU 第一次使用|742|398|-344|
|rank_g2s_r3_b256|tensor|u8|3|256|已经提前送入 TMAU|501|398|-103|
|rank_g2s_r3_b32768|tensor|u8|3|32768|在 L2，TMAU 第一次使用|1046|717|-329|
|rank_g2s_r3_b32768|tensor|u8|3|32768|已经提前送入 TMAU|803|740|-63|
|rank_g2s_r3_b4096|tensor|u8|3|4096|在 L2，TMAU 第一次使用|801|466|-335|
|rank_g2s_r3_b4096|tensor|u8|3|4096|已经提前送入 TMAU|566|482|-84|
|rank_g2s_r4_b128|tensor|u8|4|128|在 L2，TMAU 第一次使用|743|407|-336|
|rank_g2s_r4_b128|tensor|u8|4|128|已经提前送入 TMAU|483|392|-91|
|rank_g2s_r4_b256|tensor|u8|4|256|在 L2，TMAU 第一次使用|741|415|-326|
|rank_g2s_r4_b256|tensor|u8|4|256|已经提前送入 TMAU|491|416|-75|
|rank_g2s_r4_b32768|tensor|u8|4|32768|在 L2，TMAU 第一次使用|1232|939|-293|
|rank_g2s_r4_b32768|tensor|u8|4|32768|已经提前送入 TMAU|1011|945|-66|
|rank_g2s_r4_b4096|tensor|u8|4|4096|在 L2，TMAU 第一次使用|861|548|-313|
|rank_g2s_r4_b4096|tensor|u8|4|4096|已经提前送入 TMAU|633|537|-96|
|rank_g2s_r5_b128|tensor|u8|5|128|在 L2，TMAU 第一次使用|729|420|-309|
|rank_g2s_r5_b128|tensor|u8|5|128|已经提前送入 TMAU|492|436|-56|
|rank_g2s_r5_b256|tensor|u8|5|256|在 L2，TMAU 第一次使用|786|435|-351|
|rank_g2s_r5_b256|tensor|u8|5|256|已经提前送入 TMAU|526|444|-82|
|rank_g2s_r5_b32768|tensor|u8|5|32768|在 L2，TMAU 第一次使用|1752|1459|-293|
|rank_g2s_r5_b32768|tensor|u8|5|32768|已经提前送入 TMAU|1517|1451|-66|
|rank_g2s_r5_b4096|tensor|u8|5|4096|在 L2，TMAU 第一次使用|1008|689|-319|
|rank_g2s_r5_b4096|tensor|u8|5|4096|已经提前送入 TMAU|770|693|-77|

## stride

|测试项|类型|数据格式|rank|字节数|TensorMap 状态|第一次|第二次|第二次-第一次|
|---|---|---|---:|---:|---|---:|---:|---:|
|stride_g2s_s128|tensor|u8|2|4096|在 L2，TMAU 第一次使用|763|442|-321|
|stride_g2s_s128|tensor|u8|2|4096|已经提前送入 TMAU|507|442|-65|
|stride_g2s_s144|tensor|u8|2|4096|在 L2，TMAU 第一次使用|802|479|-323|
|stride_g2s_s144|tensor|u8|2|4096|已经提前送入 TMAU|552|484|-68|
|stride_g2s_s256|tensor|u8|2|4096|在 L2，TMAU 第一次使用|749|439|-310|
|stride_g2s_s256|tensor|u8|2|4096|已经提前送入 TMAU|515|442|-73|
|stride_g2s_s512|tensor|u8|2|4096|在 L2，TMAU 第一次使用|766|436|-330|
|stride_g2s_s512|tensor|u8|2|4096|已经提前送入 TMAU|515|444|-71|

## subbox

|测试项|类型|数据格式|rank|字节数|TensorMap 状态|第一次|第二次|第二次-第一次|
|---|---|---|---:|---:|---|---:|---:|---:|
|subbox_g2s_large2d_offset|tensor|u8|2|4096|在 L2，TMAU 第一次使用|782|463|-319|
|subbox_g2s_large2d_offset|tensor|u8|2|4096|已经提前送入 TMAU|541|475|-66|
|subbox_g2s_large2d_plain|tensor|u8|2|4096|在 L2，TMAU 第一次使用|766|442|-324|
|subbox_g2s_large2d_plain|tensor|u8|2|4096|已经提前送入 TMAU|510|442|-68|
|subbox_g2s_large3d_offset|tensor|u8|3|4096|在 L2，TMAU 第一次使用|814|488|-326|
|subbox_g2s_large3d_offset|tensor|u8|3|4096|已经提前送入 TMAU|566|489|-77|
|subbox_g2s_large3d_plain|tensor|u8|3|4096|在 L2，TMAU 第一次使用|790|461|-329|
|subbox_g2s_large3d_plain|tensor|u8|3|4096|已经提前送入 TMAU|558|467|-91|
|subbox_g2s_small2d_offset|tensor|u8|2|128|在 L2，TMAU 第一次使用|734|402|-332|
|subbox_g2s_small2d_offset|tensor|u8|2|128|已经提前送入 TMAU|484|409|-75|
|subbox_g2s_small2d_plain|tensor|u8|2|128|在 L2，TMAU 第一次使用|716|374|-342|
|subbox_g2s_small2d_plain|tensor|u8|2|128|已经提前送入 TMAU|457|391|-66|

## swizzle

|测试项|类型|数据格式|rank|字节数|TensorMap 状态|第一次|第二次|第二次-第一次|
|---|---|---|---:|---:|---|---:|---:|---:|
|swizzle128_g2s_b1024_plain|tensor|u8|2|1024|在 L2，TMAU 第一次使用|734|412|-322|
|swizzle128_g2s_b1024_plain|tensor|u8|2|1024|已经提前送入 TMAU|489|422|-67|
|swizzle128_g2s_b1024_swz|tensor|u8|2|1024|在 L2，TMAU 第一次使用|734|412|-322|
|swizzle128_g2s_b1024_swz|tensor|u8|2|1024|已经提前送入 TMAU|489|404|-85|
|swizzle128_g2s_b16384_plain|tensor|u8|3|16384|在 L2，TMAU 第一次使用|892|567|-325|
|swizzle128_g2s_b16384_plain|tensor|u8|3|16384|已经提前送入 TMAU|646|580|-66|
|swizzle128_g2s_b16384_swz|tensor|u8|3|16384|在 L2，TMAU 第一次使用|877|571|-306|
|swizzle128_g2s_b16384_swz|tensor|u8|3|16384|已经提前送入 TMAU|654|578|-76|
|swizzle128_g2s_b4096_plain|tensor|u8|3|4096|在 L2，TMAU 第一次使用|796|455|-341|
|swizzle128_g2s_b4096_plain|tensor|u8|3|4096|已经提前送入 TMAU|524|446|-78|
|swizzle128_g2s_b4096_swz|tensor|u8|3|4096|在 L2，TMAU 第一次使用|777|461|-316|
|swizzle128_g2s_b4096_swz|tensor|u8|3|4096|已经提前送入 TMAU|524|450|-74|
|swizzle32_g2s_b1024_plain|tensor|u8|2|1024|在 L2，TMAU 第一次使用|758|424|-334|
|swizzle32_g2s_b1024_plain|tensor|u8|2|1024|已经提前送入 TMAU|470|422|-48|
|swizzle32_g2s_b1024_swz|tensor|u8|2|1024|在 L2，TMAU 第一次使用|745|422|-323|
|swizzle32_g2s_b1024_swz|tensor|u8|2|1024|已经提前送入 TMAU|497|420|-77|
|swizzle32_g2s_b16384_plain|tensor|u8|3|16384|在 L2，TMAU 第一次使用|1257|922|-335|
|swizzle32_g2s_b16384_plain|tensor|u8|3|16384|已经提前送入 TMAU|1001|932|-69|
|swizzle32_g2s_b16384_swz|tensor|u8|3|16384|在 L2，TMAU 第一次使用|1224|922|-302|
|swizzle32_g2s_b16384_swz|tensor|u8|3|16384|已经提前送入 TMAU|1001|918|-83|
|swizzle32_g2s_b4096_plain|tensor|u8|3|4096|在 L2，TMAU 第一次使用|859|538|-321|
|swizzle32_g2s_b4096_plain|tensor|u8|3|4096|已经提前送入 TMAU|622|548|-74|
|swizzle32_g2s_b4096_swz|tensor|u8|3|4096|在 L2，TMAU 第一次使用|864|538|-326|
|swizzle32_g2s_b4096_swz|tensor|u8|3|4096|已经提前送入 TMAU|606|538|-68|
|swizzle64_g2s_b1024_plain|tensor|u8|2|1024|在 L2，TMAU 第一次使用|734|420|-314|
|swizzle64_g2s_b1024_plain|tensor|u8|2|1024|已经提前送入 TMAU|484|409|-75|
|swizzle64_g2s_b1024_swz|tensor|u8|2|1024|在 L2，TMAU 第一次使用|734|412|-322|
|swizzle64_g2s_b1024_swz|tensor|u8|2|1024|已经提前送入 TMAU|470|409|-61|
|swizzle64_g2s_b16384_plain|tensor|u8|3|16384|在 L2，TMAU 第一次使用|979|674|-305|
|swizzle64_g2s_b16384_plain|tensor|u8|3|16384|已经提前送入 TMAU|742|671|-71|
|swizzle64_g2s_b16384_swz|tensor|u8|3|16384|在 L2，TMAU 第一次使用|997|676|-321|
|swizzle64_g2s_b16384_swz|tensor|u8|3|16384|已经提前送入 TMAU|740|684|-56|
|swizzle64_g2s_b4096_plain|tensor|u8|3|4096|在 L2，TMAU 第一次使用|787|480|-307|
|swizzle64_g2s_b4096_plain|tensor|u8|3|4096|已经提前送入 TMAU|542|482|-60|
|swizzle64_g2s_b4096_swz|tensor|u8|3|4096|在 L2，TMAU 第一次使用|792|466|-326|
|swizzle64_g2s_b4096_swz|tensor|u8|3|4096|已经提前送入 TMAU|542|458|-84|

## tensor_capacity

|测试项|类型|数据格式|rank|字节数|TensorMap 状态|第一次|第二次|第二次-第一次|
|---|---|---|---:|---:|---|---:|---:|---:|
|tensor_capacity_g2s_b1024|tensor|u8|2|1024|在 L2，TMAU 第一次使用|713|412|-301|
|tensor_capacity_g2s_b1024|tensor|u8|2|1024|已经提前送入 TMAU|489|418|-71|
|tensor_capacity_g2s_b128|tensor|u8|2|128|在 L2，TMAU 第一次使用|697|386|-311|
|tensor_capacity_g2s_b128|tensor|u8|2|128|已经提前送入 TMAU|489|409|-80|
|tensor_capacity_g2s_b16384|tensor|u8|2|16384|在 L2，TMAU 第一次使用|854|548|-306|
|tensor_capacity_g2s_b16384|tensor|u8|2|16384|已经提前送入 TMAU|624|550|-74|
|tensor_capacity_g2s_b2048|tensor|u8|2|2048|在 L2，TMAU 第一次使用|745|420|-325|
|tensor_capacity_g2s_b2048|tensor|u8|2|2048|已经提前送入 TMAU|497|422|-75|
|tensor_capacity_g2s_b256|tensor|u8|2|256|在 L2，TMAU 第一次使用|716|397|-319|
|tensor_capacity_g2s_b256|tensor|u8|2|256|已经提前送入 TMAU|470|404|-66|
|tensor_capacity_g2s_b32768|tensor|u8|2|32768|在 L2，TMAU 第一次使用|1007|703|-304|
|tensor_capacity_g2s_b32768|tensor|u8|2|32768|已经提前送入 TMAU|768|710|-58|
|tensor_capacity_g2s_b4096|tensor|u8|2|4096|在 L2，TMAU 第一次使用|766|439|-327|
|tensor_capacity_g2s_b4096|tensor|u8|2|4096|已经提前送入 TMAU|510|440|-70|
|tensor_capacity_g2s_b512|tensor|u8|2|512|在 L2，TMAU 第一次使用|734|412|-322|
|tensor_capacity_g2s_b512|tensor|u8|2|512|已经提前送入 TMAU|489|416|-73|
|tensor_capacity_g2s_b8192|tensor|u8|2|8192|在 L2，TMAU 第一次使用|806|479|-327|
|tensor_capacity_g2s_b8192|tensor|u8|2|8192|已经提前送入 TMAU|550|484|-66|

## 怎样理解两种 Tensor 结果

“在 L2，TMAU 第一次使用”的第一列包含从 L2 取回 TensorMap并交给 TMAU 处理的时间；第二列是同一个 TensorMap 的立即复用。“已经提前送入 TMAU”是独立控制项，第一次发令前显式 prefetch，第二次仍是相同地址的热复用。
