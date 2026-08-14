# B200 G2S 完成周期

每个测试点只分配一个新的 TensorMap。第一次测量是该地址第一次进入 TMAU，完成后在同一个 kernel 内立刻用完全相同的地址和内容测第二次。主测试没有 TensorMap prefetch；普通全局读取只把 descriptor 放进 L2。

表格分别显示第一次和第二次，不再把两次平均成一个数字。每个数字来自一轮固定的 640 点扫描：80 个静态 kernel bank 各负责 8 个点，bank 在计时外选择；计时路径没有索引跳转，相邻点静态增加一条比较、一条统一分支和一条不访问数据通路的 PM-event。请求步长为三周期，实际物理间隔由 clock64 记录，原始测量点、地址和指纹保存在同目录压缩 CSV。

测试组：`tensor__g2s__fresh-map-2d`

## tensor_2d_length_u16_row128

|测试项|类型|数据格式|rank|字节数|TensorMap 状态|第一次|第二次|第二次-第一次|
|---|---|---|---:|---:|---|---:|---:|---:|
|tensor_2d_length_g2s_u16_row128_b16384|tensor|u16|2|16384|在 L2，TMAU 第一次使用|938|594|-344|
|tensor_2d_length_g2s_u16_row128_b16384|tensor|u16|2|16384|已经提前送入 TMAU|690|599|-91|
|tensor_2d_length_g2s_u16_row128_b32768|tensor|u16|2|32768|在 L2，TMAU 第一次使用|1069|727|-342|
|tensor_2d_length_g2s_u16_row128_b32768|tensor|u16|2|32768|已经提前送入 TMAU|829|732|-97|

## tensor_2d_length_u16_row32

|测试项|类型|数据格式|rank|字节数|TensorMap 状态|第一次|第二次|第二次-第一次|
|---|---|---|---:|---:|---|---:|---:|---:|
|tensor_2d_length_g2s_u16_row32_b1024|tensor|u16|2|1024|在 L2，TMAU 第一次使用|829|487|-342|
|tensor_2d_length_g2s_u16_row32_b1024|tensor|u16|2|1024|已经提前送入 TMAU|539|487|-52|
|tensor_2d_length_g2s_u16_row32_b128|tensor|u16|2|128|在 L2，TMAU 第一次使用|820|425|-395|
|tensor_2d_length_g2s_u16_row32_b128|tensor|u16|2|128|已经提前送入 TMAU|508|418|-90|
|tensor_2d_length_g2s_u16_row32_b256|tensor|u16|2|256|在 L2，TMAU 第一次使用|777|459|-318|
|tensor_2d_length_g2s_u16_row32_b256|tensor|u16|2|256|已经提前送入 TMAU|537|434|-103|
|tensor_2d_length_g2s_u16_row32_b4096|tensor|u16|2|4096|在 L2，TMAU 第一次使用|921|552|-369|
|tensor_2d_length_g2s_u16_row32_b4096|tensor|u16|2|4096|已经提前送入 TMAU|644|561|-83|
|tensor_2d_length_g2s_u16_row32_b8192|tensor|u16|2|8192|在 L2，TMAU 第一次使用|1034|690|-344|
|tensor_2d_length_g2s_u16_row32_b8192|tensor|u16|2|8192|已经提前送入 TMAU|772|681|-91|

## tensor_2d_length_u8_row128

|测试项|类型|数据格式|rank|字节数|TensorMap 状态|第一次|第二次|第二次-第一次|
|---|---|---|---:|---:|---|---:|---:|---:|
|tensor_2d_length_g2s_u8_row128_b1024|tensor|u8|2|1024|在 L2，TMAU 第一次使用|812|459|-353|
|tensor_2d_length_g2s_u8_row128_b1024|tensor|u8|2|1024|已经提前送入 TMAU|543|462|-81|
|tensor_2d_length_g2s_u8_row128_b128|tensor|u8|2|128|在 L2，TMAU 第一次使用|829|440|-389|
|tensor_2d_length_g2s_u8_row128_b128|tensor|u8|2|128|已经提前送入 TMAU|528|442|-86|
|tensor_2d_length_g2s_u8_row128_b16256|tensor|u8|2|16256|在 L2，TMAU 第一次使用|945|590|-355|
|tensor_2d_length_g2s_u8_row128_b16256|tensor|u8|2|16256|已经提前送入 TMAU|685|594|-91|
|tensor_2d_length_g2s_u8_row128_b16384|tensor|u8|2|16384|在 L2，TMAU 第一次使用|956|597|-359|
|tensor_2d_length_g2s_u8_row128_b16384|tensor|u8|2|16384|已经提前送入 TMAU|692|594|-98|
|tensor_2d_length_g2s_u8_row128_b16512|tensor|u8|2|16512|在 L2，TMAU 第一次使用|969|603|-366|
|tensor_2d_length_g2s_u8_row128_b16512|tensor|u8|2|16512|已经提前送入 TMAU|690|599|-91|
|tensor_2d_length_g2s_u8_row128_b17408|tensor|u8|2|17408|在 L2，TMAU 第一次使用|977|601|-376|
|tensor_2d_length_g2s_u8_row128_b17408|tensor|u8|2|17408|已经提前送入 TMAU|703|612|-91|
|tensor_2d_length_g2s_u8_row128_b18432|tensor|u8|2|18432|在 L2，TMAU 第一次使用|969|603|-366|
|tensor_2d_length_g2s_u8_row128_b18432|tensor|u8|2|18432|已经提前送入 TMAU|705|609|-96|
|tensor_2d_length_g2s_u8_row128_b2048|tensor|u8|2|2048|在 L2，TMAU 第一次使用|820|449|-371|
|tensor_2d_length_g2s_u8_row128_b2048|tensor|u8|2|2048|已经提前送入 TMAU|559|450|-109|
|tensor_2d_length_g2s_u8_row128_b20480|tensor|u8|2|20480|在 L2，TMAU 第一次使用|1004|638|-366|
|tensor_2d_length_g2s_u8_row128_b20480|tensor|u8|2|20480|已经提前送入 TMAU|724|633|-91|
|tensor_2d_length_g2s_u8_row128_b22528|tensor|u8|2|22528|在 L2，TMAU 第一次使用|1011|645|-366|
|tensor_2d_length_g2s_u8_row128_b22528|tensor|u8|2|22528|已经提前送入 TMAU|744|653|-91|
|tensor_2d_length_g2s_u8_row128_b24576|tensor|u8|2|24576|在 L2，TMAU 第一次使用|1041|666|-375|
|tensor_2d_length_g2s_u8_row128_b24576|tensor|u8|2|24576|已经提前送入 TMAU|757|663|-94|
|tensor_2d_length_g2s_u8_row128_b256|tensor|u8|2|256|在 L2，TMAU 第一次使用|818|459|-359|
|tensor_2d_length_g2s_u8_row128_b256|tensor|u8|2|256|已经提前送入 TMAU|537|460|-77|
|tensor_2d_length_g2s_u8_row128_b28672|tensor|u8|2|28672|在 L2，TMAU 第一次使用|1054|697|-357|
|tensor_2d_length_g2s_u8_row128_b28672|tensor|u8|2|28672|已经提前送入 TMAU|792|701|-91|
|tensor_2d_length_g2s_u8_row128_b32768|tensor|u8|2|32768|在 L2，TMAU 第一次使用|1100|729|-371|
|tensor_2d_length_g2s_u8_row128_b32768|tensor|u8|2|32768|已经提前送入 TMAU|826|729|-97|
|tensor_2d_length_g2s_u8_row128_b4096|tensor|u8|2|4096|在 L2，TMAU 第一次使用|864|487|-377|
|tensor_2d_length_g2s_u8_row128_b4096|tensor|u8|2|4096|已经提前送入 TMAU|594|490|-104|
|tensor_2d_length_g2s_u8_row128_b512|tensor|u8|2|512|在 L2，TMAU 第一次使用|796|449|-347|
|tensor_2d_length_g2s_u8_row128_b512|tensor|u8|2|512|已经提前送入 TMAU|539|450|-89|
|tensor_2d_length_g2s_u8_row128_b8192|tensor|u8|2|8192|在 L2，TMAU 第一次使用|890|518|-372|
|tensor_2d_length_g2s_u8_row128_b8192|tensor|u8|2|8192|已经提前送入 TMAU|624|530|-94|

## tensor_2d_length_u8_row32

|测试项|类型|数据格式|rank|字节数|TensorMap 状态|第一次|第二次|第二次-第一次|
|---|---|---|---:|---:|---|---:|---:|---:|
|tensor_2d_length_g2s_u8_row32_b1024|tensor|u8|2|1024|在 L2，TMAU 第一次使用|849|452|-397|
|tensor_2d_length_g2s_u8_row32_b1024|tensor|u8|2|1024|已经提前送入 TMAU|570|487|-83|
|tensor_2d_length_g2s_u8_row32_b1056|tensor|u8|2|1056|在 L2，TMAU 第一次使用|853|483|-370|
|tensor_2d_length_g2s_u8_row32_b1056|tensor|u8|2|1056|已经提前送入 TMAU|543|484|-59|
|tensor_2d_length_g2s_u8_row32_b1152|tensor|u8|2|1152|在 L2，TMAU 第一次使用|805|468|-337|
|tensor_2d_length_g2s_u8_row32_b1152|tensor|u8|2|1152|已经提前送入 TMAU|570|479|-91|
|tensor_2d_length_g2s_u8_row32_b128|tensor|u8|2|128|在 L2，TMAU 第一次使用|825|459|-366|
|tensor_2d_length_g2s_u8_row32_b128|tensor|u8|2|128|已经提前送入 TMAU|555|460|-95|
|tensor_2d_length_g2s_u8_row32_b1280|tensor|u8|2|1280|在 L2，TMAU 第一次使用|849|496|-353|
|tensor_2d_length_g2s_u8_row32_b1280|tensor|u8|2|1280|已经提前送入 TMAU|567|497|-70|
|tensor_2d_length_g2s_u8_row32_b1408|tensor|u8|2|1408|在 L2，TMAU 第一次使用|853|476|-377|
|tensor_2d_length_g2s_u8_row32_b1408|tensor|u8|2|1408|已经提前送入 TMAU|581|484|-97|
|tensor_2d_length_g2s_u8_row32_b1536|tensor|u8|2|1536|在 L2，TMAU 第一次使用|842|476|-366|
|tensor_2d_length_g2s_u8_row32_b1536|tensor|u8|2|1536|已经提前送入 TMAU|581|479|-102|
|tensor_2d_length_g2s_u8_row32_b160|tensor|u8|2|160|在 L2，TMAU 第一次使用|814|462|-352|
|tensor_2d_length_g2s_u8_row32_b160|tensor|u8|2|160|已经提前送入 TMAU|565|450|-115|
|tensor_2d_length_g2s_u8_row32_b1664|tensor|u8|2|1664|在 L2，TMAU 第一次使用|868|501|-367|
|tensor_2d_length_g2s_u8_row32_b1664|tensor|u8|2|1664|已经提前送入 TMAU|581|484|-97|
|tensor_2d_length_g2s_u8_row32_b1792|tensor|u8|2|1792|在 L2，TMAU 第一次使用|862|498|-364|
|tensor_2d_length_g2s_u8_row32_b1792|tensor|u8|2|1792|已经提前送入 TMAU|586|479|-107|
|tensor_2d_length_g2s_u8_row32_b192|tensor|u8|2|192|在 L2，TMAU 第一次使用|805|462|-343|
|tensor_2d_length_g2s_u8_row32_b192|tensor|u8|2|192|已经提前送入 TMAU|537|450|-87|
|tensor_2d_length_g2s_u8_row32_b1920|tensor|u8|2|1920|在 L2，TMAU 第一次使用|853|511|-342|
|tensor_2d_length_g2s_u8_row32_b1920|tensor|u8|2|1920|已经提前送入 TMAU|600|503|-97|
|tensor_2d_length_g2s_u8_row32_b2016|tensor|u8|2|2016|在 L2，TMAU 第一次使用|833|492|-341|
|tensor_2d_length_g2s_u8_row32_b2016|tensor|u8|2|2016|已经提前送入 TMAU|580|496|-84|
|tensor_2d_length_g2s_u8_row32_b2048|tensor|u8|2|2048|在 L2，TMAU 第一次使用|877|504|-373|
|tensor_2d_length_g2s_u8_row32_b2048|tensor|u8|2|2048|已经提前送入 TMAU|607|514|-93|
|tensor_2d_length_g2s_u8_row32_b2080|tensor|u8|2|2080|在 L2，TMAU 第一次使用|868|522|-346|
|tensor_2d_length_g2s_u8_row32_b2080|tensor|u8|2|2080|已经提前送入 TMAU|589|496|-93|
|tensor_2d_length_g2s_u8_row32_b224|tensor|u8|2|224|在 L2，TMAU 第一次使用|794|462|-332|
|tensor_2d_length_g2s_u8_row32_b224|tensor|u8|2|224|已经提前送入 TMAU|565|450|-115|
|tensor_2d_length_g2s_u8_row32_b256|tensor|u8|2|256|在 L2，TMAU 第一次使用|809|462|-347|
|tensor_2d_length_g2s_u8_row32_b256|tensor|u8|2|256|已经提前送入 TMAU|565|450|-115|
|tensor_2d_length_g2s_u8_row32_b2560|tensor|u8|2|2560|在 L2，TMAU 第一次使用|890|522|-368|
|tensor_2d_length_g2s_u8_row32_b2560|tensor|u8|2|2560|已经提前送入 TMAU|618|514|-104|
|tensor_2d_length_g2s_u8_row32_b288|tensor|u8|2|288|在 L2，TMAU 第一次使用|833|438|-395|
|tensor_2d_length_g2s_u8_row32_b288|tensor|u8|2|288|已经提前送入 TMAU|528|456|-72|
|tensor_2d_length_g2s_u8_row32_b3072|tensor|u8|2|3072|在 L2，TMAU 第一次使用|890|528|-362|
|tensor_2d_length_g2s_u8_row32_b3072|tensor|u8|2|3072|已经提前送入 TMAU|618|529|-89|
|tensor_2d_length_g2s_u8_row32_b32|tensor|u8|2|32|在 L2，TMAU 第一次使用|825|459|-366|
|tensor_2d_length_g2s_u8_row32_b32|tensor|u8|2|32|已经提前送入 TMAU|555|460|-95|
|tensor_2d_length_g2s_u8_row32_b320|tensor|u8|2|320|在 L2，TMAU 第一次使用|836|452|-384|
|tensor_2d_length_g2s_u8_row32_b320|tensor|u8|2|320|已经提前送入 TMAU|528|433|-95|
|tensor_2d_length_g2s_u8_row32_b352|tensor|u8|2|352|在 L2，TMAU 第一次使用|833|452|-381|
|tensor_2d_length_g2s_u8_row32_b352|tensor|u8|2|352|已经提前送入 TMAU|528|456|-72|
|tensor_2d_length_g2s_u8_row32_b3584|tensor|u8|2|3584|在 L2，TMAU 第一次使用|892|559|-333|
|tensor_2d_length_g2s_u8_row32_b3584|tensor|u8|2|3584|已经提前送入 TMAU|655|540|-115|
|tensor_2d_length_g2s_u8_row32_b384|tensor|u8|2|384|在 L2，TMAU 第一次使用|836|452|-384|
|tensor_2d_length_g2s_u8_row32_b384|tensor|u8|2|384|已经提前送入 TMAU|528|456|-72|
|tensor_2d_length_g2s_u8_row32_b4064|tensor|u8|2|4064|在 L2，TMAU 第一次使用|936|549|-387|
|tensor_2d_length_g2s_u8_row32_b4064|tensor|u8|2|4064|已经提前送入 TMAU|655|564|-91|
|tensor_2d_length_g2s_u8_row32_b4096|tensor|u8|2|4096|在 L2，TMAU 第一次使用|925|553|-372|
|tensor_2d_length_g2s_u8_row32_b4096|tensor|u8|2|4096|已经提前送入 TMAU|642|551|-91|
|tensor_2d_length_g2s_u8_row32_b4128|tensor|u8|2|4128|在 L2，TMAU 第一次使用|936|583|-353|
|tensor_2d_length_g2s_u8_row32_b4128|tensor|u8|2|4128|已经提前送入 TMAU|679|594|-85|
|tensor_2d_length_g2s_u8_row32_b416|tensor|u8|2|416|在 L2，TMAU 第一次使用|812|459|-353|
|tensor_2d_length_g2s_u8_row32_b416|tensor|u8|2|416|已经提前送入 TMAU|537|462|-75|
|tensor_2d_length_g2s_u8_row32_b448|tensor|u8|2|448|在 L2，TMAU 第一次使用|812|459|-353|
|tensor_2d_length_g2s_u8_row32_b448|tensor|u8|2|448|已经提前送入 TMAU|537|462|-75|
|tensor_2d_length_g2s_u8_row32_b4608|tensor|u8|2|4608|在 L2，TMAU 第一次使用|934|579|-355|
|tensor_2d_length_g2s_u8_row32_b4608|tensor|u8|2|4608|已经提前送入 TMAU|668|567|-101|
|tensor_2d_length_g2s_u8_row32_b480|tensor|u8|2|480|在 L2，TMAU 第一次使用|812|459|-353|
|tensor_2d_length_g2s_u8_row32_b480|tensor|u8|2|480|已经提前送入 TMAU|543|462|-81|
|tensor_2d_length_g2s_u8_row32_b512|tensor|u8|2|512|在 L2，TMAU 第一次使用|812|459|-353|
|tensor_2d_length_g2s_u8_row32_b512|tensor|u8|2|512|已经提前送入 TMAU|537|462|-75|
|tensor_2d_length_g2s_u8_row32_b5120|tensor|u8|2|5120|在 L2，TMAU 第一次使用|958|590|-368|
|tensor_2d_length_g2s_u8_row32_b5120|tensor|u8|2|5120|已经提前送入 TMAU|716|605|-111|
|tensor_2d_length_g2s_u8_row32_b5632|tensor|u8|2|5632|在 L2，TMAU 第一次使用|977|601|-376|
|tensor_2d_length_g2s_u8_row32_b5632|tensor|u8|2|5632|已经提前送入 TMAU|685|594|-91|
|tensor_2d_length_g2s_u8_row32_b6144|tensor|u8|2|6144|在 L2，TMAU 第一次使用|977|631|-346|
|tensor_2d_length_g2s_u8_row32_b6144|tensor|u8|2|6144|已经提前送入 TMAU|709|612|-97|
|tensor_2d_length_g2s_u8_row32_b64|tensor|u8|2|64|在 L2，TMAU 第一次使用|812|459|-353|
|tensor_2d_length_g2s_u8_row32_b64|tensor|u8|2|64|已经提前送入 TMAU|555|460|-95|
|tensor_2d_length_g2s_u8_row32_b640|tensor|u8|2|640|在 L2，TMAU 第一次使用|820|459|-361|
|tensor_2d_length_g2s_u8_row32_b640|tensor|u8|2|640|已经提前送入 TMAU|565|460|-105|
|tensor_2d_length_g2s_u8_row32_b6656|tensor|u8|2|6656|在 L2，TMAU 第一次使用|988|655|-333|
|tensor_2d_length_g2s_u8_row32_b6656|tensor|u8|2|6656|已经提前送入 TMAU|757|660|-97|
|tensor_2d_length_g2s_u8_row32_b7168|tensor|u8|2|7168|在 L2，TMAU 第一次使用|1025|675|-350|
|tensor_2d_length_g2s_u8_row32_b7168|tensor|u8|2|7168|已经提前送入 TMAU|751|660|-91|
|tensor_2d_length_g2s_u8_row32_b768|tensor|u8|2|768|在 L2，TMAU 第一次使用|842|478|-364|
|tensor_2d_length_g2s_u8_row32_b768|tensor|u8|2|768|已经提前送入 TMAU|533|456|-77|
|tensor_2d_length_g2s_u8_row32_b7680|tensor|u8|2|7680|在 L2，TMAU 第一次使用|1054|679|-375|
|tensor_2d_length_g2s_u8_row32_b7680|tensor|u8|2|7680|已经提前送入 TMAU|768|677|-91|
|tensor_2d_length_g2s_u8_row32_b8160|tensor|u8|2|8160|在 L2，TMAU 第一次使用|1054|681|-373|
|tensor_2d_length_g2s_u8_row32_b8160|tensor|u8|2|8160|已经提前送入 TMAU|786|701|-85|
|tensor_2d_length_g2s_u8_row32_b8192|tensor|u8|2|8192|在 L2，TMAU 第一次使用|1052|686|-366|
|tensor_2d_length_g2s_u8_row32_b8192|tensor|u8|2|8192|已经提前送入 TMAU|792|701|-91|
|tensor_2d_length_g2s_u8_row32_b896|tensor|u8|2|896|在 L2，TMAU 第一次使用|829|476|-353|
|tensor_2d_length_g2s_u8_row32_b896|tensor|u8|2|896|已经提前送入 TMAU|576|484|-92|
|tensor_2d_length_g2s_u8_row32_b96|tensor|u8|2|96|在 L2，TMAU 第一次使用|825|459|-366|
|tensor_2d_length_g2s_u8_row32_b96|tensor|u8|2|96|已经提前送入 TMAU|555|460|-95|
|tensor_2d_length_g2s_u8_row32_b992|tensor|u8|2|992|在 L2，TMAU 第一次使用|818|487|-331|
|tensor_2d_length_g2s_u8_row32_b992|tensor|u8|2|992|已经提前送入 TMAU|584|487|-97|

## tensor_2d_length_u8_row64

|测试项|类型|数据格式|rank|字节数|TensorMap 状态|第一次|第二次|第二次-第一次|
|---|---|---|---:|---:|---|---:|---:|---:|
|tensor_2d_length_g2s_u8_row64_b10240|tensor|u8|2|10240|在 L2，TMAU 第一次使用|977|607|-370|
|tensor_2d_length_g2s_u8_row64_b10240|tensor|u8|2|10240|已经提前送入 TMAU|703|618|-85|
|tensor_2d_length_g2s_u8_row64_b10752|tensor|u8|2|10752|在 L2，TMAU 第一次使用|993|614|-379|
|tensor_2d_length_g2s_u8_row64_b10752|tensor|u8|2|10752|已经提前送入 TMAU|716|609|-107|
|tensor_2d_length_g2s_u8_row64_b11264|tensor|u8|2|11264|在 L2，TMAU 第一次使用|969|638|-331|
|tensor_2d_length_g2s_u8_row64_b11264|tensor|u8|2|11264|已经提前送入 TMAU|720|629|-91|
|tensor_2d_length_g2s_u8_row64_b11776|tensor|u8|2|11776|在 L2，TMAU 第一次使用|980|633|-347|
|tensor_2d_length_g2s_u8_row64_b11776|tensor|u8|2|11776|已经提前送入 TMAU|733|636|-97|
|tensor_2d_length_g2s_u8_row64_b12288|tensor|u8|2|12288|在 L2，TMAU 第一次使用|1006|645|-361|
|tensor_2d_length_g2s_u8_row64_b12288|tensor|u8|2|12288|已经提前送入 TMAU|730|639|-91|
|tensor_2d_length_g2s_u8_row64_b12800|tensor|u8|2|12800|在 L2，TMAU 第一次使用|997|638|-359|
|tensor_2d_length_g2s_u8_row64_b12800|tensor|u8|2|12800|已经提前送入 TMAU|738|653|-85|
|tensor_2d_length_g2s_u8_row64_b13312|tensor|u8|2|13312|在 L2，TMAU 第一次使用|1006|651|-355|
|tensor_2d_length_g2s_u8_row64_b13312|tensor|u8|2|13312|已经提前送入 TMAU|740|649|-91|
|tensor_2d_length_g2s_u8_row64_b13824|tensor|u8|2|13824|在 L2，TMAU 第一次使用|1021|673|-348|
|tensor_2d_length_g2s_u8_row64_b13824|tensor|u8|2|13824|已经提前送入 TMAU|754|663|-91|
|tensor_2d_length_g2s_u8_row64_b14336|tensor|u8|2|14336|在 L2，TMAU 第一次使用|1045|655|-390|
|tensor_2d_length_g2s_u8_row64_b14336|tensor|u8|2|14336|已经提前送入 TMAU|762|671|-91|
|tensor_2d_length_g2s_u8_row64_b14848|tensor|u8|2|14848|在 L2，TMAU 第一次使用|1052|679|-373|
|tensor_2d_length_g2s_u8_row64_b14848|tensor|u8|2|14848|已经提前送入 TMAU|778|687|-91|
|tensor_2d_length_g2s_u8_row64_b15360|tensor|u8|2|15360|在 L2，TMAU 第一次使用|1065|696|-369|
|tensor_2d_length_g2s_u8_row64_b15360|tensor|u8|2|15360|已经提前送入 TMAU|782|690|-92|
|tensor_2d_length_g2s_u8_row64_b15872|tensor|u8|2|15872|在 L2，TMAU 第一次使用|1060|703|-357|
|tensor_2d_length_g2s_u8_row64_b15872|tensor|u8|2|15872|已经提前送入 TMAU|805|714|-91|
|tensor_2d_length_g2s_u8_row64_b16320|tensor|u8|2|16320|在 L2，TMAU 第一次使用|1052|705|-347|
|tensor_2d_length_g2s_u8_row64_b16320|tensor|u8|2|16320|已经提前送入 TMAU|816|719|-97|
|tensor_2d_length_g2s_u8_row64_b16384|tensor|u8|2|16384|在 L2，TMAU 第一次使用|1060|705|-355|
|tensor_2d_length_g2s_u8_row64_b16384|tensor|u8|2|16384|已经提前送入 TMAU|799|701|-98|
|tensor_2d_length_g2s_u8_row64_b8128|tensor|u8|2|8128|在 L2，TMAU 第一次使用|929|585|-344|
|tensor_2d_length_g2s_u8_row64_b8128|tensor|u8|2|8128|已经提前送入 TMAU|655|566|-89|
|tensor_2d_length_g2s_u8_row64_b8192|tensor|u8|2|8192|在 L2，TMAU 第一次使用|929|583|-346|
|tensor_2d_length_g2s_u8_row64_b8192|tensor|u8|2|8192|已经提前送入 TMAU|652|561|-91|
|tensor_2d_length_g2s_u8_row64_b8256|tensor|u8|2|8256|在 L2，TMAU 第一次使用|938|594|-344|
|tensor_2d_length_g2s_u8_row64_b8256|tensor|u8|2|8256|已经提前送入 TMAU|682|585|-97|
|tensor_2d_length_g2s_u8_row64_b8704|tensor|u8|2|8704|在 L2，TMAU 第一次使用|938|594|-344|
|tensor_2d_length_g2s_u8_row64_b8704|tensor|u8|2|8704|已经提前送入 TMAU|668|577|-91|
|tensor_2d_length_g2s_u8_row64_b9216|tensor|u8|2|9216|在 L2，TMAU 第一次使用|940|594|-346|
|tensor_2d_length_g2s_u8_row64_b9216|tensor|u8|2|9216|已经提前送入 TMAU|682|591|-91|
|tensor_2d_length_g2s_u8_row64_b9728|tensor|u8|2|9728|在 L2，TMAU 第一次使用|956|601|-355|
|tensor_2d_length_g2s_u8_row64_b9728|tensor|u8|2|9728|已经提前送入 TMAU|692|599|-93|

## 怎样理解两种 Tensor 结果

“在 L2，TMAU 第一次使用”的第一列包含从 L2 取回 TensorMap并交给 TMAU 处理的时间；第二列是同一个 TensorMap 的立即复用。“已经提前送入 TMAU”是独立控制项，第一次发令前显式 prefetch，第二次仍是相同地址的热复用。
