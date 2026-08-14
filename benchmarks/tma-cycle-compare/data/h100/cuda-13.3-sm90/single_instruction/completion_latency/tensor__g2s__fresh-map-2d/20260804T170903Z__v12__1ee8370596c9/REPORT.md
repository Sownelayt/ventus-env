# H100 G2S 完成周期

每个测试点只分配一个新的 TensorMap。第一次测量是该地址第一次进入 TMAU，完成后在同一个 kernel 内立刻用完全相同的地址和内容测第二次。主测试没有 TensorMap prefetch；普通全局读取只把 descriptor 放进 L2。

表格分别显示第一次和第二次，不再把两次平均成一个数字。每个数字来自一轮固定的 640 点扫描：80 个静态 kernel bank 各负责 8 个点，bank 在计时外选择；计时路径没有索引跳转，相邻点静态增加一条比较、一条统一分支和一条不访问数据通路的 PM-event。请求步长为三周期，实际物理间隔由 clock64 记录，原始测量点、地址和指纹保存在同目录压缩 CSV。

测试组：`tensor__g2s__fresh-map-2d`

## tensor_2d_length_u16_row128

|测试项|类型|数据格式|rank|字节数|TensorMap 状态|第一次|第二次|第二次-第一次|
|---|---|---|---:|---:|---|---:|---:|---:|
|tensor_2d_length_g2s_u16_row128_b16384|tensor|u16|2|16384|在 L2，TMAU 第一次使用|865|541|-324|
|tensor_2d_length_g2s_u16_row128_b16384|tensor|u16|2|16384|已经提前送入 TMAU|616|547|-69|
|tensor_2d_length_g2s_u16_row128_b32768|tensor|u16|2|32768|在 L2，TMAU 第一次使用|1007|688|-319|
|tensor_2d_length_g2s_u16_row128_b32768|tensor|u16|2|32768|已经提前送入 TMAU|765|697|-68|

## tensor_2d_length_u16_row32

|测试项|类型|数据格式|rank|字节数|TensorMap 状态|第一次|第二次|第二次-第一次|
|---|---|---|---:|---:|---|---:|---:|---:|
|tensor_2d_length_g2s_u16_row32_b1024|tensor|u16|2|1024|在 L2，TMAU 第一次使用|730|418|-312|
|tensor_2d_length_g2s_u16_row32_b1024|tensor|u16|2|1024|已经提前送入 TMAU|470|418|-52|
|tensor_2d_length_g2s_u16_row32_b128|tensor|u16|2|128|在 L2，TMAU 第一次使用|680|386|-294|
|tensor_2d_length_g2s_u16_row32_b128|tensor|u16|2|128|已经提前送入 TMAU|465|385|-80|
|tensor_2d_length_g2s_u16_row32_b256|tensor|u16|2|256|在 L2，TMAU 第一次使用|682|364|-318|
|tensor_2d_length_g2s_u16_row32_b256|tensor|u16|2|256|已经提前送入 TMAU|457|366|-91|
|tensor_2d_length_g2s_u16_row32_b4096|tensor|u16|2|4096|在 L2，TMAU 第一次使用|819|511|-308|
|tensor_2d_length_g2s_u16_row32_b4096|tensor|u16|2|4096|已经提前送入 TMAU|574|508|-66|
|tensor_2d_length_g2s_u16_row32_b8192|tensor|u16|2|8192|在 L2，TMAU 第一次使用|946|620|-326|
|tensor_2d_length_g2s_u16_row32_b8192|tensor|u16|2|8192|已经提前送入 TMAU|709|628|-81|

## tensor_2d_length_u8_row128

|测试项|类型|数据格式|rank|字节数|TensorMap 状态|第一次|第二次|第二次-第一次|
|---|---|---|---:|---:|---|---:|---:|---:|
|tensor_2d_length_g2s_u8_row128_b1024|tensor|u8|2|1024|在 L2，TMAU 第一次使用|725|415|-310|
|tensor_2d_length_g2s_u8_row128_b1024|tensor|u8|2|1024|已经提前送入 TMAU|491|409|-82|
|tensor_2d_length_g2s_u8_row128_b128|tensor|u8|2|128|在 L2，TMAU 第一次使用|710|386|-324|
|tensor_2d_length_g2s_u8_row128_b128|tensor|u8|2|128|已经提前送入 TMAU|465|385|-80|
|tensor_2d_length_g2s_u8_row128_b16256|tensor|u8|2|16256|在 L2，TMAU 第一次使用|859|543|-316|
|tensor_2d_length_g2s_u8_row128_b16256|tensor|u8|2|16256|已经提前送入 TMAU|608|547|-61|
|tensor_2d_length_g2s_u8_row128_b16384|tensor|u8|2|16384|在 L2，TMAU 第一次使用|872|548|-324|
|tensor_2d_length_g2s_u8_row128_b16384|tensor|u8|2|16384|已经提前送入 TMAU|616|550|-66|
|tensor_2d_length_g2s_u8_row128_b16512|tensor|u8|2|16512|在 L2，TMAU 第一次使用|871|538|-333|
|tensor_2d_length_g2s_u8_row128_b16512|tensor|u8|2|16512|已经提前送入 TMAU|613|538|-75|
|tensor_2d_length_g2s_u8_row128_b17408|tensor|u8|2|17408|在 L2，TMAU 第一次使用|859|557|-302|
|tensor_2d_length_g2s_u8_row128_b17408|tensor|u8|2|17408|已经提前送入 TMAU|624|560|-64|
|tensor_2d_length_g2s_u8_row128_b18432|tensor|u8|2|18432|在 L2，TMAU 第一次使用|878|564|-314|
|tensor_2d_length_g2s_u8_row128_b18432|tensor|u8|2|18432|已经提前送入 TMAU|640|566|-74|
|tensor_2d_length_g2s_u8_row128_b2048|tensor|u8|2|2048|在 L2，TMAU 第一次使用|743|426|-317|
|tensor_2d_length_g2s_u8_row128_b2048|tensor|u8|2|2048|已经提前送入 TMAU|489|428|-61|
|tensor_2d_length_g2s_u8_row128_b20480|tensor|u8|2|20480|在 L2，TMAU 第一次使用|893|583|-310|
|tensor_2d_length_g2s_u8_row128_b20480|tensor|u8|2|20480|已经提前送入 TMAU|651|584|-67|
|tensor_2d_length_g2s_u8_row128_b22528|tensor|u8|2|22528|在 L2，TMAU 第一次使用|920|605|-315|
|tensor_2d_length_g2s_u8_row128_b22528|tensor|u8|2|22528|已经提前送入 TMAU|670|601|-69|
|tensor_2d_length_g2s_u8_row128_b24576|tensor|u8|2|24576|在 L2，TMAU 第一次使用|931|614|-317|
|tensor_2d_length_g2s_u8_row128_b24576|tensor|u8|2|24576|已经提前送入 TMAU|685|619|-66|
|tensor_2d_length_g2s_u8_row128_b256|tensor|u8|2|256|在 L2，TMAU 第一次使用|706|378|-328|
|tensor_2d_length_g2s_u8_row128_b256|tensor|u8|2|256|已经提前送入 TMAU|470|375|-95|
|tensor_2d_length_g2s_u8_row128_b28672|tensor|u8|2|28672|在 L2，TMAU 第一次使用|977|653|-324|
|tensor_2d_length_g2s_u8_row128_b28672|tensor|u8|2|28672|已经提前送入 TMAU|723|658|-65|
|tensor_2d_length_g2s_u8_row128_b32768|tensor|u8|2|32768|在 L2，TMAU 第一次使用|1009|692|-317|
|tensor_2d_length_g2s_u8_row128_b32768|tensor|u8|2|32768|已经提前送入 TMAU|771|691|-80|
|tensor_2d_length_g2s_u8_row128_b4096|tensor|u8|2|4096|在 L2，TMAU 第一次使用|745|436|-309|
|tensor_2d_length_g2s_u8_row128_b4096|tensor|u8|2|4096|已经提前送入 TMAU|507|438|-69|
|tensor_2d_length_g2s_u8_row128_b512|tensor|u8|2|512|在 L2，TMAU 第一次使用|734|402|-332|
|tensor_2d_length_g2s_u8_row128_b512|tensor|u8|2|512|已经提前送入 TMAU|465|404|-61|
|tensor_2d_length_g2s_u8_row128_b8192|tensor|u8|2|8192|在 L2，TMAU 第一次使用|793|474|-319|
|tensor_2d_length_g2s_u8_row128_b8192|tensor|u8|2|8192|已经提前送入 TMAU|541|475|-66|

## tensor_2d_length_u8_row32

|测试项|类型|数据格式|rank|字节数|TensorMap 状态|第一次|第二次|第二次-第一次|
|---|---|---|---:|---:|---|---:|---:|---:|
|tensor_2d_length_g2s_u8_row32_b1024|tensor|u8|2|1024|在 L2，TMAU 第一次使用|752|397|-355|
|tensor_2d_length_g2s_u8_row32_b1024|tensor|u8|2|1024|已经提前送入 TMAU|497|404|-93|
|tensor_2d_length_g2s_u8_row32_b1056|tensor|u8|2|1056|在 L2，TMAU 第一次使用|743|426|-317|
|tensor_2d_length_g2s_u8_row32_b1056|tensor|u8|2|1056|已经提前送入 TMAU|497|433|-64|
|tensor_2d_length_g2s_u8_row32_b1152|tensor|u8|2|1152|在 L2，TMAU 第一次使用|721|422|-299|
|tensor_2d_length_g2s_u8_row32_b1152|tensor|u8|2|1152|已经提前送入 TMAU|489|422|-67|
|tensor_2d_length_g2s_u8_row32_b128|tensor|u8|2|128|在 L2，TMAU 第一次使用|719|383|-336|
|tensor_2d_length_g2s_u8_row32_b128|tensor|u8|2|128|已经提前送入 TMAU|457|385|-72|
|tensor_2d_length_g2s_u8_row32_b1280|tensor|u8|2|1280|在 L2，TMAU 第一次使用|749|420|-329|
|tensor_2d_length_g2s_u8_row32_b1280|tensor|u8|2|1280|已经提前送入 TMAU|470|422|-48|
|tensor_2d_length_g2s_u8_row32_b1408|tensor|u8|2|1408|在 L2，TMAU 第一次使用|721|439|-282|
|tensor_2d_length_g2s_u8_row32_b1408|tensor|u8|2|1408|已经提前送入 TMAU|497|440|-57|
|tensor_2d_length_g2s_u8_row32_b1536|tensor|u8|2|1536|在 L2，TMAU 第一次使用|745|436|-309|
|tensor_2d_length_g2s_u8_row32_b1536|tensor|u8|2|1536|已经提前送入 TMAU|507|438|-69|
|tensor_2d_length_g2s_u8_row32_b160|tensor|u8|2|160|在 L2，TMAU 第一次使用|706|407|-299|
|tensor_2d_length_g2s_u8_row32_b160|tensor|u8|2|160|已经提前送入 TMAU|438|404|-34|
|tensor_2d_length_g2s_u8_row32_b1664|tensor|u8|2|1664|在 L2，TMAU 第一次使用|754|431|-323|
|tensor_2d_length_g2s_u8_row32_b1664|tensor|u8|2|1664|已经提前送入 TMAU|497|433|-64|
|tensor_2d_length_g2s_u8_row32_b1792|tensor|u8|2|1792|在 L2，TMAU 第一次使用|763|442|-321|
|tensor_2d_length_g2s_u8_row32_b1792|tensor|u8|2|1792|已经提前送入 TMAU|523|442|-81|
|tensor_2d_length_g2s_u8_row32_b192|tensor|u8|2|192|在 L2，TMAU 第一次使用|691|397|-294|
|tensor_2d_length_g2s_u8_row32_b192|tensor|u8|2|192|已经提前送入 TMAU|438|391|-47|
|tensor_2d_length_g2s_u8_row32_b1920|tensor|u8|2|1920|在 L2，TMAU 第一次使用|754|436|-318|
|tensor_2d_length_g2s_u8_row32_b1920|tensor|u8|2|1920|已经提前送入 TMAU|523|438|-85|
|tensor_2d_length_g2s_u8_row32_b2016|tensor|u8|2|2016|在 L2，TMAU 第一次使用|763|444|-319|
|tensor_2d_length_g2s_u8_row32_b2016|tensor|u8|2|2016|已经提前送入 TMAU|515|440|-75|
|tensor_2d_length_g2s_u8_row32_b2048|tensor|u8|2|2048|在 L2，TMAU 第一次使用|782|446|-336|
|tensor_2d_length_g2s_u8_row32_b2048|tensor|u8|2|2048|已经提前送入 TMAU|497|446|-51|
|tensor_2d_length_g2s_u8_row32_b2080|tensor|u8|2|2080|在 L2，TMAU 第一次使用|791|450|-341|
|tensor_2d_length_g2s_u8_row32_b2080|tensor|u8|2|2080|已经提前送入 TMAU|526|444|-82|
|tensor_2d_length_g2s_u8_row32_b224|tensor|u8|2|224|在 L2，TMAU 第一次使用|716|407|-309|
|tensor_2d_length_g2s_u8_row32_b224|tensor|u8|2|224|已经提前送入 TMAU|438|404|-34|
|tensor_2d_length_g2s_u8_row32_b256|tensor|u8|2|256|在 L2，TMAU 第一次使用|701|402|-299|
|tensor_2d_length_g2s_u8_row32_b256|tensor|u8|2|256|已经提前送入 TMAU|438|391|-47|
|tensor_2d_length_g2s_u8_row32_b2560|tensor|u8|2|2560|在 L2，TMAU 第一次使用|793|446|-347|
|tensor_2d_length_g2s_u8_row32_b2560|tensor|u8|2|2560|已经提前送入 TMAU|523|446|-77|
|tensor_2d_length_g2s_u8_row32_b288|tensor|u8|2|288|在 L2，TMAU 第一次使用|721|386|-335|
|tensor_2d_length_g2s_u8_row32_b288|tensor|u8|2|288|已经提前送入 TMAU|465|389|-76|
|tensor_2d_length_g2s_u8_row32_b3072|tensor|u8|2|3072|在 L2，TMAU 第一次使用|782|487|-295|
|tensor_2d_length_g2s_u8_row32_b3072|tensor|u8|2|3072|已经提前送入 TMAU|560|494|-66|
|tensor_2d_length_g2s_u8_row32_b32|tensor|u8|2|32|在 L2，TMAU 第一次使用|710|400|-310|
|tensor_2d_length_g2s_u8_row32_b32|tensor|u8|2|32|已经提前送入 TMAU|465|401|-64|
|tensor_2d_length_g2s_u8_row32_b320|tensor|u8|2|320|在 L2，TMAU 第一次使用|695|386|-309|
|tensor_2d_length_g2s_u8_row32_b320|tensor|u8|2|320|已经提前送入 TMAU|465|385|-80|
|tensor_2d_length_g2s_u8_row32_b352|tensor|u8|2|352|在 L2，TMAU 第一次使用|719|386|-333|
|tensor_2d_length_g2s_u8_row32_b352|tensor|u8|2|352|已经提前送入 TMAU|465|389|-76|
|tensor_2d_length_g2s_u8_row32_b3584|tensor|u8|2|3584|在 L2，TMAU 第一次使用|797|509|-288|
|tensor_2d_length_g2s_u8_row32_b3584|tensor|u8|2|3584|已经提前送入 TMAU|576|484|-92|
|tensor_2d_length_g2s_u8_row32_b384|tensor|u8|2|384|在 L2，TMAU 第一次使用|713|386|-327|
|tensor_2d_length_g2s_u8_row32_b384|tensor|u8|2|384|已经提前送入 TMAU|465|385|-80|
|tensor_2d_length_g2s_u8_row32_b4064|tensor|u8|2|4064|在 L2，TMAU 第一次使用|845|504|-341|
|tensor_2d_length_g2s_u8_row32_b4064|tensor|u8|2|4064|已经提前送入 TMAU|595|508|-87|
|tensor_2d_length_g2s_u8_row32_b4096|tensor|u8|2|4096|在 L2，TMAU 第一次使用|817|504|-313|
|tensor_2d_length_g2s_u8_row32_b4096|tensor|u8|2|4096|已经提前送入 TMAU|595|508|-87|
|tensor_2d_length_g2s_u8_row32_b4128|tensor|u8|2|4128|在 L2，TMAU 第一次使用|850|514|-336|
|tensor_2d_length_g2s_u8_row32_b4128|tensor|u8|2|4128|已经提前送入 TMAU|589|512|-77|
|tensor_2d_length_g2s_u8_row32_b416|tensor|u8|2|416|在 L2，TMAU 第一次使用|728|393|-335|
|tensor_2d_length_g2s_u8_row32_b416|tensor|u8|2|416|已经提前送入 TMAU|457|391|-66|
|tensor_2d_length_g2s_u8_row32_b448|tensor|u8|2|448|在 L2，TMAU 第一次使用|691|383|-308|
|tensor_2d_length_g2s_u8_row32_b448|tensor|u8|2|448|已经提前送入 TMAU|457|375|-82|
|tensor_2d_length_g2s_u8_row32_b4608|tensor|u8|2|4608|在 L2，TMAU 第一次使用|839|517|-322|
|tensor_2d_length_g2s_u8_row32_b4608|tensor|u8|2|4608|已经提前送入 TMAU|589|518|-71|
|tensor_2d_length_g2s_u8_row32_b480|tensor|u8|2|480|在 L2，TMAU 第一次使用|728|389|-339|
|tensor_2d_length_g2s_u8_row32_b480|tensor|u8|2|480|已经提前送入 TMAU|465|391|-74|
|tensor_2d_length_g2s_u8_row32_b512|tensor|u8|2|512|在 L2，TMAU 第一次使用|728|374|-354|
|tensor_2d_length_g2s_u8_row32_b512|tensor|u8|2|512|已经提前送入 TMAU|457|391|-66|
|tensor_2d_length_g2s_u8_row32_b5120|tensor|u8|2|5120|在 L2，TMAU 第一次使用|841|536|-305|
|tensor_2d_length_g2s_u8_row32_b5120|tensor|u8|2|5120|已经提前送入 TMAU|616|538|-78|
|tensor_2d_length_g2s_u8_row32_b5632|tensor|u8|2|5632|在 L2，TMAU 第一次使用|859|557|-302|
|tensor_2d_length_g2s_u8_row32_b5632|tensor|u8|2|5632|已经提前送入 TMAU|627|560|-67|
|tensor_2d_length_g2s_u8_row32_b6144|tensor|u8|2|6144|在 L2，TMAU 第一次使用|887|572|-315|
|tensor_2d_length_g2s_u8_row32_b6144|tensor|u8|2|6144|已经提前送入 TMAU|646|571|-75|
|tensor_2d_length_g2s_u8_row32_b64|tensor|u8|2|64|在 L2，TMAU 第一次使用|701|383|-318|
|tensor_2d_length_g2s_u8_row32_b64|tensor|u8|2|64|已经提前送入 TMAU|448|385|-63|
|tensor_2d_length_g2s_u8_row32_b640|tensor|u8|2|640|在 L2，TMAU 第一次使用|719|386|-333|
|tensor_2d_length_g2s_u8_row32_b640|tensor|u8|2|640|已经提前送入 TMAU|465|380|-85|
|tensor_2d_length_g2s_u8_row32_b6656|tensor|u8|2|6656|在 L2，TMAU 第一次使用|889|609|-280|
|tensor_2d_length_g2s_u8_row32_b6656|tensor|u8|2|6656|已经提前送入 TMAU|667|608|-59|
|tensor_2d_length_g2s_u8_row32_b7168|tensor|u8|2|7168|在 L2，TMAU 第一次使用|927|607|-320|
|tensor_2d_length_g2s_u8_row32_b7168|tensor|u8|2|7168|已经提前送入 TMAU|672|608|-64|
|tensor_2d_length_g2s_u8_row32_b768|tensor|u8|2|768|在 L2，TMAU 第一次使用|721|407|-314|
|tensor_2d_length_g2s_u8_row32_b768|tensor|u8|2|768|已经提前送入 TMAU|477|409|-68|
|tensor_2d_length_g2s_u8_row32_b7680|tensor|u8|2|7680|在 L2，TMAU 第一次使用|927|625|-302|
|tensor_2d_length_g2s_u8_row32_b7680|tensor|u8|2|7680|已经提前送入 TMAU|699|625|-74|
|tensor_2d_length_g2s_u8_row32_b8160|tensor|u8|2|8160|在 L2，TMAU 第一次使用|955|644|-311|
|tensor_2d_length_g2s_u8_row32_b8160|tensor|u8|2|8160|已经提前送入 TMAU|715|646|-69|
|tensor_2d_length_g2s_u8_row32_b8192|tensor|u8|2|8192|在 L2，TMAU 第一次使用|955|638|-317|
|tensor_2d_length_g2s_u8_row32_b8192|tensor|u8|2|8192|已经提前送入 TMAU|720|630|-90|
|tensor_2d_length_g2s_u8_row32_b896|tensor|u8|2|896|在 L2，TMAU 第一次使用|725|400|-325|
|tensor_2d_length_g2s_u8_row32_b896|tensor|u8|2|896|已经提前送入 TMAU|470|401|-69|
|tensor_2d_length_g2s_u8_row32_b96|tensor|u8|2|96|在 L2，TMAU 第一次使用|719|400|-319|
|tensor_2d_length_g2s_u8_row32_b96|tensor|u8|2|96|已经提前送入 TMAU|465|401|-64|
|tensor_2d_length_g2s_u8_row32_b992|tensor|u8|2|992|在 L2，TMAU 第一次使用|739|407|-332|
|tensor_2d_length_g2s_u8_row32_b992|tensor|u8|2|992|已经提前送入 TMAU|477|409|-68|

## tensor_2d_length_u8_row64

|测试项|类型|数据格式|rank|字节数|TensorMap 状态|第一次|第二次|第二次-第一次|
|---|---|---|---:|---:|---|---:|---:|---:|
|tensor_2d_length_g2s_u8_row64_b10240|tensor|u8|2|10240|在 L2，TMAU 第一次使用|850|548|-302|
|tensor_2d_length_g2s_u8_row64_b10240|tensor|u8|2|10240|已经提前送入 TMAU|624|547|-77|
|tensor_2d_length_g2s_u8_row64_b10752|tensor|u8|2|10752|在 L2，TMAU 第一次使用|869|568|-301|
|tensor_2d_length_g2s_u8_row64_b10752|tensor|u8|2|10752|已经提前送入 TMAU|613|547|-66|
|tensor_2d_length_g2s_u8_row64_b11264|tensor|u8|2|11264|在 L2，TMAU 第一次使用|878|564|-314|
|tensor_2d_length_g2s_u8_row64_b11264|tensor|u8|2|11264|已经提前送入 TMAU|637|562|-75|
|tensor_2d_length_g2s_u8_row64_b11776|tensor|u8|2|11776|在 L2，TMAU 第一次使用|887|577|-310|
|tensor_2d_length_g2s_u8_row64_b11776|tensor|u8|2|11776|已经提前送入 TMAU|648|571|-77|
|tensor_2d_length_g2s_u8_row64_b12288|tensor|u8|2|12288|在 L2，TMAU 第一次使用|878|572|-306|
|tensor_2d_length_g2s_u8_row64_b12288|tensor|u8|2|12288|已经提前送入 TMAU|656|582|-74|
|tensor_2d_length_g2s_u8_row64_b12800|tensor|u8|2|12800|在 L2，TMAU 第一次使用|898|590|-308|
|tensor_2d_length_g2s_u8_row64_b12800|tensor|u8|2|12800|已经提前送入 TMAU|656|577|-79|
|tensor_2d_length_g2s_u8_row64_b13312|tensor|u8|2|13312|在 L2，TMAU 第一次使用|898|601|-297|
|tensor_2d_length_g2s_u8_row64_b13312|tensor|u8|2|13312|已经提前送入 TMAU|664|601|-63|
|tensor_2d_length_g2s_u8_row64_b13824|tensor|u8|2|13824|在 L2，TMAU 第一次使用|924|605|-319|
|tensor_2d_length_g2s_u8_row64_b13824|tensor|u8|2|13824|已经提前送入 TMAU|667|601|-66|
|tensor_2d_length_g2s_u8_row64_b14336|tensor|u8|2|14336|在 L2，TMAU 第一次使用|920|612|-308|
|tensor_2d_length_g2s_u8_row64_b14336|tensor|u8|2|14336|已经提前送入 TMAU|680|610|-70|
|tensor_2d_length_g2s_u8_row64_b14848|tensor|u8|2|14848|在 L2，TMAU 第一次使用|935|620|-315|
|tensor_2d_length_g2s_u8_row64_b14848|tensor|u8|2|14848|已经提前送入 TMAU|694|622|-72|
|tensor_2d_length_g2s_u8_row64_b15360|tensor|u8|2|15360|在 L2，TMAU 第一次使用|935|614|-321|
|tensor_2d_length_g2s_u8_row64_b15360|tensor|u8|2|15360|已经提前送入 TMAU|694|628|-66|
|tensor_2d_length_g2s_u8_row64_b15872|tensor|u8|2|15872|在 L2，TMAU 第一次使用|946|649|-297|
|tensor_2d_length_g2s_u8_row64_b15872|tensor|u8|2|15872|已经提前送入 TMAU|699|632|-67|
|tensor_2d_length_g2s_u8_row64_b16320|tensor|u8|2|16320|在 L2，TMAU 第一次使用|946|638|-308|
|tensor_2d_length_g2s_u8_row64_b16320|tensor|u8|2|16320|已经提前送入 TMAU|720|638|-82|
|tensor_2d_length_g2s_u8_row64_b16384|tensor|u8|2|16384|在 L2，TMAU 第一次使用|965|633|-332|
|tensor_2d_length_g2s_u8_row64_b16384|tensor|u8|2|16384|已经提前送入 TMAU|712|649|-63|
|tensor_2d_length_g2s_u8_row64_b8128|tensor|u8|2|8128|在 L2，TMAU 第一次使用|826|514|-312|
|tensor_2d_length_g2s_u8_row64_b8128|tensor|u8|2|8128|已经提前送入 TMAU|598|514|-84|
|tensor_2d_length_g2s_u8_row64_b8192|tensor|u8|2|8192|在 L2，TMAU 第一次使用|848|498|-350|
|tensor_2d_length_g2s_u8_row64_b8192|tensor|u8|2|8192|已经提前送入 TMAU|595|523|-72|
|tensor_2d_length_g2s_u8_row64_b8256|tensor|u8|2|8256|在 L2，TMAU 第一次使用|848|535|-313|
|tensor_2d_length_g2s_u8_row64_b8256|tensor|u8|2|8256|已经提前送入 TMAU|613|532|-81|
|tensor_2d_length_g2s_u8_row64_b8704|tensor|u8|2|8704|在 L2，TMAU 第一次使用|841|529|-312|
|tensor_2d_length_g2s_u8_row64_b8704|tensor|u8|2|8704|已经提前送入 TMAU|595|518|-77|
|tensor_2d_length_g2s_u8_row64_b9216|tensor|u8|2|9216|在 L2，TMAU 第一次使用|845|523|-322|
|tensor_2d_length_g2s_u8_row64_b9216|tensor|u8|2|9216|已经提前送入 TMAU|605|526|-79|
|tensor_2d_length_g2s_u8_row64_b9728|tensor|u8|2|9728|在 L2，TMAU 第一次使用|848|535|-313|
|tensor_2d_length_g2s_u8_row64_b9728|tensor|u8|2|9728|已经提前送入 TMAU|608|532|-76|

## 怎样理解两种 Tensor 结果

“在 L2，TMAU 第一次使用”的第一列包含从 L2 取回 TensorMap并交给 TMAU 处理的时间；第二列是同一个 TensorMap 的立即复用。“已经提前送入 TMAU”是独立控制项，第一次发令前显式 prefetch，第二次仍是相同地址的热复用。
