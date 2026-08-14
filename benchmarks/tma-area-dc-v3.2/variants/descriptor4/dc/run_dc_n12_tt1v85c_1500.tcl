# Standalone Ventus TMA V3.2 area synthesis at TSMC N12 TT 1.0 V 85 C.

proc getenv_default {name default} {
  if {[info exists ::env($name)] && $::env($name) ne ""} {
    return $::env($name)
  }
  return $default
}

proc apply_clock {port period} {
  set previous [get_clocks -quiet $port]
  if {[sizeof_collection $previous] > 0} {
    remove_clock $previous
  }
  create_clock [get_ports $port] -name $port -period $period \
    -waveform [list 0 [expr {$period / 2.0}]]
  set_clock_latency 0.10 [get_clocks $port]
  set_clock_uncertainty -setup 0.10 [get_clocks $port]
  set_clock_uncertainty -hold 0.03 [get_clocks $port]
  set_clock_transition 0.10 [get_clocks $port]
  set_dont_touch_network [get_clocks $port]
}

set script_dir [file normalize [file dirname [info script]]]
set export_root [file normalize [file join $script_dir ..]]
cd $export_root

set design_name       [getenv_default "DESIGN_TOP" "tma"]
set clock_port        [getenv_default "CLOCK_PORT" "clock"]
set synth_freq_mhz    [getenv_default "SYNTH_FREQ_MHZ" 1500.0]
set report_freq_mhz   [getenv_default "REPORT_FREQ_MHZ" 1500.0]
set synth_period      [expr {1000.0 / $synth_freq_mhz}]
set report_period     [expr {1000.0 / $report_freq_mhz}]
set max_cores         [getenv_default "MAX_CORES" 12]
set rtl_filelist      [getenv_default "RTL_FILELIST" \
  [file join $export_root dc filelists tma_core_sv.f]]
set target_library_db [getenv_default "TARGET_LIBRARY_DB" \
  "/vmshare/pdk/install/tsmc_n12_CLN12FFCLL/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn12ffcllbwp16p90cpd_100c/tcbn12ffcllbwp16p90cpdtt1v85c.db"]
set target_library_name [getenv_default "TARGET_LIBRARY_NAME" \
  "tcbn12ffcllbwp16p90cpdtt1v85c"]
set operating_condition [getenv_default "OPERATING_CONDITION" "tt1v85c"]
set variant [getenv_default "VARIANT" "tma_v3_2_area"]
set workdir [getenv_default "OUTDIR" [file join $export_root DC_log $variant]]
set rptdir [file join $workdir report]
set datadir [file join $workdir data]
file mkdir $workdir
file mkdir $rptdir
file mkdir $datadir

if {$max_cores > 12} {
  error "MAX_CORES=$max_cores exceeds the project limit of 12"
}
if {![file exists $target_library_db]} {
  error "TARGET_LIBRARY_DB does not exist: $target_library_db"
}
if {![file exists $rtl_filelist]} {
  error "RTL_FILELIST does not exist: $rtl_filelist"
}

set target_library [list $target_library_db]
set link_library [list "*" $target_library_db]
define_design_lib WORK -path $workdir
set_app_var alib_library_analysis_path [file join $workdir alib]
set sh_command_log_file [file join $workdir command.log]
set_svf [file join $datadir ${design_name}.svf]
set_host_options -max_cores $max_cores
set_units -time ns

set files {}
set fp [open $rtl_filelist r]
while {[gets $fp line] >= 0} {
  set line [string trim $line]
  if {$line eq "" || [string index $line 0] eq "#"} {
    continue
  }
  lappend files [file normalize [file join $export_root $line]]
}
close $fp

puts "============================================================"
puts "Ventus TMA V3.2 standalone DC area sweep"
puts "variant=$variant top=$design_name"
puts "process=N12 corner=TT_1.0V_85C library=$target_library_db"
puts "synth=${synth_freq_mhz}MHz period=${synth_period}ns"
puts "report=${report_freq_mhz}MHz period=${report_period}ns"
puts "cores=$max_cores"
puts "============================================================"

foreach file $files {
  puts "INFO: analyze $file"
  if {![analyze -work WORK -format sverilog $file]} {
    error "SystemVerilog analysis failed: $file"
  }
}
elaborate $design_name -work WORK
if {[sizeof_collection [get_designs -quiet $design_name]] == 0} {
  error "Elaboration did not create design: $design_name"
}
current_design $design_name
uniquify
link
check_design > [file join $rptdir check_design_pre_compile.rpt]

set_operating_conditions -library $target_library_name $operating_condition
set_wire_load_model -name ZeroWireload -library $target_library_name
apply_clock $clock_port $synth_period
set_fix_multiple_port_nets -all -buffer_constants
set verilogout_no_tri true
remove_unconnected_ports [get_cells -hier {*}]
set_max_area 0

foreach pattern [list \
    "core/ingress" \
    "core/ingress/windowPlanner" \
    "core/ingress/binder" \
    "core/subsystem" \
    "core/subsystem/descriptor" \
    "core/subsystem/engine"] {
  set cells [get_cells -quiet $pattern]
  if {[sizeof_collection $cells] > 0} {
    set_ungroup $cells false
    puts "INFO: preserve hierarchy $pattern"
  }
}

report_hierarchy > [file join $rptdir hierarchy_pre_compile.rpt]
report_resources > [file join $rptdir resources_pre_compile.rpt]
check_design > [file join $rptdir check_design_post_constraints.rpt]

if {[string equal -nocase [getenv_default "STOP_BEFORE_COMPILE" "false"] "true"]} {
  report_clock -skew > [file join $rptdir dry_run_clock.rpt]
  report_lib $target_library_name > [file join $rptdir library.rpt]
  puts "INFO: STOP_BEFORE_COMPILE dry-run passed"
  exit
}

compile_ultra -retime
compile_ultra -incremental
change_names -rules verilog -hierarchy

apply_clock $clock_port $report_period
update_timing
report_qor > [file join $rptdir qor_summary.rpt]
report_area > [file join $rptdir area.rpt]
report_area -hierarchy -nosplit > [file join $rptdir area_hierarchy.rpt]
report_area -designware > [file join $rptdir area_designware.rpt]
report_timing -delay_type max -max_paths 100 -nworst 10 -path full \
  -significant_digits 4 > [file join $rptdir timing_setup.rpt]
report_timing -delay_type min -max_paths 100 -nworst 10 -path full \
  -significant_digits 4 > [file join $rptdir timing_hold.rpt]
report_power > [file join $rptdir power.rpt]
report_power -hierarchy -levels 5 > [file join $rptdir power_hierarchy.rpt]
report_resources > [file join $rptdir resources.rpt]
report_reference -hierarchy > [file join $rptdir reference.rpt]
report_constraints -all_violators > \
  [file join $rptdir constraints_violations.rpt]
report_clock -skew > [file join $rptdir clock.rpt]

set summary [open [file join $rptdir run_summary.rpt] w]
puts $summary "Design: $design_name"
puts $summary "Variant: $variant"
puts $summary "Process: TSMC N12 CLN12FFCLL"
puts $summary "Target_Library_DB: $target_library_db"
puts $summary "Operating_Condition: $operating_condition"
puts $summary "Synthesis_Frequency_MHz: $synth_freq_mhz"
puts $summary "Synthesis_Clock_Period_ns: $synth_period"
puts $summary "Report_Frequency_MHz: $report_freq_mhz"
puts $summary "Report_Clock_Period_ns: $report_period"
puts $summary "PMU: disabled at elaboration"
puts $summary "Max_Cores: $max_cores"
puts $summary "Wire_Load: ZeroWireload"
puts $summary "Max_Area: 0"
puts $summary "Compile: compile_ultra -retime; compile_ultra -incremental"
close $summary

write -hierarchy -format ddc -output [file join $datadir ${design_name}.ddc]
write -hierarchy -format verilog \
  -output [file join $datadir ${design_name}_post.v]
write_sdf [file join $datadir ${design_name}.sdf]
puts "INFO: TMA V3.2 N12 TT 1.0V 85C 1.5GHz synthesis finished"
exit
