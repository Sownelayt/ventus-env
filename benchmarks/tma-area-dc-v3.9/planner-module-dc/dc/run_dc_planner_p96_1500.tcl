# Module-level TmaV34WindowPlanner comparison at TSMC N12 6T P96.
#
# The frozen V3.8 and experimental V3.9 projects use this identical script.
# Only rtl/planner.sv and VARIANT differ, so the result isolates the 2+3
# mixed-radix pipeline.  Boundary constraints match the standalone TMA runs.

proc getenv_default {name default} {
  if {[info exists ::env($name)] && $::env($name) ne ""} {
    return $::env($name)
  }
  return $default
}

proc remove_named_path_group {group_name} {
  foreach_in_collection old_group [get_path_groups *] {
    if {[get_object_name $old_group] eq $group_name} {
      remove_path_group $group_name
    }
  }
}

proc refresh_planner_path_groups {critical_range} {
  foreach group_name [list PLANNER_REG PLANNER_OUT] {
    remove_named_path_group $group_name
  }
  set regs [all_registers]
  set outputs [all_outputs]
  if {[sizeof_collection $regs] == 0} {
    error "module-level Planner contains no registers"
  }
  if {[sizeof_collection $outputs] == 0} {
    error "module-level Planner contains no outputs"
  }
  group_path -name PLANNER_REG -to $regs \
    -critical_range $critical_range -weight 5.0
  group_path -name PLANNER_OUT -to $outputs \
    -critical_range $critical_range -weight 4.0
  foreach group_name [list PLANNER_REG PLANNER_OUT] {
    set probe [get_timing_paths -group $group_name -max_paths 1]
    if {[sizeof_collection $probe] == 0} {
      error "required path group $group_name contains no timing path"
    }
  }
}

set script_dir [file normalize [file dirname [info script]]]
set project_root [file normalize [file join $script_dir ..]]
cd $project_root

set design_name "TmaV34WindowPlanner"
set variant [getenv_default "VARIANT" "planner_v3_9_rank2p3_p96_1500"]
set clock_port [getenv_default "CLOCK_PORT" "clock"]
set synth_freq_mhz [getenv_default "SYNTH_FREQ_MHZ" 1500.0]
set report_freq_mhz [getenv_default "REPORT_FREQ_MHZ" 1500.0]
set synth_period [expr {1000.0 / $synth_freq_mhz}]
set report_period [expr {1000.0 / $report_freq_mhz}]
set max_cores [getenv_default "MAX_CORES" 6]
set max_area [getenv_default "MAX_AREA" 50000.0]
set critical_range [getenv_default "CRITICAL_RANGE_NS" 0.20]
set rtl_file [file join $project_root rtl planner.sv]
set workdir [getenv_default \
  "OUTDIR" [file join $project_root DC_log $variant]]
set rptdir [file join $workdir report]
set datadir [file join $workdir data]

set target_library_db [getenv_default "TARGET_LIBRARY_DB" \
  "/vmshare/pdk/install/tsmc_n12_CLN12FFCLL/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn12ffcllbwp6t16p96cpd_120a/tcbn12ffcllbwp6t16p96cpdtt1v85c.db"]
set target_library_name [getenv_default "TARGET_LIBRARY_NAME" \
  "tcbn12ffcllbwp6t16p96cpdtt1v85c"]
set operating_condition [getenv_default "OPERATING_CONDITION" "tt1v85c"]
set driving_cell [getenv_default \
  "BOUNDARY_DRIVING_CELL" "BUFFD2BWP6T16P96CPD"]

if {$max_cores > 12} {
  error "MAX_CORES=$max_cores exceeds the per-project limit of 12"
}
if {![file exists $rtl_file]} {
  error "Planner RTL does not exist: $rtl_file"
}
if {![file exists $target_library_db]} {
  error "target library does not exist: $target_library_db"
}

file mkdir $workdir
file mkdir $rptdir
file mkdir $datadir
set target_library [list $target_library_db]
set link_library [list "*" $target_library_db]
define_design_lib WORK -path $workdir
set_app_var alib_library_analysis_path [file join $workdir alib]
set sh_command_log_file [file join $workdir command.log]
set_svf [file join $datadir ${design_name}.svf]
set_host_options -max_cores $max_cores
set_units -time ns

analyze -work WORK -format sverilog $rtl_file
elaborate $design_name -work WORK
current_design $design_name
uniquify
link
check_design > [file join $rptdir check_design_pre_compile.rpt]

set_operating_conditions -library $target_library_name $operating_condition
set_wire_load_model -name ZeroWireload -library $target_library_name
create_clock [get_ports $clock_port] -name $clock_port \
  -period $synth_period -waveform [list 0 [expr {$synth_period / 2.0}]]
set_clock_latency 0.10 [get_clocks $clock_port]
set_clock_uncertainty -setup 0.10 [get_clocks $clock_port]
set_clock_uncertainty -hold 0.03 [get_clocks $clock_port]
set_clock_transition 0.10 [get_clocks $clock_port]
set_dont_touch_network [get_clocks $clock_port]

set data_inputs [remove_from_collection [all_inputs] [get_ports $clock_port]]
set outputs [all_outputs]
set driver [get_lib_cells -quiet ${target_library_name}/${driving_cell}]
if {[sizeof_collection $driver] == 0} {
  error "boundary driving cell is unavailable: ${target_library_name}/${driving_cell}"
}
set_driving_cell -lib_cell $driving_cell $data_inputs
set_input_delay -clock [get_clocks $clock_port] -max 0.12 $data_inputs
set_input_delay -clock [get_clocks $clock_port] -min 0.03 $data_inputs
set_output_delay -clock [get_clocks $clock_port] -max 0.12 $outputs
set_output_delay -clock [get_clocks $clock_port] -min 0.03 $outputs
set_load 0.010 $outputs
set_max_transition 0.12 [current_design]
set_max_fanout 16 [current_design]
set_fix_multiple_port_nets -all -buffer_constants
set verilogout_no_tri true
set_max_area $max_area
set_critical_range $critical_range [current_design]

refresh_planner_path_groups $critical_range
report_path_group > [file join $rptdir path_groups_pre_compile.rpt]
report_resources > [file join $rptdir resources_pre_compile.rpt]
check_timing > [file join $rptdir check_timing_pre_compile.rpt]

if {[string equal -nocase \
    [getenv_default "STOP_BEFORE_COMPILE" "false"] "true"]} {
  puts "INFO: Planner module STOP_BEFORE_COMPILE dry-run passed"
  exit
}

compile_ultra -retime -timing_high_effort_script
update_timing
refresh_planner_path_groups $critical_range
compile_ultra -incremental
change_names -rules verilog -hierarchy

remove_clock [get_clocks $clock_port]
create_clock [get_ports $clock_port] -name $clock_port \
  -period $report_period -waveform [list 0 [expr {$report_period / 2.0}]]
set_clock_latency 0.10 [get_clocks $clock_port]
set_clock_uncertainty -setup 0.10 [get_clocks $clock_port]
set_clock_uncertainty -hold 0.03 [get_clocks $clock_port]
set_clock_transition 0.10 [get_clocks $clock_port]
set_dont_touch_network [get_clocks $clock_port]
set_input_delay -clock [get_clocks $clock_port] -max 0.12 $data_inputs
set_input_delay -clock [get_clocks $clock_port] -min 0.03 $data_inputs
set_output_delay -clock [get_clocks $clock_port] -max 0.12 $outputs
set_output_delay -clock [get_clocks $clock_port] -min 0.03 $outputs
update_timing
refresh_planner_path_groups $critical_range
update_timing

report_qor > [file join $rptdir qor.rpt]
report_area > [file join $rptdir area.rpt]
report_area -hierarchy -nosplit > [file join $rptdir area_hierarchy.rpt]
report_reference -hierarchy > [file join $rptdir reference.rpt]
report_resources > [file join $rptdir resources.rpt]
report_constraints -all_violators > \
  [file join $rptdir constraints_violations.rpt]
check_timing > [file join $rptdir check_timing.rpt]
report_timing -delay_type max -max_paths 200 -nworst 20 -path full \
  -significant_digits 4 > [file join $rptdir timing_setup.rpt]
report_timing -delay_type min -max_paths 100 -nworst 10 -path full \
  -significant_digits 4 > [file join $rptdir timing_hold.rpt]
foreach group_name [list PLANNER_REG PLANNER_OUT] {
  redirect [file join $rptdir timing_group_${group_name}.rpt] {
    report_timing -group $group_name -delay_type max \
      -max_paths 200 -nworst 20 -path full -significant_digits 4
  }
}

set summary [open [file join $rptdir run_summary.rpt] w]
puts $summary "Design: $design_name"
puts $summary "Variant: $variant"
puts $summary "Library: $target_library_name"
puts $summary "Operating_Condition: $operating_condition"
puts $summary "Synthesis_Frequency_MHz: $synth_freq_mhz"
puts $summary "Report_Frequency_MHz: $report_freq_mhz"
puts $summary "Timing_Model: boundary_typical plus ZeroWireload"
puts $summary "Boundary_Driving_Cell: $driving_cell"
puts $summary "Max_Cores: $max_cores"
puts $summary "Max_Area: $max_area"
puts $summary "Critical_Range_ns: $critical_range"
puts $summary "Compile: compile_ultra -retime -timing_high_effort_script; compile_ultra -incremental"
close $summary

write -hierarchy -format ddc \
  -output [file join $datadir ${design_name}.ddc]
write -hierarchy -format verilog \
  -output [file join $datadir ${design_name}_post.v]
write_sdf [file join $datadir ${design_name}.sdf]
puts "INFO: Planner module P96 TT1V85C 1.5GHz synthesis finished"
exit
