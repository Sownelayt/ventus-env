# Standalone Ventus TMA V3.7 all-path synthesis at TSMC N12 TT 1.0 V 85 C.
#
# TIMING_MODEL selects one of two deliberately separate constraint sets:
#   zero_wire       - matched comparison with the historical V3.4 run.
#   boundary_typical - same internal ZeroWireload assumption plus an explicit
#                      block-boundary drive/load and I/O timing envelope.

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

proc apply_boundary_typical {
    clock_port target_library_name driving_cell
    input_delay_max input_delay_min output_delay_max output_delay_min
    output_load max_transition max_fanout} {
  set data_inputs [remove_from_collection [all_inputs] [get_ports $clock_port]]
  set outputs [all_outputs]
  set driver [get_lib_cells -quiet \
    ${target_library_name}/${driving_cell}]
  if {[sizeof_collection $driver] == 0} {
    error "Boundary driving cell is unavailable: ${target_library_name}/${driving_cell}"
  }
  if {[sizeof_collection $data_inputs] > 0} {
    set_driving_cell -lib_cell $driving_cell $data_inputs
    set_input_delay -clock [get_clocks $clock_port] \
      -max $input_delay_max $data_inputs
    set_input_delay -clock [get_clocks $clock_port] \
      -min $input_delay_min $data_inputs
  }
  if {[sizeof_collection $outputs] > 0} {
    set_output_delay -clock [get_clocks $clock_port] \
      -max $output_delay_max $outputs
    set_output_delay -clock [get_clocks $clock_port] \
      -min $output_delay_min $outputs
    set_load $output_load $outputs
  }
  set_max_transition $max_transition [current_design]
  set_max_fanout $max_fanout [current_design]
}

# Retime can replace and rename registers.  Path groups built only before
# compile therefore become empty and silently stop steering the incremental
# pass.  Rebuild endpoint collections from surviving module ref_name values at
# every optimization boundary, then prove that each material TMA group owns a
# real timing path.
proc registers_under_ref {ref_pattern} {
  set all_regs [all_registers]
  set result [filter_collection \
    $all_regs "full_name == __tma_no_such_register__"]
  set owners [get_cells -hierarchical -quiet \
    -filter "ref_name =~ $ref_pattern"]
  foreach_in_collection owner $owners {
    set owner_name [get_object_name $owner]
    set owned [filter_collection \
      $all_regs "full_name =~ ${owner_name}/*"]
    set result [add_to_collection $result $owned]
  }
  return $result
}

proc remove_named_path_group {group_name} {
  foreach_in_collection old_group [get_path_groups *] {
    if {[get_object_name $old_group] eq $group_name} {
      remove_path_group $old_group
    }
  }
}

proc create_path_class_group {
    group_name from_objects to_objects critical_range weight required} {
  if {[sizeof_collection $from_objects] == 0 || \
      [sizeof_collection $to_objects] == 0} {
    if {$required} {
      error "empty endpoint collection for required path group $group_name"
    }
    puts "INFO: optional path group $group_name has an empty endpoint class"
    return
  }
  group_path -name $group_name -from $from_objects -to $to_objects \
    -critical_range $critical_range -weight $weight
  # get_timing_paths does not accept a collection of sequential cells as a
  # direct -from/-to list in this DC release.  Probe the just-created group,
  # which resolves the registers to legal timing start/end pins internally.
  set probe [get_timing_paths -group $group_name -max_paths 1]
  if {[sizeof_collection $probe] == 0} {
    if {$required} {
      error "no timing path exists for required path group $group_name"
    }
    puts "INFO: optional path group $group_name has no timing path"
  }
}

proc create_endpoint_group {
    group_name endpoint_regs critical_range weight} {
  if {[sizeof_collection $endpoint_regs] == 0} {
    error "no registers found for required TMA group $group_name"
  }
  group_path -name $group_name -to $endpoint_regs \
    -critical_range $critical_range -weight $weight
  set probe [get_timing_paths -group $group_name -max_paths 1]
  if {[sizeof_collection $probe] == 0} {
    error "required TMA path group $group_name contains no timing path"
  }
}

proc refresh_tma_path_groups {critical_range clock_port} {
  foreach group_name [list \
      REG_TO_REG INPUT_TO_REG REG_TO_OUTPUT INPUT_TO_OUTPUT \
      INGRESS PLANNER BINDER DESCRIPTOR ENGINE DMA_CORE_CTRL] {
    remove_named_path_group $group_name
  }

  set all_regs [all_registers]
  set planner_regs [registers_under_ref "TmaV34WindowPlanner*"]
  set binder_regs [registers_under_ref "TmaV2CommandBinder*"]
  set descriptor_regs [registers_under_ref "TmaV2DescriptorService*"]
  set engine_regs [registers_under_ref "TmaV34WindowEngine*"]
  set ingress_regs [registers_under_ref "TmaV3Ingress*"]
  set dma_regs [registers_under_ref "TmaV2DmaCore*"]

  # Parent collections include child modules.  Give each register exactly one
  # architectural owner so no group masks another group's negative paths.
  set ingress_regs [remove_from_collection $ingress_regs $planner_regs]
  set ingress_regs [remove_from_collection $ingress_regs $binder_regs]
  foreach child_regs [list $ingress_regs $planner_regs $binder_regs \
      $descriptor_regs $engine_regs] {
    set dma_regs [remove_from_collection $dma_regs $child_regs]
  }

  # Create disjoint endpoint groups first.  The old flow assigned every
  # register path to REG_TO_REG/INPUT_TO_REG and then attempted to overlay
  # module groups; this DC release keeps the first matching group, leaving the
  # later module groups empty.  Generic classes must therefore use only the
  # registers not owned by a material TMA module.
  create_endpoint_group INGRESS $ingress_regs $critical_range 4.0
  create_endpoint_group PLANNER $planner_regs $critical_range 5.0
  create_endpoint_group BINDER $binder_regs $critical_range 4.0
  create_endpoint_group DESCRIPTOR $descriptor_regs $critical_range 4.0
  create_endpoint_group ENGINE $engine_regs $critical_range 5.0
  create_endpoint_group DMA_CORE_CTRL $dma_regs $critical_range 4.0

  set generic_regs $all_regs
  foreach owned_regs [list $ingress_regs $planner_regs $binder_regs \
      $descriptor_regs $engine_regs $dma_regs] {
    set generic_regs [remove_from_collection $generic_regs $owned_regs]
  }
  set data_inputs [remove_from_collection \
    [all_inputs] [get_ports $clock_port]]
  set outputs [all_outputs]
  create_path_class_group REG_TO_REG $all_regs $generic_regs \
    $critical_range 2.0 0
  create_path_class_group INPUT_TO_REG $data_inputs $generic_regs \
    $critical_range 2.0 0
  create_path_class_group REG_TO_OUTPUT $all_regs $outputs \
    $critical_range 2.0 0
  create_path_class_group INPUT_TO_OUTPUT $data_inputs $outputs \
    $critical_range 2.0 0
}

set script_dir [file normalize [file dirname [info script]]]
set export_root [file normalize [file join $script_dir ..]]
cd $export_root

set design_name [getenv_default "DESIGN_TOP" "tma"]
set clock_port [getenv_default "CLOCK_PORT" "clock"]
set synth_freq_mhz [getenv_default "SYNTH_FREQ_MHZ" 1500.0]
set report_freq_mhz [getenv_default "REPORT_FREQ_MHZ" 1500.0]
set synth_period [expr {1000.0 / $synth_freq_mhz}]
set report_period [expr {1000.0 / $report_freq_mhz}]
set max_cores [getenv_default "MAX_CORES" 12]
set rtl_filelist [getenv_default "RTL_FILELIST" \
  [file join $export_root dc filelists tma_core_sv.f]]
set target_library_db [getenv_default "TARGET_LIBRARY_DB" \
  "/vmshare/pdk/install/tsmc_n12_CLN12FFCLL/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn12ffcllbwp16p90cpd_100c/tcbn12ffcllbwp16p90cpdtt1v85c.db"]
set target_library_name [getenv_default "TARGET_LIBRARY_NAME" \
  "tcbn12ffcllbwp16p90cpdtt1v85c"]
set operating_condition [getenv_default "OPERATING_CONDITION" "tt1v85c"]
set variant [getenv_default \
  "VARIANT" "v3_7_line40_payload6_zerowire_1500"]
set max_area [getenv_default "MAX_AREA" 100000.0]
set critical_range [getenv_default "CRITICAL_RANGE_NS" 0.20]
set timing_model [string tolower [getenv_default "TIMING_MODEL" "zero_wire"]]
set workdir [getenv_default "OUTDIR" [file join $export_root DC_log $variant]]
set rptdir [file join $workdir report]
set datadir [file join $workdir data]

set boundary_driving_cell [getenv_default \
  "BOUNDARY_DRIVING_CELL" "BUFFD2BWP16P90CPD"]
set boundary_input_delay_max [getenv_default \
  "BOUNDARY_INPUT_DELAY_MAX_NS" 0.12]
set boundary_input_delay_min [getenv_default \
  "BOUNDARY_INPUT_DELAY_MIN_NS" 0.03]
set boundary_output_delay_max [getenv_default \
  "BOUNDARY_OUTPUT_DELAY_MAX_NS" 0.12]
set boundary_output_delay_min [getenv_default \
  "BOUNDARY_OUTPUT_DELAY_MIN_NS" 0.03]
set boundary_output_load [getenv_default "BOUNDARY_OUTPUT_LOAD_PF" 0.010]
set boundary_max_transition [getenv_default \
  "BOUNDARY_MAX_TRANSITION_NS" 0.12]
set boundary_max_fanout [getenv_default "BOUNDARY_MAX_FANOUT" 16]

if {$timing_model ni [list "zero_wire" "boundary_typical"]} {
  error "unsupported TIMING_MODEL=$timing_model"
}
if {$max_cores > 12} {
  error "MAX_CORES=$max_cores exceeds the project limit of 12"
}
if {![file exists $target_library_db]} {
  error "TARGET_LIBRARY_DB does not exist: $target_library_db"
}
if {![file exists $rtl_filelist]} {
  error "RTL_FILELIST does not exist: $rtl_filelist"
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

set fp [open $rtl_filelist r]
set files {}
while {[gets $fp line] >= 0} {
  set line [string trim $line]
  if {$line ne "" && [string index $line 0] ne "#"} {
    lappend files [file normalize [file join $export_root $line]]
  }
}
close $fp

puts "Ventus TMA V3.7 all-path DC: $variant top=$design_name"
puts "N12 TT 1.0V 85C, ${synth_freq_mhz}MHz, cores=$max_cores"
puts "Timing model: $timing_model"
foreach file $files {
  if {![analyze -work WORK -format sverilog $file]} {
    error "SystemVerilog analysis failed: $file"
  }
}
elaborate $design_name -work WORK
current_design $design_name
uniquify
link

check_design > [file join $rptdir check_design_pre_compile.rpt]
set_operating_conditions -library $target_library_name $operating_condition
set_wire_load_model -name ZeroWireload -library $target_library_name
apply_clock $clock_port $synth_period
if {$timing_model eq "boundary_typical"} {
  apply_boundary_typical \
    $clock_port $target_library_name $boundary_driving_cell \
    $boundary_input_delay_max $boundary_input_delay_min \
    $boundary_output_delay_max $boundary_output_delay_min \
    $boundary_output_load $boundary_max_transition $boundary_max_fanout
}
set_fix_multiple_port_nets -all -buffer_constants
set verilogout_no_tri true
remove_unconnected_ports [get_cells -hier {*}]
# This is a timing-first release run.  The 100k-cell cap leaves headroom for
# the explicit all-path pipeline boundaries while remaining below V3.4.
set_max_area $max_area
set_critical_range $critical_range [current_design]

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
  }
}

# Do not let the single worst endpoint monopolize optimization.  The groups
# are ref_name-derived, mutually owned, and include DmaCore-local queues.
refresh_tma_path_groups $critical_range $clock_port
report_path_group > [file join $rptdir path_groups_pre_compile.rpt]
report_hierarchy > [file join $rptdir hierarchy_pre_compile.rpt]
report_resources > [file join $rptdir resources_pre_compile.rpt]
check_timing > [file join $rptdir check_timing_pre_compile.rpt]

if {[string equal -nocase \
    [getenv_default "STOP_BEFORE_COMPILE" "false"] "true"]} {
  report_clock -skew > [file join $rptdir dry_run_clock.rpt]
  puts "INFO: STOP_BEFORE_COMPILE dry-run passed"
  exit
}

# High timing effort plus a 0.20 ns critical range makes DC work the complete
# negative-slack population in every group instead of polishing only WNS.
compile_ultra -retime -timing_high_effort_script
# The retime pass has created/replaced registers.  Refresh now so the
# incremental optimizer sees every current endpoint instead of stale groups.
update_timing
refresh_tma_path_groups $critical_range $clock_port
report_path_group > \
  [file join $rptdir path_groups_post_retime_pre_incremental.rpt]
compile_ultra -incremental
change_names -rules verilog -hierarchy
apply_clock $clock_port $report_period
if {$timing_model eq "boundary_typical"} {
  apply_boundary_typical \
    $clock_port $target_library_name $boundary_driving_cell \
    $boundary_input_delay_max $boundary_input_delay_min \
    $boundary_output_delay_max $boundary_output_delay_min \
    $boundary_output_load $boundary_max_transition $boundary_max_fanout
}
update_timing
# change_names can also invalidate name-based collections.  Recreate once
# more for final reporting and fail the run if any material group is empty.
refresh_tma_path_groups $critical_range $clock_port
update_timing
report_qor > [file join $rptdir qor_summary.rpt]
report_path_group > [file join $rptdir path_groups.rpt]
report_area > [file join $rptdir area.rpt]
report_area -hierarchy -nosplit > [file join $rptdir area_hierarchy.rpt]
report_area -designware > [file join $rptdir area_designware.rpt]
report_timing -delay_type max -max_paths 100 -nworst 10 -path full \
  -significant_digits 4 > [file join $rptdir timing_setup.rpt]
report_timing -delay_type min -max_paths 100 -nworst 10 -path full \
  -significant_digits 4 > [file join $rptdir timing_hold.rpt]
foreach group_name [list \
    REG_TO_REG INPUT_TO_REG REG_TO_OUTPUT INPUT_TO_OUTPUT \
    INGRESS PLANNER BINDER DESCRIPTOR ENGINE DMA_CORE_CTRL] {
  set group_present 0
  foreach_in_collection group [get_path_groups *] {
    if {[get_object_name $group] eq $group_name} {
      set group_present 1
    }
  }
  if {$group_present} {
    redirect [file join $rptdir timing_group_${group_name}.rpt] {
      report_timing -group $group_name -delay_type max \
        -max_paths 100 -nworst 10 -path full -significant_digits 4
    }
  } else {
    set absent [open \
      [file join $rptdir timing_group_${group_name}.rpt] w]
    puts $absent "INFO: optional path class $group_name is absent"
    close $absent
  }
}
report_power > [file join $rptdir power.rpt]
report_power -hierarchy -levels 5 > \
  [file join $rptdir power_hierarchy.rpt]
report_resources > [file join $rptdir resources.rpt]
report_reference -hierarchy > [file join $rptdir reference.rpt]
report_constraints -all_violators > \
  [file join $rptdir constraints_violations.rpt]
check_timing > [file join $rptdir check_timing.rpt]

set summary [open [file join $rptdir run_summary.rpt] w]
puts $summary "Design: $design_name"
puts $summary "Variant: $variant"
puts $summary "Process: TSMC N12 CLN12FFCLL"
puts $summary "Operating_Condition: $operating_condition"
puts $summary "Synthesis_Frequency_MHz: $synth_freq_mhz"
puts $summary "Report_Frequency_MHz: $report_freq_mhz"
puts $summary "Timing_Model: $timing_model"
puts $summary "Internal_Wireload: ZeroWireload"
if {$timing_model eq "boundary_typical"} {
  puts $summary "Boundary_Driving_Cell: $boundary_driving_cell"
  puts $summary "Boundary_Input_Delay_Max_ns: $boundary_input_delay_max"
  puts $summary "Boundary_Input_Delay_Min_ns: $boundary_input_delay_min"
  puts $summary "Boundary_Output_Delay_Max_ns: $boundary_output_delay_max"
  puts $summary "Boundary_Output_Delay_Min_ns: $boundary_output_delay_min"
  puts $summary "Boundary_Output_Load_pF: $boundary_output_load"
  puts $summary "Boundary_Max_Transition_ns: $boundary_max_transition"
  puts $summary "Boundary_Max_Fanout: $boundary_max_fanout"
}
puts $summary "PMU_Elaborated: false"
puts $summary "Max_Cores: $max_cores"
puts $summary "Max_Area: $max_area"
puts $summary "Critical_Range_ns: $critical_range"
puts $summary "Path_Groups: REG_TO_REG INPUT_TO_REG REG_TO_OUTPUT INPUT_TO_OUTPUT INGRESS PLANNER BINDER DESCRIPTOR ENGINE DMA_CORE_CTRL"
puts $summary "Setup_Acceptance: every group WNS >= 0, TNS = 0, violating paths = 0"
puts $summary "Hold_Acceptance: reported only; physical closure deferred to CTS/routing"
puts $summary "Compile: compile_ultra -retime -timing_high_effort_script; compile_ultra -incremental"
close $summary

write -hierarchy -format ddc \
  -output [file join $datadir ${design_name}.ddc]
write -hierarchy -format verilog \
  -output [file join $datadir ${design_name}_post.v]
write_sdf [file join $datadir ${design_name}.sdf]
puts "INFO: TMA V3.7 N12 TT 1.0V 85C 1.5GHz all-path synthesis finished"
exit
