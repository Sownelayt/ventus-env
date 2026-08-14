# Read-only post-map timing diagnosis for the completed V3.6 DC run.

proc getenv_required {name} {
  if {![info exists ::env($name)] || $::env($name) eq ""} {
    error "missing environment variable $name"
  }
  return $::env($name)
}

set ddc [getenv_required "TMA_DDC"]
set output_dir [getenv_required "TMA_REPORT_DIR"]
set target_library_db \
  "/vmshare/pdk/install/tsmc_n12_CLN12FFCLL/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn12ffcllbwp16p90cpd_100c/tcbn12ffcllbwp16p90cpdtt1v85c.db"
set target_library [list $target_library_db]
set link_library [list "*" $target_library_db]
file mkdir $output_dir
read_ddc $ddc
current_design tma
link
update_timing
set all_reg_pins [all_registers -data_pins]

foreach family [list \
    [list planner_cursor "core_ingress/windowPlanner/*cursor*"] \
    [list planner_coordinate "core_ingress/windowPlanner/coordinateStage*"] \
    [list planner_global "core_ingress/windowPlanner/globalStage*"] \
    [list planner_lane "core_ingress/windowPlanner/laneStage*"] \
    [list engine_payload_data "core_subsystem/engine/payloadData*"] \
    [list engine_payload_meta "core_subsystem/engine/payloadMeta*"] \
    [list engine_line_route "core_subsystem/engine/*line*Route*"] \
    [list engine_other "core_subsystem/engine/*"] \
    [list cache_egress "core_cacheEgress*"] \
    [list shared_egress "core_sharedEgress*"] \
    [list window_ingress "core_windowIngress*"] \
    [list descriptor "core_subsystem/descriptor/*"] \
    [list binder "core_ingress/binder/*"] \
    [list ingress "core_ingress/*"]] {
  set name [lindex $family 0]
  set pattern [lindex $family 1]
  set endpoints [filter_collection \
    $all_reg_pins "full_name =~ $pattern"]
  set count [sizeof_collection $endpoints]
  set fp [open [file join $output_dir ${name}_count.txt] w]
  puts $fp $count
  close $fp
  if {$count > 0} {
    redirect [file join $output_dir ${name}.rpt] {
      report_timing -delay_type max -group REG_TO_REG -to $endpoints -max_paths 20 \
        -nworst 1 -path full -significant_digits 4
    }
  }
}
exit
